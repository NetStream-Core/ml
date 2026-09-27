"""Loads a CICFlowMeter-style CSV (CIC-IDS2017, CIC-DDoS2019) and reduces it
to the subset of `features.FEATURE_COLUMNS` that can honestly be
reconstructed from it, so the lab-trained detectors can be re-evaluated on
traffic they have never seen — the point of M10.

CICFlowMeter's unit of aggregation is already a single bidirectional flow
(one row per 5-tuple), not a per-source-IP time window like ours, and it
never touches DNS payloads at all. That rules out most of `FEATURE_COLUMNS`
outright, not just as a data-cleaning inconvenience:

- every DNS_FEATURE_COLUMNS entry: no payload inspection in this dataset.
- `receive_ratio`: our value depends on a source sharing a window with
  traffic addressed *to* it from unrelated flows; CICFlowMeter's fwd/bwd
  split is direction *within* one flow (who opened it), a different axis.
- `synack_ratio`: no combined SYN+ACK flag count column, only separate
  SYN/ACK counts.
- `aggregated_ratio`: eBPF/NIC packet-aggregation is not a concept that
  exists in a flow-level CSV.
- every `*_ratio` size bin: CICFlowMeter gives per-flow min/max/mean/std
  packet length, not a histogram over individual packet sizes, so the bin
  ratios cannot be recovered even approximately.

`RECONSTRUCTABLE_FEATURES` is what survives: packet/byte/flow counts,
destination fan-out, SYN and RST ratios, and inter-arrival time mean/std
(recombined from each flow's own mean/std the same way the online
accumulator combines them, since a window here spans several flows).
`evaluate transfer` retrains each baseline on the lab data restricted to
exactly this subset, so the external-dataset numbers are comparable rather
than an apples-to-oranges mismatch against models trained on the full
feature set.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from netstream_ml.features import FEATURE_COLUMNS, WindowSpec

RECONSTRUCTABLE_FEATURES = (
    "packets_total",
    "payload_bytes_total",
    "flows_count",
    "unique_dst_ips",
    "unique_dst_ports",
    "syn_ratio",
    "rst_ratio",
    "iat_mean_us",
    "iat_std_us",
)

UNRECONSTRUCTABLE_FEATURES = tuple(f for f in FEATURE_COLUMNS if f not in RECONSTRUCTABLE_FEATURES)

BENIGN_LABEL_RAW = "BENIGN"

_RAW_COLUMNS = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Destination Port": "dst_port",
    "Timestamp": "ts_raw",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "SYN Flag Count": "syn_count",
    "RST Flag Count": "rst_count",
    "Flow IAT Mean": "iat_mean_us_raw",
    "Flow IAT Std": "iat_std_us_raw",
    "Label": "label_raw",
}

# CICFlowMeter's timestamp format differs across dataset releases (with or
# without seconds, with or without an AM/PM marker, CIC-DDoS2019's own
# ISO-like "%Y-%m-%d %H:%M:%S.%f"); tried in order, and the first format
# that parses without turning every row into a null wins.
_TIMESTAMP_FORMATS = (
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S%.f",
)


def load_cic_csv(path: Path) -> pl.DataFrame:
    """Reads one CICFlowMeter CSV and normalises it to the raw columns
    `external_window_features` needs, dropping everything else.

    Column names in these files are inconsistently padded with stray
    leading/trailing spaces (e.g. `" Destination Port"`), which is why
    every lookup here goes through a stripped-name mapping instead of the
    literal header text.
    """
    raw = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)
    rename = {name: name.strip() for name in raw.columns}
    raw = raw.rename(rename)

    missing = [name for name in _RAW_COLUMNS if name not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing expected CICFlowMeter columns: {missing}")

    selected = raw.select(
        [pl.col(name).alias(alias) for name, alias in _RAW_COLUMNS.items()]
    )
    return selected.with_columns(
        _parse_timestamp(selected["ts_raw"]),
        pl.col("dst_port").cast(pl.Int64, strict=False),
        *(
            pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0)
            for c in (
                "fwd_packets",
                "bwd_packets",
                "fwd_bytes",
                "bwd_bytes",
                "syn_count",
                "rst_count",
                "iat_mean_us_raw",
                "iat_std_us_raw",
            )
        ),
    ).drop("ts_raw")


def _parse_timestamp(column: pl.Series) -> pl.Expr:
    best: pl.Expr | None = None
    best_nulls = column.len() + 1
    for fmt in _TIMESTAMP_FORMATS:
        parsed = column.str.strip_chars().str.to_datetime(fmt, strict=False)
        nulls = int(parsed.is_null().sum())
        if nulls < best_nulls:
            best, best_nulls = pl.col("ts_raw").str.strip_chars().str.to_datetime(
                fmt, strict=False
            ), nulls
    if best is None:
        raise ValueError("could not parse Timestamp column with any known CICFlowMeter format")
    return best.alias("ts")


def external_window_features(flows: pl.DataFrame, window: WindowSpec) -> pl.DataFrame:
    """One row per (src_ip, window_start), restricted to
    `RECONSTRUCTABLE_FEATURES` plus `label`/`scenario`. `label` is
    `"benign"` for the dataset's own `BENIGN` rows and `"attack"`
    otherwise; `scenario` keeps the dataset's original label string
    (lowercased, spaces to underscores) for per-scenario breakdowns.
    """
    if flows.height == 0:
        return _empty_external_features()

    bucketed = flows.with_columns(
        ((pl.col("ts").dt.epoch("s") // window.seconds) * window.seconds).alias("window_start"),
        (pl.col("fwd_packets") + pl.col("bwd_packets")).alias("packets"),
        pl.col("label_raw")
        .str.to_lowercase()
        .str.replace_all(r"\s+", "_")
        .alias("scenario"),
    ).with_columns(
        pl.when(pl.col("label_raw") == BENIGN_LABEL_RAW)
        .then(pl.lit("benign"))
        .otherwise(pl.lit("attack"))
        .alias("label"),
        (pl.col("packets") - 1).clip(lower_bound=0).alias("iat_count"),
    ).with_columns(
        (pl.col("iat_mean_us_raw") * pl.col("iat_count")).alias("iat_sum_us"),
        (
            (pl.col("iat_std_us_raw") ** 2 + pl.col("iat_mean_us_raw") ** 2)
            * pl.col("iat_count")
        ).alias("iat_sumsq_us"),
    )

    def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator > 0).then(numerator / denominator).otherwise(0.0)

    return (
        bucketed.sort("ts")
        .group_by(["src_ip", "window_start"], maintain_order=True)
        .agg(
            pl.len().alias("flows_count"),
            pl.col("packets").sum().alias("packets_total"),
            (pl.col("fwd_bytes") + pl.col("bwd_bytes")).sum().alias("payload_bytes_total"),
            pl.col("dst_ip").n_unique().alias("unique_dst_ips"),
            pl.col("dst_port").n_unique().alias("unique_dst_ports"),
            _ratio(pl.col("syn_count").sum(), pl.col("packets").sum()).alias("syn_ratio"),
            _ratio(pl.col("rst_count").sum(), pl.col("packets").sum()).alias("rst_ratio"),
            pl.col("iat_count").sum().alias("_iat_count"),
            pl.col("iat_sum_us").sum().alias("_iat_sum_us"),
            pl.col("iat_sumsq_us").sum().alias("_iat_sumsq_us"),
            pl.col("label").last().alias("label"),
            pl.col("scenario").last().alias("scenario"),
        )
        .with_columns(
            _ratio(pl.col("_iat_sum_us"), pl.col("_iat_count")).alias("iat_mean_us"),
        )
        .with_columns(
            (
                _ratio(pl.col("_iat_sumsq_us"), pl.col("_iat_count"))
                - pl.col("iat_mean_us") ** 2
            )
            .clip(lower_bound=0.0)
            .sqrt()
            .alias("iat_std_us"),
        )
        .drop("_iat_count", "_iat_sum_us", "_iat_sumsq_us")
    )


def _empty_external_features() -> pl.DataFrame:
    schema = {
        "src_ip": pl.Utf8,
        "window_start": pl.Int64,
        **{name: pl.Float64 for name in RECONSTRUCTABLE_FEATURES},
        "label": pl.Utf8,
        "scenario": pl.Utf8,
    }
    schema["flows_count"] = pl.UInt32
    schema["unique_dst_ips"] = pl.UInt32
    schema["unique_dst_ports"] = pl.UInt32
    return pl.DataFrame(schema=schema)
