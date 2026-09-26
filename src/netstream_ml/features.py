"""Windowed features for the attack detector, computed once and used twice:
`window_features` for offline training (a fast vectorised polars pipeline)
and `WindowAccumulator` for the online detector (a plain incremental
accumulator that can be fed one record at a time from a live stream).

Both compute the exact same feature set from the exact same fields, so a
model trained on the batch path sees the same numbers the online path will
serve it later. `tests/test_features.py` cross-checks them against each
other on generated data, and `tests/test_features_parity.py` cross-checks
them on a real exported dataset when one is available.

A window is a fixed-size, epoch-aligned bucket (`floor(ts / seconds)`) of
one source IP's traffic. Grouping by source rather than by flow keeps the
feature set small and lets a single row describe "what is this host doing
right now", which is what a scan, a flood or a beacon shows up as. The
label and scenario of a window are taken from its last record by
timestamp: in the lab an attacking host doesn't share a window with
background traffic (attacker and client are different addresses), so this
matters only at a run's boundary, where the traffic just after it started
is the more informative half.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

SIZE_BIN_COLUMNS = (
    "size_le64",
    "size_le128",
    "size_le256",
    "size_le512",
    "size_le1024",
    "size_gt1024",
)

FLOW_FEATURE_COLUMNS = (
    "packets_total",
    "ip_bytes_total",
    "payload_bytes_total",
    "flows_count",
    "unique_dst_ips",
    "unique_dst_ports",
    "receive_ratio",
    "syn_ratio",
    "synack_ratio",
    "rst_ratio",
    "aggregated_ratio",
    *(f"{c}_ratio" for c in SIZE_BIN_COLUMNS),
    "iat_mean_us",
    "iat_std_us",
)

DNS_FEATURE_COLUMNS = (
    "dns_queries_count",
    "dns_unique_qnames",
    "dns_qname_length_mean",
    "dns_qname_length_max",
    "dns_entropy_mean",
    "dns_entropy_max",
    "dns_digit_ratio_mean",
    "dns_txt_null_ratio",
    "dns_tcp_ratio",
    "dns_unique_subdomains_max",
)

FEATURE_COLUMNS = FLOW_FEATURE_COLUMNS + DNS_FEATURE_COLUMNS

DNS_COUNT_LIKE_COLUMNS = ("dns_queries_count", "dns_unique_qnames", "dns_unique_subdomains_max")


@dataclass(frozen=True, slots=True)
class WindowSpec:
    seconds: int

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"window seconds must be positive, got {self.seconds}")

    def bucket(self, epoch_seconds: float) -> int:
        return int(epoch_seconds // self.seconds) * self.seconds


def _as_utc_epoch(timestamp: dt.datetime) -> float:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.UTC)
    return timestamp.timestamp()


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(0.0)


def _window_bucket(window: WindowSpec, ts_column: str = "ts") -> pl.Expr:
    epoch = pl.col(ts_column).dt.epoch("s")
    return ((epoch // window.seconds) * window.seconds).alias("window_start")


def flow_window_features(frame: pl.DataFrame, window: WindowSpec) -> pl.DataFrame:
    """One row per (src_ip, window_start) summarising that source's flows.

    `synack_ratio` is the share of this source's own SYN packets that carry
    the ACK flag too, i.e. how much of this source's traffic looks like it's
    accepting connections rather than opening them. It cannot show whether
    the connections *this* source opened got answered: the reply is a
    separate flow recorded under the other side's src_ip, in a window keyed
    by that address, not this one. A source that only ever attacks will
    read close to 0 here for that reason, not because its floods went
    unanswered.
    """
    if frame.height == 0:
        return _empty_flow_features()

    with_bucket = frame.with_columns(_window_bucket(window))
    ordered = with_bucket.sort("ts")

    size_ratio_exprs = [
        _ratio(pl.col(c).sum(), pl.col("packets").sum()).alias(f"{c}_ratio")
        for c in SIZE_BIN_COLUMNS
    ]
    carry_split = [pl.col("split").last()] if "split" in frame.columns else []
    carry_run_id = [pl.col("run_id").last()] if "run_id" in frame.columns else []

    return (
        ordered.group_by(["src_ip", "window_start"], maintain_order=True)
        .agg(
            pl.len().alias("flows_count"),
            pl.col("packets").sum().alias("packets_total"),
            pl.col("ip_bytes").sum().alias("ip_bytes_total"),
            pl.col("payload_bytes").sum().alias("payload_bytes_total"),
            pl.col("dst_ip").n_unique().alias("unique_dst_ips"),
            pl.col("dst_port").n_unique().alias("unique_dst_ports"),
            _ratio(
                pl.col("packets").filter(pl.col("direction") == "receive").sum(),
                pl.col("packets").sum(),
            ).alias("receive_ratio"),
            _ratio(pl.col("tcp_syn").sum(), pl.col("packets").sum()).alias("syn_ratio"),
            _ratio(pl.col("tcp_synack").sum(), pl.col("tcp_syn").sum()).alias("synack_ratio"),
            _ratio(pl.col("tcp_rst").sum(), pl.col("packets").sum()).alias("rst_ratio"),
            _ratio(
                pl.col("packets").filter(pl.col("aggregated") > 0).sum(),
                pl.col("packets").sum(),
            ).alias("aggregated_ratio"),
            *size_ratio_exprs,
            _ratio(pl.col("iat_sum_us").sum(), pl.col("iat_count").sum()).alias("iat_mean_us"),
            pl.col("iat_count").sum().alias("_iat_count"),
            pl.col("iat_sum_us").sum().alias("_iat_sum_us"),
            pl.col("iat_sumsq_us").sum().alias("_iat_sumsq_us"),
            pl.col("label").last().alias("label"),
            pl.col("scenario").last().alias("scenario"),
            *carry_split,
            *carry_run_id,
        )
        .with_columns(
            _iat_std_expr(
                pl.col("_iat_count"), pl.col("_iat_sum_us"), pl.col("_iat_sumsq_us")
            ).alias("iat_std_us")
        )
        .drop("_iat_count", "_iat_sum_us", "_iat_sumsq_us")
    )


def _iat_std_expr(count: pl.Expr, total: pl.Expr, total_sq: pl.Expr) -> pl.Expr:
    mean = _ratio(total, count)
    variance = _ratio(total_sq, count) - mean * mean
    return pl.when(count > 0).then(variance.clip(lower_bound=0.0).sqrt()).otherwise(0.0)


def _empty_flow_features() -> pl.DataFrame:
    schema = {
        "src_ip": pl.Utf8,
        "window_start": pl.Int64,
        **{name: pl.Float64 for name in FLOW_FEATURE_COLUMNS},
        "label": pl.Utf8,
        "scenario": pl.Utf8,
    }
    schema["flows_count"] = pl.UInt32
    schema["unique_dst_ips"] = pl.UInt32
    schema["unique_dst_ports"] = pl.UInt32
    return pl.DataFrame(schema=schema)


def dns_window_features(frame: pl.DataFrame, window: WindowSpec) -> pl.DataFrame:
    """One row per (src_ip, window_start) summarising that source's DNS queries."""
    if frame.height == 0:
        return _empty_dns_features()

    with_bucket = frame.with_columns(_window_bucket(window))

    return with_bucket.group_by(["src_ip", "window_start"], maintain_order=True).agg(
        pl.len().alias("dns_queries_count"),
        pl.col("qname").n_unique().alias("dns_unique_qnames"),
        pl.col("qname_length").mean().alias("dns_qname_length_mean"),
        pl.col("qname_length").max().alias("dns_qname_length_max"),
        pl.col("entropy").mean().alias("dns_entropy_mean"),
        pl.col("entropy").max().alias("dns_entropy_max"),
        pl.col("digit_ratio").mean().alias("dns_digit_ratio_mean"),
        (pl.col("qtype").is_in(["TXT", "NULL"]).sum() / pl.len()).alias("dns_txt_null_ratio"),
        (pl.col("transport") == "tcp").sum().truediv(pl.len()).alias("dns_tcp_ratio"),
        pl.col("unique_subdomains").max().alias("dns_unique_subdomains_max"),
    )


def _empty_dns_features() -> pl.DataFrame:
    schema = {
        "src_ip": pl.Utf8,
        "window_start": pl.Int64,
        **{name: pl.Float64 for name in DNS_FEATURE_COLUMNS},
    }
    schema["dns_queries_count"] = pl.UInt32
    schema["dns_unique_qnames"] = pl.UInt32
    return pl.DataFrame(schema=schema)


def combine_features(flow_features: pl.DataFrame, dns_features: pl.DataFrame) -> pl.DataFrame:
    """Left-joins DNS features onto flow features by (src_ip, window_start).

    A window with flows but no DNS activity gets zero counts and null
    aggregates for the DNS columns, since a source that made no queries has
    no query lengths or entropy to average.
    """
    joined = flow_features.join(
        dns_features, on=["src_ip", "window_start"], how="left", suffix="_dns"
    )
    return joined.with_columns([pl.col(c).fill_null(0) for c in DNS_COUNT_LIKE_COLUMNS])


def build_features(flows: pl.DataFrame, dns: pl.DataFrame, window: WindowSpec) -> pl.DataFrame:
    return combine_features(flow_window_features(flows, window), dns_window_features(dns, window))


@dataclass
class _FlowAccumulator:
    flows_count: int = 0
    packets_total: int = 0
    ip_bytes_total: int = 0
    payload_bytes_total: int = 0
    dst_ips: set[str] = field(default_factory=set)
    dst_ports: set[int] = field(default_factory=set)
    receive_packets: int = 0
    tcp_syn: int = 0
    tcp_synack: int = 0
    tcp_rst: int = 0
    aggregated_packets: int = 0
    size_bins: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SIZE_BIN_COLUMNS, 0))
    iat_count: int = 0
    iat_sum_us: int = 0
    iat_sumsq_us: int = 0
    label: str = "benign"
    scenario: str = "background"

    def add(self, record: dict[str, Any]) -> None:
        packets = int(record["packets"])
        self.flows_count += 1
        self.packets_total += packets
        self.ip_bytes_total += int(record["ip_bytes"])
        self.payload_bytes_total += int(record["payload_bytes"])
        self.dst_ips.add(str(record["dst_ip"]))
        self.dst_ports.add(int(record["dst_port"]))
        if record["direction"] == "receive":
            self.receive_packets += packets
        self.tcp_syn += int(record["tcp_syn"])
        self.tcp_synack += int(record["tcp_synack"])
        self.tcp_rst += int(record["tcp_rst"])
        if int(record["aggregated"]) > 0:
            self.aggregated_packets += packets
        for column in SIZE_BIN_COLUMNS:
            self.size_bins[column] += int(record[column])
        self.iat_count += int(record["iat_count"])
        self.iat_sum_us += int(record["iat_sum_us"])
        self.iat_sumsq_us += int(record["iat_sumsq_us"])
        self.label = str(record["label"])
        self.scenario = str(record["scenario"])

    def snapshot(self) -> dict[str, object]:
        packets = self.packets_total
        mean = self.iat_sum_us / self.iat_count if self.iat_count else 0.0
        variance = (self.iat_sumsq_us / self.iat_count - mean * mean) if self.iat_count else 0.0
        return {
            "flows_count": self.flows_count,
            "packets_total": packets,
            "ip_bytes_total": self.ip_bytes_total,
            "payload_bytes_total": self.payload_bytes_total,
            "unique_dst_ips": len(self.dst_ips),
            "unique_dst_ports": len(self.dst_ports),
            "receive_ratio": self.receive_packets / packets if packets else 0.0,
            "syn_ratio": self.tcp_syn / packets if packets else 0.0,
            "synack_ratio": self.tcp_synack / self.tcp_syn if self.tcp_syn else 0.0,
            "rst_ratio": self.tcp_rst / packets if packets else 0.0,
            "aggregated_ratio": self.aggregated_packets / packets if packets else 0.0,
            **{
                f"{column}_ratio": (self.size_bins[column] / packets if packets else 0.0)
                for column in SIZE_BIN_COLUMNS
            },
            "iat_mean_us": mean,
            "iat_std_us": math.sqrt(max(variance, 0.0)),
            "label": self.label,
            "scenario": self.scenario,
        }


@dataclass
class _DnsAccumulator:
    queries_count: int = 0
    qnames: set[str] = field(default_factory=set)
    qname_length_sum: int = 0
    qname_length_max: int = 0
    entropy_sum: float = 0.0
    entropy_max: float = 0.0
    digit_ratio_sum: float = 0.0
    txt_null_count: int = 0
    tcp_count: int = 0
    unique_subdomains_max: int = 0

    def add(self, record: dict[str, Any]) -> None:
        self.queries_count += 1
        self.qnames.add(str(record["qname"]))
        length = int(record["qname_length"])
        self.qname_length_sum += length
        self.qname_length_max = max(self.qname_length_max, length)
        entropy = float(record["entropy"])
        self.entropy_sum += entropy
        self.entropy_max = max(self.entropy_max, entropy)
        self.digit_ratio_sum += float(record["digit_ratio"])
        if record["qtype"] in ("TXT", "NULL"):
            self.txt_null_count += 1
        if record["transport"] == "tcp":
            self.tcp_count += 1
        self.unique_subdomains_max = max(
            self.unique_subdomains_max,
            int(record["unique_subdomains"]),
        )

    def snapshot(self) -> dict[str, object]:
        n = self.queries_count
        return {
            "dns_queries_count": n,
            "dns_unique_qnames": len(self.qnames),
            "dns_qname_length_mean": self.qname_length_sum / n if n else None,
            "dns_qname_length_max": self.qname_length_max if n else None,
            "dns_entropy_mean": self.entropy_sum / n if n else None,
            "dns_entropy_max": self.entropy_max if n else None,
            "dns_digit_ratio_mean": self.digit_ratio_sum / n if n else None,
            "dns_txt_null_ratio": self.txt_null_count / n if n else None,
            "dns_tcp_ratio": self.tcp_count / n if n else None,
            "dns_unique_subdomains_max": self.unique_subdomains_max if n else None,
        }


class WindowAccumulator:
    """Incremental, streaming twin of `build_features`.

    Feed it flow and DNS records as they arrive (in any order across
    sources, but in timestamp order within a source) and read out a
    window's feature row with `pop_completed`, once a later record shows
    that window has closed.
    """

    def __init__(self, window: WindowSpec) -> None:
        self._window = window
        self._flows: dict[tuple[str, int], _FlowAccumulator] = {}
        self._dns: dict[tuple[str, int], _DnsAccumulator] = {}
        self._completed: list[dict[str, object]] = []
        self._latest_bucket: dict[str, int] = {}

    def add_flow_record(self, record: dict[str, Any]) -> None:
        src_ip = str(record["src_ip"])
        bucket = self._window.bucket(_as_utc_epoch(record["ts"]))
        self._advance(src_ip, bucket)
        key = (src_ip, bucket)
        self._flows.setdefault(key, _FlowAccumulator()).add(record)

    def add_dns_record(self, record: dict[str, Any]) -> None:
        src_ip = str(record["src_ip"])
        bucket = self._window.bucket(_as_utc_epoch(record["ts"]))
        self._advance(src_ip, bucket)
        key = (src_ip, bucket)
        self._dns.setdefault(key, _DnsAccumulator()).add(record)

    def _advance(self, src_ip: str, bucket: int) -> None:
        previous = self._latest_bucket.get(src_ip)
        if previous is not None and bucket > previous:
            self._close(src_ip, previous)
        self._latest_bucket[src_ip] = max(previous or bucket, bucket)

    def _close(self, src_ip: str, bucket: int) -> None:
        key = (src_ip, bucket)
        flow_acc = self._flows.pop(key, None)
        dns_acc = self._dns.pop(key, None)
        if flow_acc is None:
            return
        row: dict[str, object] = {"src_ip": src_ip, "window_start": bucket}
        row.update(flow_acc.snapshot())
        if dns_acc is not None:
            row.update(dns_acc.snapshot())
        else:
            row.update(_DnsAccumulator().snapshot())
            for name in DNS_COUNT_LIKE_COLUMNS:
                row[name] = 0
        self._completed.append(row)

    def flush(self) -> None:
        """Closes every window still open, for use at the end of a stream."""
        for src_ip, bucket in list(self._latest_bucket.items()):
            self._close(src_ip, bucket)
        self._latest_bucket.clear()

    def pop_completed(self) -> list[dict[str, object]]:
        completed, self._completed = self._completed, []
        return completed
