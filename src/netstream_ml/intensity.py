"""Whether a detector's recall holds up as an attack gets quieter, not just
whether it catches the loud, obvious version the lab happens to have most
runs of. `deploy/lab/run.sh` already exposes a `rate` parameter for every
flood/scan/tunnel scenario, and the campaign (`lab/campaigns/v1.yaml`)
already runs several scenarios at more than one rate — so this does not
need a new campaign, only a way to join a feature window back to the rate
of the run it came from.

That join goes through `labels.rate` (parsed out of the run's `params`
JSON) and `run_id`, which `features.py` now carries through from
`labeled_flows`/`labeled_dns` alongside `label`/`scenario`/`split`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import polars as pl

from netstream_ml.baselines import ClassificationMetrics, evaluate_predictions


def _extract_rate(params: str) -> float | None:
    try:
        parsed = json.loads(params)
    except (json.JSONDecodeError, TypeError):
        return None
    rate = parsed.get("rate")
    return float(rate) if rate is not None else None


def run_rates(labels: pl.DataFrame) -> pl.DataFrame:
    """One row per run_id with its `rate` parameter, or null if the
    scenario has no `rate` (e.g. `iodine_tunnel`, `benign`)."""
    return labels.select(
        "run_id",
        pl.col("params").map_elements(_extract_rate, return_dtype=pl.Float64).alias("rate"),
    )


def attach_rate(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Left-joins each feature window onto the `rate` of the run it came
    from. A window with no `run_id` (background traffic) or whose scenario
    has no `rate` parameter gets a null rate."""
    if "run_id" not in features.columns:
        raise ValueError(
            "features frame has no run_id column; rebuild it with a version of "
            "`features build` that carries run_id through (after data split)"
        )
    return features.join(run_rates(labels), on="run_id", how="left")


@dataclass(frozen=True, slots=True)
class RateGroupMetrics:
    scenario: str
    rate: float | None
    n: int
    metrics: ClassificationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "rate": self.rate,
            "n": self.n,
            **self.metrics.to_dict(),
        }


def evaluate_by_rate(
    predict: pl.Series,
    frame: pl.DataFrame,
    label_column: str = "label",
    benign_label: str = "benign",
) -> list[RateGroupMetrics]:
    """`predict` must already be the model's boolean prediction for every
    row of `frame`, in the same order (call the model's own `.predict`
    first — this function is model-agnostic on purpose, the same grouping
    works for any of the three baselines). Groups by `scenario` and `rate`,
    columns `attach_rate` guarantees are present. Only scenarios that
    actually have a `rate` (attacks with an intensity knob) produce groups;
    `benign` and rate-less scenarios are skipped, since "sensitivity to
    intensity" is not a question that applies to them."""
    y_true = frame[label_column] != benign_label
    results = []
    with_pred = frame.with_columns(predict.alias("_predicted"), y_true.alias("_true"))

    groups = (
        with_pred.filter(pl.col("rate").is_not_null())
        .select("scenario", "rate")
        .unique()
        .sort(["scenario", "rate"])
    )
    for scenario, rate in groups.iter_rows():
        subset = with_pred.filter((pl.col("scenario") == scenario) & (pl.col("rate") == rate))
        metrics = evaluate_predictions(subset["_true"], subset["_predicted"])
        results.append(
            RateGroupMetrics(scenario=scenario, rate=rate, n=subset.height, metrics=metrics)
        )
    return results
