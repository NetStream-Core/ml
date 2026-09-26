"""A threshold baseline: for each candidate feature, pick the value above which
no more than `target_fpr` of the training set's benign traffic falls, then
flag a window as an attack if any feature crosses its threshold.

This is deliberately the simplest possible detector — one number per feature,
no training beyond reading off a quantile, easy to explain to anyone auditing
it. It exists to answer one question before any model is built: is there
already enough separation in the features for a naive rule to do reasonably
well? If a random forest can't clear this bar by a wide margin, the forest
isn't earning its complexity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

DEFAULT_CANDIDATE_FEATURES = (
    "aggregated_ratio",
    "unique_dst_ports",
    "packets_total",
    "rst_ratio",
    "syn_ratio",
    "dns_txt_null_ratio",
    "dns_entropy_mean",
    "dns_unique_subdomains_max",
)


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    feature: str
    threshold: float
    direction: Literal["above", "below"] = "above"

    def fires(self, frame: pl.DataFrame) -> pl.Series:
        column = frame[self.feature].fill_null(
            float("-inf") if self.direction == "above" else float("inf")
        )
        if self.direction == "above":
            return column > self.threshold
        return column < self.threshold

    def to_dict(self) -> dict[str, object]:
        return {"feature": self.feature, "threshold": self.threshold, "direction": self.direction}

    def excess(self, frame: pl.DataFrame) -> pl.Series:
        """How far past this rule's own threshold each row's value falls,
        as a fraction of the threshold — 0 at the threshold, positive
        beyond it, growing without bound the further past it a row is.
        Used only to rank rows by "how attack-like", not to decide
        pass/fail (`fires` does that); a null value counts as not exceeding
        the threshold, same as in `fires`."""
        column = frame[self.feature].fill_null(0.0)
        threshold = self.threshold if self.threshold != 0 else 1.0
        if self.direction == "above":
            return ((column - self.threshold) / abs(threshold)).alias("excess")
        return ((self.threshold - column) / abs(threshold)).alias("excess")


@dataclass(frozen=True, slots=True)
class ThresholdBaseline:
    rules: tuple[ThresholdRule, ...]

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        if not self.rules:
            return pl.Series("predicted", [False] * frame.height)
        fired = [rule.fires(frame) for rule in self.rules]
        result = fired[0]
        for series in fired[1:]:
            result = result | series
        return result.alias("predicted")

    def decision_score(self, frame: pl.DataFrame) -> pl.Series:
        """The largest normalized excess over any single rule's threshold —
        a continuous stand-in for "how attack-like", for tracing a
        precision/recall curve rather than reading off the one point
        `predict`'s fixed thresholds happen to land on."""
        if not self.rules:
            return pl.Series("score", [0.0] * frame.height)
        excesses = np.stack([rule.excess(frame).to_numpy() for rule in self.rules])
        return pl.Series("score", excesses.max(axis=0))

    def to_dict(self) -> list[dict[str, object]]:
        return [rule.to_dict() for rule in self.rules]


def fit_threshold_baseline(
    train: pl.DataFrame,
    features_to_consider: Sequence[str] = DEFAULT_CANDIDATE_FEATURES,
    label_column: str = "label",
    benign_label: str = "benign",
    target_fpr: float = 0.01,
) -> ThresholdBaseline:
    """Picks, for each feature, the (1 - target_fpr) quantile of its value
    among benign training rows, so that feature alone would false-alarm on
    about `target_fpr` of benign traffic. A feature missing from the frame,
    or with no benign observations, is skipped rather than raising: DNS
    features are absent for windows with no DNS activity at all."""
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be between 0 and 1, got {target_fpr}")

    benign = train.filter(pl.col(label_column) == benign_label)
    rules = []
    for feature in features_to_consider:
        if feature not in train.columns:
            continue
        values = benign[feature].drop_nulls()
        if values.len() == 0:
            continue
        threshold = values.quantile(1 - target_fpr)
        if threshold is None:
            continue
        rules.append(ThresholdRule(feature=feature, threshold=float(threshold)))
    return ThresholdBaseline(tuple(rules))


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
        }


def evaluate_predictions(y_true: pl.Series, y_pred: pl.Series) -> ClassificationMetrics:
    true_positives = int((y_true & y_pred).sum())
    false_positives = int((~y_true & y_pred).sum())
    false_negatives = int((y_true & ~y_pred).sum())
    true_negatives = int((~y_true & ~y_pred).sum())

    positive_predictions = true_positives + false_positives
    actual_positives = true_positives + false_negatives
    actual_negatives = false_positives + true_negatives
    total = true_positives + false_positives + false_negatives + true_negatives

    precision = true_positives / positive_predictions if positive_predictions else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = false_positives / actual_negatives if actual_negatives else 0.0
    accuracy = (true_positives + true_negatives) / total if total else 0.0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        accuracy=accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )


def evaluate_baseline(
    baseline: ThresholdBaseline,
    frame: pl.DataFrame,
    label_column: str = "label",
    benign_label: str = "benign",
) -> ClassificationMetrics:
    y_true = frame[label_column] != benign_label
    y_pred = baseline.predict(frame)
    return evaluate_predictions(y_true, y_pred)


def evaluate_baseline_by_scenario(
    baseline: ThresholdBaseline,
    frame: pl.DataFrame,
    scenario_column: str = "scenario",
    label_column: str = "label",
    benign_label: str = "benign",
) -> dict[str, ClassificationMetrics]:
    """Per-scenario recall (for attacks) or false-positive rate (for anything
    labelled benign, including a benign-looking profile that shares a
    scenario name with a real attack, like `admin_scan` next to
    `port_scan`)."""
    y_pred = baseline.predict(frame)
    results = {}
    for scenario in frame[scenario_column].unique(maintain_order=True).sort():
        mask = frame[scenario_column] == scenario
        subset_true = frame.filter(mask)[label_column] != benign_label
        subset_pred = y_pred.filter(mask)
        results[scenario] = evaluate_predictions(subset_true, subset_pred)
    return results
