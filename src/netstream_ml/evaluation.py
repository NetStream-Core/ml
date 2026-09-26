"""A reusable evaluation toolkit, independent of any one model.

`baselines.py`, `anomaly.py` and `supervised.py` each report a single point
on train/val/test — precision, recall, F1 at whatever threshold or
contamination that model happened to use. That single point hides two
things this module is for:

1. How much of that number is the model's own randomness, not the data?
   `seed_sensitivity` refits the same configuration under different random
   seeds and reports the spread.
2. Is one model's advantage over another real, or could the same test set
   have gone the other way by chance? `mcnemar_test` answers that for a
   pair of models evaluated on the same rows.

`precision_recall_curve` additionally traces the whole precision/recall
trade-off for a model that exposes a continuous score, rather than reading
off the single point its chosen threshold happens to land on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve as _sklearn_pr_curve

from netstream_ml.baselines import ClassificationMetrics


@dataclass(frozen=True, slots=True)
class PrecisionRecallCurve:
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    thresholds: tuple[float, ...]
    average_precision: float

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": list(self.precision),
            "recall": list(self.recall),
            "thresholds": list(self.thresholds),
            "average_precision": self.average_precision,
        }


def precision_recall_curve(y_true: pl.Series, y_score: pl.Series) -> PrecisionRecallCurve:
    """`y_score` is a continuous score where higher means "more likely an
    attack" — a model's decision function or predicted probability, not its
    thresholded prediction. Traces every precision/recall pair the model
    could reach by moving its decision threshold, so a model can be judged
    by the whole curve instead of by whatever single point its current
    threshold or contamination setting happens to land on."""
    true_np = y_true.to_numpy()
    score_np = y_score.to_numpy()
    precision, recall, thresholds = _sklearn_pr_curve(true_np, score_np)
    return PrecisionRecallCurve(
        precision=tuple(float(p) for p in precision),
        recall=tuple(float(r) for r in recall),
        thresholds=tuple(float(t) for t in thresholds),
        average_precision=float(average_precision_score(true_np, score_np)),
    )


@dataclass(frozen=True, slots=True)
class SeedSensitivity:
    """Summarises a metric's spread across refits with different random
    seeds, all else (data, hyperparameters) held fixed."""

    metric_name: str
    values: tuple[float, ...]
    mean: float
    std: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric_name,
            "values": list(self.values),
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


def seed_sensitivity(
    fit_and_evaluate: Callable[[int], ClassificationMetrics],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    metric_name: str = "f1",
) -> SeedSensitivity:
    """`fit_and_evaluate(seed)` should fit a model with that random seed
    (same data and hyperparameters otherwise) and return its metrics on
    whichever split is being reported on. A narrow spread means the
    reported metric is a property of the model and data, not a lucky draw
    of the random forest's bootstrap samples or the isolation forest's
    tree structure; a wide one means a single seed's number should not be
    trusted on its own."""
    if not seeds:
        raise ValueError("seeds must not be empty")

    values = tuple(getattr(fit_and_evaluate(seed), metric_name) for seed in seeds)
    array = np.array(values)
    return SeedSensitivity(
        metric_name=metric_name,
        values=values,
        mean=float(array.mean()),
        std=float(array.std()),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """`a_only_correct`/`b_only_correct` are the counts of rows where
    exactly one model got the right answer — the only rows that carry any
    information about which model is better, per McNemar's test. Rows
    where both models agree (both right or both wrong) are uninformative
    and dropped, which is why this needs no split-size caveat the way a
    plain accuracy comparison would."""

    a_only_correct: int
    b_only_correct: int
    p_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "a_only_correct": self.a_only_correct,
            "b_only_correct": self.b_only_correct,
            "p_value": self.p_value,
        }


def mcnemar_test(y_true: pl.Series, pred_a: pl.Series, pred_b: pl.Series) -> McNemarResult:
    """Whether model A's predictions differ from model B's by more than
    chance, on the same test rows. Uses an exact two-sided binomial test
    rather than the usual chi-squared approximation, since the lab's
    per-scenario slices are small enough that the approximation would not
    be reliable."""
    true_np = y_true.to_numpy()
    correct_a = pred_a.to_numpy() == true_np
    correct_b = pred_b.to_numpy() == true_np

    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue

    return McNemarResult(a_only_correct=a_only, b_only_correct=b_only, p_value=p_value)
