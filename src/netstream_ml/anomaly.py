"""An Isolation Forest baseline: unlike the threshold baseline, it is fit on
*all* feature columns at once and can separate an attack from benign traffic
by an unusual combination of otherwise-ordinary values, not just by any
single feature crossing a fixed line. It is still trained on benign traffic
only — an isolation forest does not need attack examples to work, which
matters here since the lab has far more benign windows than attack ones and
some attack scenarios (a handful of runs each) are too rare to hold out a
reliable validation slice of their own.

Whether the extra complexity is worth it over the threshold baseline is
exactly the question `evaluate_isolation_forest_by_scenario` is meant to
answer, scenario by scenario, the same way `baselines.py` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from netstream_ml.baselines import ClassificationMetrics, evaluate_predictions
from netstream_ml.features import FEATURE_COLUMNS


def _to_matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """Selects the feature columns and fills nulls with 0.

    A null here only ever means "no DNS activity in this window" (see
    `combine_features`): the count-like columns are already 0-filled, and
    the DNS mean/max aggregates are left null because there is nothing to
    average. A model needs a number, so those become 0 too — a source with
    no DNS traffic is not distinguishable from one with maximally
    "un-suspicious" DNS by this imputation, which is a real, if minor, loss
    of information relative to the null-preserving batch/streaming feature
    pipeline itself.
    """
    return frame.select(features).fill_null(0).to_numpy()


@dataclass(frozen=True, slots=True)
class IsolationForestBaseline:
    model: IsolationForest
    features: tuple[str, ...]

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        matrix = _to_matrix(frame, self.features)
        raw = self.model.predict(matrix)
        return pl.Series("predicted", raw == -1)

    def decision_score(self, frame: pl.DataFrame) -> pl.Series:
        """Higher means more attack-like. `decision_function` is the
        opposite convention (higher means more normal), so this negates it."""
        matrix = _to_matrix(frame, self.features)
        return pl.Series("score", -self.model.decision_function(matrix))

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "random_state": self.model.random_state,
        }


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    features: tuple[str, ...] = FEATURE_COLUMNS
    contamination: float = 0.01
    n_estimators: int = 100
    random_state: int = 0
    label_column: str = "label"
    benign_label: str = "benign"

    def __post_init__(self) -> None:
        if not 0 < self.contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {self.contamination}")


DEFAULT_ISOLATION_FOREST_CONFIG = IsolationForestConfig()


def fit_isolation_forest(
    train: pl.DataFrame, config: IsolationForestConfig = DEFAULT_ISOLATION_FOREST_CONFIG
) -> IsolationForestBaseline:
    """Fits on the benign rows of `train` only.

    `contamination` calibrates the model's internal decision threshold: it
    is the share of the *benign training data itself* that ends up flagged
    as an outlier, playing the same role `target_fpr` plays for the
    threshold baseline — a lower value makes the model more permissive.
    """
    available = tuple(f for f in config.features if f in train.columns)
    if not available:
        raise ValueError("none of the requested features are present in the frame")

    benign = train.filter(pl.col(config.label_column) == config.benign_label)
    if benign.height == 0:
        raise ValueError("no benign rows in the training split to fit on")

    matrix = _to_matrix(benign, available)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    model.fit(matrix)
    return IsolationForestBaseline(model=model, features=available)


def evaluate_isolation_forest(
    baseline: IsolationForestBaseline,
    frame: pl.DataFrame,
    label_column: str = "label",
    benign_label: str = "benign",
) -> ClassificationMetrics:
    y_true = frame[label_column] != benign_label
    y_pred = baseline.predict(frame)
    return evaluate_predictions(y_true, y_pred)


def evaluate_isolation_forest_by_scenario(
    baseline: IsolationForestBaseline,
    frame: pl.DataFrame,
    scenario_column: str = "scenario",
    label_column: str = "label",
    benign_label: str = "benign",
) -> dict[str, ClassificationMetrics]:
    y_pred = baseline.predict(frame)
    results = {}
    for scenario in frame[scenario_column].unique(maintain_order=True).sort():
        mask = frame[scenario_column] == scenario
        subset_true = frame.filter(mask)[label_column] != benign_label
        subset_pred = y_pred.filter(mask)
        results[scenario] = evaluate_predictions(subset_true, subset_pred)
    return results
