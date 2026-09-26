"""A Random Forest baseline: unlike the threshold and Isolation Forest
baselines, it is trained on attack examples too, not only benign ones. That
is the whole point of comparing it against them — the lab has enough
labelled attack windows to try a supervised model, and the question is
whether seeing real attacks during training buys more than an unsupervised
model gets from a good feature set alone.

Hyperparameters are picked by a small grid search scored on the
**validation** split, never on test, so the reported test metrics are not
themselves the result of tuning against the same data they measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import polars as pl
from sklearn.ensemble import RandomForestClassifier

from netstream_ml.anomaly import _to_matrix
from netstream_ml.baselines import ClassificationMetrics, evaluate_predictions
from netstream_ml.features import FEATURE_COLUMNS

ClassWeight = Literal["balanced"] | None

# Kept small and fixed rather than exposed as CLI flags: the lab dataset is a
# couple of thousand rows, so this grid fits in well under a second, and a
# handful of sane candidates is enough to show whether tuning matters at all.
N_ESTIMATORS_GRID = (100, 300)
MAX_DEPTH_GRID: tuple[int | None, ...] = (None, 10, 20)
CLASS_WEIGHT_GRID: tuple[ClassWeight, ...] = (None, "balanced")


@dataclass(frozen=True, slots=True)
class RandomForestConfig:
    features: tuple[str, ...] = FEATURE_COLUMNS
    n_estimators: int = 300
    max_depth: int | None = None
    class_weight: ClassWeight = "balanced"
    random_state: int = 0
    label_column: str = "label"
    benign_label: str = "benign"


DEFAULT_RANDOM_FOREST_CONFIG = RandomForestConfig()


@dataclass(frozen=True, slots=True)
class RandomForestBaseline:
    model: RandomForestClassifier
    features: tuple[str, ...]

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        matrix = _to_matrix(frame, self.features)
        raw = self.model.predict(matrix)
        return pl.Series("predicted", raw.astype(bool))

    def decision_score(self, frame: pl.DataFrame) -> pl.Series:
        """The model's predicted probability of the attack (`True`) class."""
        matrix = _to_matrix(frame, self.features)
        attack_index = list(self.model.classes_).index(True)
        return pl.Series("score", self.model.predict_proba(matrix)[:, attack_index])

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "class_weight": self.model.class_weight,
            "random_state": self.model.random_state,
        }


def fit_random_forest(
    train: pl.DataFrame, config: RandomForestConfig = DEFAULT_RANDOM_FOREST_CONFIG
) -> RandomForestBaseline:
    """Fits on every row of `train`, benign and attack alike."""
    available = tuple(f for f in config.features if f in train.columns)
    if not available:
        raise ValueError("none of the requested features are present in the frame")
    if train.height == 0:
        raise ValueError("no rows in the training split to fit on")

    matrix = _to_matrix(train, available)
    y = (train[config.label_column] != config.benign_label).to_numpy()
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    model.fit(matrix, y)
    return RandomForestBaseline(model=model, features=available)


def evaluate_random_forest(
    baseline: RandomForestBaseline,
    frame: pl.DataFrame,
    label_column: str = "label",
    benign_label: str = "benign",
) -> ClassificationMetrics:
    y_true = frame[label_column] != benign_label
    y_pred = baseline.predict(frame)
    return evaluate_predictions(y_true, y_pred)


def evaluate_random_forest_by_scenario(
    baseline: RandomForestBaseline,
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


@dataclass(frozen=True, slots=True)
class TuningResult:
    config: RandomForestConfig
    val_metrics: ClassificationMetrics
    candidates_tried: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "class_weight": self.config.class_weight,
            "val_f1": self.val_metrics.f1,
            "candidates_tried": self.candidates_tried,
        }


def tune_random_forest(
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: tuple[str, ...] = FEATURE_COLUMNS,
    random_state: int = 0,
) -> TuningResult:
    """A small grid search over `N_ESTIMATORS_GRID` x `MAX_DEPTH_GRID` x
    `CLASS_WEIGHT_GRID`, scored by F1 on `val`. Returns the best config
    fitted on `train` alone — call `fit_random_forest` again on train+val
    combined afterwards if the final model should use all non-test data."""
    if val.height == 0:
        raise ValueError("no rows in the validation split to score candidates on")

    best: tuple[RandomForestConfig, ClassificationMetrics] | None = None
    candidates = 0
    for n_estimators, max_depth, class_weight in product(
        N_ESTIMATORS_GRID, MAX_DEPTH_GRID, CLASS_WEIGHT_GRID
    ):
        candidates += 1
        config = RandomForestConfig(
            features=features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
        )
        model = fit_random_forest(train, config)
        metrics = evaluate_random_forest(model, val)
        if best is None or metrics.f1 > best[1].f1:
            best = (config, metrics)

    assert best is not None  # candidates is never empty
    return TuningResult(config=best[0], val_metrics=best[1], candidates_tried=candidates)
