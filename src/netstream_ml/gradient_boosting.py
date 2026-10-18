"""An XGBoost baseline: the same supervised setup as `supervised.py`'s
Random Forest (trained on attack examples too, hyperparameters picked by a
grid search scored on validation, never on test), but boosted trees instead
of bagged ones. The comparison this enables is not "is gradient boosting
better than a forest in general" — on a dataset this size that question
has no stable answer — but whether the two agree on which attacks are hard,
which is a more honest signal than either model's score alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import polars as pl
from xgboost import XGBClassifier

from netstream_ml.anomaly import _to_matrix
from netstream_ml.baselines import ClassificationMetrics, evaluate_predictions
from netstream_ml.features import FEATURE_COLUMNS

# Kept small and fixed rather than exposed as CLI flags, same reasoning as
# supervised.py's grid: the lab dataset is a couple of thousand rows, so this
# fits in well under a second.
N_ESTIMATORS_GRID = (100, 300)
MAX_DEPTH_GRID = (3, 6)
LEARNING_RATE_GRID = (0.1, 0.3)


@dataclass(frozen=True, slots=True)
class GradientBoostingConfig:
    features: tuple[str, ...] = FEATURE_COLUMNS
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 0
    label_column: str = "label"
    benign_label: str = "benign"


DEFAULT_GRADIENT_BOOSTING_CONFIG = GradientBoostingConfig()


@dataclass(frozen=True, slots=True)
class GradientBoostingBaseline:
    model: XGBClassifier
    features: tuple[str, ...]

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        matrix = _to_matrix(frame, self.features)
        raw = self.model.predict(matrix)
        return pl.Series("predicted", raw.astype(bool))

    def decision_score(self, frame: pl.DataFrame) -> pl.Series:
        """The model's predicted probability of the attack (`True`) class."""
        matrix = _to_matrix(frame, self.features)
        return pl.Series("score", self.model.predict_proba(matrix)[:, 1])

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "random_state": self.model.random_state,
        }


def fit_gradient_boosting(
    train: pl.DataFrame,
    config: GradientBoostingConfig = DEFAULT_GRADIENT_BOOSTING_CONFIG,
) -> GradientBoostingBaseline:
    """Fits on every row of `train`, benign and attack alike."""
    available = tuple(f for f in config.features if f in train.columns)
    if not available:
        raise ValueError("none of the requested features are present in the frame")
    if train.height == 0:
        raise ValueError("no rows in the training split to fit on")

    matrix = _to_matrix(train, available)
    y = (train[config.label_column] != config.benign_label).to_numpy()
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        eval_metric="logloss",
    )
    model.fit(matrix, y)
    return GradientBoostingBaseline(model=model, features=available)


def evaluate_gradient_boosting(
    baseline: GradientBoostingBaseline,
    frame: pl.DataFrame,
    label_column: str = "label",
    benign_label: str = "benign",
) -> ClassificationMetrics:
    y_true = frame[label_column] != benign_label
    y_pred = baseline.predict(frame)
    return evaluate_predictions(y_true, y_pred)


def evaluate_gradient_boosting_by_scenario(
    baseline: GradientBoostingBaseline,
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
class GradientBoostingTuningResult:
    config: GradientBoostingConfig
    val_metrics: ClassificationMetrics
    candidates_tried: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "val_f1": self.val_metrics.f1,
            "candidates_tried": self.candidates_tried,
        }


def tune_gradient_boosting(
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: tuple[str, ...] = FEATURE_COLUMNS,
    random_state: int = 0,
) -> GradientBoostingTuningResult:
    """A small grid search over `N_ESTIMATORS_GRID` x `MAX_DEPTH_GRID` x
    `LEARNING_RATE_GRID`, scored by F1 on `val`. Returns the best config
    fitted on `train` alone — call `fit_gradient_boosting` again on
    train+val combined afterwards if the final model should use all
    non-test data."""
    if val.height == 0:
        raise ValueError("no rows in the validation split to score candidates on")

    best: tuple[GradientBoostingConfig, ClassificationMetrics] | None = None
    candidates = 0
    for n_estimators, max_depth, learning_rate in product(
        N_ESTIMATORS_GRID, MAX_DEPTH_GRID, LEARNING_RATE_GRID
    ):
        candidates += 1
        config = GradientBoostingConfig(
            features=features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        model = fit_gradient_boosting(train, config)
        metrics = evaluate_gradient_boosting(model, val)
        if best is None or metrics.f1 > best[1].f1:
            best = (config, metrics)

    assert best is not None  # candidates is never empty
    return GradientBoostingTuningResult(
        config=best[0], val_metrics=best[1], candidates_tried=candidates
    )
