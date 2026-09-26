import polars as pl
import pytest

from netstream_ml import evaluation
from netstream_ml.baselines import ClassificationMetrics


def _metrics(f1: float) -> ClassificationMetrics:
    return ClassificationMetrics(
        precision=f1,
        recall=f1,
        f1=f1,
        fpr=0.0,
        accuracy=f1,
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        true_negatives=0,
    )


def test_precision_recall_curve_scores_a_perfect_separator_at_one() -> None:
    y_true = pl.Series([False, False, True, True])
    y_score = pl.Series([0.1, 0.2, 0.8, 0.9])

    curve = evaluation.precision_recall_curve(y_true, y_score)

    assert curve.average_precision == pytest.approx(1.0)
    assert max(curve.precision) == pytest.approx(1.0)


def test_precision_recall_curve_scores_a_random_separator_lower() -> None:
    y_true = pl.Series([False, True, False, True])
    y_score = pl.Series([0.5, 0.5, 0.5, 0.5])  # no separation at all

    curve = evaluation.precision_recall_curve(y_true, y_score)

    assert curve.average_precision == pytest.approx(0.5, abs=0.01)


def test_seed_sensitivity_rejects_an_empty_seed_list() -> None:
    with pytest.raises(ValueError, match="seeds must not be empty"):
        evaluation.seed_sensitivity(lambda seed: _metrics(0.5), seeds=())


def test_seed_sensitivity_summarises_a_constant_metric_with_zero_spread() -> None:
    result = evaluation.seed_sensitivity(lambda seed: _metrics(0.75), seeds=(0, 1, 2))

    assert result.values == (0.75, 0.75, 0.75)
    assert result.mean == pytest.approx(0.75)
    assert result.std == pytest.approx(0.0)
    assert result.minimum == pytest.approx(0.75)
    assert result.maximum == pytest.approx(0.75)


def test_seed_sensitivity_reports_the_actual_spread() -> None:
    by_seed = {0: 0.5, 1: 0.7, 2: 0.9}
    result = evaluation.seed_sensitivity(lambda seed: _metrics(by_seed[seed]), seeds=(0, 1, 2))

    assert result.values == (0.5, 0.7, 0.9)
    assert result.mean == pytest.approx(0.7)
    assert result.minimum == pytest.approx(0.5)
    assert result.maximum == pytest.approx(0.9)


def test_mcnemar_test_finds_no_significant_difference_when_models_agree() -> None:
    y_true = pl.Series([True, False, True, False, True, False])
    pred_a = pl.Series([True, False, True, False, True, False])
    pred_b = pl.Series([True, False, True, False, True, False])

    result = evaluation.mcnemar_test(y_true, pred_a, pred_b)

    assert result.a_only_correct == 0
    assert result.b_only_correct == 0
    assert result.p_value == 1.0


def test_mcnemar_test_finds_a_significant_difference_when_one_model_is_clearly_better() -> None:
    y_true = pl.Series([True] * 20 + [False] * 20)
    pred_a = pl.Series([True] * 20 + [False] * 20)  # perfect
    pred_b = pl.Series([False] * 20 + [False] * 20)  # misses every attack

    result = evaluation.mcnemar_test(y_true, pred_a, pred_b)

    assert result.a_only_correct == 20
    assert result.b_only_correct == 0
    assert result.p_value < 0.01


def test_mcnemar_test_is_symmetric_in_the_p_value() -> None:
    y_true = pl.Series([True, True, True, False, False])
    pred_a = pl.Series([True, True, False, False, False])
    pred_b = pl.Series([True, False, True, True, False])

    forward = evaluation.mcnemar_test(y_true, pred_a, pred_b)
    backward = evaluation.mcnemar_test(y_true, pred_b, pred_a)

    assert forward.a_only_correct == backward.b_only_correct
    assert forward.p_value == pytest.approx(backward.p_value)
