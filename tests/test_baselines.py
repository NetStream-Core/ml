import polars as pl
import pytest

from netstream_ml import baselines


def _frame(
    packets_total: list[int],
    label: list[str],
    scenario: list[str] | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "packets_total": packets_total,
            "label": label,
            "scenario": scenario or label,
        }
    )


def test_fit_threshold_baseline_rejects_an_invalid_target_fpr() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        baselines.fit_threshold_baseline(_frame([1], ["benign"]), target_fpr=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        baselines.fit_threshold_baseline(_frame([1], ["benign"]), target_fpr=1.0)


def test_fit_threshold_baseline_picks_a_quantile_of_benign_values() -> None:
    benign_values = list(range(1, 101))  # 1..100
    train = _frame(benign_values, ["benign"] * 100)

    baseline = baselines.fit_threshold_baseline(
        train, features_to_consider=["packets_total"], target_fpr=0.10
    )

    assert len(baseline.rules) == 1
    rule = baseline.rules[0]
    assert rule.feature == "packets_total"
    assert rule.threshold == pytest.approx(90.0)


def test_fit_threshold_baseline_skips_a_feature_missing_from_the_frame() -> None:
    train = _frame([1, 2, 3], ["benign", "benign", "benign"])
    baseline = baselines.fit_threshold_baseline(
        train, features_to_consider=["packets_total", "no_such_feature"]
    )
    assert {rule.feature for rule in baseline.rules} == {"packets_total"}


def test_fit_threshold_baseline_skips_a_feature_with_no_benign_observations() -> None:
    train = _frame([1, 2, 3], ["syn_flood", "syn_flood", "syn_flood"])
    baseline = baselines.fit_threshold_baseline(train, features_to_consider=["packets_total"])
    assert baseline.rules == ()


def test_threshold_rule_fires_above_or_below_as_configured() -> None:
    frame = pl.DataFrame({"x": [1.0, 5.0, 10.0]})
    above = baselines.ThresholdRule("x", 5.0, direction="above")
    below = baselines.ThresholdRule("x", 5.0, direction="below")
    assert above.fires(frame).to_list() == [False, False, True]
    assert below.fires(frame).to_list() == [True, False, False]


def test_threshold_rule_treats_a_null_as_never_firing() -> None:
    frame = pl.DataFrame({"x": [None, 5.0, 10.0]}, schema={"x": pl.Float64})
    above = baselines.ThresholdRule("x", 1.0, direction="above")
    below = baselines.ThresholdRule("x", 1.0, direction="below")
    assert above.fires(frame).to_list() == [False, True, True]
    assert below.fires(frame).to_list() == [False, False, False]


def test_baseline_with_no_rules_never_fires() -> None:
    baseline = baselines.ThresholdBaseline(())
    frame = pl.DataFrame({"x": [1, 2, 3]})
    assert baseline.predict(frame).to_list() == [False, False, False]


def test_baseline_predicts_true_if_any_rule_fires() -> None:
    frame = pl.DataFrame({"a": [10, 0, 0], "b": [0, 10, 0]})
    baseline = baselines.ThresholdBaseline(
        (
            baselines.ThresholdRule("a", 5.0),
            baselines.ThresholdRule("b", 5.0),
        )
    )
    assert baseline.predict(frame).to_list() == [True, True, False]


def test_threshold_rule_excess_is_zero_at_the_threshold_and_grows_past_it() -> None:
    frame = pl.DataFrame({"x": [5.0, 10.0, 15.0]})
    rule = baselines.ThresholdRule("x", 10.0, direction="above")
    excess = rule.excess(frame).to_list()
    assert excess[1] == pytest.approx(0.0)
    assert excess[0] < excess[1] < excess[2]


def test_baseline_with_no_rules_scores_everything_at_zero() -> None:
    baseline = baselines.ThresholdBaseline(())
    frame = pl.DataFrame({"x": [1, 2, 3]})
    assert baseline.decision_score(frame).to_list() == [0.0, 0.0, 0.0]


def test_decision_score_takes_the_largest_excess_across_rules() -> None:
    frame = pl.DataFrame({"a": [20.0, 0.0], "b": [0.0, 20.0]})
    baseline = baselines.ThresholdBaseline(
        (
            baselines.ThresholdRule("a", 10.0),
            baselines.ThresholdRule("b", 10.0),
        )
    )
    scores = baseline.decision_score(frame).to_list()
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[0] == pytest.approx(scores[1])


def test_evaluate_predictions_computes_standard_metrics() -> None:
    y_true = pl.Series([True, True, False, False])
    y_pred = pl.Series([True, False, True, False])

    metrics = baselines.evaluate_predictions(y_true, y_pred)

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.fpr == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_evaluate_predictions_handles_a_perfect_classifier() -> None:
    y_true = pl.Series([True, False])
    y_pred = pl.Series([True, False])
    metrics = baselines.evaluate_predictions(y_true, y_pred)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.fpr == 0.0


def test_evaluate_predictions_handles_no_predicted_positives_without_dividing_by_zero() -> None:
    y_true = pl.Series([True, False])
    y_pred = pl.Series([False, False])
    metrics = baselines.evaluate_predictions(y_true, y_pred)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_evaluate_baseline_uses_the_label_column_to_derive_ground_truth() -> None:
    frame = _frame([10, 1, 10, 1], ["syn_flood", "benign", "benign", "benign"])
    baseline = baselines.ThresholdBaseline((baselines.ThresholdRule("packets_total", 5.0),))

    metrics = baselines.evaluate_baseline(baseline, frame)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 2


def test_evaluate_baseline_by_scenario_separates_lookalike_scenarios() -> None:
    frame = _frame(
        [500, 500],
        ["port_scan", "benign"],
        scenario=["port_scan", "admin_scan"],
    )
    baseline = baselines.ThresholdBaseline((baselines.ThresholdRule("packets_total", 100.0),))

    by_scenario = baselines.evaluate_baseline_by_scenario(baseline, frame)

    assert by_scenario["port_scan"].recall == 1.0
    assert by_scenario["admin_scan"].fpr == 1.0
