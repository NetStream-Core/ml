import polars as pl
import pytest

from netstream_ml import gradient_boosting
from netstream_ml.gradient_boosting import GradientBoostingConfig


def _correlated_frame(n: int = 100) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Benign rows where packets_total and unique_dst_ports scale together,
    plus a handful of attack rows that break the correlation (as in
    test_supervised.py) — both splits also get labelled attack rows, since
    a supervised model is allowed to see them during training."""
    values = list(range(1, n + 1))
    benign = pl.DataFrame(
        {
            "packets_total": values,
            "unique_dst_ports": values,
            "label": ["benign"] * n,
            "scenario": ["background"] * n,
        }
    )
    attack = pl.DataFrame(
        {
            "packets_total": [50, 55, 60, 65, 70],
            "unique_dst_ports": [1, 1, 1, 1, 1],
            "label": ["port_scan"] * 5,
            "scenario": ["port_scan"] * 5,
        }
    )
    return benign, attack


def _labelled_dataset() -> pl.DataFrame:
    benign, attack = _correlated_frame()
    return pl.concat([benign, attack])


def test_fit_gradient_boosting_fails_when_no_requested_feature_is_present() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(features=("no_such_feature",))
    with pytest.raises(ValueError, match="none of the requested features"):
        gradient_boosting.fit_gradient_boosting(train, config)


def test_fit_gradient_boosting_uses_only_features_present_in_the_frame() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(features=("packets_total", "no_such_feature"))

    model = gradient_boosting.fit_gradient_boosting(train, config)

    assert model.features == ("packets_total",)


def test_fit_gradient_boosting_fails_on_an_empty_training_frame() -> None:
    train = _labelled_dataset().clear()
    config = GradientBoostingConfig(features=("packets_total",))
    with pytest.raises(ValueError, match="no rows in the training split"):
        gradient_boosting.fit_gradient_boosting(train, config)


def test_gradient_boosting_learns_the_off_manifold_combination() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = gradient_boosting.fit_gradient_boosting(train, config)

    metrics = gradient_boosting.evaluate_gradient_boosting(model, train)

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_evaluate_gradient_boosting_by_scenario_separates_lookalike_scenarios() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = gradient_boosting.fit_gradient_boosting(train, config)

    test = pl.concat(
        [
            train,
            pl.DataFrame(
                {
                    "packets_total": [50],
                    "unique_dst_ports": [50],
                    "label": ["benign"],
                    "scenario": ["admin_scan"],
                }
            ),
        ]
    )

    by_scenario = gradient_boosting.evaluate_gradient_boosting_by_scenario(model, test)

    assert by_scenario["port_scan"].recall == 1.0
    assert by_scenario["admin_scan"].fpr == 0.0


def test_to_dict_reports_model_configuration() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(
        features=("packets_total",),
        n_estimators=17,
        max_depth=4,
        learning_rate=0.2,
        random_state=3,
    )

    model = gradient_boosting.fit_gradient_boosting(train, config)
    as_dict = model.to_dict()

    assert as_dict["features"] == ["packets_total"]
    assert as_dict["n_estimators"] == 17
    assert as_dict["max_depth"] == 4
    assert as_dict["learning_rate"] == pytest.approx(0.2)
    assert as_dict["random_state"] == 3


def test_tune_gradient_boosting_fails_without_validation_rows() -> None:
    train = _labelled_dataset()
    empty_val = train.clear()
    with pytest.raises(ValueError, match="no rows in the validation split"):
        gradient_boosting.tune_gradient_boosting(train, empty_val, features=("packets_total",))


def test_tune_gradient_boosting_picks_a_config_that_scores_well_on_validation() -> None:
    train = _labelled_dataset()
    val = _labelled_dataset()

    result = gradient_boosting.tune_gradient_boosting(
        train, val, features=("packets_total", "unique_dst_ports")
    )

    assert result.candidates_tried > 1
    assert result.val_metrics.f1 > 0.9
    model = gradient_boosting.fit_gradient_boosting(train, result.config)
    assert model.features == ("packets_total", "unique_dst_ports")


def test_decision_score_ranks_attacks_above_benign_rows() -> None:
    train = _labelled_dataset()
    config = GradientBoostingConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = gradient_boosting.fit_gradient_boosting(train, config)

    scores = model.decision_score(train)

    benign_scores = scores.filter(train["label"] == "benign").to_numpy()
    attack_scores = scores.filter(train["label"] != "benign").to_numpy()
    assert attack_scores.min() > benign_scores.max()
