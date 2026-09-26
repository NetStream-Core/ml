import polars as pl
import pytest

from netstream_ml import supervised
from netstream_ml.supervised import RandomForestConfig


def _correlated_frame(n: int = 100) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Benign rows where packets_total and unique_dst_ports scale together,
    plus a handful of attack rows that break the correlation (as in
    test_anomaly.py) — but this time both splits also get labelled attack
    rows, since a supervised model is allowed to see them during training."""
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


def test_fit_random_forest_fails_when_no_requested_feature_is_present() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(features=("no_such_feature",))
    with pytest.raises(ValueError, match="none of the requested features"):
        supervised.fit_random_forest(train, config)


def test_fit_random_forest_uses_only_features_present_in_the_frame() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(features=("packets_total", "no_such_feature"))

    model = supervised.fit_random_forest(train, config)

    assert model.features == ("packets_total",)


def test_random_forest_learns_the_off_manifold_combination() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = supervised.fit_random_forest(train, config)

    metrics = supervised.evaluate_random_forest(model, train)

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_evaluate_random_forest_by_scenario_separates_lookalike_scenarios() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = supervised.fit_random_forest(train, config)

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

    by_scenario = supervised.evaluate_random_forest_by_scenario(model, test)

    assert by_scenario["port_scan"].recall == 1.0
    assert by_scenario["admin_scan"].fpr == 0.0


def test_to_dict_reports_model_configuration() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(
        features=("packets_total",),
        n_estimators=17,
        max_depth=5,
        class_weight="balanced",
        random_state=3,
    )

    model = supervised.fit_random_forest(train, config)
    as_dict = model.to_dict()

    assert as_dict["features"] == ["packets_total"]
    assert as_dict["n_estimators"] == 17
    assert as_dict["max_depth"] == 5
    assert as_dict["class_weight"] == "balanced"
    assert as_dict["random_state"] == 3


def test_tune_random_forest_fails_without_validation_rows() -> None:
    train = _labelled_dataset()
    empty_val = train.clear()
    with pytest.raises(ValueError, match="no rows in the validation split"):
        supervised.tune_random_forest(train, empty_val, features=("packets_total",))


def test_tune_random_forest_picks_a_config_that_scores_well_on_validation() -> None:
    train = _labelled_dataset()
    val = _labelled_dataset()

    result = supervised.tune_random_forest(
        train, val, features=("packets_total", "unique_dst_ports")
    )

    assert result.candidates_tried > 1
    assert result.val_metrics.f1 > 0.9
    model = supervised.fit_random_forest(train, result.config)
    assert model.features == ("packets_total", "unique_dst_ports")


def test_decision_score_ranks_attacks_above_benign_rows() -> None:
    train = _labelled_dataset()
    config = RandomForestConfig(
        features=("packets_total", "unique_dst_ports"), n_estimators=50, random_state=0
    )
    model = supervised.fit_random_forest(train, config)

    scores = model.decision_score(train)

    benign_scores = scores.filter(train["label"] == "benign").to_numpy()
    attack_scores = scores.filter(train["label"] != "benign").to_numpy()
    assert attack_scores.min() > benign_scores.max()
