import polars as pl
import pytest

from netstream_ml import anomaly
from netstream_ml.anomaly import IsolationForestConfig


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


def test_isolation_forest_config_rejects_an_invalid_contamination() -> None:
    with pytest.raises(ValueError, match=r"contamination must be in \(0, 0\.5\]"):
        IsolationForestConfig(features=("packets_total",), contamination=0.0)
    with pytest.raises(ValueError, match=r"contamination must be in \(0, 0\.5\]"):
        IsolationForestConfig(features=("packets_total",), contamination=0.6)


def test_fit_isolation_forest_fails_without_benign_rows() -> None:
    train = _frame([1, 2, 3], ["syn_flood", "syn_flood", "syn_flood"])
    config = IsolationForestConfig(features=("packets_total",))
    with pytest.raises(ValueError, match="no benign rows"):
        anomaly.fit_isolation_forest(train, config)


def test_fit_isolation_forest_fails_when_no_requested_feature_is_present() -> None:
    train = _frame([1, 2, 3], ["benign", "benign", "benign"])
    config = IsolationForestConfig(features=("no_such_feature",))
    with pytest.raises(ValueError, match="none of the requested features"):
        anomaly.fit_isolation_forest(train, config)


def test_fit_isolation_forest_uses_only_features_present_in_the_frame() -> None:
    train = _frame(list(range(1, 51)), ["benign"] * 50)
    config = IsolationForestConfig(features=("packets_total", "no_such_feature"), contamination=0.1)

    model = anomaly.fit_isolation_forest(train, config)

    assert model.features == ("packets_total",)


def _correlated_benign_frame(n: int = 100) -> pl.DataFrame:
    # packets_total and unique_dst_ports scale together for benign traffic:
    # a busier host also talks to proportionally more distinct destinations.
    # Neither column alone is unusual for an "attack" row that breaks this
    # correlation — only the *combination* is, which is exactly what a
    # per-feature threshold cannot catch but Isolation Forest can.
    values = list(range(1, n + 1))
    return pl.DataFrame(
        {
            "packets_total": values,
            "unique_dst_ports": values,
            "label": ["benign"] * n,
            "scenario": ["background"] * n,
        }
    )


def test_isolation_forest_flags_an_off_manifold_feature_combination() -> None:
    train = _correlated_benign_frame()
    config = IsolationForestConfig(
        features=("packets_total", "unique_dst_ports"), contamination=0.05, random_state=0
    )
    model = anomaly.fit_isolation_forest(train, config)

    # mid-range packets_total paired with a single destination port: each
    # value is ordinary on its own, the pairing is not.
    attack = pl.DataFrame(
        {
            "packets_total": [50, 50, 50],
            "unique_dst_ports": [1, 1, 1],
            "label": ["port_scan"] * 3,
            "scenario": ["port_scan"] * 3,
        }
    )
    metrics = anomaly.evaluate_isolation_forest(model, pl.concat([train, attack]))

    assert metrics.recall == 1.0
    assert metrics.fpr == pytest.approx(0.05, abs=0.02)


def test_decision_score_ranks_the_off_manifold_attack_above_benign_rows() -> None:
    train = _correlated_benign_frame()
    config = IsolationForestConfig(
        features=("packets_total", "unique_dst_ports"), contamination=0.05, random_state=0
    )
    model = anomaly.fit_isolation_forest(train, config)

    attack = pl.DataFrame(
        {
            "packets_total": [50, 50, 50],
            "unique_dst_ports": [1, 1, 1],
            "label": ["port_scan"] * 3,
            "scenario": ["port_scan"] * 3,
        }
    )
    scores = model.decision_score(pl.concat([train, attack]))

    benign_scores = scores.slice(0, train.height).to_numpy()
    attack_scores = scores.slice(train.height, attack.height).to_numpy()
    assert attack_scores.min() > benign_scores.mean()


def test_evaluate_isolation_forest_by_scenario_separates_lookalike_scenarios() -> None:
    train = _correlated_benign_frame()
    config = IsolationForestConfig(
        features=("packets_total", "unique_dst_ports"), contamination=0.05, random_state=0
    )
    model = anomaly.fit_isolation_forest(train, config)

    test = pl.concat(
        [
            train,
            pl.DataFrame(
                {
                    "packets_total": [50, 50],
                    "unique_dst_ports": [1, 50],
                    "label": ["port_scan", "benign"],
                    "scenario": ["port_scan", "admin_scan"],
                }
            ),
        ]
    )

    by_scenario = anomaly.evaluate_isolation_forest_by_scenario(model, test)

    assert by_scenario["port_scan"].recall == 1.0
    # same packets_total as the attack row, but on the benign manifold
    # (ports scale with packets too) — unlike a single-feature threshold on
    # packets_total alone, the combination gives this one away as ordinary.
    assert by_scenario["admin_scan"].fpr == 0.0


def test_to_dict_reports_model_configuration() -> None:
    train = _frame(list(range(1, 51)), ["benign"] * 50)
    config = IsolationForestConfig(
        features=("packets_total",), contamination=0.05, n_estimators=17, random_state=3
    )

    model = anomaly.fit_isolation_forest(train, config)
    as_dict = model.to_dict()

    assert as_dict["features"] == ["packets_total"]
    assert as_dict["n_estimators"] == 17
    assert as_dict["contamination"] == 0.05
    assert as_dict["random_state"] == 3
