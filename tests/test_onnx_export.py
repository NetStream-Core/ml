import numpy as np
import polars as pl

from netstream_ml.anomaly import IsolationForestConfig, fit_isolation_forest
from netstream_ml.baselines import fit_threshold_baseline
from netstream_ml.gradient_boosting import GradientBoostingConfig, fit_gradient_boosting
from netstream_ml.onnx_export import run_onnx, to_onnx
from netstream_ml.supervised import RandomForestConfig, fit_random_forest


def _correlated_frame(n: int = 100) -> pl.DataFrame:
    """Benign rows where packets_total and unique_dst_ports scale together,
    plus attack rows that break the correlation — same shape used by the
    other baselines' own test fixtures (test_supervised.py,
    test_gradient_boosting.py), here with a clear margin around every
    baseline's decision boundary so float32 ONNX rounding can't flip a
    `predict` that float64 polars/sklearn got right."""
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
            "packets_total": [500, 550, 600, 650, 700],
            "unique_dst_ports": [1, 1, 1, 1, 1],
            "label": ["port_scan"] * 5,
            "scenario": ["port_scan"] * 5,
        }
    )
    return pl.concat([benign, attack])


def test_threshold_baseline_onnx_matches_the_original() -> None:
    frame = _correlated_frame()
    baseline = fit_threshold_baseline(
        frame, features_to_consider=("packets_total", "unique_dst_ports"), target_fpr=0.01
    )

    model, feature_order, kind = to_onnx(baseline)
    onnx_predict, onnx_score = run_onnx(model, frame, feature_order, kind)

    assert kind == "threshold"
    assert list(baseline.predict(frame)) == list(onnx_predict)
    np.testing.assert_allclose(baseline.decision_score(frame).to_numpy(), onnx_score, atol=1e-4)


def test_threshold_baseline_with_no_rules_exports_an_always_false_model() -> None:
    baseline = fit_threshold_baseline(
        pl.DataFrame({"packets_total": [1, 2], "label": ["benign", "benign"]}),
        features_to_consider=("no_such_feature",),
    )

    model, feature_order, kind = to_onnx(baseline)
    frame = pl.DataFrame({"packets_total": [1.0, 2.0, 3.0]})
    onnx_predict, onnx_score = run_onnx(model, frame, feature_order, kind)

    assert feature_order == ()
    assert list(onnx_predict) == [False, False, False]
    assert list(onnx_score) == [0.0, 0.0, 0.0]


def test_isolation_forest_onnx_matches_the_original() -> None:
    frame = _correlated_frame()
    baseline = fit_isolation_forest(
        frame,
        IsolationForestConfig(
            features=("packets_total", "unique_dst_ports"), contamination=0.1, random_state=0
        ),
    )

    model, feature_order, kind = to_onnx(baseline)
    onnx_predict, onnx_score = run_onnx(model, frame, feature_order, kind)

    assert kind == "isolation_forest"
    assert list(baseline.predict(frame)) == list(onnx_predict)
    np.testing.assert_allclose(baseline.decision_score(frame).to_numpy(), onnx_score, atol=1e-4)


def test_random_forest_onnx_matches_the_original() -> None:
    frame = _correlated_frame()
    baseline = fit_random_forest(
        frame, RandomForestConfig(features=("packets_total", "unique_dst_ports"), random_state=0)
    )

    model, feature_order, kind = to_onnx(baseline)
    onnx_predict, onnx_score = run_onnx(model, frame, feature_order, kind)

    assert kind == "random_forest"
    assert list(baseline.predict(frame)) == list(onnx_predict)
    np.testing.assert_allclose(baseline.decision_score(frame).to_numpy(), onnx_score, atol=1e-4)


def test_gradient_boosting_onnx_matches_the_original() -> None:
    frame = _correlated_frame()
    baseline = fit_gradient_boosting(
        frame,
        GradientBoostingConfig(features=("packets_total", "unique_dst_ports"), random_state=0),
    )

    model, feature_order, kind = to_onnx(baseline)
    onnx_predict, onnx_score = run_onnx(model, frame, feature_order, kind)

    assert kind == "gradient_boosting"
    assert list(baseline.predict(frame)) == list(onnx_predict)
    np.testing.assert_allclose(baseline.decision_score(frame).to_numpy(), onnx_score, atol=1e-4)
