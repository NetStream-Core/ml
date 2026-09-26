import json

import polars as pl
import pytest

from netstream_ml import intensity


def _labels(rows: list[tuple[str, dict[str, object]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [r[0] for r in rows],
            "params": [json.dumps(r[1]) for r in rows],
        }
    )


def test_run_rates_extracts_the_rate_parameter() -> None:
    labels = _labels([("r1", {"rate": 1000, "duration": 20}), ("r2", {"duration": 60})])

    rates = intensity.run_rates(labels)

    assert rates.filter(pl.col("run_id") == "r1")["rate"].item() == 1000.0
    assert rates.filter(pl.col("run_id") == "r2")["rate"].item() is None


def test_run_rates_handles_malformed_params_gracefully() -> None:
    labels = pl.DataFrame({"run_id": ["r1"], "params": ["not json"]})

    rates = intensity.run_rates(labels)

    assert rates["rate"].item() is None


def test_attach_rate_requires_a_run_id_column() -> None:
    features = pl.DataFrame({"src_ip": ["10.0.0.1"]})
    labels = _labels([])

    with pytest.raises(ValueError, match="run_id"):
        intensity.attach_rate(features, labels)


def test_attach_rate_joins_by_run_id() -> None:
    features = pl.DataFrame({"src_ip": ["10.0.0.1", "10.0.0.2"], "run_id": ["r1", "r2"]})
    labels = _labels([("r1", {"rate": 200}), ("r2", {"rate": 5000})])

    joined = intensity.attach_rate(features, labels)

    assert joined.filter(pl.col("run_id") == "r1")["rate"].item() == 200.0
    assert joined.filter(pl.col("run_id") == "r2")["rate"].item() == 5000.0


def test_evaluate_by_rate_groups_by_scenario_and_rate() -> None:
    frame = pl.DataFrame(
        {
            "label": ["syn_flood", "syn_flood", "syn_flood", "benign"],
            "scenario": ["syn_flood", "syn_flood", "syn_flood", "background"],
            "rate": [200.0, 200.0, 5000.0, None],
        }
    )
    predict = pl.Series([False, True, True, False])  # misses one of the low-rate windows

    results = intensity.evaluate_by_rate(predict, frame)

    by_rate = {(r.scenario, r.rate): r for r in results}
    assert (200.0 in [r.rate for r in results if r.scenario == "syn_flood"]) is True
    low = by_rate[("syn_flood", 200.0)]
    high = by_rate[("syn_flood", 5000.0)]
    assert low.n == 2
    assert low.metrics.recall == pytest.approx(0.5)
    assert high.n == 1
    assert high.metrics.recall == pytest.approx(1.0)
    # benign / rate-less rows never form a group
    assert "background" not in [r.scenario for r in results]


def test_evaluate_by_rate_returns_nothing_when_no_row_has_a_rate() -> None:
    frame = pl.DataFrame(
        {
            "label": ["benign"],
            "scenario": ["background"],
            "rate": [None],
        },
        schema={"label": pl.Utf8, "scenario": pl.Utf8, "rate": pl.Float64},
    )
    predict = pl.Series([False])

    assert intensity.evaluate_by_rate(predict, frame) == []
