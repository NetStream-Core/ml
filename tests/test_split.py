import datetime as dt

import polars as pl
import pytest

from netstream_ml import split


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        split.SplitRatios(train=0.5, val=0.5, test=0.5)


def test_ratios_reject_a_negative_share() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        split.SplitRatios(train=1.5, val=0.0, test=-0.5)


def test_run_splits_gives_every_run_id_a_split() -> None:
    runs = {"syn_flood": [f"r{i}" for i in range(10)], "benign": [f"b{i}" for i in range(5)]}
    assignment = split.run_splits(runs, seed=0)

    all_runs = [r for group in runs.values() for r in group]
    assert set(assignment) == set(all_runs)
    assert set(assignment.values()) <= set(split.SPLITS)


def test_run_splits_is_deterministic_for_the_same_seed() -> None:
    runs = {"port_scan": [f"r{i}" for i in range(12)]}
    first = split.run_splits(runs, seed=7)
    second = split.run_splits(runs, seed=7)
    assert first == second


def test_run_splits_can_change_with_a_different_seed() -> None:
    runs = {"port_scan": [f"r{i}" for i in range(30)]}
    first = split.run_splits(runs, seed=1)
    second = split.run_splits(runs, seed=2)
    assert first != second


def test_a_scenario_with_at_least_three_runs_appears_in_every_split() -> None:
    runs = {"c2_beacon": [f"r{i}" for i in range(6)]}
    assignment = split.run_splits(runs, seed=0)
    assert set(assignment.values()) == set(split.SPLITS)


def test_run_splits_proportions_track_the_ratios_as_runs_grow() -> None:
    runs = {"syn_flood": [f"r{i}" for i in range(300)]}
    assignment = split.run_splits(runs, seed=0)
    counts = {name: sum(1 for v in assignment.values() if v == name) for name in split.SPLITS}
    assert counts["train"] / 300 == pytest.approx(0.6, abs=0.05)
    assert counts["val"] / 300 == pytest.approx(0.2, abs=0.05)
    assert counts["test"] / 300 == pytest.approx(0.2, abs=0.05)


def _ts(seconds: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1) + dt.timedelta(seconds=seconds)


def test_add_split_column_uses_the_run_assignment_when_a_run_id_is_present() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "ts": [_ts(0), _ts(1), _ts(2)],
        }
    )
    assignment = {"r1": "train", "r2": "test"}
    result = split.add_split_column(frame, assignment)
    assert result["split"].to_list() == ["train", "train", "test"]


def test_add_split_column_buckets_background_rows_by_time() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["", "", ""],
            "ts": [_ts(0), _ts(1), _ts(1000)],
        }
    )
    result = split.add_split_column(frame, {}, split.SplitConfig(bucket_ms=30_000))
    labels = result["split"].to_list()
    assert labels[0] == labels[1]
    assert all(label in split.SPLITS for label in labels)


def test_check_run_leakage_passes_a_clean_split() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "split": ["train", "train", "test"],
        }
    )
    report = split.check_run_leakage(frame)
    assert report.ok
    assert report.leaking_run_ids == []
    assert report.split_counts == {"train": 2, "test": 1}


def test_check_run_leakage_catches_a_run_split_across_train_and_test() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "split": ["train", "test", "test"],
        }
    )
    report = split.check_run_leakage(frame)
    assert not report.ok
    assert report.leaking_run_ids == ["r1"]


def test_check_run_leakage_ignores_background_rows_without_a_run_id() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["", "", "r1"],
            "split": ["train", "test", "train"],
        }
    )
    report = split.check_run_leakage(frame)
    assert report.ok


def test_run_ids_by_scenario_groups_labels() -> None:
    labels = pl.DataFrame(
        {
            "scenario": ["syn_flood", "syn_flood", "benign"],
            "run_id": ["r1", "r2", "r3"],
        }
    )
    grouped = split.run_ids_by_scenario(labels)
    assert set(grouped["syn_flood"]) == {"r1", "r2"}
    assert grouped["benign"] == ["r3"]
