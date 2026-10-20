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


def test_a_scenario_with_exactly_three_runs_appears_in_every_split() -> None:
    # Regression test: at exactly 3 runs and the default 60/20/20 ratios,
    # plain largest-remainder rounding gives (2, 1, 0) — both val and
    # test's fractional parts (0.6) round down, and the one remaining slot
    # goes to train's larger fractional part first, leaving test with
    # nothing. Found in practice (M11's drift experiment, both scenarios
    # had exactly 3 runs before a 4th was added as a workaround) before
    # this reservation logic existed.
    runs = {"dns_tunnel": [f"r{i}" for i in range(3)]}
    assignment = split.run_splits(runs, seed=0)
    assert set(assignment.values()) == set(split.SPLITS)


@pytest.mark.parametrize("seed", range(20))
def test_three_runs_appear_in_every_split_across_many_seeds(seed: int) -> None:
    # The specific case above could coincidentally pass for one seed even
    # with the bug if the hash-based shuffle happened to break the tie
    # differently — the real guarantee has to hold regardless of seed.
    runs = {"dns_tunnel": [f"r{i}" for i in range(3)]}
    assignment = split.run_splits(runs, seed=seed)
    assert set(assignment.values()) == set(split.SPLITS)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10, 15, 50, 100, 300])
def test_run_splits_assigns_every_run_for_a_range_of_scenario_sizes(n: int) -> None:
    # Regression test for a bug introduced (and caught here) while fixing
    # the n=3 coverage gap above: a first attempt at reserving one run per
    # split distributed the rest via `order[:remaining]`, which only adds
    # at most 1 per split (`order` has 3 entries) — silently dropping
    # almost every run once a scenario had more than 5 runs. 300 runs
    # with only 2 landing in train is exactly how that surfaced.
    runs = {"syn_flood": [f"r{i}" for i in range(n)]}
    assignment = split.run_splits(runs, seed=0)
    assert set(assignment) == set(runs["syn_flood"])
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


def test_scenario_split_coverage_is_empty_when_run_splits_covered_everything() -> None:
    runs = {"dns_tunnel": [f"r{i}" for i in range(4)], "benign": [f"b{i}" for i in range(2)]}
    assignment = split.run_splits(runs, seed=0)

    gaps = split.scenario_split_coverage(runs, assignment)

    assert gaps == {}


def test_scenario_split_coverage_flags_a_split_with_zero_runs() -> None:
    # A hand-built assignment that skips `test` entirely for a scenario
    # with enough runs to have covered it — this is what
    # `_largest_remainder_with_minimum_coverage` is supposed to prevent
    # `run_splits` from ever producing; this test checks the detector
    # itself works, independent of whether that guarantee holds.
    runs = {"dns_tunnel": ["r0", "r1", "r2"]}
    assignment = {"r0": "train", "r1": "train", "r2": "val"}

    gaps = split.scenario_split_coverage(runs, assignment)

    assert gaps == {"dns_tunnel": ["test"]}


def test_scenario_split_coverage_skips_a_scenario_with_too_few_runs_to_cover_everything() -> None:
    # 2 runs can never cover 3 splits regardless of allocation strategy —
    # this is the known, documented limitation, not something to flag.
    runs = {"dns_tunnel": ["r0", "r1"]}
    assignment = {"r0": "train", "r1": "val"}

    gaps = split.scenario_split_coverage(runs, assignment)

    assert gaps == {}
