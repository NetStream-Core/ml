"""Assigns dataset rows to train/val/test without leaking a run or a burst of
background traffic across splits.

Every attack run and every explicit benign profile run gets a `run_id` from the
lab (see `deploy`'s `labeled_flows`/`labeled_dns`). Ambient background traffic
between runs has no `run_id`, so it is grouped into fixed-size time buckets and
each bucket is assigned as a whole. This is an approximation: a background flow
that straddles a bucket boundary can still have some of its interval records on
either side of a split. It is a much smaller leak than putting one run's
records in both train and test, and cheap to compute, but it is not perfect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLITS = (TRAIN, VAL, TEST)

BACKGROUND_BUCKET_MS = 30_000


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value < 0:
                raise ValueError(f"{name} ratio must not be negative, got {value}")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.train, self.val, self.test)


DEFAULT_RATIOS = SplitRatios()


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Groups the knobs `add_split_column` needs beyond the frame and the run
    assignment, so the function itself stays under the arguments-count limit."""

    ts_column: str = "ts"
    run_id_column: str = "run_id"
    seed: int = 0
    ratios: SplitRatios = DEFAULT_RATIOS
    bucket_ms: int = BACKGROUND_BUCKET_MS


DEFAULT_SPLIT_CONFIG = SplitConfig()


def _bucket_hash(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _hash_assign(key: str, seed: int, ratios: SplitRatios) -> str:
    position = _bucket_hash(key, seed)
    if position < ratios.train:
        return TRAIN
    if position < ratios.train + ratios.val:
        return VAL
    return TEST


def _largest_remainder(n: int, ratios: SplitRatios) -> tuple[int, int, int]:
    raw = [n * r for r in ratios.as_tuple()]
    base = [int(x) for x in raw]
    remainder = n - sum(base)
    order = sorted(range(3), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base[0], base[1], base[2]


def run_splits(
    run_ids_by_scenario: Mapping[str, list[str]],
    seed: int = 0,
    ratios: SplitRatios = DEFAULT_RATIOS,
) -> dict[str, str]:
    """Assigns every run_id to train, val or test, one scenario at a time.

    Runs of the same scenario are shuffled deterministically (seeded) and then
    split by the largest-remainder method, so a scenario with at least three
    runs is represented in every split, and the overall proportions still
    track `ratios` as the number of runs grows.
    """
    assignment: dict[str, str] = {}
    for scenario in sorted(run_ids_by_scenario):
        run_ids = sorted(set(run_ids_by_scenario[scenario]))
        order = sorted(run_ids, key=lambda run_id: _bucket_hash(f"{seed}:{scenario}:{run_id}", 0))
        counts = _largest_remainder(len(order), ratios)
        cursor = 0
        for split, count in zip(SPLITS, counts, strict=True):
            for run_id in order[cursor : cursor + count]:
                assignment[run_id] = split
            cursor += count
    return assignment


def run_ids_by_scenario(labels: pl.DataFrame) -> dict[str, list[str]]:
    grouped = labels.group_by("scenario").agg(pl.col("run_id"))
    return {row["scenario"]: row["run_id"] for row in grouped.iter_rows(named=True)}


def add_split_column(
    frame: pl.DataFrame,
    run_assignment: Mapping[str, str],
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> pl.DataFrame:
    """Adds a `split` column: `run_assignment[run_id]` where a run_id is present,
    otherwise the assignment of that row's background time bucket."""
    bucket = (pl.col(config.ts_column).dt.epoch("ms") // config.bucket_ms).alias("_bucket")
    with_bucket = frame.with_columns(bucket)

    buckets = with_bucket.select("_bucket").unique().to_series().to_list()
    bucket_assignment = {
        bucket: _hash_assign(f"background:{bucket}", config.seed, config.ratios)
        for bucket in buckets
    }

    run_split = pl.col(config.run_id_column).replace_strict(
        dict(run_assignment), default=None, return_dtype=pl.Utf8
    )
    bucket_split = pl.col("_bucket").replace_strict(
        bucket_assignment, default=None, return_dtype=pl.Utf8
    )

    run_id = pl.col(config.run_id_column)
    has_run = run_id.is_not_null() & (run_id != "")
    split = pl.when(has_run).then(run_split).otherwise(bucket_split).alias("split")

    return with_bucket.with_columns(split).drop("_bucket")


@dataclass(frozen=True, slots=True)
class LeakageReport:
    split_counts: dict[str, int]
    leaking_run_ids: list[str]

    @property
    def ok(self) -> bool:
        return not self.leaking_run_ids


def check_run_leakage(frame: pl.DataFrame, run_id_column: str = "run_id") -> LeakageReport:
    """Fails if any run_id's records ended up in more than one split."""
    split_counts = {
        row["split"]: row["len"]
        for row in frame.group_by("split").agg(pl.len()).iter_rows(named=True)
    }
    has_run = frame.filter(pl.col(run_id_column).is_not_null() & (pl.col(run_id_column) != ""))
    per_run_splits = has_run.group_by(run_id_column).agg(
        pl.col("split").n_unique().alias("n_splits")
    )
    leaking = (
        per_run_splits.filter(pl.col("n_splits") > 1).select(run_id_column).to_series().to_list()
    )
    return LeakageReport(split_counts=split_counts, leaking_run_ids=sorted(leaking))
