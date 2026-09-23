import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from netstream_ml import __version__
from netstream_ml.split import (
    SplitConfig,
    SplitRatios,
    add_split_column,
    check_run_leakage,
    run_ids_by_scenario,
    run_splits,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netstream-ml")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    data = subparsers.add_parser("data", help="build and inspect datasets")
    data_subparsers = data.add_subparsers(dest="data_command")

    split = data_subparsers.add_parser(
        "split",
        help="assign train/val/test splits to a dataset without leaking a run across them",
    )
    split.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `data export` (flows.parquet, dns.parquet, labels.parquet)",
    )
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train", type=float, default=0.6)
    split.add_argument("--val", type=float, default=0.2)
    split.add_argument("--test", type=float, default=0.2)

    return parser


def _run_split(args: argparse.Namespace) -> int:
    ratios = SplitRatios(train=args.train, val=args.val, test=args.test)
    labels = pl.read_parquet(args.dataset_dir / "labels.parquet")
    assignment = run_splits(run_ids_by_scenario(labels), seed=args.seed, ratios=ratios)

    config = SplitConfig(seed=args.seed, ratios=ratios)
    reports: dict[str, object] = {}
    ok = True
    for name in ("flows", "dns"):
        path = args.dataset_dir / f"{name}.parquet"
        frame = add_split_column(pl.read_parquet(path), assignment, config)
        frame.write_parquet(path)
        report = check_run_leakage(frame)
        reports[name] = {
            "split_counts": report.split_counts,
            "leaking_run_ids": report.leaking_run_ids,
        }
        ok = ok and report.ok

    manifest_path = args.dataset_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
                "run_assignment": assignment,
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"split manifest written to {manifest_path}")

    if not ok:
        print("leakage detected: some run_id appears in more than one split", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "data" and args.data_command == "split":
        return _run_split(args)

    parser.print_help()
    return 2
