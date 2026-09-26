import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

from netstream_ml import __version__
from netstream_ml.anomaly import (
    IsolationForestConfig,
    evaluate_isolation_forest,
    evaluate_isolation_forest_by_scenario,
    fit_isolation_forest,
)
from netstream_ml.baselines import (
    evaluate_baseline,
    evaluate_baseline_by_scenario,
    fit_threshold_baseline,
)
from netstream_ml.clickhouse import ClickHouseConfig
from netstream_ml.dataset import export_dataset
from netstream_ml.features import WindowSpec, build_features
from netstream_ml.split import (
    SplitConfig,
    SplitRatios,
    add_split_column,
    check_run_leakage,
    run_ids_by_scenario,
    run_splits,
)
from netstream_ml.supervised import (
    evaluate_random_forest,
    evaluate_random_forest_by_scenario,
    fit_random_forest,
    tune_random_forest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netstream-ml")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    data = subparsers.add_parser("data", help="build and inspect datasets")
    data_subparsers = data.add_subparsers(dest="data_command")

    export = data_subparsers.add_parser(
        "export", help="export labelled flows, DNS queries and labels to parquet"
    )
    export.add_argument(
        "out_dir", type=Path, help="directory to write the parquet files and manifest into"
    )
    export.add_argument(
        "--url",
        default=os.environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123"),
        help="ClickHouse HTTP interface URL",
    )
    export.add_argument("--user", default=os.environ.get("CLICKHOUSE_USER", "netstream"))
    export.add_argument(
        "--password", default=os.environ.get("CLICKHOUSE_PASSWORD", "netstream-dev")
    )
    export.add_argument("--database", default=os.environ.get("CLICKHOUSE_DATABASE", "netstream"))
    export.add_argument(
        "--campaign-manifest",
        type=Path,
        default=None,
        help="campaign manifest (from deploy's lab/campaign.py) to embed for traceability",
    )
    export.add_argument(
        "--max-memory-usage",
        type=int,
        default=int(os.environ.get("CLICKHOUSE_MAX_MEMORY_USAGE", "2000000000")),
        help="per-query memory limit in bytes; labeled_flows/labeled_dns scan every label "
        "window per row and can need more than the server's default",
    )

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

    features = subparsers.add_parser(
        "features", help="build windowed features for training or evaluation"
    )
    features_subparsers = features.add_subparsers(dest="features_command")

    build = features_subparsers.add_parser(
        "build", help="aggregate flows and DNS queries into per-source, per-window features"
    )
    build.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `data export` (flows.parquet, dns.parquet)",
    )
    build.add_argument(
        "--window",
        dest="windows",
        type=int,
        action="append",
        default=None,
        help="window size in seconds; repeat for more than one (default: 1, 5, 10)",
    )

    baseline = subparsers.add_parser("baseline", help="fit and evaluate baseline detectors")
    baseline_subparsers = baseline.add_subparsers(dest="baseline_command")

    threshold = baseline_subparsers.add_parser(
        "threshold", help="a per-feature threshold rule, fit on the training split"
    )
    threshold.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split column)",
    )
    threshold.add_argument("--window", type=int, default=5, help="window size in seconds")
    threshold.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="each feature's threshold false-alarms on about this share of training benign traffic",
    )

    isolation_forest = baseline_subparsers.add_parser(
        "isolation-forest",
        help="an Isolation Forest fit on benign windows across all features at once",
    )
    isolation_forest.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split column)",
    )
    isolation_forest.add_argument("--window", type=int, default=5, help="window size in seconds")
    isolation_forest.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="share of training benign traffic the model itself flags as an outlier",
    )
    isolation_forest.add_argument("--n-estimators", type=int, default=100)
    isolation_forest.add_argument("--random-state", type=int, default=0)

    random_forest = baseline_subparsers.add_parser(
        "random-forest",
        help="a Random Forest fit on labelled attack and benign windows, tuned on validation",
    )
    random_forest.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split column)",
    )
    random_forest.add_argument("--window", type=int, default=5, help="window size in seconds")
    random_forest.add_argument("--random-state", type=int, default=0)

    return parser


def _run_export(args: argparse.Namespace) -> int:
    config = ClickHouseConfig(
        url=args.url,
        user=args.user,
        password=args.password,
        database=args.database,
        max_memory_usage=args.max_memory_usage,
    )
    manifest_path = export_dataset(config, args.out_dir, args.campaign_manifest)
    print(f"manifest written to {manifest_path}")
    return 0


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


def _run_features_build(args: argparse.Namespace) -> int:
    windows = args.windows or [1, 5, 10]
    flows = pl.read_parquet(args.dataset_dir / "flows.parquet")
    dns = pl.read_parquet(args.dataset_dir / "dns.parquet")

    built: dict[str, object] = {}
    for seconds in windows:
        window = WindowSpec(seconds)
        table = build_features(flows, dns, window)
        out_path = args.dataset_dir / f"features_w{seconds}.parquet"
        table.write_parquet(out_path)
        built[out_path.name] = table.height
        print(f"{out_path}: {table.height} rows")

    manifest_path = args.dataset_dir / "features_manifest.json"
    manifest_path.write_text(
        json.dumps({"windows_seconds": windows, "files": built}, indent=2) + "\n"
    )
    print(f"features manifest written to {manifest_path}")
    return 0


def _run_baseline_threshold(args: argparse.Namespace) -> int:
    path = args.dataset_dir / f"features_w{args.window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return 1

    train = frame.filter(pl.col("split") == "train")
    model = fit_threshold_baseline(train, target_fpr=args.target_fpr)

    metrics_by_split: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        subset = frame.filter(pl.col("split") == split_name)
        if subset.height == 0:
            continue
        metrics = evaluate_baseline(model, subset)
        metrics_by_split[split_name] = metrics.to_dict()
        print(
            f"{split_name}: precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
            f"f1={metrics.f1:.3f} fpr={metrics.fpr:.4f} (n={subset.height})"
        )

    test = frame.filter(pl.col("split") == "test")
    by_scenario = (
        {
            scenario: metrics.to_dict()
            for scenario, metrics in evaluate_baseline_by_scenario(model, test).items()
        }
        if test.height > 0
        else {}
    )

    result = {
        "window_seconds": args.window,
        "target_fpr": args.target_fpr,
        "rules": model.to_dict(),
        "metrics_by_split": metrics_by_split,
        "test_metrics_by_scenario": by_scenario,
    }
    out_path = args.dataset_dir / f"baseline_threshold_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"baseline report written to {out_path}")
    return 0


def _run_baseline_isolation_forest(args: argparse.Namespace) -> int:
    path = args.dataset_dir / f"features_w{args.window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return 1

    train = frame.filter(pl.col("split") == "train")
    config = IsolationForestConfig(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    model = fit_isolation_forest(train, config)

    metrics_by_split: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        subset = frame.filter(pl.col("split") == split_name)
        if subset.height == 0:
            continue
        metrics = evaluate_isolation_forest(model, subset)
        metrics_by_split[split_name] = metrics.to_dict()
        print(
            f"{split_name}: precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
            f"f1={metrics.f1:.3f} fpr={metrics.fpr:.4f} (n={subset.height})"
        )

    test = frame.filter(pl.col("split") == "test")
    by_scenario = (
        {
            scenario: metrics.to_dict()
            for scenario, metrics in evaluate_isolation_forest_by_scenario(model, test).items()
        }
        if test.height > 0
        else {}
    )

    result = {
        "window_seconds": args.window,
        "contamination": args.contamination,
        "model": model.to_dict(),
        "metrics_by_split": metrics_by_split,
        "test_metrics_by_scenario": by_scenario,
    }
    out_path = args.dataset_dir / f"isolation_forest_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"isolation forest report written to {out_path}")
    return 0


def _run_baseline_random_forest(args: argparse.Namespace) -> int:
    path = args.dataset_dir / f"features_w{args.window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return 1

    train = frame.filter(pl.col("split") == "train")
    val = frame.filter(pl.col("split") == "val")
    tuning = tune_random_forest(train, val, random_state=args.random_state)
    model = fit_random_forest(train, tuning.config)
    print(
        f"tuned: n_estimators={tuning.config.n_estimators} max_depth={tuning.config.max_depth} "
        f"class_weight={tuning.config.class_weight} (val f1={tuning.val_metrics.f1:.3f}, "
        f"{tuning.candidates_tried} candidates)"
    )

    metrics_by_split: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        subset = frame.filter(pl.col("split") == split_name)
        if subset.height == 0:
            continue
        metrics = evaluate_random_forest(model, subset)
        metrics_by_split[split_name] = metrics.to_dict()
        print(
            f"{split_name}: precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
            f"f1={metrics.f1:.3f} fpr={metrics.fpr:.4f} (n={subset.height})"
        )

    test = frame.filter(pl.col("split") == "test")
    by_scenario = (
        {
            scenario: metrics.to_dict()
            for scenario, metrics in evaluate_random_forest_by_scenario(model, test).items()
        }
        if test.height > 0
        else {}
    )

    result = {
        "window_seconds": args.window,
        "tuning": tuning.to_dict(),
        "model": model.to_dict(),
        "metrics_by_split": metrics_by_split,
        "test_metrics_by_scenario": by_scenario,
    }
    out_path = args.dataset_dir / f"random_forest_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"random forest report written to {out_path}")
    return 0


_HANDLERS: dict[tuple[str, str | None], Callable[[argparse.Namespace], int]] = {
    ("data", "export"): _run_export,
    ("data", "split"): _run_split,
    ("features", "build"): _run_features_build,
    ("baseline", "threshold"): _run_baseline_threshold,
    ("baseline", "isolation-forest"): _run_baseline_isolation_forest,
    ("baseline", "random-forest"): _run_baseline_random_forest,
}

_SUBCOMMAND_ATTR = {
    "data": "data_command",
    "features": "features_command",
    "baseline": "baseline_command",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    subcommand = getattr(args, _SUBCOMMAND_ATTR.get(args.command, ""), None)
    handler = _HANDLERS.get((args.command, subcommand))
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)
