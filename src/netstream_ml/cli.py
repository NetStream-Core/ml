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
    ClassificationMetrics,
    evaluate_baseline,
    evaluate_baseline_by_scenario,
    fit_threshold_baseline,
)
from netstream_ml.clickhouse import ClickHouseConfig
from netstream_ml.dataset import export_dataset
from netstream_ml.evaluation import mcnemar_test, precision_recall_curve, seed_sensitivity
from netstream_ml.external import (
    RECONSTRUCTABLE_FEATURES,
    UNRECONSTRUCTABLE_FEATURES,
    external_window_features,
    load_cic_csv,
)
from netstream_ml.features import WindowSpec, build_features
from netstream_ml.intensity import attach_rate, evaluate_by_rate
from netstream_ml.split import (
    SplitConfig,
    SplitRatios,
    add_split_column,
    check_run_leakage,
    run_ids_by_scenario,
    run_splits,
)
from netstream_ml.supervised import (
    RandomForestConfig,
    evaluate_random_forest,
    evaluate_random_forest_by_scenario,
    fit_random_forest,
    tune_random_forest,
)
from netstream_ml.tracking import track_run


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

    _add_evaluate_subparsers(subparsers)

    return parser


def _add_evaluate_subparsers(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    evaluate = subparsers.add_parser(
        "evaluate", help="model-agnostic evaluation tools (PR curves, seed spread, significance)"
    )
    evaluate_subparsers = evaluate.add_subparsers(dest="evaluate_command")

    compare = evaluate_subparsers.add_parser(
        "compare",
        help="compare the threshold, Isolation Forest and Random Forest baselines: PR "
        "curves, seed sensitivity, and pairwise significance tests on the same test rows",
    )
    compare.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split column)",
    )
    compare.add_argument("--window", type=int, default=5, help="window size in seconds")
    compare.add_argument("--target-fpr", type=float, default=0.01)
    compare.add_argument("--contamination", type=float, default=0.01)
    compare.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="random seeds to refit the Isolation Forest and Random Forest under",
    )

    intensity = evaluate_subparsers.add_parser(
        "intensity",
        help="check whether recall holds up as an attack's rate drops, per scenario, "
        "for each of the three baselines",
    )
    intensity.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split "
        "column and a run_id column)",
    )
    intensity.add_argument("--window", type=int, default=5, help="window size in seconds")
    intensity.add_argument("--target-fpr", type=float, default=0.01)
    intensity.add_argument("--contamination", type=float, default=0.01)
    intensity.add_argument("--random-state", type=int, default=0)

    transfer = evaluate_subparsers.add_parser(
        "transfer",
        help="retrain each baseline on a reconstructable feature subset and check "
        "generalisation to an external CICFlowMeter dataset (CIC-IDS2017/CIC-DDoS2019)",
    )
    transfer.add_argument(
        "dataset_dir",
        type=Path,
        help="directory produced by `features build` (features_w<N>.parquet, with a split column)",
    )
    transfer.add_argument(
        "external_csv", type=Path, help="a CICFlowMeter CSV (e.g. one CIC-IDS2017 day file)"
    )
    transfer.add_argument("--window", type=int, default=5, help="window size in seconds")
    transfer.add_argument("--target-fpr", type=float, default=0.01)
    transfer.add_argument("--contamination", type=float, default=0.01)
    transfer.add_argument("--random-state", type=int, default=0)


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

    with track_run(
        "baseline-threshold", {"window_seconds": args.window, "target_fpr": args.target_fpr}
    ) as run:
        run.log_metrics(metrics_by_split)
        run.log_artifact(out_path)
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

    with track_run(
        "baseline-isolation-forest",
        {
            "window_seconds": args.window,
            "contamination": args.contamination,
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
        },
    ) as run:
        run.log_metrics(metrics_by_split)
        run.log_artifact(out_path)
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

    with track_run(
        "baseline-random-forest",
        {"window_seconds": args.window, "random_state": args.random_state, **tuning.to_dict()},
    ) as run:
        run.log_metrics(metrics_by_split)
        run.log_artifact(out_path)
    return 0


def _run_evaluate_compare(args: argparse.Namespace) -> int:
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
    test = frame.filter(pl.col("split") == "test")
    if test.height == 0:
        print(f"{path} has no test rows to compare on", file=sys.stderr)
        return 1
    y_test = test["label"] != "benign"

    threshold_model = fit_threshold_baseline(train, target_fpr=args.target_fpr)
    threshold_pred = threshold_model.predict(test)
    threshold_curve = precision_recall_curve(y_test, threshold_model.decision_score(test))

    iso_config = IsolationForestConfig(contamination=args.contamination)
    iso_model = fit_isolation_forest(train, iso_config)
    iso_pred = iso_model.predict(test)
    iso_curve = precision_recall_curve(y_test, iso_model.decision_score(test))
    iso_seeds = seed_sensitivity(
        lambda seed: evaluate_isolation_forest(
            fit_isolation_forest(
                train, IsolationForestConfig(contamination=args.contamination, random_state=seed)
            ),
            test,
        ),
        seeds=args.seeds,
    )

    rf_tuning = tune_random_forest(train, val, random_state=args.seeds[0])
    rf_model = fit_random_forest(train, rf_tuning.config)
    rf_pred = rf_model.predict(test)
    rf_curve = precision_recall_curve(y_test, rf_model.decision_score(test))
    rf_seeds = seed_sensitivity(
        lambda seed: evaluate_random_forest(
            fit_random_forest(
                train,
                RandomForestConfig(
                    n_estimators=rf_tuning.config.n_estimators,
                    max_depth=rf_tuning.config.max_depth,
                    class_weight=rf_tuning.config.class_weight,
                    random_state=seed,
                ),
            ),
            test,
        ),
        seeds=args.seeds,
    )

    threshold_vs_iso = mcnemar_test(y_test, threshold_pred, iso_pred)
    threshold_vs_rf = mcnemar_test(y_test, threshold_pred, rf_pred)
    iso_vs_rf = mcnemar_test(y_test, iso_pred, rf_pred)

    print(
        f"average_precision: threshold={threshold_curve.average_precision:.3f} "
        f"isolation-forest={iso_curve.average_precision:.3f} "
        f"random-forest={rf_curve.average_precision:.3f}"
    )
    print(
        f"F1 across seeds {list(args.seeds)}: "
        f"isolation-forest mean={iso_seeds.mean:.3f} std={iso_seeds.std:.3f}; "
        f"random-forest mean={rf_seeds.mean:.3f} std={rf_seeds.std:.3f}"
    )
    print(
        f"threshold vs isolation-forest: p={threshold_vs_iso.p_value:.4f}; "
        f"threshold vs random-forest: p={threshold_vs_rf.p_value:.4f}; "
        f"isolation-forest vs random-forest: p={iso_vs_rf.p_value:.4f}"
    )

    result = {
        "window_seconds": args.window,
        "threshold": {
            "target_fpr": args.target_fpr,
            "precision_recall_curve": threshold_curve.to_dict(),
        },
        "isolation_forest": {
            "contamination": args.contamination,
            "precision_recall_curve": iso_curve.to_dict(),
            "seed_sensitivity": iso_seeds.to_dict(),
        },
        "random_forest": {
            "tuning": rf_tuning.to_dict(),
            "precision_recall_curve": rf_curve.to_dict(),
            "seed_sensitivity": rf_seeds.to_dict(),
        },
        "mcnemar": {
            "threshold_vs_isolation_forest": threshold_vs_iso.to_dict(),
            "threshold_vs_random_forest": threshold_vs_rf.to_dict(),
            "isolation_forest_vs_random_forest": iso_vs_rf.to_dict(),
        },
    }
    out_path = args.dataset_dir / f"evaluate_compare_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"comparison report written to {out_path}")

    with track_run(
        "evaluate-compare",
        {
            "window_seconds": args.window,
            "target_fpr": args.target_fpr,
            "contamination": args.contamination,
            "seeds": args.seeds,
        },
    ) as run:
        run.log_metrics(result)
        run.log_artifact(out_path)
    return 0


def _run_evaluate_intensity(args: argparse.Namespace) -> int:
    path = args.dataset_dir / f"features_w{args.window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return 1
    if "run_id" not in frame.columns:
        print(
            f"{path} has no 'run_id' column; rebuild features with a version of this "
            "tool that carries run_id through",
            file=sys.stderr,
        )
        return 1

    labels = pl.read_parquet(args.dataset_dir / "labels.parquet")
    frame = attach_rate(frame, labels)

    train = frame.filter(pl.col("split") == "train")
    val = frame.filter(pl.col("split") == "val")
    held_out = frame.filter(pl.col("split") != "train")
    if held_out.height == 0:
        print(f"{path} has no held-out (val/test) rows to evaluate on", file=sys.stderr)
        return 1

    threshold_model = fit_threshold_baseline(train, target_fpr=args.target_fpr)
    iso_model = fit_isolation_forest(train, IsolationForestConfig(contamination=args.contamination))
    rf_tuning = tune_random_forest(train, val, random_state=args.random_state)
    rf_model = fit_random_forest(train, rf_tuning.config)

    result: dict[str, object] = {"window_seconds": args.window}
    for name, model in (
        ("threshold", threshold_model),
        ("isolation_forest", iso_model),
        ("random_forest", rf_model),
    ):
        by_rate = evaluate_by_rate(model.predict(held_out), held_out)
        result[name] = [g.to_dict() for g in by_rate]
        print(f"--- {name} ---")
        for group in by_rate:
            print(
                f"{group.scenario} rate={group.rate}: recall={group.metrics.recall:.3f} "
                f"fpr={group.metrics.fpr:.3f} (n={group.n})"
            )

    out_path = args.dataset_dir / f"evaluate_intensity_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"intensity report written to {out_path}")
    return 0


def _transfer_entry(
    name: str,
    lab_metrics: ClassificationMetrics | None,
    external_metrics: ClassificationMetrics,
    external_by_scenario: dict[str, ClassificationMetrics],
) -> dict[str, object]:
    entry: dict[str, object] = {}
    if lab_metrics is not None:
        entry["lab_test_same_features"] = lab_metrics.to_dict()
        print(
            f"{name} (lab test, reduced features): precision={lab_metrics.precision:.3f} "
            f"recall={lab_metrics.recall:.3f} f1={lab_metrics.f1:.3f}"
        )
    entry["external"] = external_metrics.to_dict()
    entry["external_by_scenario"] = {
        scenario: metrics.to_dict() for scenario, metrics in external_by_scenario.items()
    }
    print(
        f"{name} (external): precision={external_metrics.precision:.3f} "
        f"recall={external_metrics.recall:.3f} f1={external_metrics.f1:.3f} "
        f"fpr={external_metrics.fpr:.3f}"
    )
    return entry


def _run_evaluate_transfer(args: argparse.Namespace) -> int:
    path = args.dataset_dir / f"features_w{args.window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return 1

    external_flows = load_cic_csv(args.external_csv)
    external = external_window_features(external_flows, WindowSpec(args.window))
    if external.height == 0:
        print(f"{args.external_csv} produced no windows", file=sys.stderr)
        return 1

    train = frame.filter(pl.col("split") == "train")
    val = frame.filter(pl.col("split") == "val")
    lab_test = frame.filter(pl.col("split") == "test")

    threshold_model = fit_threshold_baseline(
        train, features_to_consider=RECONSTRUCTABLE_FEATURES, target_fpr=args.target_fpr
    )
    iso_model = fit_isolation_forest(
        train,
        IsolationForestConfig(
            features=RECONSTRUCTABLE_FEATURES, contamination=args.contamination
        ),
    )
    rf_tuning = tune_random_forest(
        train, val, features=RECONSTRUCTABLE_FEATURES, random_state=args.random_state
    )
    rf_model = fit_random_forest(train, rf_tuning.config)

    result: dict[str, object] = {
        "window_seconds": args.window,
        "reconstructable_features": list(RECONSTRUCTABLE_FEATURES),
        "unreconstructable_features": list(UNRECONSTRUCTABLE_FEATURES),
        "external_windows": external.height,
        "external_scenarios": sorted(external["scenario"].unique().to_list()),
        "threshold": _transfer_entry(
            "threshold",
            evaluate_baseline(threshold_model, lab_test) if lab_test.height > 0 else None,
            evaluate_baseline(threshold_model, external),
            evaluate_baseline_by_scenario(threshold_model, external),
        ),
        "isolation_forest": _transfer_entry(
            "isolation_forest",
            evaluate_isolation_forest(iso_model, lab_test) if lab_test.height > 0 else None,
            evaluate_isolation_forest(iso_model, external),
            evaluate_isolation_forest_by_scenario(iso_model, external),
        ),
        "random_forest": _transfer_entry(
            "random_forest",
            evaluate_random_forest(rf_model, lab_test) if lab_test.height > 0 else None,
            evaluate_random_forest(rf_model, external),
            evaluate_random_forest_by_scenario(rf_model, external),
        ),
    }

    out_path = args.dataset_dir / f"evaluate_transfer_w{args.window}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"transfer report written to {out_path}")

    with track_run(
        "evaluate-transfer",
        {
            "window_seconds": args.window,
            "target_fpr": args.target_fpr,
            "contamination": args.contamination,
            "random_state": args.random_state,
            "external_csv": str(args.external_csv),
        },
    ) as run:
        run.log_metrics(result)
        run.log_artifact(out_path)
    return 0


_HANDLERS: dict[tuple[str, str | None], Callable[[argparse.Namespace], int]] = {
    ("data", "export"): _run_export,
    ("data", "split"): _run_split,
    ("features", "build"): _run_features_build,
    ("baseline", "threshold"): _run_baseline_threshold,
    ("baseline", "isolation-forest"): _run_baseline_isolation_forest,
    ("baseline", "random-forest"): _run_baseline_random_forest,
    ("evaluate", "compare"): _run_evaluate_compare,
    ("evaluate", "intensity"): _run_evaluate_intensity,
    ("evaluate", "transfer"): _run_evaluate_transfer,
}

_SUBCOMMAND_ATTR = {
    "data": "data_command",
    "features": "features_command",
    "baseline": "baseline_command",
    "evaluate": "evaluate_command",
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
