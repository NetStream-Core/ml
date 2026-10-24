"""The lifecycle shared by all four `baseline` subcommands (threshold,
isolation-forest, random-forest, gradient-boosting) in `cli.py`: load the
split dataset, score it across train/val/test and per scenario on test,
write the JSON report, track the run, then save/register/export the
model. Fitting the model itself — and anything specific to its
hyperparameters or tuning — stays in `cli.py`, one function per baseline,
since that is the one part that actually differs between them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from netstream_ml.baselines import ClassificationMetrics
from netstream_ml.detector import Baseline, save_model
from netstream_ml.model_registry import ModelMetadata, RegistryConfig, RegistryError, publish_model
from netstream_ml.onnx_export import run_onnx, save_onnx_model, to_onnx
from netstream_ml.tracking import track_run

# All four baselines' evaluate_*/evaluate_*_by_scenario functions share this
# one return type (`ClassificationMetrics`, defined once in `baselines.py`)
# even though each takes its own concrete `Baseline` subtype as its first
# argument — `...` sidesteps the resulting parameter-type mismatch, since
# this is CLI glue calling whichever one the caller already fit, not a
# place that needs to call it generically with unknown arguments.
EvaluateFn = Callable[..., ClassificationMetrics]
EvaluateByScenarioFn = Callable[..., dict[str, ClassificationMetrics]]

_SPLIT_NAMES = ("train", "val", "test")


def load_split_dataset(dataset_dir: Path, window: int) -> pl.DataFrame | None:
    """Reads `features_w<window>.parquet` and checks it already has a
    `split` column (`data split` must run before `features build`) — the
    one check every baseline needs before it can even start fitting.
    Prints the same error and returns `None` if it's missing, instead of
    raising, so the caller can turn that into a `main()` exit code."""
    path = dataset_dir / f"features_w{window}.parquet"
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        print(
            f"{path} has no 'split' column; run `data split` before `features build`",
            file=sys.stderr,
        )
        return None
    return frame


def metrics_by_split(
    model: Baseline, frame: pl.DataFrame, evaluate: EvaluateFn
) -> dict[str, object]:
    """Scores `model` on whichever of train/val/test are non-empty in
    `frame`, printing a one-line summary for each as it goes — the exact
    loop every baseline command ran before this was extracted."""
    result: dict[str, object] = {}
    for split_name in _SPLIT_NAMES:
        subset = frame.filter(pl.col("split") == split_name)
        if subset.height == 0:
            continue
        metrics = evaluate(model, subset)
        result[split_name] = metrics.to_dict()
        print(
            f"{split_name}: precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
            f"f1={metrics.f1:.3f} fpr={metrics.fpr:.4f} (n={subset.height})"
        )
    return result


def metrics_by_scenario(
    model: Baseline, test: pl.DataFrame, evaluate_by_scenario: EvaluateByScenarioFn
) -> dict[str, object]:
    """Per-scenario test metrics, or `{}` if there is no test split at all
    (a dataset too small to have one, or one with a skewed split)."""
    if test.height == 0:
        return {}
    return {
        scenario: metrics.to_dict()
        for scenario, metrics in evaluate_by_scenario(model, test).items()
    }


def _maybe_register_model(
    args: argparse.Namespace, algorithm: str, by_split: dict[str, object]
) -> int:
    if args.register_model is None:
        return 0
    if args.save_model is None:
        print("--register-model requires --save-model", file=sys.stderr)
        return 1
    if args.model_version is None:
        print("--register-model requires --model-version", file=sys.stderr)
        return 1
    if args.signing_key is None:
        print("--register-model requires --signing-key", file=sys.stderr)
        return 1

    registry = RegistryConfig(backend_url=args.backend_url)
    metadata = ModelMetadata(
        name=args.register_model,
        version=args.model_version,
        algorithm=algorithm,
        features_version=f"w{args.window}",
        metrics=by_split,
    )
    try:
        publish_model(registry, args.save_model, metadata, args.signing_key)
    except RegistryError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"model registered: {args.register_model}@{args.model_version} ({args.backend_url})")
    return 0


def _maybe_export_onnx(args: argparse.Namespace, model: Baseline, test: pl.DataFrame) -> int:
    if args.export_onnx is None:
        return 0

    try:
        onnx_model, feature_order, kind = to_onnx(model)
    except TypeError as error:
        print(str(error), file=sys.stderr)
        return 1

    if test.height > 0:
        onnx_predict, onnx_score = run_onnx(onnx_model, test, feature_order, kind)
        original_predict = model.predict(test).to_numpy()
        original_score = model.decision_score(test).to_numpy()
        if not np.array_equal(onnx_predict, original_predict) or not np.allclose(
            onnx_score, original_score, atol=1e-4
        ):
            print(
                "ONNX export does not match the original model's predict/decision_score "
                "on the test split; refusing to write a silently wrong file",
                file=sys.stderr,
            )
            return 1

    save_onnx_model(onnx_model, args.export_onnx)
    print(f"model exported to ONNX at {args.export_onnx}")
    return 0


def finish_baseline_run(  # noqa: PLR0913 - ten independent, non-reusable-as-a-group pieces
    args: argparse.Namespace,
    *,
    model: Baseline,
    test: pl.DataFrame,
    model_kind: str,
    report_label: str,
    out_path: Path,
    result: dict[str, object],
    by_split: dict[str, object],
    track_name: str,
    track_params: dict[str, object],
) -> int:
    """Everything after a baseline has been fit and scored: write the JSON
    report, track the run (metrics + the report itself as an artifact),
    then — in this order, each gated on the last succeeding — save the
    model, register it, and export it to ONNX. Returns the process exit
    code the caller's `_run_baseline_*` should return directly."""
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{report_label} written to {out_path}")

    with track_run(track_name, track_params) as run:
        run.log_metrics(by_split)
        run.log_artifact(out_path)

    if args.save_model is not None:
        save_model(model, args.save_model)
        print(f"model saved to {args.save_model}")
    register_exit_code = _maybe_register_model(args, model_kind, by_split)
    if register_exit_code != 0:
        return register_exit_code
    return _maybe_export_onnx(args, model, test)
