"""Logs each `baseline`/`evaluate` CLI run to MLflow, so releases can be
compared by browsing runs (`mlflow ui`) instead of diffing JSON reports by
hand. Tracking is local-file-based by default (`./mlruns`, nothing to stand
up) and points at a real server via `MLFLOW_TRACKING_URI` when one exists —
the same override pattern the CLI already uses for ClickHouse.

Failing to log (MLflow unreachable, disk full, whatever) must never fail the
command that produced the actual result: every entry point here catches and
warns rather than raises.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # must import after MLFLOW_DISABLE_AGENT_HINT is set above

DEFAULT_EXPERIMENT = "netstream-baselines"


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub_value, out)
    elif isinstance(value, list):
        out[prefix] = str(value)
    elif value is not None:
        out[prefix] = value


@contextmanager
def track_run(
    run_name: str,
    params: dict[str, Any],
    experiment: str = DEFAULT_EXPERIMENT,
) -> Iterator[_Run]:
    """Starts an MLflow run for the duration of the `with` block. Params are
    logged immediately; call `.log_metrics(...)` and `.log_artifact(...)` on
    the yielded handle once results are available. Any MLflow failure is
    caught and printed as a warning — tracking is a convenience, not a
    dependency the CLI command should fail on."""
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            flat_params: dict[str, Any] = {}
            _flatten("", params, flat_params)
            mlflow.log_params(flat_params)
            yield _Run(run.info.run_id)
    except Exception as exc:
        print(f"warning: MLflow tracking failed, continuing without it: {exc}", file=sys.stderr)
        yield _Run(None)


class _Run:
    def __init__(self, run_id: str | None) -> None:
        self._run_id = run_id

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self._run_id is None:
            return
        try:
            flat: dict[str, Any] = {}
            _flatten("", metrics, flat)
            numeric = {k: v for k, v in flat.items() if isinstance(v, int | float)}
            mlflow.log_metrics(numeric)
        except Exception as exc:
            print(f"warning: MLflow metric logging failed: {exc}", file=sys.stderr)

    def log_artifact(self, path: Path) -> None:
        if self._run_id is None:
            return
        try:
            mlflow.log_artifact(str(path))
        except Exception as exc:
            print(f"warning: MLflow artifact logging failed: {exc}", file=sys.stderr)
