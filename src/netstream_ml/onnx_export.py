"""ONNX export for the four baseline types `ml` trains — a real follow-up to
M14's model registry, which until now only ever moved the `joblib` artifact
around as an opaque blob. Three of the four baselines wrap a scikit-learn/
XGBoost estimator and convert through existing libraries (`skl2onnx`,
`onnxmltools`); `ThresholdBaseline` has no underlying estimator at all — it
is a handful of per-feature comparisons — so its graph is built by hand with
`onnx.helper`, one `Greater`/`Less` plus one normalized-excess computation
per rule, `Or`-reduced for `predict` and max-reduced for `decision_score`.

Every exported model takes a single `float32` input tensor named `input` of
shape `(batch, len(feature_order))` — `feature_order` is returned alongside
the model, and is the exact column order the graph expects; getting that
order wrong silently produces wrong predictions, since ONNX graphs carry no
column names. The four converters disagree on output names and conventions
(IsolationForest's `scores` is the opposite sign of `decision_score`, for
instance) — `run_onnx` below knows each one's quirks by the `kind` string
`to_onnx` tags its result with, rather than guessing from output names at
run time.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import onnx
import onnxruntime as rt
import polars as pl
from onnx import TensorProto, helper
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import to_onnx as _skl2onnx_to_onnx
from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType

from netstream_ml.anomaly import IsolationForestBaseline
from netstream_ml.baselines import ThresholdBaseline
from netstream_ml.detector import Baseline
from netstream_ml.gradient_boosting import GradientBoostingBaseline
from netstream_ml.supervised import RandomForestBaseline

_OPSET = 18
# onnxruntime pinned here tops out at IR version 13; the onnx package
# pinned here defaults new models to 14. Without this, `InferenceSession`
# refuses to load anything `to_onnx`/`convert_xgboost` just produced.
_IR_VERSION = 9


def _finalize(model: onnx.ModelProto) -> onnx.ModelProto:
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model)
    return model


def _threshold_feature_order(baseline: ThresholdBaseline) -> tuple[str, ...]:
    seen: list[str] = []
    for rule in baseline.rules:
        if rule.feature not in seen:
            seen.append(rule.feature)
    return tuple(seen)


def _threshold_to_onnx(baseline: ThresholdBaseline) -> tuple[onnx.ModelProto, tuple[str, ...]]:
    feature_order = _threshold_feature_order(baseline)
    index_of = {name: i for i, name in enumerate(feature_order)}
    n_features = max(len(feature_order), 1)

    nodes: list[onnx.NodeProto] = []
    fires_names: list[str] = []
    excess_names: list[str] = []

    for i, rule in enumerate(baseline.rules):
        idx_name, thr_name, denom_name, col_name = f"idx_{i}", f"thr_{i}", f"denom_{i}", f"col_{i}"
        denom = abs(rule.threshold) if rule.threshold != 0 else 1.0
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [idx_name],
                value=helper.make_tensor(
                    f"{idx_name}_t", TensorProto.INT64, [1], [index_of[rule.feature]]
                ),
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [thr_name],
                value=helper.make_tensor(f"{thr_name}_t", TensorProto.FLOAT, [1], [rule.threshold]),
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [denom_name],
                value=helper.make_tensor(f"{denom_name}_t", TensorProto.FLOAT, [1], [denom]),
            )
        )
        nodes.append(helper.make_node("Gather", ["input", idx_name], [col_name], axis=1))

        fires_name = f"fires_{i}"
        comparator = "Greater" if rule.direction == "above" else "Less"
        nodes.append(helper.make_node(comparator, [col_name, thr_name], [fires_name]))
        fires_names.append(fires_name)

        diff_name = f"diff_{i}"
        if rule.direction == "above":
            nodes.append(helper.make_node("Sub", [col_name, thr_name], [diff_name]))
        else:
            nodes.append(helper.make_node("Sub", [thr_name, col_name], [diff_name]))
        excess_name = f"excess_{i}"
        nodes.append(helper.make_node("Div", [diff_name, denom_name], [excess_name]))
        excess_names.append(excess_name)

    if not baseline.rules:
        # No rules fitted: predict is always False, score is always 0 for
        # every row — built from the input's own row count so it still
        # works for a dynamic batch size.
        nodes.append(helper.make_node("Shape", ["input"], ["in_shape"]))
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["row_axis"],
                value=helper.make_tensor("row_axis_t", TensorProto.INT64, [1], [0]),
            )
        )
        nodes.append(helper.make_node("Gather", ["in_shape", "row_axis"], ["n_rows"], axis=0))
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["one"],
                value=helper.make_tensor("one_t", TensorProto.INT64, [1], [1]),
            )
        )
        nodes.append(helper.make_node("Concat", ["n_rows", "one"], ["n_rows_by_1"], axis=0))
        nodes.append(
            helper.make_node(
                "ConstantOfShape",
                ["n_rows_by_1"],
                ["predict"],
                value=helper.make_tensor("false_t", TensorProto.BOOL, [1], [0]),
            )
        )
        nodes.append(
            helper.make_node(
                "ConstantOfShape",
                ["n_rows_by_1"],
                ["score"],
                value=helper.make_tensor("zero_t", TensorProto.FLOAT, [1], [0.0]),
            )
        )
    else:
        predict_name = fires_names[0]
        for name in fires_names[1:]:
            or_name = f"or_{name}"
            nodes.append(helper.make_node("Or", [predict_name, name], [or_name]))
            predict_name = or_name
        nodes.append(helper.make_node("Identity", [predict_name], ["predict"]))

        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["unsqueeze_axes"],
                value=helper.make_tensor("unsqueeze_axes_t", TensorProto.INT64, [1], [1]),
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["reduce_axes"],
                value=helper.make_tensor("reduce_axes_t", TensorProto.INT64, [1], [1]),
            )
        )
        reshaped = []
        for name in excess_names:
            r = f"{name}_r"
            nodes.append(helper.make_node("Unsqueeze", [name, "unsqueeze_axes"], [r]))
            reshaped.append(r)
        nodes.append(helper.make_node("Concat", reshaped, ["excess_concat"], axis=1))
        nodes.append(
            helper.make_node("ReduceMax", ["excess_concat", "reduce_axes"], ["score"], keepdims=0)
        )

    graph = helper.make_graph(
        nodes,
        "threshold_baseline",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
        [
            helper.make_tensor_value_info("predict", TensorProto.BOOL, [None, 1]),
            helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    return _finalize(model), feature_order


def _skl_to_onnx(estimator: object, n_features: int) -> onnx.ModelProto:
    return _finalize(
        _skl2onnx_to_onnx(
            estimator,
            initial_types=[("input", SklFloatTensorType([None, n_features]))],
            target_opset={"": _OPSET, "ai.onnx.ml": 3},
        )
    )


def to_onnx(baseline: Baseline) -> tuple[onnx.ModelProto, tuple[str, ...], str]:
    """Converts any of the four baselines to ONNX. Returns the model, the
    column order it expects its input tensor in, and a `kind` tag that
    `run_onnx` needs to know how to read that specific converter's
    output convention back into `predict`/`decision_score`."""
    if isinstance(baseline, ThresholdBaseline):
        model, feature_order = _threshold_to_onnx(baseline)
        return model, feature_order, "threshold"
    if isinstance(baseline, IsolationForestBaseline):
        return (
            _skl_to_onnx(baseline.model, len(baseline.features)),
            baseline.features,
            "isolation_forest",
        )
    if isinstance(baseline, RandomForestBaseline):
        return (
            _skl_to_onnx(baseline.model, len(baseline.features)),
            baseline.features,
            "random_forest",
        )
    if isinstance(baseline, GradientBoostingBaseline):
        model = convert_xgboost(
            baseline.model,
            initial_types=[("input", FloatTensorType([None, len(baseline.features)]))],
        )
        return _finalize(model), baseline.features, "gradient_boosting"
    raise TypeError(f"no ONNX conversion for {type(baseline).__name__}")


def run_onnx(
    model: onnx.ModelProto, frame: pl.DataFrame, feature_order: tuple[str, ...], kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the exported graph the same way `OnlineDetector` would: select
    `feature_order`'s columns in that exact order and fill nulls with 0
    (same imputation every baseline's own `.predict`/`.decision_score`
    already does). Returns `(predict, score)` as flat numpy arrays, in the
    same convention as the original baseline — not whatever convention the
    particular ONNX converter for `kind` happened to pick."""
    if feature_order:
        matrix = frame.select(list(feature_order)).fill_null(0).to_numpy().astype(np.float32)
    else:
        # A ThresholdBaseline with no rules still declares a one-column
        # dummy input (ONNX graphs can't take a zero-width tensor) — its
        # value is never read, only the row count matters.
        matrix = np.zeros((frame.height, 1), dtype=np.float32)
    session = rt.InferenceSession(model.SerializeToString())
    names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(names, session.run(None, {"input": matrix}), strict=True))

    if kind == "threshold":
        predict = np.asarray(outputs["predict"]).reshape(-1).astype(bool)
        score = np.asarray(outputs["score"]).reshape(-1).astype(np.float64)
        return predict, score

    if kind == "isolation_forest":
        # label is 1 (normal) / -1 (outlier); `scores` is `decision_function`
        # (higher = more normal). IsolationForestBaseline uses the opposite
        # convention for both.
        label = np.asarray(outputs["label"]).reshape(-1)
        predict = label == -1
        score = -np.asarray(outputs["scores"]).reshape(-1).astype(np.float64)
        return predict, score

    if kind == "random_forest":
        # skl2onnx's classifier output defaults to a list of {class: proba}
        # dicts (a ZipMap node) rather than a plain array; the estimator's
        # `classes_` here are bool (`True`/`False`), but a ZipMap over a
        # bool classifier serializes its keys as int64, so the dict is
        # keyed 0/1, not False/True.
        predict = np.asarray(outputs["output_label"]).reshape(-1).astype(bool)
        proba = cast(list[dict[int, float]], outputs["output_probability"])
        score = np.array([row[1] for row in proba], dtype=np.float64)
        return predict, score

    if kind == "gradient_boosting":
        predict = np.asarray(outputs["label"]).reshape(-1).astype(bool)
        score = np.asarray(outputs["probabilities"])[:, 1].astype(np.float64)
        return predict, score

    raise ValueError(f"unknown ONNX export kind: {kind!r}")
