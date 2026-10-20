import datetime as dt
from pathlib import Path

import confluent_kafka
import polars as pl
import pytest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
from opentelemetry.proto.resource.v1.resource_pb2 import Resource

from netstream_ml import detector
from netstream_ml.detector import (
    OnlineDetector,
    _commit_if_anything_to_commit,
    _detection_key,
    _produce_detections,
)
from netstream_ml.features import WindowSpec


def _any_value(value: object) -> AnyValue:
    av = AnyValue()
    if isinstance(value, bool):
        av.bool_value = value
    elif isinstance(value, int):
        av.int_value = value
    elif isinstance(value, float):
        av.double_value = value
    elif isinstance(value, str):
        av.string_value = value
    return av


def _kv(key: str, value: object) -> KeyValue:
    return KeyValue(key=key, value=_any_value(value))


_FLOW_ATTRS: dict[str, object] = {
    "destination.address": "10.0.0.6",
    "destination.port": 443,
    "network.io.direction": "transmit",
    "netstream.flow.packets": 12,
    "netstream.flow.bytes.ip": 1200,
    "netstream.flow.bytes.payload": 1000,
    "netstream.flow.tcp.syn": 1,
    "netstream.flow.tcp.synack": 0,
    "netstream.flow.tcp.rst": 0,
    "netstream.flow.aggregated": 0,
    "netstream.flow.size.le64": 0,
    "netstream.flow.size.le128": 0,
    "netstream.flow.size.le256": 0,
    "netstream.flow.size.le512": 0,
    "netstream.flow.size.le1024": 12,
    "netstream.flow.size.gt1024": 0,
    "netstream.flow.iat.count": 11,
    "netstream.flow.iat.sum_us": 1100,
    "netstream.flow.iat.sumsq_us": 110000,
}


def _flow_payload(src_ip: str, time_unix_nano: int, host_id: str = "sensor-1") -> bytes:
    attrs = {**_FLOW_ATTRS, "source.address": src_ip}
    resource = Resource(attributes=[_kv("host.id", host_id)])
    log_record = LogRecord(
        time_unix_nano=time_unix_nano,
        event_name="netstream.flow",
        attributes=[_kv(k, v) for k, v in attrs.items()],
    )
    resource_logs = ResourceLogs(
        resource=resource, scope_logs=[ScopeLogs(log_records=[log_record])]
    )
    return bytes(ExportLogsServiceRequest(resource_logs=[resource_logs]).SerializeToString())


class _FakeModel:
    """Flags a window as an attack whenever it has more than one flow."""

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        return frame["flows_count"] > 1

    def decision_score(self, frame: pl.DataFrame) -> pl.Series:
        return frame["flows_count"].cast(pl.Float64)


def _ns(seconds_from_epoch: int) -> int:
    return seconds_from_epoch * 1_000_000_000


def test_process_message_produces_no_detection_while_the_window_is_still_open() -> None:
    online = OnlineDetector(_FakeModel(), WindowSpec(5))

    results = online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000)))

    assert results == []


def test_process_message_scores_a_window_once_a_later_record_closes_it() -> None:
    online = OnlineDetector(_FakeModel(), WindowSpec(5))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000)))

    results = online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_010)))

    assert len(results) == 1
    result = results[0]
    assert result.host_id == "sensor-1"
    assert result.src_ip == "10.0.0.5"
    assert result.is_attack is False  # only one flow landed in the closed window
    assert result.score == 1.0


def test_process_message_keeps_sensors_with_the_same_src_ip_separate() -> None:
    online = OnlineDetector(_FakeModel(), WindowSpec(5))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000), host_id="sensor-a"))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_001), host_id="sensor-a"))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000), host_id="sensor-b"))

    results = online.process_message(
        _flow_payload("10.0.0.5", _ns(1_700_000_010), host_id="sensor-b")
    )

    # sensor-b's window has 1 flow (not an attack by the fake model); it
    # must not have been contaminated by sensor-a's two flows sharing the
    # same src_ip.
    assert len(results) == 1
    assert results[0].host_id == "sensor-b"
    assert results[0].is_attack is False


def test_process_message_reports_increasing_latency_for_an_older_window() -> None:
    online = OnlineDetector(_FakeModel(), WindowSpec(5))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000)))

    now = dt.datetime.fromtimestamp(1_700_000_020, tz=dt.UTC)
    results = online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_010)), now=now)

    assert len(results) == 1
    assert results[0].latency_ms > 0


def test_flush_closes_a_window_that_never_received_a_later_record() -> None:
    online = OnlineDetector(_FakeModel(), WindowSpec(5))
    online.process_message(_flow_payload("10.0.0.5", _ns(1_700_000_000)))

    results = online.flush()

    assert len(results) == 1
    assert results[0].src_ip == "10.0.0.5"
    assert results[0].is_attack is False


def test_detection_result_to_dict_round_trips_fields() -> None:
    result = detector.DetectionResult(
        host_id="sensor-1",
        src_ip="10.0.0.5",
        window_start=1_700_000_000,
        window_seconds=5,
        score=0.9,
        is_attack=True,
        latency_ms=12.3,
    )
    as_dict = result.to_dict()
    assert as_dict["host_id"] == "sensor-1"
    assert as_dict["is_attack"] is True
    assert as_dict["score"] == pytest.approx(0.9)


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    model = _FakeModel()
    path = tmp_path / "model.joblib"

    detector.save_model(model, path)
    loaded = detector.load_model(path)

    frame = pl.DataFrame({"flows_count": [1, 5]})
    assert loaded.predict(frame).to_list() == [False, True]


def _detection(**overrides: object) -> detector.DetectionResult:
    defaults: dict[str, object] = {
        "host_id": "sensor-1",
        "src_ip": "10.0.0.5",
        "window_start": 1_700_000_000,
        "window_seconds": 5,
        "score": 0.9,
        "is_attack": True,
        "latency_ms": 12.3,
    }
    defaults.update(overrides)
    return detector.DetectionResult(**defaults)  # type: ignore[arg-type]


def test_detection_key_is_the_same_for_the_same_window_reprocessed() -> None:
    # The whole point: a window reprocessed after a crash before its input
    # offset committed must produce the same key, so a downstream reader
    # can dedupe on it rather than seeing an unexplained duplicate.
    first = _detection_key(_detection())
    second = _detection_key(_detection())
    assert first == second


def test_detection_key_differs_for_a_different_window() -> None:
    assert _detection_key(_detection(window_start=1)) != _detection_key(_detection(window_start=2))
    assert _detection_key(_detection(src_ip="10.0.0.6")) != _detection_key(_detection())
    assert _detection_key(_detection(host_id="sensor-2")) != _detection_key(_detection())


class _FakeProducer:
    """Stands in for `confluent_kafka.Producer`: records every `produce()`
    call and, on `flush()`, immediately invokes each callback as either a
    success or (for keys in `failing_keys`) a delivery error — exercising
    `_produce_detections`'s commit-gating logic without a real broker."""

    def __init__(
        self, failing_keys: set[bytes] | None = None, remaining_after_flush: int = 0
    ) -> None:
        self.failing_keys = failing_keys or set()
        self.remaining_after_flush = remaining_after_flush
        self.produced: list[tuple[str, bytes, bytes]] = []
        self._pending: list[tuple[bytes, object]] = []

    def produce(self, topic: str, value: bytes, key: bytes, callback: object) -> None:
        self.produced.append((topic, value, key))
        self._pending.append((key, callback))

    def flush(self, timeout: float) -> int:
        for key, callback in self._pending:
            if key in self.failing_keys:
                callback(RuntimeError("delivery failed"), None)  # type: ignore[operator]
            else:
                callback(None, None)  # type: ignore[operator]
        self._pending.clear()
        return self.remaining_after_flush


def test_produce_detections_returns_true_when_everything_is_acked() -> None:
    producer = _FakeProducer()
    latencies: list[float] = []

    delivered = _produce_detections(
        producer,  # type: ignore[arg-type]
        "netstream.detections.v1",
        [_detection()],
        5.0,
        latencies,
    )

    assert delivered is True
    assert len(producer.produced) == 1
    assert latencies == [12.3]


def test_produce_detections_returns_false_when_a_delivery_fails() -> None:
    result = _detection()
    producer = _FakeProducer(failing_keys={_detection_key(result)})

    delivered = _produce_detections(producer, "netstream.detections.v1", [result], 5.0, [])  # type: ignore[arg-type]

    assert delivered is False


def test_produce_detections_returns_false_when_the_broker_times_out() -> None:
    producer = _FakeProducer(remaining_after_flush=1)

    delivered = _produce_detections(
        producer,  # type: ignore[arg-type]
        "netstream.detections.v1",
        [_detection()],
        5.0,
        [],
    )

    assert delivered is False


def test_produce_detections_returns_true_for_an_empty_batch_without_touching_the_producer() -> None:
    producer = _FakeProducer()

    delivered = _produce_detections(producer, "netstream.detections.v1", [], 5.0, [])  # type: ignore[arg-type]

    assert delivered is True
    assert producer.produced == []


class _FakeConsumer:
    """Stands in for `confluent_kafka.Consumer` for `_commit_if_anything_to_commit`:
    raises whatever `KafkaException` it is constructed with, or returns
    normally — exercising the real failure mode found against a live
    broker (a bare `commit()` raises `_NO_OFFSET` when nothing was stored
    since the last commit, which must be swallowed, not propagated)."""

    def __init__(self, exception: confluent_kafka.KafkaException | None = None) -> None:
        self.exception = exception
        self.commit_calls = 0

    def commit(self, asynchronous: bool) -> None:
        self.commit_calls += 1
        if self.exception is not None:
            raise self.exception


def _kafka_exception(code: int) -> confluent_kafka.KafkaException:
    return confluent_kafka.KafkaException(confluent_kafka.KafkaError(code))


def test_commit_if_anything_to_commit_swallows_the_nothing_stored_error() -> None:
    # This is the exact failure observed against a real broker: an
    # idle-flush or shutdown commit with no new offsets stored since the
    # last per-message commit must not crash the detector loop.
    consumer = _FakeConsumer(exception=_kafka_exception(confluent_kafka.KafkaError._NO_OFFSET))

    _commit_if_anything_to_commit(consumer)  # type: ignore[arg-type]

    assert consumer.commit_calls == 1


def test_commit_if_anything_to_commit_reraises_any_other_kafka_error() -> None:
    consumer = _FakeConsumer(exception=_kafka_exception(confluent_kafka.KafkaError._TRANSPORT))

    with pytest.raises(confluent_kafka.KafkaException):
        _commit_if_anything_to_commit(consumer)  # type: ignore[arg-type]


def test_commit_if_anything_to_commit_commits_normally_when_there_is_something_to_commit() -> None:
    consumer = _FakeConsumer()

    _commit_if_anything_to_commit(consumer)  # type: ignore[arg-type]

    assert consumer.commit_calls == 1
