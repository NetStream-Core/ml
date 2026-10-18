import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
from opentelemetry.proto.resource.v1.resource_pb2 import Resource

from netstream_ml import detector
from netstream_ml.detector import OnlineDetector
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
