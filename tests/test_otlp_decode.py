from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
from opentelemetry.proto.resource.v1.resource_pb2 import Resource

from netstream_ml import otlp_decode


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
    else:
        raise TypeError(f"unsupported attribute value type: {type(value)}")
    return av


def _kv(key: str, value: object) -> KeyValue:
    return KeyValue(key=key, value=_any_value(value))


def _build_payload(
    event_name: str,
    attrs: dict[str, object],
    host_id: str = "sensor-1",
    time_unix_nano: int = 1_700_000_000_000_000_000,
) -> bytes:
    resource = Resource(attributes=[_kv("host.id", host_id)])
    log_record = LogRecord(
        time_unix_nano=time_unix_nano,
        event_name=event_name,
        attributes=[_kv(k, v) for k, v in attrs.items()],
    )
    resource_logs = ResourceLogs(
        resource=resource, scope_logs=[ScopeLogs(log_records=[log_record])]
    )
    request = ExportLogsServiceRequest(resource_logs=[resource_logs])
    return bytes(request.SerializeToString())


_FLOW_ATTRS = {
    "source.address": "10.0.0.5",
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

_DNS_ATTRS = {
    "source.address": "10.0.0.5",
    "dns.question.name": "www.example.com",
    "netstream.dns.qname.length": 15,
    "netstream.dns.qname.entropy": 3.2,
    "netstream.dns.qname.digit_ratio": 0.0,
    "netstream.dns.question.type": "A",
    "network.transport": "udp",
    "netstream.dns.unique_subdomains": 1,
}


def test_iter_records_decodes_a_flow_event() -> None:
    payload = _build_payload(otlp_decode.EVENT_FLOW, _FLOW_ATTRS)

    records = otlp_decode.iter_records(payload)

    assert len(records) == 1
    kind, host_id, record = records[0]
    assert kind == "flow"
    assert host_id == "sensor-1"
    assert record["src_ip"] == "10.0.0.5"
    assert record["dst_ip"] == "10.0.0.6"
    assert record["packets"] == 12
    assert record["direction"] == "transmit"
    assert record["label"] == otlp_decode.NO_LABEL
    assert record["scenario"] == otlp_decode.NO_LABEL
    assert record["ts"].year == 2023


def test_iter_records_decodes_a_dns_event() -> None:
    payload = _build_payload(otlp_decode.EVENT_DNS_QUERY, _DNS_ATTRS)

    records = otlp_decode.iter_records(payload)

    assert len(records) == 1
    kind, _host_id, record = records[0]
    assert kind == "dns"
    assert record["qname"] == "www.example.com"
    assert record["qtype"] == "A"
    assert record["unique_subdomains"] == 1


def test_iter_records_skips_unknown_event_names() -> None:
    payload = _build_payload("netstream.blocklist.hit", {"source.address": "10.0.0.5"})

    assert otlp_decode.iter_records(payload) == []


def test_iter_records_skips_a_flow_event_missing_a_required_attribute() -> None:
    incomplete = dict(_FLOW_ATTRS)
    del incomplete["netstream.flow.packets"]
    payload = _build_payload(otlp_decode.EVENT_FLOW, incomplete)

    assert otlp_decode.iter_records(payload) == []


def test_iter_records_handles_multiple_records_in_one_batch() -> None:
    resource = Resource(attributes=[_kv("host.id", "sensor-1")])
    flow_record = LogRecord(
        time_unix_nano=1_700_000_000_000_000_000,
        event_name=otlp_decode.EVENT_FLOW,
        attributes=[_kv(k, v) for k, v in _FLOW_ATTRS.items()],
    )
    dns_record = LogRecord(
        time_unix_nano=1_700_000_000_000_000_000,
        event_name=otlp_decode.EVENT_DNS_QUERY,
        attributes=[_kv(k, v) for k, v in _DNS_ATTRS.items()],
    )
    resource_logs = ResourceLogs(
        resource=resource, scope_logs=[ScopeLogs(log_records=[flow_record, dns_record])]
    )
    payload = ExportLogsServiceRequest(resource_logs=[resource_logs]).SerializeToString()

    records = otlp_decode.iter_records(payload)

    assert [kind for kind, _, _ in records] == ["flow", "dns"]
