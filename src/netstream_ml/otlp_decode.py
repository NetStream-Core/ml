"""Decodes the OTLP log batches `otel-edge`'s Kafka exporter writes to
`netstream.events.v1` (`encoding: otlp_proto` — a serialized
`ExportLogsServiceRequest`, the same bytes `otel-gateway` reads to fill
ClickHouse) into the plain dicts `features.WindowAccumulator` expects.

The attribute names and their mapping here must match `agent`'s
`telemetry/logs.rs` (`flow_attributes`/`dns_attributes`) and the contract
documented in `deploy/README.md` ("Контракт событий") exactly — this module
has no schema of its own, it is a consumer of that one.

Real-time events carry no ground-truth label (that is exactly what the
model exists to predict): `label`/`scenario` on the records built here are
filled with a placeholder, never read back out by any baseline — they only
consume `features.FEATURE_COLUMNS`, not those two fields.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord

EVENT_FLOW = "netstream.flow"
EVENT_DNS_QUERY = "netstream.dns.query"

NO_LABEL = "unknown"


def _any_value_to_python(value: AnyValue) -> object:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bool_value":
        return value.bool_value
    return None


def _attrs_to_dict(attributes: list[KeyValue]) -> dict[str, object]:
    return {kv.key: _any_value_to_python(kv.value) for kv in attributes}


def _event_time(record: LogRecord) -> dt.datetime:
    return dt.datetime.fromtimestamp(record.time_unix_nano / 1e9, tz=dt.UTC)


def _flow_record(attrs: dict[str, object], event_time: dt.datetime) -> dict[str, Any] | None:
    try:
        return {
            "src_ip": attrs["source.address"],
            "ts": event_time,
            "dst_ip": attrs["destination.address"],
            "dst_port": attrs["destination.port"],
            "direction": attrs["network.io.direction"],
            "packets": attrs["netstream.flow.packets"],
            "ip_bytes": attrs["netstream.flow.bytes.ip"],
            "payload_bytes": attrs["netstream.flow.bytes.payload"],
            "tcp_syn": attrs["netstream.flow.tcp.syn"],
            "tcp_synack": attrs["netstream.flow.tcp.synack"],
            "tcp_rst": attrs["netstream.flow.tcp.rst"],
            "aggregated": attrs["netstream.flow.aggregated"],
            "size_le64": attrs["netstream.flow.size.le64"],
            "size_le128": attrs["netstream.flow.size.le128"],
            "size_le256": attrs["netstream.flow.size.le256"],
            "size_le512": attrs["netstream.flow.size.le512"],
            "size_le1024": attrs["netstream.flow.size.le1024"],
            "size_gt1024": attrs["netstream.flow.size.gt1024"],
            "iat_count": attrs["netstream.flow.iat.count"],
            "iat_sum_us": attrs["netstream.flow.iat.sum_us"],
            "iat_sumsq_us": attrs["netstream.flow.iat.sumsq_us"],
            "label": NO_LABEL,
            "scenario": NO_LABEL,
        }
    except KeyError:
        return None


def _dns_record(attrs: dict[str, object], event_time: dt.datetime) -> dict[str, Any] | None:
    try:
        return {
            "src_ip": attrs["source.address"],
            "ts": event_time,
            "qname": attrs["dns.question.name"],
            "qname_length": attrs["netstream.dns.qname.length"],
            "entropy": attrs["netstream.dns.qname.entropy"],
            "digit_ratio": attrs["netstream.dns.qname.digit_ratio"],
            "qtype": attrs["netstream.dns.question.type"],
            "transport": attrs["network.transport"],
            "unique_subdomains": attrs["netstream.dns.unique_subdomains"],
        }
    except KeyError:
        return None


def iter_records(payload: bytes) -> list[tuple[str, str, dict[str, Any]]]:
    """Parses one Kafka message's bytes into `(kind, host_id, record)`
    triples, `kind` being `"flow"` or `"dns"`. Any other event name
    (`netstream.blocklist.hit`, future additions) and any record missing an
    attribute this module knows about are skipped rather than raising —
    a message straddling a schema change should degrade, not take the
    whole consumer loop down.
    """
    request = ExportLogsServiceRequest()
    request.ParseFromString(payload)

    records: list[tuple[str, str, dict[str, Any]]] = []
    for resource_logs in request.resource_logs:
        resource_attrs = _attrs_to_dict(resource_logs.resource.attributes)
        host_id = str(resource_attrs.get("host.id", NO_LABEL))
        for scope_logs in resource_logs.scope_logs:
            for log_record in scope_logs.log_records:
                attrs = _attrs_to_dict(log_record.attributes)
                event_time = _event_time(log_record)
                if log_record.event_name == EVENT_FLOW:
                    flow = _flow_record(attrs, event_time)
                    if flow is not None:
                        records.append(("flow", host_id, flow))
                elif log_record.event_name == EVENT_DNS_QUERY:
                    dns = _dns_record(attrs, event_time)
                    if dns is not None:
                        records.append(("dns", host_id, dns))
    return records
