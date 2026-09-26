"""Cross-checks the batch pipeline (`build_features`, used for training) against
the incremental accumulator (`WindowAccumulator`, used by the online detector)
on the same generated data, since the whole point of computing features once
is that both callers see the same numbers."""

import datetime as dt
import random
from typing import Any, cast

import polars as pl
import pytest

from netstream_ml import features

NUMERIC_COLUMNS = [
    c
    for c in features.FEATURE_COLUMNS
    if c not in ("flows_count", "unique_dst_ips", "unique_dst_ports")
]


def _generate(
    seed: int, sources: int, duration_s: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    base = dt.datetime(2026, 1, 1)
    flow_records: list[dict[str, Any]] = []
    dns_records: list[dict[str, Any]] = []

    for source in range(sources):
        src_ip = f"10.10.0.{10 + source}"
        for second in range(duration_s):
            ts = base + dt.timedelta(seconds=second)
            for _ in range(rng.randint(0, 3)):
                packets = rng.randint(1, 50)
                tcp_syn = rng.randint(0, packets)
                flow_records.append(
                    {
                        "ts": ts,
                        "src_ip": src_ip,
                        "dst_ip": f"10.20.0.{rng.randint(10, 15)}",
                        "dst_port": rng.choice([80, 443, 22, 53]),
                        "direction": rng.choice(["receive", "transmit"]),
                        "packets": packets,
                        "ip_bytes": packets * rng.randint(40, 1500),
                        "payload_bytes": packets * rng.randint(0, 1400),
                        "tcp_syn": tcp_syn,
                        "tcp_synack": rng.randint(0, tcp_syn),
                        "tcp_rst": rng.randint(0, packets),
                        "aggregated": rng.choice([0, 0, 0, 1, 2]),
                        "size_le64": rng.randint(0, packets),
                        "size_le128": rng.randint(0, packets),
                        "size_le256": 0,
                        "size_le512": 0,
                        "size_le1024": 0,
                        "size_gt1024": 0,
                        "iat_count": max(packets - 1, 0),
                        "iat_sum_us": rng.randint(0, 10_000),
                        "iat_sumsq_us": rng.randint(0, 1_000_000),
                        "label": "syn_flood" if source == 0 else "benign",
                        "scenario": "syn_flood" if source == 0 else "background",
                    }
                )
            for _ in range(rng.randint(0, 2)):
                length = rng.randint(5, 60)
                dns_records.append(
                    {
                        "ts": ts,
                        "src_ip": src_ip,
                        "qname": f"{rng.randint(0, 9999)}.example.test",
                        "qname_length": length,
                        "entropy": rng.uniform(1.0, 4.5),
                        "digit_ratio": rng.uniform(0.0, 0.5),
                        "qtype": rng.choice(["A", "AAAA", "TXT", "NULL"]),
                        "transport": rng.choice(["udp", "udp", "udp", "tcp"]),
                        "unique_subdomains": rng.randint(1, 20),
                    }
                )

    flow_records.sort(key=lambda r: r["ts"])
    dns_records.sort(key=lambda r: r["ts"])
    return flow_records, dns_records


def _batch_rows(
    flow_records: list[dict[str, Any]],
    dns_records: list[dict[str, Any]],
    window: features.WindowSpec,
) -> dict[tuple[str, int], dict[str, Any]]:
    flows = pl.DataFrame(flow_records)
    dns = pl.DataFrame(dns_records)
    combined = features.build_features(flows, dns, window)
    return {(str(row["src_ip"]), int(row["window_start"])): row for row in combined.to_dicts()}


def _streaming_rows(
    flow_records: list[dict[str, Any]],
    dns_records: list[dict[str, Any]],
    window: features.WindowSpec,
) -> dict[tuple[str, int], dict[str, Any]]:
    accumulator = features.WindowAccumulator(window)
    events = sorted(
        [("flow", r) for r in flow_records] + [("dns", r) for r in dns_records],
        key=lambda item: item[1]["ts"],
    )
    for kind, record in events:
        if kind == "flow":
            accumulator.add_flow_record(record)
        else:
            accumulator.add_dns_record(record)
    accumulator.flush()
    return {
        (cast(str, row["src_ip"]), cast(int, row["window_start"])): row
        for row in accumulator.pop_completed()
    }


@pytest.mark.parametrize("window_seconds", [1, 5, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_and_streaming_agree_on_generated_traffic(seed: int, window_seconds: int) -> None:
    flow_records, dns_records = _generate(seed, sources=3, duration_s=30)
    window = features.WindowSpec(window_seconds)

    batch = _batch_rows(flow_records, dns_records, window)
    streaming = _streaming_rows(flow_records, dns_records, window)

    assert set(batch) == set(streaming)

    for key, batch_row in batch.items():
        streaming_row = streaming[key]
        for column in NUMERIC_COLUMNS:
            batch_value = batch_row[column]
            streaming_value = streaming_row[column]
            if batch_value is None or streaming_value is None:
                assert batch_value == streaming_value, column
            else:
                assert batch_value == pytest.approx(streaming_value, abs=1e-9), column
        assert batch_row["label"] == streaming_row["label"]
        assert batch_row["scenario"] == streaming_row["scenario"]


def test_batch_and_streaming_agree_when_a_source_has_only_dns_traffic() -> None:
    window = features.WindowSpec(5)
    ts = dt.datetime(2026, 1, 1)
    dns_only = [
        {
            "ts": ts,
            "src_ip": "10.10.0.20",
            "qname": "a.example.test",
            "qname_length": 14,
            "entropy": 3.1,
            "digit_ratio": 0.0,
            "qtype": "A",
            "transport": "udp",
            "unique_subdomains": 1,
        }
    ]

    batch = _batch_rows([], dns_only, window)
    streaming = _streaming_rows([], dns_only, window)

    assert set(batch) == set(streaming) == set()
