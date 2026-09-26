import datetime as dt

import polars as pl
import pytest

from netstream_ml import features


def _ts(seconds: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1) + dt.timedelta(seconds=seconds)


def _flow_row(  # noqa: PLR0913
    ts: int,
    *,
    src_ip: str = "10.10.0.10",
    dst_ip: str = "10.20.0.10",
    dst_port: int = 80,
    direction: str = "receive",
    packets: int = 10,
    ip_bytes: int = 6000,
    payload_bytes: int = 5000,
    tcp_syn: int = 0,
    tcp_synack: int = 0,
    tcp_rst: int = 0,
    aggregated: int = 0,
    size_le64: int = 0,
    size_le128: int = 0,
    size_le256: int = 0,
    size_le512: int = 0,
    size_le1024: int = 0,
    size_gt1024: int = 0,
    iat_count: int = 0,
    iat_sum_us: int = 0,
    iat_sumsq_us: int = 0,
    label: str = "benign",
    scenario: str = "background",
) -> dict[str, object]:
    return {
        "ts": _ts(ts),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "direction": direction,
        "packets": packets,
        "ip_bytes": ip_bytes,
        "payload_bytes": payload_bytes,
        "tcp_syn": tcp_syn,
        "tcp_synack": tcp_synack,
        "tcp_rst": tcp_rst,
        "aggregated": aggregated,
        "size_le64": size_le64,
        "size_le128": size_le128,
        "size_le256": size_le256,
        "size_le512": size_le512,
        "size_le1024": size_le1024,
        "size_gt1024": size_gt1024,
        "iat_count": iat_count,
        "iat_sum_us": iat_sum_us,
        "iat_sumsq_us": iat_sumsq_us,
        "label": label,
        "scenario": scenario,
    }


def _dns_row(  # noqa: PLR0913
    ts: int,
    *,
    src_ip: str = "10.10.0.10",
    qname: str = "a.example.com",
    qname_length: int = 13,
    entropy: float = 3.0,
    digit_ratio: float = 0.0,
    qtype: str = "A",
    transport: str = "udp",
    unique_subdomains: int = 1,
) -> dict[str, object]:
    return {
        "ts": _ts(ts),
        "src_ip": src_ip,
        "qname": qname,
        "qname_length": qname_length,
        "entropy": entropy,
        "digit_ratio": digit_ratio,
        "qtype": qtype,
        "transport": transport,
        "unique_subdomains": unique_subdomains,
    }


def test_window_spec_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        features.WindowSpec(0)


def test_window_bucket_floors_to_the_window_size() -> None:
    window = features.WindowSpec(5)
    assert window.bucket(1767225600.0) == window.bucket(1767225604.0)
    assert window.bucket(1767225604.0) != window.bucket(1767225605.0)


def test_flow_window_features_are_computed_from_known_inputs() -> None:
    rows = pl.DataFrame(
        [
            _flow_row(
                0,
                packets=10,
                ip_bytes=6000,
                tcp_syn=4,
                tcp_synack=1,
                tcp_rst=1,
                size_le64=6,
                size_gt1024=4,
                iat_count=9,
                iat_sum_us=900,
                iat_sumsq_us=100_000,
                label="syn_flood",
                scenario="syn_flood",
            ),
            _flow_row(
                1,
                dst_port=443,
                direction="transmit",
                packets=5,
                ip_bytes=3000,
                size_le64=5,
                label="syn_flood",
                scenario="syn_flood",
            ),
        ]
    )

    result = features.flow_window_features(rows, features.WindowSpec(5))

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["flows_count"] == 2
    assert row["packets_total"] == 15
    assert row["ip_bytes_total"] == 9000
    assert row["unique_dst_ips"] == 1
    assert row["unique_dst_ports"] == 2
    assert row["receive_ratio"] == pytest.approx(10 / 15)
    assert row["syn_ratio"] == pytest.approx(4 / 15)
    assert row["synack_ratio"] == pytest.approx(1 / 4)
    assert row["rst_ratio"] == pytest.approx(1 / 15)
    assert row["size_le64_ratio"] == pytest.approx(11 / 15)
    assert row["size_gt1024_ratio"] == pytest.approx(4 / 15)
    assert row["iat_mean_us"] == pytest.approx(100.0)
    assert row["iat_std_us"] == pytest.approx((100_000 / 9 - (900 / 9) ** 2) ** 0.5)
    assert row["label"] == "syn_flood"
    assert row["scenario"] == "syn_flood"


def test_flow_window_features_uses_the_last_records_label() -> None:
    rows = pl.DataFrame(
        [
            _flow_row(0, label="syn_flood", scenario="syn_flood"),
            _flow_row(1, label="benign", scenario="background"),
        ]
    )
    result = features.flow_window_features(rows, features.WindowSpec(5))
    row = result.row(0, named=True)
    assert row["label"] == "benign"
    assert row["scenario"] == "background"


def test_flow_window_features_separates_two_windows() -> None:
    rows = pl.DataFrame([_flow_row(0), _flow_row(6)])
    result = features.flow_window_features(rows, features.WindowSpec(5)).sort("window_start")
    base = int(_ts(0).replace(tzinfo=dt.UTC).timestamp())
    assert result.height == 2
    assert result["window_start"].to_list() == [base, base + 5]


def test_flow_window_features_of_an_empty_frame_has_the_expected_schema() -> None:
    empty = pl.DataFrame([_flow_row(0)]).clear()
    result = features.flow_window_features(empty, features.WindowSpec(5))
    assert result.height == 0
    assert set(features.FLOW_FEATURE_COLUMNS) <= set(result.columns)


def test_aggregated_ratio_reflects_the_share_of_budget_exceeded_packets() -> None:
    rows = pl.DataFrame(
        [
            _flow_row(0, packets=100, aggregated=0),
            _flow_row(0, dst_port=81, packets=300, aggregated=1),
        ]
    )
    row = features.flow_window_features(rows, features.WindowSpec(5)).row(0, named=True)
    assert row["aggregated_ratio"] == pytest.approx(300 / 400)


def test_dns_window_features_are_computed_from_known_inputs() -> None:
    rows = pl.DataFrame(
        [
            _dns_row(
                0,
                qname="a.tunnel.test",
                qname_length=40,
                entropy=3.9,
                qtype="TXT",
                transport="tcp",
                unique_subdomains=5,
            ),
            _dns_row(
                1,
                qname="b.tunnel.test",
                qname_length=44,
                entropy=4.1,
                qtype="NULL",
                unique_subdomains=6,
            ),
            _dns_row(
                2,
                qname="a.tunnel.test",
                qname_length=40,
                entropy=3.9,
                qtype="A",
                unique_subdomains=6,
            ),
        ]
    )
    row = features.dns_window_features(rows, features.WindowSpec(5)).row(0, named=True)
    assert row["dns_queries_count"] == 3
    assert row["dns_unique_qnames"] == 2
    assert row["dns_qname_length_mean"] == pytest.approx((40 + 44 + 40) / 3)
    assert row["dns_qname_length_max"] == 44
    assert row["dns_entropy_max"] == pytest.approx(4.1)
    assert row["dns_txt_null_ratio"] == pytest.approx(2 / 3)
    assert row["dns_tcp_ratio"] == pytest.approx(1 / 3)
    assert row["dns_unique_subdomains_max"] == 6


def test_dns_window_features_of_an_empty_frame_has_the_expected_schema() -> None:
    empty = pl.DataFrame([_dns_row(0)]).clear()
    result = features.dns_window_features(empty, features.WindowSpec(5))
    assert result.height == 0
    assert set(features.DNS_FEATURE_COLUMNS) <= set(result.columns)


def test_combine_features_fills_dns_counts_with_zero_when_a_source_made_no_queries() -> None:
    flows = features.flow_window_features(pl.DataFrame([_flow_row(0)]), features.WindowSpec(5))
    dns = features.dns_window_features(pl.DataFrame([_dns_row(0)]).clear(), features.WindowSpec(5))

    combined = features.combine_features(flows, dns)

    row = combined.row(0, named=True)
    assert row["dns_queries_count"] == 0
    assert row["dns_unique_qnames"] == 0
    assert row["dns_qname_length_mean"] is None


def test_combine_features_joins_matching_windows() -> None:
    flows = features.flow_window_features(pl.DataFrame([_flow_row(0)]), features.WindowSpec(5))
    dns = features.dns_window_features(pl.DataFrame([_dns_row(0)]), features.WindowSpec(5))

    combined = features.combine_features(flows, dns)

    row = combined.row(0, named=True)
    assert row["dns_queries_count"] == 1
    assert row["packets_total"] == 10


def test_flow_window_features_carries_through_a_split_column_when_present() -> None:
    rows = pl.DataFrame([_flow_row(0)]).with_columns(pl.lit("train").alias("split"))
    result = features.flow_window_features(rows, features.WindowSpec(5))
    assert result.row(0, named=True)["split"] == "train"


def test_flow_window_features_has_no_split_column_when_absent() -> None:
    rows = pl.DataFrame([_flow_row(0)])
    result = features.flow_window_features(rows, features.WindowSpec(5))
    assert "split" not in result.columns
