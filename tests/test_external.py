from pathlib import Path

import polars as pl
import pytest

from netstream_ml.external import (
    RECONSTRUCTABLE_FEATURES,
    UNRECONSTRUCTABLE_FEATURES,
    external_window_features,
    load_cic_csv,
)
from netstream_ml.features import FEATURE_COLUMNS, WindowSpec

_HEADER = (
    " Source IP, Destination IP, Destination Port, Timestamp,"
    " Total Fwd Packets, Total Backward Packets,"
    " Total Length of Fwd Packets, Total Length of Bwd Packets,"
    " SYN Flag Count, RST Flag Count, Flow IAT Mean, Flow IAT Std, Label"
)


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "cic.csv"
    path.write_text(_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_reconstructable_and_unreconstructable_partition_feature_columns() -> None:
    assert set(RECONSTRUCTABLE_FEATURES) | set(UNRECONSTRUCTABLE_FEATURES) == set(FEATURE_COLUMNS)
    assert set(RECONSTRUCTABLE_FEATURES) & set(UNRECONSTRUCTABLE_FEATURES) == set()


def test_load_cic_csv_strips_padded_column_names_and_parses_timestamp(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            "10.0.0.1,10.0.0.2,80,5/7/2017 8:42:00,3,2,300,200,1,0,1000.0,50.0,BENIGN",
            "10.0.0.1,10.0.0.3,22,5/7/2017 8:42:05,10,0,600,0,5,3,100.0,10.0,FTP-Patator",
        ],
    )
    frame = load_cic_csv(path)
    assert frame.height == 2
    assert frame["src_ip"].to_list() == ["10.0.0.1", "10.0.0.1"]
    assert frame["ts"].null_count() == 0


def test_load_cic_csv_parses_cic_ddos2019_iso_like_timestamp(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            "172.16.0.5,192.168.50.1,4463,2018-12-01 13:04:45.928673,2,0,766,0,0,0,1.0,0.0,BENIGN",
        ],
    )
    frame = load_cic_csv(path)
    assert frame["ts"].null_count() == 0


def test_load_cic_csv_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("Source IP,Label\n10.0.0.1,BENIGN\n")
    with pytest.raises(ValueError, match="missing expected"):
        load_cic_csv(path)


def test_external_window_features_labels_benign_and_attack(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            "10.0.0.1,10.0.0.2,80,5/7/2017 8:42:00,3,2,300,200,1,0,1000.0,50.0,BENIGN",
            "10.0.0.9,10.0.0.3,22,5/7/2017 8:42:05,10,0,600,0,10,3,100.0,10.0,FTP-Patator",
        ],
    )
    flows = load_cic_csv(path)
    features = external_window_features(flows, WindowSpec(60))

    assert set(features.columns) >= set(RECONSTRUCTABLE_FEATURES) | {
        "src_ip",
        "window_start",
        "label",
        "scenario",
    }
    by_ip = {row["src_ip"]: row for row in features.to_dicts()}
    assert by_ip["10.0.0.1"]["label"] == "benign"
    assert by_ip["10.0.0.9"]["label"] == "attack"
    assert by_ip["10.0.0.9"]["scenario"] == "ftp-patator"
    assert by_ip["10.0.0.9"]["flows_count"] == 1
    assert by_ip["10.0.0.9"]["packets_total"] == 10


def test_external_window_features_syn_and_rst_ratio(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        ["10.0.0.5,10.0.0.6,443,5/7/2017 9:00:00,8,2,800,200,4,2,50.0,5.0,PortScan"],
    )
    flows = load_cic_csv(path)
    features = external_window_features(flows, WindowSpec(60))
    row = features.to_dicts()[0]
    assert row["packets_total"] == 10
    assert row["syn_ratio"] == pytest.approx(0.4)
    assert row["rst_ratio"] == pytest.approx(0.2)


def test_external_window_features_combines_iat_across_flows_in_same_window(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            "10.0.0.7,10.0.0.8,80,5/7/2017 9:00:00,3,0,300,0,0,0,100.0,0.0,BENIGN",
            "10.0.0.7,10.0.0.9,80,5/7/2017 9:00:10,3,0,300,0,0,0,300.0,0.0,BENIGN",
        ],
    )
    flows = load_cic_csv(path)
    features = external_window_features(flows, WindowSpec(60))
    row = features.to_dicts()[0]
    assert row["iat_mean_us"] == pytest.approx(200.0)


def test_external_window_features_empty_input_has_expected_schema() -> None:
    empty = pl.DataFrame(
        schema={
            "src_ip": pl.Utf8,
            "dst_ip": pl.Utf8,
            "dst_port": pl.Int64,
            "fwd_packets": pl.Float64,
            "bwd_packets": pl.Float64,
            "fwd_bytes": pl.Float64,
            "bwd_bytes": pl.Float64,
            "syn_count": pl.Float64,
            "rst_count": pl.Float64,
            "iat_mean_us_raw": pl.Float64,
            "iat_std_us_raw": pl.Float64,
            "label_raw": pl.Utf8,
            "ts": pl.Datetime,
        }
    )
    features = external_window_features(empty, WindowSpec(60))
    assert features.height == 0
    assert set(RECONSTRUCTABLE_FEATURES) <= set(features.columns)
