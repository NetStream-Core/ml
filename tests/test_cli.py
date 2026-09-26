import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from netstream_ml import __version__, cli
from netstream_ml.cli import main
from netstream_ml.clickhouse import ClickHouseConfig


def test_main_without_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: netstream-ml" in capsys.readouterr().out


def test_version_flag_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_data_without_a_subcommand_prints_help_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["data"]) == 2
    assert "usage: netstream-ml" in capsys.readouterr().out


def test_data_export_calls_export_dataset_with_parsed_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen_config: ClickHouseConfig | None = None
    seen_out_dir: Path | None = None
    seen_campaign_manifest: Path | None = None

    def fake_export_dataset(
        config: ClickHouseConfig, out_dir: Path, campaign_manifest: Path | None
    ) -> Path:
        nonlocal seen_config, seen_out_dir, seen_campaign_manifest
        seen_config = config
        seen_out_dir = out_dir
        seen_campaign_manifest = campaign_manifest
        manifest_path = out_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}")
        return manifest_path

    monkeypatch.setattr(cli, "export_dataset", fake_export_dataset)

    campaign_manifest = tmp_path / "campaign.json"
    campaign_manifest.write_text("{}")
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "data",
            "export",
            str(out_dir),
            "--url",
            "http://ch:8123",
            "--user",
            "u",
            "--password",
            "p",
            "--database",
            "netstream",
            "--campaign-manifest",
            str(campaign_manifest),
        ]
    )

    assert exit_code == 0
    assert seen_out_dir == out_dir
    assert seen_campaign_manifest == campaign_manifest
    assert seen_config == ClickHouseConfig(
        url="http://ch:8123", user="u", password="p", database="netstream"
    )
    assert "manifest written to" in capsys.readouterr().out


def test_data_export_reads_clickhouse_defaults_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLICKHOUSE_URL", "http://envhost:8123")
    monkeypatch.setenv("CLICKHOUSE_USER", "envuser")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "envpass")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "envdb")

    seen_config: ClickHouseConfig | None = None

    def fake_export_dataset(
        config: ClickHouseConfig, out_dir: Path, campaign_manifest: Path | None
    ) -> Path:
        nonlocal seen_config
        seen_config = config
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "manifest.json"

    monkeypatch.setattr(cli, "export_dataset", fake_export_dataset)

    main(["data", "export", str(tmp_path / "out")])

    assert seen_config == ClickHouseConfig(
        url="http://envhost:8123", user="envuser", password="envpass", database="envdb"
    )


def _write_flow_dns_dataset(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = dt.datetime(2026, 1, 1)
    flows = pl.DataFrame(
        {
            "ts": [base, base],
            "src_ip": ["10.10.0.10", "10.10.0.10"],
            "dst_ip": ["10.20.0.10", "10.20.0.11"],
            "dst_port": [80, 443],
            "direction": ["receive", "receive"],
            "packets": [10, 5],
            "ip_bytes": [6000, 3000],
            "payload_bytes": [5000, 2500],
            "tcp_syn": [1, 1],
            "tcp_synack": [1, 1],
            "tcp_fin": [0, 0],
            "tcp_rst": [0, 0],
            "aggregated": [0, 0],
            "size_le64": [10, 5],
            "size_le128": [0, 0],
            "size_le256": [0, 0],
            "size_le512": [0, 0],
            "size_le1024": [0, 0],
            "size_gt1024": [0, 0],
            "iat_count": [9, 4],
            "iat_sum_us": [900, 400],
            "iat_sumsq_us": [90000, 40000],
            "label": ["benign", "benign"],
            "scenario": ["background", "background"],
        }
    )
    dns = pl.DataFrame(
        {
            "ts": [base],
            "src_ip": ["10.10.0.10"],
            "qname": ["a.example.com"],
            "qname_length": [13],
            "entropy": [3.0],
            "digit_ratio": [0.0],
            "qtype": ["A"],
            "transport": ["udp"],
            "unique_subdomains": [1],
        }
    )
    flows.write_parquet(out_dir / "flows.parquet")
    dns.write_parquet(out_dir / "dns.parquet")


def test_features_build_writes_one_file_per_window_and_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_flow_dns_dataset(tmp_path)

    exit_code = main(["features", "build", str(tmp_path), "--window", "1", "--window", "5"])

    assert exit_code == 0
    for seconds in (1, 5):
        table = pl.read_parquet(tmp_path / f"features_w{seconds}.parquet")
        assert table.height == 1
        assert table.row(0, named=True)["packets_total"] == 15

    manifest = json.loads((tmp_path / "features_manifest.json").read_text())
    assert manifest["windows_seconds"] == [1, 5]
    assert "features manifest written to" in capsys.readouterr().out


def test_features_build_defaults_to_three_window_sizes(tmp_path: Path) -> None:
    _write_flow_dns_dataset(tmp_path)

    main(["features", "build", str(tmp_path)])

    manifest = json.loads((tmp_path / "features_manifest.json").read_text())
    assert manifest["windows_seconds"] == [1, 5, 10]


def _write_split_dataset(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = dt.datetime(2026, 1, 1)
    labels = pl.DataFrame(
        {
            "run_id": ["r1", "r2", "r3", "r4", "r5"],
            "scenario": ["syn_flood"] * 5,
        }
    )
    flows = pl.DataFrame(
        {
            "run_id": ["r1", "r2", "r3", "r4", "r5", ""],
            "ts": [base, base, base, base, base, base],
        }
    )
    dns = pl.DataFrame({"run_id": ["r1", ""], "ts": [base, base]})
    labels.write_parquet(out_dir / "labels.parquet")
    flows.write_parquet(out_dir / "flows.parquet")
    dns.write_parquet(out_dir / "dns.parquet")


def test_data_split_writes_a_split_column_and_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_split_dataset(tmp_path)

    exit_code = main(["data", "split", str(tmp_path), "--seed", "0"])

    assert exit_code == 0
    flows = pl.read_parquet(tmp_path / "flows.parquet")
    assert "split" in flows.columns
    assert flows["split"].null_count() == 0

    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["seed"] == 0
    assert manifest["reports"]["flows"]["leaking_run_ids"] == []
    assert "split manifest written to" in capsys.readouterr().out


def _write_features_dataset_with_split(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    benign_packets = list(range(1, 51))
    attack_packets = [500] * 5
    n_benign = len(benign_packets)
    n_attack = len(attack_packets)
    splits = (
        ["train"] * (n_benign // 2)
        + ["val"] * (n_benign // 4)
        + ["test"] * (n_benign - n_benign // 2 - n_benign // 4)
    )
    frame = pl.DataFrame(
        {
            "src_ip": [f"10.0.0.{i}" for i in range(n_benign + n_attack)],
            "window_start": list(range(n_benign + n_attack)),
            "packets_total": benign_packets + attack_packets,
            "label": ["benign"] * n_benign + ["syn_flood"] * n_attack,
            "scenario": ["background"] * n_benign + ["syn_flood"] * n_attack,
            "split": [*splits, "train", "train", "val", "test", "test"],
        }
    )
    frame.write_parquet(out_dir / "features_w5.parquet")


def test_baseline_threshold_writes_a_report_with_metrics_by_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_features_dataset_with_split(tmp_path)

    exit_code = main(["baseline", "threshold", str(tmp_path), "--window", "5"])

    assert exit_code == 0
    report = json.loads((tmp_path / "baseline_threshold_w5.json").read_text())
    assert report["window_seconds"] == 5
    assert "train" in report["metrics_by_split"]
    assert "test" in report["test_metrics_by_scenario"] or report["test_metrics_by_scenario"]
    assert "baseline report written to" in capsys.readouterr().out


def test_baseline_threshold_fails_clearly_without_a_split_column(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"packets_total": [1, 2], "label": ["benign", "benign"]}).write_parquet(
        tmp_path / "features_w5.parquet"
    )

    exit_code = main(["baseline", "threshold", str(tmp_path)])

    assert exit_code == 1


def test_baseline_isolation_forest_writes_a_report_with_metrics_by_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_features_dataset_with_split(tmp_path)

    exit_code = main(
        ["baseline", "isolation-forest", str(tmp_path), "--window", "5", "--contamination", "0.1"]
    )

    assert exit_code == 0
    report = json.loads((tmp_path / "isolation_forest_w5.json").read_text())
    assert report["window_seconds"] == 5
    assert report["contamination"] == 0.1
    assert "train" in report["metrics_by_split"]
    assert "test" in report["test_metrics_by_scenario"] or report["test_metrics_by_scenario"]
    assert "isolation forest report written to" in capsys.readouterr().out


def test_baseline_isolation_forest_fails_clearly_without_a_split_column(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"packets_total": [1, 2], "label": ["benign", "benign"]}).write_parquet(
        tmp_path / "features_w5.parquet"
    )

    exit_code = main(["baseline", "isolation-forest", str(tmp_path)])

    assert exit_code == 1


def test_baseline_random_forest_writes_a_report_with_metrics_by_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_features_dataset_with_split(tmp_path)

    exit_code = main(["baseline", "random-forest", str(tmp_path), "--window", "5"])

    assert exit_code == 0
    report = json.loads((tmp_path / "random_forest_w5.json").read_text())
    assert report["window_seconds"] == 5
    assert "candidates_tried" in report["tuning"]
    assert "train" in report["metrics_by_split"]
    assert "test" in report["test_metrics_by_scenario"] or report["test_metrics_by_scenario"]
    out = capsys.readouterr().out
    assert "tuned:" in out
    assert "random forest report written to" in out


def test_baseline_random_forest_fails_clearly_without_a_split_column(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"packets_total": [1, 2], "label": ["benign", "benign"]}).write_parquet(
        tmp_path / "features_w5.parquet"
    )

    exit_code = main(["baseline", "random-forest", str(tmp_path)])

    assert exit_code == 1


def test_evaluate_compare_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_features_dataset_with_split(tmp_path)

    exit_code = main(["evaluate", "compare", str(tmp_path), "--window", "5", "--seeds", "0", "1"])

    assert exit_code == 0
    report = json.loads((tmp_path / "evaluate_compare_w5.json").read_text())
    assert report["window_seconds"] == 5
    assert report["threshold"]["precision_recall_curve"]["average_precision"] > 0
    assert len(report["isolation_forest"]["seed_sensitivity"]["values"]) == 2
    assert len(report["random_forest"]["seed_sensitivity"]["values"]) == 2
    assert "candidates_tried" in report["random_forest"]["tuning"]
    assert "threshold_vs_isolation_forest" in report["mcnemar"]
    assert "threshold_vs_random_forest" in report["mcnemar"]
    assert "isolation_forest_vs_random_forest" in report["mcnemar"]
    out = capsys.readouterr().out
    assert "average_precision" in out
    assert "comparison report written to" in out


def test_evaluate_compare_fails_clearly_without_test_rows(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "packets_total": [1, 2, 3],
            "label": ["benign", "benign", "benign"],
            "split": ["train", "train", "val"],
        }
    ).write_parquet(tmp_path / "features_w5.parquet")

    exit_code = main(["evaluate", "compare", str(tmp_path)])

    assert exit_code == 1
