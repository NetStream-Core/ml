from pathlib import Path

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
