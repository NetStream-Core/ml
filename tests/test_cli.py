import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from netstream_ml import __version__
from netstream_ml.cli import main


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


def _write_dataset(out_dir: Path) -> None:
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
    _write_dataset(tmp_path)

    exit_code = main(["data", "split", str(tmp_path), "--seed", "0"])

    assert exit_code == 0
    flows = pl.read_parquet(tmp_path / "flows.parquet")
    assert "split" in flows.columns
    assert flows["split"].null_count() == 0

    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["seed"] == 0
    assert manifest["reports"]["flows"]["leaking_run_ids"] == []
    assert "split manifest written to" in capsys.readouterr().out
