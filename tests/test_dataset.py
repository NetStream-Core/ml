import json
from pathlib import Path

import pytest

from netstream_ml import dataset
from netstream_ml.clickhouse import ClickHouseConfig


def test_sha256_of_matches_a_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello")
    assert (
        dataset.sha256_of(path)
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_export_dataset_writes_three_files_and_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_counts = {"labeled_flows": 10, "labeled_dns": 3, "labels": 2}
    queries: list[str] = []

    def fake_query_value(config: ClickHouseConfig, query: str) -> str:
        queries.append(query)
        if query == "SELECT version()":
            return "25.8.33"
        if "groupUniqArray" in query:
            return "benign,syn_flood"
        for table, count in row_counts.items():
            if table in query:
                return str(count)
        raise AssertionError(f"unexpected query: {query}")

    def fake_export_to_file(config: ClickHouseConfig, query: str, out_path: Path) -> int:
        out_path.write_bytes(b"parquet:" + query.encode())
        return out_path.stat().st_size

    monkeypatch.setattr(dataset, "query_value", fake_query_value)
    monkeypatch.setattr(dataset, "export_to_file", fake_export_to_file)

    config = ClickHouseConfig(url="http://ch:8123", database="netstream")
    manifest_path = dataset.export_dataset(config, tmp_path)

    assert manifest_path == tmp_path / "manifest.json"
    for name in ("flows.parquet", "dns.parquet", "labels.parquet"):
        assert (tmp_path / name).exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["clickhouse"] == {
        "url": "http://ch:8123",
        "database": "netstream",
        "server_version": "25.8.33",
    }
    assert manifest["scenarios"] == ["benign", "syn_flood"]
    assert manifest["campaign_manifest"] is None
    assert {f["name"]: f["rows"] for f in manifest["files"]} == {
        "flows.parquet": 10,
        "dns.parquet": 3,
        "labels.parquet": 2,
    }
    for entry in manifest["files"]:
        digest = dataset.sha256_of(tmp_path / entry["name"])
        assert entry["sha256"] == digest


def test_export_dataset_embeds_the_campaign_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset, "query_value", lambda config, query: "0")
    monkeypatch.setattr(
        dataset, "export_to_file", lambda config, query, out_path: out_path.write_bytes(b"x") or 1
    )

    campaign_manifest = tmp_path / "campaign.json"
    campaign_manifest.write_text(json.dumps({"campaign_file": "v1.yaml", "runs_planned": 46}))

    manifest_path = dataset.export_dataset(
        ClickHouseConfig(), tmp_path / "out", campaign_manifest=campaign_manifest
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["campaign_manifest"]["campaign_file"] == "v1.yaml"
    assert manifest["campaign_manifest"]["runs_planned"] == 46
    assert manifest["campaign_manifest"]["_source_path"] == str(campaign_manifest)
    assert manifest["campaign_manifest"]["_source_sha256"] == dataset.sha256_of(campaign_manifest)
