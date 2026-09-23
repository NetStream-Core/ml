from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from netstream_ml import __version__
from netstream_ml.clickhouse import ClickHouseConfig, export_to_file, query_value

FLOWS_QUERY = "SELECT * FROM labeled_flows ORDER BY ts FORMAT Parquet"
DNS_QUERY = "SELECT * FROM labeled_dns ORDER BY ts FORMAT Parquet"
LABELS_QUERY = "SELECT * FROM labels ORDER BY start_ts FORMAT Parquet"

EXPORTS: tuple[tuple[str, str, str], ...] = (
    ("flows.parquet", "labeled_flows", FLOWS_QUERY),
    ("dns.parquet", "labeled_dns", DNS_QUERY),
    ("labels.parquet", "labels", LABELS_QUERY),
)


@dataclass(frozen=True, slots=True)
class ExportedFile:
    name: str
    table: str
    query: str
    rows: int
    bytes: int
    sha256: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(config: ClickHouseConfig, table: str) -> int:
    return int(query_value(config, f"SELECT count() FROM {table}"))


def _scenarios(config: ClickHouseConfig) -> list[str]:
    raw = query_value(
        config, "SELECT arrayStringConcat(arraySort(groupUniqArray(scenario)), ',') FROM labels"
    )
    return raw.split(",") if raw else []


def _campaign_manifest(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    data = cast(dict[str, object], json.loads(path.read_text()))
    data["_source_path"] = str(path)
    data["_source_sha256"] = sha256_of(path)
    return data


def export_dataset(
    config: ClickHouseConfig,
    out_dir: Path,
    campaign_manifest: Path | None = None,
) -> Path:
    """Exports the labelled flow, DNS and labels tables to Parquet files under out_dir.

    Writes a manifest.json alongside them recording the exporter version, the
    ClickHouse server version, row counts and the sha256 of every file, so the
    export can be traced back to the data it came from.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exported_at_ms = int(time.time() * 1000)

    files = []
    for name, table, query in EXPORTS:
        out_path = out_dir / name
        rows = _row_count(config, table)
        size = export_to_file(config, query, out_path)
        files.append(
            ExportedFile(
                name=name,
                table=table,
                query=query,
                rows=rows,
                bytes=size,
                sha256=sha256_of(out_path),
            )
        )

    manifest = {
        "netstream_ml_version": __version__,
        "exported_at_ms": exported_at_ms,
        "clickhouse": {
            "url": config.url,
            "database": config.database,
            "server_version": query_value(config, "SELECT version()"),
        },
        "files": [asdict(f) for f in files],
        "scenarios": _scenarios(config),
        "campaign_manifest": _campaign_manifest(campaign_manifest),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest_path
