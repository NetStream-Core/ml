import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from netstream_ml import __version__
from netstream_ml.clickhouse import ClickHouseConfig
from netstream_ml.dataset import export_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netstream-ml")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    data = subparsers.add_parser("data", help="build and inspect datasets")
    data_subparsers = data.add_subparsers(dest="data_command")

    export = data_subparsers.add_parser(
        "export", help="export labelled flows, DNS queries and labels to parquet"
    )
    export.add_argument(
        "out_dir", type=Path, help="directory to write the parquet files and manifest into"
    )
    export.add_argument(
        "--url",
        default=os.environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123"),
        help="ClickHouse HTTP interface URL",
    )
    export.add_argument("--user", default=os.environ.get("CLICKHOUSE_USER", "netstream"))
    export.add_argument(
        "--password", default=os.environ.get("CLICKHOUSE_PASSWORD", "netstream-dev")
    )
    export.add_argument("--database", default=os.environ.get("CLICKHOUSE_DATABASE", "netstream"))
    export.add_argument(
        "--campaign-manifest",
        type=Path,
        default=None,
        help="campaign manifest (from deploy's lab/campaign.py) to embed for traceability",
    )
    export.add_argument(
        "--max-memory-usage",
        type=int,
        default=int(os.environ.get("CLICKHOUSE_MAX_MEMORY_USAGE", "2000000000")),
        help="per-query memory limit in bytes; labeled_flows/labeled_dns scan every label "
        "window per row and can need more than the server's default",
    )

    return parser


def _run_export(args: argparse.Namespace) -> int:
    config = ClickHouseConfig(
        url=args.url,
        user=args.user,
        password=args.password,
        database=args.database,
        max_memory_usage=args.max_memory_usage,
    )
    manifest_path = export_dataset(config, args.out_dir, args.campaign_manifest)
    print(f"manifest written to {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "data" and args.data_command == "export":
        return _run_export(args)

    parser.print_help()
    return 2
