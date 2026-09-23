from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request as urllib_request
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class ClickHouseConfig:
    url: str = "http://127.0.0.1:8123"
    user: str = "netstream"
    password: str = "netstream-dev"
    database: str = "netstream"
    max_memory_usage: int = 2_000_000_000


class ClickHouseError(RuntimeError):
    pass


def _request(config: ClickHouseConfig, query: str, timeout: float) -> bytes:
    params = urllib.parse.urlencode(
        {
            "user": config.user,
            "password": config.password,
            "database": config.database,
            "max_memory_usage": config.max_memory_usage,
        }
    )
    request = urllib_request.Request(f"{config.url}/?{params}", data=query.encode(), method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return cast(bytes, response.read())
    except urllib.error.HTTPError as error:
        raise ClickHouseError(error.read().decode(errors="replace")) from error
    except urllib.error.URLError as error:
        raise ClickHouseError(f"could not reach {config.url}: {error.reason}") from error


def query_text(config: ClickHouseConfig, query: str, timeout: float = 300) -> str:
    return _request(config, query, timeout).decode()


def query_value(config: ClickHouseConfig, query: str, timeout: float = 300) -> str:
    return query_text(config, query, timeout).strip()


def export_to_file(
    config: ClickHouseConfig, query: str, out_path: Path, timeout: float = 600
) -> int:
    """Runs `query` (must end with a FORMAT clause) and writes the response to out_path.

    Returns the number of bytes written.
    """
    body = _request(config, query, timeout)
    out_path.write_bytes(body)
    return len(body)
