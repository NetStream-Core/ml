import email.message
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from netstream_ml import clickhouse


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_query_text_sends_query_as_body_and_credentials_as_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_url = ""
    seen_data: bytes = b""
    seen_timeout = 0.0

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        nonlocal seen_url, seen_data, seen_timeout
        seen_url = request.full_url
        assert isinstance(request.data, bytes)
        seen_data = request.data
        seen_timeout = timeout
        return FakeResponse(b"42\n")

    monkeypatch.setattr("netstream_ml.clickhouse.urllib_request.urlopen", fake_urlopen)

    config = clickhouse.ClickHouseConfig(
        url="http://ch:8123", user="u", password="p", database="netstream"
    )
    result = clickhouse.query_text(config, "SELECT 42", timeout=5)

    assert result == "42\n"
    assert seen_data == b"SELECT 42"
    assert seen_timeout == 5
    assert "user=u" in seen_url
    assert "password=p" in seen_url
    assert "database=netstream" in seen_url


def test_query_value_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "netstream_ml.clickhouse.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(b"  7\n"),
    )
    config = clickhouse.ClickHouseConfig()
    assert clickhouse.query_value(config, "SELECT count()") == "7"


def test_export_to_file_writes_the_response_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "netstream_ml.clickhouse.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(b"parquet-bytes"),
    )
    config = clickhouse.ClickHouseConfig()
    out_path = tmp_path / "flows.parquet"

    written = clickhouse.export_to_file(config, "SELECT * FROM flows FORMAT Parquet", out_path)

    assert written == len(b"parquet-bytes")
    assert out_path.read_bytes() == b"parquet-bytes"


def test_http_errors_are_wrapped_with_the_server_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_http_error(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            email.message.Message(),
            io.BytesIO(b"Table not found"),
        )

    monkeypatch.setattr("netstream_ml.clickhouse.urllib_request.urlopen", raise_http_error)
    config = clickhouse.ClickHouseConfig()

    with pytest.raises(clickhouse.ClickHouseError, match="Table not found"):
        clickhouse.query_text(config, "SELECT * FROM missing")


def test_connection_errors_are_wrapped_with_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_url_error(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("netstream_ml.clickhouse.urllib_request.urlopen", raise_url_error)
    config = clickhouse.ClickHouseConfig(url="http://ch:8123")

    with pytest.raises(clickhouse.ClickHouseError, match="http://ch:8123"):
        clickhouse.query_text(config, "SELECT 1")
