import email.message
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pytest

from netstream_ml import model_registry


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_list_models_parses_the_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "netstream_ml.model_registry.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(b'[{"name": "threshold", "version": "v1"}]'),
    )
    config = model_registry.RegistryConfig()

    assert model_registry.list_models(config) == [{"name": "threshold", "version": "v1"}]


def test_latest_version_returns_the_first_match_for_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backend orders List() by (name, created_at DESC), so the first row
    # for a given name is already its newest version.
    monkeypatch.setattr(
        "netstream_ml.model_registry.list_models",
        lambda config: [
            {"name": "random-forest", "version": "2026-10-18"},
            {"name": "random-forest", "version": "2026-10-01"},
            {"name": "threshold", "version": "2026-09-01"},
        ],
    )
    config = model_registry.RegistryConfig()

    assert model_registry.latest_version(config, "random-forest") == "2026-10-18"


def test_latest_version_fails_clearly_when_the_name_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("netstream_ml.model_registry.list_models", lambda config: [])
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    with pytest.raises(model_registry.RegistryError, match="no-such-model"):
        model_registry.latest_version(config, "no-such-model")


def test_register_model_posts_json_to_the_models_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["content_type"] = request.get_header("Content-type")
        assert isinstance(request.data, bytes)
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(seen["body"]).encode())

    monkeypatch.setattr("netstream_ml.model_registry.urllib_request.urlopen", fake_urlopen)
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    result = model_registry.register_model(
        config,
        model_registry.ModelMetadata(
            name="threshold",
            version="v1",
            algorithm="threshold",
            features_version="w5",
            dataset_ref="data/v1",
            git_sha="abc123",
            metrics={"test": {"f1": 0.9}},
        ),
    )

    assert seen["url"] == "http://backend:8080/api/v1/models"
    assert seen["method"] == "POST"
    assert seen["content_type"] == "application/json"
    assert seen["body"] == {
        "name": "threshold",
        "version": "v1",
        "algorithm": "threshold",
        "features_version": "w5",
        "dataset_ref": "data/v1",
        "git_sha": "abc123",
        "metrics": {"test": {"f1": 0.9}},
    }
    assert result == seen["body"]


def test_upload_artifact_puts_the_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake-joblib-bytes")
    seen: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["data"] = request.data
        return FakeResponse(b'{"status": "ok"}')

    monkeypatch.setattr("netstream_ml.model_registry.urllib_request.urlopen", fake_urlopen)
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    model_registry.upload_artifact(config, "threshold", "v1", model_path)

    assert seen["url"] == "http://backend:8080/api/v1/models/threshold/v1/artifact"
    assert seen["method"] == "PUT"
    assert seen["data"] == b"fake-joblib-bytes"


def test_download_artifact_follows_the_redirect_transparently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # urllib.request.urlopen follows HTTP redirects on its own, so
    # download_artifact never needs to see the backend's 302 or talk to S3
    # directly — this test only checks it returns what urlopen hands back.
    monkeypatch.setattr(
        "netstream_ml.model_registry.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(b"artifact-bytes"),
    )
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    assert model_registry.download_artifact(config, "threshold", "v1") == b"artifact-bytes"


def test_publish_model_registers_then_uploads_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"bytes")
    calls: list[str] = []

    def fake_register_model(
        config: model_registry.RegistryConfig, metadata: model_registry.ModelMetadata
    ) -> dict[str, object]:
        calls.append("register")
        return {}

    def fake_upload_artifact(
        config: model_registry.RegistryConfig, name: str, version: str, path: Path
    ) -> dict[str, object]:
        calls.append("upload")
        return {}

    monkeypatch.setattr("netstream_ml.model_registry.register_model", fake_register_model)
    monkeypatch.setattr("netstream_ml.model_registry.upload_artifact", fake_upload_artifact)
    config = model_registry.RegistryConfig()
    metadata = model_registry.ModelMetadata(name="threshold", version="v1", algorithm="threshold")

    model_registry.publish_model(config, model_path, metadata)

    assert calls == ["register", "upload"]


def test_load_model_resolves_latest_version_and_joblib_loads_the_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    joblib.dump({"fake": "model"}, buffer)

    monkeypatch.setattr("netstream_ml.model_registry.latest_version", lambda config, name: "v2")
    monkeypatch.setattr(
        "netstream_ml.model_registry.download_artifact",
        lambda config, name, version: buffer.getvalue(),
    )
    config = model_registry.RegistryConfig()

    loaded: object = model_registry.load_model(config, "threshold")
    assert loaded == {"fake": "model"}


def test_http_errors_are_wrapped_with_the_status_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_http_error(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            email.message.Message(),
            io.BytesIO(b"model not found"),
        )

    monkeypatch.setattr("netstream_ml.model_registry.urllib_request.urlopen", raise_http_error)
    config = model_registry.RegistryConfig()

    with pytest.raises(model_registry.RegistryError, match="model not found"):
        model_registry.list_models(config)


def test_connection_errors_are_wrapped_with_the_backend_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("netstream_ml.model_registry.urllib_request.urlopen", raise_url_error)
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    with pytest.raises(model_registry.RegistryError, match="http://backend:8080"):
        model_registry.list_models(config)
