import email.message
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from netstream_ml import model_registry


def _write_ed25519_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """A real Ed25519 keypair, written as PEM files — sign/verify tests use
    real `cryptography` calls throughout, not mocked crypto."""
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signing_key.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path = tmp_path / "verify_key.pem"
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


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


def test_upload_artifact_puts_the_file_bytes_with_a_real_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"fake-joblib-bytes")
    signing_key_path, verify_key_path = _write_ed25519_keypair(tmp_path)
    seen: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["data"] = request.data
        seen["sha256"] = request.get_header("X-artifact-sha256")
        seen["signature"] = request.get_header("X-artifact-signature")
        return FakeResponse(b'{"status": "ok"}')

    monkeypatch.setattr("netstream_ml.model_registry.urllib_request.urlopen", fake_urlopen)
    config = model_registry.RegistryConfig(backend_url="http://backend:8080")

    model_registry.upload_artifact(config, "threshold", "v1", model_path, signing_key_path)

    assert seen["url"] == "http://backend:8080/api/v1/models/threshold/v1/artifact"
    assert seen["method"] == "PUT"
    assert seen["data"] == b"fake-joblib-bytes"
    assert isinstance(seen["sha256"], str) and len(seen["sha256"]) == 64
    # The signature is real, not a stub — it must actually verify.
    assert isinstance(seen["signature"], str)
    model_registry.verify_artifact(
        b"fake-joblib-bytes", seen["sha256"], seen["signature"], verify_key_path
    )


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
    signing_key_path, _ = _write_ed25519_keypair(tmp_path)
    calls: list[str] = []

    def fake_register_model(
        config: model_registry.RegistryConfig, metadata: model_registry.ModelMetadata
    ) -> dict[str, object]:
        calls.append("register")
        return {}

    def fake_upload_artifact(
        config: model_registry.RegistryConfig,
        name: str,
        version: str,
        path: Path,
        signing_key_path: Path,
    ) -> dict[str, object]:
        calls.append("upload")
        return {}

    monkeypatch.setattr("netstream_ml.model_registry.register_model", fake_register_model)
    monkeypatch.setattr("netstream_ml.model_registry.upload_artifact", fake_upload_artifact)
    config = model_registry.RegistryConfig()
    metadata = model_registry.ModelMetadata(name="threshold", version="v1", algorithm="threshold")

    model_registry.publish_model(config, model_path, metadata, signing_key_path)

    assert calls == ["register", "upload"]


def test_sign_and_verify_artifact_round_trip_with_a_real_keypair(tmp_path: Path) -> None:
    signing_key_path, verify_key_path = _write_ed25519_keypair(tmp_path)
    data = b"a real model artifact's bytes"

    sha256_hex, signature_b64 = model_registry.sign_artifact(data, signing_key_path)

    model_registry.verify_artifact(data, sha256_hex, signature_b64, verify_key_path)


def test_verify_artifact_rejects_tampered_bytes(tmp_path: Path) -> None:
    signing_key_path, verify_key_path = _write_ed25519_keypair(tmp_path)
    sha256_hex, signature_b64 = model_registry.sign_artifact(b"original bytes", signing_key_path)

    with pytest.raises(model_registry.RegistryError, match="sha256"):
        model_registry.verify_artifact(
            b"tampered bytes", sha256_hex, signature_b64, verify_key_path
        )


def test_verify_artifact_rejects_a_signature_from_a_different_key(tmp_path: Path) -> None:
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    attacker_signing_key, _ = _write_ed25519_keypair(attacker_dir)
    _, real_verify_key = _write_ed25519_keypair(real_dir)
    data = b"a real model artifact's bytes"
    sha256_hex, forged_signature = model_registry.sign_artifact(data, attacker_signing_key)

    with pytest.raises(model_registry.RegistryError, match="signature"):
        model_registry.verify_artifact(data, sha256_hex, forged_signature, real_verify_key)


def test_load_model_resolves_latest_version_verifies_and_joblib_loads_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signing_key_path, verify_key_path = _write_ed25519_keypair(tmp_path)
    buffer = io.BytesIO()
    joblib.dump({"fake": "model"}, buffer)
    artifact = buffer.getvalue()
    sha256_hex, signature_b64 = model_registry.sign_artifact(artifact, signing_key_path)

    monkeypatch.setattr("netstream_ml.model_registry.latest_version", lambda config, name: "v2")
    monkeypatch.setattr(
        "netstream_ml.model_registry.get_model",
        lambda config, name, version: {
            "artifact_sha256": sha256_hex,
            "artifact_signature": signature_b64,
        },
    )
    monkeypatch.setattr(
        "netstream_ml.model_registry.download_artifact",
        lambda config, name, version: artifact,
    )
    config = model_registry.RegistryConfig()

    loaded: object = model_registry.load_model(config, "threshold", None, verify_key_path)
    assert loaded == {"fake": "model"}


def test_load_model_refuses_an_artifact_with_no_recorded_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, verify_key_path = _write_ed25519_keypair(tmp_path)
    monkeypatch.setattr(
        "netstream_ml.model_registry.get_model",
        lambda config, name, version: {"artifact_sha256": "", "artifact_signature": ""},
    )
    config = model_registry.RegistryConfig()

    with pytest.raises(model_registry.RegistryError, match="no recorded"):
        model_registry.load_model(config, "threshold", "v1", verify_key_path)


def test_load_model_refuses_a_tampered_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    signing_key_path, verify_key_path = _write_ed25519_keypair(tmp_path)
    sha256_hex, signature_b64 = model_registry.sign_artifact(b"original bytes", signing_key_path)

    monkeypatch.setattr(
        "netstream_ml.model_registry.get_model",
        lambda config, name, version: {
            "artifact_sha256": sha256_hex,
            "artifact_signature": signature_b64,
        },
    )
    monkeypatch.setattr(
        "netstream_ml.model_registry.download_artifact",
        lambda config, name, version: b"tampered bytes",
    )
    config = model_registry.RegistryConfig()

    with pytest.raises(model_registry.RegistryError, match="sha256"):
        model_registry.load_model(config, "threshold", "v1", verify_key_path)


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
