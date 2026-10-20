from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request as urllib_request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

from netstream_ml.detector import Baseline


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    backend_url: str = "http://127.0.0.1:9999"
    timeout: float = 30.0


class RegistryError(RuntimeError):
    pass


def _request(
    config: RegistryConfig,
    path: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    request = urllib_request.Request(
        f"{config.backend_url}{path}", data=data, method=method, headers=headers or {}
    )
    try:
        with urllib_request.urlopen(request, timeout=config.timeout) as response:
            return cast(bytes, response.read())
    except urllib.error.HTTPError as error:
        raise RegistryError(
            f"{method} {path} failed: {error.code} {error.read().decode(errors='replace')}"
        ) from error
    except urllib.error.URLError as error:
        raise RegistryError(f"could not reach {config.backend_url}: {error.reason}") from error


def list_models(config: RegistryConfig) -> list[dict[str, object]]:
    body = _request(config, "/api/v1/models")
    return cast(list[dict[str, object]], json.loads(body))


def get_model(config: RegistryConfig, name: str, version: str) -> dict[str, object]:
    body = _request(config, f"/api/v1/models/{name}/{version}")
    return cast(dict[str, object], json.loads(body))


def latest_version(config: RegistryConfig, name: str) -> str:
    """The backend orders `List` by `(name, created_at DESC)`, so the first
    match for `name` is already its newest registered version — there is no
    separate "latest" alias in the registry."""
    for model in list_models(config):
        if model["name"] == name:
            return cast(str, model["version"])
    raise RegistryError(f"no model named {name!r} is registered in {config.backend_url}")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    name: str
    version: str
    algorithm: str
    features_version: str = ""
    dataset_ref: str = ""
    git_sha: str = ""
    metrics: dict[str, object] | None = None


def register_model(config: RegistryConfig, metadata: ModelMetadata) -> dict[str, object]:
    payload = json.dumps(
        {
            "name": metadata.name,
            "version": metadata.version,
            "algorithm": metadata.algorithm,
            "features_version": metadata.features_version,
            "dataset_ref": metadata.dataset_ref,
            "git_sha": metadata.git_sha,
            "metrics": metadata.metrics or {},
        }
    ).encode()
    body = _request(
        config,
        "/api/v1/models",
        method="POST",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    return cast(dict[str, object], json.loads(body))


def _load_ed25519_private_key(path: Path) -> Ed25519PrivateKey:
    key = load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RegistryError(f"{path} is not an Ed25519 private key")
    return key


def _load_ed25519_public_key(path: Path) -> Ed25519PublicKey:
    key = load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise RegistryError(f"{path} is not an Ed25519 public key")
    return key


def sign_artifact(data: bytes, signing_key_path: Path) -> tuple[str, str]:
    """Returns `(sha256_hex, signature_b64)`: the digest signed, and the
    Ed25519 signature over that digest — both go to the registry alongside
    the artifact, for `verify_artifact` to check before any consumer
    deserializes bytes it downloaded from object storage."""
    key = _load_ed25519_private_key(signing_key_path)
    digest = hashlib.sha256(data).digest()
    signature = key.sign(digest)
    return digest.hex(), base64.b64encode(signature).decode()


def verify_artifact(
    data: bytes, sha256_hex: str, signature_b64: str, verify_key_path: Path
) -> None:
    """Raises `RegistryError` if `data` doesn't match the recorded digest,
    or the digest isn't validly signed by `verify_key_path` — the only
    thing standing between a compromised object store (or a backend with
    its auth bypassed) and `joblib.load` running arbitrary code from
    whatever bytes it was handed."""
    digest = hashlib.sha256(data).digest()
    if digest.hex() != sha256_hex:
        raise RegistryError(
            "artifact sha256 does not match the registry's recorded value — refusing to "
            "load a corrupted or tampered artifact"
        )
    key = _load_ed25519_public_key(verify_key_path)
    try:
        key.verify(base64.b64decode(signature_b64), digest)
    except InvalidSignature as error:
        raise RegistryError(
            "artifact signature verification failed — refusing to load an unsigned or "
            "tampered artifact"
        ) from error


def upload_artifact(
    config: RegistryConfig, name: str, version: str, path: Path, signing_key_path: Path
) -> dict[str, object]:
    data = path.read_bytes()
    sha256_hex, signature_b64 = sign_artifact(data, signing_key_path)
    body = _request(
        config,
        f"/api/v1/models/{name}/{version}/artifact",
        method="PUT",
        data=data,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Artifact-Sha256": sha256_hex,
            "X-Artifact-Signature": signature_b64,
        },
    )
    return cast(dict[str, object], json.loads(body))


def download_artifact(config: RegistryConfig, name: str, version: str) -> bytes:
    """The backend 302-redirects this to a presigned, time-limited S3 URL —
    `urllib.request.urlopen` follows that redirect transparently, so this
    never needs its own S3 credentials."""
    return _request(config, f"/api/v1/models/{name}/{version}/artifact")


def load_model(
    config: RegistryConfig, name: str, version: str | None, verify_key_path: Path
) -> Baseline:
    resolved_version = version or latest_version(config, name)
    model_record = get_model(config, name, resolved_version)
    sha256_hex = cast(str, model_record.get("artifact_sha256") or "")
    signature_b64 = cast(str, model_record.get("artifact_signature") or "")
    if not sha256_hex or not signature_b64:
        raise RegistryError(
            f"{name}@{resolved_version} has no recorded artifact_sha256/artifact_signature — "
            "it predates signed artifacts, or was uploaded without --signing-key; refusing to "
            "load an artifact with no integrity record"
        )

    artifact = download_artifact(config, name, resolved_version)
    verify_artifact(artifact, sha256_hex, signature_b64, verify_key_path)

    model: Baseline = joblib.load(io.BytesIO(artifact))
    return model


def publish_model(
    config: RegistryConfig, model_path: Path, metadata: ModelMetadata, signing_key_path: Path
) -> dict[str, object]:
    """Registers the catalog entry, then uploads the `joblib`-saved artifact
    it points to, signed with `signing_key_path` — `UploadModelArtifact` on
    the backend 404s until the entry exists, so the order matters."""
    register_model(config, metadata)
    return upload_artifact(config, metadata.name, metadata.version, model_path, signing_key_path)
