from __future__ import annotations

import io
import json
import urllib.error
import urllib.request as urllib_request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib

from netstream_ml.detector import Baseline


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    backend_url: str = "http://127.0.0.1:8080"
    timeout: float = 30.0


class RegistryError(RuntimeError):
    pass


def _request(
    config: RegistryConfig,
    path: str,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
) -> bytes:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib_request.Request(
        f"{config.backend_url}{path}", data=data, method=method, headers=headers
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
        config, "/api/v1/models", method="POST", data=payload, content_type="application/json"
    )
    return cast(dict[str, object], json.loads(body))


def upload_artifact(
    config: RegistryConfig, name: str, version: str, path: Path
) -> dict[str, object]:
    body = _request(
        config,
        f"/api/v1/models/{name}/{version}/artifact",
        method="PUT",
        data=path.read_bytes(),
        content_type="application/octet-stream",
    )
    return cast(dict[str, object], json.loads(body))


def download_artifact(config: RegistryConfig, name: str, version: str) -> bytes:
    """The backend 302-redirects this to a presigned, time-limited S3 URL —
    `urllib.request.urlopen` follows that redirect transparently, so this
    never needs its own S3 credentials."""
    return _request(config, f"/api/v1/models/{name}/{version}/artifact")


def load_model(config: RegistryConfig, name: str, version: str | None = None) -> Baseline:
    resolved_version = version or latest_version(config, name)
    artifact = download_artifact(config, name, resolved_version)
    model: Baseline = joblib.load(io.BytesIO(artifact))
    return model


def publish_model(
    config: RegistryConfig, model_path: Path, metadata: ModelMetadata
) -> dict[str, object]:
    """Registers the catalog entry, then uploads the `joblib`-saved artifact
    it points to — `UploadModelArtifact` on the backend 404s until the entry
    exists, so the order matters."""
    register_model(config, metadata)
    return upload_artifact(config, metadata.name, metadata.version, model_path)
