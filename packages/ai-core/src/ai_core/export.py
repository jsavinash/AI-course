"""Signed model export: the packaging step of the model lifecycle.

Lifecycle: train.py -> MLflow -> export.py -> artifacts/ -> FastAPI -> prediction
                                              (signed file)

`export_signed_model` packages a trained model artifact into a single
self-describing, HMAC-SHA256-signed file (`model.signed`). The serving layer
(`verify_signed_model`) refuses to load a package whose signature or payload
hash does not verify, guaranteeing that only artifacts produced by a trusted
export step ever reach prediction.

Signed-file format (JSON)::

    {
      "format": "mlops-signed-model",
      "format_version": 1,
      "manifest": { ... metadata, incl. payload_sha256 ... },
      "signature": "<hex hmac-sha256 over the canonical manifest bytes>",
      "payload_base64": "<base64 of the model artifact bytes>"
    }"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess  # noqa: S404 - used only to read the git SHA
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SIGNING_KEY_ENV = "MODEL_SIGNING_KEY"
FORMAT_NAME = "mlops-signed-model"
FORMAT_VERSION = 1


class SignatureError(Exception):
    """Raised when a signed model fails signature or integrity verification."""


def _canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes so signatures are stable across platforms."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signing_key(explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key
    return os.getenv(SIGNING_KEY_ENV) or None


def get_git_sha(repo_dir: Path | None = None) -> str | None:
    """Best-effort current git commit SHA (for build provenance metadata)."""
    try:
        return (
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
                cwd=str(repo_dir) if repo_dir else None,
            ).stdout.strip()
        )
    except Exception:
        return None


def sign_payload(manifest: dict[str, Any], signing_key: str) -> str:
    """Return the hex HMAC-SHA256 signature over the canonical manifest bytes."""
    return hmac.new(
        signing_key.encode("utf-8"), _canonical_json(manifest), hashlib.sha256
    ).hexdigest()


def export_signed_model(
    model_path: Path,
    out_dir: Path,
    manifest: dict[str, Any],
    signing_key: str | None = None,
    filename: str = "model.signed",
) -> Path:
    """Package a trained model artifact into a signed, self-describing file.

    Args:
        model_path: Path to the trained model artifact (e.g. ``model.npz``).
        out_dir: Directory to write ``<filename>`` (and ``manifest.json``) into.
        manifest: Metadata dict (metrics, params, mlflow_run_id, git_sha, ...).
            ``payload_sha256``, sizes and timestamps are added automatically.
        signing_key: HMAC key. Falls back to the ``MODEL_SIGNING_KEY`` env var.
        filename: Output file name.

    Returns:
        Path to the written signed file.

    Raises:
        ValueError: If no signing key is available.
    """
    key = _signing_key(signing_key)
    if not key:
        raise ValueError(
            f"No signing key provided. Pass signing_key or set the {SIGNING_KEY_ENV} env var."
        )

    payload = Path(model_path).read_bytes()
    full_manifest = {
        **manifest,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "exported_at": datetime.now(UTC).isoformat(),
    }
    if "git_sha" not in full_manifest:
        full_manifest["git_sha"] = get_git_sha()

    container = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "manifest": full_manifest,
        "signature": sign_payload(full_manifest, key),
        "payload_base64": base64.b64encode(payload).decode("ascii"),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(container, indent=2))

    # Human-readable side-car (same manifest, no payload) for inspection/CI gates.
    (out_dir / "manifest.json").write_text(json.dumps(full_manifest, indent=2))

    logger.info(
        "Signed model exported",
        path=str(out_path),
        model=full_manifest.get("model_name"),
        version=full_manifest.get("model_version"),
        sha256=full_manifest["payload_sha256"],
    )
    return out_path


def verify_signed_model(
    signed_path: Path,
    signing_key: str | None = None,
    require_key: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Verify a signed model file and return ``(payload_bytes, manifest)``.

    Verification fails (raising :class:`SignatureError`) if the file is not a
    valid signed container, the HMAC does not match, or the payload SHA-256
    does not match the signed manifest.

    Args:
        signed_path: Path to the ``model.signed`` file.
        signing_key: Explicit HMAC key; falls back to ``MODEL_SIGNING_KEY``.
        require_key: If True (production default), a missing key is an error.
            If False, verification is skipped with a warning (dev mode).
    """
    path = Path(signed_path)
    try:
        container = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SignatureError(f"Cannot read signed model {path}: {e}") from e

    if container.get("format") != FORMAT_NAME:
        raise SignatureError(f"Unrecognized model package format in {path}")

    manifest = container.get("manifest")
    signature = container.get("signature")
    if not isinstance(manifest, dict) or not isinstance(signature, str):
        raise SignatureError(f"Malformed signed model package in {path}")

    key = _signing_key(signing_key)
    if key:
        expected = sign_payload(manifest, key)
        if not hmac.compare_digest(expected, signature):
            raise SignatureError(
                f"Signature verification FAILED for {path}. "
                "The artifact may have been tampered with or was signed by another party."
            )
    elif require_key:
        raise SignatureError(
            f"No signing key configured ({SIGNING_KEY_ENV}); refusing to load {path}."
        )
    else:
        logger.warning(
            "Signature verification SKIPPED (no signing key configured) - dev mode only",
            path=str(path),
        )

    payload = base64.b64decode(container.get("payload_base64", ""))
    if hashlib.sha256(payload).hexdigest() != manifest.get("payload_sha256"):
        raise SignatureError(f"Payload hash mismatch for {path}: artifact is corrupted.")

    logger.info(
        "Signed model verified",
        path=str(path),
        model=manifest.get("model_name"),
        version=manifest.get("model_version"),
        sha256=manifest.get("payload_sha256"),
    )
    return payload, manifest


__all__ = [
    "SIGNING_KEY_ENV",
    "SignatureError",
    "export_signed_model",
    "get_git_sha",
    "sign_payload",
    "verify_signed_model",
]
