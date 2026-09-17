"""Tests for the signed model export lifecycle (export.py -> verify)."""

import base64
import json

import numpy as np
import pytest
from ai_core.export import (
    SIGNING_KEY_ENV,
    SignatureError,
    export_signed_model,
    sign_payload,
    verify_signed_model,
)


@pytest.fixture()
def model_npz(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(path, weights=np.array([0.1, 0.2, 0.3]), bias=np.float64(0.05))
    return path


def _export(tmp_path, model_npz, key="test-key", **kwargs):
    return export_signed_model(
        model_path=model_npz,
        out_dir=tmp_path / "out" / "1.0.0",
        manifest={"model_name": "test-model", "model_version": "1.0.0"},
        signing_key=key,
        **kwargs,
    )


def test_export_and_verify_roundtrip(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz)
    assert signed.name == "model.signed"
    assert (signed.parent / "manifest.json").exists()

    payload, manifest = verify_signed_model(signed, signing_key="test-key")
    assert manifest["model_name"] == "test-model"
    assert manifest["payload_sha256"]
    # Payload round-trips exactly
    npz_file = tmp_path / "roundtrip.npz"
    npz_file.write_bytes(payload)
    restored = np.load(npz_file)
    assert np.allclose(restored["weights"], [0.1, 0.2, 0.3])
    assert float(restored["bias"]) == pytest.approx(0.05)


def test_tampered_signature_rejected(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz)

    container = json.loads(signed.read_text())
    container["signature"] = "0" * 64
    signed.write_text(json.dumps(container))

    with pytest.raises(SignatureError, match="Signature verification FAILED"):
        verify_signed_model(signed, signing_key="test-key")


def test_tampered_payload_rejected(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz)

    container = json.loads(signed.read_text())
    payload = bytearray(base64.b64decode(container["payload_base64"]))
    payload[0] ^= 0xFF  # flip one bit inside the model weights
    container["payload_base64"] = base64.b64encode(bytes(payload)).decode()
    signed.write_text(json.dumps(container))

    with pytest.raises(SignatureError, match="hash mismatch|Signature verification"):
        verify_signed_model(signed, signing_key="test-key")


def test_wrong_key_rejected(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz, key="key-a")
    with pytest.raises(SignatureError):
        verify_signed_model(signed, signing_key="key-b")


def test_missing_key_in_production_mode_rejected(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz)
    with pytest.raises(SignatureError, match="No signing key configured"):
        verify_signed_model(signed, signing_key=None, require_key=True)


def test_missing_key_in_dev_mode_warns_but_loads(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    signed = _export(tmp_path, model_npz)
    payload, manifest = verify_signed_model(signed, signing_key=None, require_key=False)
    assert manifest["model_version"] == "1.0.0"
    assert payload[:2] == b"PK"  # npz is a zip container


def test_signing_key_from_env(tmp_path, model_npz, monkeypatch):
    monkeypatch.setenv(SIGNING_KEY_ENV, "env-key")
    signed = export_signed_model(
        model_path=model_npz,
        out_dir=tmp_path / "out",
        manifest={"model_name": "m"},
    )
    # Verifiable with the same env key
    payload, _ = verify_signed_model(signed)
    assert payload
    # But not with a different key
    with pytest.raises(SignatureError):
        verify_signed_model(signed, signing_key="other")


def test_export_requires_key(tmp_path, model_npz, monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match="No signing key"):
        export_signed_model(
            model_path=model_npz, out_dir=tmp_path / "out", manifest={}, signing_key=None
        )


def test_signature_is_deterministic():
    manifest = {"model_name": "m", "payload_sha256": "abc"}
    assert sign_payload(manifest, "k") == sign_payload(dict(manifest), "k")
