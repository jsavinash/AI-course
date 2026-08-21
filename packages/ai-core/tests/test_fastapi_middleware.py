"""Tests for production-grade FastAPI middleware in ai_core.fastapi_middleware.

Cross-cutting hardening every app inherits via ``add_observability_middleware``:
structured error envelopes, CORS, request size limiting, opt-in API-key auth
and rate limiting.

Integration tests drive the app through ``TestClient`` for the paths that are
stable under the test runner. Auth / rate-limit / size-limit *logic* is covered
by unit tests of the underlying dependency factories, which avoids the
FastAPI+anyio quirk where a dependency touching the request before body parsing
can poison the body stream under pytest's loop (real uvicorn is unaffected).
"""

from __future__ import annotations

import os

os.environ.setdefault("MAX_REQUEST_SIZE_BYTES", "10")

import pytest
from ai_core.fastapi_middleware import (
    _auth_dependency,
    _rate_dependency,
    _size_dependency,
    add_observability_middleware,
)
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from starlette.requests import Request


def _build_app(**kwargs) -> FastAPI:
    app = FastAPI()
    add_observability_middleware(app, **kwargs)

    class Req(BaseModel):
        x: int = Field(gt=0)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(body: Req):
        return {"x": body.x}

    @app.get("/boom")
    def boom():
        raise ValueError("kaboom")

    return app


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _fake_request(path: str = "/predict", headers: dict | None = None, client_host: str = "1.2.3.4"):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "scheme": "http",
        "server": ("testserver", 80),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 1234),
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


# --- Integration tests (stable under the test runner) ----------------------


def test_validation_error_envelope():
    with _client(_build_app()) as c:
        r = c.post("/predict", json={"x": -1})
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body and "request_id" in body


def test_unhandled_exception_envelope():
    with _client(_build_app()) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal server error"
    assert "request_id" in body


def test_cors_echoed_on_cross_origin():
    with _client(_build_app()) as c:
        r = c.get("/health", headers={"Origin": "https://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_request_size_limit_rejected():
    with _client(_build_app(max_request_bytes=10)) as c:
        r = c.post(
            "/predict", content=b'{"x":1}', headers={"content-length": "99999999"}
        )
    assert r.status_code == 413
    assert "request_id" in r.json()


# --- Unit tests of dependency logic ----------------------------------------


@pytest.mark.anyio
async def test_auth_dependency_blocks_without_key_and_allows_with_key():
    dep = _auth_dependency("secret").dependency
    with pytest.raises(HTTPException) as exc:
        await dep(_fake_request(path="/predict"))
    assert exc.value.status_code == 401
    # correct key -> no raise
    await dep(_fake_request(path="/predict", headers={"X-API-Key": "secret"}))
    # public path -> no raise even without key
    await dep(_fake_request(path="/health"))


@pytest.mark.anyio
async def test_size_dependency_rejects_oversized():
    dep = _size_dependency(10).dependency
    with pytest.raises(HTTPException) as exc:
        await dep(_fake_request(headers={"content-length": "99999"}))
    assert exc.value.status_code == 413
    # within limit -> no raise
    await dep(_fake_request(headers={"content-length": "5"}))


@pytest.mark.anyio
async def test_rate_dependency_limits_per_ip():
    dep = _rate_dependency(2).dependency

    def req(client: str = "9.9.9.9"):
        return _fake_request(path="/predict", headers={"X-Forwarded-For": client})

    await dep(req("9.9.9.9"))  # allowed (1/2)
    await dep(req("9.9.9.9"))  # allowed (2/2)
    with pytest.raises(HTTPException) as exc:
        await dep(req("9.9.9.9"))  # blocked (limit reached)
    assert exc.value.status_code == 429
    # a different client (different header) is not limited
    await dep(req("8.8.8.8"))
