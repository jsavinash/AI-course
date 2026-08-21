"""FastAPI middleware for production-grade MLOps serving.

Cross-cutting concerns are implemented with the most robust mechanism for each:

* :class:`ProductionMiddleware` is a **raw ASGI middleware** (not
  ``BaseHTTPMiddleware``) providing correlation-id logging, metrics and security
  headers. Raw ASGI avoids the well-known ``BaseHTTPMiddleware`` request-body
  pitfall (it never buffers/receives the body).
* auth, request-size limiting and rate limiting are **global FastAPI
  dependencies** (``app.router.dependencies``), which run *after* the body has
  been parsed, so they cannot corrupt it and they apply to every route.

Every app that calls :func:`add_observability_middleware` inherits all of this.
Auth and rate limiting are opt-in via env (``API_KEY``,
``RATE_LIMIT_PER_MINUTE``) and are disabled by default.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_core.logging import (
    clear_request_context,
    generate_request_id,
    get_logger,
    set_request_context,
)
from ai_core.metrics import MetricsCollector

logger = get_logger("mlops.middleware")

DEFAULT_MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(8 * 1024 * 1024)))
PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/"}


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or generate_request_id()


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:]
    return None


def _constant_time_eq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b, strict=False):
        result |= ord(x) ^ ord(y)
    return result == 0


class ProductionMiddleware:
    """Raw ASGI middleware: logging, metrics, security headers.

    Implemented as plain ASGI (no ``BaseHTTPMiddleware``) so it never receives
    or buffers the request body -- eliminating the Starlette body-corruption bug.
    """

    def __init__(self, app, metrics: MetricsCollector | None = None, enable_security_headers: bool = True):
        self.app = app
        self.metrics = metrics
        self.enable_security_headers = enable_security_headers

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = scope.get("headers", [])
        rid = _decode(raw_headers, b"x-request-id") or generate_request_id()
        tid = _decode(raw_headers, b"x-trace-id") or rid
        set_request_context(request_id=rid, trace_id=tid)

        method = scope.get("method", "")
        path = scope.get("path", "")
        metrics = self.metrics
        start = time.time()
        responded = {}
        if metrics:
            metrics.inc_active_requests()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers", []))
                hdrs.append((b"x-request-id", rid.encode()))
                if self.enable_security_headers:
                    hdrs.append((b"x-content-type-options", b"nosniff"))
                    hdrs.append((b"x-frame-options", b"deny"))
                    hdrs.append((b"x-xss-protection", b"1; mode=block"))
                    hdrs.append((b"cache-control", b"no-store"))
                    hdrs.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message["headers"] = hdrs
                responded["status"] = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            duration = time.time() - start
            status = responded.get("status", 500)
            if metrics:
                metrics.record_request(
                    method=method, endpoint=path, status=status, duration=duration
                )
            logger.info(
                "request_completed",
                method=method,
                path=path,
                status=status,
                duration_ms=round(duration * 1000, 2),
            )
        except Exception as e:
            duration = time.time() - start
            if metrics:
                metrics.record_request(
                    method=method, endpoint=path, status=500, duration=duration
                )
            logger.exception("request_failed", method=method, path=path, error=str(e))
            if "status" not in responded:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"x-request-id", rid.encode()),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps(
                            {"detail": "Internal server error", "request_id": rid}
                        ).encode(),
                    }
                )
        finally:
            if metrics:
                metrics.dec_active_requests()
            clear_request_context()


def _decode(headers, key: bytes) -> str | None:
    for k, v in headers:
        if k == key:
            return v.decode()
    return None


# --- Global dependencies (run after body parsing) -------------------------


def _size_dependency(max_bytes: int):
    async def dep(request: Request):
        if not max_bytes:
            return
        raw = request.headers.get("content-length")
        if raw and raw.isdigit() and int(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: max {max_bytes} bytes",
            )

    return Depends(dep)


def _auth_dependency(api_key: str | None):
    async def dep(request: Request):
        if not api_key or request.url.path in PUBLIC_PATHS:
            return
        provided = request.headers.get("X-API-Key") or _bearer(request)
        if not provided or not _constant_time_eq(provided, api_key):
            await request.body()  # drain so keep-alive connections stay valid
            raise HTTPException(status_code=401, detail="Unauthorized")

    return Depends(dep)


def _rate_dependency(limit: int):
    hits: dict[str, deque] = defaultdict(deque)

    async def dep(request: Request):
        if limit <= 0 or request.url.path in PUBLIC_PATHS:
            return
        client = (
            request.headers.get("X-Forwarded-For")
            or request.headers.get("X-Real-IP")
            or "unknown"
        )
        now = time.monotonic()
        window = hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            await request.body()  # drain so keep-alive connections stay valid
            raise HTTPException(status_code=429, detail="Too many requests")
        window.append(now)

    return Depends(dep)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "request_id": _request_id(request)},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": _request_id(request)},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": _request_id(request)},
        )


def add_observability_middleware(
    app: FastAPI,
    metrics: MetricsCollector | None = None,
    enable_security_headers: bool = True,
    enable_cors: bool = True,
    enable_request_size_limit: bool = True,
    max_request_bytes: int | None = None,
    api_key: str | None = None,
    rate_limit_per_minute: int = 0,
) -> None:
    """Add production-grade middleware/handlers to a FastAPI app.

    Auth and rate limiting are opt-in: ``api_key`` (or ``API_KEY`` env) and
    ``rate_limit_per_minute`` (or ``RATE_LIMIT_PER_MINUTE`` env). Disabled by
    default so existing behavior is preserved until explicitly configured.
    """
    resolved_max = (
        max_request_bytes
        if max_request_bytes is not None
        else (DEFAULT_MAX_REQUEST_BYTES if enable_request_size_limit else 0)
    )
    resolved_key = api_key if api_key is not None else os.getenv("API_KEY")
    resolved_rate = (
        rate_limit_per_minute
        if rate_limit_per_minute
        else int(os.getenv("RATE_LIMIT_PER_MINUTE", "0") or "0")
    )

    # Global dependencies (auth / size / rate) -- run after body parsing.
    app.router.dependencies.append(_size_dependency(resolved_max))
    app.router.dependencies.append(_auth_dependency(resolved_key))
    app.router.dependencies.append(_rate_dependency(resolved_rate))

    # Raw ASGI middleware for logging / metrics / security headers.
    app.add_middleware(
        ProductionMiddleware,
        metrics=metrics,
        enable_security_headers=enable_security_headers,
    )

    if enable_cors:
        origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials="*" not in origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _register_exception_handlers(app)
