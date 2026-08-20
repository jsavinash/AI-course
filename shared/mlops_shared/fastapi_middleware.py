"""FastAPI middleware for production-grade MLOps serving."""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mlops_shared.logging import (
    clear_request_context,
    generate_request_id,
    get_logger,
    set_request_context,
)
from mlops_shared.metrics import MetricsCollector

logger = get_logger("mlops.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with correlation IDs and measure latency."""

    def __init__(self, app, metrics: MetricsCollector | None = None):
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        trace_id = request.headers.get("X-Trace-ID") or request_id

        set_request_context(request_id=request_id, trace_id=trace_id)

        metrics = self.metrics or getattr(request.app.state, "metrics", None)

        start = time.time()
        if metrics:
            metrics.inc_active_requests()

        try:
            response = await call_next(request)
            duration = time.time() - start
            status = response.status_code

            if metrics:
                metrics.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=status,
                    duration=duration,
                )

            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round(duration * 1000, 2),
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            duration = time.time() - start
            if metrics:
                metrics.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=500,
                    duration=duration,
                )
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})
        finally:
            if metrics:
                metrics.dec_active_requests()
            clear_request_context()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def add_observability_middleware(
    app: FastAPI,
    metrics: MetricsCollector | None = None,
    enable_security_headers: bool = True,
) -> None:
    """Add observability and security middleware to a FastAPI app."""
    app.add_middleware(RequestLoggingMiddleware, metrics=metrics)
    if enable_security_headers:
        app.add_middleware(SecurityHeadersMiddleware)
