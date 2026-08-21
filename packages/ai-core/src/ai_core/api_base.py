"""Base API / observability wiring for FastAPI services.

Re-exported from :mod:`ai_core.fastapi_middleware` so the package exposes a
stable ``ai_core.api_base`` namespace (target layout) without duplicating code.
"""

from ai_core.fastapi_middleware import add_observability_middleware

__all__ = ["add_observability_middleware"]
