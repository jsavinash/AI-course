# SERVING CONTAINER - Multi-stage build
##############################################################################
# Build for a specific app:
#   docker build -f docker/serve.Dockerfile --build-arg APP_MODULE=spam_classification .
##############################################################################

# Build stage
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy workspace files
COPY pyproject.toml uv.lock* ./
COPY packages/ packages/
COPY apps/ apps/

# Create virtual environment and install ALL workspace members with the
# frozen lockfile (see train.Dockerfile note on --all-packages).
RUN uv venv /app/.venv && \
    uv sync --python /app/.venv/bin/python --all-packages --no-dev

# Runtime stage
FROM python:3.11-slim AS runtime

# Python import module of the app to serve, e.g. spam_classification
ARG APP_MODULE=pizza_price
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/packages/ai-core/src:/app/apps" \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    APP_MODULE=${APP_MODULE}

WORKDIR /app

# Create non-root user first
RUN useradd -m -u 1000 appuser && mkdir -p /models && chown -R appuser:appuser /app /models

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code. Model artifacts are NOT baked into the image — they are
# pulled from the model registry / mounted volumes at runtime.
COPY packages/ packages/
COPY apps/ apps/
COPY pyproject.toml .

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command (overridden by K8s Deployment)
CMD ["sh", "-c", "uvicorn ${APP_MODULE}.api:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips *"]
