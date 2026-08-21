"""Serving API for Large Language Model (LLM)."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_large_language_model_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from large_language_model.data import MAX_SEQ_LEN, VOCAB_SIZE, generate_synthetic_data
from large_language_model.model import LargeLanguageModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("LLM_METRICS_PORT", "8013"))


class PredictRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=64)
    max_len: int = Field(default=10, ge=1, le=32)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=10, ge=1, le=100)


class PredictResponse(BaseModel):
    generated_tokens: list[int]
    next_token_probabilities: list[float]
    model_version: str
    training_mode: str


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: LargeLanguageModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_reference_data: np.ndarray | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("large_language_model", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_large_language_model_schema())

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="large-language-model",
        model_version=_model_version,
        model_type="classification",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="large-language-model", version=_model_version)

    yield
    logger.info("Shutting down large-language-model API")


def _load_model() -> tuple[LargeLanguageModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "large-language-model"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("llm_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return LargeLanguageModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "large-language-model" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("llm_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return LargeLanguageModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "llm_model.npz"
    if npz_path.exists():
        return LargeLanguageModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/llm_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "llm_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return LargeLanguageModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base = generate_synthetic_data(n_samples=50, vocab_size=VOCAB_SIZE, random_seed=42)
    model = LargeLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=256,
        max_seq_len=MAX_SEQ_LEN,
        learning_rate=0.001,
        n_iterations=30,
        random_seed=42,
    )
    model.fit(X_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base = generate_synthetic_data(n_samples=50, vocab_size=VOCAB_SIZE, random_seed=42)
    return X_base.reshape(-1, 1)


app = FastAPI(
    title="Large Language Model API",
    description="Transformer-based LLM with self-attention, multi-head attention, positional encoding, and autoregressive decoding",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "large_language_model-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "stats": "GET /stats",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
    }


@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name="large-language-model",
                model_version=_model_version,
                model_type="classification",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded", model="large-language-model", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.layers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info["vocab_size"],
        d_model=info["d_model"],
        n_heads=info["n_heads"],
        n_layers=info["n_layers"],
        d_ff=info["d_ff"],
        training_mode=info["training_mode"],
        n_epochs_run=info["n_epochs_run"],
        final_loss=info["final_loss"],
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Generate next tokens using LLM with temperature and top-k sampling."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(body.tokens).reshape(1, -1)
    validation = _validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        generated = _model.predict(X, max_len=body.max_len, temperature=body.temperature, top_k=body.top_k)
        next_probs = _model.predict_proba(X)[0]

        probs_list = [float(p) for p in next_probs.flatten()]
        top_probs = probs_list[:10] + [0.0] * (10 - min(len(probs_list), 10))

        response = PredictResponse(
            generated_tokens=generated.tolist(),
            next_token_probabilities=top_probs,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e
