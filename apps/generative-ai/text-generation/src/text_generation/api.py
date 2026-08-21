"""Serving API for Text Generation."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from text_gen.data import DEFAULT_VOCAB_SIZE
from text_gen.model import TextGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("TEXT_GENERATION_METRICS_PORT", "9024"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    model_version: str


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    reference_text: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    score: float
    model_version: str


class StatsResponse(BaseModel):
    model_id: str
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int
    temperature: float
    top_k: int
    top_p: float
    model_version: str


_model: TextGenerationModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("text_gen_generative", port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f"token_{i}" for i in range(DEFAULT_VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="text-generation",
        model_version=_model_version,
        model_type="generative",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="text-generation", version=_model_version)

    yield
    logger.info("Shutting down text-generation API")


def _load_model() -> tuple[TextGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            tg_models = [m for m in models if m.get("model_name") == "text-generation"]
            if tg_models:
                tg_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = tg_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("text_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return TextGenerationModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "text-generation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("text_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return TextGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "text_generation.npz"
    if npz_path.exists():
        return TextGenerationModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/text_generation_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "text_generation_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return TextGenerationModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    model = TextGenerationModel(model_id="baseline", vocab_size=DEFAULT_VOCAB_SIZE)
    model._init()
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    from nlp_text_generation.data import generate_synthetic_text
    X_base, _ = generate_synthetic_text(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title="Text Generation API",
    description="Transformer-based autoregressive text generation with temperature, top-k, and top-p sampling",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "text-generation-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "endpoints": {
            "health": "/health",
            "generate": "POST /generate",
            "evaluate": "POST /evaluate",
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
        "model_id": _model.model_id if _model else "unknown",
    }


@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
def generate_text(body: GenerateRequest):
    """Generate text from a prompt using the transformer model."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        _model.temperature = body.temperature
        _model.top_k = body.top_k
        _model.top_p = body.top_p
        generated_text = _model.generate(body.prompt, max_new_tokens=body.max_new_tokens)

        response = GenerateResponse(
            generated_text=generated_text,
            prompt=body.prompt,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(len(body.prompt.split()))])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="generation")
        logger.exception("Text generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Text generation failed") from e


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_text(body: EvaluateRequest):
    """Evaluate generated text against a reference."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        generated = _model.generate(body.prompt, max_new_tokens=50)
        gen_words = set(generated.lower().split())
        ref_words = set(body.reference_text.lower().split())
        score = len(gen_words.intersection(ref_words)) / max(len(ref_words), 1)

        response = EvaluateResponse(
            score=round(score, 4),
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="evaluation")
        logger.exception("Text evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Text evaluation failed") from e


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        model_id=info.get("model_id", "unknown"),
        vocab_size=info.get("vocab_size", DEFAULT_VOCAB_SIZE),
        d_model=info.get("d_model", 256),
        n_layers=info.get("n_layers", 2),
        max_seq_len=info.get("max_seq_len", 128),
        temperature=info.get("temperature", 0.8),
        top_k=info.get("top_k", 50),
        top_p=info.get("top_p", 0.9),
        model_version=_model_version,
    )
