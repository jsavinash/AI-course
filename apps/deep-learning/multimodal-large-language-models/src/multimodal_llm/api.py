"""Serving API for Multimodal Language Modeling."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from mlops_shared.drift import DriftDetector
from mlops_shared.fastapi_middleware import add_observability_middleware
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.metrics import MetricsCollector
from mlops_shared.model_registry import ModelRegistry
from pydantic import BaseModel, Field

from multimodal_llm.data import VOCAB_SIZE, generate_synthetic_multimodal_data
from multimodal_llm.model import MultimodalLLM, softmax

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("MULTIMODAL_LLM_METRICS_PORT", "8012"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class MultimodalPredictRequest(BaseModel):
    text_tokens: list[int] = Field(..., min_length=1, max_length=64)
    image_patches: list[list[float]] | None = Field(default=None)
    mel_spectrogram: list[list[float]] | None = Field(default=None)
    max_len: int = Field(default=10, ge=1, le=32)


class MultimodalPredictResponse(BaseModel):
    generated_tokens: list[int]
    predicted_token: int
    confidence: float
    model_version: str
    training_mode: str
    modalities_used: list[str]


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    connector_dim: int
    fusion_type: str
    n_encoder_layers: int
    n_decoder_layers: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: MultimodalLLM | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("multimodal_llm", port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f"token_{i}" for i in range(VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="multimodal-llm",
        model_version=_model_version,
        model_type="multimodal",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="multimodal-llm", version=_model_version)

    yield
    logger.info("Shutting down multimodal-llm API")


def _load_model() -> tuple[MultimodalLLM, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            mm_models = [m for m in models if m.get("model_name") == "multimodal-llm"]
            if mm_models:
                mm_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = mm_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("multimodal_llm_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return MultimodalLLM.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "multimodal-llm" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("multimodal_llm_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return MultimodalLLM.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "multimodal_llm.npz"
    if npz_path.exists():
        return MultimodalLLM.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/multimodal_llm_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "multimodal_llm_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return MultimodalLLM.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_multimodal_data(n_samples=100, random_seed=42)
    model = MultimodalLLM(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        text_encoder_dim=128,
        image_encoder_dim=256,
        audio_encoder_dim=64,
        connector_dim=256,
        fusion_type="hybrid",
        max_seq_len=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=256,
        learning_rate=0.001,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base["text_tokens"], y_base, image_patches=X_base.get("image_patches"), mel_spectrogram=X_base.get("mel_spectrogram"))
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_multimodal_data(n_samples=100, random_seed=42)
    return X_base["text_tokens"].astype(float)


app = FastAPI(
    title="Multimodal LLM API",
    description="Multimodal Large Language Model that integrates text, image, and audio inputs using modality encoders, connectors, fusion mechanisms, and LLM backbone",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "multimodal_llm-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "stats": "GET /stats",
            "drift": "GET /drift",
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
                model_name="multimodal-llm",
                model_version=_model_version,
                model_type="multimodal",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded", model="multimodal-llm", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {"total_features": VOCAB_SIZE, "drifted_features": 0, "drift_ratio": 0.0, "drifted": [], "all_results": []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.text_encoder:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info["vocab_size"],
        d_model=info["d_model"],
        connector_dim=info["connector_dim"],
        fusion_type=info["fusion_type"],
        n_encoder_layers=info["n_encoder_layers"],
        n_decoder_layers=info["n_decoder_layers"],
        training_mode=info["training_mode"],
        n_epochs_run=info["n_epochs_run"],
        final_loss=info["final_loss"],
        model_version=_model_version,
    )


@app.post("/predict", response_model=MultimodalPredictResponse)
def predict(body: MultimodalPredictRequest):
    """Generate next-token prediction using multimodal LLM with text, image, and audio inputs."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text_tokens = np.array(body.text_tokens).reshape(1, -1)
    image_patches = np.array(body.image_patches).reshape(1, -1, 768) if body.image_patches else None
    mel_spectrogram = np.array(body.mel_spectrogram).reshape(1, -1, 80) if body.mel_spectrogram else None

    modalities_used = ["text"]
    if image_patches is not None:
        modalities_used.append("image")
    if mel_spectrogram is not None:
        modalities_used.append("audio")

    start = time.time()
    try:
        generated = _model.predict(text_tokens, image_patches=image_patches, mel_spectrogram=mel_spectrogram, max_len=body.max_len)
        predicted_token = int(generated[0]) if len(generated) > 0 else 0

        logits = _model.llm_backbone.forward(text_tokens)
        probs = softmax(logits.flatten())
        confidence = float(probs[predicted_token]) if predicted_token < len(probs) else 0.0

        response = MultimodalPredictResponse(
            generated_tokens=generated.tolist(),
            predicted_token=predicted_token,
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
            modalities_used=modalities_used,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(t) for t in body.text_tokens])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e
