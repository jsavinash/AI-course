"""Serving API for image captioning (RNN)."""

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
from ai_core.validation import DataValidator, create_image_captioning_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from vision_image_captioning.data import (
    CAPTION_LEN,
    N_PIXELS,
    VOCAB_SIZE,
    VOCAB_TOKENS,
    generate_synthetic_data,
)
from vision_image_captioning.model import ImageCaptioningRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("IMAGE_CAPTIONING_METRICS_PORT", "8019"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=N_PIXELS, max_length=N_PIXELS)


class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    caption_tokens: list[int]
    caption: str
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class StatsResponse(BaseModel):
    n_pixels: int
    vocab_size: int
    caption_len: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: ImageCaptioningRNN | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("image_captioning", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_image_captioning_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"pixel_{i}" for i in range(N_PIXELS)],
        feature_types={f"pixel_{i}": "float" for i in range(N_PIXELS)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="image-captioning",
        model_version=_model_version,
        model_type="rnn_image_captioning",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="image-captioning", version=_model_version)

    yield
    logger.info("Shutting down image-captioning API")


def _load_model() -> tuple[ImageCaptioningRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            ic_models = [m for m in models if m.get("model_name") == "image-captioning"]
            if ic_models:
                ic_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = ic_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("image_captioning_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return ImageCaptioningRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "image-captioning" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("image_captioning_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return ImageCaptioningRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "image_captioning_model.npz"
    if npz_path.exists():
        return ImageCaptioningRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/image_captioning_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "image_captioning_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return ImageCaptioningRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = ImageCaptioningRNN(
        n_pixels=N_PIXELS,
        vocab_size=VOCAB_SIZE,
        caption_len=CAPTION_LEN,
        hidden_dim=32,
        learning_rate=0.05,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title="Image Captioning API",
    description="RNN for generating descriptive captions from image pixel sequences",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "image-captioning-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_pixels": N_PIXELS,
        "vocab_size": VOCAB_SIZE,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict/bulk": "POST /predict/bulk",
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
                model_name="image-captioning",
                model_version=_model_version,
                model_type="rnn_image_captioning",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="image-captioning", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift")
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {
            "total_features": N_PIXELS,
            "drifted_features": 0,
            "drift_ratio": 0.0,
            "drifted": [],
            "all_results": [],
        }
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        n_pixels=N_PIXELS,
        vocab_size=VOCAB_SIZE,
        caption_len=CAPTION_LEN,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction(pixels: list[float]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([pixels])
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        captions = _model.predict(X)
        caption_tokens = captions[0].tolist()
        caption_str = " ".join(VOCAB_TOKENS[t % len(VOCAB_TOKENS)] for t in caption_tokens)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        _recent_predictions.append(pixels)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            caption_tokens=[int(t) for t in caption_tokens],
            caption=caption_str,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.pixels)


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for pixels in body.requests:
        predictions.append(_compute_prediction(pixels))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
