"""Serving API for speech recognition (RNN)."""

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
from ai_core.validation import DataValidator, create_speech_recognition_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from speech_audio_recognition.data import (
    N_CLASSES,
    N_FEATURES,
    SEQ_LEN,
    WORD_NAMES,
    generate_synthetic_data,
)
from speech_audio_recognition.model import SpeechRecognitionRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("SPEECH_RECOGNITION_METRICS_PORT", "8015"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    audio_features: list[list[float]] = Field(..., min_length=1, max_length=SEQ_LEN)


class PredictBulkRequest(BaseModel):
    requests: list[list[list[float]]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    word: str
    word_index: int
    confidence: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class StatsResponse(BaseModel):
    n_features: int
    seq_len: int
    n_classes: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: SpeechRecognitionRNN | None = None
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
    _metrics = MetricsCollector("speech_recognition", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_speech_recognition_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"frame_{i}" for i in range(N_FEATURES)],
        feature_types={f"frame_{i}": "float" for i in range(N_FEATURES)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="speech-recognition",
        model_version=_model_version,
        model_type="rnn_sequence_classification",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="speech-recognition", version=_model_version)

    yield
    logger.info("Shutting down speech-recognition API")


def _load_model() -> tuple[SpeechRecognitionRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            sr_models = [m for m in models if m.get("model_name") == "speech-recognition"]
            if sr_models:
                sr_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = sr_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("speech_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return SpeechRecognitionRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "speech-recognition" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("speech_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return SpeechRecognitionRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "speech_recognition_model.npz"
    if npz_path.exists():
        return SpeechRecognitionRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/speech_recognition_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "speech_recognition_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return SpeechRecognitionRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = SpeechRecognitionRNN(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        n_classes=N_CLASSES,
        hidden_dim=32,
        learning_rate=0.05,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    # Flatten for drift detection (first timestep features)
    return X_base[:, 0, :].reshape(-1, 1) if X_base.ndim == 3 else X_base


app = FastAPI(
    title="Speech Recognition API",
    description="RNN for speech-to-text feature sequence classification",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "speech-recognition-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
        "n_classes": N_CLASSES,
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
                model_name="speech-recognition",
                model_version=_model_version,
                model_type="rnn_sequence_classification",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="speech-recognition", version=_model_version
        )
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
            "total_features": N_FEATURES,
            "drifted_features": 0,
            "drift_ratio": 0.0,
            "drifted": [],
            "all_results": [],
        }
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data[:, :N_FEATURES], current[:, :N_FEATURES])
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        n_classes=N_CLASSES,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction(audio_features: list[list[float]]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([audio_features])

    # Validate each frame
    for frame in audio_features:
        X_flat = np.array([frame])
        validation = _validator.validate(X_flat)
        if not validation.valid:
            raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        word_idx = int(_model.predict(X)[0])
        probas = _model.predict_proba(X)[0]
        confidence = float(np.max(probas))
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        flat = [v for frame in audio_features for v in frame]
        _recent_predictions.append(flat)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            word=WORD_NAMES[word_idx] if word_idx < len(WORD_NAMES) else f"word_{word_idx}",
            word_index=word_idx,
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.audio_features)


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for audio_features in body.requests:
        predictions.append(_compute_prediction(audio_features))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
