"""Serving API for Transfer Learning."""

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

from transfer_learning.data import VOCAB_SIZE
from transfer_learning.model import TransferLearningModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("TRANSFER_LEARNING_METRICS_PORT", "8013"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=32)
    max_len: int = Field(default=1, ge=1, le=1)


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_probabilities: list[float]
    model_version: str
    base_model_frozen: bool
    fine_tune_layers: int


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_base_layers: int
    d_ff: int
    n_classes: int
    freeze_base: bool
    fine_tune_layers: int
    learning_rate: float
    fine_tune_lr: float
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: TransferLearningModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("transfer_learning", port=METRICS_PORT)
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
        model_name="transfer-learning",
        model_version=_model_version,
        model_type="classification",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="transfer-learning", version=_model_version)

    yield
    logger.info("Shutting down transfer-learning API")


def _load_model() -> tuple[TransferLearningModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            tl_models = [m for m in models if m.get("model_name") == "transfer-learning"]
            if tl_models:
                tl_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = tl_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("transfer_learning_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return TransferLearningModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "transfer-learning" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("transfer_learning_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return TransferLearningModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "transfer_learning.npz"
    if npz_path.exists():
        return TransferLearningModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/transfer_learning_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "transfer_learning_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return TransferLearningModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    from transfer_learning.data import generate_synthetic_data
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = TransferLearningModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_base_layers=1,
        d_ff=256,
        max_seq_len=32,
        n_classes=10,
        freeze_base=True,
        fine_tune_layers=0,
        learning_rate=0.001,
        fine_tune_lr=0.0001,
        n_iterations=10,
        random_seed=42,
    )
    model.fit(X_base, y_base, n_iterations=10)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    from transfer_learning.data import generate_synthetic_data
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title="Transfer Learning API",
    description="Transfer Learning model with frozen base model and trainable classification head, supporting fine-tuning of top layers",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "transfer_learning-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "base_model_frozen": _model.base_model.frozen if _model and _model.base_model else True,
        "fine_tune_layers": _model.fine_tune_layers if _model else 0,
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
        "base_model_frozen": _model.base_model.frozen if _model and _model.base_model else True,
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
                model_name="transfer-learning",
                model_version=_model_version,
                model_type="classification",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded", model="transfer-learning", version=_model_version)
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
    if _model is None or _model.base_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info["vocab_size"],
        d_model=info["d_model"],
        n_base_layers=info["n_base_layers"],
        d_ff=info["d_ff"],
        n_classes=info["n_classes"],
        freeze_base=bool(info["freeze_base"]),
        fine_tune_layers=info["fine_tune_layers"],
        learning_rate=info["learning_rate"],
        fine_tune_lr=info["fine_tune_lr"],
        n_epochs_run=info["n_epochs_run"],
        final_loss=info["final_loss"],
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Predict class using transfer learning model with frozen base and trainable head."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(body.tokens).reshape(1, -1)

    start = time.time()
    try:
        probs = _model.predict_proba(X)
        predicted_class = int(np.argmax(probs[0]))
        confidence = float(probs[0][predicted_class])

        response = PredictResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            class_probabilities=probs[0].tolist(),
            model_version=_model_version,
            base_model_frozen=_model.base_model.frozen if _model.base_model else True,
            fine_tune_layers=_model.fine_tune_layers,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(t) for t in body.tokens])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e
