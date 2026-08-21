"""Serving API for Attention Mechanism."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_attention_mechanism_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from attention_mechanism.data import INPUT_DIM, SEQ_LEN, generate_synthetic_data
from attention_mechanism.model import AttentionModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("ATTENTION_METRICS_PORT", "8012"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    input_sequence: list[Annotated[list[float], Field(min_length=SEQ_LEN * INPUT_DIM, max_length=SEQ_LEN * INPUT_DIM)]] = Field(..., min_length=1, max_length=1)
    attention_type: str | None = None


class PredictResponse(BaseModel):
    output: list[float]
    attention_weights: list[float] | None = None
    model_version: str
    training_mode: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    input_dim: int
    hidden_dim: int
    output_dim: int
    attention_type: str
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: AttentionModel | None = None
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
    _metrics = MetricsCollector("attention_mechanism", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_attention_mechanism_schema())
    feature_names = [f"feature_{i}" for i in range(SEQ_LEN * INPUT_DIM)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="attention-mechanism",
        model_version=_model_version,
        model_type="regression",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="attention-mechanism", version=_model_version)

    yield
    logger.info("Shutting down attention-mechanism API")


def _load_model() -> tuple[AttentionModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "attention-mechanism"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("attention_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return AttentionModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "attention-mechanism" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("attention_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return AttentionModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "attention_model.npz"
    if npz_path.exists():
        return AttentionModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/attention_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "attention_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return AttentionModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = AttentionModel(
        input_dim=INPUT_DIM,
        hidden_dim=64,
        output_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        attention_type="multi_head",
        learning_rate=0.01,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base.reshape(-1, 1)


app = FastAPI(
    title="Attention Mechanism API",
    description="Focuses on relevant input parts by computing attention weights over sequences",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "attention_mechanism-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "attention_type": _model.attention_type if _model else "unknown",
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
                model_name="attention-mechanism",
                model_version=_model_version,
                model_type="regression",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded", model="attention-mechanism", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {"total_features": SEQ_LEN * INPUT_DIM, "drifted_features": 0, "drift_ratio": 0.0, "drifted": [], "all_results": []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or _model.encoder_rnn is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        input_dim=info["input_dim"],
        hidden_dim=info["hidden_dim"],
        output_dim=info["output_dim"],
        attention_type=info["attention_type"],
        training_mode=info["training_mode"],
        n_epochs_run=info["n_epochs_run"],
        final_loss=info["final_loss"],
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Generate prediction with attention weights visualization."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(body.input_sequence).reshape(1, -1)
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        X_reshaped = X.reshape(1, SEQ_LEN, INPUT_DIM) if X.size >= SEQ_LEN * INPUT_DIM else X.reshape(1, X.shape[1] // INPUT_DIM, INPUT_DIM)

        if body.attention_type and body.attention_type in ("soft_additive", "self", "multi_head", "hard") and body.attention_type != _model.attention_type:
            _model.attention_type = body.attention_type

        output, attn_weights = _model.predict(X_reshaped)

        attn_list = []
        if attn_weights is not None:
            attn_flat = np.array(attn_weights).flatten()
            attn_list = attn_flat.tolist()

        response = PredictResponse(
            output=output.flatten().tolist(),
            attention_weights=attn_list if attn_list else None,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(X.flatten().tolist())
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e
