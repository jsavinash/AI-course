"""Serving API for stock market prediction (RNN)."""

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
from mlops_shared.validation import DataValidator, create_stock_prediction_schema
from pydantic import BaseModel, Field

from stock_market_prediction.data import N_FEATURES, SEQ_LEN, generate_synthetic_data
from stock_market_prediction.model import StockMarketRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("STOCK_PREDICTION_METRICS_PORT", "8017"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    feature_sequences: list[list[float]] = Field(..., min_length=1, max_length=SEQ_LEN)


class PredictBulkRequest(BaseModel):
    requests: list[list[list[float]]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class StatsResponse(BaseModel):
    n_features: int
    seq_len: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: StockMarketRNN | None = None
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
    _metrics = MetricsCollector("stock_market_prediction", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_stock_prediction_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"step_{i}" for i in range(N_FEATURES)],
        feature_types={f"step_{i}": "float" for i in range(N_FEATURES)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="stock-market-prediction",
        model_version=_model_version,
        model_type="rnn_sequence_regression",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="stock-market-prediction", version=_model_version)

    yield
    logger.info("Shutting down stock-market-prediction API")


def _load_model() -> tuple[StockMarketRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            sm_models = [m for m in models if m.get("model_name") == "stock-market-prediction"]
            if sm_models:
                sm_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = sm_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("stock_market_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return StockMarketRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "stock-market-prediction" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("stock_market_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return StockMarketRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "stock_market_model.npz"
    if npz_path.exists():
        return StockMarketRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/stock_market_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "stock_market_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return StockMarketRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = StockMarketRNN(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        learning_rate=0.01,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    # Flatten first timestep features for drift detection
    return X_base[:, 0, :].reshape(-1, N_FEATURES)


app = FastAPI(
    title="Stock Market Prediction API",
    description="RNN for stock price prediction from temporal feature sequences",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "stock-market-prediction-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
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
                model_name="stock-market-prediction",
                model_version=_model_version,
                model_type="rnn_sequence_regression",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="stock-market-prediction", version=_model_version
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
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction(feature_sequences: list[list[float]]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([feature_sequences])

    # Validate
    X_flat = X.reshape(-1, N_FEATURES)
    validation = _validator.validate(X_flat)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        price = float(_model.predict(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        flat = [v for frame in feature_sequences for v in frame]
        _recent_predictions.append(flat)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            predicted_price=round(price, 2),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.feature_sequences)


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for feature_seq in body.requests:
        predictions.append(_compute_prediction(feature_seq))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
