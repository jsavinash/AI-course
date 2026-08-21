"""Production serving API for credit card fraud detection via autoencoder reconstruction error."""

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

from anomaly_detection_fraud.data import FEATURE_NAMES
from anomaly_detection_fraud.model import FraudDetectionAutoencoder

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("FRAUD_DETECTION_METRICS_PORT", "8010")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class FraudRequest(BaseModel):
    """Single credit card transaction for fraud detection."""

    time_since_last_transaction: float = Field(
        ..., ge=0, description="Minutes since last transaction"
    )
    transaction_amount: float = Field(..., ge=0, description="Transaction amount in USD")
    merchant_category: float = Field(..., ge=0, le=11, description="Merchant category code (0-11)")
    merchant_risk_score: float = Field(..., ge=0, le=1, description="Merchant risk score (0-1)")
    cardholder_risk_score: float = Field(..., ge=0, le=1, description="Cardholder risk score (0-1)")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in miles")
    is_online: float = Field(..., ge=0, le=1, description="Whether transaction is online (0/1)")
    is_foreign: float = Field(..., ge=0, le=1, description="Whether transaction is foreign (0/1)")
    hour_of_day: float = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: float = Field(..., ge=0, le=6, description="Day of week (0-6)")
    account_age_days: float = Field(..., ge=0, description="Account age in days")
    recent_transaction_count: float = Field(..., ge=0, description="Recent transaction count")
    avg_transaction_amount_24h: float = Field(..., ge=0, description="Avg transaction amount (24h)")
    device_risk_score: float = Field(..., ge=0, le=1, description="Device risk score (0-1)")
    ip_risk_score: float = Field(..., ge=0, le=1, description="IP risk score (0-1)")


class FraudBulkRequest(BaseModel):
    """Bulk fraud detection request."""

    samples: list[FraudRequest] = Field(..., min_length=1, max_length=100)


class FraudResponse(BaseModel):
    """Fraud detection response for a single transaction."""

    is_fraud: bool
    fraud_probability: float
    reconstruction_error: float
    anomaly_threshold: float
    model_version: str
    training_mode: str


class BulkFraudResponse(BaseModel):
    """Bulk fraud detection response."""

    samples: list[FraudResponse]
    n_frauds: int
    n_samples: int
    model_version: str


class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    """Model statistics response."""

    n_features: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    threshold: float
    model_version: str


# Global model state
_model: FraudDetectionAutoencoder | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("credit_card_fraud_detection", port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "float" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="credit-card-fraud-detection",
        model_version=_model_version,
        model_type="anomaly_detection",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="credit-card-fraud-detection", version=_model_version)

    yield

    logger.info("Shutting down credit-card-fraud-detection API")


def _load_model() -> tuple[FraudDetectionAutoencoder, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            fraud_models = [
                m for m in models if m.get("model_name") == "credit-card-fraud-detection"
            ]
            if fraud_models:
                fraud_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = fraud_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("fraud_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return FraudDetectionAutoencoder.load(str(npz_files[0])), latest[
                        "model_version"
                    ]
        else:
            model_dir = MODEL_DIR / "credit-card-fraud-detection" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("fraud_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return FraudDetectionAutoencoder.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "fraud_detection_model.npz"
    if npz_path.exists():
        return FraudDetectionAutoencoder.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/fraud_detection_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "fraud_detection_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return FraudDetectionAutoencoder.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found on disk. Initializing baseline model.")
    from anomaly_detection_fraud.data import generate_normal_data

    X_base = generate_normal_data(n_samples=2000, random_seed=42)
    model = FraudDetectionAutoencoder(
        hidden_dim=8, learning_rate=0.001, n_iterations=500, random_seed=42
    )
    model.fit(X_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    candidate_csvs = [
        MODEL_DIR / "credit-card-fraud-detection" / _model_version / "training_data.csv",
        MODEL_DIR / "training_data.csv",
        Path("/app/artifacts/models/training_data.csv"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "training_data.csv",
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if all(f in df.columns for f in FEATURE_NAMES):
                    return df[FEATURE_NAMES].values
            except Exception as e:
                logger.warning("Could not read reference csv", path=str(csv_path), error=str(e))

    from anomaly_detection_fraud.data import generate_normal_data

    X_base = generate_normal_data(n_samples=500, random_seed=42)
    return X_base


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Feedforward autoencoder for detecting fraudulent credit card transactions",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "credit-card-fraud-detection-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "features": FEATURE_NAMES,
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
                model_name="credit-card-fraud-detection",
                model_version=_model_version,
                model_type="anomaly_detection",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically",
            model="credit-card-fraud-detection",
            version=_model_version,
        )
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")

    if len(_recent_predictions) < 10:
        return DriftResponse(
            total_features=len(FEATURE_NAMES),
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])

    return DriftResponse(**summary)


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or _model.W1 is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StatsResponse(
        n_features=_model.input_dim,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        threshold=_model.threshold,
        model_version=_model_version,
    )


def _extract_features(obs: FraudRequest) -> list[float]:
    return [
        obs.time_since_last_transaction,
        obs.transaction_amount,
        obs.merchant_category,
        obs.merchant_risk_score,
        obs.cardholder_risk_score,
        obs.distance_from_home,
        obs.is_online,
        obs.is_foreign,
        obs.hour_of_day,
        obs.day_of_week,
        obs.account_age_days,
        obs.recent_transaction_count,
        obs.avg_transaction_amount_24h,
        obs.device_risk_score,
        obs.ip_risk_score,
    ]


def _compute_fraud(obs: FraudRequest) -> FraudResponse:
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _extract_features(obs)
    X = np.array([features])

    start = time.time()
    try:
        recon_error = float(_model.reconstruction_error(X)[0])
        is_fraud = bool(_model.is_fraud(X)[0])
        proba = float(_model.predict_proba(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(features)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return FraudResponse(
            is_fraud=is_fraud,
            fraud_probability=round(proba, 4),
            reconstruction_error=round(recon_error, 4),
            anomaly_threshold=round(_model.threshold, 4),
            model_version=_model_version,
            training_mode=_model.training_mode if _model else "unknown",
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Fraud detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Fraud detection failed") from e


@app.post("/predict", response_model=FraudResponse)
def predict_fraud(body: FraudRequest):
    """Detect fraud for a single transaction."""
    return _compute_fraud(body)


@app.post("/predict/bulk", response_model=BulkFraudResponse)
def predict_fraud_bulk(body: FraudBulkRequest):
    """Detect fraud for multiple transactions."""
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(body.samples) < 1 or len(body.samples) > 100:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 100")

    X = np.array([_extract_features(s) for s in body.samples])

    start = time.time()
    try:
        recon_errors = _model.reconstruction_error(X)
        anomalies = _model.is_fraud(X)
        probas = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            FraudResponse(
                is_fraud=bool(anom),
                fraud_probability=round(float(proba), 4),
                reconstruction_error=round(float(recon_error), 4),
                anomaly_threshold=round(_model.threshold, 4),
                model_version=_model_version,
                training_mode=_model.training_mode if _model else "unknown",
            )
            for anom, recon_error, proba in zip(anomalies, recon_errors, probas, strict=False)
        ]

        return BulkFraudResponse(
            samples=results,
            n_frauds=int(np.sum(anomalies)),
            n_samples=len(results),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Bulk fraud detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Bulk fraud detection failed") from e
