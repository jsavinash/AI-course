"""Serving API for PINN Heat Equation Solver."""

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
from mlops_shared.validation import DataValidator, create_pinn_heat_equation_schema
from pydantic import BaseModel, Field

from pinn_heat_equation.data import N_FEATURES, generate_synthetic_data
from pinn_heat_equation.model import PINNHeatEquation

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("PINN_METRICS_PORT", "8030"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    t: float = Field(..., ge=0.0, le=0.5)


class PredictBulkRequest(BaseModel):
    requests: list[dict] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    temperature: float
    physics_residual: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    alpha: float
    hidden_dim: int
    n_layers: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: PINNHeatEquation | None = None
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
    _metrics = MetricsCollector("pinn_heat_equation", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_pinn_heat_equation_schema())
    feature_names = ["x", "t"]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={"x": "float", "t": "float"},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="pinn-heat-equation",
        model_version=_model_version,
        model_type="regression",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="pinn-heat-equation", version=_model_version)

    yield
    logger.info("Shutting down pinn-heat-equation API")


def _load_model() -> tuple[PINNHeatEquation, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "pinn-heat-equation"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("pinn_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return PINNHeatEquation.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "pinn-heat-equation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("pinn_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return PINNHeatEquation.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "pinn_model.npz"
    if npz_path.exists():
        return PINNHeatEquation.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/pinn_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "pinn_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return PINNHeatEquation.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, u_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = PINNHeatEquation(
        alpha=0.01,
        hidden_dim=16,
        n_layers=2,
        learning_rate=0.01,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base, u_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title="PINN Heat Equation Solver API",
    description="Solves supervised learning tasks while respecting physical laws described by differential equations",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "pinn_heat_equation-api",
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
                model_name="pinn-heat-equation",
                model_version=_model_version,
                model_type="regression",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded", model="pinn-heat-equation", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {"total_features": N_FEATURES, "drifted_features": 0, "drift_ratio": 0.0, "drifted": [], "all_results": []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.weights:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        alpha=_model.alpha,
        hidden_dim=_model.hidden_dim,
        n_layers=_model.n_layers,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Predict temperature u(x, t) using physics-informed network."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[body.x, body.t]])
    validation = _validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        u_pred = _model.predict(X)[0]
        residual = _model.predict_proba(X)[0]
        response = PredictResponse(
            temperature=round(float(u_pred), 6),
            physics_residual=round(float(residual), 6),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([body.x, body.t])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Make multiple predictions."""
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for req in body.requests:
        x = float(req.get("x", 0.5))
        t = float(req.get("t", 0.1))
        X = np.array([[x, t]])
        u_pred = _model.predict(X)[0]
        residual = _model.predict_proba(X)[0]
        predictions.append(PredictResponse(
            temperature=round(float(u_pred), 6),
            physics_residual=round(float(residual), 6),
            model_version=_model_version,
            training_mode=_model.training_mode,
        ))
        _recent_predictions.append([x, t])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
