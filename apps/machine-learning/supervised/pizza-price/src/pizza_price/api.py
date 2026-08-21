"""Production serving API for pizza price prediction."""

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
from ai_core.validation import DataValidator, create_pizza_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from pizza_price.model import LinearRegression

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("PIZZA_METRICS_PORT", "8001")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    """Single pizza price prediction request."""

    diameter: float = Field(..., gt=0, le=50, description="Pizza diameter in inches")


class PredictBulkRequest(BaseModel):
    """Bulk pizza price prediction request."""

    diameters: list[float] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    """Prediction response."""

    diameter: float
    predicted_price: float
    equation: str
    model_version: str


class BulkPredictResponse(BaseModel):
    """Bulk prediction response."""

    predictions: list[PredictResponse]
    model_version: str


class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: LinearRegression | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("pizza_price", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_pizza_schema())
    _drift_detector = DriftDetector(
        feature_names=["diameter"],
        feature_types={"diameter": "float"},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="pizza-price", model_version=_model_version, model_type="regression"
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="pizza-price", version=_model_version)

    yield

    logger.info("Shutting down pizza-price API")


def _load_model() -> tuple[LinearRegression, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            pizza_models = [m for m in models if m.get("model_name") == "pizza-price"]
            if pizza_models:
                pizza_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = pizza_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("pizza_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return LinearRegression.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "pizza-price" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("pizza_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return LinearRegression.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "pizza_model.npz"
    if npz_path.exists():
        return LinearRegression.load(str(npz_path)), "legacy"

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/pizza_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "pizza_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return LinearRegression.load(str(p)), "1.0.0-bundled"

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning("No pre-existing model found on disk. Initializing baseline linear model.")
    from pizza_price.data import load_training_data

    X_base, y_base = load_training_data(None)
    model = LinearRegression(learning_rate=0.001, n_iterations=2000)
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_csvs = [
        MODEL_DIR / "pizza-price" / _model_version / "training_data.csv",
        MODEL_DIR / "training_data.csv",
        Path("/app/artifacts/models/training_data.csv"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "training_data.csv",
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if "diameter" in df.columns:
                    return df[["diameter"]].values
            except Exception as e:
                logger.warning("Could not read reference csv", path=str(csv_path), error=str(e))

    from pizza_price.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base.reshape(-1, 1)


# Create FastAPI app
app = FastAPI(
    title="Pizza Price Prediction API",
    description="Linear Regression model for predicting pizza prices from diameter",
    version="1.0.0",
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "pizza-price-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_bulk": "POST /predict/bulk",
            "drift": "GET /drift",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health_check():
    """Kubernetes liveness/readiness probe."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    """Dynamically reload the model from disk/registry."""
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name="pizza-price", model_version=_model_version, model_type="regression"
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="pizza-price", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    """Check for data drift between reference and recent predictions."""
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")

    if len(_recent_predictions) < 10:
        return DriftResponse(
            total_features=1,
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:]).reshape(-1, 1)
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    return DriftResponse(**summary)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Predict pizza price for a single diameter."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    validation = _validator.validate(np.array([body.diameter]))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        price = _model.predict(np.array([body.diameter]))[0]
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(body.diameter)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            diameter=body.diameter,
            predicted_price=round(float(price), 2),
            equation=f"price = {_model.weight:.4f} * diameter + {_model.bias:.4f}",
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version)
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Predict pizza prices for multiple diameters."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    validation = _validator.validate(np.array(body.diameters))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        diameters = np.array(body.diameters)
        prices = _model.predict(diameters)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(body.diameters)
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]

        predictions = [
            PredictResponse(
                diameter=float(d),
                predicted_price=round(float(p), 2),
                equation=f"price = {_model.weight:.4f} * diameter + {_model.bias:.4f}",
                model_version=_model_version,
            )
            for d, p in zip(diameters, prices, strict=False)
        ]
        return BulkPredictResponse(predictions=predictions, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version)
        logger.exception("Bulk prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Bulk prediction failed") from e
