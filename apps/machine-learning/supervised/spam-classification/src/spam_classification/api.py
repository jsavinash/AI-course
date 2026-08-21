"""Production serving API for spam email classification."""

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_spam_schema
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spam_classification.data import FEATURE_NAMES
from spam_classification.model import LogisticRegression

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("SPAM_METRICS_PORT", "8002")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    """Request with explicit feature values."""

    features: list[int] = Field(
        ..., min_length=5, max_length=5, description="[free, win, link, !!!, meeting]"
    )
    threshold: float | None = SPAM_THRESHOLD


class EmailRequest(BaseModel):
    """Request with raw email text (features are auto-extracted)."""

    text: str = Field(..., min_length=1, max_length=10000)
    threshold: float | None = SPAM_THRESHOLD


class PredictResponse(BaseModel):
    """Response with prediction and probability."""

    is_spam: bool
    spam_probability: float
    threshold: float
    features: list[int]
    feature_names: list[str]
    label: str
    model_version: str


class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: LogisticRegression | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[int]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("spam_classification", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_spam_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "binary" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="spam-classification", model_version=_model_version, model_type="classification"
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="spam-classification", version=_model_version)

    yield

    logger.info("Shutting down spam-classification API")


def _load_model() -> tuple[LogisticRegression, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            spam_models = [m for m in models if m.get("model_name") == "spam-classification"]
            if spam_models:
                spam_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = spam_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("spam_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return LogisticRegression.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "spam-classification" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("spam_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return LogisticRegression.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "spam_model.npz"
    if npz_path.exists():
        return LogisticRegression.load(str(npz_path)), "legacy"

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/spam_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "spam_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return LogisticRegression.load(str(p)), "1.0.0-bundled"

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(
        "No pre-existing model found on disk. Initializing baseline spam classification model."
    )
    from spam_classification.data import load_training_data

    X_base, y_base = load_training_data(None)
    model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_csvs = [
        MODEL_DIR / "spam-classification" / _model_version / "training_data.csv",
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

    from spam_classification.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base


def extract_features(text: str) -> list[int]:
    """Extract 5 binary features from raw email text."""
    text_lower = text.lower()
    return [
        1 if "free" in text_lower else 0,
        1 if re.search(r"\bwin\b", text_lower) else 0,
        1 if re.search(r"https?://|www\.", text_lower) else 0,
        1 if text.count("!") >= 3 else 0,
        1 if "meeting" in text_lower else 0,
    ]


# Create FastAPI app
app = FastAPI(
    title="Spam Email Detection API",
    description="Logistic Regression model for classifying emails as SPAM or NOT spam",
    version="1.0.0",
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "spam-classification-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "threshold": SPAM_THRESHOLD,
        "features": FEATURE_NAMES,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_email": "POST /predict/email",
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
    from fastapi import Response
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
                model_name="spam-classification",
                model_version=_model_version,
                model_type="classification",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="spam-classification", version=_model_version
        )
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


def _compute_prediction(features_list: list[int], threshold: float) -> PredictResponse:
    """Core prediction logic shared by all predict endpoints."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(features_list) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(features_list)}",
        )

    # Validate input
    validation = _validator.validate(np.array([features_list]))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        X = np.array(features_list, dtype=float).reshape(1, -1)
        prob = float(_model.predict_proba(X)[0])
        is_spam = prob >= threshold
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(features_list)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            is_spam=is_spam,
            spam_probability=round(prob, 4),
            threshold=threshold,
            features=[int(f) for f in features_list],
            feature_names=FEATURE_NAMES,
            label="SPAM" if is_spam else "NOT spam",
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict_features(body: PredictRequest):
    """Classify an email given explicit feature values."""
    return _compute_prediction(body.features, body.threshold)


@app.post("/predict/email", response_model=PredictResponse)
def predict_email(body: EmailRequest):
    """Classify an email given raw text. Features are auto-extracted."""
    features = extract_features(body.text)
    return _compute_prediction(features, body.threshold)
