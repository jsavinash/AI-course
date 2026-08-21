"""Production serving API for PCA-based anomaly detection."""

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
from ai_core.validation import DataValidator, create_anomaly_detection_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from anomaly_detection.data import FEATURE_NAMES
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("ANOMALY_METRICS_PORT", "8005")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class MetricsRequest(BaseModel):
    """Single metrics observation for anomaly detection."""

    request_count: float = Field(..., ge=0, description="Number of requests")
    bytes_per_request: float = Field(..., ge=0, description="Average bytes per request")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_io: float = Field(..., ge=0, description="Disk I/O operations per second")
    network_in: float = Field(..., ge=0, description="Network inbound MB/s")
    network_out: float = Field(..., ge=0, description="Network outbound MB/s")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage")
    connection_count: float = Field(..., ge=0, description="Active connections")
    response_time: float = Field(..., ge=0, description="Average response time in ms")


class MetricsBulkRequest(BaseModel):
    """Bulk metrics request for anomaly detection."""

    samples: list[MetricsRequest] = Field(..., min_length=1, max_length=100)


class AnomalyResponse(BaseModel):
    """Anomaly detection response for a single observation."""

    is_anomaly: bool
    anomaly_score: float
    anomaly_probability: float
    reconstruction_error: float
    anomaly_threshold: float
    model_version: str


class BulkAnomalyResponse(BaseModel):
    """Bulk anomaly detection response."""

    samples: list[AnomalyResponse]
    n_anomalies: int
    n_samples: int
    model_version: str


class StatsResponse(BaseModel):
    """Model statistics response."""

    n_features: int
    n_components: int
    explained_variance_ratio: float
    reconstruction_threshold: float
    threshold_method: str
    mean_reconstruction_error: float
    max_reconstruction_error: float
    model_version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    n_components: int
    n_features: int
    feature_names: list[str]
    cumulative_variance_ratio: float
    reconstruction_threshold: float
    model_version: str


class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: PCAAnomalyDetector | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("anomaly_detection", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_anomaly_detection_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "float" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="anomaly-detection", model_version=_model_version, model_type="anomaly_detection"
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="anomaly-detection", version=_model_version)

    yield

    logger.info("Shutting down anomaly-detection API")


def _load_model() -> tuple[PCAAnomalyDetector, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            ad_models = [m for m in models if m.get("model_name") == "anomaly-detection"]
            if ad_models:
                ad_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = ad_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("anomaly_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "anomaly-detection" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("anomaly_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "anomaly_detection_model.npz"
    if npz_path.exists():
        return PCAAnomalyDetector.load(str(npz_path)), "legacy"

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/anomaly_detection_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "anomaly_detection_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return PCAAnomalyDetector.load(str(p)), "1.0.0-bundled"

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning("No pre-existing model found on disk. Initializing baseline PCA model.")
    from anomaly_detection.data import load_training_data

    X_base, y_base = load_training_data(None)
    X_normal = X_base[y_base == 0]
    model = PCAAnomalyDetector(n_components=0.95, threshold_method="percentile", random_seed=42)
    model.fit(X_normal)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_csvs = [
        MODEL_DIR / "anomaly-detection" / _model_version / "training_data.csv",
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

    from anomaly_detection.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base


# Create FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="PCA-based anomaly detection using dimensionality reduction",
    version="1.0.0",
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "anomaly-detection-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "features": FEATURE_NAMES,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_bulk": "POST /predict/bulk",
            "stats": "GET /stats",
            "model_info": "GET /model/info",
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
                model_name="anomaly-detection",
                model_version=_model_version,
                model_type="anomaly_detection",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="anomaly-detection", version=_model_version)
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


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return model statistics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    evr = _model.explained_variance_ratio

    return StatsResponse(
        n_features=_model.n_features,
        n_components=_model.n_components_selected,
        explained_variance_ratio=float(
            np.sum(evr[: _model.n_components_selected]) if evr is not None else 0.0
        ),
        reconstruction_threshold=round(_model.threshold, 4),
        threshold_method=_model.threshold_method,
        mean_reconstruction_error=float(
            np.mean(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        max_reconstruction_error=float(
            np.max(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        model_version=_model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """Return detailed model information."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        n_components=_model.n_components_selected,
        n_features=_model.n_features,
        feature_names=FEATURE_NAMES,
        cumulative_variance_ratio=_model.cumulative_variance_ratio,
        reconstruction_threshold=round(_model.threshold, 4),
        model_version=_model_version,
    )


def _compute_anomaly(observation: MetricsRequest) -> AnomalyResponse:
    """Core anomaly detection logic shared by all detection endpoints."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    X = np.array(
        [
            [
                observation.request_count,
                observation.bytes_per_request,
                observation.cpu_usage,
                observation.memory_usage,
                observation.disk_io,
                observation.network_in,
                observation.network_out,
                observation.error_rate,
                observation.connection_count,
                observation.response_time,
            ]
        ]
    )
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        recon_error = float(_model.reconstruction_error(X)[0])
        is_anom = bool(_model.is_anomaly(X)[0])
        proba = float(_model.predict_proba(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(
            [
                observation.request_count,
                observation.bytes_per_request,
                observation.cpu_usage,
                observation.memory_usage,
                observation.disk_io,
                observation.network_in,
                observation.network_out,
                observation.error_rate,
                observation.connection_count,
                observation.response_time,
            ]
        )
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return AnomalyResponse(
            is_anomaly=is_anom,
            anomaly_score=round(recon_error, 4),
            anomaly_probability=round(proba, 4),
            reconstruction_error=round(recon_error, 4),
            anomaly_threshold=round(_model.threshold, 4),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Anomaly detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Anomaly detection failed") from e


@app.post("/predict", response_model=AnomalyResponse)
def predict_anomaly(body: MetricsRequest):
    """Detect anomaly for a single metrics observation."""
    return _compute_anomaly(body)


@app.post("/predict/bulk", response_model=BulkAnomalyResponse)
def predict_anomaly_bulk(body: MetricsBulkRequest):
    """Detect anomalies for multiple metrics observations."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(
        [
            [
                obs.request_count,
                obs.bytes_per_request,
                obs.cpu_usage,
                obs.memory_usage,
                obs.disk_io,
                obs.network_in,
                obs.network_out,
                obs.error_rate,
                obs.connection_count,
                obs.response_time,
            ]
            for obs in body.samples
        ]
    )

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        recon_errors = _model.reconstruction_error(X)
        anomalies = _model.is_anomaly(X)
        probas = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            AnomalyResponse(
                is_anomaly=bool(anom),
                anomaly_score=round(float(recon_error), 4),
                anomaly_probability=round(float(proba), 4),
                reconstruction_error=round(float(recon_error), 4),
                anomaly_threshold=round(_model.threshold, 4),
                model_version=_model_version,
            )
            for anom, proba, recon_error in zip(anomalies, probas, recon_errors, strict=False)
        ]

        n_anomalies = int(np.sum(anomalies))
        return BulkAnomalyResponse(
            samples=results,
            n_anomalies=n_anomalies,
            n_samples=len(results),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Bulk anomaly detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Bulk anomaly detection failed") from e
