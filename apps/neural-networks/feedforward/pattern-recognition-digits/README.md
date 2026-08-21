# pattern-recognition-digits

## ∫ Mathematics & Theory

Digit Recognition / Classification — Underlying equations and derivations

$$Z = WX + b$$

$$A = \text{ReLU}(Z)$$

$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

$$\hat{y} = \text{softmax}(Z_{out})$$

### Step-by-Step Derivation

Feedforward networks learn hierarchical feature representations. Each layer computes a linear transformation followed by a non-linearity. Cross-entropy loss penalizes misclassification. Backpropagation computes gradients via the chain rule.

### Interactive Visualization

Interactive decision boundary; feature visualization for hidden layers; confusion matrix explorer.

## ⚙ Architecture

Model structure, data flow, and layer breakdown

### Class Hierarchy

```
  DigitRecognitionNN
```

### Data Flow

```mermaid
graph TD
  A[Input Data] --> B[Preprocessing]
  B --> C[Model Training]
  C --> D[Evaluation]
  D --> E[Model Registry]
  E --> F[Serving API]
```

## ⚡ API Reference

FastAPI endpoints and model interfaces

| Method | Endpoint |
| --- | --- |
| `GET` | `/` |
| `GET` | `/health` |
| `GET` | `/metrics` |
| `POST` | `/reload` |

## ▶ Usage

Code examples and CLI commands

### Training Script

```python
"""Training pipeline for handwritten digit recognition using a feedforward neural network."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from pattern_recognition_digits.data import (
    save_training_data,
    train_test_split,
)
from pattern_recognition_digits.model import DigitRecognitionNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 1000,
    hidden_dim: int = 64,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
    weight_decay: float = 0.0001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    noise_level: float = 0.3,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the digit recognition neural network and save artifacts.

    Uses softmax cross-entropy loss for multi-class classification of handwritten digits (0-9).
    """
    X, y = load_training_data_fn(data_path, n_samples, noise_level, random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info(
        "Data split",
        n_train=len(X_train),
        n_test=len(X_test),
        test_size=test_size,
        random_seed=random_seed,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.csv")

    model = DigitRecognitionNN(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        train_accuracy=train_metrics["accuracy"],
        test_accuracy=test_metrics["accuracy"],
    )

    model_path = model_dir / f"digit_recognition_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "n_features": float(X_train.shape[1]),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="handwritten-digit-recognition",
        model_version=model_version,
        model_type="pattern_recognition",
        metrics=metrics,
        parameters={
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"digit_recognition_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={
            "framework": "numpy",
            "task": "pattern_recognition",
            "model_type": "feedforward_neural_network",
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="handwritten-digit-recognition",
            model_version=model_version,
            metrics=metrics,
            params={
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "weight_decay": weight_decay,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"handwritten_digit_recognition_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "pattern_recognition", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow",
            model="handwritten-digit-recognition",
            version=model_version,
        )

    return metrics

def load_training_data_fn(data_path, n_samples, noise_level, random_seed):
    """Wrapper to avoid circular import."""
    from pattern_recognition_digits.data import load_training_data

    return load_training_data(data_path, n_samples, noise_level, random_seed)

def _save_chart(model: DigitRecognitionNN, output_dir: Path, version: str) -> None:
    """Save the training loss chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5, label="Training Loss")
    if model.val_accuracy_history:
        ax2 = ax.twinx()
        ax2.plot(
            range(0, len(model.val_accuracy_history) * 50, 50),
            model.val_accuracy_history,
            color="coral",
            linewidth=1.5,
            label="Validation Accuracy",
        )
        ax2.set_ylabel("Validation Accuracy", color="coral")
        ax2.legend(loc="center right")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss (Cross-Entropy + L2)")
    ax.set_title("Handwritten Digit Recognition NN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"handwritten_digit_recognition_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(
        description="Train handwritten digit recognition neural network"
    )
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "1000")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "64")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.1"))
    )
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "1000")))
    parser.add_argument(
        "--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.0001"))
    )
    parser.add_argument("--noise-level", type=float, default=float(os.getenv("NOISE_LEVEL", "0.3")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        noise_level=args.noise_level,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))

if __name__ == "__main__":
    main()
```

### API Server

```python
"""Production serving API for handwritten digit recognition via feedforward neural network."""

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

from pattern_recognition_digits.data import FEATURE_NAMES, N_CLASSES
from pattern_recognition_digits.model import DigitRecognitionNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("DIGIT_RECOGNITION_METRICS_PORT", "8011")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    """Handwritten digit recognition request."""

    pixels: list[float] = Field(
        ..., min_length=64, max_length=64, description="8x8=64 pixel values (0-1)"
    )

class PredictBulkRequest(BaseModel):
    """Bulk digit recognition request."""

    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    """Digit recognition prediction response."""

    digit: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    training_mode: str

class BulkPredictResponse(BaseModel):
    """Bulk digit recognition prediction response."""

    predictions: list[PredictResponse]
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
    n_classes: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: DigitRecognitionNN | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("handwritten_digit_recognition", port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "float" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="handwritten-digit-recognition",
        model_version=_model_version,
        model_type="pattern_recognition",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="handwritten-digit-recognition", version=_model_version)

    yield

    logger.info("Shutting down handwritten-digit-recognition API")

def _load_model() -> tuple[DigitRecognitionNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [
                m for m in models if m.get("model_name") == "handwritten-digit-recognition"
            ]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("digit_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return DigitRecognitionNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "handwritten-digit-recognition" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("digit_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return DigitRecognitionNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "digit_recognition_model.npz"
    if npz_path.exists():
        return DigitRecognitionNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/digit_recognition_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "digit_recognition_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return DigitRecognitionNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found on disk. Initializing baseline model.")
    from pattern_recognition_digits.data import generate_synthetic_data

    X_base, y_base = generate_synthetic_data(n_samples=500, random_seed=42)
    model = DigitRecognitionNN(hidden_dim=64, learning_rate=0.1, n_iterations=500, random_seed=42)
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    candidate_csvs = [
        MODEL_DIR / "handwritten-digit-recognition" / _model_version / "training_data.csv",
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

    from pattern_recognition_digits.data import generate_synthetic_data

    X_base, _ = generate_synthetic_data(n_samples=500, random_seed=42)
    return X_base

app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="Feedforward neural network for recognizing handwritten digits (0-9) from 8x8 pixel images",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "handwritten-digit-recognition-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
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
                model_name="handwritten-digit-recognition",
                model_version=_model_version,
                model_type="pattern_recognition",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically",
            model="handwritten-digit-recognition",
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
        n_classes=_model.n_classes,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_digit(pixels: list[float]) -> PredictResponse:
    if _model is None or _metrics is None or _drift_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([pixels])

    start = time.time()
    try:
        probs = _model.predict_proba(X)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(pixels)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        prob_dict = {str(i): round(float(probs[i]), 4) for i in range(N_CLASSES)}

        return PredictResponse(
            digit=digit,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            model_version=_model_version,
            training_mode=_model.training_mode if _model else "unknown",
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict_digit(body: PredictRequest):
    """Recognize a single handwritten digit."""
    return _compute_digit(body.pixels)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_digit_bulk(body: PredictBulkRequest):
    """Recognize multiple handwritten digits."""
    global _recent_predictions
    if _model is None or _metrics is None or _drift_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    X = np.array(body.requests)

    start = time.time()
    try:
        all_probs = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(body.requests)
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]

        predictions = []
        for probs in all_probs:
            digit = int(np.argmax(probs))
            confidence = float(np.max(probs))
            prob_dict = {str(i): round(float(probs[i]), 4) for i in range(N_CLASSES)}
            predictions.append(
                PredictResponse(
                    digit=digit,
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                    model_version=_model_version,
                    training_mode=_model.training_mode if _model else "unknown",
                )
            )

        return BulkPredictResponse(predictions=predictions, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Bulk prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Bulk prediction failed") from e
```

### CLI Commands

```bash
uv run python -m pattern_recognition_digits.train --model-dir ./artifacts/models
```

## 📊 Benchmarks

Test results and performance metrics

Run `pytest tests/test_models.py` and `pytest tests/test_apis.py` for detailed metrics.

### Related Apps

- [capsnet-text-recognition](../capsnet-text-recognition/README.md)

- [cnn-facial-recognition](../cnn-facial-recognition/README.md)

- [speech-audio-recognition](../speech-audio-recognition/README.md)

Generated documentation for **pattern-recognition-digits**
