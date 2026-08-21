# gan

## ∫ Mathematics & Theory

Generative Adversarial Network (GAN) — Underlying equations and derivations

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

$$G^*(x) = \arg\min_G \max_D V(D, G)$$

$$\nabla_\theta G \frac{1}{m} \sum_{i=1}^{m} \log(1 - D(G(z^{(i)})))$$

$$\mathcal{L}_{D} = -\frac{1}{m} \sum_{i=1}^{m} [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$$

### Step-by-Step Derivation

GANs consist of two networks competing in a minimax game. The discriminator $D$ maximizes its ability to distinguish real from fake samples. The generator $G$ minimizes the probability of its samples being detected as fake. At Nash equilibrium, $G$ reproduces the true data distribution $p_{data}$.

### Interactive Visualization

Interactive GAN training dashboard: generator/discriminator loss curves, sample evolution grid, decision boundary.

## ⚙ Architecture

Model structure, data flow, and layer breakdown

### Class Hierarchy

```
  GAN
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
"""Training pipeline for GAN Image Generation."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_gan_image_generation_schema

from gan_image_generation.data import (
    IMAGE_SIZE,
    N_CHANNELS,
    load_training_data,
    save_training_data,
    train_test_split,
)
from gan_image_generation.model import GAN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
     n_samples: int = 500,
     latent_dim: int = 16,
     hidden_dim: int = 32,
     learning_rate: float = 0.01,
    n_iterations: int = 200,
    weight_decay: float = 0.0001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the GAN model and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_gan_image_generation_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test), test_size=test_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = GAN(
         latent_dim=latent_dim,
         img_size=IMAGE_SIZE,
         n_channels=N_CHANNELS,
         hidden_dim=hidden_dim,
         learning_rate=learning_rate,
         n_iterations=n_iterations,
         weight_decay=weight_decay,
         random_seed=random_seed,
     )
    model.fit(X_train)

    model.evaluate(X_train)
    test_metrics = model.evaluate(X_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"gan_image_generation_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "gan",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "latent_dim": float(latent_dim),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="gan-image-generation",
        model_version=model_version,
        model_type="generative",
        metrics=metrics,
        parameters={
            "latent_dim": latent_dim,
            "img_size": IMAGE_SIZE,
            "n_channels": N_CHANNELS,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"gan_image_generation_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "gan_image_generation", "model_type": "GAN"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="gan-image-generation",
            model_version=model_version,
            metrics=metrics,
            params={
                "latent_dim": latent_dim,
                "img_size": IMAGE_SIZE,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"gan_image_generation_v{model_version}.png"),
            },
            tags={"model_type": "gan_image_generation", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="gan-image-generation", version=model_version)

    return metrics

def _save_chart(model, output_dir: Path, version: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5, label="Total Loss")
    if model._gen_loss_history:
        ax.plot(model._gen_loss_history, color="orange", linewidth=1.0, label="Generator Loss")
    if model._disc_loss_history:
        ax.plot(model._disc_loss_history, color="green", linewidth=1.0, label="Discriminator Loss")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Image Generation Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"gan_image_generation_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train GAN Image Generation model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "16")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.01")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "200")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.0001")))
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
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))

if __name__ == "__main__":
    main()
```

### API Server

```python
"""Serving API for GAN Image Generation.

Provides image generation via the trained GAN generator and real/fake
classification via the discriminator.
"""

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
from ai_core.validation import DataValidator, create_gan_image_generation_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from gan_image_generation.data import IMAGE_SIZE, N_CHANNELS, N_FEATURES, generate_synthetic_data
from gan_image_generation.model import GAN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("GAN_IMAGE_GENERATION_METRICS_PORT", "8022"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    latent_vector: list[float] = Field(..., min_length=16, max_length=16)

class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    generated_pixels: list[float]
    confidence: float
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
    latent_dim: int
    img_size: int
    n_channels: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: GAN | None = None
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
    _metrics = MetricsCollector("gan_image_generation", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_gan_image_generation_schema())
    feature_names = [f"pixel_{i}" for i in range(N_FEATURES)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="gan-image-generation",
        model_version=_model_version,
        model_type="generative",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="gan-image-generation", version=_model_version)

    yield
    logger.info("Shutting down gan-image-generation API")

def _load_model() -> tuple[GAN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "gan-image-generation"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("gan_image_generation_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return GAN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "gan-image-generation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("gan_image_generation_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return GAN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "gan_image_generation_model.npz"
    if npz_path.exists():
        return GAN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/gan_image_generation_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "gan_image_generation_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return GAN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    model = GAN(
        latent_dim=16,
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        hidden_dim=32,
        learning_rate=0.01,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base

app = FastAPI(
    title="GAN Image Generation API",
    description="Creates new synthetic images using a generator network trained via adversarial competition",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "gan_image_generation-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": N_FEATURES,
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
                model_name="gan-image-generation",
                model_version=_model_version,
                model_type="generative",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="gan-image-generation", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e

@app.get("/drift", response_model=DriftResponse)
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
    if _model is None or _model._gen_W1 is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        latent_dim=_model.latent_dim,
        img_size=_model.img_size,
        n_channels=_model.n_channels,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(latent_vector: list[float]):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    z = np.array([latent_vector])
    predictions = _model.predict(z)[0]

    response = PredictResponse(
        generated_pixels=predictions.flatten().tolist(),
        confidence=round(float(np.max(np.abs(predictions))), 4),
        model_version=_model_version,
        training_mode=_model.training_mode,
    )
    _recent_predictions.append(latent_vector)
    if len(_recent_predictions) > 1000:
        _recent_predictions.pop(0)

    return response

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Generate an image from a latent vector."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        response = _compute_prediction(body.latent_vector)
        duration = time.time() - start
        if _metrics:
            _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        if _metrics:
            _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Generate multiple images from latent vectors."""
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for latent_vector in body.requests:
        predictions.append(_compute_prediction(latent_vector))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
```

### CLI Commands

```bash
uv run python -m gan.train --model-dir ./artifacts/models
```

## 📊 Benchmarks

Test results and performance metrics

Run `pytest tests/test_models.py` and `pytest tests/test_apis.py` for detailed metrics.

Generated documentation for **gan**
