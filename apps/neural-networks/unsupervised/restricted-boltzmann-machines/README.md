# restricted-boltzmann-machines



Machine Learning Fundamentals — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Machine Learning Fundamentals**. The equations below
drive every forward and backward pass in the implementation.

$$\hat{y} = f(x; \theta)$$

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, \hat{y}_i)$$

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

### Derivation

Machine learning models learn parameters $\theta$ by minimizing a loss function $\mathcal{L}$. Gradient descent iteratively updates parameters in the direction of steepest descent. The learning rate $\alpha$ controls step size. Stochastic gradient descent (SGD) uses mini-batches for computational efficiency.

### Worked Numerical Example

$$z = w \cdot x + b$$

Illustrative forward-pass evaluation (scalar example):

Input  x        = 12.0   (e.g. pizza diameter, inches)
Weights w       =  0.85
Bias    b       =  0.30
---------------------------------
z = w*x + b
  = 0.85 * 12.0 + 0.30
  = 10.20 + 0.30
  = 10.50   <- model output

### Conceptual Diagram

        Math concept (placeholder)
   [ Input x ] --> ( w · x + b ) --> [ Output z ]
                       |
                  [ activation ]
                       |
                  [ prediction ]

![Math & architecture diagram](./assets/math-concept.png)

Interactive loss landscape explorer; gradient descent trajectory; learning rate scheduler.

## 2. Core Logic & Architecture

The example follows a consistent **data → train → evaluate → serve**
pipeline. Inputs are loaded and validated, transformed by the core algorithm, scored against
held-out data, and exposed through a REST API.

  Raw dataset→
  load + validate (data.py)→
  fit / transform (model.py)→
  evaluate + persist (train.py)→
  serve (api.py)

### Primary Components

| Class | Public methods | Responsibility |
| --- | --- | --- |
| `PredictRequest` | — |  |
| `PredictBulkRequest` | — |  |
| `PredictResponse` | — |  |
| `BulkPredictResponse` | — |  |
| `DriftResponse` | — |  |
| `StatsResponse` | — |  |
| `RBM` | _init_weights, _sample_h, _sample_v, fit, transform, reconstruct, predict_proba, predict, evaluate, save, load, to_dict | Restricted Boltzmann Machine for unsupervised feature learning.  Learns a probability distribution over binary inputs and extracts hierarchical features through its hidden representation.  Args:     n_features: Number of visible units (input features)     n_hidden: Number of hidden units     learning_rate: Learning rate     n_cd_steps: Contrastive Divergence steps (CD-k)     n_epochs: Number of training epochs     weight_decay: L2 regularization     random_seed: Random seed |

### Data Flow



1. **Load** — `data.py` reads the source dataset and splits train/test.



2. **Validate** — a Pydantic schema guards input shape/dtypes before training.



3. **Fit / Transform** — `model.py` applies the mathematics from Section 1.



4. **Evaluate** — metrics (MSE/RMSE/R², accuracy, etc.) are computed and logged.



5. **Persist** — weights/artifacts are saved and registered in the model registry.



6. **Serve** — `api.py` exposes prediction endpoints with drift detection.

### Design Patterns & Performance

Key design choices in this module: a pure-NumPy implementation (no PyTorch/TensorFlow), schema validation via `ai_core.validation`, structured JSON logging through `ai_core.logging`, Prometheus metrics from `ai_core.metrics`, and MLflow/model-registry persistence via `ai_core.model_registry`. The FastAPI service wraps the trained model with observability middleware from `ai_core.fastapi_middleware`.

## 3. Detailed Code Walkthrough

The most important behaviour is summarised below; full source for each module is collapsible
so the page stays readable while remaining self-contained.

### `RBM.fit(X, n_epochs)`

Train the RBM using Contrastive Divergence.

Args:
    X: Binary input data (n_samples, n_features) in [0, 1]

### `RBM.predict(X)`

Return hidden representations (binary samples).

### Source Files

<details>
<summary>model.py</summary>

```
"""Restricted Boltzmann Machine for feature learning.

Architecture:
    Binary visible units (n_features) <-> Hidden units (n_hidden)
    Fully connected undirected bipartite graph

Training: Contrastive Divergence (CD-k)
Loss: Reconstruction cross-entropy
"""

from dataclasses import dataclass, field

import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def _bernoulli_sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(probs.shape) < probs).astype(float)

@dataclass
class RBM:
    """Restricted Boltzmann Machine for unsupervised feature learning.

    Learns a probability distribution over binary inputs and extracts
    hierarchical features through its hidden representation.

    Args:
        n_features: Number of visible units (input features)
        n_hidden: Number of hidden units
        learning_rate: Learning rate
        n_cd_steps: Contrastive Divergence steps (CD-k)
        n_epochs: Number of training epochs
        weight_decay: L2 regularization
        random_seed: Random seed
    """

    n_features: int = 32
    n_hidden: int = 16
    learning_rate: float = 0.05
    n_cd_steps: int = 1
    n_epochs: int = 100
    weight_decay: float = 0.001
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    c: np.ndarray | None = None
    loss_history: list[float] = field(default_factory=list)
    training_mode: str = "unsupervised"

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W = rng.normal(0, 0.01, (self.n_features, self.n_hidden))
        self.b = np.zeros(self.n_features)
        self.c = np.zeros(self.n_hidden)

    def _sample_h(self, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(v @ self.W + self.c)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def _sample_v(self, h: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(h @ self.W.T + self.b)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def fit(
        self,
        X: np.ndarray,
        n_epochs: int | None = None,
    ) -> "RBM":
        """Train the RBM using Contrastive Divergence.

        Args:
            X: Binary input data (n_samples, n_features) in [0, 1]
        """
        if self.W is None:
            self._init_weights()

        if n_epochs is None:
            n_epochs = self.n_epochs

        rng = np.random.default_rng(self.random_seed)
        n_samples = X.shape[0]

        for _epoch in range(n_epochs):
            epoch_loss = 0.0
            X_shuffled = X[rng.permutation(n_samples)]

            for i in range(n_samples):
                v = X_shuffled[i:i + 1]

                h_prob, h_sample = self._sample_h(v, rng)

                for _ in range(self.n_cd_steps - 1):
                    v_prob, v_sample = self._sample_v(h_sample, rng)
                    h_prob, h_sample = self._sample_h(v_prob, rng)

                v_k_prob, v_k_sample = self._sample_v(h_sample, rng)
                h_k_prob, _ = self._sample_h(v_k_sample, rng)

                dW = np.outer(v[0], h_prob[0]) - np.outer(v_k_prob[0], h_k_prob[0])
                db = v[0] - v_k_prob[0]
                dc = h_prob[0] - h_k_prob[0]

                self.W -= self.learning_rate * (dW + self.weight_decay * self.W)
                self.b -= self.learning_rate * db
                self.c -= self.learning_rate * dc

                epoch_loss += np.mean((v[0] - v_k_prob[0]) ** 2)

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input data to hidden representation."""
        return _sigmoid(X @ self.W + self.c)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the RBM."""
        h = _sigmoid(X @ self.W + self.c)
        return _sigmoid(h @ self.W.T + self.b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return hidden feature activations."""
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hidden representations (binary samples)."""
        h_probs = _sigmoid(X @ self.W + self.c)
        return (h_probs > 0.5).astype(float)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        recon = self.reconstruct(X)
        mse = float(np.mean((X - recon) ** 2))
        return {"reconstruction_error": mse, "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "W": self.W, "b": self.b, "c": self.c,
            "n_features": np.array([self.n_features]),
            "n_hidden": np.array([self.n_hidden]),
            "learning_rate": np.array([self.learning_rate]),
            "n_cd_steps": np.array([self.n_cd_steps]),
            "n_epochs": np.array([self.n_epochs]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "RBM":
        data = np.load(path)
        obj = cls(
            n_features=int(data["n_features"].item()),
            n_hidden=int(data["n_hidden"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_cd_steps=int(data["n_cd_steps"].item()),
            n_epochs=int(data["n_epochs"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj.W = data["W"]
        obj.b = data["b"]
        obj.c = data["c"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "n_cd_steps": self.n_cd_steps,
            "n_epochs": self.n_epochs,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
```

</details>

<details>
<summary>train.py</summary>

```
"""Training pipeline for RBM Feature Learning."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_rbm_feature_learning_schema

from rbm_feature_learning.data import (
    N_FEATURES,
    load_training_data,
    save_training_data,
    train_test_split,
)
from rbm_feature_learning.model import RBM

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_hidden: int = 16,
    learning_rate: float = 0.05,
    n_cd_steps: int = 1,
    n_epochs: int = 100,
    weight_decay: float = 0.001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_rbm_feature_learning_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")

    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    X_bin = (X_train > 0.5).astype(float)
    X_test_bin = (X_test > 0.5).astype(float)

    model = RBM(
        n_features=N_FEATURES,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        n_cd_steps=n_cd_steps,
        n_epochs=n_epochs,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_bin)

    test_metrics = model.evaluate(X_test_bin)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"rbm_model_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "training_mode": "unsupervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="rbm-feature-learning",
        model_version=model_version,
        model_type="generative",
        metrics=metrics,
        parameters={
            "n_features": N_FEATURES,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "n_cd_steps": n_cd_steps,
            "n_epochs": n_epochs,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"rbm_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "rbm_feature_learning", "model_type": "RBM"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="rbm-feature-learning",
            model_version=model_version,
            metrics=metrics,
            params={"n_features": N_FEATURES, "n_hidden": n_hidden, "learning_rate": learning_rate, "n_epochs": n_epochs},
            artifacts={"model": str(model_path)},
            tags={"model_type": "rbm", "framework": "numpy"},
        )

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train RBM Feature Learning model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--n-hidden", type=int, default=int(os.getenv("N_HIDDEN", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.05")))
    parser.add_argument("--n-cd-steps", type=int, default=int(os.getenv("N_CD_STEPS", "1")))
    parser.add_argument("--n-epochs", type=int, default=int(os.getenv("N_EPOCHS", "100")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        n_hidden=args.n_hidden,
        learning_rate=args.learning_rate,
        n_cd_steps=args.n_cd_steps,
        n_epochs=args.n_epochs,
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

</details>

<details>
<summary>data.py</summary>

```
"""Data loading and preprocessing for RBM feature learning."""

from pathlib import Path

import numpy as np

N_FEATURES = 32
N_HIDDEN = 16

DEFAULT_N_SAMPLES = 500

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary feature data for RBM training.

    Returns:
        X: (n_samples, N_FEATURES) binary feature vectors in {0, 1}
        y: (n_samples,) uniform labels (placeholder)
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES), dtype=float)

    for i in range(n_samples):
        pattern = rng.integers(0, 3)
        if pattern == 0:
            X[i, :N_FEATURES // 4] = 1.0
            X[i, rng.random(N_FEATURES) > 0.95] = 1.0
        elif pattern == 1:
            X[i, N_FEATURES // 4:N_FEATURES // 2] = 1.0
            X[i, rng.random(N_FEATURES) > 0.95] = 1.0
        else:
            X[i, :N_FEATURES // 2] = (rng.random(N_FEATURES // 2) > noise_level).astype(float)

    y = np.ones(n_samples, dtype=int)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]

def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)
```

</details>

<details>
<summary>api.py</summary>

```
"""Serving API for RBM Feature Learning."""

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
from ai_core.validation import DataValidator, create_rbm_feature_learning_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from rbm_feature_learning.data import N_FEATURES, generate_synthetic_data
from rbm_feature_learning.model import RBM

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("RBM_METRICS_PORT", "8027"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)

class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    latent_activations: list[float]
    reconstruction: list[float]
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
    n_features: int
    n_hidden: int
    learning_rate: float
    n_cd_steps: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: RBM | None = None
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
    _metrics = MetricsCollector("rbm_feature_learning", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_rbm_feature_learning_schema())
    feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="rbm-feature-learning",
        model_version=_model_version,
        model_type="generative",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="rbm-feature-learning", version=_model_version)

    yield
    logger.info("Shutting down rbm-feature-learning API")

def _load_model() -> tuple[RBM, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "rbm-feature-learning"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("rbm_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return RBM.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "rbm-feature-learning" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("rbm_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return RBM.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "rbm_model.npz"
    if npz_path.exists():
        return RBM.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/rbm_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "rbm_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return RBM.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    model = RBM(
        n_features=N_FEATURES,
        n_hidden=16,
        learning_rate=0.05,
        n_cd_steps=1,
        n_epochs=50,
        random_seed=42,
    )
    model.fit(X_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base

app = FastAPI(
    title="RBM Feature Learning API",
    description="Stochastic neural networks that learn probability distributions over binary inputs",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "rbm_feature_learning-api",
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
                model_name="rbm-feature-learning",
                model_version=_model_version,
                model_type="generative",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="rbm-feature-learning", version=_model_version)
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
    if _model is None or _model.W is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        n_features=_model.n_features,
        n_hidden=_model.n_hidden,
        learning_rate=_model.learning_rate,
        n_cd_steps=_model.n_cd_steps,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(features: list[float]):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([features]).reshape(1, -1)
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        X_bin = (X > 0.5).astype(float)
        latent = _model.transform(X_bin)[0]
        recon = _model.reconstruct(X_bin)[0]
        confidence = float(np.max(latent))
        response = PredictResponse(
            latent_activations=latent.tolist(),
            reconstruction=recon.tolist(),
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(features)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.features)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")
    predictions = [_compute_prediction(f) for f in body.requests]
    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
```

</details>

## 4. Monorepo Integration

This example is a first-class consumer of the shared `packages/ai-core` library.
It reuses the following foundation modules instead of re-implementing infrastructure:

ai_core.drift
ai_core.fastapi_middleware
ai_core.logging
ai_core.metrics
ai_core.model_registry
ai_core.validation

### How it plugs in



- **Configuration** — 12-factor config from `ai_core.config`.



- **Observability** — structured logging + Prometheus metrics are wired in automatically.



- **Validation** — input schema validation prevents bad data reaching the model.



- **Registry** — trained artifacts are versioned and registered for reproducible serving.



- **Serving** — the FastAPI app mounts shared observability middleware for tracing & metrics.

Because every example shares `ai_core`, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.
