# cnn-video-surveillance



Convolutional Neural Network — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Convolutional Neural Network**. The equations below
drive every forward and backward pass in the implementation.

$$Z^{(l)} = W^{(l)} * X^{(l)} + b^{(l)}$$

$$A^{(l)} = \text{ReLU}(Z^{(l)})$$

$$\text{MaxPool}(X)_{i,j} = \max_{m \in \mathcal{R}_i, n \in \mathcal{R}_j} X_{m,n}$$

$$\text{Softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### Derivation

CNNs apply learned filters across spatial dimensions. Convolution slides a kernel $W$ over the input, computing dot products at each position. ReLU introduces non-linearity. Pooling reduces spatial dimensions. The softmax converts final logits to class probabilities.

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

Interactive filter visualization; feature map heatmap; receptive field calculator; Grad-CAM overlay.

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
| `VideoSurveillanceCNN` | fit, predict_proba, predict_class, predict, accuracy, evaluate, save, load, to_dict | CNN for video surveillance.  Args:     img_size: Size of input images (square)     n_channels: Number of input channels     n_filters: Number of convolution filters     kernel_size: Convolution kernel size     hidden_dim: Hidden units in dense layer     output_dim: Output dimension     learning_rate: Gradient descent step size     n_iterations: Number of training epochs     weight_decay: L2 regularization strength     clip_value: Maximum gradient norm for clipping     random_seed: Random seed for reproducibility |

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

### `VideoSurveillanceCNN.fit(X, y, X_val, y_val)`

Train the CNN using backpropagation.

Args:
    X: Image pixel arrays (n_samples, N_FEATURES)
    y: Class labels (n_samples,)

Returns:
    self

### `VideoSurveillanceCNN.predict(X, threshold)`

Alias for predict_class.

### Source Files

<details>
<summary>model.py</summary>

```
"""Video Surveillance using a Convolutional Neural Network.

Architecture:
    Input (1 x 8x8) -> Conv2D (1->8, 3x3, ReLU)
    -> MaxPool2D (2x2) -> Flatten -> Dense (32, ReLU) -> Dense (3, softmax)

Loss: categorical cross-entropy (many-to-one: classifies image into a class label)
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.cnn import SimpleCNN

from cnn_video_surveillance.data import reshape_image

@dataclass
class VideoSurveillanceCNN:
    """CNN for video surveillance.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input channels
        n_filters: Number of convolution filters
        kernel_size: Convolution kernel size
        hidden_dim: Hidden units in dense layer
        output_dim: Output dimension
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm for clipping
        random_seed: Random seed for reproducibility
    """

    IMG_SIZE: int = 8
    N_CHANNELS: int = 1
    n_filters: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    output_dim: int = 3
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleCNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "VideoSurveillanceCNN":
        """Train the CNN using backpropagation.

        Args:
            X: Image pixel arrays (n_samples, N_FEATURES)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_arr = np.asarray(y, dtype=float)

        if self.output_dim == 1:
            y_arr = y_arr.reshape(-1, 1)
        else:
            onehot = np.zeros((len(y_arr), self.output_dim))
            onehot[np.arange(len(y_arr)), y_arr.astype(int)] = 1.0
            y_arr = onehot

        self.model = SimpleCNN(
            input_shape=(self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE),
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_activation="softmax",
            output_loss="cross_entropy",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
        )
        self.model.fit(X_img, y_arr, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return output probabilities for each sample."""
        X_img = reshape_image(X)
        return self.model.predict_proba(X_img)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return predicted class indices."""
        if self.output_dim == 1:
            probas = self.predict_proba(X).flatten()
            return (probas >= threshold).astype(int)
        X_img = reshape_image(X)
        return self.model.predict(X_img)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Alias for predict_class."""
        return self.predict_class(X, threshold)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict_class(X)
        return float(np.mean(preds == y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict_class(X)
        acc = float(np.mean(preds == y))
        return {
            "accuracy": acc,
            "n_samples": float(len(y)),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "VideoSurveillanceCNN":
        model = SimpleCNN.load(path)
        obj = cls()
        obj.model = model
        obj.loss_history = model.loss_history
        obj.output_dim = model.output_dim
        return obj

    def to_dict(self) -> dict:
        return {
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
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
"""Training pipeline for Video Surveillance (CNN)."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_video_surveillance_schema

from cnn_video_surveillance.data import (
    load_training_data,
    save_training_data,
    train_test_split,
)
from cnn_video_surveillance.model import VideoSurveillanceCNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_filters: int = 8,
    kernel_size: int = 3,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    n_iterations: int = 300,
    weight_decay: float = 0.001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the video surveillance model and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    if "classification" in ("classification", "binary_classification"):
        validator = DataValidator(create_video_surveillance_schema())
        validation = validator.validate(X.reshape(-1, 1))
        if not validation.valid:
            logger.error("Training data validation failed", errors=validation.errors)
            raise ValueError(f"Training data validation failed: {validation.errors}")
        logger.info("Training data validated", stats=validation.stats)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test), test_size=test_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = VideoSurveillanceCNN(
        IMG_SIZE=8,
        N_CHANNELS=1,
        n_filters=n_filters,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"video_surveillance_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "n_filters": float(n_filters),
        "learning_rate": float(learning_rate),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="video-surveillance",
        model_version=model_version,
        model_type="classification",
        metrics=metrics,
        parameters={
            "img_size": 8,
            "n_channels": 1,
            "n_filters": n_filters,
            "kernel_size": kernel_size,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"video_surveillance_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "video_surveillance", "model_type": "CNN"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="video-surveillance",
            model_version=model_version,
            metrics=metrics,
            params={
                "img_size": 8,
                "n_filters": n_filters,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"video_surveillance_v{model_version}.png"),
            },
            tags={"model_type": "video_surveillance", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="video-surveillance", version=model_version)

    return metrics

def _save_chart(model, output_dir: Path, version: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Video Surveillance CNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"video_surveillance_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train Video Surveillance model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--n-filters", type=int, default=int(os.getenv("N_FILTERS", "8")))
    parser.add_argument("--kernel-size", type=int, default=int(os.getenv("KERNEL_SIZE", "3")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.05")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "300")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
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
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
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

</details>

<details>
<summary>data.py</summary>

```
"""Data loading and preprocessing for Video Surveillance (CNN).

Generates synthetic 8x8 grayscale images and their labels.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = 8
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 3

DEFAULT_N_SAMPLES = 500

LABEL_NAMES = ['normal', 'activity', 'threat']

def _create_template(label: int, rng: np.random.Generator) -> np.ndarray:
    """Create a 8x8 template pattern for a given class."""
    grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    # Distinct spatial patterns per class
    patterns = [
        lambda r, c: IMAGE_SIZE // 4 <= r <= 3 * IMAGE_SIZE // 4 and IMAGE_SIZE // 4 <= c <= 3 * IMAGE_SIZE // 4,
        lambda r, c: r < IMAGE_SIZE // 2,
        lambda r, c: (r + c) % 3 == 0,
        lambda r, c: r == 0 or r == IMAGE_SIZE - 1 or c == 0 or c == IMAGE_SIZE - 1,
        lambda r, c: (r - IMAGE_SIZE // 2) ** 2 + (c - IMAGE_SIZE // 2) ** 2 <= 4,
        lambda r, c: r > IMAGE_SIZE // 2 and c > IMAGE_SIZE // 2,
        lambda r, c: (r + c) % 2 == 0,
        lambda r, c: r == c,
        lambda r, c: r + c == IMAGE_SIZE - 1,
        lambda r, c: True,
    ]
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if label < len(patterns) and patterns[label](r, c):
                grid[r, c] = 0.9
    return grid.flatten()

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their labels.

    Returns:
        X: (n_samples, N_FEATURES) flattened image pixel arrays
        y: (n_samples,) class labels
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = rng.integers(0, N_CLASSES) if N_CLASSES > 0 else rng.integers(0, 2)
        template = _create_template(label, rng)
        X[i] = np.clip(template + rng.normal(0, noise_level, N_FEATURES), 0, 1)
        y[i] = label

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]

def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, noise_level=noise_level, random_seed=random_seed)

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

def reshape_image(X: np.ndarray) -> np.ndarray:
    """Reshape flattened images to (N, 1, IMAGE_SIZE, IMAGE_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
```

</details>

<details>
<summary>api.py</summary>

```
"""Serving API for Video Surveillance (CNN)."""

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
from ai_core.validation import DataValidator, create_video_surveillance_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from cnn_video_surveillance.data import IMAGE_SIZE, N_CHANNELS, generate_synthetic_data
from cnn_video_surveillance.model import VideoSurveillanceCNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("VIDEO_SURVEILLANCE_METRICS_PORT", "8018"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    activity: list[float] = Field(..., min_length=64, max_length=64)

class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    activity: str | list[float] | bool
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
    img_size: int
    n_channels: int
    n_filters: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: VideoSurveillanceCNN | None = None
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
    _metrics = MetricsCollector("video_surveillance", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_video_surveillance_schema())
    feature_names = [f"pixel_{i}" for i in range(64)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="video-surveillance",
        model_version=_model_version,
        model_type="classification",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="video-surveillance", version=_model_version)

    yield
    logger.info("Shutting down video-surveillance API")

def _load_model() -> tuple[VideoSurveillanceCNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "video-surveillance"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("video_surveillance_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return VideoSurveillanceCNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "video-surveillance" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("video_surveillance_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return VideoSurveillanceCNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "video_surveillance_model.npz"
    if npz_path.exists():
        return VideoSurveillanceCNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/video_surveillance_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "video_surveillance_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return VideoSurveillanceCNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = VideoSurveillanceCNN(
        IMG_SIZE=8,
        N_CHANNELS=1,
        n_filters=8,
        kernel_size=3,
        hidden_dim=32,
        learning_rate=0.05,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base

app = FastAPI(
    title="Video Surveillance API",
    description="Tracks crowd movement and spots security threats in real-time frame images",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "video_surveillance-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": 64,
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
                model_name="video-surveillance",
                model_version=_model_version,
                model_type="classification",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="video-surveillance", version=_model_version)
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
            "total_features": 64,
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
    if _model is None or getattr(_model, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        n_filters=getattr(_model, "n_filters", 8),
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(activity: list[float]):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([activity]).reshape(1, -1)

    if "classification" in ("classification", "binary_classification"):
        validation = _validator.validate(X)
    else:
        validation = _validator.validate(X)

    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        probas = _model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probas))
        confidence = float(np.max(probas))
        response = PredictResponse(
            activity=str(pred_idx),
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(activity)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Make a video surveillance prediction."""
    return _compute_prediction(body.activity)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Make multiple video surveillance predictions."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for activity in body.requests:
        predictions.append(_compute_prediction(activity))

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
ai_core.nn_utils
ai_core.validation

### How it plugs in



- **Configuration** — 12-factor config from `ai_core.config`.



- **Observability** — structured logging + Prometheus metrics are wired in automatically.



- **Validation** — input schema validation prevents bad data reaching the model.



- **Registry** — trained artifacts are versioned and registered for reproducible serving.



- **Serving** — the FastAPI app mounts shared observability middleware for tracing & metrics.

Because every example shares `ai_core`, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.
