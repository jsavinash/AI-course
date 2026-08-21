# advanced-super-resolution



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

![Math Explanation (placeholder)](./assets/math-concept.png)

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
| `_Conv2D` | _init, forward, backward, update_params | Minimal conv2d that tracks its own gradients for the DN encoder. |
| `ImageSuperResolutionDN` | _build, fit, predict, predict_proba, mse, rmse, evaluate, save, load, to_dict | Deconvolutional network for image super-resolution.  Args:     img_size: Size of input images (square)     n_channels: Number of input/output channels     n_filters: Number of filters in conv layers     kernel_size: Convolution kernel size     learning_rate: Gradient descent step size     n_iterations: Number of training epochs     weight_decay: L2 regularization strength     random_seed: Random seed |

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

### `ImageSuperResolutionDN.fit(X, y, X_val, y_val)`

Train the deconvolutional network.

Args:
    X: Input images (n_samples, N_FEATURES)
    y: Target images (n_samples, N_FEATURES)

Returns:
    self

### `ImageSuperResolutionDN.predict(X)`

Return predictions for a batch of images.

### Source Files

<details>
<summary>model.py</summary>

```
"""Image Super-Resolution using a Deconvolutional Network (DN).

Architecture:
    Input (1 x 8x8) -> Conv2D (8, 3x3, ReLU) -> MaxPool2D (2x2)
    -> Deconv2D (8, 3x3, ReLU) -> Deconv2D (1, 3x3, linear)

Loss: mean squared error (many-to-many: outputs pixel-level reconstruction)
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.cnn import Activation, Deconv2D

from advanced_super_resolution.data import reshape_image

@dataclass
class _Conv2D:
    """Minimal conv2d that tracks its own gradients for the DN encoder."""

    n_filters: int = 8
    kernel_size: int = 3
    random_seed: int = 42
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    def _init(self, C: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        fan_in = C * self.kernel_size * self.kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), (self.n_filters, C, self.kernel_size, self.kernel_size))
        self.b = np.zeros(self.n_filters)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            self._init(X.shape[1])
        N, C, H, W = X.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        out = np.zeros((N, self.n_filters, H_out, W_out))
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        region = X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        out[n, f, h, w] = np.sum(region * self.W[f]) + self.b[f]
        self._cache = {"X": X, "H_out": H_out, "W_out": W_out}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        X = c["X"]
        N, C, H, W = X.shape
        H_out, W_out = c["H_out"], c["W_out"]
        actual_filters = min(self.n_filters, dout.shape[1])
        dW = np.zeros_like(self.W)
        db = np.zeros(self.n_filters)
        dX = np.zeros_like(X)
        for n in range(N):
            for f in range(actual_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        val = dout[n, f, h, w]
                        dW[f] += val * X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        db[f] += val
                        dX[n, :, h:h + self.kernel_size, w:w + self.kernel_size] += val * self.W[f]
        self.dW = dW / N
        self.db = db / N
        return dX

    def update_params(self, lr: float, wd: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + wd * self.W)
        self.b -= lr * self.db

@dataclass
class ImageSuperResolutionDN:
    """Deconvolutional network for image super-resolution.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input/output channels
        n_filters: Number of filters in conv layers
        kernel_size: Convolution kernel size
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        random_seed: Random seed
    """

    IMG_SIZE: int = 8
    N_CHANNELS: int = 1
    n_filters: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.0001
    clip_value: float = 1.0
    random_seed: int = 42

    _layers: list = field(default_factory=list, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        # Encoder-decoder: 8x8 -> Conv(3) -> 6x6 -> Conv(3) -> 4x4
        # -> Deconv(3,stride=1) -> 6x6 -> Deconv(3,stride=1) -> 8x8
        self._layers = [
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed),
            Activation("relu"),
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed + 1),
            Activation("relu"),
            Deconv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 2),
            Activation("relu"),
            Deconv2D(n_filters=self.N_CHANNELS, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 3),
            Activation("linear"),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "ImageSuperResolutionDN":
        """Train the deconvolutional network.

        Args:
            X: Input images (n_samples, N_FEATURES)
            y: Target images (n_samples, N_FEATURES)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_img = reshape_image(y)
        self._build()

        N = X_img.shape[0]
        eps = 1e-12

        for epoch in range(self.n_iterations):
            total_loss = 0.0
            for i in range(N):
                out = X_img[i:i + 1]
                for layer in self._layers:
                    out = layer.forward(out)

                target = y_img[i:i + 1]
                if "mse" == "mse":
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(np.mean((target - clip_out) ** 2))
                else:
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(-np.mean(target * np.log(clip_out) + (1 - target) * np.log(1 - clip_out)))
                total_loss += loss

                if "mse" == "mse":
                    dout = 2 * (out - target) / max(out.shape[0], 1)
                else:
                    dout = (out - target) / max(out.shape[0], 1)

                for layer in reversed(self._layers):
                    dout = layer.backward(dout)

                # Gradient clipping
                grad_norm = 0.0
                for layer in self._layers:
                    if hasattr(layer, "dW") and layer.dW is not None:
                        grad_norm += float(np.sum(layer.dW ** 2))
                    if hasattr(layer, "db") and layer.db is not None:
                        grad_norm += float(np.sum(layer.db ** 2))
                grad_norm = np.sqrt(grad_norm)
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    for layer in self._layers:
                        if hasattr(layer, "dW") and layer.dW is not None:
                            layer.dW *= scale
                        if hasattr(layer, "db") and layer.db is not None:
                            layer.db *= scale

                for layer in self._layers:
                    layer.update_params(self.learning_rate, self.weight_decay)

            self.loss_history.append(total_loss / N)
            if epoch > 50 and len(self.loss_history) > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-8:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of images."""
        X_img = reshape_image(X)
        results = []
        for i in range(X_img.shape[0]):
            out = X_img[i:i + 1]
            for layer in self._layers:
                out = layer.forward(out)
            results.append(out[0])
        return np.array(results)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds.flatten() - y.flatten()) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        mse_val = self.mse(X, y)
        return {"mse": mse_val, "rmse": float(np.sqrt(mse_val)), "n_samples": float(X.shape[0])}

    def save(self, path: str) -> None:
        arrays = {"loss_history": np.array(self.loss_history)}
        for i, layer in enumerate(self._layers):
            if isinstance(layer, _Conv2D):
                if layer.W is not None:
                    arrays[f"conv_{i}_W"] = layer.W
                    arrays[f"conv_{i}_b"] = layer.b
            elif isinstance(layer, Deconv2D) and layer.W is not None:
                arrays[f"deconv_{i}_W"] = layer.W
                arrays[f"deconv_{i}_b"] = layer.b
        arrays["n_filters"] = np.array(self.n_filters)
        arrays["learning_rate"] = np.array(self.learning_rate)
        arrays["n_iterations"] = np.array(self.n_iterations)
        arrays["weight_decay"] = np.array(self.weight_decay)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "ImageSuperResolutionDN":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_filters=int(data.get("n_filters", 8)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            n_iterations=int(data.get("n_iterations", 300)),
            weight_decay=float(data.get("weight_decay", 0.0001)),
            random_seed=42,
        )
        obj._build()
        for i, layer in enumerate(obj._layers):
            if isinstance(layer, _Conv2D) and f"conv_{i}_W" in data:
                layer.W = data[f"conv_{i}_W"]
                layer.b = data[f"conv_{i}_b"]
            elif isinstance(layer, Deconv2D) and f"deconv_{i}_W" in data:
                layer.W = data[f"deconv_{i}_W"]
                layer.b = data[f"deconv_{i}_b"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
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
"""Training pipeline for Image Super-Resolution (DN)."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_image_super_resolution_schema

from advanced_super_resolution.data import (
    load_training_data,
    save_training_data,
    train_test_split,
)
from advanced_super_resolution.model import ImageSuperResolutionDN

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
    """Train the image super-resolution model and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    if "super_resolution" in ("classification", "binary_classification"):
        validator = DataValidator(create_image_super_resolution_schema())
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

    model = ImageSuperResolutionDN(
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
    model.fit(X_train, y_train)

    model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"image_super_resolution_model_v{model_version}.npz"
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
        model_name="image-super-resolution",
        model_version=model_version,
        model_type="regression",
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
            f"image_super_resolution_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "image_super_resolution", "model_type": "DN"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="image-super-resolution",
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
                "chart": str(model_dir / f"image_super_resolution_v{model_version}.png"),
            },
            tags={"model_type": "image_super_resolution", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="image-super-resolution", version=model_version)

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
    ax.set_title("Image Super-Resolution DN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"image_super_resolution_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train Image Super-Resolution model")
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
"""Data loading and preprocessing for Image Super-Resolution (DN).

Generates synthetic 8x8 images for pixel-to-pixel tasks.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = 8
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 0

DEFAULT_N_SAMPLES = 500

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic low-quality images and their target images.

    Returns:
        X: (n_samples, N_FEATURES) input image pixels
        y: (n_samples, N_FEATURES) target image pixels
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        grid_hr = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
        cx, cy = rng.integers(2, IMAGE_SIZE - 2, size=2)
        r = rng.integers(2, 4)
        for gy in range(IMAGE_SIZE):
            for gx in range(IMAGE_SIZE):
                dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                if dist <= r:
                    grid_hr[gy, gx] = 0.9
                elif dist <= r + 1:
                    grid_hr[gy, gx] = 0.6
                elif dist <= r + 2:
                    grid_hr[gy, gx] = 0.3
        y[i] = grid_hr.flatten()
        X[i] = np.clip(grid_hr.flatten() + rng.normal(0, noise_level + 0.2, N_FEATURES), 0, 1)

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
    return X.reshape(-1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
```

</details>

<details>
<summary>api.py</summary>

```
"""Serving API for Image Super-Resolution (DN)."""

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
from ai_core.validation import DataValidator, create_image_super_resolution_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from advanced_super_resolution.data import IMAGE_SIZE, N_CHANNELS, generate_synthetic_data
from advanced_super_resolution.model import ImageSuperResolutionDN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("IMAGE_SUPER_RESOLUTION_METRICS_PORT", "8020"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    high_res_pixels: list[float] = Field(..., min_length=64, max_length=64)

class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    high_res_pixels: str | list[float] | bool
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

_model: ImageSuperResolutionDN | None = None
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
    _metrics = MetricsCollector("image_super_resolution", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_image_super_resolution_schema())
    feature_names = [f"pixel_{i}" for i in range(64)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="image-super-resolution",
        model_version=_model_version,
        model_type="super_resolution",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="image-super-resolution", version=_model_version)

    yield
    logger.info("Shutting down image-super-resolution API")

def _load_model() -> tuple[ImageSuperResolutionDN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "image-super-resolution"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("image_super_resolution_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return ImageSuperResolutionDN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "image-super-resolution" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("image_super_resolution_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return ImageSuperResolutionDN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "image_super_resolution_model.npz"
    if npz_path.exists():
        return ImageSuperResolutionDN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/image_super_resolution_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "image_super_resolution_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return ImageSuperResolutionDN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = ImageSuperResolutionDN(
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
    title="Image Super-Resolution API",
    description="Upscales low-resolution photos and blurry security footage into sharp images",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "image_super_resolution-api",
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
                model_name="image-super-resolution",
                model_version=_model_version,
                model_type="super_resolution",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="image-super-resolution", version=_model_version)
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

def _compute_prediction(high_res_pixels: list[float]):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([high_res_pixels]).reshape(1, -1)

    if "super_resolution" in ("classification", "binary_classification"):
        validation = _validator.validate(X)
    else:
        validation = _validator.validate(X)

    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        preds = _model.predict(X)[0]
        response = PredictResponse(
            high_res_pixels=preds.flatten().tolist(),
            confidence=round(float(np.max(np.abs(preds))), 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(high_res_pixels)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Make a image super-resolution prediction."""
    return _compute_prediction(body.high_res_pixels)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Make multiple image super-resolution predictions."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for high_res_pixels in body.requests:
        predictions.append(_compute_prediction(high_res_pixels))

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
