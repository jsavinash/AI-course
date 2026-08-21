# pattern-recognition-digits



Digit Recognition / Classification — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Digit Recognition / Classification**. The equations below
drive every forward and backward pass in the implementation.

$$Z = WX + b$$

$$A = \text{ReLU}(Z)$$

$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

$$\hat{y} = \text{softmax}(Z_{out})$$

### Derivation

Feedforward networks learn hierarchical feature representations. Each layer computes a linear transformation followed by a non-linearity. Cross-entropy loss penalizes misclassification. Backpropagation computes gradients via the chain rule.

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

Interactive decision boundary; feature visualization for hidden layers; confusion matrix explorer.

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
| `PredictRequest` | — | Handwritten digit recognition request. |
| `PredictBulkRequest` | — | Bulk digit recognition request. |
| `PredictResponse` | — | Digit recognition prediction response. |
| `BulkPredictResponse` | — | Bulk digit recognition prediction response. |
| `DriftResponse` | — | Drift detection response. |
| `StatsResponse` | — | Model statistics response. |
| `DigitRecognitionNN` | _he_init, _xavier_init, _forward, fit, predict_proba, predict, predict_classes, accuracy, precision_per_class, recall_per_class, f1_per_class, macro_f1, confusion_matrix, evaluate, save, load, to_dict | Feedforward neural network for handwritten digit recognition.  Architecture: Input -> Hidden (ReLU) -> Output (Softmax)  Args:     hidden_dim: Number of neurons in the hidden layer     learning_rate: Gradient descent step size     n_iterations: Maximum number of training iterations     weight_decay: L2 regularization strength     hidden_activation: Activation for hidden layer ('relu' or 'tanh')     random_seed: Random seed for reproducibility |

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

### `DigitRecognitionNN.fit(X, y, X_val, y_val)`

Train the neural network using batch gradient descent.

Args:
    X: Training features (n_samples, n_features)
    y: Training labels (n_samples,) — digit 0-9
    X_val: Optional validation features
    y_val: Optional validation labels

Returns:
    self

### `DigitRecognitionNN.predict(X)`

Return predicted digit class for each sample.

### `DigitRecognitionNN.evaluate(X, y)`

Compute all evaluation metrics.

### Source Files

<details>
<summary>model.py</summary>

```
"""Feedforward neural network for handwritten digit recognition (multi-class classification).

A multi-layer perceptron (MLP) with one hidden layer, trained via backpropagation
and batch gradient descent using softmax cross-entropy loss. Built from scratch
with NumPy — no external ML libraries.

Architecture:
    Input (64 pixels) -> Hidden (hidden_dim, ReLU) -> Output (10 classes, Softmax)

Loss: Categorical Cross-Entropy (softmax)
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pattern_recognition_digits.data import N_CLASSES

def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, z)

def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (z > 0).astype(z.dtype)

def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax activation with numerical stability."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def _cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Categorical cross-entropy loss.

    Args:
        y_true: One-hot encoded labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)

    Returns:
        Mean cross-entropy loss
    """
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

@dataclass
class DigitRecognitionNN:
    """Feedforward neural network for handwritten digit recognition.

    Architecture: Input -> Hidden (ReLU) -> Output (Softmax)

    Args:
        hidden_dim: Number of neurons in the hidden layer
        learning_rate: Gradient descent step size
        n_iterations: Maximum number of training iterations
        weight_decay: L2 regularization strength
        hidden_activation: Activation for hidden layer ('relu' or 'tanh')
        random_seed: Random seed for reproducibility
    """

    hidden_dim: int = 64
    learning_rate: float = 0.1
    n_iterations: int = 1000
    weight_decay: float = 0.0001
    hidden_activation: Literal["relu", "tanh"] = "relu"
    random_seed: int = 42

    input_dim: int = 0
    n_classes: int = N_CLASSES
    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None

    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    val_accuracy_history: list[float] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _he_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """He initialization for ReLU networks."""
        return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))

    def _xavier_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Xavier initialization for tanh networks."""
        return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Returns: (probabilities, hidden_activations, z1)
        """
        z1 = np.dot(X, self.W1) + self.b1

        a1 = _relu(z1) if self.hidden_activation == "relu" else np.tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        probs = _softmax(z2)
        return probs, a1, z1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "DigitRecognitionNN":
        """Train the neural network using batch gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) — digit 0-9
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).flatten()

        n_samples, n_features = X.shape
        self.input_dim = n_features
        self.n_classes = N_CLASSES

        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y] = 1.0

        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self.mean_) / self.std_

        X_val_norm = None
        y_val_onehot = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.mean_) / self.std_
            y_val_flat = np.asarray(y_val, dtype=int).flatten()
            y_val_onehot = np.zeros((len(y_val_flat), self.n_classes))
            y_val_onehot[np.arange(len(y_val_flat)), y_val_flat] = 1.0
            self.val_accuracy_history = []

        if self.hidden_activation == "relu":
            self.W1 = self._he_init(n_features, self.hidden_dim, rng)
        else:
            self.W1 = self._xavier_init(n_features, self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._xavier_init(self.hidden_dim, self.n_classes, rng)
        self.b2 = np.zeros(self.n_classes)

        self.loss_history = []

        for epoch in range(self.n_iterations):
            probs, a1, z1 = self._forward(X_norm)
            loss = _cross_entropy_loss(y_onehot, probs)

            l2_penalty = self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
            loss += l2_penalty
            self.loss_history.append(loss)

            m = n_samples
            dz2 = (probs - y_onehot) / m
            dW2 = np.dot(a1.T, dz2) + self.weight_decay * self.W2
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, self.W2.T)
            if self.hidden_activation == "relu":
                dz1 = da1 * _relu_derivative(z1)
            else:
                dz1 = da1 * (1 - np.tanh(z1) ** 2)

            dW1 = np.dot(X_norm.T, dz1) + self.weight_decay * self.W1
            db1 = np.sum(dz1, axis=0)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if X_val_norm is not None and y_val_flat is not None and epoch % 50 == 0:
                val_probs, _, _ = self._forward(X_val_norm)
                val_preds = np.argmax(val_probs, axis=1)
                val_acc = float(np.mean(val_preds == y_val_flat))
                self.val_accuracy_history.append(val_acc)

            if epoch > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-7:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        probs, _, _ = self._forward(X_norm)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted digit class for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def precision_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class precision."""
        y_pred = self.predict(X)
        precisions = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            tp = np.sum((y_pred == c) & (y == c))
            fp = np.sum((y_pred == c) & (y != c))
            precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precisions

    def recall_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class recall."""
        y_pred = self.predict(X)
        recalls = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            tp = np.sum((y_pred == c) & (y == c))
            fn = np.sum((y_pred != c) & (y == c))
            recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return recalls

    def f1_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class F1 score."""
        precisions = self.precision_per_class(X, y)
        recalls = self.recall_per_class(X, y)
        f1s = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            if precisions[c] + recalls[c] > 0:
                f1s[c] = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
        return f1s

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute macro-averaged F1 score."""
        return float(np.mean(self.f1_per_class(X, y)))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        y_pred = self.predict(X)
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(y, y_pred, strict=False):
            cm[true, pred] += 1
        return cm

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics."""
        y_pred = self.predict(X)
        accuracy = float(np.mean(y_pred == y))

        precisions = self.precision_per_class(X, y)
        recalls = self.recall_per_class(X, y)

        return {
            "accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(self.f1_per_class(X, y))),
            "per_class_precision": precisions.tolist(),
            "per_class_recall": recalls.tolist(),
            "per_class_f1": self.f1_per_class(X, y).tolist(),
        }

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.W1 is None:
            raise ValueError("Cannot save untrained model")

        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=np.array([self.input_dim]),
            n_classes=np.array([self.n_classes]),
            hidden_dim=np.array([self.hidden_dim]),
            learning_rate=np.array([self.learning_rate]),
            n_iterations=np.array([self.n_iterations]),
            weight_decay=np.array([self.weight_decay]),
            hidden_activation=np.array([self.hidden_activation]),
            random_seed=np.array([self.random_seed]),
            mean_=self.mean_,
            std_=self.std_,
            loss_history=np.array(self.loss_history),
            val_accuracy_history=np.array(self.val_accuracy_history),
            training_mode=np.array([self.training_mode]),
        )

    @classmethod
    def load(cls, path: str) -> "DigitRecognitionNN":
        """Load model parameters from disk."""
        data = np.load(path)

        model = cls(
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            hidden_activation=str(data["hidden_activation"].item()),
            random_seed=int(data["random_seed"].item()),
        )

        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.input_dim = int(data["input_dim"].item())
        model.n_classes = int(data["n_classes"].item())
        model.mean_ = data["mean_"]
        model.std_ = data["std_"]
        model.loss_history = list(data["loss_history"])
        model.val_accuracy_history = list(data["val_accuracy_history"])
        model.training_mode = str(data["training_mode"].item())

        return model

    def to_dict(self) -> dict:
        """Return model configuration as a dict."""
        return {
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "hidden_activation": self.hidden_activation,
            "random_seed": self.random_seed,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
```

</details>

<details>
<summary>train.py</summary>

```
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

</details>

<details>
<summary>data.py</summary>

```
"""Data generation and preprocessing for handwritten digit recognition.

Generates synthetic digit feature vectors (8x8 = 64 features) representing
simplified handwritten digits 0-9. Each digit is represented as a flattened
8x8 grayscale image with pixel values normalized to [0, 1].
"""

from pathlib import Path

import numpy as np
import pandas as pd

GRID_SIZE = 8
N_FEATURES = GRID_SIZE * GRID_SIZE
N_CLASSES = 10

FEATURE_NAMES = [f"pixel_{i}" for i in range(N_FEATURES)]

DEFAULT_N_SAMPLES = 1000

def _create_digit_template(digit: int) -> np.ndarray:
    """Create a template 8x8 pattern for a given digit (0-9)."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    if digit == 0:
        grid[1:7, 1:7] = 1.0
        grid[2:6, 2:6] = 0.1
        grid[3:5, 3:5] = 0.0

    elif digit == 1:
        grid[1:7, 3:5] = 1.0

    elif digit == 2:
        grid[1, 1:7] = 1.0
        grid[2:4, 6] = 1.0
        grid[4, 1:7] = 1.0
        grid[5:6, 1] = 1.0
        grid[6, 1:7] = 1.0

    elif digit == 3:
        grid[1:3, 5:7] = 1.0
        grid[3, 1:6] = 1.0
        grid[4:6, 5:7] = 1.0
        grid[6, 1:6] = 1.0

    elif digit == 4:
        grid[1:5, 1] = 1.0
        grid[4, 1:5] = 1.0
        grid[1:5, 4] = 1.0
        grid[1, 4] = 1.0

    elif digit == 5:
        grid[1, 1:7] = 1.0
        grid[2:4, 1] = 1.0
        grid[1:3, 5:7] = 1.0
        grid[3, 4:6] = 1.0
        grid[4:7, 5] = 1.0
        grid[6, 1:6] = 1.0

    elif digit == 6:
        grid[1, 4:7] = 1.0
        grid[2:4, 3] = 1.0
        grid[2, 5:7] = 1.0
        grid[4, 3:6] = 1.0
        grid[5:7, 6] = 1.0
        grid[6, 3:5] = 1.0

    elif digit == 7:
        grid[1, 1:7] = 1.0
        grid[2:7, 5:6] = 1.0
        grid[3:7, 3:4] = 1.0

    elif digit == 8:
        grid[1:7, 1:3] = 1.0
        grid[1:2, 4:7] = 1.0
        grid[3:4, 4:7] = 1.0
        grid[5:6, 4:7] = 1.0
        grid[6:7, 1:3] = 1.0
        grid[1, 3:4] = 1.0
        grid[7, 3:4] = 1.0

    elif digit == 9:
        grid[1, 1:4] = 1.0
        grid[2:4, 4] = 1.0
        grid[4:5, 1:5] = 1.0
        grid[5:7, 5:6] = 1.0
        grid[6, 1:5] = 1.0

    return grid.flatten()

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic handwritten digit data.

    Each sample is an 8x8=64 pixel image of a digit (0-9) with added noise.

    Returns:
        Tuple of (X, y) where X is (n_samples, 64) and y is digit labels (0-9).
    """
    rng = np.random.default_rng(random_seed)

    templates = {d: _create_digit_template(d) for d in range(N_CLASSES)}

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        digit = rng.integers(0, N_CLASSES)
        template = templates[digit].copy()

        noise = rng.normal(0, noise_level, N_FEATURES)
        noisy = template + noise
        noisy = np.clip(noisy, 0, 1)

        X[i] = noisy
        y[i] = digit

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]

def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate digit data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c.startswith("pixel_")]
        X = df[feature_cols].values.astype(float)
        y = df["label"].values.astype(int)
        return X, y

    return generate_synthetic_data(
        n_samples=n_samples, noise_level=noise_level, random_seed=random_seed
    )

def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv(path, index=False)

def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
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

def one_hot_encode(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

def image_to_string(X: np.ndarray, threshold: float = 0.5) -> str:
    """Convert a flattened 8x8 image to an ASCII string for debug display."""
    grid = X.reshape(GRID_SIZE, GRID_SIZE)
    chars = []
    for row in grid:
        line = ""
        for val in row:
            line += "\u2588" if val >= threshold else " "
        chars.append(line)
    return "\n".join(chars)
```

</details>

<details>
<summary>api.py</summary>

```
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

</details>

## 4. Monorepo Integration

This example is a first-class consumer of the shared `packages/ai-core` library.
It reuses the following foundation modules instead of re-implementing infrastructure:

ai_core.drift
ai_core.fastapi_middleware
ai_core.logging
ai_core.metrics
ai_core.model_registry

### How it plugs in



- **Configuration** — 12-factor config from `ai_core.config`.



- **Observability** — structured logging + Prometheus metrics are wired in automatically.



- **Validation** — input schema validation prevents bad data reaching the model.



- **Registry** — trained artifacts are versioned and registered for reproducible serving.



- **Serving** — the FastAPI app mounts shared observability middleware for tracing & metrics.

Because every example shares `ai_core`, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.
