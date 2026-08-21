# regression-house-price



Linear Regression — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Linear Regression**. The equations below
drive every forward and backward pass in the implementation.

$$\hat{y} = w \cdot x + b$$

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$

### Derivation

Starting from the hypothesis $h(x) = wx + b$, we minimize the MSE loss. Taking partial derivatives w.r.t. $w$ and $b$ and applying gradient descent yields the update rules. The learning rate $\alpha$ controls step size; too large causes divergence, too small causes slow convergence.

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

        Core transformation flow
   [ Input x ] --> ( w · x + b ) --> [ Output z ]
                       |
                  [ activation ]
                       |
                  [ prediction ]

![Linear Regression diagram](./assets/regression-house-price.png)

Interactive scatter plot with regression line, showing loss descent over iterations.

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
| `PredictRequest` | — | House price prediction request. |
| `PredictBulkRequest` | — | Bulk house price prediction request. |
| `PredictResponse` | — | Prediction response. |
| `BulkPredictResponse` | — | Bulk prediction response. |
| `DriftResponse` | — | Drift detection response. |
| `StatsResponse` | — | Model statistics response. |
| `HousePriceNN` | _he_init, _xavier_init, _forward, fit, predict, mse, rmse, mae, r2_score, evaluate, save, load, to_dict | Feedforward neural network for house price regression.  Architecture: Input -> Hidden (ReLU) -> Output (Linear)  Args:     hidden_dim: Number of neurons in the hidden layer     learning_rate: Gradient descent step size     n_iterations: Maximum number of training iterations     weight_decay: L2 regularization strength     hidden_activation: Activation for hidden layer ('relu' or 'tanh')     random_seed: Random seed for reproducibility |

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

### `HousePriceNN.fit(X, y, X_val, y_val)`

Train the neural network using batch gradient descent.

Args:
    X: Training features (n_samples, n_features)
    y: Training targets (n_samples,) — house prices
    X_val: Optional validation features
    y_val: Optional validation targets

Returns:
    self

### `HousePriceNN.predict(X)`

Predict house prices for given features.

### `HousePriceNN.evaluate(X, y)`

Compute all evaluation metrics.

### Source Files

<details>
<summary>model.py</summary>

```
"""Feedforward neural network for house price prediction (regression).

A multi-layer perceptron (MLP) with one hidden layer, trained via
backpropagation and batch gradient descent. Built from scratch with NumPy.

Architecture:
    Input (n_features) -> Hidden (hidden_dim, ReLU) -> Output (1, Linear)

Loss: Mean Squared Error
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, z)

def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (z > 0).astype(z.dtype)

def _tanh(z: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(z)

def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((y_true - y_pred) ** 2))

@dataclass
class HousePriceNN:
    """Feedforward neural network for house price regression.

    Architecture: Input -> Hidden (ReLU) -> Output (Linear)

    Args:
        hidden_dim: Number of neurons in the hidden layer
        learning_rate: Gradient descent step size
        n_iterations: Maximum number of training iterations
        weight_decay: L2 regularization strength
        hidden_activation: Activation for hidden layer ('relu' or 'tanh')
        random_seed: Random seed for reproducibility
    """

    hidden_dim: int = 32
    learning_rate: float = 0.001
    n_iterations: int = 2000
    weight_decay: float = 0.0001
    hidden_activation: Literal["relu", "tanh"] = "relu"
    random_seed: int = 42

    # Learned parameters
    input_dim: int = 0
    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None

    # Training metadata
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    y_mean_: float | None = None
    y_std_: float | None = None

    def _he_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """He initialization for ReLU networks."""
        return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))

    def _xavier_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Xavier initialization for tanh networks."""
        return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Returns: (output, hidden_activations, z1)
        """
        z1 = np.dot(X, self.W1) + self.b1

        a1 = _relu(z1) if self.hidden_activation == "relu" else _tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        return z2.flatten(), a1, z1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "HousePriceNN":
        """Train the neural network using batch gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) — house prices
            X_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()

        n_samples, n_features = X.shape
        self.input_dim = n_features

        # Normalize features
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self.mean_) / self.std_

        # Normalize targets
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if y.std() > 1e-8 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        # Normalize validation set
        X_val_norm = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.mean_) / self.std_
            self.val_loss_history = []

        # Initialize weights
        if self.hidden_activation == "relu":
            self.W1 = self._he_init(n_features, self.hidden_dim, rng)
        else:
            self.W1 = self._xavier_init(n_features, self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._xavier_init(self.hidden_dim, 1, rng)
        self.b2 = np.zeros(1)

        self.loss_history = []

        for epoch in range(self.n_iterations):
            # Forward pass
            output, a1, z1 = self._forward(X_norm)
            loss = _mse_loss(y_norm, output)

            # L2 regularization term
            l2_penalty = self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
            loss += l2_penalty

            self.loss_history.append(loss)

            # Backpropagation
            m = n_samples
            dz2 = 2 * (output - y_norm) / m  # dL/dz2
            dW2 = np.dot(a1.T, dz2.reshape(-1, 1)) + self.weight_decay * self.W2
            db2 = np.sum(dz2)

            da1 = np.dot(dz2.reshape(-1, 1), self.W2.T)
            if self.hidden_activation == "relu":
                dz1 = da1 * _relu_derivative(z1)
            else:
                dz1 = da1 * (1 - _tanh(z1) ** 2)

            dW1 = np.dot(X_norm.T, dz1) + self.weight_decay * self.W1
            db1 = np.sum(dz1, axis=0)

            # Gradient descent
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            # Track validation loss
            if X_val_norm is not None and y_val is not None and epoch % 50 == 0:
                val_output, _, _ = self._forward(X_val_norm)
                y_val_norm = (y_val - self.y_mean_) / self.y_std_
                val_loss = _mse_loss(y_val_norm, val_output)
                self.val_loss_history.append(val_loss)

            # Early stopping
            if epoch > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-7:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict house prices for given features."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        output, _, _ = self._forward(X_norm)
        return output * self.y_std_ + self.y_mean_

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return float(np.mean((self.predict(X) - y) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(self.predict(X) - y)))

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² (coefficient of determination) score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics."""
        return {
            "mse": self.mse(X, y),
            "rmse": self.rmse(X, y),
            "mae": self.mae(X, y),
            "r2": self.r2_score(X, y),
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
            hidden_dim=np.array([self.hidden_dim]),
            learning_rate=np.array([self.learning_rate]),
            n_iterations=np.array([self.n_iterations]),
            weight_decay=np.array([self.weight_decay]),
            hidden_activation=np.array([self.hidden_activation]),
            random_seed=np.array([self.random_seed]),
            mean_=self.mean_,
            std_=self.std_,
            y_mean_=np.array([self.y_mean_]) if self.y_mean_ is not None else np.array([0.0]),
            y_std_=np.array([self.y_std_]) if self.y_std_ is not None else np.array([1.0]),
            loss_history=np.array(self.loss_history),
            val_loss_history=np.array(self.val_loss_history),
            training_mode=np.array([self.training_mode]),
        )

    @classmethod
    def load(cls, path: str) -> "HousePriceNN":
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
        model.mean_ = data["mean_"]
        model.std_ = data["std_"]
        model.y_mean_ = float(data["y_mean_"].item())
        model.y_std_ = float(data["y_std_"].item())
        model.loss_history = list(data["loss_history"])
        model.val_loss_history = list(data["val_loss_history"])
        model.training_mode = str(data["training_mode"].item())

        return model

    def to_dict(self) -> dict:
        """Return model configuration as a dict."""
        return {
            "input_dim": self.input_dim,
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
"""Training pipeline for house price prediction using a feedforward neural network."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_house_price_schema

from regression_house_price.data import load_training_data, save_training_data, train_test_split
from regression_house_price.model import HousePriceNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 1000,
    hidden_dim: int = 32,
    learning_rate: float = 0.001,
    n_iterations: int = 2000,
    weight_decay: float = 0.0001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the house price prediction neural network and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    # Validate training data
    validator = DataValidator(create_house_price_schema())
    validation = validator.validate(X, y)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    # Split train/test
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

    # Save training data for reproducibility
    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.csv")

    # Train model
    model = HousePriceNN(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    # Save model
    model_path = model_dir / f"house_price_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Combined metrics for registry
    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "n_features": float(X_train.shape[1]),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="house-price-prediction",
        model_version=model_version,
        model_type="regression",
        metrics=metrics,
        parameters={
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"house_price_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={
            "framework": "numpy",
            "task": "regression",
            "model_type": "feedforward_neural_network",
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="house-price-prediction",
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
                "chart": str(model_dir / f"house_price_regression_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "regression", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow", model="house-price-prediction", version=model_version
        )

    return metrics

def _save_chart(model: HousePriceNN, output_dir: Path, version: str) -> None:
    """Save the training loss chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss (MSE + L2)")
    ax.set_title("House Price Prediction NN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"house_price_regression_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train house price prediction neural network")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "1000")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001"))
    )
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "2000")))
    parser.add_argument(
        "--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.0001"))
    )
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
"""Data generation and preprocessing for house price prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "location_score",
    "age",
    "garage",
    "lot_size",
    "year_built",
    "property_type",
    "school_rating",
]

# Location multipliers for synthetic data
LOCATIONS = ["downtown", "suburban", "riverside", "mountain", "beach"]
LOCATION_SCORES = {"downtown": 85, "suburban": 70, "riverside": 60, "mountain": 55, "beach": 90}

DEFAULT_N_SAMPLES = 1000

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic house data with features and prices.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is house prices.
    """
    rng = np.random.default_rng(random_seed)

    sqft = rng.integers(800, 5000, n_samples).astype(float)
    bedrooms = rng.integers(1, 7, n_samples).astype(float)
    bathrooms = rng.integers(1, 5, n_samples).astype(float)
    location_indices = rng.integers(0, len(LOCATIONS), n_samples)
    location_score = np.array([LOCATION_SCORES[LOCATIONS[i]] for i in location_indices]).astype(
        float
    )
    age = rng.integers(0, 80, n_samples).astype(float)
    garage = rng.integers(0, 4, n_samples).astype(float)
    lot_size = rng.integers(2000, 15000, n_samples).astype(float)
    year_built = 2024 - age
    property_type = rng.integers(0, 4, n_samples).astype(
        float
    )  # 0=single, 1=condo, 2=townhome, 3=villa
    school_rating = rng.uniform(4.0, 10.0, n_samples)

    # Price formula: base + sqft * factor + location premium + other features + noise
    price = (
        50000
        + sqft * rng.uniform(80, 150, n_samples)
        + bedrooms * 15000
        + bathrooms * 20000
        + location_score * 1500
        - age * 1500
        + garage * 8000
        + lot_size * 2
        + school_rating * 25000
        + property_type * 30000
        + rng.normal(0, 20000, n_samples)
    )
    price = np.round(price, 2)

    X = np.column_stack(
        [
            sqft,
            bedrooms,
            bathrooms,
            location_score,
            age,
            garage,
            lot_size,
            year_built,
            property_type,
            school_rating,
        ]
    )

    return X.astype(float), price

def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate house data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["price"].values.astype(float)
        return X, y

    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["price"] = y
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
```

</details>

<details>
<summary>api.py</summary>

```
"""Production serving API for house price prediction via feedforward neural network."""

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
from ai_core.validation import DataValidator, create_house_price_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from regression_house_price.data import FEATURE_NAMES
from regression_house_price.model import HousePriceNN

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("HOUSE_PRICE_METRICS_PORT", "8009")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    """House price prediction request."""

    features: list[float] = Field(..., min_length=10, max_length=10)

class PredictBulkRequest(BaseModel):
    """Bulk house price prediction request."""

    requests: list[list[float]] = Field(..., min_length=1, max_length=100)

class PredictResponse(BaseModel):
    """Prediction response."""

    predicted_price: float
    model_version: str
    training_mode: str

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

class StatsResponse(BaseModel):
    """Model statistics response."""

    n_features: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

# Global model state
_model: HousePriceNN | None = None
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
    _metrics = MetricsCollector("house_price_prediction", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_house_price_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "float" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="house-price-prediction",
        model_version=_model_version,
        model_type="regression",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="house-price-prediction", version=_model_version)

    yield

    logger.info("Shutting down house-price-prediction API")

def _load_model() -> tuple[HousePriceNN, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            hp_models = [m for m in models if m.get("model_name") == "house-price-prediction"]
            if hp_models:
                hp_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = hp_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("house_price_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return HousePriceNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "house-price-prediction" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("house_price_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return HousePriceNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "house_price_model.npz"
    if npz_path.exists():
        return HousePriceNN.load(str(npz_path)), "legacy"

    # Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/house_price_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "house_price_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return HousePriceNN.load(str(p)), "1.0.0-bundled"

    # In-memory baseline fallback
    logger.warning("No pre-existing model found on disk. Initializing baseline NN model.")
    from regression_house_price.data import generate_synthetic_data

    X_base, y_base = generate_synthetic_data(n_samples=200, random_seed=42)
    model = HousePriceNN(hidden_dim=32, learning_rate=0.001, n_iterations=500, random_seed=42)
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_csvs = [
        MODEL_DIR / "house-price-prediction" / _model_version / "training_data.csv",
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

    from regression_house_price.data import generate_synthetic_data

    X_base, _ = generate_synthetic_data(n_samples=200, random_seed=42)
    return X_base

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Feedforward neural network for predicting house prices from features",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "house-price-prediction-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "features": FEATURE_NAMES,
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
    """Kubernetes liveness/readiness probe."""
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
                model_name="house-price-prediction",
                model_version=_model_version,
                model_type="regression",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="house-price-prediction", version=_model_version
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

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return model statistics."""
    if _model is None or _model.W1 is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StatsResponse(
        n_features=_model.input_dim,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(features: list[float]) -> PredictResponse:
    """Core prediction logic shared by all prediction endpoints."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([features])

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        price = float(_model.predict(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(features)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            predicted_price=round(price, 2),
            model_version=_model_version,
            training_mode=_model.training_mode if _model else "unknown",
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Predict house price for a single property."""
    return _compute_prediction(body.features)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Predict house prices for multiple properties."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(body.requests) < 1 or len(body.requests) > 100:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 100")

    X = np.array(body.requests)

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        predictions = _model.predict(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(body.requests)
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            PredictResponse(
                predicted_price=round(float(p), 2),
                model_version=_model_version,
                training_mode=_model.training_mode if _model else "unknown",
            )
            for p in predictions
        ]
        return BulkPredictResponse(predictions=results, model_version=_model_version)
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
ai_core.validation

### How it plugs in



- **Configuration** — 12-factor config from `ai_core.config`.



- **Observability** — structured logging + Prometheus metrics are wired in automatically.



- **Validation** — input schema validation prevents bad data reaching the model.



- **Registry** — trained artifacts are versioned and registered for reproducible serving.



- **Serving** — the FastAPI app mounts shared observability middleware for tracing & metrics.

Because every example shares `ai_core`, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.
