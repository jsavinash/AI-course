# time-series-weather



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
| `StatsResponse` | — |  |
| `WeatherForecastingRNN` | fit, predict, predict_proba, mse, rmse, mae, r2_score_per_feature, evaluate, save, load, to_dict | RNN for multi-feature weather regression (many-to-one).  Args:     n_features: Number of weather features per timestep     seq_len: Number of timesteps in input sequences     hidden_dim: Number of hidden units     learning_rate: Gradient descent step size     n_iterations: Number of training epochs     weight_decay: L2 regularization strength     clip_value: Maximum gradient norm     random_seed: Random seed for reproducibility |

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

### `WeatherForecastingRNN.fit(X, y, X_val, y_val)`

Train the RNN with BPTT.

Args:
    X: Weather feature sequences (n_samples, seq_len, n_features)
    y: Target weather vectors (n_samples, n_features) — next-day values

Returns:
    self

### `WeatherForecastingRNN.predict(X)`

Predict next-day weather vectors for a batch of sequences.

### Source Files

<details>
<summary>model.py</summary>

```
"""Recurrent neural network for weather forecasting.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared nn_utils.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, n_features) -> Hidden (hidden_dim, tanh) -> Output (n_features, linear)

Loss: Mean Squared Error (many-to-one: predicts next-day weather vector)

The model learns temporal patterns in a sequence of weather measurements
(temperature, humidity, pressure, wind-speed, precipitation) and predicts
the weather vector for the next day.
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN

@dataclass
class WeatherForecastingRNN:
    """RNN for multi-feature weather regression (many-to-one).

    Args:
        n_features: Number of weather features per timestep
        seq_len: Number of timesteps in input sequences
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_features: int = 5
    seq_len: int = 30
    hidden_dim: int = 32
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "WeatherForecastingRNN":
        """Train the RNN with BPTT.

        Args:
            X: Weather feature sequences (n_samples, seq_len, n_features)
            y: Target weather vectors (n_samples, n_features) — next-day values

        Returns:
            self
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Normalize features per-feature
        all_X = X.reshape(-1, self.n_features)
        self.feature_mean_ = all_X.mean(axis=0)
        self.feature_std_ = np.where(all_X.std(axis=0) < 1e-8, 1.0, all_X.std(axis=0))
        X_norm = (X - self.feature_mean_) / self.feature_std_

        # Normalize targets per-feature
        y_mean = y.mean(axis=0)
        y_std = np.where(y.std(axis=0) < 1e-8, 1.0, y.std(axis=0))
        y_norm = (y - y_mean) / y_std

        self.model = SimpleRNN(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_features,
            output_activation="linear",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="mse",
        )
        self.model.fit(X_norm, y_norm, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history

        # Store normalization params for prediction
        self._y_mean = y_mean
        self._y_std = y_std
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day weather vectors for a batch of sequences."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.feature_mean_) / self.feature_std_
        preds_norm = self.model.predict_proba(X_norm)
        return preds_norm * self._y_std + self._y_mean

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return float(np.mean((self.predict(X) - y) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return float(np.mean(np.abs(self.predict(X) - y)))

    def r2_score_per_feature(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "mse": self.mse(X, y),
            "rmse": self.rmse(X, y),
            "mae": self.mae(X, y),
            "r2": self.r2_score_per_feature(X, y),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)
        np.savez(
            path + ".norm.npz",
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            y_mean=self._y_mean,
            y_std=self._y_std,
        )

    @classmethod
    def load(cls, path: str) -> "WeatherForecastingRNN":
        model = SimpleRNN.load(path)

        feature_mean = None
        feature_std = None
        y_mean = None
        y_std = None
        try:
            norm_data = np.load(path + ".norm.npz")
            feature_mean = norm_data["feature_mean"]
            feature_std = norm_data["feature_std"]
            y_mean = norm_data["y_mean"]
            y_std = norm_data["y_std"]
        except FileNotFoundError:
            pass

        obj = cls(
            n_features=model.input_dim,
            seq_len=20,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        obj.feature_mean_ = feature_mean
        obj.feature_std_ = feature_std
        obj._y_mean = y_mean
        obj._y_std = y_std
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
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
"""Training pipeline for weather forecasting (RNN)."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_weather_forecasting_schema

from time_series_weather.data import (
    N_FEATURES,
    SEQ_LEN,
    load_training_data,
    save_training_data,
    train_test_split,
)
from time_series_weather.model import WeatherForecastingRNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_features: int = N_FEATURES,
    seq_len: int = SEQ_LEN,
    hidden_dim: int = 32,
    learning_rate: float = 0.01,
    n_iterations: int = 300,
    weight_decay: float = 0.001,
    clip_value: float = 5.0,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_weather_forecasting_schema())
    X_flat = X[:, 0, :].reshape(-1, n_features)
    validation = validator.validate(X_flat)
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

    model = WeatherForecastingRNN(
        n_features=n_features,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        clip_value=clip_value,
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
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"weather_forecasting_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "train_mse": train_metrics["mse"],
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
        "n_features": float(n_features),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="weather-forecasting",
        model_version=model_version,
        model_type="rnn_sequence_regression",
        metrics=metrics,
        parameters={
            "n_features": n_features,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"weather_forecasting_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "weather_forecasting", "model_type": "simple_rnn"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="weather-forecasting",
            model_version=model_version,
            metrics=metrics,
            params={
                "n_features": n_features,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "weight_decay": weight_decay,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"weather_v{model_version}.png"),
            },
            tags={"model_type": "weather_forecasting", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow", model="weather-forecasting", version=model_version
        )

    return metrics

def _save_chart(model: WeatherForecastingRNN, output_dir: Path, version: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Weather Forecasting RNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"weather_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train weather forecasting RNN")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument(
        "--n-features", type=int, default=int(os.getenv("N_FEATURES", str(N_FEATURES)))
    )
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", str(SEQ_LEN))))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.01"))
    )
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "300")))
    parser.add_argument(
        "--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001"))
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
        n_features=args.n_features,
        seq_len=args.seq_len,
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
"""Data loading and preprocessing for weather forecasting (RNN).

Generates synthetic weather time-series feature sequences for next-day forecasting.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 5  # temperature, humidity, pressure, wind_speed, precipitation
SEQ_LEN = 30  # days of history

DEFAULT_N_SAMPLES = 500

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic weather feature sequences.

    Each sample is a sequence of SEQ_LEN days, each with 5 features:
    temperature, humidity, pressure, wind_speed, precipitation.

    The target is the weather vector for the next day, derived from
    temporal patterns (seasonal trend + autocorrelation + noise).

    Returns:
        X: (n_samples, SEQ_LEN, N_FEATURES) weather feature sequences
        y: (n_samples, N_FEATURES) next-day weather vector
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES))
    y = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        # Random seasonal phase
        phase = rng.uniform(0, 2 * np.pi)

        # Generate base pattern for temperature (seasonal + trend)
        temps = np.zeros(SEQ_LEN + 1)
        for t in range(SEQ_LEN + 1):
            seasonal = np.sin(phase + t * 0.2) * 10 + 15
            trend = t * 0.05
            temps[t] = seasonal + trend + rng.normal(0, 2)

        # Humidity (anti-correlated with temperature)
        humidity = 80 - (temps[: SEQ_LEN + 1] - 10) * 0.5 + rng.normal(0, 3, SEQ_LEN + 1)
        humidity = np.clip(humidity, 0, 100)

        # Pressure (slowly varying)
        pressure_base = rng.uniform(990, 1030)
        pressure = pressure_base + np.cumsum(rng.normal(0, 0.3, SEQ_LEN + 1))

        # Wind speed
        wind = np.abs(rng.normal(8, 4, SEQ_LEN + 1)) + np.sin(np.arange(SEQ_LEN + 1) * 0.3) * 3

        # Precipitation (correlated with humidity)
        precip = np.where(
            humidity > 75, rng.uniform(0.5, 3.0, SEQ_LEN + 1), rng.uniform(0, 0.3, SEQ_LEN + 1)
        )

        # Fill X (first SEQ_LEN days)
        X[i, :, 0] = temps[:SEQ_LEN]  # temperature
        X[i, :, 1] = humidity[:SEQ_LEN]  # humidity
        X[i, :, 2] = pressure[:SEQ_LEN]  # pressure
        X[i, :, 3] = wind[:SEQ_LEN]  # wind speed
        X[i, :, 4] = precip[:SEQ_LEN]  # precipitation

        # Target: next day (day SEQ_LEN)
        y[i, 0] = temps[SEQ_LEN]
        y[i, 1] = humidity[SEQ_LEN]
        y[i, 2] = pressure[SEQ_LEN]
        y[i, 3] = wind[SEQ_LEN]
        y[i, 4] = precip[SEQ_LEN]

    # Shuffle
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
"""Serving API for weather forecasting (RNN)."""

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
from ai_core.validation import DataValidator, create_weather_forecasting_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from time_series_weather.data import N_FEATURES, SEQ_LEN, generate_synthetic_data
from time_series_weather.model import WeatherForecastingRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("WEATHER_FORECASTING_METRICS_PORT", "8018"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

FEATURE_NAMES = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]

class PredictRequest(BaseModel):
    feature_sequences: list[list[float]] = Field(..., min_length=1, max_length=SEQ_LEN)

class PredictBulkRequest(BaseModel):
    requests: list[list[list[float]]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    predicted_weather: dict
    model_version: str
    training_mode: str

class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str

class StatsResponse(BaseModel):
    n_features: int
    seq_len: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: WeatherForecastingRNN | None = None
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
    _metrics = MetricsCollector("weather_forecasting", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_weather_forecasting_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"step_{i}" for i in range(N_FEATURES)],
        feature_types={f"step_{i}": "float" for i in range(N_FEATURES)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="weather-forecasting",
        model_version=_model_version,
        model_type="rnn_sequence_regression",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="weather-forecasting", version=_model_version)

    yield
    logger.info("Shutting down weather-forecasting API")

def _load_model() -> tuple[WeatherForecastingRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            wf_models = [m for m in models if m.get("model_name") == "weather-forecasting"]
            if wf_models:
                wf_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = wf_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("weather_forecasting_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return WeatherForecastingRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "weather-forecasting" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("weather_forecasting_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return WeatherForecastingRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "weather_forecasting_model.npz"
    if npz_path.exists():
        return WeatherForecastingRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/weather_forecasting_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "weather_forecasting_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return WeatherForecastingRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = WeatherForecastingRNN(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        learning_rate=0.01,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base[:, 0, :].reshape(-1, N_FEATURES)

app = FastAPI(
    title="Weather Forecasting API",
    description="RNN for next-day weather prediction from temporal feature sequences",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "weather-forecasting-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
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
                model_name="weather-forecasting",
                model_version=_model_version,
                model_type="rnn_sequence_regression",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="weather-forecasting", version=_model_version
        )
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e

@app.get("/drift")
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
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(feature_sequences: list[list[float]]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([feature_sequences])
    X_flat = X.reshape(-1, N_FEATURES)
    validation = _validator.validate(X_flat)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        pred = _model.predict(X)[0]  # (n_features,)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        flat = [v for frame in feature_sequences for v in frame]
        _recent_predictions.append(flat)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        weather = {name: round(float(pred[i]), 2) for i, name in enumerate(FEATURE_NAMES)}
        return PredictResponse(
            predicted_weather=weather,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.feature_sequences)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for feature_seq in body.requests:
        predictions.append(_compute_prediction(feature_seq))

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
