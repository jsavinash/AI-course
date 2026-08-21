# speech-audio-recognition



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

Concrete forward-pass / update evaluation using the algorithm's own equations:

RNN hidden-state update (one timestep).
  h_{t-1}=0.30, x_t=0.50, W_hh=W_xh=0.5, b=0
  pre = 0.5*0.30 + 0.5*0.50 = 0.40
  h_t = tanh(0.40) = 0.380

### Conceptual Diagram

        Math concept (placeholder)
   [ Input x ] --> ( w · x + b ) --> [ Output z ]
                       |
                  [ activation ]
                       |
                  [ prediction ]

![Machine Learning Fundamentals diagram](./assets/speech-audio-recognition.png)

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
| `SpeechRecognitionRNN` | _to_onehot, fit, predict_proba, predict, accuracy, precision, recall, f1_score, evaluate, save, load, to_dict | RNN for speech-to-text classification (many-to-one).  Args:     n_features: Number of acoustic features per timestep (e.g., MFCCs)     seq_len: Number of timesteps in each audio sequence     n_classes: Number of recognizable words     hidden_dim: Number of hidden units     learning_rate: Gradient descent step size     n_iterations: Number of training epochs     weight_decay: L2 regularization strength     clip_value: Maximum gradient norm     random_seed: Random seed for reproducibility |

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

### `SpeechRecognitionRNN.fit(X, y, X_val, y_val)`

Train the RNN with BPTT.

Args:
    X: Audio feature sequences (n_samples, seq_len, n_features)
    y: Word class labels (n_samples,)

Returns:
    self

### `SpeechRecognitionRNN.predict(X)`

Return predicted word class indices.

### Source Files

<details>
<summary>model.py</summary>

```
"""Recurrent neural network for speech recognition.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared nn_utils.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, n_mfcc_features) -> Hidden (hidden_dim, tanh) -> Output (n_chars, softmax)

Loss: Cross-Entropy (many-to-one: predicts word at final timestep)

This is a simplified speech-to-text model that classifies an audio feature
sequence into one of a small vocabulary of spoken words.
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN

@dataclass
class SpeechRecognitionRNN:
    """RNN for speech-to-text classification (many-to-one).

    Args:
        n_features: Number of acoustic features per timestep (e.g., MFCCs)
        seq_len: Number of timesteps in each audio sequence
        n_classes: Number of recognizable words
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_features: int = 16
    seq_len: int = 20
    n_classes: int = 10
    hidden_dim: int = 32
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _to_onehot(self, indices: np.ndarray, dim: int) -> np.ndarray:
        result = np.zeros((len(indices), dim))
        for i, idx in enumerate(indices):
            result[i, int(idx)] = 1.0
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SpeechRecognitionRNN":
        """Train the RNN with BPTT.

        Args:
            X: Audio feature sequences (n_samples, seq_len, n_features)
            y: Word class labels (n_samples,)

        Returns:
            self
        """
        y_onehot = self._to_onehot(y, self.n_classes)

        self.model = SimpleRNN(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes,
            output_activation="softmax",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="cross_entropy",
        )
        self.model.fit(X, y_onehot, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted word class indices."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        classes = np.unique(np.concatenate([y, preds]))
        precisions = []
        for c in classes:
            tp = np.sum((preds == c) & (y == c))
            fp = np.sum((preds == c) & (y != c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return float(np.mean(precisions))

    def recall(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        classes = np.unique(np.concatenate([y, preds]))
        recalls = []
        for c in classes:
            tp = np.sum((preds == c) & (y == c))
            fn = np.sum((preds != c) & (y == c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return float(np.mean(recalls))

    def f1_score(self, X: np.ndarray, y: np.ndarray) -> float:
        p, r = self.precision(X, y), self.recall(X, y)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": self.accuracy(X, y),
            "precision": self.precision(X, y),
            "recall": self.recall(X, y),
            "f1": self.f1_score(X, y),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "SpeechRecognitionRNN":
        model = SimpleRNN.load(path)
        obj = cls(
            n_features=model.input_dim,
            seq_len=20,
            n_classes=model.output_dim,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "n_classes": self.n_classes,
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
"""Training pipeline for speech recognition (RNN)."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_speech_recognition_schema

from speech_audio_recognition.data import (
    N_CLASSES,
    N_FEATURES,
    SEQ_LEN,
    load_training_data,
    save_training_data,
    train_test_split,
)
from speech_audio_recognition.model import SpeechRecognitionRNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_features: int = N_FEATURES,
    seq_len: int = SEQ_LEN,
    n_classes: int = N_CLASSES,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
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

    validator = DataValidator(create_speech_recognition_schema())
    X_flat = X[:, 0, :].reshape(-1, N_FEATURES)
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

    model = SpeechRecognitionRNN(
        n_features=n_features,
        seq_len=seq_len,
        n_classes=n_classes,
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

    model_path = model_dir / f"speech_recognition_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "train_accuracy": train_metrics["accuracy"],
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
        "n_classes": float(n_classes),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="speech-recognition",
        model_version=model_version,
        model_type="rnn_sequence_classification",
        metrics=metrics,
        parameters={
            "n_features": n_features,
            "seq_len": seq_len,
            "n_classes": n_classes,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"speech_recognition_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "speech_recognition", "model_type": "simple_rnn"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="speech-recognition",
            model_version=model_version,
            metrics=metrics,
            params={
                "n_features": n_features,
                "n_classes": n_classes,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "weight_decay": weight_decay,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"speech_v{model_version}.png"),
            },
            tags={"model_type": "speech_recognition", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="speech-recognition", version=model_version)

    return metrics

def _save_chart(model: SpeechRecognitionRNN, output_dir: Path, version: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Speech Recognition RNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"speech_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train speech recognition RNN")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument(
        "--n-features", type=int, default=int(os.getenv("N_FEATURES", str(N_FEATURES)))
    )
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", str(SEQ_LEN))))
    parser.add_argument(
        "--n-classes", type=int, default=int(os.getenv("N_CLASSES", str(N_CLASSES)))
    )
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.05"))
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
        n_classes=args.n_classes,
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
"""Data loading and preprocessing for speech recognition (RNN).

Generates synthetic audio-like feature sequences (e.g., MFCC-like 16-dim vectors)
labeled with spoken word classes.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 16
SEQ_LEN = 20
N_CLASSES = 10

DEFAULT_N_SAMPLES = 500

WORD_NAMES = [
    "hello",
    "world",
    "yes",
    "no",
    "good",
    "bad",
    "up",
    "down",
    "left",
    "right",
]

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic audio feature sequences and their word labels.

    Each word class has a characteristic feature pattern (mean vector).
    The RNN must learn to classify the sequence of acoustic frames.

    Returns:
        X: (n_samples, SEQ_LEN, N_FEATURES) audio feature sequences
        y: (n_samples,) word class indices
    """
    rng = np.random.default_rng(random_seed)

    # Each class has a characteristic mean feature vector
    class_means = rng.normal(0, 1, size=(N_CLASSES, N_FEATURES))

    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = rng.integers(0, N_CLASSES)
        y[i] = label

        # Generate a sequence that gradually reveals the class
        base = class_means[label]
        for t in range(SEQ_LEN):
            # Early frames are noisy, later frames are clearer (as word is pronounced)
            noise_scale = 1.0 - (t / SEQ_LEN) * 0.4
            X[i, t] = base + rng.normal(0, noise_scale, size=N_FEATURES)

        # Normalize each sequence
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)

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
"""Serving API for speech recognition (RNN)."""

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
from ai_core.validation import DataValidator, create_speech_recognition_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from speech_audio_recognition.data import (
    N_CLASSES,
    N_FEATURES,
    SEQ_LEN,
    WORD_NAMES,
    generate_synthetic_data,
)
from speech_audio_recognition.model import SpeechRecognitionRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("SPEECH_RECOGNITION_METRICS_PORT", "8015"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    audio_features: list[list[float]] = Field(..., min_length=1, max_length=SEQ_LEN)

class PredictBulkRequest(BaseModel):
    requests: list[list[list[float]]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    word: str
    word_index: int
    confidence: float
    model_version: str
    training_mode: str

class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str

class StatsResponse(BaseModel):
    n_features: int
    seq_len: int
    n_classes: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: SpeechRecognitionRNN | None = None
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
    _metrics = MetricsCollector("speech_recognition", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_speech_recognition_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"frame_{i}" for i in range(N_FEATURES)],
        feature_types={f"frame_{i}": "float" for i in range(N_FEATURES)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="speech-recognition",
        model_version=_model_version,
        model_type="rnn_sequence_classification",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="speech-recognition", version=_model_version)

    yield
    logger.info("Shutting down speech-recognition API")

def _load_model() -> tuple[SpeechRecognitionRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            sr_models = [m for m in models if m.get("model_name") == "speech-recognition"]
            if sr_models:
                sr_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = sr_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("speech_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return SpeechRecognitionRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "speech-recognition" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("speech_recognition_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return SpeechRecognitionRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "speech_recognition_model.npz"
    if npz_path.exists():
        return SpeechRecognitionRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/speech_recognition_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "speech_recognition_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return SpeechRecognitionRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = SpeechRecognitionRNN(
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        n_classes=N_CLASSES,
        hidden_dim=32,
        learning_rate=0.05,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    # Flatten for drift detection (first timestep features)
    return X_base[:, 0, :].reshape(-1, 1) if X_base.ndim == 3 else X_base

app = FastAPI(
    title="Speech Recognition API",
    description="RNN for speech-to-text feature sequence classification",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "speech-recognition-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
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
                model_name="speech-recognition",
                model_version=_model_version,
                model_type="rnn_sequence_classification",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="speech-recognition", version=_model_version
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
    results = _drift_detector.detect_drift(_reference_data[:, :N_FEATURES], current[:, :N_FEATURES])
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
        n_classes=N_CLASSES,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(audio_features: list[list[float]]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([audio_features])

    # Validate each frame
    for frame in audio_features:
        X_flat = np.array([frame])
        validation = _validator.validate(X_flat)
        if not validation.valid:
            raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        word_idx = int(_model.predict(X)[0])
        probas = _model.predict_proba(X)[0]
        confidence = float(np.max(probas))
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        flat = [v for frame in audio_features for v in frame]
        _recent_predictions.append(flat)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            word=WORD_NAMES[word_idx] if word_idx < len(WORD_NAMES) else f"word_{word_idx}",
            word_index=word_idx,
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.audio_features)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for audio_features in body.requests:
        predictions.append(_compute_prediction(audio_features))

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
