# vision-image-captioning



Image Generation (GAN/VAE/Diffusion) — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Image Generation (GAN/VAE/Diffusion)**. The equations below
drive every forward and backward pass in the implementation.

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$

### Derivation

Image generation models learn to synthesize realistic images. GANs use adversarial training between generator and discriminator. VAEs learn a structured latent space via reconstruction and KL regularization. Diffusion models iteratively denoise from Gaussian noise, offering stable training and diverse outputs.

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

![Image Generation (GAN/VAE/Diffusion) diagram](./assets/vision-image-captioning.png)

Interactive latent space explorer; denoising trajectory viewer; FID score vs training steps.

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
| `ImageCaptioningRNN` | _to_onehot_seq, _encode_image, _encode_image_batch, fit, predict, predict_proba, evaluate, save, load, to_dict | RNN for image captioning (image encoder + RNN language model).  Args:     n_pixels: Number of input image pixels (e.g., 8x8=64)     vocab_size: Size of the word vocabulary     caption_len: Number of words in each caption     hidden_dim: Number of hidden units     learning_rate: Gradient descent step size     n_iterations: Number of training epochs     weight_decay: L2 regularization strength     clip_value: Maximum gradient norm     random_seed: Random seed for reproducibility |

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

### `ImageCaptioningRNN.fit(X_images, captions, X_val, y_val)`

Train the image encoder + RNN decoder with BPTT.

Args:
    X_images: Image pixel arrays (n_samples, n_pixels)
    captions: Caption word indices (n_samples, caption_len)
    X_val: Optional validation images
    y_val: Optional validation captions

Returns:
    self

### `ImageCaptioningRNN.predict(X_images)`

Generate captions for a batch of images (greedy decoding).

### Source Files

<details>
<summary>model.py</summary>

```
"""Recurrent neural network for image captioning.

Combines a dense image encoder with a SimpleRNN (Elman network) decoder,
trained with Backpropagation Through Time (BPTT). Built from scratch with NumPy.

Architecture:
    Image (64 pixels) -> Dense (hidden_dim) -> RNN (hidden_dim, tanh) -> Output (vocab_size, softmax)

The image is encoded as a dense projection, then repeated as the first input
to a many-to-many RNN that generates a sequence of word tokens.

Loss: Cross-Entropy (many-to-many: predicts next word at each timestep)
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN

@dataclass
class ImageCaptioningRNN:
    """RNN for image captioning (image encoder + RNN language model).

    Args:
        n_pixels: Number of input image pixels (e.g., 8x8=64)
        vocab_size: Size of the word vocabulary
        caption_len: Number of words in each caption
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_pixels: int = 64
    vocab_size: int = 20
    caption_len: int = 8
    hidden_dim: int = 32
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    # Image encoder weights
    W_img: np.ndarray | None = None
    b_img: np.ndarray | None = None

    def _to_onehot_seq(self, seq: np.ndarray, dim: int) -> np.ndarray:
        seq = np.atleast_1d(seq).astype(int)
        result = np.zeros((len(seq), dim))
        result[np.arange(len(seq)), seq % dim] = 1.0
        return result

    def _encode_image(self, X_img: np.ndarray) -> np.ndarray:
        """Encode image pixels to a dense vector, then expand to RNN input dim.

        Returns (vocab_size,) one-hot-like vector (argmax-based one-hot).
        """
        if self.W_img is None:
            raise ValueError("Image encoder not initialized")
        z = X_img @ self.W_img + self.b_img
        z = np.tanh(z)
        # Project hidden representation to vocab_size-dim one-hot-like input
        onehot = np.zeros(self.vocab_size)
        onehot[int(np.argmax(z))] = 1.0
        return onehot

    def _encode_image_batch(self, X_images: np.ndarray) -> np.ndarray:
        """Encode a batch of images to one-hot-like start tokens.

        Returns: (n_samples, 1, vocab_size)
        """
        result = np.zeros((len(X_images), 1, self.vocab_size))
        for i in range(len(X_images)):
            result[i, 0] = self._encode_image(X_images[i])
        return result

    def fit(
        self,
        X_images: np.ndarray,
        captions: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ImageCaptioningRNN":
        """Train the image encoder + RNN decoder with BPTT.

        Args:
            X_images: Image pixel arrays (n_samples, n_pixels)
            captions: Caption word indices (n_samples, caption_len)
            X_val: Optional validation images
            y_val: Optional validation captions

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        # Initialize image encoder (project pixels to vocab_size-dim for one-hot encoding)
        scale = np.sqrt(1.0 / self.n_pixels)
        self.W_img = rng.normal(0, scale, (self.n_pixels, self.vocab_size))
        self.b_img = np.zeros(self.vocab_size)

        # Build RNN input sequences: [start_token, caption[:-1]]
        # The start token is the encoded image (one-hot like)
        n_samples = X_images.shape[0]
        seq_len = self.caption_len
        X_rnn = np.zeros((n_samples, seq_len, self.vocab_size))

        for i in range(n_samples):
            img_encoded = self._encode_image(X_images[i])
            # Shift captions: predict word t from image + words 0..t-1
            cap = captions[i] % self.vocab_size
            X_rnn[i, 0] = img_encoded  # first input is image
            for t in range(1, seq_len):
                prev_idx = int(cap[t - 1])
                X_rnn[i, t] = self._to_onehot_seq(np.array([prev_idx]), self.vocab_size)[0]

        # Build targets: captions one-hot
        y_onehot = np.zeros((n_samples, seq_len, self.vocab_size))
        for i in range(n_samples):
            cap = captions[i] % self.vocab_size
            for t in range(seq_len):
                y_onehot[i, t, int(cap[t])] = 1.0

        self.model = SimpleRNN(
            input_dim=self.vocab_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.vocab_size,
            output_activation="softmax",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="cross_entropy",
        )
        self.model.fit(X_rnn, y_onehot, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history

        # Fine-tune image encoder via RNN gradients (simplified: keep image encoder fixed)
        # In a full implementation, we would backprop through the image encoder too.
        return self

    def predict(self, X_images: np.ndarray) -> list[np.ndarray]:
        """Generate captions for a batch of images (greedy decoding)."""
        X_rnn_start = self._encode_image_batch(X_images)
        captions = []
        for i in range(len(X_images)):
            seq = X_rnn_start[i]  # (1, vocab_size)
            generated = []
            for _t in range(self.caption_len):
                outputs = self.model.predict_many_to_many(seq)
                next_word = int(np.argmax(outputs[-1]))
                generated.append(next_word)
                # Append next input (greedy)
                next_input = np.zeros((1, self.vocab_size))
                next_input[0, next_word] = 1.0
                seq = np.vstack([seq, next_input])
            captions.append(np.array(generated))
        return captions

    def predict_proba(self, X_images: np.ndarray) -> np.ndarray:
        """Return word probabilities for the first predicted word."""
        X_rnn_start = self._encode_image_batch(X_images)
        results = []
        for i in range(len(X_images)):
            outputs = self.model.predict_many_to_many(X_rnn_start[i])
            results.append(outputs[-1])
        return np.array(results)

    def evaluate(self, X_images: np.ndarray, captions: np.ndarray) -> dict[str, float]:
        preds = self.predict(X_images)
        correct = sum(
            np.array_equal(preds[i], captions[i] % self.vocab_size) for i in range(len(preds))
        )
        return {
            "accuracy": float(correct / max(len(preds), 1)),
            "n_samples": float(len(preds)),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)
        np.savez(
            path + ".img_encoder.npz",
            W_img=self.W_img,
            b_img=self.b_img,
        )

    @classmethod
    def load(cls, path: str) -> "ImageCaptioningRNN":
        model = SimpleRNN.load(path)

        W_img = None
        b_img = None
        try:
            img_data = np.load(path + ".img_encoder.npz")
            W_img = img_data["W_img"]
            b_img = img_data["b_img"]
        except FileNotFoundError:
            rng = np.random.default_rng(42)
            W_img = rng.normal(0, 0.1, (model.input_dim, model.input_dim))
            b_img = np.zeros(model.input_dim)

        obj = cls(
            n_pixels=model.input_dim,
            vocab_size=model.input_dim,
            caption_len=8,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        obj.W_img = W_img
        obj.b_img = b_img
        return obj

    def to_dict(self) -> dict:
        return {
            "n_pixels": self.n_pixels,
            "vocab_size": self.vocab_size,
            "caption_len": self.caption_len,
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
"""Training pipeline for image captioning (RNN)."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_image_captioning_schema

from vision_image_captioning.data import (
    CAPTION_LEN,
    N_PIXELS,
    VOCAB_SIZE,
    load_training_data,
    save_training_data,
    train_test_split,
)
from vision_image_captioning.model import ImageCaptioningRNN

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_pixels: int = N_PIXELS,
    vocab_size: int = VOCAB_SIZE,
    caption_len: int = CAPTION_LEN,
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

    validator = DataValidator(create_image_captioning_schema())
    X_flat = X[:, :N_PIXELS]
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

    model = ImageCaptioningRNN(
        n_pixels=n_pixels,
        vocab_size=vocab_size,
        caption_len=caption_len,
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

    model_path = model_dir / f"image_captioning_model_v{model_version}.npz"
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
        "vocab_size": float(vocab_size),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="image-captioning",
        model_version=model_version,
        model_type="rnn_image_captioning",
        metrics=metrics,
        parameters={
            "n_pixels": n_pixels,
            "vocab_size": vocab_size,
            "caption_len": caption_len,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"image_captioning_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "image_captioning", "model_type": "simple_rnn"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="image-captioning",
            model_version=model_version,
            metrics=metrics,
            params={
                "n_pixels": n_pixels,
                "vocab_size": vocab_size,
                "caption_len": caption_len,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "weight_decay": weight_decay,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"caption_v{model_version}.png"),
            },
            tags={"model_type": "image_captioning", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="image-captioning", version=model_version)

    return metrics

def _save_chart(model: ImageCaptioningRNN, output_dir: Path, version: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Image Captioning RNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"caption_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train image captioning RNN")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--n-pixels", type=int, default=int(os.getenv("N_PIXELS", str(N_PIXELS))))
    parser.add_argument(
        "--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", str(VOCAB_SIZE)))
    )
    parser.add_argument(
        "--caption-len", type=int, default=int(os.getenv("CAPTION_LEN", str(CAPTION_LEN)))
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
        n_pixels=args.n_pixels,
        vocab_size=args.vocab_size,
        caption_len=args.caption_len,
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
"""Data loading and preprocessing for image captioning (RNN).

Generates synthetic image pixel arrays (8x8=64) and corresponding caption word sequences.
"""

from pathlib import Path

import numpy as np

N_PIXELS = 64
VOCAB_SIZE = 20
CAPTION_LEN = 8

DEFAULT_N_SAMPLES = 500

# Simple vocabulary: objects + descriptors
VOCAB_TOKENS = [
    "start",
    "a",
    "the",
    "object",
    "bright",
    "dark",
    "round",
    "square",
    "small",
    "large",
    "circle",
    "box",
    "shape",
    "is",
    "this",
    "red",
    "blue",
    "green",
    "pattern",
    "end",
]

def _create_image_template(
    pattern_type: int, noise_level: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate an 8x8 image with a specific pattern.

    pattern_type determines the pattern (0=circle-like, 1=corner-like, etc.)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    img = np.zeros(N_PIXELS)

    if pattern_type == 0:
        # Circle in center
        img[27:29] = 0.9
        img[28:31] += 0.8
        img[27:29] += 0.4
    elif pattern_type == 1:
        # Corner
        img[0:3] = 0.9
        img[8:11] = 0.8
    elif pattern_type == 2:
        # Horizontal bar
        img[28:36] = 0.9
    elif pattern_type == 3:
        # Vertical bar
        img[::8] = 0.9
    elif pattern_type == 4:
        # Diagonal
        img[::9] = 0.9
    else:
        img.flat[rng.integers(0, N_PIXELS, size=10)] = 0.9

    img = np.clip(img + rng.normal(0, noise_level, N_PIXELS), 0, 1)
    return img

def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their caption sequences.

    Each image has one of 5 basic patterns, and the caption describes it.

    Returns:
        X_images: (n_samples, N_PIXELS) image pixel arrays
        captions: (n_samples, CAPTION_LEN) word token indices
    """
    rng = np.random.default_rng(random_seed)

    # Pattern-to-caption mapping
    pattern_captions = {
        0: [0, 1, 3, 10, 2, 4, 5, 19],  # start a object circle the bright blue end
        1: [0, 1, 3, 11, 2, 6, 12, 19],  # start a object box the round shape end
        2: [0, 1, 3, 2, 7, 13, 14, 19],  # start a object the square is this end
        3: [0, 1, 3, 11, 2, 8, 15, 19],  # start a object box the small red end
        4: [0, 1, 3, 10, 2, 9, 16, 19],  # start a object circle the large blue end
    }

    X_images = np.zeros((n_samples, N_PIXELS))
    captions = np.zeros((n_samples, CAPTION_LEN), dtype=int)

    for i in range(n_samples):
        pattern = rng.integers(0, 5)
        X_images[i] = _create_image_template(pattern, noise_level, rng)
        captions[i] = pattern_captions[pattern]

    # Shuffle
    perm = rng.permutation(n_samples)
    return X_images[perm], captions[perm]

def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(
        n_samples=n_samples, noise_level=noise_level, random_seed=random_seed
    )

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
"""Serving API for image captioning (RNN)."""

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
from ai_core.validation import DataValidator, create_image_captioning_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from vision_image_captioning.data import (
    CAPTION_LEN,
    N_PIXELS,
    VOCAB_SIZE,
    VOCAB_TOKENS,
    generate_synthetic_data,
)
from vision_image_captioning.model import ImageCaptioningRNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("IMAGE_CAPTIONING_METRICS_PORT", "8019"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class PredictRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=N_PIXELS, max_length=N_PIXELS)

class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)

class PredictResponse(BaseModel):
    caption_tokens: list[int]
    caption: str
    model_version: str
    training_mode: str

class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str

class StatsResponse(BaseModel):
    n_pixels: int
    vocab_size: int
    caption_len: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str

_model: ImageCaptioningRNN | None = None
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
    _metrics = MetricsCollector("image_captioning", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_image_captioning_schema())
    _drift_detector = DriftDetector(
        feature_names=[f"pixel_{i}" for i in range(N_PIXELS)],
        feature_types={f"pixel_{i}": "float" for i in range(N_PIXELS)},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="image-captioning",
        model_version=_model_version,
        model_type="rnn_image_captioning",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="image-captioning", version=_model_version)

    yield
    logger.info("Shutting down image-captioning API")

def _load_model() -> tuple[ImageCaptioningRNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            ic_models = [m for m in models if m.get("model_name") == "image-captioning"]
            if ic_models:
                ic_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = ic_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("image_captioning_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return ImageCaptioningRNN.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "image-captioning" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("image_captioning_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return ImageCaptioningRNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "image_captioning_model.npz"
    if npz_path.exists():
        return ImageCaptioningRNN.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/image_captioning_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "image_captioning_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return ImageCaptioningRNN.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline RNN model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = ImageCaptioningRNN(
        n_pixels=N_PIXELS,
        vocab_size=VOCAB_SIZE,
        caption_len=CAPTION_LEN,
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
    title="Image Captioning API",
    description="RNN for generating descriptive captions from image pixel sequences",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "image-captioning-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_pixels": N_PIXELS,
        "vocab_size": VOCAB_SIZE,
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
                model_name="image-captioning",
                model_version=_model_version,
                model_type="rnn_image_captioning",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="image-captioning", version=_model_version)
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
            "total_features": N_PIXELS,
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
        n_pixels=N_PIXELS,
        vocab_size=VOCAB_SIZE,
        caption_len=CAPTION_LEN,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )

def _compute_prediction(pixels: list[float]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([pixels])
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        captions = _model.predict(X)
        caption_tokens = captions[0].tolist()
        caption_str = " ".join(VOCAB_TOKENS[t % len(VOCAB_TOKENS)] for t in caption_tokens)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        _recent_predictions.append(pixels)
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            caption_tokens=[int(t) for t in caption_tokens],
            caption=caption_str,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.pixels)

@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for pixels in body.requests:
        predictions.append(_compute_prediction(pixels))

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
