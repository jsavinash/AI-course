# image-generation



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
| `GenerateImageRequest` | — |  |
| `GenerateImageResponse` | — |  |
| `StatsResponse` | — |  |
| `ImageTokenizer` | __post_init__, encode, decode, batch_encode |  |
| `MultiHeadAttention` | __post_init__, init_weights, _split_heads, _combine_heads, forward, backward, update_params |  |
| `AddNorm` | init_params, forward, backward |  |
| `FeedForward` | init_weights, forward, backward, update_params |  |
| `TransformerBlock` | __post_init__, forward |  |
| `TextConditioning` | _init, forward |  |
| `VariationalAutoencoder` | _init, encode, reparameterize, decode |  |
| `DiffusionModel` | _init, forward_process, predict_noise, reverse_process |  |
| `ImageGenerationModel` | _init, encode_text, encode_image, decode_latent, generate_from_text, fit, evaluate, save, load, to_dict |  |

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

No docstring-annotated key methods.

### Source Files

<details>
<summary>model.py</summary>

```
"""Image Generation implementation from scratch using NumPy.

Architecture:
    1. ImageTokenizer: Encodes text prompts and decodes image latents
    2. VariationalAutoencoder: Encoder-decoder with latent space compression
    3. DiffusionModel: Denoising diffusion probabilistic model (DDPM)
    4. TextConditioning: Projects text embeddings into latent space
    5. ImageGenerationModel: Main model orchestrating text-to-image generation

Core capabilities:
    - Text-to-Image: Generate images from text descriptions
    - Latent Space Compression: Compress images for faster processing
    - Diffusion Process: Iteratively denoise latents to form crisp images
    - Text Conditioning: Use transformer embeddings to guide generation

Args:
    vocab_size: vocabulary size for text tokens
    d_model: text embedding dimension
    img_size: output image size (img_size x img_size)
    latent_dim: latent space dimension
    n_diffusion_steps: number of denoising steps
    n_heads: number of attention heads
    n_layers: number of transformer layers
    random_seed: random seed for reproducibility
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    attn = softmax(scores, axis=-1)
    return attn @ V

@dataclass
class ImageTokenizer:
    vocab_size: int = 1000
    max_seq_len: int = 64
    random_seed: int = 42

    token_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    id_to_token: dict[int, str] = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<BOS>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        visual_words = ["image", "photo", "picture", "portrait", "landscape", "scene", "art", "painting",
                        "sketch", "drawing", "digital", "3d", "render", "style", "color", "light", "dark",
                        "bright", "sunset", "sunrise", "night", "day", "forest", "ocean", "mountain",
                        "city", "street", "building", "car", "person", "face", "cat", "dog", "bird",
                        "flower", "tree", "sky", "cloud", "water", "fire", "abstract", "realistic"]
        for i, word in enumerate(visual_words):
            idx = len(special_tokens) + i
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word

    def encode(self, text: str) -> list[int]:
        tokens = text.lower().split()
        encoded = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        return encoded

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        max_len = max(len(self.encode(t)) for t in texts)
        max_len = min(max_len, self.max_seq_len)
        batch = np.full((len(texts), max_len), self.token_to_id["<PAD>"], dtype=int)
        for i, text in enumerate(texts):
            encoded = self.encode(text)
            batch[i, :len(encoded)] = encoded
        return batch

@dataclass
class MultiHeadAttention:
    d_model: int = 256
    n_heads: int = 8
    random_seed: int = 42
    trainable: bool = True

    d_k: int = field(init=False)
    W_q: np.ndarray | None = None
    W_k: np.ndarray | None = None
    W_v: np.ndarray | None = None
    W_o: np.ndarray | None = None
    dw_q: np.ndarray | None = None
    dw_k: np.ndarray | None = None
    dw_v: np.ndarray | None = None
    dw_o: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W_q = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_k = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_v = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_o = rng.normal(0, scale, (self.d_model, self.d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, _, seq_len, _ = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.W_q is None:
            self.init_weights()
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Q_split = self._split_heads(Q)
        K_split = self._split_heads(K)
        V_split = self._split_heads(V)
        if mask is not None and mask.ndim == 2:
            mask = mask[np.newaxis, np.newaxis, :, :].astype(bool)
        elif mask is not None and mask.ndim == 3:
            mask = mask[:, np.newaxis, :, :].astype(bool)
        out = np.zeros_like(Q_split)
        for h in range(self.n_heads):
            q_h = Q_split[:, h, :, :]
            k_h = K_split[:, h, :, :]
            v_h = V_split[:, h, :, :]
            out[:, h, :, :] = scaled_dot_product_attention(q_h, k_h, v_h, mask)
        out = self._combine_heads(out)
        result = out @ self.W_o
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "out": out, "result": result}
        return result

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dw_o = c["out"].T @ dout
        return dout @ self.W_o.T @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None or not self.trainable:
            return
        self.W_q -= lr * (self.dw_q + weight_decay * self.W_q) if self.dw_q is not None else self.W_q
        self.W_k -= lr * (self.dw_k + weight_decay * self.W_k) if self.dw_k is not None else self.W_k
        self.W_v -= lr * (self.dw_v + weight_decay * self.W_v) if self.dw_v is not None else self.W_v
        self.W_o -= lr * (self.dw_o + weight_decay * self.W_o) if self.dw_o is not None else self.W_o

@dataclass
class AddNorm:
    d_model: int = 256
    random_seed: int = 42
    trainable: bool = True

    gamma: np.ndarray | None = None
    beta: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_params(self) -> None:
        self.gamma = np.ones(self.d_model)
        self.beta = np.zeros(self.d_model)

    def forward(self, residual: np.ndarray, output: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            self.init_params()
        combined = residual + output
        normed = layer_norm(combined, self.gamma, self.beta)
        self._cache = {"residual": residual, "output": output, "combined": combined, "normed": normed}
        return normed

    def backward(self, dout: np.ndarray) -> np.ndarray:
        eps = 1e-5
        c = self._cache
        x = c["combined"]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]
        return dx

@dataclass
class FeedForward:
    d_model: int = 256
    d_ff: int = 1024
    random_seed: int = 42
    trainable: bool = True

    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None
    dw1: np.ndarray | None = None
    db1: np.ndarray | None = None
    dw2: np.ndarray | None = None
    db2: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W1 = rng.normal(0, scale, (self.d_model, self.d_ff))
        self.b1 = np.zeros(self.d_ff)
        self.W2 = rng.normal(0, scale, (self.d_ff, self.d_model))
        self.b2 = np.zeros(self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W1 is None:
            self.init_weights()
        z1 = x @ self.W1 + self.b1
        a1 = gelu(z1)
        out = a1 @ self.W2 + self.b2
        self._cache = {"x": x, "z1": z1, "a1": a1}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dw2 = c["a1"].T @ dout
        self.db2 = np.sum(dout, axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * (c["a1"] * (1 - c["a1"]))
        self.dw1 = c["x"].T @ dz1
        self.db1 = np.sum(dz1, axis=0)
        return dz1 @ self.W1.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W1 is None or not self.trainable:
            return
        self.W1 -= lr * (self.dw1 + weight_decay * self.W1)
        self.b1 -= lr * self.db1
        self.W2 -= lr * (self.dw2 + weight_decay * self.W2)
        self.b2 -= lr * self.db2

@dataclass
class TransformerBlock:
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    random_seed: int = 42
    trainable: bool = True

    self_attn: MultiHeadAttention | None = None
    add_norm1: AddNorm | None = None
    ffn: FeedForward | None = None
    add_norm2: AddNorm | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.n_heads, self.random_seed, trainable=self.trainable)
        self.add_norm1 = AddNorm(self.d_model, self.random_seed + 1, trainable=self.trainable)
        self.ffn = FeedForward(self.d_model, self.d_ff, self.random_seed + 2, trainable=self.trainable)
        self.add_norm2 = AddNorm(self.d_model, self.random_seed + 3, trainable=self.trainable)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.self_attn.forward(x, mask)
        x = self.add_norm1.forward(x, attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.add_norm2.forward(x, ffn_out)
        return x

@dataclass
class TextConditioning:
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 64
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    layers: list[TransformerBlock] = field(default_factory=list, repr=False)
    W_proj: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)
        self.layers = [
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.random_seed + i, trainable=True)
            for i in range(self.n_layers)
        ]
        self.W_proj = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.d_model, self.d_model))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        seq_len = token_ids.shape[1] if token_ids.ndim > 1 else 1
        embedded = self.embedding[token_ids] * np.sqrt(self.d_model)
        if token_ids.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(token_ids)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]
        for layer in self.layers:
            embedded = layer.forward(embedded)
        pooled = np.mean(embedded, axis=1, keepdims=True)
        conditioning = pooled @ self.W_proj
        self._cache = {"embedded": embedded, "pooled": pooled, "conditioning": conditioning}
        return conditioning

def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                pe[pos, i] = np.sin(angle)
            else:
                pe[pos, i] = np.cos(angle)
    return pe

@dataclass
class VariationalAutoencoder:
    img_size: int = 32
    latent_dim: int = 64
    random_seed: int = 42

    encoder_w: np.ndarray | None = None
    encoder_b: np.ndarray | None = None
    mu_w: np.ndarray | None = None
    mu_b: np.ndarray | None = None
    logvar_w: np.ndarray | None = None
    logvar_b: np.ndarray | None = None
    decoder_w: np.ndarray | None = None
    decoder_b: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        input_dim = self.img_size * self.img_size * 3
        hidden_dim = 256
        scale_enc = np.sqrt(2.0 / input_dim)
        self.encoder_w = rng.normal(0, scale_enc, (input_dim, hidden_dim))
        self.encoder_b = np.zeros(hidden_dim)
        scale_lat = np.sqrt(2.0 / hidden_dim)
        self.mu_w = rng.normal(0, scale_lat, (hidden_dim, self.latent_dim))
        self.mu_b = np.zeros(self.latent_dim)
        self.logvar_w = rng.normal(0, scale_lat, (hidden_dim, self.latent_dim))
        self.logvar_b = np.zeros(self.latent_dim)
        scale_dec = np.sqrt(2.0 / self.latent_dim)
        self.decoder_w = rng.normal(0, scale_dec, (self.latent_dim, hidden_dim))
        self.decoder_b = np.zeros(hidden_dim)
        self._decoder_out_W = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, input_dim))
        self._decoder_out_b = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.encoder_w is None:
            self._init()
        h = gelu(x @ self.encoder_w + self.encoder_b)
        mu = h @ self.mu_w + self.mu_b
        logvar = h @ self.logvar_w + self.logvar_b
        self._cache = {"x": x, "h": h, "mu": mu, "logvar": logvar}
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        std = np.exp(0.5 * logvar)
        eps = rng.normal(0, 1, size=std.shape)
        return mu + eps * std
... (truncated) ...
```

</details>

<details>
<summary>train.py</summary>

```
"""Training pipeline for Image Generation."""

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from image_generation.data import load_image_dataset, save_dataset, train_test_split_images
from image_generation.model import ImageGenerationModel

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    img_size: int = 32,
    latent_dim: int = 64,
    model_id: str = "image-generation-v1",
    n_diffusion_steps: int = 1000,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading image dataset", n_samples=n_samples, img_size=img_size)
    images, prompts = load_image_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, prompts_train, prompts_test = train_test_split_images(images, prompts, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(images, prompts, model_dir / "training_data.npz")

    model = ImageGenerationModel(
        model_id=model_id,
        img_size=img_size,
        latent_dim=latent_dim,
        n_diffusion_steps=n_diffusion_steps,
        random_seed=random_seed,
    )
    model._init()

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    metrics = model.fit(X_train_flat, np.zeros(len(X_train_flat)), n_iterations=10)
    logger.info("Training finished", metrics=metrics)

    eval_metrics = model.evaluate(X_test_flat, np.zeros(len(X_test_flat)))
    logger.info("Evaluation metrics", metrics=eval_metrics)

    model_path = model_dir / f"image_generation_v{model_version}.npz"
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        "img_size": float(img_size),
        "latent_dim": float(latent_dim),
        "n_diffusion_steps": float(n_diffusion_steps),
        "n_samples": float(n_samples),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="image-generation",
        model_version=model_version,
        model_type="generative",
        metrics=combined_metrics,
        parameters={
            "model_id": model_id,
            "img_size": img_size,
            "latent_dim": latent_dim,
            "n_diffusion_steps": n_diffusion_steps,
            "n_samples": n_samples,
            "random_seed": random_seed,
        },
        artifacts={f"image_generation_v{model_version}.npz": model_path, "training_data.npz": model_dir / "training_data.npz"},
        tags={"framework": "numpy", "task": "image_generation", "model_type": "ImageGeneration"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="image-generation",
            model_version=model_version,
            metrics=combined_metrics,
            params={"model_id": model_id, "img_size": img_size, "latent_dim": latent_dim, "n_diffusion_steps": n_diffusion_steps, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "image_generation", "framework": "numpy"},
        )

    return combined_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Image Generation Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--img-size", type=int, default=int(os.getenv("IMG_SIZE", "32")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "64")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "image-generation-v1"))
    parser.add_argument("--n-diffusion-steps", type=int, default=int(os.getenv("N_DIFFUSION_STEPS", "1000")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
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
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        model_id=args.model_id,
        n_diffusion_steps=args.n_diffusion_steps,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
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
"""Data loading and preprocessing for Image Generation."""

from pathlib import Path

import numpy as np

DEFAULT_N_SAMPLES = 500
DEFAULT_IMG_SIZE = 32
DEFAULT_LATENT_DIM = 64

def generate_synthetic_images(n_samples: int = DEFAULT_N_SAMPLES, img_size: int = DEFAULT_IMG_SIZE, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    images = rng.uniform(0, 1, size=(n_samples, img_size, img_size, 3))
    labels = rng.integers(0, 10, size=n_samples)
    return images, labels

def generate_synthetic_prompts(n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> list[str]:
    rng = np.random.default_rng(random_seed)
    templates = [
        "a photo of a {object} in {setting}",
        "a painting of {object} with {style} style",
        "a digital render of {object} in {color} lighting",
        "an abstract image of {object} and {object}",
    ]
    objects = ["cat", "dog", "bird", "car", "city", "forest", "ocean", "mountain", "flower", "sunset"]
    settings = ["a forest", "the ocean", "a city", "a mountain", "space", "a desert", "a garden", "the sky"]
    styles = ["modern", "classic", "futuristic", "vintage", "minimalist", "impressionist", "surreal"]
    colors = ["bright", "dark", "warm", "cool", "neon", "soft", "vivid"]
    prompts = []
    for _ in range(n_samples):
        template = rng.choice(templates)
        prompt = template.format(
            object=rng.choice(objects),
            setting=rng.choice(settings),
            style=rng.choice(styles),
            color=rng.choice(colors),
        )
        prompts.append(prompt)
    return prompts

def load_image_dataset(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> tuple[np.ndarray, list[str]]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["images"], data["prompts"].tolist()
    images, _ = generate_synthetic_images(n_samples=n_samples, random_seed=random_seed)
    prompts = generate_synthetic_prompts(n_samples=n_samples, random_seed=random_seed)
    return images, prompts

def train_test_split_images(images: np.ndarray, prompts: list[str], test_size: float = 0.2, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    n = len(images)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return images[train_idx], images[test_idx], [prompts[i] for i in train_idx], [prompts[i] for i in test_idx]

def save_dataset(images: np.ndarray, prompts: list[str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images, prompts=np.array(prompts, dtype=object))
```

</details>

<details>
<summary>api.py</summary>

```
"""Serving API for Image Generation."""

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

from image_generation.data import DEFAULT_IMG_SIZE, DEFAULT_LATENT_DIM
from image_generation.model import ImageGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("IMAGE_GENERATION_METRICS_PORT", "9021"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n_steps: int = Field(default=50, ge=1, le=200)

class GenerateImageResponse(BaseModel):
    image_shape: tuple[int, int, int]
    prompt: str
    model_version: str

class StatsResponse(BaseModel):
    model_id: str
    img_size: int
    latent_dim: int
    n_diffusion_steps: int
    model_version: str

_model: ImageGenerationModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("image_generation", port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f"latent_{i}" for i in range(DEFAULT_LATENT_DIM)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="image-generation",
        model_version=_model_version,
        model_type="generative",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="image-generation", version=_model_version)

    yield
    logger.info("Shutting down image-generation API")

def _load_model() -> tuple[ImageGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            ig_models = [m for m in models if m.get("model_name") == "image-generation"]
            if ig_models:
                ig_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = ig_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("image_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return ImageGenerationModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "image-generation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("image_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return ImageGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "image_generation.npz"
    if npz_path.exists():
        return ImageGenerationModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/image_generation_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "image_generation_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return ImageGenerationModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    model = ImageGenerationModel(model_id="baseline", img_size=DEFAULT_IMG_SIZE, latent_dim=DEFAULT_LATENT_DIM)
    model._init()
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    from image_generation.data import generate_synthetic_images
    images, _ = generate_synthetic_images(n_samples=100, random_seed=42)
    return images.reshape(100, -1).astype(float)

app = FastAPI(
    title="Image Generation API",
    description="Diffusion model-based text-to-image generation with VAE latent compression",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "image-generation-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "endpoints": {
            "health": "/health",
            "generate": "POST /generate",
            "stats": "GET /stats",
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
        "model_id": _model.model_id if _model else "unknown",
    }

@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateImageResponse)
def generate_image(body: GenerateImageRequest):
    """Generate an image from a text prompt using the diffusion model."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        image = _model.generate_from_text(body.prompt, n_steps=body.n_steps)
        response = GenerateImageResponse(
            image_shape=image.shape,
            prompt=body.prompt,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(body.n_steps)])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="generation")
        logger.exception("Image generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Image generation failed") from e

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        model_id=info.get("model_id", "unknown"),
        img_size=info.get("img_size", DEFAULT_IMG_SIZE),
        latent_dim=info.get("latent_dim", DEFAULT_LATENT_DIM),
        n_diffusion_steps=info.get("n_diffusion_steps", 1000),
        model_version=_model_version,
    )
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
