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

    def decode(self, z: np.ndarray) -> np.ndarray:
        if self.decoder_w is None:
            self._init()
        h = gelu(z @ self.decoder_w + self.decoder_b)
        out = h @ self._decoder_out_W + self._decoder_out_b
        out = 1.0 / (1.0 + np.exp(-out))
        return out


@dataclass
class DiffusionModel:
    img_size: int = 32
    latent_dim: int = 64
    n_diffusion_steps: int = 1000
    random_seed: int = 42

    beta: np.ndarray | None = None
    alpha: np.ndarray | None = None
    alpha_bar: np.ndarray | None = None
    noise_predictor_w: np.ndarray | None = None
    noise_predictor_b: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.beta = np.linspace(1e-4, 0.02, self.n_diffusion_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        input_dim = self.latent_dim
        scale = np.sqrt(2.0 / input_dim)
        self.noise_predictor_w = rng.normal(0, scale, (input_dim, input_dim))
        self.noise_predictor_b = np.zeros(input_dim)

    def forward_process(self, z_0: np.ndarray, t: int) -> np.ndarray:
        if self.alpha_bar is None:
            self._init()
        noise = np.random.default_rng(self.random_seed + t).normal(0, 1, size=z_0.shape)
        sqrt_alpha_bar = np.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar[t])
        return sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise, noise

    def predict_noise(self, z_t: np.ndarray, t: int, text_cond: np.ndarray | None = None) -> np.ndarray:
        if self.noise_predictor_w is None:
            self._init()
        t_norm = np.full((z_t.shape[0], 1), t / self.n_diffusion_steps)
        if text_cond is not None:
            cond = np.repeat(text_cond, z_t.shape[0], axis=0)
            x = np.concatenate([z_t, t_norm, cond], axis=-1)
        else:
            x = np.concatenate([z_t, t_norm], axis=-1)
        h = gelu(x @ self.noise_predictor_w.T + self.noise_predictor_b)
        return h[:, :z_t.shape[-1]]

    def reverse_process(self, z_T: np.ndarray, text_cond: np.ndarray | None = None, n_steps: int = 50) -> np.ndarray:
        if self.alpha is None:
            self._init()
        z = z_T.copy()
        n_steps = min(n_steps, self.n_diffusion_steps)
        for t in reversed(range(n_steps)):
            noise_pred = self.predict_noise(z, t, text_cond)
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]
            z = (1.0 / np.sqrt(alpha)) * (z - (1.0 - alpha) / np.sqrt(1.0 - alpha_bar) * noise_pred)
            if t > 0:
                z += np.sqrt(beta) * np.random.default_rng(self.random_seed + t).normal(0, 1, size=z.shape)
        return z


@dataclass
class ImageGenerationModel:
    vocab_size: int = 1000
    d_model: int = 256
    img_size: int = 32
    latent_dim: int = 64
    n_diffusion_steps: int = 1000
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    random_seed: int = 42
    learning_rate: float = 0.001
    n_iterations: int = 100
    weight_decay: float = 0.01

    tokenizer: ImageTokenizer | None = None
    text_conditioning: TextConditioning | None = None
    vae: VariationalAutoencoder | None = None
    diffusion: DiffusionModel | None = None
    loss_history: list[float] = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        self.tokenizer = ImageTokenizer(vocab_size=self.vocab_size, random_seed=self.random_seed)
        self.text_conditioning = TextConditioning(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            random_seed=self.random_seed + 1,
        )
        self.vae = VariationalAutoencoder(img_size=self.img_size, latent_dim=self.latent_dim, random_seed=self.random_seed + 2)
        self.diffusion = DiffusionModel(img_size=self.img_size, latent_dim=self.latent_dim, n_diffusion_steps=self.n_diffusion_steps, random_seed=self.random_seed + 3)

    def encode_text(self, text: str) -> np.ndarray:
        if self.text_conditioning is None:
            self._init()
        tokens = self.tokenizer.encode(text)
        token_array = np.array([tokens])
        return self.text_conditioning.forward(token_array)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        if self.vae is None:
            self._init()
        flat = image.reshape(1, -1)
        mu, logvar = self.vae.encode(flat)
        return self.vae.reparameterize(mu, logvar)

    def decode_latent(self, z: np.ndarray) -> np.ndarray:
        if self.vae is None:
            self._init()
        reconstructed = self.vae.decode(z)
        return reconstructed.reshape(self.img_size, self.img_size, 3)

    def generate_from_text(self, text: str, n_steps: int = 50) -> np.ndarray:
        if self.diffusion is None or self.text_conditioning is None:
            self._init()
        text_cond = self.encode_text(text)
        z_T = np.random.default_rng(self.random_seed).normal(0, 1, size=(1, self.latent_dim))
        z_0 = self.diffusion.reverse_process(z_T, text_cond=text_cond, n_steps=n_steps)
        return self.decode_latent(z_0)

    def fit(self, X_images: np.ndarray, X_texts: np.ndarray, n_iterations: int | None = None) -> dict:
        if self.vae is None or self.diffusion is None:
            self._init()
        if n_iterations is None:
            n_iterations = self.n_iterations
        n_samples = X_images.shape[0]
        rng = np.random.default_rng(self.random_seed)
        total_loss = 0.0
        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_img_shuffled = X_images[perm]
            epoch_loss = 0.0
            for i in range(n_samples):
                image = X_img_shuffled[i].reshape(1, -1)
                mu, logvar = self.vae.encode(image)
                z = self.vae.reparameterize(mu, logvar)
                reconstructed = self.vae.decode(z)
                recon_loss = float(np.mean((reconstructed - image) ** 2))
                kl_loss = -0.5 * float(np.mean(1 + logvar - mu ** 2 - np.exp(logvar)))
                t = int(rng.integers(0, self.n_diffusion_steps))
                z_t, noise = self.diffusion.forward_process(z, t)
                noise_pred = self.diffusion.predict_noise(z_t, t)
                diff_loss = float(np.mean((noise_pred - noise) ** 2))
                loss = recon_loss + kl_loss + diff_loss
                epoch_loss += loss
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            total_loss += avg_loss
        return self.to_dict()

    def evaluate(self, X_images: np.ndarray, X_texts: np.ndarray) -> dict[str, float]:
        n_samples = X_images.shape[0]

        total_recon = 0.0
        for i in range(n_samples):
            image = X_images[i].reshape(1, -1)
            mu, _ = self.vae.encode(image)
            z = mu
            reconstructed = self.vae.decode(z)
            total_recon += float(np.mean((reconstructed - image) ** 2))
        return {"reconstruction_mse": total_recon / n_samples, "n_samples": float(n_samples)}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "img_size": np.array([self.img_size]),
            "latent_dim": np.array([self.latent_dim]),
            "n_diffusion_steps": np.array([self.n_diffusion_steps]),
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_layers": np.array([self.n_layers]),
            "d_ff": np.array([self.d_ff]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        if self.vae and self.vae.encoder_w is not None:
            arrays["encoder_w"] = self.vae.encoder_w
            arrays["encoder_b"] = self.vae.encoder_b
            arrays["mu_w"] = self.vae.mu_w
            arrays["mu_b"] = self.vae.mu_b
            arrays["logvar_w"] = self.vae.logvar_w
            arrays["logvar_b"] = self.vae.logvar_b
            arrays["decoder_w"] = self.vae.decoder_w
            arrays["decoder_b"] = self.vae.decoder_b
        if self.diffusion and self.diffusion.noise_predictor_w is not None:
            arrays["noise_predictor_w"] = self.diffusion.noise_predictor_w
            arrays["noise_predictor_b"] = self.diffusion.noise_predictor_b
            arrays["alpha"] = self.diffusion.alpha
            arrays["alpha_bar"] = self.diffusion.alpha_bar
            arrays["beta"] = self.diffusion.beta
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "ImageGenerationModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            img_size=int(data["img_size"].item()),
            latent_dim=int(data["latent_dim"].item()),
            n_diffusion_steps=int(data["n_diffusion_steps"].item()),
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_layers=int(data["n_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj._init()
        if "encoder_w" in data and obj.vae:
            obj.vae.encoder_w = data["encoder_w"]
            obj.vae.encoder_b = data["encoder_b"]
            obj.vae.mu_w = data["mu_w"]
            obj.vae.mu_b = data["mu_b"]
            obj.vae.logvar_w = data["logvar_w"]
            obj.vae.logvar_b = data["logvar_b"]
            obj.vae.decoder_w = data["decoder_w"]
            obj.vae.decoder_b = data["decoder_b"]
        if "noise_predictor_w" in data and obj.diffusion:
            obj.diffusion.noise_predictor_w = data["noise_predictor_w"]
            obj.diffusion.noise_predictor_b = data["noise_predictor_b"]
            obj.diffusion.alpha = data["alpha"]
            obj.diffusion.alpha_bar = data["alpha_bar"]
            obj.diffusion.beta = data["beta"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "img_size": self.img_size,
            "latent_dim": self.latent_dim,
            "n_diffusion_steps": self.n_diffusion_steps,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
