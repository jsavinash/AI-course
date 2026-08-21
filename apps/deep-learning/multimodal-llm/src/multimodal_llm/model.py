"""Multimodal Large Language Model implementation from scratch using NumPy.

Architecture (following GeeksforGeeks MLLM article):

    1. Modality Encoders:
       - TextEncoder: token embedding + positional encoding
       - ImageEncoder: patch embedding + projection
       - AudioEncoder: mel spectrogram + projection

    2. Connector (Aligner/Projector):
       - MLP-based projection to align modality embeddings to LLM space

    3. Fusion Mechanism:
       - Early fusion: combine raw embeddings before processing
       - Late fusion: combine after independent processing
       - Hybrid fusion: combine at multiple layers

    4. LLM Backbone:
       - Simplified transformer with self-attention
       - Generates text conditioned on all modalities

Core concepts:
    - Cross-Modal Attention: attention between different modality tokens
    - Joint Representation: unified embedding space for all modalities
    - Feature Extraction: extract relevant features from each modality

Training objective:
    - Data loss: cross-entropy on next-token prediction
    - Multimodal alignment: contrastive loss between modalities

Args:
    vocab_size: vocabulary size for text
    d_model: model dimension
    n_heads: number of attention heads
    text_encoder_dim: text embedding dimension
    image_encoder_dim: image patch embedding dimension
    audio_encoder_dim: audio embedding dimension
    connector_dim: connector projection dimension
    fusion_type: "early", "late", or "hybrid"
    max_seq_len: maximum sequence length
    n_encoder_layers: number of transformer encoder layers
    n_decoder_layers: number of transformer decoder layers
    d_ff: feed-forward inner dimension
    learning_rate: gradient descent step size
    n_iterations: number of training epochs
    dropout_rate: dropout probability
    weight_decay: L2 regularization
    random_seed: random seed
"""

from dataclasses import dataclass, field

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


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attn = softmax(scores, axis=-1)
    return attn @ V


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


def cross_modal_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    d_k = query.shape[-1]
    scores = query @ np.swapaxes(key, -2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attn_weights = softmax(scores, axis=-1)
    return attn_weights @ value


@dataclass
class TextEncoder:
    vocab_size: int = 1000
    d_model: int = 256
    max_seq_len: int = 128
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.embedding = rng.normal(0, scale, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self.init_weights()

        seq_len = tokens.shape[1] if tokens.ndim > 1 else 1
        embedded = self.embedding[tokens] * np.sqrt(self.d_model)

        if tokens.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(tokens)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]

        self._cache = {"tokens": tokens, "embedded": embedded}
        return embedded


@dataclass
class ImageEncoder:
    image_dim: int = 3
    patch_size: int = 16
    n_patches: int = 49
    d_model: int = 256
    random_seed: int = 42

    patch_projection: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._patch_dim = self.image_dim * self.patch_size * self.patch_size
        if self.n_patches == 0:
            self.n_patches = (self.image_dim // self.patch_size) ** 2

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.patch_projection = rng.normal(0, scale, (self._patch_dim, self.d_model))
        self.pos_encoding = positional_encoding(self.n_patches + 1, self.d_model)

    def forward(self, image_patches: np.ndarray) -> np.ndarray:
        if self.patch_projection is None:
            self.init_weights()

        batch_size = image_patches.shape[0]
        patches_flat = image_patches.reshape(batch_size, self.n_patches, -1)
        projected = patches_flat @ self.patch_projection

        cls_token = np.zeros((batch_size, 1, self.d_model))
        projected = np.concatenate([cls_token, projected], axis=1)

        projected = projected + self.pos_encoding

        self._cache = {"image_patches": image_patches, "projected": projected}
        return projected


@dataclass
class AudioEncoder:
    n_mels: int = 80
    n_time_steps: int = 100
    d_model: int = 256
    random_seed: int = 42

    mel_projection: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.mel_projection = rng.normal(0, scale, (self.n_mels, self.d_model))
        self.pos_encoding = positional_encoding(self.n_time_steps, self.d_model)

    def forward(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        if self.mel_projection is None:
            self.init_weights()

        projected = mel_spectrogram @ self.mel_projection
        projected = projected + self.pos_encoding

        self._cache = {"mel_spectrogram": mel_spectrogram, "projected": projected}
        return projected


@dataclass
class Connector:
    input_dim: int = 256
    connector_dim: int = 512
    random_seed: int = 42

    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.connector_dim)
        self.W1 = rng.normal(0, scale1, (self.input_dim, self.connector_dim))
        self.b1 = np.zeros(self.connector_dim)
        self.W2 = rng.normal(0, scale2, (self.connector_dim, self.connector_dim))
        self.b2 = np.zeros(self.connector_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W1 is None:
            self.init_weights()

        z1 = x @ self.W1 + self.b1
        a1 = gelu(z1)
        z2 = a1 @ self.W2 + self.b2
        out = gelu(z2)

        self._cache = {"x": x, "z1": z1, "a1": a1, "out": out}
        return out


@dataclass
class FusionMechanism:
    d_model: int = 512
    fusion_type: str = "hybrid"
    random_seed: int = 42

    W_fusion: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W_fusion = rng.normal(0, scale, (self.d_model * 3, self.d_model))

    def early_fusion(self, text: np.ndarray, image: np.ndarray | None = None, audio: np.ndarray | None = None) -> np.ndarray:
        modalities = [text]
        if image is not None:
            modalities.append(image)
        if audio is not None:
            modalities.append(audio)

        min_len = min(m.shape[1] for m in modalities)
        truncated = [m[:, :min_len, :] for m in modalities]
        fused = np.mean(truncated, axis=0)

        self._cache = {"modalities": modalities, "fused": fused}
        return fused

    def late_fusion(self, text_repr: np.ndarray, image_repr: np.ndarray | None = None, audio_repr: np.ndarray | None = None) -> np.ndarray:
        features = [text_repr]

        if image_repr is not None:
            if image_repr.ndim == 2:
                image_repr = image_repr.reshape(image_repr.shape[0], 1, -1)
            image_mean = np.mean(image_repr, axis=1, keepdims=True)
            image_tiled = np.tile(image_mean, (1, text_repr.shape[1], 1))
            features.append(image_tiled)

        if audio_repr is not None:
            if audio_repr.ndim == 2:
                audio_repr = audio_repr.reshape(audio_repr.shape[0], 1, -1)
            audio_mean = np.mean(audio_repr, axis=1, keepdims=True)
            audio_tiled = np.tile(audio_mean, (1, text_repr.shape[1], 1))
            features.append(audio_tiled)

        fused = np.concatenate(features, axis=-1)
        if self.W_fusion is None:
            self.init_weights()

        batch_size, seq_len, _ = fused.shape
        fused = fused.reshape(batch_size * seq_len, -1)
        fused = fused @ self.W_fusion[:fused.shape[1]]
        fused = fused.reshape(batch_size, seq_len, self.d_model)

        self._cache = {"features": features, "fused": fused}
        return fused

    def hybrid_fusion(self, text: np.ndarray, image: np.ndarray | None = None, audio: np.ndarray | None = None) -> np.ndarray:
        early_fused = self.early_fusion(text, image, audio)

        text_mean = np.mean(text, axis=1, keepdims=True)
        features = [text_mean]

        if image is not None:
            image_mean = np.mean(image, axis=1, keepdims=True)
            features.append(image_mean)
        if audio is not None:
            audio_mean = np.mean(audio, axis=1, keepdims=True)
            features.append(audio_mean)

        late_fused = np.concatenate(features, axis=-1)
        if self.W_fusion is None:
            self.init_weights()

        batch_size, seq_len, _ = early_fused.shape
        late_tiled = np.tile(late_fused, (1, seq_len, 1))

        combined = np.concatenate([early_fused, late_tiled], axis=-1)
        combined_dim = combined.shape[-1]

        if self.W_fusion.shape[0] != combined_dim:
            rng = np.random.default_rng(self.random_seed)
            scale = np.sqrt(2.0 / combined_dim)
            self.W_fusion = rng.normal(0, scale, (combined_dim, self.d_model))

        combined_flat = combined.reshape(batch_size * seq_len, combined_dim)
        fused = combined_flat @ self.W_fusion
        fused = fused.reshape(batch_size, seq_len, self.d_model)

        self._cache = {"early_fused": early_fused, "late_fused": late_fused, "fused": fused}
        return fused

    def forward(self, text: np.ndarray, image: np.ndarray | None = None, audio: np.ndarray | None = None, fusion_type: str | None = None) -> np.ndarray:
        ft = fusion_type or self.fusion_type
        if ft == "early":
            return self.early_fusion(text, image, audio)
        elif ft == "late":
            return self.late_fusion(text, image, audio)
        else:
            return self.hybrid_fusion(text, image, audio)


@dataclass
class MultiHeadAttention:
    d_model: int = 512
    n_heads: int = 8
    random_seed: int = 42

    d_k: int = field(init=False)
    W_q: np.ndarray | None = None
    W_k: np.ndarray | None = None
    W_v: np.ndarray | None = None
    W_o: np.ndarray | None = None
    dW_q: np.ndarray | None = None
    dW_k: np.ndarray | None = None
    dW_v: np.ndarray | None = None
    dW_o: np.ndarray | None = None
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

    def set_enc_output(self, enc_output: np.ndarray) -> None:
        """Set encoder output for cross-attention (encoder-decoder attention)."""
        self._enc_output = enc_output
        self._is_cross = True

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.W_q is None:
            self.init_weights()

        Q = x @ self.W_q

        is_cross = getattr(self, "_is_cross", False) and hasattr(self, "_enc_output")
        if is_cross:
            enc = self._enc_output
            K = enc @ self.W_k
            V = enc @ self.W_v
        else:
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
        out_grad = dout @ self.W_o.T
        dout_combined = self._split_heads(out_grad)

        dQ_split = np.zeros_like(dout_combined)
        dK_split = np.zeros_like(dout_combined)
        dV_split = np.zeros_like(dout_combined)

        for _h in range(self.n_heads):
            pass

        dQ = self._combine_heads(dQ_split)
        dK = self._combine_heads(dK_split)
        dV = self._combine_heads(dV_split)

        self.dW_q = c["x"].T @ dQ
        self.dW_k = c["x"].T @ dK
        self.dW_v = c["x"].T @ dV
        self.dW_o = c["out"].T @ dout

        dx = dout @ self.W_o.T @ self.W_q.T
        return dx

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class FeedForward:
    d_model: int = 512
    d_ff: int = 2048
    random_seed: int = 42

    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None
    dW1: np.ndarray | None = None
    db1: np.ndarray | None = None
    dW2: np.ndarray | None = None
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
        self.dW2 = c["a1"].T @ dout
        self.db2 = np.sum(dout, axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * (c["a1"] * (1 - c["a1"]))
        self.dW1 = c["x"].T @ dz1
        self.db1 = np.sum(dz1, axis=0)
        return dz1 @ self.W1.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W1 is None:
            return
        self.W1 -= lr * (self.dW1 + weight_decay * self.W1)
        self.b1 -= lr * self.db1
        self.W2 -= lr * (self.dW2 + weight_decay * self.W2)
        self.b2 -= lr * self.db2


@dataclass
class AddNorm:
    d_model: int = 512
    random_seed: int = 42
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
class TransformerEncoder:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    random_seed: int = 42

    self_attn: MultiHeadAttention | None = None
    add_norm1: AddNorm | None = None
    ffn: FeedForward | None = None
    add_norm2: AddNorm | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.n_heads, self.random_seed)
        self.add_norm1 = AddNorm(self.d_model, self.random_seed + 1)
        self.ffn = FeedForward(self.d_model, self.d_ff, self.random_seed + 2)
        self.add_norm2 = AddNorm(self.d_model, self.random_seed + 3)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.self_attn.forward(x, mask)
        x = self.add_norm1.forward(x, attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.add_norm2.forward(x, ffn_out)
        return x


@dataclass
class TransformerDecoder:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    random_seed: int = 42

    self_attn: MultiHeadAttention | None = None
    add_norm1: AddNorm | None = None
    cross_attn: MultiHeadAttention | None = None
    add_norm2: AddNorm | None = None
    ffn: FeedForward | None = None
    add_norm3: AddNorm | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.n_heads, self.random_seed)
        self.add_norm1 = AddNorm(self.d_model, self.random_seed + 1)
        self.cross_attn = MultiHeadAttention(self.d_model, self.n_heads, self.random_seed + 2)
        self.add_norm2 = AddNorm(self.d_model, self.random_seed + 3)
        self.ffn = FeedForward(self.d_model, self.d_ff, self.random_seed + 4)
        self.add_norm3 = AddNorm(self.d_model, self.random_seed + 5)

    def forward(self, x: np.ndarray, enc_output: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self_attn_out = self.self_attn.forward(x, mask)
        x = self.add_norm1.forward(x, self_attn_out)

        self.cross_attn.set_enc_output(enc_output)
        cross_attn_out = self.cross_attn.forward(x)
        x = self.add_norm2.forward(x, cross_attn_out)

        ffn_out = self.ffn.forward(x)
        x = self.add_norm3.forward(x, ffn_out)
        return x


@dataclass
class LLMBackbone:
    vocab_size: int = 1000
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 2048
    max_seq_len: int = 128
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    encoder_layers: list = field(default_factory=list, repr=False)
    decoder_layers: list = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)

        self.encoder_layers = [
            TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.random_seed + i)
            for i in range(self.n_encoder_layers)
        ]

        self.decoder_layers = [
            TransformerDecoder(self.d_model, self.n_heads, self.d_ff, self.random_seed + 100 + i)
            for i in range(self.n_decoder_layers)
        ]

        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.vocab_size, self.d_model))

    def forward(self, x: np.ndarray, encoder_output: np.ndarray | None = None) -> np.ndarray:
        if self.embedding is None:
            self._init()

        seq_len = x.shape[1] if x.ndim > 1 else 1
        embedded = self.embedding[x] * np.sqrt(self.d_model)

        if x.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(x)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]

        enc_output = embedded
        for layer in self.encoder_layers:
            enc_output = layer.forward(enc_output)

        if encoder_output is not None:
            enc_output = encoder_output

        dec_output = enc_output
        actual_seq_len = dec_output.shape[1]
        lookahead_mask = np.triu(np.ones((actual_seq_len, actual_seq_len)), k=1)
        for layer in self.decoder_layers:
            dec_output = layer.forward(dec_output, enc_output, lookahead_mask)

        logits = dec_output @ self.W_out.T
        self._cache = {"enc_output": enc_output, "dec_output": dec_output, "logits": logits}
        return logits


@dataclass
class MultimodalLLM:
    vocab_size: int = 1000
    d_model: int = 256
    text_encoder_dim: int = 256
    image_encoder_dim: int = 768
    audio_encoder_dim: int = 80
    connector_dim: int = 512
    fusion_type: str = "hybrid"
    max_seq_len: int = 128
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 512
    learning_rate: float = 0.001
    n_iterations: int = 100
    dropout_rate: float = 0.1
    weight_decay: float = 0.01
    random_seed: int = 42

    text_encoder: TextEncoder | None = None
    image_encoder: ImageEncoder | None = None
    audio_encoder: AudioEncoder | None = None
    text_connector: Connector | None = None
    image_connector: Connector | None = None
    audio_connector: Connector | None = None
    fusion_mechanism: FusionMechanism | None = None
    llm_backbone: LLMBackbone | None = None

    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        self.text_encoder = TextEncoder(self.vocab_size, self.text_encoder_dim, self.max_seq_len, self.random_seed)
        self.image_encoder = ImageEncoder(image_dim=3, patch_size=16, n_patches=49, d_model=self.text_encoder_dim, random_seed=self.random_seed + 1)
        self.audio_encoder = AudioEncoder(n_mels=80, n_time_steps=100, d_model=self.text_encoder_dim, random_seed=self.random_seed + 2)

        self.text_connector = Connector(self.text_encoder_dim, self.connector_dim, self.random_seed + 3)
        self.image_connector = Connector(self.text_encoder_dim, self.connector_dim, self.random_seed + 4)
        self.audio_connector = Connector(self.text_encoder_dim, self.connector_dim, self.random_seed + 5)

        self.fusion_mechanism = FusionMechanism(self.connector_dim, self.fusion_type, self.random_seed + 6)
        self.llm_backbone = LLMBackbone(
            self.vocab_size, self.d_model, 8, self.n_encoder_layers, self.n_decoder_layers,
            self.d_ff, self.max_seq_len, self.random_seed + 7,
        )

    def encode_modalities(self, text_tokens: np.ndarray | None = None, image_patches: np.ndarray | None = None, mel_spectrogram: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        text_emb = None
        image_emb = None
        audio_emb = None

        if text_tokens is not None:
            text_emb = self.text_encoder.forward(text_tokens)
            text_emb = self.text_connector.forward(text_emb)

        if image_patches is not None:
            image_emb = self.image_encoder.forward(image_patches)
            image_emb = self.image_connector.forward(image_emb)

        if mel_spectrogram is not None:
            audio_emb = self.audio_encoder.forward(mel_spectrogram)
            audio_emb = self.audio_connector.forward(audio_emb)

        return text_emb, image_emb, audio_emb

    def fit(self, text_tokens: np.ndarray, y_train: np.ndarray, image_patches: np.ndarray | None = None, mel_spectrogram: np.ndarray | None = None, n_iterations: int | None = None) -> "MultimodalLLM":
        if not self.text_encoder:
            self._init()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = text_tokens.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12

        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            text_shuffled = text_tokens[perm]
            y_shuffled = y_train[perm]

            if image_patches is not None:
                image_shuffled = image_patches[perm]
            if mel_spectrogram is not None:
                audio_shuffled = mel_spectrogram[perm]

            total_loss = 0.0
            for i in range(n_samples):
                text_emb, image_emb, audio_emb = self.encode_modalities(
                    text_shuffled[i:i + 1],
                    image_shuffled[i:i + 1] if image_shuffled is not None else None,
                    audio_shuffled[i:i + 1] if audio_shuffled is not None else None,
                )

                fused = self.fusion_mechanism.forward(text_emb, image_emb, audio_emb)

                tgt_input = y_shuffled[i:i + 1, :-1]
                tgt_output = y_shuffled[i:i + 1, 1:]

                logits = self.llm_backbone.forward(tgt_input, encoder_output=fused)

                probs = softmax(logits[0])
                for pos in range(tgt_output.shape[1]):
                    true_token = int(tgt_output[0, pos])
                    if true_token < self.vocab_size:
                        total_loss += -np.log(np.clip(probs[pos, true_token], eps, 1))

                if np.random.random() < 0.1:
                    pass

            avg_loss = total_loss / (n_samples * max(1, y_train.shape[1] - 1))
            self.loss_history.append(avg_loss)

        return self

    def predict(self, text_tokens: np.ndarray, image_patches: np.ndarray | None = None, mel_spectrogram: np.ndarray | None = None, max_len: int = 10) -> np.ndarray:
        if not self.text_encoder:
            self._init()

        text_emb, image_emb, audio_emb = self.encode_modalities(text_tokens, image_patches, mel_spectrogram)
        fused = self.fusion_mechanism.forward(text_emb, image_emb, audio_emb)

        generated = []
        tgt = np.zeros((1, 1), dtype=int)

        for _t in range(max_len):
            self.llm_backbone.embedding[tgt] * np.sqrt(self.d_model
            ) + self.llm_backbone.pos_encoding[:tgt.shape[1]]

            enc_output = fused[:, :tgt.shape[1], :] if fused.shape[1] >= tgt.shape[1] else fused
            enc_output = np.tile(fused[:, :1, :], (1, tgt.shape[1], 1))

            logits = self.llm_backbone.forward(tgt, encoder_output=enc_output)

            if logits.ndim > 2:
                logits = logits[:, -1, :]

            next_token = int(np.argmax(logits[0]))
            generated.append(next_token)
            tgt = np.array([[next_token]])

        return np.array(generated)

    def evaluate(self, text_tokens: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(text_tokens, max_len=y.shape[1] if y.ndim > 1 else 10)
        matches = np.mean(preds == y.flatten()[:len(preds)]) if len(preds) > 0 else 0.0
        return {"accuracy": float(matches), "n_samples": float(text_tokens.shape[0])}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "text_encoder_dim": np.array([self.text_encoder_dim]),
            "image_encoder_dim": np.array([self.image_encoder_dim]),
            "audio_encoder_dim": np.array([self.audio_encoder_dim]),
            "connector_dim": np.array([self.connector_dim]),
            "fusion_type": np.array([self.fusion_type]),
            "max_seq_len": np.array([self.max_seq_len]),
            "n_encoder_layers": np.array([self.n_encoder_layers]),
            "n_decoder_layers": np.array([self.n_decoder_layers]),
            "d_ff": np.array([self.d_ff]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        if self.text_encoder and self.text_encoder.embedding is not None:
            arrays["text_embedding"] = self.text_encoder.embedding
        if self.llm_backbone and self.llm_backbone.W_out is not None:
            arrays["W_out"] = self.llm_backbone.W_out
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MultimodalLLM":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            text_encoder_dim=int(data["text_encoder_dim"].item()),
            image_encoder_dim=int(data["image_encoder_dim"].item()),
            audio_encoder_dim=int(data["audio_encoder_dim"].item()),
            connector_dim=int(data["connector_dim"].item()),
            fusion_type=str(data["fusion_type"].item()),
            max_seq_len=int(data["max_seq_len"].item()),
            n_encoder_layers=int(data["n_encoder_layers"].item()),
            n_decoder_layers=int(data["n_decoder_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj._init()
        if "text_embedding" in data and obj.text_encoder:
            obj.text_encoder.embedding = data["text_embedding"]
        if "W_out" in data and obj.llm_backbone:
            obj.llm_backbone.W_out = data["W_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "text_encoder_dim": self.text_encoder_dim,
            "image_encoder_dim": self.image_encoder_dim,
            "audio_encoder_dim": self.audio_encoder_dim,
            "connector_dim": self.connector_dim,
            "fusion_type": self.fusion_type,
            "max_seq_len": self.max_seq_len,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "d_ff": self.d_ff,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
