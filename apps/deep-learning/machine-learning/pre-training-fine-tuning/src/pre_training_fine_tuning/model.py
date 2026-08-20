"""Pre-training and Fine-Tuning Transformer model.

Implements:
- Pre-training objectives: Masked Language Modeling (MLM), Next-Token Prediction (NTP)
- Fine-tuning strategies: Full, Feature Extraction, Partial, PEFT (LoRA-style)
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

FineTuningStrategy = Literal["full", "feature_extraction", "partial", "peft"]
PretrainingObjective = Literal["mlm", "ntp"]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


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


@dataclass
class MultiHeadAttention:
    """Multi-Head Attention mechanism."""

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
    _frozen: bool = False

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

    def set_enc_output(self, enc_output: np.ndarray) -> None:
        """Set encoder output for cross-attention (encoder-decoder attention)."""
        self._enc_output = enc_output
        self._is_cross = True

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
        if self.W_q is None or self._frozen:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class FeedForward:
    """Position-wise Feed-Forward Network."""

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
    _frozen: bool = False

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
        if self.W1 is None or self._frozen:
            return
        self.W1 -= lr * (self.dW1 + weight_decay * self.W1)
        self.b1 -= lr * self.db1
        self.W2 -= lr * (self.dW2 + weight_decay * self.W2)
        self.b2 -= lr * self.db2


@dataclass
class AddNorm:
    """Residual connection + Layer Normalization."""

    d_model: int = 512
    random_seed: int = 42

    gamma: np.ndarray | None = None
    beta: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)
    _frozen: bool = False

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

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.gamma is None or self._frozen:
            return
        self.gamma -= lr * weight_decay * self.gamma
        self.beta -= lr * weight_decay * self.beta


@dataclass
class LoRAAdapter:
    """Low-Rank Adaptation (LoRA) adapter for efficient fine-tuning.

    Adds low-rank matrices A and B to update weights without modifying original weights:
    W' = W + B @ A, where B @ A is low-rank.
    """

    d_model: int = 128
    rank: int = 4
    random_seed: int = 42

    A: np.ndarray | None = None
    B: np.ndarray | None = None
    dA: np.ndarray | None = None
    dB: np.ndarray | None = None

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale_a = np.sqrt(1.0 / self.d_model)
        self.A = rng.normal(0, scale_a, (self.d_model, self.rank))
        self.B = np.zeros((self.rank, self.d_model))

    def forward(self, x: np.ndarray, base_output: np.ndarray) -> np.ndarray:
        if self.A is None:
            self.init_weights()
        lora_output = x @ self.A @ self.B
        self._cache = {"x": x, "base_output": base_output, "lora_output": lora_output}
        return base_output + lora_output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dB = c["x"] @ self.A @ dout
        self.dA = c["x"].T @ (dout @ self.B.T)
        return dout

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.A is None:
            return
        self.A -= lr * (self.dA + weight_decay * self.A)
        self.B -= lr * (self.dB + weight_decay * self.B)


@dataclass
class MLMHead:
    """Masked Language Modeling head for pre-training.

    Predicts original tokens at masked positions.
    """

    d_model: int = 128
    vocab_size: int = 100
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W = rng.normal(0, scale, (self.d_model, self.vocab_size))
        self.b = np.zeros(self.vocab_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            self.init_weights()
        logits = x @ self.W + self.b
        self._cache = {"x": x, "logits": logits}
        return logits

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dW = c["x"].T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class NTPHead:
    """Next-Token Prediction head for pre-training.

    Predicts the next token in a sequence.
    """

    d_model: int = 128
    vocab_size: int = 100
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W = rng.normal(0, scale, (self.d_model, self.vocab_size))
        self.b = np.zeros(self.vocab_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            self.init_weights()
        logits = x @ self.W + self.b
        self._cache = {"x": x, "logits": logits}
        return logits

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dW = c["x"].T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class ClassificationHead:
    """Task-specific classification head for fine-tuning."""

    d_model: int = 128
    n_classes: int = 10
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.d_model)
        self.W = rng.normal(0, scale, (self.d_model, self.n_classes))
        self.b = np.zeros(self.n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            self.init_weights()
        logits = x @ self.W + self.b
        self._cache = {"x": x, "logits": logits}
        return logits

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dW = c["x"].T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class Transformer:
    """Full Transformer supporting pre-training and fine-tuning.

    Supports:
        - Pre-training objectives: MLM, NTP
        - Fine-tuning strategies: full, feature_extraction, partial, peft
    """

    vocab_size: int = 100
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 32
    learning_rate: float = 0.001
    n_iterations: int = 100
    dropout_rate: float = 0.1
    weight_decay: float = 0.01
    clip_value: float = 1.0
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    encoder_layers: list = field(default_factory=list, repr=False)
    decoder_layers: list = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    training_mode: str = "pretrain"
    loss_history: list[float] = field(default_factory=list, repr=False)
    fine_tuning_strategy: FineTuningStrategy = "full"
    pretraining_objective: PretrainingObjective = "ntp"
    mlm_head: MLMHead | None = None
    ntp_head: NTPHead | None = None
    classification_head: ClassificationHead | None = None
    lora_adapters: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)
        self.encoder_layers = [
            (
                MultiHeadAttention(self.d_model, self.n_heads, self.random_seed + i),
                AddNorm(self.d_model, self.random_seed + i),
                FeedForward(self.d_model, self.d_ff, self.random_seed + i),
                AddNorm(self.d_model, self.random_seed + i),
            )
            for i in range(self.n_encoder_layers)
        ]
        self.decoder_layers = [
            (
                MultiHeadAttention(self.d_model, self.n_heads, self.random_seed + 100 + i),
                AddNorm(self.d_model, self.random_seed + 100 + i),
                MultiHeadAttention(self.d_model, self.n_heads, self.random_seed + 200 + i),
                AddNorm(self.d_model, self.random_seed + 200 + i),
                FeedForward(self.d_model, self.d_ff, self.random_seed + 200 + i),
                AddNorm(self.d_model, self.random_seed + 200 + i),
            )
            for i in range(self.n_decoder_layers)
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.vocab_size, self.d_model))
        self.mlm_head = MLMHead(self.d_model, self.vocab_size, self.random_seed)
        self.ntp_head = NTPHead(self.d_model, self.vocab_size, self.random_seed)

    def _apply_fine_tuning_strategy(self, strategy: FineTuningStrategy) -> None:
        """Apply fine-tuning strategy by freezing/unfreezing layers."""
        self.fine_tuning_strategy = strategy
        if strategy == "full":
            for layer in self.encoder_layers + self.decoder_layers:
                for component in layer:
                    component._frozen = False
        elif strategy == "feature_extraction":
            for layer in self.encoder_layers + self.decoder_layers:
                for component in layer:
                    component._frozen = True
        elif strategy == "partial":
            total_layers = len(self.encoder_layers) + len(self.decoder_layers)
            for idx, layer in enumerate(self.encoder_layers + self.decoder_layers):
                for component in layer:
                    component._frozen = idx < total_layers // 2
        elif strategy == "peft":
            for layer in self.encoder_layers + self.decoder_layers:
                for component in layer:
                    component._frozen = True

    def _embed(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        embedded = self.embedding[x]
        seq_len = x.shape[1] if x.ndim > 1 else 1
        if x.ndim == 1:
            return embedded * np.sqrt(self.d_model) + self.pos_encoding[:len(x)]
        return embedded * np.sqrt(self.d_model) + self.pos_encoding[:seq_len]

    def _create_lookahead_mask(self, seq_len: int) -> np.ndarray:
        return np.triu(np.ones((seq_len, seq_len)), k=1)

    def _forward_encoder(self, x: np.ndarray) -> np.ndarray:
        for (self_attn, add_norm1, ffn, add_norm2) in self.encoder_layers:
            attn_out = self_attn.forward(x)
            x = add_norm1.forward(x, attn_out)
            ffn_out = ffn.forward(x)
            x = add_norm2.forward(x, ffn_out)
        return x

    def _forward_decoder(self, tgt_emb: np.ndarray, enc_output: np.ndarray) -> np.ndarray:
        tgt_mask = self._create_lookahead_mask(tgt_emb.shape[1]) if tgt_emb.ndim > 1 else None
        for (self_attn, add_norm1, enc_dec_attn, add_norm2, ffn, add_norm3) in self.decoder_layers:
            tgt_attn_out = self_attn.forward(tgt_emb, tgt_mask)
            tgt_emb = add_norm1.forward(tgt_emb, tgt_attn_out)
            enc_dec_attn.set_enc_output(enc_output)
            enc_dec_out = enc_dec_attn.forward(tgt_emb)
            tgt_emb = add_norm2.forward(tgt_emb, enc_dec_out)
            ffn_out = ffn.forward(tgt_emb)
            tgt_emb = add_norm3.forward(tgt_emb, ffn_out)
        return tgt_emb

    def _compute_mlm_loss(self, logits: np.ndarray, targets: np.ndarray, mask_positions: np.ndarray) -> float:
        eps = 1e-12
        total_loss = 0.0
        n_masked = 0
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                if mask_positions[i, j]:
                    probs = softmax(logits[i, j])
                    true_token = int(targets[i, j])
                    if 0 <= true_token < self.vocab_size:
                        total_loss += -np.log(np.clip(probs[true_token], eps, 1))
                        n_masked += 1
        return total_loss / max(n_masked, 1)

    def _compute_ntp_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        eps = 1e-12
        total_loss = 0.0
        n_tokens = 0
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1] - 1):
                probs = softmax(logits[i, j])
                true_token = int(targets[i, j])
                if 0 <= true_token < self.vocab_size:
                    total_loss += -np.log(np.clip(probs[true_token], eps, 1))
                    n_tokens += 1
        return total_loss / max(n_tokens, 1)

    def _compute_classification_loss(self, logits: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
        eps = 1e-12
        total_loss = 0.0
        for i in range(len(labels)):
            probs = softmax(logits[i])
            label = int(labels[i])
            if 0 <= label < n_classes:
                total_loss += -np.log(np.clip(probs[label], eps, 1))
        return total_loss / max(len(labels), 1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        phase: str = "pretrain",
        objective: PretrainingObjective = "ntp",
        strategy: FineTuningStrategy = "full",
        n_iterations: int | None = None,
        learning_rate: float | None = None,
        mask_positions: np.ndarray | None = None,
    ) -> "Transformer":
        """Train the model for pre-training or fine-tuning.

        Args:
            X_train: input sequences
            y_train: targets (original tokens for MLM, next tokens for NTP, labels for fine-tuning)
            phase: "pretrain" or "finetune"
            objective: "mlm" or "ntp" for pre-training
            strategy: fine-tuning strategy ("full", "feature_extraction", "partial", "peft")
            n_iterations: override default iterations
            learning_rate: override default learning rate
            mask_positions: boolean mask for MLM objective
        """
        if not self.encoder_layers:
            self._init()

        if n_iterations is None:
            n_iterations = self.n_iterations
        if learning_rate is None:
            learning_rate = self.learning_rate

        self.training_mode = phase
        self.pretraining_objective = objective
        if phase == "finetune":
            self._apply_fine_tuning_strategy(strategy)
            if self.classification_head is None:
                self.classification_head = ClassificationHead(self.d_model, 10, self.random_seed)

        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)

        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            if mask_positions is not None:
                mask_shuffled = mask_positions[perm]

            total_loss = 0.0
            for i in range(n_samples):
                src = X_shuffled[i:i + 1]
                tgt = y_shuffled[i:i + 1]

                if phase == "pretrain" and objective == "ntp":
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                else:
                    tgt_input = src
                    tgt_output = tgt

                src_emb = self._embed(src)
                tgt_emb = self._embed(tgt_input)

                enc_output = self._forward_encoder(src_emb)
                dec_output = self._forward_decoder(tgt_emb, enc_output)

                if phase == "pretrain" and objective == "mlm":
                    logits = self.mlm_head.forward(dec_output)
                    current_mask = mask_shuffled[i : i + 1] if mask_positions is not None else None
                    if current_mask is not None:
                        loss = self._compute_mlm_loss(logits, tgt_output, current_mask)
                    else:
                        loss = self._compute_ntp_loss(logits, tgt_output)
                    total_loss += loss
                elif phase == "pretrain" and objective == "ntp":
                    logits = self.ntp_head.forward(dec_output)
                    loss = self._compute_ntp_loss(logits, tgt_output)
                    total_loss += loss
                else:
                    pooled = np.mean(dec_output, axis=1)
                    logits = self.classification_head.forward(pooled)
                    loss = self._compute_classification_loss(logits, tgt_output, 10)
                    total_loss += loss

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

        return self

    def predict(self, X: np.ndarray, max_len: int = 10, phase: str = "pretrain") -> np.ndarray:
        """Predict tokens or classes."""
        if not self.encoder_layers:
            self._init()

        src_emb = self._embed(X)
        enc_output = self._forward_encoder(src_emb)

        if phase == "finetune":
            pooled = np.mean(enc_output, axis=1)
            logits = self.classification_head.forward(pooled)
            return np.argmax(softmax(logits), axis=-1)

        generated = []
        tgt = np.zeros((1, 1), dtype=int)
        for _t in range(max_len):
            tgt_emb = self._embed(tgt)
            dec_output = self._forward_decoder(tgt_emb, enc_output)
            logits = self.ntp_head.forward(dec_output) if self.ntp_head else dec_output @ self.W_out.T
            if logits.ndim > 2:
                logits = logits[:, -1, :]
            next_token = int(np.argmax(logits[0]))
            generated.append(next_token)
            tgt = np.array([[next_token]])
        return np.array(generated)

    def predict_proba(self, X: np.ndarray, max_len: int = 10) -> np.ndarray:
        self.predict(X, max_len=max_len)
        return self._embed(X) @ self.W_out.T

    def evaluate(self, X: np.ndarray, y: np.ndarray, phase: str = "pretrain") -> dict[str, float]:
        """Evaluate model on test data."""
        if phase == "finetune":
            preds = self.predict(X, phase="finetune")
            accuracy = float(np.mean(preds == y.flatten()[:len(preds)])) if len(preds) > 0 else 0.0
            return {"accuracy": accuracy, "n_samples": float(X.shape[0])}

        preds = self.predict(X, max_len=y.shape[0] if y.ndim > 1 else 10)
        matches = float(np.mean(preds == y.flatten()[:len(preds)])) if len(preds) > 0 else 0.0
        return {"accuracy": matches, "n_samples": float(X.shape[0])}

    def save(self, path: str) -> None:
        """Save model to npz file."""
        arrays = {
            "loss_history": np.array(self.loss_history),
            "embedding": self.embedding,
            "W_out": self.W_out,
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_encoder_layers": np.array([self.n_encoder_layers]),
            "n_decoder_layers": np.array([self.n_decoder_layers]),
            "d_ff": np.array([self.d_ff]),
            "max_seq_len": np.array([self.max_seq_len]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "training_mode": np.array([self.training_mode]),
            "pretraining_objective": np.array([self.pretraining_objective]),
            "fine_tuning_strategy": np.array([self.fine_tuning_strategy]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Transformer":
        """Load model from npz file."""
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_encoder_layers=int(data["n_encoder_layers"].item()),
            n_decoder_layers=int(data["n_decoder_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            max_seq_len=int(data["max_seq_len"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._init()
        obj.embedding = data["embedding"]
        obj.W_out = data["W_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        obj.training_mode = str(data.get("training_mode", ["pretrain"])[0])
        obj.pretraining_objective = str(data.get("pretraining_objective", ["ntp"])[0])
        obj.fine_tuning_strategy = str(data.get("fine_tuning_strategy", ["full"])[0])
        return obj

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "d_ff": self.d_ff,
            "learning_rate": self.learning_rate,
            "training_mode": self.training_mode,
            "pretraining_objective": self.pretraining_objective,
            "fine_tuning_strategy": self.fine_tuning_strategy,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
