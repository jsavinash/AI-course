"""Transfer Learning model implementation from scratch using NumPy.

Architecture:
    1. BaseModel: Pre-trained model with frozen/trainable layers
    2. TransferClassifier: New task-specific head added on top of frozen base
    3. FineTuner: Handles gradual unfreezing of layers for fine-tuning

Core concepts:
    - Frozen Layers: weights kept fixed, preserve general features
    - Trainable Layers: weights updated during training, adapt to new task
    - Transfer Layers: new layers added for the target task
    - Fine-tuning: gradual unfreezing of top layers

Training objective:
    - Cross-entropy loss for classification
    - Layer-wise learning rates for fine-tuning

Args:
    base_model: pre-trained base model
    n_classes: number of output classes for new task
    freeze_base: whether to freeze base model initially
    fine_tune_layers: number of top layers to fine-tune
    learning_rate: learning rate for new layers
    fine_tune_lr: learning rate for fine-tuning base layers
    random_seed: random seed
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    n = probs.shape[0]
    clipped = np.clip(probs, eps, 1.0)
    if y_true.ndim == 0:
        return -np.log(clipped[int(y_true)])
    return -np.sum(np.log(clipped[np.arange(n), y_true])) / n


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


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
class DenseLayer:
    input_dim: int = 128
    output_dim: int = 64
    random_seed: int = 42
    trainable: bool = True

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.input_dim)
        self.W = rng.normal(0, scale, (self.input_dim, self.output_dim))
        self.b = np.zeros(self.output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            self.init_weights()
        out = x @ self.W + self.b
        self._cache = {"x": x, "out": out}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        self.dW = c["x"].T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None or not self.trainable:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class MultiHeadAttention:
    d_model: int = 128
    n_heads: int = 4
    random_seed: int = 42
    trainable: bool = True

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
        return dout @ self.W_o.T @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None or not self.trainable:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class AddNorm:
    d_model: int = 128
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
    d_model: int = 128
    d_ff: int = 512
    random_seed: int = 42
    trainable: bool = True

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
        if self.W1 is None or not self.trainable:
            return
        self.W1 -= lr * (self.dW1 + weight_decay * self.W1)
        self.b1 -= lr * self.db1
        self.W2 -= lr * (self.dW2 + weight_decay * self.W2)
        self.b2 -= lr * self.db2


@dataclass
class TransformerBlock:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
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
class BaseModel:
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 32
    random_seed: int = 42
    frozen: bool = True

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    layers: list[TransformerBlock] = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)
        self.layers = [
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.random_seed + i, trainable=not self.frozen)
            for i in range(self.n_layers)
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.vocab_size, self.d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        seq_len = x.shape[1] if x.ndim > 1 else 1
        embedded = self.embedding[x] * np.sqrt(self.d_model)
        if x.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(x)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]
        for layer in self.layers:
            embedded = layer.forward(embedded)
        self._features = embedded
        logits = embedded @ self.W_out.T
        return logits

    def get_features(self, x: np.ndarray) -> np.ndarray:
        if self.embedding is None:
            self._init()
        seq_len = x.shape[1] if x.ndim > 1 else 1
        embedded = self.embedding[x] * np.sqrt(self.d_model)
        if x.ndim == 1:
            embedded = embedded + self.pos_encoding[:len(x)]
        else:
            embedded = embedded + self.pos_encoding[:seq_len]
        for layer in self.layers:
            embedded = layer.forward(embedded)
        self._features = embedded
        return embedded

    def set_trainable(self, trainable: bool) -> None:
        self.frozen = not trainable
        for layer in self.layers:
            layer.trainable = trainable
            layer.self_attn.trainable = trainable
            layer.add_norm1.trainable = trainable
            layer.ffn.trainable = trainable
            layer.add_norm2.trainable = trainable

    def set_top_layers_trainable(self, n_top_layers: int) -> None:
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            trainable = i >= n_layers - n_top_layers
            layer.trainable = trainable
            layer.self_attn.trainable = trainable
            layer.add_norm1.trainable = trainable
            layer.ffn.trainable = trainable
            layer.add_norm2.trainable = trainable
        self.frozen = all(not l.trainable for l in self.layers)

    def get_trainable_params_count(self) -> int:
        count = 0
        for layer in self.layers:
            if layer.self_attn.W_q is not None and layer.self_attn.trainable:
                count += layer.self_attn.W_q.size + layer.self_attn.W_k.size + layer.self_attn.W_v.size + layer.self_attn.W_o.size
            if layer.ffn.W1 is not None and layer.ffn.trainable:
                count += layer.ffn.W1.size + layer.ffn.b1.size + layer.ffn.W2.size + layer.ffn.b2.size
        return count


@dataclass
class TransferClassifier:
    base_model: BaseModel | None = None
    n_classes: int = 10
    d_model: int = 128
    random_seed: int = 42
    trainable: bool = True

    pooling: DenseLayer | None = None
    classifier_head: DenseLayer | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        if self.base_model is None:
            raise ValueError("Base model must be provided")
        self.pooling = DenseLayer(self.d_model, self.d_model, self.random_seed, trainable=self.trainable)
        self.classifier_head = DenseLayer(self.d_model, self.n_classes, self.random_seed + 1, trainable=self.trainable)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        if self.pooling is None:
            self._init()
        features = self.base_model.get_features(x)
        pooled = np.mean(features, axis=1)
        hidden = gelu(self.pooling.forward(pooled))
        logits = self.classifier_head.forward(hidden)
        self._cache = {"pooled": pooled, "hidden": hidden, "logits": logits}
        return logits

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        dpooled = self.classifier_head.backward(dout)
        dhidden = dpooled * (c["hidden"] * (1 - c["hidden"]))
        dx = self.pooling.backward(dhidden)
        return dx

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.classifier_head is not None and self.classifier_head.trainable:
            self.classifier_head.update_params(lr, weight_decay)
        if self.pooling is not None and self.pooling.trainable:
            self.pooling.update_params(lr, weight_decay)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = trainable
        if self.pooling is not None:
            self.pooling.trainable = trainable
        if self.classifier_head is not None:
            self.classifier_head.trainable = trainable


@dataclass
class TransferLearningModel:
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_base_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 32
    n_classes: int = 10
    random_seed: int = 42
    freeze_base: bool = True
    fine_tune_layers: int = 0
    learning_rate: float = 0.001
    fine_tune_lr: float = 0.0001
    weight_decay: float = 0.01

    base_model: BaseModel | None = None
    transfer_classifier: TransferClassifier | None = None
    loss_history: list[float] = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        self.base_model = BaseModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_base_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            random_seed=self.random_seed,
            frozen=self.freeze_base,
        )
        self.transfer_classifier = TransferClassifier(
            base_model=self.base_model,
            n_classes=self.n_classes,
            d_model=self.d_model,
            random_seed=self.random_seed + 100,
            trainable=True,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_iterations: int = 100, fine_tune_at: int | None = None) -> dict:
        if self.base_model is None:
            self._init()

        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12

        if fine_tune_at is not None:
            self.base_model.set_top_layers_trainable(self.fine_tune_layers)

        for epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            total_loss = 0.0
            for i in range(n_samples):
                x = X_shuffled[i:i + 1]
                y = y_shuffled[i:i + 1]

                logits = self.transfer_classifier.forward(x)
                probs = softmax(logits[0])
                loss = cross_entropy_loss(probs, y[0])
                total_loss += loss

                dlogits = probs.copy()
                true_class = int(y[0])
                dlogits[true_class] -= 1
                dlogits = dlogits.reshape(1, -1)

                dx = self.transfer_classifier.backward(dlogits)

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            if fine_tune_at is not None and epoch == fine_tune_at:
                self.base_model.set_top_layers_trainable(self.fine_tune_layers)

        return self.to_dict()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            self._init()
        logits = self.transfer_classifier.forward(X)
        return np.argmax(logits, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            self._init()
        logits = self.transfer_classifier.forward(X)
        return softmax(logits)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy, "n_samples": float(len(y))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_base_layers": np.array([self.n_base_layers]),
            "d_ff": np.array([self.d_ff]),
            "max_seq_len": np.array([self.max_seq_len]),
            "n_classes": np.array([self.n_classes]),
            "random_seed": np.array([self.random_seed]),
            "freeze_base": np.array([1 if self.freeze_base else 0]),
            "fine_tune_layers": np.array([self.fine_tune_layers]),
            "learning_rate": np.array([self.learning_rate]),
            "fine_tune_lr": np.array([self.fine_tune_lr]),
            "weight_decay": np.array([self.weight_decay]),
        }
        if self.base_model and self.base_model.embedding is not None:
            arrays["base_embedding"] = self.base_model.embedding
        if self.transfer_classifier and self.transfer_classifier.classifier_head and self.transfer_classifier.classifier_head.W is not None:
            arrays["classifier_W"] = self.transfer_classifier.classifier_head.W
            arrays["classifier_b"] = self.transfer_classifier.classifier_head.b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "TransferLearningModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_base_layers=int(data["n_base_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            max_seq_len=int(data["max_seq_len"].item()),
            n_classes=int(data["n_classes"].item()),
            random_seed=int(data["random_seed"].item()),
            freeze_base=bool(data["freeze_base"].item()),
            fine_tune_layers=int(data["fine_tune_layers"].item()),
            learning_rate=float(data["learning_rate"].item()),
            fine_tune_lr=float(data["fine_tune_lr"].item()),
            weight_decay=float(data["weight_decay"].item()),
        )
        obj._init()
        if "base_embedding" in data and obj.base_model:
            obj.base_model.embedding = data["base_embedding"]
        if "classifier_W" in data and obj.transfer_classifier and obj.transfer_classifier.classifier_head:
            obj.transfer_classifier.classifier_head.W = data["classifier_W"]
            obj.transfer_classifier.classifier_head.b = data["classifier_b"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        base_trainable = self.base_model.get_trainable_params_count() if self.base_model else 0
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_base_layers": self.n_base_layers,
            "d_ff": self.d_ff,
            "n_classes": self.n_classes,
            "freeze_base": self.freeze_base,
            "fine_tune_layers": self.fine_tune_layers,
            "learning_rate": self.learning_rate,
            "fine_tune_lr": self.fine_tune_lr,
            "base_trainable_params": base_trainable,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
