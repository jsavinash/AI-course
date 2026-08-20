"""Text Generation implementation from scratch using NumPy.

Architecture:
    1. TextTokenizer: Tokenizes text into token IDs with vocabulary
    2. TextGeneratorModel: Transformer-based autoregressive model
    3. SamplingStrategy: Temperature, top-k, top-p sampling strategies
    4. GenerationConfig: Configuration for text generation

Core capabilities:
    - Autoregressive Generation: Predict next token given previous tokens
    - Transformer-based: Multi-head attention and feed-forward layers
    - Sampling Strategies: Greedy, temperature, top-k, top-p sampling
    - Prompt Conditioning: Generate text from seed prompts

Args:
    vocab_size: vocabulary size for text tokens
    d_model: model dimension
    n_heads: number of attention heads
    n_layers: number of transformer layers
    d_ff: feed-forward inner dimension
    max_seq_len: maximum sequence length
    temperature: sampling temperature for generation
    top_k: top-k sampling parameter
    top_p: nucleus sampling parameter
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
class TextTokenizer:
    vocab_size: int = 1000
    max_seq_len: int = 128
    random_seed: int = 42

    token_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    id_to_token: dict[int, str] = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<BOS>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        common_words = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did", "will", "would", "could",
                        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                        "for", "on", "with", "at", "by", "from", "as", "into", "through",
                        "during", "before", "after", "above", "below", "between", "out", "off",
                        "over", "under", "again", "further", "then", "once", "here", "there",
                        "when", "where", "why", "how", "all", "each", "every", "both", "few",
                        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
                        "own", "same", "so", "than", "too", "very", "just", "because", "but",
                        "and", "or", "if", "while", "although", "though", "even", "that", "this",
                        "it", "its", "he", "she", "they", "we", "you", "i", "me", "my", "your"]
        for i, word in enumerate(common_words):
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
class BaseTextModel:
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 128
    random_seed: int = 42

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
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.random_seed + i, trainable=True)
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
        logits = embedded @ self.W_out.T
        return logits


@dataclass
class SamplingStrategy:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    random_seed: int = 42

    def sample(self, logits: np.ndarray) -> int:
        rng = np.random.default_rng(self.random_seed)
        scaled = logits / self.temperature
        sorted_idx = np.argsort(scaled)[::-1]
        sorted_logits = scaled[sorted_idx]
        cumulative_probs = np.cumsum(softmax(sorted_logits))
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_idx[sorted_indices_to_remove]
        scaled[indices_to_remove] = -float("inf")
        if self.top_k > 0:
            top_k_idx = np.argsort(scaled)[-self.top_k:]
            scaled = np.full_like(scaled, -float("inf"))
            scaled[top_k_idx] = logits[top_k_idx] / self.temperature
        probs = softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


@dataclass
class TextGenerationModel:
    vocab_size: int = 1000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    random_seed: int = 42
    learning_rate: float = 0.001
    n_iterations: int = 100
    weight_decay: float = 0.01

    base_model: BaseTextModel | None = None
    tokenizer: TextTokenizer | None = None
    sampler: SamplingStrategy | None = None
    loss_history: list[float] = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _init(self) -> None:
        self.base_model = BaseTextModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            random_seed=self.random_seed,
        )
        self.tokenizer = TextTokenizer(vocab_size=self.vocab_size, max_seq_len=self.max_seq_len, random_seed=self.random_seed)
        self.sampler = SamplingStrategy(temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, random_seed=self.random_seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_iterations: int | None = None) -> dict:
        if self.base_model is None:
            self._init()
        if n_iterations is None:
            n_iterations = self.n_iterations
        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12
        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            total_loss = 0.0
            for i in range(n_samples):
                x = X_shuffled[i:i + 1]
                y = y_shuffled[i:i + 1]
                logits = self.base_model.forward(x)
                probs = softmax(logits[0])
                for pos in range(y.shape[1]):
                    true_token = int(y[0, pos])
                    if true_token < self.vocab_size:
                        total_loss += -np.log(np.clip(probs[pos, true_token], eps, 1))
            avg_loss = total_loss / (n_samples * max(1, y_train.shape[1]))
            self.loss_history.append(avg_loss)
        return self.to_dict()

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        if self.base_model is None or self.tokenizer is None or self.sampler is None:
            self._init()
        tokens = self.tokenizer.encode(prompt)
        generated = tokens[:]
        for _ in range(max_new_tokens):
            context = np.array([generated[-(self.max_seq_len - 1):]])
            logits = self.base_model.forward(context)
            next_token = self.sampler.sample(logits[0, -1, :])
            generated.append(next_token)
            if next_token == self.tokenizer.token_to_id.get("<EOS>", -1):
                break
        return self.tokenizer.decode(generated)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = []
        for i in range(len(X)):
            x = X[i:i + 1]
            logits = self.base_model.forward(x)
            pred = int(np.argmax(logits[0, 0, :]))
            preds.append(pred)
        matches = np.mean(np.array(preds) == y.flatten()[:len(preds)]) if len(preds) > 0 else 0.0
        return {"accuracy": float(matches), "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_layers": np.array([self.n_layers]),
            "d_ff": np.array([self.d_ff]),
            "max_seq_len": np.array([self.max_seq_len]),
            "temperature": np.array([self.temperature]),
            "top_k": np.array([self.top_k]),
            "top_p": np.array([self.top_p]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        if self.base_model and self.base_model.embedding is not None:
            arrays["embedding"] = self.base_model.embedding
        if self.base_model and self.base_model.W_out is not None:
            arrays["W_out"] = self.base_model.W_out
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "TextGenerationModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_layers=int(data["n_layers"].item()),
            d_ff=int(data["d_ff"].item()),
            max_seq_len=int(data["max_seq_len"].item()),
            temperature=float(data["temperature"].item()),
            top_k=int(data["top_k"].item()),
            top_p=float(data["top_p"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj._init()
        if "embedding" in data and obj.base_model:
            obj.base_model.embedding = data["embedding"]
        if "W_out" in data and obj.base_model:
            obj.base_model.W_out = data["W_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
