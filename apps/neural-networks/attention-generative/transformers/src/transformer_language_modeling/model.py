"""Transformer-based language modeling using self-attention mechanisms.

Architecture:
    Input (batch, seq_len) token IDs -> Token Embedding + Positional Encoding
    -> Multi-Head Self-Attention (causal mask) -> Add & Norm
    -> Feed-Forward -> Add & Norm
    -> Dense (d_model, vocab_size) -> softmax

Loss: categorical cross-entropy (next-token prediction)
"""

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SimpleAttention:
    """Simplified self-attention layer for transformer language modeling.

    Args:
        d_model: Model dimension
        random_seed: Random seed
    """

    d_model: int = 32
    random_seed: int = 42
    W_q: np.ndarray | None = None
    W_k: np.ndarray | None = None
    W_v: np.ndarray | None = None
    W_o: np.ndarray | None = None
    dW_q: np.ndarray | None = None
    dW_k: np.ndarray | None = None
    dW_v: np.ndarray | None = None
    dW_o: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(1.0 / self.d_model)
        self.W_q = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_k = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_v = rng.normal(0, scale, (self.d_model, self.d_model))
        self.W_o = rng.normal(0, scale, (self.d_model, self.d_model))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with causal self-attention. X: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        if self.W_q is None:
            self._init_weights()

        batch, seq_len, d_model = X.shape

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores[:, causal_mask] = -1e9
        attn = softmax(scores)

        out = attn @ V
        out = out @ self.W_o

        self._cache = {"X": X, "Q": Q, "K": K, "V": V, "attn": attn, "batch": batch, "seq_len": seq_len}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        X = c["X"]
        batch = c["batch"]
        seq_len = c["seq_len"]
        d_model = X.shape[-1]

        self.dW_o = c["Q"].reshape(-1, d_model).T @ dout.reshape(-1, d_model) / batch
        dX_pre = dout @ self.W_o.T

        d_attn = dX_pre.reshape(batch, seq_len, seq_len) @ c["V"].transpose(0, 2, 1)
        dV = c["attn"].transpose(0, 2, 1) @ dX_pre

        dx_scores = attn_backward(d_attn, c["attn"], c["V"], seq_len, batch)
        dQ = dx_scores @ c["K"] / np.sqrt(d_model)
        dK = c["Q"].transpose(0, 2, 1) @ dx_scores / np.sqrt(d_model).T
        dV_total = dV

        X_flat = X.reshape(-1, d_model)
        dQ_flat = dQ.reshape(-1, d_model)
        dK_flat = dK.reshape(-1, d_model)
        dV_flat = dV_total.reshape(-1, d_model)

        self.dW_q = X_flat.T @ dQ_flat / batch
        self.dW_k = X_flat.T @ dK_flat / batch
        self.dW_v = X_flat.T @ dV_flat / batch

        dX = dX_pre @ self.W_q.T
        return dX

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


def attn_backward(d_attn: np.ndarray, attn: np.ndarray, V: np.ndarray, seq_len: int, batch: int) -> np.ndarray:
    """Compute gradient through softmax attention."""
    dx_scores = np.zeros_like(d_attn)
    for b in range(batch):
        for i in range(seq_len):
            for j in range(seq_len):
                for k in range(seq_len):
                    dx_scores[b, i, j] += attn[b, i, k] * (1 if k == j else 0) * d_attn[b, i, k]
    dx_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
    return dx_scores


@dataclass
class LayerNorm:
    gamma: np.ndarray | None = None
    beta: np.ndarray | None = None
    dgamma: np.ndarray | None = None
    dbeta: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            self.gamma = np.ones(X.shape[-1])
            self.beta = np.zeros(X.shape[-1])
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-5)
        X_norm = (X - mean) / std
        out = X_norm * self.gamma + self.beta
        self._cache = {"X": X, "X_norm": X_norm, "std": std, "mean": mean}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self._cache["X"]
        X_norm = self._cache["X_norm"]
        std = self._cache["std"]

        self.dgamma = np.sum(dout * X_norm, axis=tuple(range(dout.ndim - 1)))
        self.dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

        dX_norm = dout * self.gamma
        dX = (1.0 / std) * (
            dX_norm
            - np.mean(dX_norm, axis=-1, keepdims=True)
            - X_norm * np.mean(dX_norm * X_norm, axis=-1, keepdims=True)
        )
        return dX

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.gamma is None:
            return
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


@dataclass
class TransformerLanguageModel:
    """Transformer-based language model with self-attention.

    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        d_model: Model dimension
        hidden_dim: Feed-forward hidden dimension
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    vocab_size: int = 100
    seq_len: int = 16
    d_model: int = 32
    hidden_dim: int = 64
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    token_embedding: np.ndarray | None = None
    position_embedding: np.ndarray | None = None
    W_ff1: np.ndarray | None = None
    b_ff1: np.ndarray | None = None
    W_ff2: np.ndarray | None = None
    b_ff2: np.ndarray | None = None
    W_out: np.ndarray | None = None
    b_out: np.ndarray | None = None
    ln1: LayerNorm | None = None
    ln2: LayerNorm | None = None
    ln3: LayerNorm | None = None
    attn: SimpleAttention | None = None

    dW_ff1: np.ndarray | None = None
    db_ff1: np.ndarray | None = None
    dW_ff2: np.ndarray | None = None
    db_ff2: np.ndarray | None = None
    dW_out: np.ndarray | None = None
    db_out: np.ndarray | None = None
    dW_token_emb: np.ndarray | None = None

    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        emb_scale = np.sqrt(1.0 / self.d_model)
        self.token_embedding = rng.normal(0, emb_scale, (self.vocab_size, self.d_model))
        self.position_embedding = rng.normal(0, emb_scale, (self.seq_len, self.d_model))

        self.W_ff1 = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.d_model, self.hidden_dim))
        self.b_ff1 = np.zeros(self.hidden_dim)
        self.W_ff2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.d_model))
        self.b_ff2 = np.zeros(self.d_model)
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.d_model, self.vocab_size))
        self.b_out = np.zeros(self.vocab_size)

        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()
        self.ln3 = LayerNorm()
        self.attn = SimpleAttention(d_model=self.d_model, random_seed=self.random_seed)

    def _forward(self, tokens: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass for a batch of token sequences.

        Args:
            tokens: (batch, seq_len) integer token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
            cache: intermediate values for backward
        """
        tokens.shape[0]
        emb = self.token_embedding[tokens] + self.position_embedding[np.arange(self.seq_len)]

        normed = self.ln1.forward(emb)
        attn_out = self.attn.forward(normed)
        h = emb + attn_out
        ff_in = self.ln2.forward(h)
        ff_hidden = np.maximum(0, ff_in @ self.W_ff1 + self.b_ff1)
        ff_out = ff_hidden @ self.W_ff2 + self.b_ff2
        h2 = h + ff_out
        logits = self.ln3.forward(h2) @ self.W_out + self.b_out

        cache = {
            "tokens": tokens, "emb": emb, "attn_out": attn_out,
            "ff_in": ff_in, "ff_hidden": ff_hidden, "ff_out": ff_out,
            "h": h, "h2": h2, "ln1_out": self.ln1._cache, "ln2_out": self.ln2._cache,
            "ln3_out": self.ln3._cache,
        }
        return logits, cache

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int | None = None,
    ) -> "TransformerLanguageModel":
        """Train the transformer language model.

        Args:
            X: Input token IDs (batch, seq_len)
            y: Target token IDs (batch, seq_len) — shifted by one position for next-token prediction
        """
        if self.token_embedding is None:
            self._build()

        if n_iterations is None:
            n_iterations = self.n_iterations

        batch = X.shape[0]
        eps = 1e-12

        for _epoch in range(n_iterations):
            logits, cache = self._forward(X)

            logits_clipped = np.clip(softmax(logits), eps, 1 - eps)
            y_onehot = np.zeros_like(logits)
            for b in range(batch):
                for s in range(self.seq_len):
                    y_onehot[b, s, y[b, s]] = 1.0

            loss = -np.sum(y_onehot * np.log(logits_clipped)) / (batch * self.seq_len)
            self.loss_history.append(loss)

            dlogits = (softmax(logits) - y_onehot) / (batch * self.seq_len)

            ln3 = self.ln3
            d_h2_pre = ln3.backward(dlogits @ self.W_out.T)
            self.dW_out = cache["h2"].reshape(-1, self.d_model).T @ dlogits.reshape(-1, self.vocab_size) / batch
            self.db_out = np.sum(dlogits.reshape(-1, self.vocab_size), axis=0) / batch

            d_h2 = d_h2_pre
            d_ff_out = d_h2
            self.dW_ff2 = cache["ff_hidden"].reshape(-1, self.hidden_dim).T @ d_ff_out.reshape(-1, self.d_model) / batch
            self.db_ff2 = np.sum(d_ff_out.reshape(-1, self.d_model), axis=0) / batch
            d_ff_hidden = d_ff_out @ self.W_ff2.T
            d_ff_hidden_pre = d_ff_hidden * (cache["ff_hidden"] > 0)
            self.dW_ff1 = cache["ff_in"].reshape(-1, self.d_model).T @ d_ff_hidden_pre.reshape(-1, self.hidden_dim) / batch
            self.db_ff1 = np.sum(d_ff_hidden_pre.reshape(-1, self.hidden_dim), axis=0) / batch
            d_ff_in = d_ff_hidden_pre @ self.W_ff1.T

            ln2 = self.ln2
            d_h_attn = ln2.backward(d_ff_in)

            d_attn_out = self.attn.backward(d_h_attn)
            ln1 = self.ln1
            d_ln1_out = ln1.backward(d_attn_out)

            d_emb = d_ln1_out + d_h2_pre

            dW_emb = np.zeros_like(self.token_embedding)
            for b in range(batch):
                for s in range(self.seq_len):
                    dW_emb[X[b, s]] += d_emb[b, s]
            dW_emb /= batch
            dW_emb += self.weight_decay * self.token_embedding

            grad_norm = np.sqrt(
                np.sum(self.dW_out**2) + np.sum(self.dW_ff1**2) + np.sum(self.dW_ff2**2)
                + np.sum(dW_emb**2)
                + sum(np.sum(g**2) for g in [self.attn.dW_q, self.attn.dW_k, self.attn.dW_v, self.attn.dW_o] if g is not None)
            )
            if grad_norm > self.clip_value:
                scale = self.clip_value / (grad_norm + 1e-8)
                dW_emb *= scale
                self.dW_out *= scale
                self.dW_ff1 *= scale
                self.dW_ff2 *= scale

            self.W_out -= self.learning_rate * (self.dW_out + self.weight_decay * self.W_out)
            self.b_out -= self.learning_rate * self.db_out
            self.W_ff1 -= self.learning_rate * (self.dW_ff1 + self.weight_decay * self.W_ff1)
            self.b_ff1 -= self.learning_rate * self.db_ff1
            self.W_ff2 -= self.learning_rate * (self.dW_ff2 + self.weight_decay * self.W_ff2)
            self.b_ff2 -= self.learning_rate * self.db_ff2
            self.token_embedding -= self.learning_rate * dW_emb

            self.attn.W_q -= self.learning_rate * (self.attn.dW_q + self.weight_decay * self.attn.W_q)
            self.attn.W_k -= self.learning_rate * (self.attn.dW_k + self.weight_decay * self.attn.W_k)
            self.attn.W_v -= self.learning_rate * (self.attn.dW_v + self.weight_decay * self.attn.W_v)
            self.attn.W_o -= self.learning_rate * (self.attn.dW_o + self.weight_decay * self.attn.W_o)
            self.ln1.update_params(self.learning_rate)
            self.ln2.update_params(self.learning_rate)
            self.ln3.update_params(self.learning_rate)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return output probabilities for each sample."""
        logits, _ = self._forward(X)
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted token IDs."""
        logits, _ = self._forward(X)
        return np.argmax(logits, axis=-1)

    def perplexity(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute perplexity."""
        logits, _ = self._forward(X)
        probs = softmax(logits)
        batch, seq_len, _ = probs.shape
        eps = 1e-12
        nll = 0.0
        count = 0
        for b in range(batch):
            for s in range(seq_len):
                nll -= np.log(probs[b, s, y[b, s]] + eps)
                count += 1
        return float(np.exp(nll / max(count, 1)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        ppl = self.perplexity(X, y)
        return {"perplexity": ppl, "n_samples": float(X.shape[0])}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "W_ff1": self.W_ff1, "b_ff1": self.b_ff1,
            "W_ff2": self.W_ff2, "b_ff2": self.b_ff2,
            "W_out": self.W_out, "b_out": self.b_out,
            "ln1_gamma": self.ln1.gamma, "ln1_beta": self.ln1.beta,
            "ln2_gamma": self.ln2.gamma, "ln2_beta": self.ln2.beta,
            "ln3_gamma": self.ln3.gamma, "ln3_beta": self.ln3.beta,
            "attn_Wq": self.attn.W_q, "attn_Wk": self.attn.W_k,
            "attn_Wv": self.attn.W_v, "attn_Wo": self.attn.W_o,
            "vocab_size": np.array([self.vocab_size]),
            "seq_len": np.array([self.seq_len]),
            "d_model": np.array([self.d_model]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "TransformerLanguageModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            seq_len=int(data["seq_len"].item()),
            d_model=int(data["d_model"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._build()
        obj.token_embedding = data["token_embedding"]
        obj.position_embedding = data["position_embedding"]
        obj.W_ff1 = data["W_ff1"]
        obj.b_ff1 = data["b_ff1"]
        obj.W_ff2 = data["W_ff2"]
        obj.b_ff2 = data["b_ff2"]
        obj.W_out = data["W_out"]
        obj.b_out = data["b_out"]
        obj.ln1.gamma = data["ln1_gamma"]
        obj.ln1.beta = data["ln1_beta"]
        obj.ln2.gamma = data["ln2_gamma"]
        obj.ln2.beta = data["ln2_beta"]
        obj.ln3.gamma = data["ln3_gamma"]
        obj.ln3.beta = data["ln3_beta"]
        obj.attn.W_q = data["attn_Wq"]
        obj.attn.W_k = data["attn_Wk"]
        obj.attn.W_v = data["attn_Wv"]
        obj.attn.W_o = data["attn_Wo"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
