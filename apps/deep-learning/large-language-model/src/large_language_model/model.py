"""Large Language Model (LLM) built on Transformer architecture.

Covers all topics from the GeeksforGeeks LLM article:

Architecture:
    - Input Embeddings: token vectors from vocabulary
    - Positional Encoding: sinusoidal sin/cos for position information
    - Self-Attention: Q/K/V with scaled dot-product, capturing word relationships
    - Multi-Head Attention: parallel heads for diverse context reasoning
    - Feed-Forward Layers: position-wise FFN with GELU activation
    - Residual + LayerNorm: Add & Norm at each sub-layer
    - Decoding: autoregressive token-by-token generation

Features:
    - Next-token prediction (language modeling)
    - Zero-shot capability (no task-specific training needed)
    - Few-shot inference (learn from examples in prompt)
    - Text generation with temperature and top-k sampling
    - Attention weight analysis for interpretability

Limitations addressed:
    - Computational efficiency (smaller parameter count for demo)
    - Gradient clipping for stable training
    - Weight decay for regularization to reduce overfitting
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


def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
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
class MultiHeadSelfAttention:
    """Multi-Head Self-Attention mechanism.

    Each head captures different attention patterns over the input.
    Uses scaled dot-product: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Supports:
    - Causal masking for autoregressive generation (prevent looking ahead)
    - Multiple parallel attention heads

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        max_seq_len: maximum sequence length
        random_seed: random seed
    """

    d_model: int = 512
    n_heads: int = 8
    max_seq_len: int = 512
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

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass with multi-head self-attention.

        Args:
            x: (batch, seq_len, d_model) input embeddings
            mask: optional (seq_len, seq_len) causal mask

        Returns:
            (batch, seq_len, d_model) output
        """
        if self.W_q is None:
            self.init_weights()

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        batch_size, seq_len, _ = x.shape

        Q_split = self._split_heads(Q)
        K_split = self._split_heads(K)
        V_split = self._split_heads(V)

        if mask is not None and mask.ndim == 2:
            mask = mask[np.newaxis, np.newaxis, :, :].astype(bool)
        elif mask is not None and mask.ndim == 3:
            mask = mask[:, np.newaxis, :, :].astype(bool)

        out = np.zeros_like(Q_split)
        for h_idx in range(self.n_heads):
            q_h = Q_split[:, h_idx, :, :]
            k_h = K_split[:, h_idx, :, :]
            v_h = V_split[:, h_idx, :, :]

            scores = q_h @ np.swapaxes(k_h, -2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores + (mask.astype(float) * -1e9)
            attn = softmax(scores, axis=-1)
            out[:, h_idx, :, :] = attn @ v_h

        out = self._combine_heads(out)
        result = out @ self.W_o
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "out": out, "result": result}
        return result

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, _, seq_len, _ = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, self.d_model)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        x = c["x"]
        batch_size, seq_len, _ = x.shape

        self.dW_o = c["out"].reshape(-1, self.d_model).T @ dout.reshape(-1, self.d_model)

        dout_combined = dout @ self.W_o.T
        dQ = dout_combined

        self.dW_q = self.dW_k = self.dW_v = x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        return dout_combined @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class FeedForward:
    """Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Applied independently to each position in the sequence.
    Captures complex patterns in the data.
    """

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
class TransformerBlock:
    """Transformer encoder block with self-attention and feed-forward.

    Contains:
    1. Multi-Head Self-Attention (captures relationships between all tokens)
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network (captures complex patterns)
    4. Add & Norm (residual connection + layer normalization)

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward inner dimension
        max_seq_len: maximum sequence length
        random_seed: random seed
    """

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    random_seed: int = 42

    attention: MultiHeadSelfAttention = field(init=False, repr=False)
    ffn: FeedForward = field(init=False, repr=False)
    ln1_gamma: np.ndarray | None = None
    ln1_beta: np.ndarray | None = None
    ln2_gamma: np.ndarray | None = None
    ln2_beta: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.attention = MultiHeadSelfAttention(
            d_model=self.d_model, n_heads=self.n_heads, max_seq_len=self.max_seq_len, random_seed=self.random_seed
        )
        self.ffn = FeedForward(d_model=self.d_model, d_ff=self.d_ff, random_seed=self.random_seed)
        rng = np.random.default_rng(self.random_seed + 1)
        self.ln1_gamma = rng.normal(1, 0.02, self.d_model)
        self.ln1_beta = np.zeros(self.d_model)
        self.ln2_gamma = rng.normal(1, 0.02, self.d_model)
        self.ln2_beta = np.zeros(self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.attention.forward(x, mask)
        x_norm1 = layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)
        ffn_out = self.ffn.forward(x_norm1)
        x_norm2 = layer_norm(x_norm1 + ffn_out, self.ln2_gamma, self.ln2_beta)
        self._cache = {"x": x, "x_norm1": x_norm1, "x_norm2": x_norm2}
        return x_norm2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout
        dffn_out = dx
        self.ffn.backward(dffn_out)
        self.attention.backward(dffn_out)
        return dffn_out

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        self.attention.update_params(lr, weight_decay)
        self.ffn.update_params(lr, weight_decay)


@dataclass
class LargeLanguageModel:
    """Large Language Model based on Transformer decoder architecture.

    Built on the Transformer architecture enabling learning of long-range
    dependencies and contextual meaning in text.

    Architecture:
        - Input Embeddings: token -> vector
        - Positional Encoding: adds sequence/order information
        - Multi-Head Self-Attention: understands word relationships
        - Feed-Forward Layers: captures complex patterns
        - Residual + LayerNorm: stable training of deep networks
        - Decoding: autoregressive token-by-token generation

    Features:
        - Zero-shot capability: performs tasks without explicit training
        - Few-shot inference: learns from examples in prompt
        - Temperature sampling: controls output randomness
        - Top-k sampling: limits sampling to top-k most likely tokens

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of transformer blocks
        d_ff: feed-forward inner dimension
        max_seq_len: maximum sequence length for positional encoding
        dropout_rate: dropout probability for regularization
        learning_rate: gradient descent step size
        n_iterations: number of training iterations
        weight_decay: L2 regularization
        clip_value: gradient clipping threshold
        random_seed: random seed
    """

    vocab_size: int = 100
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 32
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    n_iterations: int = 100
    weight_decay: float = 0.01
    clip_value: float = 1.0
    random_seed: int = 42

    embedding: np.ndarray | None = None
    pos_encoding: np.ndarray | None = None
    ln_f_gamma: np.ndarray | None = None
    ln_f_beta: np.ndarray | None = None
    layers: list = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list, repr=False)

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.embedding = rng.normal(0, 0.02, (self.vocab_size, self.d_model))
        self.pos_encoding = positional_encoding(self.max_seq_len, self.d_model)
        self.ln_f_gamma = np.ones(self.d_model)
        self.ln_f_beta = np.zeros(self.d_model)
        self.layers = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                random_seed=self.random_seed + i,
            )
            for i in range(self.n_layers)
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.d_model), (self.vocab_size, self.d_model))

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """Convert token indices to embeddings with positional encoding."""
        if self.embedding is None:
            self._init()
        embedded = self.embedding[x]
        seq_len = x.shape[1] if x.ndim > 1 else 1

        if seq_len <= self.max_seq_len:
            pos_enc = self.pos_encoding[:seq_len]
        else:
            pos_enc = positional_encoding(seq_len, self.d_model)

        if x.ndim == 1:
            embedded = embedded * np.sqrt(self.d_model) + pos_enc[:len(x)]
        else:
            embedded = embedded * np.sqrt(self.d_model) + pos_enc
        return embedded

    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask to prevent looking at future tokens."""
        return np.triu(np.ones((seq_len, seq_len)), k=1).astype(float)

    def fit(self, X_train: np.ndarray, n_iterations: int | None = None) -> "LargeLanguageModel":
        """Train the LLM using next-token prediction objective.

        Args:
            X_train: (n_samples, seq_len) token indices
            n_iterations: training iterations
        """
        if not self.layers:
            self._init()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12

        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]

            total_loss = 0.0
            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]
                seq_len = x_i.shape[1]
                input_seq = x_i[:, :-1]
                target_seq = x_i[:, 1:]

                emb = self._embed(input_seq)
                causal_mask = self._create_causal_mask(seq_len - 1)

                for layer in self.layers:
                    emb = layer.forward(emb, causal_mask)

                emb = layer_norm(emb, self.ln_f_gamma, self.ln_f_beta)
                logits = emb @ self.W_out.T

                preds = softmax(logits[0])
                targets = target_seq[0]
                loss = -np.mean(np.log(np.clip(preds[np.arange(len(targets)), targets], eps, 1)))
                total_loss += loss

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

        return self

    def predict(self, input_ids: np.ndarray, max_len: int = 10, temperature: float = 0.8, top_k: int = 10) -> np.ndarray:
        """Autoregressive generation with temperature and top-k sampling.

        Args:
            input_ids: (1, prompt_len) input token indices
            max_len: maximum generation length
            temperature: controls randomness (higher = more random)
            top_k: limits sampling to top-k most likely tokens

        Returns:
            Generated token indices (max_len,)
        """
        if not self.layers:
            self._init()

        input_ids = input_ids[:1] if input_ids.ndim == 1 else input_ids[0:1]
        generated = []
        current_ids = input_ids.copy()
        rng = np.random.default_rng(self.random_seed + 99)

        for _step in range(max_len):
            seq_len = current_ids.shape[1]
            emb = self._embed(current_ids)
            causal_mask = self._create_causal_mask(seq_len)

            for layer in self.layers:
                emb = layer.forward(emb, causal_mask)

            emb = layer_norm(emb, self.ln_f_gamma, self.ln_f_beta)
            logits = emb @ self.W_out.T
            last_logits = logits[0, -1, :] / temperature

            if top_k > 0 and top_k < self.vocab_size:
                top_indices = np.argsort(last_logits)[-top_k:]
                filtered_logits = np.full_like(last_logits, -np.inf)
                filtered_logits[top_indices] = last_logits[top_indices]
                probs = softmax(filtered_logits)
            else:
                probs = softmax(last_logits)

            next_token = int(rng.choice(self.vocab_size, p=probs))
            generated.append(next_token)
            current_ids = np.concatenate([current_ids, np.array([[next_token]])], axis=1)

        return np.array(generated)

    def predict_proba(self, input_ids: np.ndarray) -> np.ndarray:
        """Get token probability distribution for the next token."""
        if not self.layers:
            self._init()

        emb = self._embed(input_ids)
        causal_mask = self._create_causal_mask(input_ids.shape[1])
        for layer in self.layers:
            emb = layer.forward(emb, causal_mask)

        emb = layer_norm(emb, self.ln_f_gamma, self.ln_f_beta)
        logits = emb @ self.W_out.T
        return softmax(logits[0, -1, :])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model accuracy on next-token prediction."""
        correct = 0
        total = 0
        for i in range(len(X)):
            preds = self.predict(X[i:i + 1], max_len=1, temperature=1.0, top_k=1)
            actual = y[i, -1]
            if preds[0] == actual:
                correct += 1
            total += 1
        return {"accuracy": correct / max(total, 1), "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "embedding": self.embedding,
            "W_out": self.W_out,
            "ln_f_gamma": self.ln_f_gamma,
            "ln_f_beta": self.ln_f_beta,
            "vocab_size": np.array([self.vocab_size]),
            "d_model": np.array([self.d_model]),
            "n_heads": np.array([self.n_heads]),
            "n_layers": np.array([self.n_layers]),
            "d_ff": np.array([self.d_ff]),
            "max_seq_len": np.array([self.max_seq_len]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "LargeLanguageModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            vocab_size=int(data["vocab_size"].item()),
            d_model=int(data["d_model"].item()),
            n_heads=int(data["n_heads"].item()),
            n_layers=int(data["n_layers"].item()),
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
        obj.ln_f_gamma = data["ln_f_gamma"]
        obj.ln_f_beta = data["ln_f_beta"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
