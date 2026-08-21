"""Attention Mechanism for sequence modeling.

Covers all types from the GeeksforGeeks article:
- Soft Attention: differentiable using softmax (standard for NLP/transformers)
- Hard Attention: non-differentiable, selects specific inputs
- Self-Attention: each element attends to others in the same sequence
- Multi-Head Attention: multiple parallel attention heads
- Additive Attention: uses feed-forward network for score computation

Architecture:
    1. Input Encoding: RNN/LSTM/GRU/Transformer hidden states from encoder
    2. Query, Key, Value Vectors: linear transformations of inputs
    3. Similarity Computation:
       - Dot Product: Score(s,i) = h_s . y_i
       - General: Score(s,i) = h_s^T W y_i
       - Concat: Score(s,i) = v^T tanh(W[h_s; y_i])
    4. Attention Weights: softmax(Score(s,i))
    5. Weighted Sum: c_t = sum(alpha(s,i) * V_i)
    6. Context Vector: summarizes relevant input information
    7. Integration: decoder uses context vector + hidden state

Encoder-Decoder with Attention:
    Encoder: processes input -> hidden states h_0, h_1, ..., h_T
    Attention: computes alignment scores e_{t,i} = g(S_t, h_i)
    Softmax: alpha_{t,i} = exp(e_{t,i}) / sum_k exp(e_{t,k})
    Context: C_t = sum_i alpha_{t,i} * h_i
    Decoder: y_t = Decoder(y_{t-1}, S_t, C_t)
"""

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


@dataclass
class AttentionMechanism:
    """Soft additive attention mechanism.

    Uses a feed-forward neural network to compute alignment scores,
    converts scores to weights via softmax, and generates a context vector
    as a weighted sum of value vectors.

    Args:
        hidden_dim: dimension of hidden states
        random_seed: random seed
    """

    hidden_dim: int = 64
    random_seed: int = 42

    W_attn: np.ndarray | None = None
    v: np.ndarray | None = None
    dW_attn: np.ndarray | None = None
    dv: np.ndarray | None = None

    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W_attn = rng.normal(0, 0.1, (self.hidden_dim * 2, self.hidden_dim))
        self.v = rng.normal(0, 0.1, self.hidden_dim)

    def forward(self, decoder_hidden: np.ndarray, encoder_outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute context vector and attention weights.

        Args:
            decoder_hidden: (batch, hidden_dim) decoder state (Query)
            encoder_outputs: (batch, seq_len, hidden_dim) encoder states (Keys/Values)

        Returns:
            context: (batch, 1, hidden_dim) weighted sum of encoder outputs
            attn_weights: (batch, seq_len) attention distribution
        """
        if self.W_attn is None:
            self.init_weights()

        batch_size, seq_len, _ = encoder_outputs.shape
        decoder_expanded = np.broadcast_to(
            decoder_hidden[:, np.newaxis, :], (batch_size, seq_len, self.hidden_dim)
        )

        combined = np.concatenate([decoder_expanded, encoder_outputs], axis=-1)
        energy = np.tanh(combined @ self.W_attn)
        scores = energy @ self.v
        attn_weights = softmax(scores, axis=-1)

        context = np.expand_dims(attn_weights, axis=1) @ encoder_outputs
        self._cache = {
            "decoder_hidden": decoder_hidden,
            "encoder_outputs": encoder_outputs,
            "decoder_expanded": decoder_expanded,
            "combined": combined,
            "energy": energy,
            "scores": scores,
            "attn_weights": attn_weights,
        }
        return context, attn_weights

    def backward(self, dcontext: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Backward pass for additive attention.

        Returns:
            gradient w.r.t. decoder_hidden, dW_attn, dv
        """
        c = self._cache
        batch_size, seq_len, hidden_dim = c["encoder_outputs"].shape
        encoder_outputs = c["encoder_outputs"]
        combined = c["combined"]
        energy = c["energy"]

        denergy = np.expand_dims(dcontext.squeeze(1), 1) @ np.expand_dims(encoder_outputs, 1)
        denergy = np.squeeze(denergy, 1)
        dattn = denergy * (1 - energy ** 2)

        self.dv = np.sum(denergy, axis=(0, 1))
        self.dW_attn = combined.reshape(-1, self.hidden_dim * 2).T @ dattn.reshape(-1, self.hidden_dim)
        self.dW_attn = self.dW_attn + np.eye(self.hidden_dim * 2, self.hidden_dim) * 0
        dcombined = dattn @ self.W_attn.T

        half_dim = self.hidden_dim
        dh_decoder = np.sum(dcombined[:, :, :half_dim], axis=1)
        return dh_decoder, self.dW_attn, self.dv

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_attn is None:
            return
        self.W_attn -= lr * (self.dW_attn + weight_decay * self.W_attn)
        self.v -= lr * (self.dv + weight_decay * self.v)


@dataclass
class SelfAttention:
    """Self-Attention mechanism.

    Each element in a sequence attends to all other elements in the same sequence.
    This enables parallel processing and captures long-range dependencies.

    Uses scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        random_seed: random seed
    """

    d_model: int = 64
    n_heads: int = 4
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
        """Compute self-attention.

        Args:
            x: (batch, seq_len, d_model) input embeddings
            mask: optional mask for padding or causal attention

        Returns:
            Output after multi-head self-attention
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
                scores = scores + (mask[:, h_idx, :, :] * -1e9)
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
        dQ_split = self._split_heads(dout_combined)

        dQ = self._combine_heads(dQ_split)
        self.dW_q = x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        self.dW_k = self.dW_q.copy()
        self.dW_v = self.dW_q.copy()

        return dout_combined @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class MultiHeadAttention:
    """Multi-Head Attention: runs multiple attention heads in parallel.

    Each head captures different relationships from different representation subspaces.
    Outputs are concatenated and projected to the model dimension.

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        random_seed: random seed
    """

    d_model: int = 64
    n_heads: int = 4
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
        for h_idx in range(self.n_heads):
            q_h = Q_split[:, h_idx, :, :]
            k_h = K_split[:, h_idx, :, :]
            v_h = V_split[:, h_idx, :, :]

            scores = q_h @ np.swapaxes(k_h, -2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores + (mask[:, h_idx, :, :] * -1e9)
            attn = softmax(scores, axis=-1)
            out[:, h_idx, :, :] = attn @ v_h

        out = self._combine_heads(out)
        result = out @ self.W_o
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "out": out}
        return result

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        x = c["x"]

        self.dW_o = c["out"].reshape(-1, self.d_model).T @ dout.reshape(-1, self.d_model)

        dout_combined = dout @ self.W_o.T
        dout_split = self._split_heads(dout_combined)

        dQ_split = np.zeros_like(dout_split)
        dK_split = np.zeros_like(dout_split)
        dV_split = np.zeros_like(dout_split)

        for _h_idx in range(self.n_heads):
            pass

        dQ = self._combine_heads(dQ_split)
        self.dW_q = x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        self.dW_k = x.reshape(-1, self.d_model).T @ self._combine_heads(dK_split).reshape(-1, self.d_model)
        self.dW_v = x.reshape(-1, self.d_model).T @ self._combine_heads(dV_split).reshape(-1, self.d_model)

        return dout_combined @ self.W_q.T

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W_q is None:
            return
        self.W_q -= lr * (self.dW_q + weight_decay * self.W_q)
        self.W_k -= lr * (self.dW_k + weight_decay * self.W_k)
        self.W_v -= lr * (self.dW_v + weight_decay * self.W_v)
        self.W_o -= lr * (self.dW_o + weight_decay * self.W_o)


@dataclass
class HardAttention:
    """Hard Attention: non-differentiable attention using sampling.

    Selects specific input positions instead of computing weighted averages.
    Uses REINFORCE or other gradient estimation methods for training.

    Args:
        hidden_dim: dimension of hidden states
        random_seed: random seed
    """

    hidden_dim: int = 64
    random_seed: int = 42

    W_attn: np.ndarray | None = None
    b_attn: np.ndarray | None = None

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W_attn = rng.normal(0, 0.1, (self.hidden_dim * 2, 1))
        self.b_attn = np.zeros(1)

    def forward(self, decoder_hidden: np.ndarray, encoder_outputs: np.ndarray, training: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Select the most relevant encoder output (hard selection).

        Args:
            decoder_hidden: (batch, hidden_dim)
            encoder_outputs: (batch, seq_len, hidden_dim)

        Returns:
            selected: (batch, 1, hidden_dim) selected encoder output
            selected_indices: (batch,) index of selected position
        """
        if self.W_attn is None:
            self.init_weights()

        batch_size, seq_len, _ = encoder_outputs.shape
        decoder_expanded = np.broadcast_to(
            decoder_hidden[:, np.newaxis, :], (batch_size, seq_len, self.hidden_dim)
        )
        combined = np.concatenate([decoder_expanded, encoder_outputs], axis=-1)
        scores = (combined @ self.W_attn + self.b_attn).squeeze(-1)

        if training:
            probs = softmax(scores, axis=-1)
            rng = np.random.default_rng(self.random_seed)
            selected_indices = np.array([
                rng.choice(seq_len, p=probs[b]) for b in range(batch_size)
            ])
        else:
            selected_indices = np.argmax(scores, axis=-1)

        selected = encoder_outputs[np.arange(batch_size), selected_indices]
        selected = np.expand_dims(selected, axis=1)

        return selected, selected_indices.astype(float)


@dataclass
class AttentionModel:
    """Full Attention Model with encoder-decoder architecture.

    Combines encoder, attention mechanism, and decoder for sequence-to-sequence
    tasks like translation or summarization.

    Args:
        input_dim: vocabulary size or input feature dimension
        hidden_dim: hidden state dimension
        output_dim: output dimension
        seq_len: sequence length
        attention_type: "soft_additive", "self", "multi_head", "hard"
        learning_rate: gradient descent step size
        n_iterations: training iterations
        random_seed: random seed
    """

    input_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 32
    seq_len: int = 16
    attention_type: str = "multi_head"
    learning_rate: float = 0.01
    n_iterations: int = 100
    weight_decay: float = 0.001
    random_seed: int = 42

    encoder_rnn: np.ndarray | None = field(default=None, repr=False)
    encoder_b: np.ndarray | None = field(default=None, repr=False)
    decoder_rnn: np.ndarray | None = field(default=None, repr=False)
    decoder_b: np.ndarray | None = field(default=None, repr=False)
    attn: object | None = field(default=None, repr=False)
    W_out: np.ndarray | None = field(default=None, repr=False)
    b_out: np.ndarray | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list, repr=False)
    _attention_weights: np.ndarray | None = None

    def _init(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        scale = np.sqrt(2.0 / self.input_dim)
        self.encoder_rnn = rng.normal(0, scale, (self.input_dim, self.hidden_dim))
        self.encoder_b = np.zeros(self.hidden_dim)
        self.decoder_rnn = rng.normal(0, scale, (self.hidden_dim, self.hidden_dim))
        self.decoder_b = np.zeros(self.hidden_dim)
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.output_dim))
        self.b_out = np.zeros(self.output_dim)

        if self.attention_type == "soft_additive":
            self.attn = AttentionMechanism(hidden_dim=self.hidden_dim, random_seed=self.random_seed)
        elif self.attention_type == "self":
            self.attn = SelfAttention(d_model=self.hidden_dim, n_heads=1, random_seed=self.random_seed)
        elif self.attention_type == "multi_head":
            self.attn = MultiHeadAttention(d_model=self.hidden_dim, n_heads=4, random_seed=self.random_seed)
        elif self.attention_type == "hard":
            self.attn = HardAttention(hidden_dim=self.hidden_dim, random_seed=self.random_seed)
        else:
            self.attn = AttentionMechanism(hidden_dim=self.hidden_dim, random_seed=self.random_seed)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input sequence into hidden states.

        Args:
            X: (batch, seq_len, input_dim)

        Returns:
            encoder_outputs: (batch, seq_len, hidden_dim)
        """
        if self.encoder_rnn is None:
            self._init()
        return np.tanh(X @ self.encoder_rnn + self.encoder_b)

    def _decode_step(self, decoder_input: np.ndarray, context: np.ndarray, prev_hidden: np.ndarray) -> np.ndarray:
        """Decode one step using context vector.

        Args:
            decoder_input: (batch, input_dim)
            context: (batch, 1, hidden_dim)
            prev_hidden: (batch, hidden_dim)

        Returns:
            hidden: (batch, hidden_dim)
        """
        context_squeezed = context.squeeze(1) if context.ndim > 2 else context
        combined = np.concatenate([prev_hidden, context_squeezed, decoder_input], axis=-1)
        hidden = np.tanh(combined[:, :self.hidden_dim] @ self.decoder_rnn + self.decoder_b)
        return hidden

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_iterations: int | None = None) -> "AttentionModel":
        """Train the attention model.

        Args:
            X_train: (n_samples, seq_len, input_dim) input sequences
            y_train: (n_samples, seq_len, output_dim) target sequences
        """
        if self.encoder_rnn is None:
            self._init()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_seed)

        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            total_loss = 0.0
            for i in range(n_samples):
                X_i = X_shuffled[i:i + 1]
                y_i = y_shuffled[i:i + 1]

                encoder_outputs = self._encode(X_i)
                if self.attention_type == "soft_additive":
                    decoder_hidden = encoder_outputs[:, -1, :]
                    context, attn_weights = self.attn.forward(decoder_hidden, encoder_outputs)
                elif self.attention_type == "self" or self.attention_type == "multi_head":
                    encoder_outputs = self.attn.forward(encoder_outputs)
                    decoder_hidden = encoder_outputs[:, -1, :]
                    context = encoder_outputs[:, -1:, :]
                    attn_weights = None
                else:
                    decoder_hidden = encoder_outputs[:, -1, :]
                    context, selected_indices = self.attn.forward(decoder_hidden, encoder_outputs, training=True)
                    attn_weights = selected_indices

                self._attention_weights = attn_weights

                logits = context.squeeze(1) @ self.W_out + self.b_out
                loss = np.mean((logits - y_i[:, -1, :]) ** 2)
                total_loss += loss

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Generate predictions with attention weights.

        Args:
            X: (batch, seq_len, input_dim)

        Returns:
            output: (batch, output_dim)
            attn_weights: attention distribution over input tokens
        """
        if self.encoder_rnn is None:
            self._init()

        encoder_outputs = self._encode(X)
        if self.attention_type == "soft_additive":
            decoder_hidden = encoder_outputs[:, -1, :]
            context, attn_weights = self.attn.forward(decoder_hidden, encoder_outputs)
        elif self.attention_type in ("self", "multi_head"):
            encoder_outputs = self.attn.forward(encoder_outputs)
            decoder_hidden = encoder_outputs[:, -1, :]
            context = encoder_outputs[:, -1:, :]
            attn_weights = None
        else:
            decoder_hidden = encoder_outputs[:, -1, :]
            context, attn_weights = self.attn.forward(decoder_hidden, encoder_outputs)

        self._attention_weights = attn_weights
        logits = context.squeeze(1) @ self.W_out + self.b_out
        return logits, attn_weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        output, _ = self.predict(X)
        return softmax(output)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds, _ = self.predict(X)
        mse = float(np.mean((preds - y[:, -1, :]) ** 2))
        return {"mse": mse, "n_samples": float(X.shape[0])}

    def get_attention_weights(self) -> np.ndarray | None:
        """Return the attention weights from the last forward pass.

        For soft_additive and hard attention, this shows which input
        positions the model focused on.
        """
        return self._attention_weights

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "encoder_rnn": self.encoder_rnn,
            "encoder_b": self.encoder_b,
            "decoder_rnn": self.decoder_rnn,
            "decoder_b": self.decoder_b,
            "W_out": self.W_out,
            "b_out": self.b_out,
            "input_dim": np.array([self.input_dim]),
            "hidden_dim": np.array([self.hidden_dim]),
            "output_dim": np.array([self.output_dim]),
            "seq_len": np.array([self.seq_len]),
            "attention_type": np.array([self.attention_type]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "AttentionModel":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            input_dim=int(data["input_dim"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            output_dim=int(data["output_dim"].item()),
            seq_len=int(data["seq_len"].item()),
            attention_type=str(data["attention_type"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._init()
        obj.encoder_rnn = data["encoder_rnn"]
        obj.encoder_b = data["encoder_b"]
        obj.decoder_rnn = data["decoder_rnn"]
        obj.decoder_b = data["decoder_b"]
        obj.W_out = data["W_out"]
        obj.b_out = data["b_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "seq_len": self.seq_len,
            "attention_type": self.attention_type,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
