"""Recurrent neural network for music generation.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared nn_utils.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, vocab_size) -> Hidden (hidden_dim, tanh) -> Output (vocab_size, softmax)

Loss: Cross-Entropy (many-to-many: predicts next note at each timestep)

The model is trained to predict the next musical note (or rest) in a sequence,
enabling autoregressive music composition.
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN


@dataclass
class MusicGenerationRNN:
    """RNN language model for character-level music generation (many-to-many).

    Args:
        vocab_size: Number of possible note/rest tokens
        seq_len: Length of input note sequences
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    vocab_size: int = 40
    seq_len: int = 20
    hidden_dim: int = 32
    learning_rate: float = 0.1
    n_iterations: int = 500
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "self-supervised"
    loss_history: list[float] = field(default_factory=list)

    def _to_onehot_seq(self, seq: np.ndarray, dim: int) -> np.ndarray:
        seq = np.atleast_1d(seq).astype(int)
        result = np.zeros((len(seq), dim))
        result[np.arange(len(seq)), seq % dim] = 1.0
        return result

    def _to_onehot_batch(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        result = np.zeros((n_samples, seq_len, self.vocab_size))
        for i in range(n_samples):
            result[i] = self._to_onehot_seq(X[i], self.vocab_size)
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MusicGenerationRNN":
        """Train the RNN with BPTT.

        For music generation, y is derived from X by shifting by one position
        (predict next note).

        Args:
            X: Note index sequences (n_samples, seq_len)
            y: Optional explicit targets (n_samples, seq_len). If None, shift X.

        Returns:
            self
        """
        X_onehot = self._to_onehot_batch(X)

        # Next-token prediction: target = input shifted by 1
        y_shifted = np.roll(X, -1, axis=1) if y is None else y

        y_onehot = np.zeros((X_onehot.shape[0], X_onehot.shape[1], self.vocab_size))
        for i in range(X_onehot.shape[0]):
            for t in range(X_onehot.shape[1]):
                idx = int(y_shifted[i, t]) % self.vocab_size
                y_onehot[i, t, idx] = 1.0

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
        self.model.fit(X_onehot, y_onehot, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict next-note probabilities for each position."""
        X_oh = self._to_onehot_seq(X_seq, self.vocab_size)
        return self.model.predict_many_to_many(X_oh)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Greedy sampling: predict one note at each position."""
        probas = self.predict_proba(X_seq)
        return np.argmax(probas, axis=1)

    def generate(self, seed_seq: np.ndarray, n_tokens: int = 10) -> np.ndarray:
        """Autoregressively generate n_tokens following a seed sequence."""
        generated = list(seed_seq)
        current_seq = seed_seq.copy()

        for _ in range(n_tokens):
            X_oh = self._to_onehot_seq(current_seq, self.vocab_size)
            outputs = self.model.predict_many_to_many(X_oh)
            next_probs = outputs[-1]
            next_idx = int(np.argmax(next_probs))
            generated.append(next_idx)
            current_seq = np.array(generated[-self.seq_len :])

        return np.array(generated)

    def perplexity(self, X: np.ndarray) -> float:
        total_loss = 0.0
        n_tokens = 0
        for i in range(X.shape[0]):
            y_shifted = np.roll(X[i], -1)
            probas = self.predict_proba(X[i])
            for t in range(len(y_shifted) - 1):
                idx = int(y_shifted[t]) % self.vocab_size
                p = max(probas[t, idx], 1e-9)
                total_loss += -np.log(p)
                n_tokens += 1
        avg_loss = total_loss / max(n_tokens, 1)
        return float(np.exp(avg_loss))

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        return {"perplexity": self.perplexity(X), "n_sequences": float(X.shape[0])}

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "MusicGenerationRNN":
        model = SimpleRNN.load(path)
        obj = cls(
            vocab_size=model.input_dim,
            seq_len=20,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        return obj

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
