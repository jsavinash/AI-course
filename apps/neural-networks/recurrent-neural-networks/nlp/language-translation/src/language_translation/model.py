"""Recurrent neural network for language translation.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared mlops_shared.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, vocab_size) -> Hidden (hidden_dim, tanh) -> Output (vocab_size, softmax)

Loss: Cross-Entropy (many-to-one: predicts translation token at final timestep)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.rnn import SimpleRNN


@dataclass
class LanguageTranslationRNN:
    """RNN for many-to-one language translation.

    Args:
        vocab_size: Size of the token vocabulary
        seq_len: Length of input token sequences
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    vocab_size: int = 40
    seq_len: int = 8
    hidden_dim: int = 32
    learning_rate: float = 0.1
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LanguageTranslationRNN":
        """Train the RNN with BPTT.

        Args:
            X: Token index sequences (n_samples, seq_len) as integers
            y: Target token indices (n_samples,) — the translated word

        Returns:
            self
        """
        X_onehot = self._to_onehot_batch(X)
        y_onehot = self._to_onehot(y, self.vocab_size)

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

        if X_val is not None and y_val is not None:
            val_acc = self.accuracy(X_val, y_val)
            self.loss_history.append(val_acc)  # track validation accuracy

        return self

    def _to_onehot(self, indices: np.ndarray, dim: int) -> np.ndarray:
        """Convert class indices to one-hot vectors."""
        result = np.zeros((len(indices), dim))
        for i, idx in enumerate(indices):
            result[i, int(idx)] = 1.0
        return result

    def _to_onehot_seq(self, seq: np.ndarray, dim: int) -> np.ndarray:
        """Convert a 1-D sequence of indices to (seq_len, dim) one-hot."""
        seq = np.atleast_1d(seq).astype(int)
        result = np.zeros((len(seq), dim))
        result[np.arange(len(seq)), seq] = 1.0
        return result

    def _to_onehot_batch(self, X: np.ndarray) -> np.ndarray:
        """Convert batch of index sequences to one-hot sequences.

        Args:
            X: (n_samples, seq_len) integer indices

        Returns:
            (n_samples, seq_len, vocab_size) one-hot
        """
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        result = np.zeros((n_samples, seq_len, self.vocab_size))
        for i in range(n_samples):
            result[i] = self._to_onehot_seq(X[i], self.vocab_size)
        return result

    def predict(self, X: np.ndarray) -> int:
        """Predict the translated token index for a single sequence."""
        X_oh = self._to_onehot_seq(X, self.vocab_size)
        outputs = self.model.predict_many_to_many(X_oh)
        return int(np.argmax(outputs[-1]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for each sample."""
        results = []
        for i in range(X.shape[0]):
            X_oh = self._to_onehot_seq(X[i], self.vocab_size)
            outputs = self.model.predict_many_to_many(X_oh)
            results.append(outputs[-1])
        return np.array(results)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = [self.predict(X[i]) for i in range(X.shape[0])]
        return float(np.mean(np.array(preds) == y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = np.array([self.predict(X[i]) for i in range(X.shape[0])]).flatten()
        y_flat = np.asarray(y).flatten()
        return {
            "accuracy": float(np.mean(preds == y_flat)),
            "n_samples": float(len(y_flat)),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "LanguageTranslationRNN":
        model = SimpleRNN.load(path)
        obj = cls(
            vocab_size=model.input_dim,
            seq_len=8,
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
