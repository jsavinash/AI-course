"""Recurrent neural network for sentiment analysis.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared mlops_shared.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, vocab_size) -> Hidden (hidden_dim, tanh) -> Output (1, sigmoid)

Loss: Binary Cross-Entropy (many-to-one: classifies sentiment at final timestep)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.rnn import SimpleRNN


@dataclass
class SentimentAnalysisRNN:
    """RNN for binary sentiment classification (many-to-one).

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

    vocab_size: int = 50
    seq_len: int = 10
    hidden_dim: int = 32
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _to_onehot(self, indices: np.ndarray, dim: int) -> np.ndarray:
        """Convert class indices to one-hot vectors."""
        result = np.zeros((len(indices), dim))
        for i, idx in enumerate(indices):
            result[i, int(idx) % dim] = 1.0
        return result

    def _to_onehot_seq(self, seq: np.ndarray, dim: int) -> np.ndarray:
        """Convert a 1-D sequence of indices to (seq_len, dim) one-hot."""
        seq = np.atleast_1d(seq).astype(int)
        result = np.zeros((len(seq), dim))
        result[np.arange(len(seq)), seq % dim] = 1.0
        return result

    def _to_onehot_batch(self, X: np.ndarray) -> np.ndarray:
        """Convert batch of index sequences to one-hot sequences."""
        n_samples = X.shape[0]
        seq_len = X.shape[1]
        result = np.zeros((n_samples, seq_len, self.vocab_size))
        for i in range(n_samples):
            result[i] = self._to_onehot_seq(X[i], self.vocab_size)
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SentimentAnalysisRNN":
        """Train the RNN with BPTT.

        Args:
            X: Token index sequences (n_samples, seq_len)
            y: Sentiment labels (n_samples,) — 1=positive, 0=negative

        Returns:
            self
        """
        X_onehot = self._to_onehot_batch(X)
        y_col = np.asarray(y, dtype=float).flatten()

        self.model = SimpleRNN(
            input_dim=self.vocab_size,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            output_activation="sigmoid",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="binary_crossentropy",
        )
        self.model.fit(X_onehot, y_col, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-sentiment probabilities for each sample."""
        X_oh = self._to_onehot_batch(X)
        probas = self.model.predict_proba(X_oh)
        return probas.flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 1 (positive) if probability >= threshold, else 0."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        tp = np.sum((preds == 1) & (y == 1))
        fn = np.sum((preds == 0) & (y == 1))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1_score(self, X: np.ndarray, y: np.ndarray) -> float:
        p, r = self.precision(X, y), self.recall(X, y)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": self.accuracy(X, y),
            "precision": self.precision(X, y),
            "recall": self.recall(X, y),
            "f1": self.f1_score(X, y),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "SentimentAnalysisRNN":
        model = SimpleRNN.load(path)
        obj = cls(
            vocab_size=model.input_dim,
            seq_len=10,
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
