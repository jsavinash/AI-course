"""Recurrent neural network for speech recognition.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared nn_utils.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, n_mfcc_features) -> Hidden (hidden_dim, tanh) -> Output (n_chars, softmax)

Loss: Cross-Entropy (many-to-one: predicts word at final timestep)

This is a simplified speech-to-text model that classifies an audio feature
sequence into one of a small vocabulary of spoken words.
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN


@dataclass
class SpeechRecognitionRNN:
    """RNN for speech-to-text classification (many-to-one).

    Args:
        n_features: Number of acoustic features per timestep (e.g., MFCCs)
        seq_len: Number of timesteps in each audio sequence
        n_classes: Number of recognizable words
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_features: int = 16
    seq_len: int = 20
    n_classes: int = 10
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
        result = np.zeros((len(indices), dim))
        for i, idx in enumerate(indices):
            result[i, int(idx)] = 1.0
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SpeechRecognitionRNN":
        """Train the RNN with BPTT.

        Args:
            X: Audio feature sequences (n_samples, seq_len, n_features)
            y: Word class labels (n_samples,)

        Returns:
            self
        """
        y_onehot = self._to_onehot(y, self.n_classes)

        self.model = SimpleRNN(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes,
            output_activation="softmax",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="cross_entropy",
        )
        self.model.fit(X, y_onehot, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted word class indices."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        classes = np.unique(np.concatenate([y, preds]))
        precisions = []
        for c in classes:
            tp = np.sum((preds == c) & (y == c))
            fp = np.sum((preds == c) & (y != c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return float(np.mean(precisions))

    def recall(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        classes = np.unique(np.concatenate([y, preds]))
        recalls = []
        for c in classes:
            tp = np.sum((preds == c) & (y == c))
            fn = np.sum((preds != c) & (y == c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return float(np.mean(recalls))

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
    def load(cls, path: str) -> "SpeechRecognitionRNN":
        model = SimpleRNN.load(path)
        obj = cls(
            n_features=model.input_dim,
            seq_len=20,
            n_classes=model.output_dim,
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
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
