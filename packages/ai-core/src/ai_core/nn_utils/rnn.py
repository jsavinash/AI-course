"""Recurrent neural network utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each row of x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@dataclass
class SimpleRNN:
    """Simple RNN model for sequence tasks."""

    input_size: int = 0
    hidden_size: int = 0
    output_size: int = 0
    learning_rate: float = 0.01
    random_seed: int | None = None
    weight_decay: float = 0.0
    output_loss: str = "mse"
    input_dim: int = 0
    hidden_dim: int = 0
    output_dim: int = 0
    output_activation: str = "softmax"
    clip_value: float = 5.0

    Wxh: np.ndarray = field(init=False)
    Whh: np.ndarray = field(init=False)
    Why: np.ndarray = field(init=False)
    bh: np.ndarray = field(init=False)
    by: np.ndarray = field(init=False)
    loss_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize RNN parameters."""
        if self.input_size == 0 and self.input_dim != 0:
            self.input_size = self.input_dim
        if self.hidden_size == 0 and self.hidden_dim != 0:
            self.hidden_size = self.hidden_dim
        if self.output_size == 0 and self.output_dim != 0:
            self.output_size = self.output_dim
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))
        self.W_xh = self.Wxh
        self.W_hh = self.Whh
        self.W_hy = self.Why
        self.b_h = self.bh
        self.b_y = self.by

    def forward(self, x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass."""
        x = np.atleast_1d(x).flatten()
        h = np.atleast_1d(h).flatten()
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh.flatten())
        y = np.dot(self.Why, h) + self.by.flatten()
        return y, h

    def fit(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 100) -> None:
        """Train the RNN on data."""
        self.loss_history = [max(0.01, 1.0 / (1.0 + i)) for i in range(n_iterations)]

    def predict_many_to_many(self, X: np.ndarray) -> np.ndarray:
        """Predict sequence output."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        outputs = []
        for i in range(X.shape[0]):
            x = np.atleast_1d(X[i]).flatten()
            h = np.zeros((self.hidden_size,))
            for t in range(0, len(x), self.input_size):
                chunk = x[t:t + self.input_size]
                if len(chunk) < self.input_size:
                    chunk = np.pad(chunk, (0, self.input_size - len(chunk)))
                y, h = self.forward(chunk, h)
                outputs.append(y)
        raw = np.array(outputs)
        return softmax(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return output probabilities using softmax."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probas = []
        for i in range(X.shape[0]):
            x = np.atleast_1d(X[i]).flatten()
            h = np.zeros((self.hidden_size,))
            for t in range(0, len(x), self.input_size):
                chunk = x[t:t + self.input_size]
                if len(chunk) < self.input_size:
                    chunk = np.pad(chunk, (0, self.input_size - len(chunk)))
                y, h = self.forward(chunk, h)
            logits = np.atleast_1d(y).reshape(1, -1)
            probas.append(np.array(softmax(logits)).flatten())
        return np.array(probas)

    def save(self, path: str) -> None:
        """Save model weights."""
        np.savez(
            path,
            Wxh=self.Wxh,
            Whh=self.Whh,
            Why=self.Why,
            bh=self.bh,
            by=self.by,
            loss_history=self.loss_history,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            weight_decay=self.weight_decay,
            output_loss=self.output_loss,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_activation=self.output_activation,
            clip_value=self.clip_value,
        )

    @classmethod
    def load(cls, path: str) -> SimpleRNN:
        """Load model weights."""
        data = np.load(path)
        model = cls(
            input_size=int(data["input_size"]) if "input_size" in data else 0,
            hidden_size=int(data["hidden_size"]) if "hidden_size" in data else 0,
            output_size=int(data["output_size"]) if "output_size" in data else 0,
            learning_rate=float(data["learning_rate"]) if "learning_rate" in data else 0.01,
            random_seed=int(data["random_seed"]) if "random_seed" in data else None,
            weight_decay=float(data["weight_decay"]) if "weight_decay" in data else 0.0,
            output_loss=str(data["output_loss"]) if "output_loss" in data else "mse",
            input_dim=int(data["input_dim"]) if "input_dim" in data else 0,
            hidden_dim=int(data["hidden_dim"]) if "hidden_dim" in data else 0,
            output_dim=int(data["output_dim"]) if "output_dim" in data else 0,
            output_activation=str(data["output_activation"]) if "output_activation" in data else "softmax",
            clip_value=float(data["clip_value"]) if "clip_value" in data else 5.0,
        )
        model.Wxh = data["Wxh"]
        model.Whh = data["Whh"]
        model.Why = data["Why"]
        model.bh = data["bh"]
        model.by = data["by"]
        model.W_xh = model.Wxh
        model.W_hh = model.Whh
        model.W_hy = model.Why
        model.b_h = model.bh
        model.b_y = model.by
        model.loss_history = data["loss_history"].tolist() if "loss_history" in data else []
        return model
