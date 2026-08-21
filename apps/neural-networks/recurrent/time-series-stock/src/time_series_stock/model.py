"""Recurrent neural network for stock market price prediction.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared nn_utils.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, n_features) -> Hidden (hidden_dim, tanh) -> Output (1, linear)

Loss: Mean Squared Error (many-to-one: predicts next price at final timestep)

The model learns temporal patterns in a sequence of feature vectors
(e.g., normalized open/high/low/close/volume) and predicts the next price.
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.rnn import SimpleRNN


@dataclass
class StockMarketRNN:
    """RNN for stock price regression (many-to-one).

    Args:
        n_features: Number of features per timestep (e.g., OHLCV = 5)
        seq_len: Number of timesteps in input sequences
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_features: int = 5
    seq_len: int = 20
    hidden_dim: int = 32
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    y_mean_: float | None = None
    y_std_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "StockMarketRNN":
        """Train the RNN with BPTT.

        Args:
            X: Feature sequences (n_samples, seq_len, n_features)
            y: Target prices (n_samples,)

        Returns:
            self
        """
        y = np.asarray(y, dtype=float).flatten()
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if y.std() > 1e-8 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        self.model = SimpleRNN(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            output_activation="linear",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="mse",
        )
        self.model.fit(X, y_norm, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stock prices for a batch of sequences."""
        preds_norm = self.model.predict_proba(X).flatten()
        return preds_norm * self.y_std_ + self.y_mean_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict (returns normalized then denormalized predictions)."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(X) - y) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.abs(self.predict(X) - y)))

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "mse": self.mse(X, y),
            "rmse": self.rmse(X, y),
            "mae": self.mae(X, y),
            "r2": self.r2_score(X, y),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)
        np.savez(
            path + ".norm.npz",
            y_mean=np.array([self.y_mean_ or 0.0]),
            y_std=np.array([self.y_std_ or 1.0]),
        )

    @classmethod
    def load(cls, path: str) -> "StockMarketRNN":
        model = SimpleRNN.load(path)
        y_mean = 0.0
        y_std = 1.0
        try:
            norm_data = np.load(path + ".norm.npz")
            y_mean = float(norm_data["y_mean"].item())
            y_std = float(norm_data["y_std"].item())
        except FileNotFoundError:
            pass

        obj = cls(
            n_features=model.input_dim,
            seq_len=20,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        obj.y_mean_ = y_mean
        obj.y_std_ = y_std
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
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
