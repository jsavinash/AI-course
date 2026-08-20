"""Recurrent neural network for weather forecasting.

A SimpleRNN (Elman network) trained with Backpropagation Through Time (BPTT).
Built from scratch with NumPy, using the shared mlops_shared.rnn.SimpleRNN base.

Architecture:
    Input (seq_len, n_features) -> Hidden (hidden_dim, tanh) -> Output (n_features, linear)

Loss: Mean Squared Error (many-to-one: predicts next-day weather vector)

The model learns temporal patterns in a sequence of weather measurements
(temperature, humidity, pressure, wind-speed, precipitation) and predicts
the weather vector for the next day.
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.rnn import SimpleRNN


@dataclass
class WeatherForecastingRNN:
    """RNN for multi-feature weather regression (many-to-one).

    Args:
        n_features: Number of weather features per timestep
        seq_len: Number of timesteps in input sequences
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_features: int = 5
    seq_len: int = 30
    hidden_dim: int = 32
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "WeatherForecastingRNN":
        """Train the RNN with BPTT.

        Args:
            X: Weather feature sequences (n_samples, seq_len, n_features)
            y: Target weather vectors (n_samples, n_features) — next-day values

        Returns:
            self
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Normalize features per-feature
        all_X = X.reshape(-1, self.n_features)
        self.feature_mean_ = all_X.mean(axis=0)
        self.feature_std_ = np.where(all_X.std(axis=0) < 1e-8, 1.0, all_X.std(axis=0))
        X_norm = (X - self.feature_mean_) / self.feature_std_

        # Normalize targets per-feature
        y_mean = y.mean(axis=0)
        y_std = np.where(y.std(axis=0) < 1e-8, 1.0, y.std(axis=0))
        y_norm = (y - y_mean) / y_std

        self.model = SimpleRNN(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_features,
            output_activation="linear",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="mse",
        )
        self.model.fit(X_norm, y_norm, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history

        # Store normalization params for prediction
        self._y_mean = y_mean
        self._y_std = y_std
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day weather vectors for a batch of sequences."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.feature_mean_) / self.feature_std_
        preds_norm = self.model.predict_proba(X_norm)
        return preds_norm * self._y_std + self._y_mean

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return float(np.mean((self.predict(X) - y) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return float(np.mean(np.abs(self.predict(X) - y)))

    def r2_score_per_feature(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "mse": self.mse(X, y),
            "rmse": self.rmse(X, y),
            "mae": self.mae(X, y),
            "r2": self.r2_score_per_feature(X, y),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)
        np.savez(
            path + ".norm.npz",
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            y_mean=self._y_mean,
            y_std=self._y_std,
        )

    @classmethod
    def load(cls, path: str) -> "WeatherForecastingRNN":
        model = SimpleRNN.load(path)

        feature_mean = None
        feature_std = None
        y_mean = None
        y_std = None
        try:
            norm_data = np.load(path + ".norm.npz")
            feature_mean = norm_data["feature_mean"]
            feature_std = norm_data["feature_std"]
            y_mean = norm_data["y_mean"]
            y_std = norm_data["y_std"]
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
        obj.feature_mean_ = feature_mean
        obj.feature_std_ = feature_std
        obj._y_mean = y_mean
        obj._y_std = y_std
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
