"""Linear Regression model for pizza price prediction.

Implements a production-ready linear regression with:
- Gradient descent training
- R² score computation
- Feature scaling
- Proper serialization with metadata
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearRegression:
    """Linear regression: price = weight * diameter + bias, trained via MSE gradient descent."""

    learning_rate: float = 0.001
    n_iterations: int = 2000
    weight: float = 0.0
    bias: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: y_hat = w * x + b."""
        return self.weight * X + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Train using gradient descent on Mean Squared Error."""
        n = len(X)
        self.loss_history = []
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(float(loss))

            dw = (2 / n) * np.sum(X * (y_pred - y))
            db = (2 / n) * np.sum(y_pred - y)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # ---------- Metrics ----------

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error on given data."""
        return float(np.mean((self.predict(X) - y) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(self.mse(X, y)))

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² (coefficient of determination) score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(self.predict(X) - y)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics."""
        return {
            "mse": self.mse(X, y),
            "rmse": self.rmse(X, y),
            "mae": self.mae(X, y),
            "r2": self.r2_score(X, y),
        }

    # ---------- Serialization ----------

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        np.savez(
            path,
            weight=self.weight,
            bias=self.bias,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            loss_history=np.array(self.loss_history),
        )

    @classmethod
    def load(cls, path: str) -> "LinearRegression":
        """Load model parameters from disk."""
        data = np.load(path)
        model = cls(
            learning_rate=float(data["learning_rate"]),
            n_iterations=int(data["n_iterations"]),
        )
        model.weight = float(data["weight"])
        model.bias = float(data["bias"])
        model.loss_history = list(data["loss_history"])
        return model

    def to_dict(self) -> dict[str, float]:
        """Return model parameters as a dict."""
        return {
            "weight": self.weight,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
        }
