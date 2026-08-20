"""Logistic Regression model for spam email classification."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LogisticRegression:
    """Logistic regression for binary classification (spam / not spam).

    Model: z = X·w + b,  p = sigmoid(z),  prediction = 1 if p >= threshold else 0
    """

    learning_rate: float = 0.1
    n_iterations: int = 2000
    weights: np.ndarray | None = None
    bias: float = 0.0
    loss_history: list[float] = field(default_factory=list)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of spam for each email."""
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 1 (spam) if probability >= threshold, else 0 (not spam)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train using gradient descent on Binary Cross-Entropy."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            probs = self.predict_proba(X)
            loss = -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
            self.loss_history.append(float(loss))

            dw = (1 / n_samples) * np.dot(X.T, (probs - y))
            db = (1 / n_samples) * np.sum(probs - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute precision (positive predictive value)."""
        predictions = self.predict(X)
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    def recall(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute recall (sensitivity)."""
        predictions = self.predict(X)
        tp = np.sum((predictions == 1) & (y == 1))
        fn = np.sum((predictions == 0) & (y == 1))
        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    def f1_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute F1 score."""
        p = self.precision(X, y)
        r = self.recall(X, y)
        if p + r == 0:
            return 0.0
        return float(2 * p * r / (p + r))

    def roc_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute ROC AUC approximation."""
        probs = self.predict_proba(X)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        rankings = np.argsort(-probs)
        sorted_y = y[rankings]
        rank_sum = np.sum(np.where(sorted_y == 1)[0] + 1)
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics."""
        return {
            "accuracy": self.accuracy(X, y),
            "precision": self.precision(X, y),
            "recall": self.recall(X, y),
            "f1": self.f1_score(X, y),
            "roc_auc": self.roc_auc(X, y),
        }

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.weights is None:
            raise ValueError("Cannot save untrained model")
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            loss_history=np.array(self.loss_history),
        )

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        """Load model parameters from disk."""
        data = np.load(path)
        model = cls(
            learning_rate=float(data["learning_rate"]),
            n_iterations=int(data["n_iterations"]),
        )
        model.weights = data["weights"]
        model.bias = float(data["bias"])
        model.loss_history = list(data["loss_history"])
        return model
