"""Feedforward neural network for email spam classification.

A multi-layer perceptron (MLP) with one hidden layer, trained via
backpropagation and batch gradient descent. Built from scratch with NumPy.

Architecture:
    Input (n_features) -> Hidden (hidden_dim, ReLU) -> Output (1, Sigmoid)

Loss: Binary Cross-Entropy
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (z > 0).astype(z.dtype)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation with numerical stability."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


@dataclass
class SpamDetectionNN:
    """Feedforward neural network for binary spam classification.

    Architecture: Input -> Hidden (ReLU) -> Output (Sigmoid)

    Args:
        hidden_dim: Number of neurons in the hidden layer
        learning_rate: Gradient descent step size
        n_iterations: Maximum number of training iterations
        weight_decay: L2 regularization strength
        random_seed: Random seed for reproducibility
    """

    hidden_dim: int = 16
    learning_rate: float = 0.01
    n_iterations: int = 1000
    weight_decay: float = 0.001
    hidden_activation: Literal["relu", "tanh"] = "relu"
    random_seed: int = 42

    # Learned parameters
    input_dim: int = 0
    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: float | None = None

    # Training metadata
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    val_accuracy_history: list[float] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _he_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """He initialization for ReLU networks."""
        return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))

    def _xavier_init(self, n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
        """Xavier initialization for tanh networks."""
        return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Returns: (output_probs, hidden_activations, z1)
        """
        z1 = np.dot(X, self.W1) + self.b1

        a1 = _relu(z1) if self.hidden_activation == "relu" else np.tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        output = _sigmoid(z2)
        return output, a1, z1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SpamDetectionNN":
        """Train the neural network using batch gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) — 1=spam, 0=ham
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()

        n_samples, n_features = X.shape
        self.input_dim = n_features

        # Normalize features
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self.mean_) / self.std_

        # Normalize validation set
        X_val_norm = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.mean_) / self.std_
            self.val_accuracy_history = []

        # Initialize weights
        if self.hidden_activation == "relu":
            self.W1 = self._he_init(n_features, self.hidden_dim, rng)
        else:
            self.W1 = self._xavier_init(n_features, self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._xavier_init(self.hidden_dim, 1, rng)
        self.b2 = 0.0

        self.loss_history = []

        for epoch in range(self.n_iterations):
            # Forward pass
            output, a1, z1 = self._forward(X_norm)
            loss = _binary_cross_entropy(y, output.flatten())

            # L2 regularization term
            l2_penalty = self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
            loss += l2_penalty

            self.loss_history.append(loss)

            # Backpropagation
            m = n_samples
            dz2 = (output.flatten() - y) / m  # dL/dz2
            dW2 = np.dot(a1.T, dz2.reshape(-1, 1)) + self.weight_decay * self.W2
            db2 = float(np.sum(dz2))

            da1 = np.dot(dz2.reshape(-1, 1), self.W2.T)
            if self.hidden_activation == "relu":
                dz1 = da1 * _relu_derivative(z1)
            else:
                dz1 = da1 * (1 - np.tanh(z1) ** 2)

            dW1 = np.dot(X_norm.T, dz1) + self.weight_decay * self.W1
            db1 = np.sum(dz1, axis=0)

            # Gradient descent
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            # Track validation accuracy
            if X_val_norm is not None and y_val is not None and epoch % 50 == 0:
                val_acc = self.accuracy(X_val_norm, y_val)
                self.val_accuracy_history.append(val_acc)

            # Early stopping
            if epoch > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-7:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return spam probability for each email."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        probs, _, _ = self._forward(X_norm)
        return probs.flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return 1 (spam) if probability >= threshold, else 0 (ham)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == y))

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
        if self.W1 is None:
            raise ValueError("Cannot save untrained model")

        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=np.array([self.b2]) if self.b2 is not None else np.array([0.0]),
            input_dim=np.array([self.input_dim]),
            hidden_dim=np.array([self.hidden_dim]),
            learning_rate=np.array([self.learning_rate]),
            n_iterations=np.array([self.n_iterations]),
            weight_decay=np.array([self.weight_decay]),
            hidden_activation=np.array([self.hidden_activation]),
            random_seed=np.array([self.random_seed]),
            mean_=self.mean_,
            std_=self.std_,
            loss_history=np.array(self.loss_history),
            val_accuracy_history=np.array(self.val_accuracy_history),
            training_mode=np.array([self.training_mode]),
        )

    @classmethod
    def load(cls, path: str) -> "SpamDetectionNN":
        """Load model parameters from disk."""
        data = np.load(path)

        model = cls(
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            hidden_activation=str(data["hidden_activation"].item()),
            random_seed=int(data["random_seed"].item()),
        )

        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = float(data["b2"].item())
        model.input_dim = int(data["input_dim"].item())
        model.mean_ = data["mean_"]
        model.std_ = data["std_"]
        model.loss_history = list(data["loss_history"])
        model.val_accuracy_history = list(data["val_accuracy_history"])
        model.training_mode = str(data["training_mode"].item())

        return model

    def to_dict(self) -> dict:
        """Return model configuration as a dict."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "hidden_activation": self.hidden_activation,
            "random_seed": self.random_seed,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
