"""Feedforward neural network for handwritten digit recognition (multi-class classification).

A multi-layer perceptron (MLP) with one hidden layer, trained via backpropagation
and batch gradient descent using softmax cross-entropy loss. Built from scratch
with NumPy — no external ML libraries.

Architecture:
    Input (64 pixels) -> Hidden (hidden_dim, ReLU) -> Output (10 classes, Softmax)

Loss: Categorical Cross-Entropy (softmax)
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from handwritten_digit_recognition.data import N_CLASSES


def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (z > 0).astype(z.dtype)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax activation with numerical stability."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Categorical cross-entropy loss.

    Args:
        y_true: One-hot encoded labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)

    Returns:
        Mean cross-entropy loss
    """
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


@dataclass
class DigitRecognitionNN:
    """Feedforward neural network for handwritten digit recognition.

    Architecture: Input -> Hidden (ReLU) -> Output (Softmax)

    Args:
        hidden_dim: Number of neurons in the hidden layer
        learning_rate: Gradient descent step size
        n_iterations: Maximum number of training iterations
        weight_decay: L2 regularization strength
        hidden_activation: Activation for hidden layer ('relu' or 'tanh')
        random_seed: Random seed for reproducibility
    """

    hidden_dim: int = 64
    learning_rate: float = 0.1
    n_iterations: int = 1000
    weight_decay: float = 0.0001
    hidden_activation: Literal["relu", "tanh"] = "relu"
    random_seed: int = 42

    input_dim: int = 0
    n_classes: int = N_CLASSES
    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None

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

        Returns: (probabilities, hidden_activations, z1)
        """
        z1 = np.dot(X, self.W1) + self.b1

        a1 = _relu(z1) if self.hidden_activation == "relu" else np.tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        probs = _softmax(z2)
        return probs, a1, z1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "DigitRecognitionNN":
        """Train the neural network using batch gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) — digit 0-9
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).flatten()

        n_samples, n_features = X.shape
        self.input_dim = n_features
        self.n_classes = N_CLASSES

        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y] = 1.0

        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self.mean_) / self.std_

        X_val_norm = None
        y_val_onehot = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val - self.mean_) / self.std_
            y_val_flat = np.asarray(y_val, dtype=int).flatten()
            y_val_onehot = np.zeros((len(y_val_flat), self.n_classes))
            y_val_onehot[np.arange(len(y_val_flat)), y_val_flat] = 1.0
            self.val_accuracy_history = []

        if self.hidden_activation == "relu":
            self.W1 = self._he_init(n_features, self.hidden_dim, rng)
        else:
            self.W1 = self._xavier_init(n_features, self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._xavier_init(self.hidden_dim, self.n_classes, rng)
        self.b2 = np.zeros(self.n_classes)

        self.loss_history = []

        for epoch in range(self.n_iterations):
            probs, a1, z1 = self._forward(X_norm)
            loss = _cross_entropy_loss(y_onehot, probs)

            l2_penalty = self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
            loss += l2_penalty
            self.loss_history.append(loss)

            m = n_samples
            dz2 = (probs - y_onehot) / m
            dW2 = np.dot(a1.T, dz2) + self.weight_decay * self.W2
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, self.W2.T)
            if self.hidden_activation == "relu":
                dz1 = da1 * _relu_derivative(z1)
            else:
                dz1 = da1 * (1 - np.tanh(z1) ** 2)

            dW1 = np.dot(X_norm.T, dz1) + self.weight_decay * self.W1
            db1 = np.sum(dz1, axis=0)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if X_val_norm is not None and y_val_flat is not None and epoch % 50 == 0:
                val_probs, _, _ = self._forward(X_val_norm)
                val_preds = np.argmax(val_probs, axis=1)
                val_acc = float(np.mean(val_preds == y_val_flat))
                self.val_accuracy_history.append(val_acc)

            if epoch > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-7:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        probs, _, _ = self._forward(X_norm)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted digit class for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def precision_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class precision."""
        y_pred = self.predict(X)
        precisions = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            tp = np.sum((y_pred == c) & (y == c))
            fp = np.sum((y_pred == c) & (y != c))
            precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precisions

    def recall_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class recall."""
        y_pred = self.predict(X)
        recalls = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            tp = np.sum((y_pred == c) & (y == c))
            fn = np.sum((y_pred != c) & (y == c))
            recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return recalls

    def f1_per_class(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-class F1 score."""
        precisions = self.precision_per_class(X, y)
        recalls = self.recall_per_class(X, y)
        f1s = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            if precisions[c] + recalls[c] > 0:
                f1s[c] = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
        return f1s

    def macro_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute macro-averaged F1 score."""
        return float(np.mean(self.f1_per_class(X, y)))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        y_pred = self.predict(X)
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(y, y_pred, strict=False):
            cm[true, pred] += 1
        return cm

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics."""
        y_pred = self.predict(X)
        accuracy = float(np.mean(y_pred == y))

        precisions = self.precision_per_class(X, y)
        recalls = self.recall_per_class(X, y)

        return {
            "accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(self.f1_per_class(X, y))),
            "per_class_precision": precisions.tolist(),
            "per_class_recall": recalls.tolist(),
            "per_class_f1": self.f1_per_class(X, y).tolist(),
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
            b2=self.b2,
            input_dim=np.array([self.input_dim]),
            n_classes=np.array([self.n_classes]),
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
    def load(cls, path: str) -> "DigitRecognitionNN":
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
        model.b2 = data["b2"]
        model.input_dim = int(data["input_dim"].item())
        model.n_classes = int(data["n_classes"].item())
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
            "n_classes": self.n_classes,
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
