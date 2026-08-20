"""Self-supervised denoising autoencoder for server metrics anomaly detection.

The autoencoder is trained to reconstruct normal server metrics from a
corrupted version of themselves. This is self-supervised because the labels
are generated from the data itself (the original uncorrupted input).

At inference time, anomalies are detected via high reconstruction error -
the model has never seen anomalous patterns during training, so it cannot
reconstruct them accurately.
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


def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class DenoisingAutoencoder:
    """Denoising autoencoder for self-supervised anomaly detection.

    Architecture:
        Input (n_features) -> Hidden (hidden_dim, ReLU) -> Output (n_features, Linear)

    Self-supervised training:
        1. Take clean input X
        2. Corrupt X to get X_noisy
        3. Train to reconstruct X from X_noisy
        4. At inference: high reconstruction error => anomaly

    Args:
        hidden_dim: Size of the hidden (encoding) layer
        learning_rate: Gradient descent learning rate
        n_iterations: Number of training iterations
        hidden_activation: Activation for hidden layer ('relu' or 'tanh')
        noise_rate: Fraction of features to corrupt during training
        random_seed: Random seed for reproducibility
    """

    hidden_dim: int = 16
    learning_rate: float = 0.01
    n_iterations: int = 5000
    hidden_activation: Literal["relu", "tanh"] = "relu"
    noise_rate: float = 0.25
    random_seed: int = 42

    # Learned parameters
    input_dim: int = 0
    W1: np.ndarray | None = None
    b1: np.ndarray | None = None
    W2: np.ndarray | None = None
    b2: np.ndarray | None = None

    # Training metadata
    training_mode: Literal["self-supervised", "supervised"] = "self-supervised"
    loss_history: list[float] = field(default_factory=list)
    threshold: float = 1.0
    threshold_percentile: float = 95.0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    accuracy_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)

    def _he_init(
        self, n_in: int, n_out: int, rng: np.random.Generator
    ) -> np.ndarray:
        """He initialization for ReLU networks."""
        return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))

    def _xavier_init(
        self, n_in: int, n_out: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Xavier initialization for tanh networks."""
        return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))

    def _corrupt(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply corruption to input for self-supervised training.

        - Dropout: randomly zero out features
        - Gaussian noise: add noise to remaining features
        """
        X_noisy = X.copy()

        # Randomly zero out features (input dropout)
        mask = rng.random(X.shape) < self.noise_rate
        X_noisy[mask] = 0.0

        # Add Gaussian noise to all features
        X_noisy += rng.normal(0, 0.1, X_noisy.shape)

        return X_noisy

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass: encode -> decode.

        Returns: (output, hidden_activation, z1, dz1_activation)
        """
        z1 = np.dot(X, self.W1) + self.b1

        if self.hidden_activation == "relu":
            a1 = _relu(z1)
            da1 = _relu_derivative(z1)
        else:
            a1 = np.tanh(z1)
            da1 = 1 - np.tanh(z1) ** 2

        z2 = np.dot(a1, self.W2) + self.b2
        return z2, a1, z1, da1

    def fit(
        self,
        X: np.ndarray,
        X_val: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> "DenoisingAutoencoder":
        """Train the denoising autoencoder in a self-supervised manner.

        The self-supervised signal is reconstruction: we corrupt the input
        and train the model to recover the original.

        Args:
            X: Normal training data (n_samples, n_features)
            X_val: Optional validation data for monitoring
            X_test: Optional test data for tracking anomaly detection metrics
            y_test: Optional test labels for tracking accuracy

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        X = np.asarray(X, dtype=float)
        self.input_dim = X.shape[1]
        self.training_mode = "self-supervised"

        # Initialize weights
        if self.hidden_activation == "relu":
            self.W1 = self._he_init(X.shape[1], self.hidden_dim, rng)
        else:
            self.W1 = self._xavier_init(X.shape[1], self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._xavier_init(self.hidden_dim, X.shape[1], rng)
        self.b2 = np.zeros(X.shape[1])

        self.loss_history = []

        # Normalize training data
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_norm = (X - self.mean_) / self.std_

        # Normalize validation data if provided (for monitoring)
        X_val_norm = None
        if X_val is not None:
            X_val_norm = (X_val - self.mean_) / self.std_
            self.val_loss_history = []

        for epoch in range(self.n_iterations):
            # Self-supervised: create corrupted input, train to reconstruct original
            X_noisy = self._corrupt(X_norm, rng)

            z2, a1, z1, da1 = self._forward(X_noisy)
            loss = _mse_loss(X_norm, z2)
            self.loss_history.append(loss)

            # Backpropagation
            m = len(X_norm)
            dz2 = 2 * (z2 - X_norm) / m  # dL/dz2
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * da1

            dW1 = np.dot(X_noisy.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # Gradient descent
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            # Track validation loss for monitoring
            if X_val_norm is not None and epoch % 50 == 0:
                val_z2, _, _, _ = self._forward(X_val_norm)
                self.val_loss_history.append(_mse_loss(X_val_norm, val_z2))

            # Early stopping if converged
            if epoch > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-7:
                break

        # Compute reconstruction threshold from training data
        self._compute_threshold(X_norm)

        # Track accuracy on test set if provided
        if X_test is not None and y_test is not None:
            X_test_norm = (X_test - self.mean_) / self.std_
            errors = self.reconstruction_error_normalized(X_test_norm)
            predictions = (errors > self.threshold).astype(int)
            self.accuracy_history = [float(np.mean(predictions == y_test))]

        return self

    def _compute_threshold(self, X_norm: np.ndarray) -> None:
        """Compute anomaly threshold from reconstruction errors on training data."""
        errors = self.reconstruction_error_normalized(X_norm)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly probability (reconstruction error normalized).

        Higher reconstruction error => higher anomaly probability.
        """
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        errors = self.reconstruction_error_normalized(X_norm)

        # Normalize to [0, 1] using threshold as reference
        max_error = max(float(errors.max()), self.threshold * 2, 1e-8)
        proba = np.clip(errors / max_error, 0.0, 1.0)
        return proba

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        thr = threshold if threshold is not None else self.threshold
        errors = self.reconstruction_error(X)
        return (errors > thr).astype(int)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE)."""
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        return self.reconstruction_error_normalized(X_norm)

    def reconstruction_error_normalized(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute reconstruction error on normalized input."""
        if self.W1 is None:
            raise ValueError("Model not trained. Call fit() first.")

        # No corruption at inference time - reconstruct from clean input
        z2, _, _, _ = self._forward(X_norm)
        errors = np.mean((X_norm - z2) ** 2, axis=1)
        return errors

    def is_anomaly(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return boolean anomaly flags."""
        return self.predict(X, threshold=threshold).astype(bool)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float | None = None) -> dict[str, float]:
        """Evaluate anomaly detection performance.

        Args:
            X: Feature matrix
            y: Labels (0 = normal, 1 = anomaly)
            threshold: Custom threshold (defaults to self.threshold)

        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        predictions = self.predict(X, threshold=threshold)

        accuracy = float(np.mean(predictions == y))

        tp = int(np.sum((predictions == 1) & (y == 1)))
        fp = int(np.sum((predictions == 1) & (y == 0)))
        fn = int(np.sum((predictions == 0) & (y == 1)))
        tn = int(np.sum((predictions == 0) & (y == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "n_true_positives": tp,
            "n_false_positives": fp,
            "n_true_negatives": tn,
            "n_false_negatives": fn,
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
            hidden_dim=np.array([self.hidden_dim]),
            learning_rate=np.array([self.learning_rate]),
            n_iterations=np.array([self.n_iterations]),
            hidden_activation=np.array([self.hidden_activation]),
            noise_rate=np.array([self.noise_rate]),
            random_seed=np.array([self.random_seed]),
            threshold=np.array([self.threshold]),
            threshold_percentile=np.array([self.threshold_percentile]),
            mean_=self.mean_,
            std_=self.std_,
            loss_history=np.array(self.loss_history),
            val_loss_history=np.array(self.val_loss_history),
            training_mode=np.array([self.training_mode]),
        )

    @classmethod
    def load(cls, path: str) -> "DenoisingAutoencoder":
        """Load model parameters from disk."""
        data = np.load(path)

        model = cls(
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            hidden_activation=str(data["hidden_activation"].item()),
            noise_rate=float(data["noise_rate"].item()),
            random_seed=int(data["random_seed"].item()),
        )

        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.input_dim = int(data["input_dim"].item())
        model.threshold = float(data["threshold"].item())
        model.threshold_percentile = float(data["threshold_percentile"].item())
        model.mean_ = data["mean_"]
        model.std_ = data["std_"]
        model.loss_history = list(data["loss_history"])
        model.val_loss_history = list(data["val_loss_history"])
        model.training_mode = str(data["training_mode"].item())

        return model

    def to_dict(self) -> dict:
        """Return model configuration as a dict."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "hidden_activation": self.hidden_activation,
            "noise_rate": self.noise_rate,
            "random_seed": self.random_seed,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
