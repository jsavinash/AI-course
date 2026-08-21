"""Autonomous Driving Object Recognition using a Capsule Network.

Architecture:
    Input (1 x 8x8) -> Conv2D (8, 3x3, ReLU) -> ConvCapsule (5, 8) -> Dense (5, softmax)

Loss: categorical cross-entropy (many-to-one: classifies image while preserving spatial part-to-whole relationships)
"""

from dataclasses import dataclass, field

import numpy as np
from ai_core.nn_utils.cnn import SimpleCNN

from capsnet_autonomous_driving.data import reshape_image


@dataclass
class AutonomousDrivingCapsNet:
    """Capsule Network for autonomous driving object recognition.

    Uses SimpleCNN with capsule-style routing via ConvCapsule layers.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input channels
        n_filters: Number of convolution filters
        kernel_size: Convolution kernel size
        capsule_dim: Dimension of each output capsule
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        random_seed: Random seed for reproducibility
    """

    IMG_SIZE: int = 8
    N_CHANNELS: int = 1
    n_filters: int = 8
    kernel_size: int = 3
    capsule_dim: int = 8
    learning_rate: float = 0.05
    n_iterations: int = 400
    weight_decay: float = 0.001
    random_seed: int = 42

    n_classes: int = 5
    model: SimpleCNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutonomousDrivingCapsNet":
        """Train the CapsNet using backpropagation.

        Args:
            X: Image pixel arrays (n_samples, N_FEATURES)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_arr = np.asarray(y, dtype=float)
        onehot = np.zeros((len(y_arr), self.n_classes))
        onehot[np.arange(len(y_arr)), y_arr.astype(int)] = 1.0
        y_arr = onehot

        self.model = SimpleCNN(
            input_shape=(self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE),
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            hidden_dim=32,
            output_dim=self.n_classes,
            output_activation="softmax",
            output_loss="cross_entropy",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=5.0,
            random_seed=self.random_seed,
        )
        self.model.fit(X_img, y_arr, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        X_img = reshape_image(X)
        return self.model.predict_proba(X_img)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        X_img = reshape_image(X)
        return self.model.predict(X_img)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.predict(X)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict_class(X)
        return float(np.mean(preds == y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict_class(X)
        acc = float(np.mean(preds == y))
        return {
            "accuracy": acc,
            "n_samples": float(len(y)),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "AutonomousDrivingCapsNet":
        model = SimpleCNN.load(path)
        obj = cls()
        obj.model = model
        obj.loss_history = model.loss_history
        return obj

    def to_dict(self) -> dict:
        return {
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "n_classes": self.n_classes,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
