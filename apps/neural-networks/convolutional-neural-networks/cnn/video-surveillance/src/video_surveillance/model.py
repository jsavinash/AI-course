"""Video Surveillance using a Convolutional Neural Network.

Architecture:
    Input (1 x 8x8) -> Conv2D (1->8, 3x3, ReLU)
    -> MaxPool2D (2x2) -> Flatten -> Dense (32, ReLU) -> Dense (3, softmax)

Loss: categorical cross-entropy (many-to-one: classifies image into a class label)
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import SimpleCNN

from video_surveillance.data import reshape_image


@dataclass
class VideoSurveillanceCNN:
    """CNN for video surveillance.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input channels
        n_filters: Number of convolution filters
        kernel_size: Convolution kernel size
        hidden_dim: Hidden units in dense layer
        output_dim: Output dimension
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm for clipping
        random_seed: Random seed for reproducibility
    """

    IMG_SIZE: int = 8
    N_CHANNELS: int = 1
    n_filters: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    output_dim: int = 3
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleCNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "VideoSurveillanceCNN":
        """Train the CNN using backpropagation.

        Args:
            X: Image pixel arrays (n_samples, N_FEATURES)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_arr = np.asarray(y, dtype=float)

        if self.output_dim == 1:
            y_arr = y_arr.reshape(-1, 1)
        else:
            onehot = np.zeros((len(y_arr), self.output_dim))
            onehot[np.arange(len(y_arr)), y_arr.astype(int)] = 1.0
            y_arr = onehot

        self.model = SimpleCNN(
            input_shape=(self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE),
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_activation="softmax",
            output_loss="cross_entropy",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
        )
        self.model.fit(X_img, y_arr, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return output probabilities for each sample."""
        X_img = reshape_image(X)
        return self.model.predict_proba(X_img)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return predicted class indices."""
        if self.output_dim == 1:
            probas = self.predict_proba(X).flatten()
            return (probas >= threshold).astype(int)
        X_img = reshape_image(X)
        return self.model.predict(X_img)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Alias for predict_class."""
        return self.predict_class(X, threshold)

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
    def load(cls, path: str) -> "VideoSurveillanceCNN":
        model = SimpleCNN.load(path)
        obj = cls()
        obj.model = model
        obj.loss_history = model.loss_history
        obj.output_dim = model.output_dim
        return obj

    def to_dict(self) -> dict:
        return {
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
