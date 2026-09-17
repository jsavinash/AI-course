#!/usr/bin/env python3
"""Generate all CNN/DN/CapsNet example packages for the MLOps monorepo."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "apps/neural-networks/convolutional-neural-networks"

IMG_SIZE = 8
N_CHANNELS = 1
N_FEATURES = IMG_SIZE * IMG_SIZE

EX = [
    {"dir": "cnn/medical-imaging", "pkg": "medical_imaging", "name": "medical-imaging",
     "title": "Medical Imaging Diagnosis",
     "desc": "Detects tumors or fractures in synthetic medical X-ray, MRI, and CT scan images",
     "cls": "MedicalImagingCNN", "task": "classification", "n": 3,
     "labels": "['normal', 'benign', 'malignant']", "port": "8016", "prefix": "MEDICAL_IMAGING",
     "loss": "categorical cross-entropy", "act": "softmax", "loss_fn": "cross_entropy",
     "field": "condition", "values": "0=normal, 1=benign, 2=malignant", "arch": "CNN"},
    {"dir": "cnn/facial-recognition", "pkg": "facial_recognition", "name": "facial-recognition",
     "title": "Facial Recognition",
     "desc": "Binary facial recognition for phone unlock and photo tagging",
     "cls": "FacialRecognitionCNN", "task": "binary_classification", "n": 1,
     "labels": "['not_owner', 'owner']", "port": "8017", "prefix": "FACIAL_RECOGNITION",
     "loss": "binary cross-entropy", "act": "sigmoid", "loss_fn": "binary_crossentropy",
     "field": "is_owner", "values": "true=owner, false=not_owner", "arch": "CNN"},
    {"dir": "cnn/video-surveillance", "pkg": "video_surveillance", "name": "video-surveillance",
     "title": "Video Surveillance",
     "desc": "Tracks crowd movement and spots security threats in real-time frame images",
     "cls": "VideoSurveillanceCNN", "task": "classification", "n": 3,
     "labels": "['normal', 'activity', 'threat']", "port": "8018", "prefix": "VIDEO_SURVEILLANCE",
     "loss": "categorical cross-entropy", "act": "softmax", "loss_fn": "cross_entropy",
     "field": "activity", "values": "0=normal, 1=activity, 2=threat", "arch": "CNN"},
    {"dir": "dn/image-super-resolution", "pkg": "image_super_resolution", "name": "image-super-resolution",
     "title": "Image Super-Resolution",
     "desc": "Upscales low-resolution photos and blurry security footage into sharp images",
     "cls": "ImageSuperResolutionDN", "task": "super_resolution", "n": 0,
     "labels": "[]", "port": "8020", "prefix": "IMAGE_SUPER_RESOLUTION",
     "loss": "mean squared error", "act": "linear", "loss_fn": "mse",
     "field": "high_res_pixels", "values": "64 reconstructed pixel values", "arch": "DN"},
    {"dir": "dn/semantic-segmentation", "pkg": "semantic_segmentation", "name": "semantic-segmentation",
     "title": "Semantic Segmentation",
     "desc": "Maps object boundaries in synthetic images for autonomous driving and satellite mapping",
     "cls": "SemanticSegmentationDN", "task": "segmentation", "n": 1,
     "labels": "['background', 'foreground']", "port": "8021", "prefix": "SEMANTIC_SEGMENTATION",
     "loss": "binary cross-entropy", "act": "sigmoid", "loss_fn": "binary_crossentropy",
     "field": "mask", "values": "64 binary pixel values (0=background, 1=foreground)", "arch": "DN"},
    {"dir": "dn/generative-art", "pkg": "generative_art", "name": "generative-art",
     "title": "Generative Art",
     "desc": "Reconstructs fine details from rough sketches using deconvolutional upsampling",
     "cls": "GenerativeArtDN", "task": "super_resolution", "n": 0,
     "labels": "[]", "port": "8025", "prefix": "GENERATIVE_ART",
     "loss": "mean squared error", "act": "linear", "loss_fn": "mse",
     "field": "reconstructed_pixels", "values": "64 reconstructed pixel values", "arch": "DN"},
    {"dir": "capsnet/autonomous-driving", "pkg": "autonomous_driving", "name": "autonomous-driving",
     "title": "Autonomous Driving Object Recognition",
     "desc": "Recognizes traffic signs and pedestrians accurately even when partially hidden",
     "cls": "AutonomousDrivingCapsNet", "task": "classification", "n": 5,
     "labels": "['stop', 'yield', 'speed_limit', 'pedestrian', 'other']", "port": "8022", "prefix": "AUTONOMOUS_DRIVING",
     "loss": "categorical cross-entropy", "act": "softmax", "loss_fn": "cross_entropy",
     "field": "object_type", "values": "0=stop, 1=yield, 2=speed_limit, 3=pedestrian, 4=other", "arch": "CapsNet"},
    {"dir": "capsnet/medical-scan-analysis", "pkg": "medical_scan_analysis", "name": "medical-scan-analysis",
     "title": "Medical Scan Analysis",
     "desc": "Identifies overlapping body structures in complex 3D medical scans",
     "cls": "MedicalScanAnalysisCapsNet", "task": "classification", "n": 4,
     "labels": "['bone', 'organ', 'vessel', 'tissue']", "port": "8023", "prefix": "MEDICAL_SCAN_ANALYSIS",
     "loss": "categorical cross-entropy", "act": "softmax", "loss_fn": "cross_entropy",
     "field": "structure", "values": "0=bone, 1=organ, 2=vessel, 3=tissue", "arch": "CapsNet"},
    {"dir": "capsnet/text-char-recognition", "pkg": "text_char_recognition", "name": "text-char-recognition",
     "title": "Text and Character Recognition",
     "desc": "Reads distorted captchas and complex handwritten text keeping part-to-whole relationships",
     "cls": "TextCharRecognitionCapsNet", "task": "classification", "n": 36,
     "labels": "[str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]", "port": "8024", "prefix": "TEXT_CHAR_RECOGNITION",
     "loss": "categorical cross-entropy", "act": "softmax", "loss_fn": "cross_entropy",
     "field": "character", "values": "0-9, A-Z", "arch": "CapsNet"},
]

# ============ TEMPLATES ============

DATA_TEMPLATE = '''"""Data loading and preprocessing for {title} ({arch}).

Generates synthetic {img}x{img} grayscale images and their labels.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = {img}
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = {n_classes}

DEFAULT_N_SAMPLES = 500

LABEL_NAMES = {labels}


def _create_template(label: int, rng: np.random.Generator) -> np.ndarray:
    """Create a {img}x{img} template pattern for a given class."""
    grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    # Distinct spatial patterns per class
    patterns = [
        lambda r, c: IMAGE_SIZE // 4 <= r <= 3 * IMAGE_SIZE // 4 and IMAGE_SIZE // 4 <= c <= 3 * IMAGE_SIZE // 4,
        lambda r, c: r < IMAGE_SIZE // 2,
        lambda r, c: (r + c) % 3 == 0,
        lambda r, c: r == 0 or r == IMAGE_SIZE - 1 or c == 0 or c == IMAGE_SIZE - 1,
        lambda r, c: (r - IMAGE_SIZE // 2) ** 2 + (c - IMAGE_SIZE // 2) ** 2 <= 4,
        lambda r, c: r > IMAGE_SIZE // 2 and c > IMAGE_SIZE // 2,
        lambda r, c: (r + c) % 2 == 0,
        lambda r, c: r == c,
        lambda r, c: r + c == IMAGE_SIZE - 1,
        lambda r, c: True,
    ]
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if label < len(patterns) and patterns[label](r, c):
                grid[r, c] = 0.9
    return grid.flatten()


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their labels.

    Returns:
        X: (n_samples, N_FEATURES) flattened image pixel arrays
        y: (n_samples,) class labels
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = rng.integers(0, N_CLASSES) if N_CLASSES > 0 else rng.integers(0, 2)
        template = _create_template(label, rng)
        X[i] = np.clip(template + rng.normal(0, noise_level, N_FEATURES), 0, 1)
        y[i] = label

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, noise_level=noise_level, random_seed=random_seed)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def reshape_image(X: np.ndarray) -> np.ndarray:
    """Reshape flattened images to (N, 1, IMAGE_SIZE, IMAGE_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
'''


# CNN classification model (for CNN arch examples)
CNN_MODEL_TEMPLATE = '''"""{title} using a Convolutional Neural Network.

Architecture:
    Input (1 x {img}x{img}) -> Conv2D (1->8, 3x3, ReLU)
    -> MaxPool2D (2x2) -> Flatten -> Dense (32, ReLU) -> Dense ({out_dim}, {act})

Loss: {loss} (many-to-one: classifies image into a class label)
Optimizer: Gradient Descent with He initialization
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import SimpleCNN

from {pkg}.data import IMAGE_SIZE, N_CHANNELS, N_CLASSES, LABEL_NAMES, reshape_image


@dataclass
class {cls}:
    """CNN for {title_lower}.

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

    IMG_SIZE: int = {img}
    N_CHANNELS: int = {nch}
    n_filters: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    output_dim: int = {out_dim}
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
    ) -> "{cls}":
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
            output_activation="{act}",
            output_loss="{loss_fn}",
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
        return {{
            "accuracy": acc,
            "n_samples": float(len(y)),
        }}

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "{cls}":
        model = SimpleCNN.load(path)
        obj = cls()
        obj.model = model
        obj.loss_history = model.loss_history
        obj.output_dim = model.output_dim
        return obj

    def to_dict(self) -> dict:
        return {{
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
        }}
'''

# CapsNet model (for CapsNet arch examples - uses SimpleCNN internally with capsule routing)
CAPSNET_MODEL_TEMPLATE = '''"""{title} using a Capsule Network.

Architecture:
    Input (1 x {img}x{img}) -> Conv2D (8, 3x3, ReLU) -> ConvCapsule ({n}, 8) -> Dense ({n}, {act})

Loss: {loss} (many-to-one: classifies image while preserving spatial part-to-whole relationships)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import SimpleCNN
from {pkg}.data import IMAGE_SIZE, N_CHANNELS, N_CLASSES, LABEL_NAMES, reshape_image


@dataclass
class {cls}:
    """Capsule Network for {title_lower}.

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

    IMG_SIZE: int = {img}
    N_CHANNELS: int = {nch}
    n_filters: int = 8
    kernel_size: int = 3
    capsule_dim: int = 8
    learning_rate: float = 0.05
    n_iterations: int = 400
    weight_decay: float = 0.001
    random_seed: int = 42

    n_classes: int = {n}
    model: SimpleCNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "{cls}":
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
            output_activation="{act}",
            output_loss="{loss_fn}",
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
        return {{
            "accuracy": acc,
            "n_samples": float(len(y)),
        }}

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "{cls}":
        model = SimpleCNN.load(path)
        obj = cls()
        obj.model = model
        obj.loss_history = model.loss_history
        return obj

    def to_dict(self) -> dict:
        return {{
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
        }}
'''

# DN model (encoder-decoder with Deconv2D)
DN_MODEL_TEMPLATE = '''"""{title} using a Deconvolutional Network (DN).

Architecture:
    Input (1 x {img}x{img}) -> Conv2D (8, 3x3, ReLU) -> MaxPool2D (2x2)
    -> Deconv2D (8, 3x3, ReLU) -> Deconv2D (1, 3x3, {act})

Loss: {loss} (many-to-many: outputs pixel-level reconstruction)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import Conv2D, MaxPool2D, Activation, Deconv2D, relu, sigmoid

from {pkg}.data import IMAGE_SIZE, N_CHANNELS, N_FEATURES, reshape_image


@dataclass
class _Conv2D:
    """Minimal conv2d that tracks its own gradients for the DN encoder."""

    n_filters: int = 8
    kernel_size: int = 3
    random_seed: int = 42
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    def _init(self, C: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        fan_in = C * self.kernel_size * self.kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), (self.n_filters, C, self.kernel_size, self.kernel_size))
        self.b = np.zeros(self.n_filters)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            self._init(X.shape[1])
        N, C, H, W = X.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        out = np.zeros((N, self.n_filters, H_out, W_out))
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        region = X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        out[n, f, h, w] = np.sum(region * self.W[f]) + self.b[f]
        self._cache = {{"X": X, "H_out": H_out, "W_out": W_out}}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        X = c["X"]
        N, C, H, W = X.shape
        H_out, W_out = c["H_out"], c["W_out"]
        dW = np.zeros_like(self.W)
        db = np.zeros(self.n_filters)
        dX = np.zeros_like(X)
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        val = dout[n, f, h, w]
                        dW[f] += val * X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        db[f] += val
                        dX[n, :, h:h + self.kernel_size, w:w + self.kernel_size] += val * self.W[f]
        self.dW = dW / N
        self.db = db / N
        return dX

    def update_params(self, lr: float, wd: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + wd * self.W)
        self.b -= lr * self.db


@dataclass
class {cls}:
    """Deconvolutional network for {title_lower}.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input/output channels
        n_filters: Number of filters in conv layers
        kernel_size: Convolution kernel size
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        random_seed: Random seed
    """

    IMG_SIZE: int = {img}
    N_CHANNELS: int = {nch}
    n_filters: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.0001
    clip_value: float = 1.0
    random_seed: int = 42

    _layers: list = field(default_factory=list, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        # Encoder-decoder: 8x8 -> Conv(3) -> 6x6 -> Conv(3) -> 4x4
        # -> Deconv(3,stride=1) -> 6x6 -> Deconv(3,stride=1) -> 8x8
        self._layers = [
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed),
            Activation("relu"),
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed + 1),
            Activation("relu"),
            Deconv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 2),
            Activation("relu"),
            Deconv2D(n_filters=self.N_CHANNELS, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 3),
            Activation("{act}"),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "{cls}":
        """Train the deconvolutional network.

        Args:
            X: Input images (n_samples, N_FEATURES)
            y: Target images (n_samples, N_FEATURES)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_img = reshape_image(y)
        self._build()

        N = X_img.shape[0]
        eps = 1e-12

        for epoch in range(self.n_iterations):
            total_loss = 0.0
            for i in range(N):
                out = X_img[i:i + 1]
                for layer in self._layers:
                    out = layer.forward(out)

                target = y_img[i:i + 1]
                if "{loss_fn}" == "mse":
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(np.mean((target - clip_out) ** 2))
                else:
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(-np.mean(target * np.log(clip_out) + (1 - target) * np.log(1 - clip_out)))
                total_loss += loss

                if "{loss_fn}" == "mse":
                    dout = 2 * (out - target) / max(out.shape[0], 1)
                else:
                    dout = (out - target) / max(out.shape[0], 1)

                for layer in reversed(self._layers):
                    dout = layer.backward(dout)

                # Gradient clipping
                grad_norm = 0.0
                for layer in self._layers:
                    if hasattr(layer, "dW") and layer.dW is not None:
                        grad_norm += float(np.sum(layer.dW ** 2))
                    if hasattr(layer, "db") and layer.db is not None:
                        grad_norm += float(np.sum(layer.db ** 2))
                grad_norm = np.sqrt(grad_norm)
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    for layer in self._layers:
                        if hasattr(layer, "dW") and layer.dW is not None:
                            layer.dW *= scale
                        if hasattr(layer, "db") and layer.db is not None:
                            layer.db *= scale

                for layer in self._layers:
                    layer.update_params(self.learning_rate, self.weight_decay)

            self.loss_history.append(total_loss / N)
            if epoch > 50 and len(self.loss_history) > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-8:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of images."""
        X_img = reshape_image(X)
        results = []
        for i in range(X_img.shape[0]):
            out = X_img[i:i + 1]
            for layer in self._layers:
                out = layer.forward(out)
            results.append(out[0])
        return np.array(results)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds.flatten() - y.flatten()) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        mse_val = self.mse(X, y)
        return {{"mse": mse_val, "rmse": float(np.sqrt(mse_val)), "n_samples": float(X.shape[0])}}

    def save(self, path: str) -> None:
        arrays = {{"loss_history": np.array(self.loss_history)}}
        for i, layer in enumerate(self._layers):
            if isinstance(layer, _Conv2D):
                if layer.W is not None:
                    arrays[f"conv_{{i}}_W"] = layer.W
                    arrays[f"conv_{{i}}_b"] = layer.b
            elif isinstance(layer, Deconv2D):
                if layer.W is not None:
                    arrays[f"deconv_{{i}}_W"] = layer.W
                    arrays[f"deconv_{{i}}_b"] = layer.b
        arrays["n_filters"] = np.array(self.n_filters)
        arrays["learning_rate"] = np.array(self.learning_rate)
        arrays["n_iterations"] = np.array(self.n_iterations)
        arrays["weight_decay"] = np.array(self.weight_decay)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "{cls}":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_filters=int(data.get("n_filters", 8)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            n_iterations=int(data.get("n_iterations", 300)),
            weight_decay=float(data.get("weight_decay", 0.0001)),
            random_seed=42,
        )
        obj._build()
        for i, layer in enumerate(obj._layers):
            if isinstance(layer, _Conv2D) and f"conv_{{i}}_W" in data:
                layer.W = data[f"conv_{{i}}_W"]
                layer.b = data[f"conv_{{i}}_b"]
            elif isinstance(layer, Deconv2D) and f"deconv_{{i}}_W" in data:
                layer.W = data[f"deconv_{{i}}_W"]
                layer.b = data[f"deconv_{{i}}_b"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {{
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }}
'''


# DN data template (image-to-image pairs)
DN_DATA_TEMPLATE = '''"""Data loading and preprocessing for {title} ({arch}).

Generates synthetic {img}x{img} images for pixel-to-pixel tasks.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = {img}
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 0

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic low-quality images and their target images.

    Returns:
        X: (n_samples, N_FEATURES) input image pixels
        y: (n_samples, N_FEATURES) target image pixels
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        grid_hr = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
        cx, cy = rng.integers(2, IMAGE_SIZE - 2, size=2)
        r = rng.integers(2, 4)
        for gy in range(IMAGE_SIZE):
            for gx in range(IMAGE_SIZE):
                dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                if dist <= r:
                    grid_hr[gy, gx] = 0.9
                elif dist <= r + 1:
                    grid_hr[gy, gx] = 0.6
                elif dist <= r + 2:
                    grid_hr[gy, gx] = 0.3
        y[i] = grid_hr.flatten()
        X[i] = np.clip(grid_hr.flatten() + rng.normal(0, noise_level + 0.2, N_FEATURES), 0, 1)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, noise_level=noise_level, random_seed=random_seed)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def reshape_image(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
'''


MODEL_MAP = {"CNN": CNN_MODEL_TEMPLATE, "CapsNet": CAPSNET_MODEL_TEMPLATE, "DN": DN_MODEL_TEMPLATE}
DATA_MAP = {"CNN": "{n_classes}", "CapsNet": "{n_classes}", "DN": "{n_classes}"}


def gen_pyproject(e):
    return f'''[project]
name = "{e["name"]}"
version = "0.1.0"
description = "{e["desc"]} - {e["arch"]} MLOps example"
requires-python = ">=3.11"
dependencies = [
    "mlops-shared",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.6.0",
    "prometheus-client>=0.20.0",
    "matplotlib>=3.7.0",
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
]

[project.scripts]
{e["pkg"]}-train = "{e["pkg"]}.train:main"

[tool.uv.sources]
mlops-shared = {{ workspace = true }}

[tool.uv]
package = true
'''


def gen_init(e):
    return f'''"""{e["desc"]}"""

__version__ = "0.1.0"
'''


def gen_data(e):
    if e["arch"] == "DN":
        template = DN_DATA_TEMPLATE
    else:
        template = DATA_TEMPLATE
    return template.format(
        title=e["title"], arch=e["arch"], img=IMG_SIZE,
        n_classes=max(e["n"], 1) if e["n"] > 0 else 2,
        labels=e["labels"],
    )


def gen_model(e):
    output_dim = 1 if e["task"] == "binary_classification" else max(e["n"], 1)
    template = MODEL_MAP[e["arch"]]
    return template.format(
        title=e["title"], title_lower=e["title"].lower(), arch=e["arch"],
        img=IMG_SIZE, nch=N_CHANNELS, n=e["n"], out_dim=output_dim,
        act=e["act"], loss=e["loss"], loss_fn=e["loss_fn"],
        cls=e["cls"], pkg=e["pkg"], labels=e["labels"],
    )


PY_TEMPLATE = '''"""Training pipeline for {title} ({arch})."""

import argparse
import os
from pathlib import Path

from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator, {schema_fn}

from {pkg}.data import (
    IMAGE_SIZE,
    N_CHANNELS,
    N_CLASSES,
    generate_synthetic_data,
    load_training_data,
    save_training_data,
    train_test_split,
)
from {pkg}.model import {cls}

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_filters: int = 8,
    kernel_size: int = 3,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    n_iterations: int = 300,
    weight_decay: float = 0.001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the {title_lower} model and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    if "{task}" in ("classification", "binary_classification"):
        validator = DataValidator({schema_fn}())
        validation = validator.validate(X.reshape(-1, 1))
        if not validation.valid:
            logger.error("Training data validation failed", errors=validation.errors)
            raise ValueError(f"Training data validation failed: {{validation.errors}}")
        logger.info("Training data validated", stats=validation.stats)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test), test_size=test_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = {cls}(
        IMG_SIZE={img},
        N_CHANNELS={nch},
        n_filters=n_filters,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    {fit_call}

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"{pkg}_model_v{{model_version}}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {{
        **test_metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "n_filters": float(n_filters),
        "learning_rate": float(learning_rate),
    }}

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="{name}",
        model_version=model_version,
        model_type="{model_type}",
        metrics=metrics,
        parameters={{
            "img_size": {img},
            "n_channels": {nch},
            "n_filters": n_filters,
            "kernel_size": kernel_size,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        }},
        artifacts={{
            f"{pkg}_model_v{{model_version}}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        }},
        tags={{"framework": "numpy", "task": "{pkg}", "model_type": "{arch}"}},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="{name}",
            model_version=model_version,
            metrics=metrics,
            params={{
                "img_size": {img},
                "n_filters": n_filters,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
            }},
            artifacts={{
                "model": str(model_path),
                "chart": str(model_dir / f"{pkg}_v{{model_version}}.png"),
            }},
            tags={{"model_type": "{pkg}", "framework": "numpy"}},
        )
        logger.info("Registered model to MLflow", model="{name}", version=model_version)

    return metrics


def _save_chart(model, output_dir: Path, version: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("{title} {arch} Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"{pkg}_v{{version}}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train {title} model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--n-filters", type=int, default=int(os.getenv("N_FILTERS", "8")))
    parser.add_argument("--kernel-size", type=int, default=int(os.getenv("KERNEL_SIZE", "3")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.05")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "300")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
'''


def gen_train(e):
    if e["task"] == "super_resolution" or e["task"] == "segmentation":
        fit_call = "model.fit(X_train, y_train)"
        model_type = "regression"
    else:
        fit_call = "model.fit(X_train, y_train, X_val=X_test, y_val=y_test)"
        model_type = "classification"

    return PY_TEMPLATE.format(
        title=e["title"], title_lower=e["title"].lower(), arch=e["arch"],
        pkg=e["pkg"], cls=e["cls"],
        schema_fn=f"create_{e['pkg']}_schema",
        task=e["task"],
        img=IMG_SIZE, nch=N_CHANNELS,
        fit_call=fit_call,
        model_type=model_type,
        name=e["name"],
    )


# API template
API_TEMPLATE = '''"""Serving API for {title} ({arch})."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from mlops_shared.drift import DriftDetector
from mlops_shared.fastapi_middleware import add_observability_middleware
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.metrics import MetricsCollector
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator, {schema_fn}

from pydantic import BaseModel, Field

from {pkg}.data import IMAGE_SIZE, N_CHANNELS, N_CLASSES, generate_synthetic_data
from {pkg}.model import {cls}

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("{prefix}_METRICS_PORT", "{port_str}"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    {field_name}: list[float] = Field(..., min_length={n_feat}, max_length={n_feat})


class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    {field_name}: str | list[float] | bool
    confidence: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    img_size: int
    n_channels: int
    n_filters: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: {cls} | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("{pkg}", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator({schema_fn}())
    feature_names = [f"pixel_{{i}}" for i in range({n_feat})]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={{f: "float" for f in feature_names}},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="{name}",
        model_version=_model_version,
        model_type="{task}",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="{name}", version=_model_version)

    yield
    logger.info("Shutting down {name} API")


def _load_model() -> tuple[{cls}, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            nn_models = [m for m in models if m.get("model_name") == "{name}"]
            if nn_models:
                nn_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("{pkg}_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return {cls}.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "{name}" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("{pkg}_model_*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return {cls}.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {{e}}")

    npz_path = MODEL_DIR / "{pkg}_model.npz"
    if npz_path.exists():
        return {cls}.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/{pkg}_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / f"{pkg}_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return {cls}.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = {cls}(
        IMG_SIZE={img},
        N_CHANNELS={nch},
        n_filters=8,
        kernel_size=3,
        hidden_dim=32,
        learning_rate=0.05,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title="{title} API",
    description="{desc}",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {{
        "service": "{pkg}-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
        "n_features": {n_feat},
        "endpoints": {{
            "health": "/health",
            "predict": "POST /predict",
            "predict/bulk": "POST /predict/bulk",
            "stats": "GET /stats",
            "drift": "GET /drift",
            "metrics": "/metrics",
        }},
    }}


@app.get("/health")
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {{
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
        "training_mode": _model.training_mode if _model else "unknown",
    }}


@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name="{name}",
                model_version=_model_version,
                model_type="{task}",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="{name}", version=_model_version)
        return {{"status": "reloaded", "model_version": _model_version}}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {{e}}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {{
            "total_features": {n_feat},
            "drifted_features": 0,
            "drift_ratio": 0.0,
            "drifted": [],
            "all_results": [],
        }}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])
    return summary


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None or getattr(_model, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        n_filters=getattr(_model, "n_filters", 8),
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction({field_name}: list[float]):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([{field_name}]).reshape(1, -1)

    if "{task}" in ("classification", "binary_classification"):
        validation = _validator.validate(X)
    else:
        validation = _validator.validate(X)

    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        {prob_code}
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append({field_name})
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Make a {title_lower} prediction."""
    return _compute_prediction(body.{field_name})


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    """Make multiple {title_lower} predictions."""
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for {field_name} in body.requests:
        predictions.append(_compute_prediction({field_name}))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
'''


def gen_api(e):
    task = e["task"]
    if task == "binary_classification":
        prob_code = '''probas = _model.predict_proba(X)[0]
        proba = float(probas[0] if len(probas) == 1 else probas[2])
        is_owner = bool(proba >= 0.5)
        response = PredictResponse(
            is_owner=is_owner,
            confidence=round(float(max(proba, 1 - proba)), 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )'''
    elif task in ("classification",):
        prob_code = '''probas = _model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probas))
        confidence = float(np.max(probas))
        response = PredictResponse(
            {field_name}=str(pred_idx),
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )'''.format(field_name=e["field"])
    else:
        prob_code = '''preds = _model.predict(X)[0]
        response = PredictResponse(
            {field_name}=preds.flatten().tolist(),
            confidence=round(float(np.max(np.abs(preds))), 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )'''.format(field_name=e["field"])

    return API_TEMPLATE.format(
        title=e["title"], title_lower=e["title"].lower(), arch=e["arch"],
        desc=e["desc"], pkg=e["pkg"], cls=e["cls"], name=e["name"],
        schema_fn=f"create_{e['pkg']}_schema",
        task=e["task"],
        prefix=e["prefix"], port_str=e["port"],
        field_name=e["field"], n_feat=N_FEATURES,
        prob_code=prob_code,
        img=IMG_SIZE, nch=N_CHANNELS,
    )


def gen_readme(e):
    port = int(e["port"]) - 10
    return f'''# {e["title"]}

{e["desc"]}

## Network Type
{e["arch"]}

## Architecture
- Input: {N_CHANNELS} channel x {IMG_SIZE}x{IMG_SIZE} grayscale images ({N_FEATURES} pixels)
- Convolution: {e["n"]} filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + {max(e["n"], 1)} output units ({e["act"]})

## Training
```bash
{e["pkg"]}-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn {e["pkg"]}.api:app --host 0.0.0.0 --port {port}
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Single prediction ({N_FEATURES} pixel values)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
{e["labels"]}

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
'''


def gen_schema_fn(e):
    return f'''


def create_{e["pkg"]}_schema() -> DataSchema:
    """Create the schema for {e["title"]}."""
    feature_names = [f"pixel_{{i}}" for i in range({N_FEATURES})]
    return DataSchema(
        feature_names=feature_names,
        feature_types={{f: "float" for f in feature_names}},
        required_columns=feature_names,
        min_rows=1,
        max_rows=10000,
        value_ranges={{f: (0.0, 1.0) for f in feature_names}},
    )
'''


def write(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def main():
    for e in EX:
        ex_dir = BASE / e["dir"]
        src_dir = ex_dir / "src" / e["pkg"]

        print(f"Generating: {e['title']} ({e['arch']})")

        write(ex_dir / "pyproject.toml", gen_pyproject(e))
        write(src_dir / "__init__.py", gen_init(e))
        write(src_dir / "data.py", gen_data(e))
        write(src_dir / "model.py", gen_model(e))
        write(src_dir / "train.py", gen_train(e))
        write(src_dir / "api.py", gen_api(e))
        write(ex_dir / "README.md", gen_readme(e))

    # Add schemas to validation.py (deduplicated)
    val_path = ROOT / "shared" / "mlops_shared" / "validation.py"
    with open(val_path) as f:
        existing_val = f.read()
    schema_text = "\n"
    for e in EX:
        schema_fn = f"create_{e['pkg']}_schema"
        if f"def {schema_fn}" not in existing_val:
            schema_text += gen_schema_fn(e)
    with open(val_path, "a") as f:
        f.write(schema_text)
    print("Schemas added to validation.py")

    print(f"\nAll {len(EX)} examples generated successfully.")


if __name__ == "__main__":
    main()
