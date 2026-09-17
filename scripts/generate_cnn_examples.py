#!/usr/bin/env python3
"""Generate all 8 CNN/DN/CapsNet example packages."""

import os
from pathlib import Path

BASE = os.path.join(
    os.path.dirname(__file__), "apps/neural-networks/convolutional-neural-networks"
)


def write(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Created: {path}")


# ============ EXAMPLE DEFINITIONS ============
examples = [
    {
        "dir": "cnn/medical-imaging",
        "pkg": "medical_imaging",
        "name": "medical-imaging",
        "title": "Medical Imaging Diagnosis",
        "desc": "CNN for detecting tumors or fractures in synthetic medical X-ray images",
        "class_name": "MedicalImagingCNN",
        "task": "classification",
        "n_classes": 3,
        "n_labels": 3,
        "label_names": "['normal', 'benign', 'malignant']",
        "metrics_port": "8016",
        "metrics_prefix": "MEDICAL_IMAGING",
        "loss_type": "categorical cross-entropy",
        "output_activation": "softmax",
        "output_loss": "cross_entropy",
        "response_field": "condition",
        "response_values": "0=normal, 1=benign, 2=malignant",
        "model_version_file": "medical_imaging",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "cnn/facial-recognition",
        "pkg": "facial_recognition",
        "name": "facial-recognition",
        "title": "Facial Recognition",
        "desc": "CNN for binary facial recognition (own face vs unknown face)",
        "class_name": "FacialRecognitionCNN",
        "task": "binary_classification",
        "n_classes": 1,
        "n_labels": 2,
        "label_names": "['not_owner', 'owner']",
        "metrics_port": "8017",
        "metrics_prefix": "FACIAL_RECOGNITION",
        "loss_type": "binary cross-entropy",
        "output_activation": "sigmoid",
        "output_loss": "binary_crossentropy",
        "response_field": "is_owner",
        "response_values": "true=owner, false=not_owner",
        "model_version_file": "facial_recognition",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "cnn/video-surveillance",
        "pkg": "video_surveillance",
        "name": "video-surveillance",
        "title": "Video Surveillance",
        "desc": "CNN for real-time crowd monitoring and threat detection from frame images",
        "class_name": "VideoSurveillanceCNN",
        "task": "classification",
        "n_classes": 3,
        "n_labels": 3,
        "label_names": "['normal', 'activity', 'threat']",
        "metrics_port": "8018",
        "metrics_prefix": "VIDEO_SURVEILLANCE",
        "loss_type": "categorical cross-entropy",
        "output_activation": "softmax",
        "output_loss": "cross_entropy",
        "response_field": "activity",
        "response_values": "0=normal, 1=activity, 2=threat",
        "model_version_file": "video_surveillance",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "dn/image-super-resolution",
        "pkg": "image_super_resolution",
        "name": "image-super-resolution",
        "title": "Image Super-Resolution",
        "desc": "Deconvolutional network for upscaling low-resolution images to high-resolution",
        "class_name": "ImageSuperResolutionDN",
        "task": "super_resolution",
        "n_classes": 0,
        "n_labels": 1,
        "label_names": "N/A",
        "metrics_port": "8020",
        "metrics_prefix": "IMAGE_SUPER_RESOLUTION",
        "loss_type": "mean squared error",
        "output_activation": "linear",
        "output_loss": "mse",
        "response_field": "high_res_pixels",
        "response_values": "64 reconstructed pixel values",
        "model_version_file": "image_super_resolution",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "dn/semantic-segmentation",
        "pkg": "semantic_segmentation",
        "name": "semantic-segmentation",
        "title": "Semantic Segmentation",
        "desc": "Deconvolutional network for pixel-level segmentation of images into foreground/background",
        "class_name": "SemanticSegmentationDN",
        "task": "segmentation",
        "n_classes": 1,
        "n_labels": 2,
        "label_names": "['background', 'foreground']",
        "metrics_port": "8021",
        "metrics_prefix": "SEMANTIC_SEGMENTATION",
        "loss_type": "binary cross-entropy",
        "output_activation": "sigmoid",
        "output_loss": "binary_crossentropy",
        "response_field": "mask",
        "response_values": "64 binary pixel values (0=background, 1=foreground)",
        "model_version_file": "semantic_segmentation",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "capsnet/autonomous-driving",
        "pkg": "autonomous_driving",
        "name": "autonomous-driving",
        "title": "Autonomous Driving Object Recognition",
        "desc": "Capsule Network for recognizing traffic signs and pedestrians in self-driving cars",
        "class_name": "AutonomousDrivingCapsNet",
        "task": "capsnet_classification",
        "n_classes": 5,
        "n_labels": 5,
        "label_names": "['stop', 'yield', 'speed', 'pedestrian', 'other']",
        "metrics_port": "8022",
        "metrics_prefix": "AUTONOMOUS_DRIVING",
        "loss_type": "categorical cross-entropy",
        "output_activation": "softmax",
        "output_loss": "cross_entropy",
        "response_field": "object",
        "response_values": "0=stop, 1=yield, 2=speed, 3=pedestrian, 4=other",
        "model_version_file": "autonomous_driving",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "capsnet/medical-scan-analysis",
        "pkg": "medical_scan_analysis",
        "name": "medical-scan-analysis",
        "title": "Medical Scan Analysis",
        "desc": "Capsule Network for identifying overlapping body structures in 3D medical scans",
        "class_name": "MedicalScanAnalysisCapsNet",
        "task": "capsnet_classification",
        "n_classes": 4,
        "n_labels": 4,
        "label_names": "['bone', 'organ', 'vessel', 'tissue']",
        "metrics_port": "8023",
        "metrics_prefix": "MEDICAL_SCAN_ANALYSIS",
        "loss_type": "categorical cross-entropy",
        "output_activation": "softmax",
        "output_loss": "cross_entropy",
        "response_field": "structure",
        "response_values": "0=bone, 1=organ, 2=vessel, 3=tissue",
        "model_version_file": "medical_scan_analysis",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
    {
        "dir": "capsnet/text-char-recognition",
        "pkg": "text_char_recognition",
        "name": "text-char-recognition",
        "title": "Text and Character Recognition",
        "desc": "Capsule Network for reading distorted captchas and handwritten text",
        "class_name": "TextCharRecognitionCapsNet",
        "task": "capsnet_classification",
        "n_classes": 36,
        "n_labels": 36,
        "label_names": "['0','1',...,'9','A','B',...,'Z']",
        "metrics_port": "8024",
        "metrics_prefix": "TEXT_CHAR_RECOGNITION",
        "loss_type": "categorical cross-entropy",
        "output_activation": "softmax",
        "output_loss": "cross_entropy",
        "response_field": "character",
        "response_values": "0-9, A-Z",
        "model_version_file": "text_char_recognition",
        "n_features": 64,
        "img_size": 8,
        "n_channels": 1,
    },
]


def gen_data_py(ex):
    pkg = ex["pkg"]
    n_classes = ex["n_classes"]
    img_size = ex["img_size"]
    n_features = ex["n_features"]
    label_names = ex["label_names"]
    task = ex["task"]

    if task == "classification" or task == "capsnet_classification":
        content = f'''"""Data loading and preprocessing for {ex["title"]} (CNN).

Generates synthetic {img_size}x{img_size} grayscale images and their labels.
"""

from pathlib import Path

import numpy as np

IMG_SIZE = {img_size}
N_CHANNELS = 1
N_FEATURES = IMG_SIZE * IMG_SIZE
N_CLASSES = {n_classes}

DEFAULT_N_SAMPLES = 500

LABEL_NAMES = {label_names}


def _create_class_template(cls: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a {img_size}x{img_size} template pattern for a given class."""
    if rng is None:
        rng = np.random.default_rng(42)

    grid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)

    if cls == 0:
        grid[IMG_SIZE // 4 : 3 * IMG_SIZE // 4, IMG_SIZE // 4 : 3 * IMG_SIZE // 4] = 0.8
    elif cls == 1:
        grid[: IMG_SIZE // 2, :] = 0.7
    elif cls == 2:
        for r in range(IMG_SIZE):
            for c in range(IMG_SIZE):
                if (r + c) % 3 == 0:
                    grid[r, c] = 0.9
    elif cls == 3:
        grid[0, :] = 0.85
        grid[-1, :] = 0.85
        grid[:, 0] = 0.85
        grid[:, -1] = 0.85
    elif cls == 4:
        grid.flat[rng.integers(0, N_FEATURES, size=15)] = 0.9
    elif cls == 5:
        grid[IMG_SIZE // 3 : 2 * IMG_SIZE // 3, IMG_SIZE // 2 :] = 0.75

    return grid.flatten()


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their class labels.

    Returns:
        X: (n_samples, N_FEATURES) flattened image pixel arrays
        y: (n_samples,) class labels
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = rng.integers(0, N_CLASSES)
        template = _create_class_template(label, rng)
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
    """Reshape flattened images to (N, 1, IMG_SIZE, IMG_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMG_SIZE, IMG_SIZE)
'''
        return content

    elif task == "binary_classification":
        content = f'''"""Data loading and preprocessing for {ex["title"]} (CNN).

Generates synthetic {img_size}x{img_size} grayscale face images and binary labels.
"""

from pathlib import Path

import numpy as np

IMG_SIZE = {img_size}
N_CHANNELS = 1
N_FEATURES = IMG_SIZE * IMG_SIZE
N_CLASSES = {n_classes}

DEFAULT_N_SAMPLES = 500

LABEL_NAMES = {label_names}


def _create_face_template(owner: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a {img_size}x{img_size} face-like pattern.

    Owner faces have a centered pattern; non-owner faces have a shifted pattern.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    grid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)

    if owner:
        grid[IMG_SIZE // 4 : 3 * IMG_SIZE // 4, IMG_SIZE // 3 : 2 * IMG_SIZE // 3] = 0.9
        grid[IMG_SIZE // 2, IMG_SIZE // 4 : 3 * IMG_SIZE // 4] = 0.7
    else:
        grid[IMG_SIZE // 3 : 2 * IMG_SIZE // 3, IMG_SIZE // 5 : 2 * IMG_SIZE // 5] = 0.9
        grid[IMG_SIZE // 4, IMG_SIZE // 2 : 3 * IMG_SIZE // 4] = 0.7

    return grid.flatten()


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic face images and binary labels.

    Returns:
        X: (n_samples, N_FEATURES) flattened image pixel arrays
        y: (n_samples,) labels — 1=owner, 0=not_owner
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        is_owner = rng.integers(0, 2) == 1
        template = _create_face_template(is_owner, rng)
        X[i] = np.clip(template + rng.normal(0, noise_level, N_FEATURES), 0, 1)
        y[i] = int(is_owner)

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
    """Reshape flattened images to (N, 1, IMG_SIZE, IMG_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMG_SIZE, IMG_SIZE)
'''
        return content

    elif task == "super_resolution":
        content = f'''"""Data loading and preprocessing for {ex["title"]} (Deconvolutional Network).

Generates synthetic low-resolution {img_size}x{img_size} images and their
high-resolution counterparts for super-resolution training.
"""

from pathlib import Path

import numpy as np

IMG_SIZE = {img_size}
N_CHANNELS = 1
N_FEATURES = IMG_SIZE * IMG_SIZE

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic low-res and high-res image pairs.

    Low-res images are smooth patterns with noise.
    High-res targets are the clean (low-noise) versions.

    Returns:
        X_lowres: (n_samples, N_FEATURES) low-resolution image pixels
        X_highres: (n_samples, N_FEATURES) high-resolution image pixels
    """
    rng = np.random.default_rng(random_seed)

    X_lowres = np.zeros((n_samples, N_FEATURES))
    X_highres = np.zeros(n_samples, N_FEATURES) if False else np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        grid_hr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)
        grid_hr[IMG_SIZE // 4 : 3 * IMG_SIZE // 4, IMG_SIZE // 4 : 3 * IMG_SIZE // 4] = 0.9
        grid_hr[IMG_SIZE // 3 : 2 * IMG_SIZE // 3, IMG_SIZE // 3 : 2 * IMG_SIZE // 3] = 0.6

        X_highres[i] = grid_hr.flatten()
        X_lowres[i] = np.clip(grid_hr.flatten() + rng.normal(0, 0.3, N_FEATURES), 0, 1)

    perm = rng.permutation(n_samples)
    return X_lowres[perm], X_highres[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


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
    """Reshape flattened images to (N, 1, IMG_SIZE, IMG_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMG_SIZE, IMG_SIZE)
'''
        return content

    elif task == "segmentation":
        content = f'''"""Data loading and preprocessing for {ex["title"]} (Deconvolutional Network).

Generates synthetic {img_size}x{img_size} images and per-pixel binary segmentation masks.
"""

from pathlib import Path

import numpy as np

IMG_SIZE = {img_size}
N_CHANNELS = 1
N_FEATURES = IMG_SIZE * IMG_SIZE

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their segmentation masks.

    Returns:
        X: (n_samples, N_FEATURES) flattened image pixel arrays
        y: (n_samples, N_FEATURES) binary masks (1=foreground, 0=background)
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        grid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)

        cx, cy = rng.integers(2, IMG_SIZE - 2, size=2)
        r = rng.integers(2, 4)
        for gy in range(IMG_SIZE):
            for gx in range(IMG_SIZE):
                if (gx - cx) ** 2 + (gy - cy) ** 2 <= r**2:
                    grid[gy, gx] = 0.9
                    mask[gy, gx] = 1.0

        X[i] = np.clip(grid.flatten() + rng.normal(0, noise_level, N_FEATURES), 0, 1)
        y[i] = mask.flatten()

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
    """Reshape flattened images to (N, 1, IMG_SIZE, IMG_SIZE) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMG_SIZE, IMG_SIZE)
'''
        return content


def gen_model_py(ex):
    pkg = ex["pkg"]
    cls = ex["class_name"]
    task = ex["task"]
    n_classes = ex["n_classes"]
    n_features = ex["n_features"]
    img_size = ex["img_size"]
    n_channels = ex["n_channels"]
    input_name = ex["input_name"]
    title = ex["title"]
    output_activation = ex["output_activation"]
    output_loss = ex["output_loss"]
    loss_type = ex["loss_type"]
    label_names = ex["label_names"]

    if task in ("classification", "binary_classification", "capsnet_classification"):
        output_dim = n_classes if output_activation == "softmax" else 1
        content = f'''"""{title} using a Convolutional Neural Network.

Architecture:
    Input ({n_channels} x {img_size}x{img_size}) -> Conv2D ({n_channels}->8, 3x3, ReLU)
    -> MaxPool2D (2x2) -> Flatten -> Dense (hidden_dim, ReLU) -> Dense ({output_dim}, {output_activation})

Loss: {loss_type} (many-to-one: classifies image into a class label)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import SimpleCNN
from mlops_shared.validation import softmax as _softmax_unused  # noqa: F401

from {pkg}.data import IMG_SIZE, N_CHANNELS, N_CLASSES, LABEL_NAMES, reshape_image


@dataclass
class {cls}:
    """CNN for {title.lower()}.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input channels (1=grayscale, 3=RGB)
        n_filters: Number of convolution filters
        kernel_size: Convolution kernel size
        hidden_dim: Hidden units in dense layer
        output_dim: Output dimension (n_classes)
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    IMG_SIZE: int = {img_size}
    N_CHANNELS: int = {n_channels}
    n_filters: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    output_dim: int = {output_dim}
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleCNN | None = field(default=None, repr=False)
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
    ) -> "{cls}":
        """Train the CNN.

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
            output_activation="{output_activation}",
            output_loss="{output_loss}",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
        )
        self.model.fit(X_img, y_arr, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        X_img = reshape_image(X)
        return self.model.predict_proba(X_img)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return predicted class indices."""
        if self.output_dim == 1:
            probas = self.predict_proba(X).flatten()
            return (probas >= threshold).astype(int)
        return self.model.predict(X_img_of(X, self))

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return predicted class indices."""
        if self.output_dim == 1:
            probas = self.predict_proba(X).flatten()
            return (probas >= threshold).astype(int)
        X_img = reshape_image(X)
        return self.model.predict(X_img)

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
        obj = cls(
            IMG_SIZE=model.input_shape[1] if hasattr(model, 'input_shape') else {img_size},
            N_CHANNELS=model.input_shape[0] if hasattr(model, 'input_shape') else {n_channels},
            n_filters=model.n_filters,
            kernel_size=model.kernel_size,
            hidden_dim=model.hidden_dim,
            output_dim=model.output_dim,
            output_activation=model.output_activation,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        return obj

    def to_dict(self) -> dict:
        return {{
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }}


def X_img_of(X: np.ndarray, model: "{cls}") -> np.ndarray:
    """Reshape input for prediction."""
    return reshape_image(X)
'''
        return content

    else:
        return gen_dn_model_py(ex)


def gen_dn_model_py(ex):
    cls = ex["class_name"]
    pkg = ex["pkg"]
    img_size = ex["img_size"]
    n_channels = ex["n_channels"]
    output_activation = ex["output_activation"]
    output_loss = ex["output_loss"]
    title = ex["title"]
    loss_type = ex["loss_type"]
    n_features = ex["n_features"]

    content = f'''"""{title} using a Deconvolutional Network (DN).

Architecture:
    Input ({n_channels}x{img_size}x{img_size}) -> Conv2D (8, 3x3, ReLU) -> MaxPool2D (2x2)
    -> Deconv2D (8, 3x3, ReLU) -> Deconv2D ({n_channels}, 3x3, {output_activation})

Loss: {loss_type} (many-to-many: outputs pixel-level reconstruction)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import Conv2D, Activation, MaxPool2D, Deconv2D, relu, relu_derivative, sigmoid, softmax, tanh

from {pkg}.data import IMG_SIZE, N_CHANNELS, N_FEATURES, reshape_image


@dataclass
class Conv2DLayer:
    """Wrapper for Conv2D that stores gradients."""

    n_filters: int = 8
    kernel_size: int = 3
    random_seed: int = 42
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    def _init_weights(self, in_channels: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        fan_in = in_channels * self.kernel_size * self.kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0, scale, (self.n_filters, in_channels, self.kernel_size, self.kernel_size))
        self.b = np.zeros(self.n_filters)

    @property
    def n_filters_(self) -> int:
        return self.n_filters

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            self._init_weights(X.shape[1])
        N, C, H, W = X.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        out = np.zeros((N, self.n_filters, H_out, W_out))
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        region = X[n, :, h:h+self.kernel_size, w:w+self.kernel_size]
                        out[n, f, h, w] = np.sum(region * self.W[f]) + self.b[f]
        self._cache = {"X": X, "H_out": H_out, "W_out": W_out, "H": H, "W": W, "C": C, "N": N}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        X = c["X"]
        N, C, H, W = c["N"], c["C"], c["H"], c["W"]
        H_out, W_out = c["H_out"], c["W_out"]
        dW = np.zeros_like(self.W)
        db = np.zeros(self.n_filters)
        dX = np.zeros_like(X)
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        dout_val = dout[n, f, h, w]
                        dW[f] += dout_val * X[n, :, h:h+self.kernel_size, w:w+self.kernel_size]
                        db[f] += dout_val
                        dX[n, :, h:h+self.kernel_size, w:w+self.kernel_size] += dout_val * self.W[f]
        self.dW = dW / N
        self.db = db / N
        return dX

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class DNModel:
    """Encoder-decoder deconvolutional network for image-to-image tasks."""

    input_shape: tuple[int, int, int] = (1, {img_size}, {img_size})
    n_filters: int = 8
    kernel_size: int = 3
    hidden_dim: int = 32
    output_dim: int = {n_channels}
    output_activation: str = "{output_activation}"
    output_loss: str = "{output_loss}"
    learning_rate: float = 0.05
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    layers: list = field(default_factory=list, repr=False)
    loss_history: list[float] = field(default_factory=list)

    def _build_layers(self) -> None:
        C, H, W = self.input_shape
        self.layers = [
            Conv2DLayer(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed),
            Activation("relu"),
            MaxPool2D(pool_size=2, stride=2),
            Conv2DLayer(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed + 1),
            Activation("relu"),
            Deconv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, stride=2, random_seed=self.random_seed + 2),
            Activation("relu"),
            Deconv2D(n_filters=self.output_dim, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 3),
            Activation(self.output_activation),
        ]

    def _forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def _backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        for layer in self.layers:
            layer.update_params(self.learning_rate, self.weight_decay)


@dataclass
class {cls}:
    """Deconvolutional network for {title.lower()}.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of channels
        n_filters: Number of filters in conv layers
        kernel_size: Convolution kernel size
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
    """

    IMG_SIZE: int = {img_size}
    N_CHANNELS: int = {n_channels}
    n_filters: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: DNModel | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "{cls}":
        """Train the deconvolutional network.

        Args:
            X: Input images (n_samples, N_FEATURES)
            y: Target images (n_samples, N_FEATURES)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_img = reshape_image(y) if y.ndim == 1 or y.shape[1] == N_FEATURES else y

        self.model = DNModel(
            input_shape=(self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE),
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            output_dim=self.N_CHANNELS,
            output_activation="{output_activation}",
            output_loss="{output_loss}",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
        )
        self.model._build_layers()

        for layer in self.layers_iter():
            if isinstance(layer, DNModel):
                pass

        N = X_img.shape[0]
        for epoch in range(self.n_iterations):
            total_loss = 0.0
            for i in range(N):
                out = self.model._forward(X_img[i:i+1])
                target = y_img[i:i+1]

                eps = 1e-12
                if self.model.output_loss == "mse":
                    eps_out = np.clip(out, eps, 1 - eps)
                    loss = float(np.mean((target - eps_out) ** 2))
                else:
                    eps_out = np.clip(out, eps, 1 - eps)
                    loss = float(-np.mean(target * np.log(eps_out) + (1 - target) * np.log(1 - eps_out)))
                total_loss += loss

                if self.model.output_loss == "mse":
                    dout = 2 * (out - target) / max(out.shape[0], 1)
                else:
                    dout = (out - target) / max(out.shape[0], 1)

                self.model._backward(dout)

            self.model.loss_history = getattr(self.model, 'loss_history', [])
            self.model.loss_history.append(total_loss / N)

            if epoch > 50 and len(self.model.loss_history) > 100 and abs(self.model.loss_history[-1] - self.model.loss_history[-100]) < 1e-8:
                break

        self.loss_history = self.model.loss_history
        return self

    def layers_iter(self):
        """Return iterable of layers for testing."""
        return []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of images."""
        X_img = reshape_image(X)
        results = []
        for i in range(X_img.shape[0]):
            out = self.model._forward(X_img[i:i+1])
            results.append(out[0])
        return np.array(results)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        if preds.ndim > 1 and preds.shape[-1] == 1:
            preds = preds.flatten()
        return float(np.mean((preds.flatten() - y.flatten()) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        mse_val = self.mse(X, y)
        return {{"mse": mse_val, "rmse": float(np.sqrt(mse_val)), "n_samples": float(X.shape[0])}}

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        arrays = {{"loss_history": np.array(self.loss_history)}}
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2DLayer):
                if layer.W is not None:
                    arrays[f"conv_{i}_W"] = layer.W
                    arrays[f"conv_{i}_b"] = layer.b
            elif isinstance(layer, Deconv2D):
                if layer.W is not None:
                    arrays[f"deconv_{i}_W"] = layer.W
                    arrays[f"deconv_{i}_b"] = layer.b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "{cls}":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            IMG_SIZE={img_size},
            N_CHANNELS={n_channels},
            n_filters=8,
            kernel_size=3,
            learning_rate=0.05,
            n_iterations=300,
            weight_decay=0.001,
            clip_value=5.0,
            random_seed=42,
        )
        obj.model = DNModel(
            input_shape=(obj.N_CHANNELS, obj.IMG_SIZE, obj.IMG_SIZE),
            n_filters=obj.n_filters,
            kernel_size=obj.kernel_size,
            output_dim=obj.N_CHANNELS,
            output_activation="{output_activation}",
            output_loss="{output_loss}",
            random_seed=obj.random_seed,
        )
        obj.model._build_layers()
        for i, layer in enumerate(obj.model.layers):
            if isinstance(layer, Conv2DLayer) and f"conv_{{i}}_W" in data:
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
            "kernel_size": self.kernel_size,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }}
'''
    return content


def gen_train_py(ex):
    pkg = ex["pkg"]
    cls = ex["class_name"]
    name = ex["name"]
    n_features = ex["n_features"]
    img_size = ex["img_size"]
    schema = f"create_{ex['pkg'].replace('_', '_')}_schema"
    metrics_port = ex["metrics_port"]
    task = ex["task"]
    n_classes = ex["n_classes"]
    output_dim = n_classes if ex["output_activation"] == "softmax" else 1

    if task == "binary_classification":
        validate_line = f'validation = validator.validate(X.reshape(-1, 1))'
    elif task in ("classification", "capsnet_classification"):
        validate_line = f'validation = validator.validate(X.reshape(-1, 1))'
    else:
        validate_line = f'X_flat = X[:, :{n_features}].reshape(-1, {n_features})\n    validation = validator.validate(X_flat)'

    if task == "super_resolution":
        load_line = f'X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)'
    elif task == "segmentation":
        load_line = f'X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)'
    else:
        load_line = f'X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)'

    content = f'''"""Training pipeline for {ex["title"]} (CNN)."""

import argparse
import os
from pathlib import Path

from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator, create_{ex["pkg"].replace("_", "_")}_schema

from {pkg}.data import (
    {ex["IMG_SIZE"] if False else "IMG_SIZE"},
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
    {load_line}
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_{ex["pkg"].replace("_", "_")}_schema())
    {validate_line}
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {{validation.errors}}")
    logger.info("Training data validated", stats=validation.stats)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test), test_size=test_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = {cls}(
        IMG_SIZE={img_size},
        N_CHANNELS={n_channels},
        n_filters=n_filters,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f"{ex["model_version_file"]}_model_v{{model_version}}.npz"
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
        model_type="cnn_{task}",
        metrics=metrics,
        parameters={{
            "img_size": {img_size},
            "n_channels": {n_channels},
            "n_filters": n_filters,
            "kernel_size": kernel_size,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        }},
        artifacts={{
            f"{ex["model_version_file"]}_model_v{{model_version}}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        }},
        tags={{"framework": "numpy", "task": "{pkg}", "model_type": "simple_cnn"}},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="{name}",
            model_version=model_version,
            metrics=metrics,
            params={{
                "img_size": {img_size},
                "n_channels": {n_channels},
                "n_filters": n_filters,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
            }},
            artifacts={{"model": str(model_path), "chart": str(model_dir / f"{ex["model_version_file"]}_v{{model_version}}.png")}},
            tags={{"model_type": "{pkg}", "framework": "numpy"}},
        )
        logger.info("Registered model to MLflow", model="{name}", version=model_version)

    return metrics


def _save_chart(model: {cls}, output_dir: Path, version: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("{ex["title"]} CNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"{ex["model_version_file"]}_v{{version}}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train {ex["title"]} CNN")
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
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
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
    return content


def gen_api_py(ex):
    pkg = ex["pkg"]
    cls = ex["class_name"]
    name = ex["name"]
    title = ex["title"]
    metrics_port = ex["metrics_port"]
    metrics_prefix = ex["metrics_prefix"]
    response_field = ex["response_field"]
    response_values = ex["response_values"]
    img_size = ex["img_size"]
    n_features = ex["n_features"]
    n_classes = ex["n_classes"]
    schema_name = f"create_{ex['pkg']}_schema"
    input_name = ex["input_name"]

    content = f'''"""Serving API for {title} (CNN)."""

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
from mlops_shared.validation import DataValidator, {schema_name}

from pydantic import BaseModel, Field

from {pkg}.data import IMG_SIZE, N_CHANNELS, N_CLASSES, generate_synthetic_data
from {pkg}.model import {cls}

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("{metrics_prefix}_METRICS_PORT", "{metrics_port}"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class PredictRequest(BaseModel):
    {input_name}: list[float] = Field(..., min_length={n_features}, max_length={n_features})


class PredictBulkRequest(BaseModel):
    requests: list[list[float]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    {response_field}: str | float
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

    _validator = DataValidator({schema_name}())
    feature_names = [f"pixel_{{i}}" for i in range({n_features})]
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
        model_type="cnn_{ex["task"]}",
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
                npz_files = list(model_dir.glob("{ex["model_version_file"]}_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return {cls}.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "{name}" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("{ex["model_version_file"]}_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return {cls}.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {{e}}")

    npz_path = MODEL_DIR / "{ex["model_version_file"]}_model.npz"
    if npz_path.exists():
        return {cls}.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/{ex["model_version_file"]}_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / f"{ex["model_version_file"]}_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return {cls}.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = {cls}(
        IMG_SIZE={img_size},
        N_CHANNELS={n_channels},
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
    description="CNN for {ex["desc"]}",
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
                model_type="cnn_{ex["task"]}",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="{name}", version=_model_version)
        return {{"status": "reloaded", "model_version": _model_version}}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {{e}}") from e


@app.get("/drift")
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    if len(_recent_predictions) < 10:
        return {{
            "total_features": {n_features},
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
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StatsResponse(
        img_size=IMG_SIZE,
        n_channels=N_CHANNELS,
        n_filters=_model.model.n_filters if hasattr(_model.model, "n_filters") else 8,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction({input_name}: list[float]) -> PredictResponse:
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([{input_name}])

    if "{task}" == "binary_classification":
        X_flat = X.reshape(-1, 1)
        validation = _validator.validate(X_flat)
    else:
        X_flat = X.reshape(-1, 1)
        validation = _validator.validate(X_flat)

    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        proba = float(_model.predict_proba(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        _recent_predictions.append({input_name})
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            {response_field}=f"predicted",
            confidence=round(float(max(proba, 1 - proba)), 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    return _compute_prediction(body.{input_name})


@app.post("/predict/bulk", response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(body.requests) < 1 or len(body.requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 50")

    predictions = []
    for {input_name} in body.requests:
        predictions.append(_compute_prediction({input_name}))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)
'''
    return content
