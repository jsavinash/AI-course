"""Data loading and preprocessing for Transfer Learning."""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 1000
MAX_SEQ_LEN = 32
DEFAULT_N_SAMPLES = 500
N_CLASSES = 10


def generate_synthetic_data(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, n_classes: int = N_CLASSES, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    y = rng.integers(0, n_classes, size=n_samples)
    return X, y


def load_mnist_like_data(n_samples: int = 1000, n_classes: int = 10, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, size=(n_samples, 28, 28, 3))
    X = (X - X.min()) / (X.max() - X.min())
    y = rng.integers(0, n_classes, size=n_samples)
    return X, y


def preprocess_images(images: np.ndarray, target_size: tuple[int, int] = (32, 32), normalize: bool = True) -> np.ndarray:
    from PIL import Image
    resized = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize(target_size)
        resized.append(np.array(pil_img) / 255.0 if normalize else np.array(pil_img))
    return np.array(resized)


def extract_features_from_base_model(base_model, X: np.ndarray) -> np.ndarray:
    features = []
    for i in range(len(X)):
        x = X[i:i + 1]
        feat = base_model.forward(x)
        features.append(np.mean(feat, axis=1))
    return np.vstack(features)


def create_joint_dataset(source_X: np.ndarray, source_y: np.ndarray, target_X: np.ndarray, target_y: np.ndarray, mix_ratio: float = 0.5, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n_source = int(len(source_X) * mix_ratio)
    n_target = len(target_X) - n_source
    source_idx = rng.choice(len(source_X), n_source, replace=False)
    target_idx = rng.choice(len(target_X), n_target, replace=False)
    X_combined = np.vstack([source_X[source_idx], target_X[target_idx]])
    y_combined = np.hstack([source_y[source_idx], target_y[target_idx]])
    perm = rng.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def save_dataset(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]
