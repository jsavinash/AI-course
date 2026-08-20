"""Data loading and preprocessing for Medical Imaging Diagnosis (CNN).

Generates synthetic 8x8 grayscale images and their labels.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = 8
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 3

DEFAULT_N_SAMPLES = 500

LABEL_NAMES = ['normal', 'benign', 'malignant']


def _create_template(label: int, rng: np.random.Generator) -> np.ndarray:
    """Create a 8x8 template pattern for a given class."""
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
