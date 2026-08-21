"""Data loading and preprocessing for SNN image classification."""

from pathlib import Path

import numpy as np

N_FEATURES = 64
N_CLASSES = 10

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic image data for SNN classification.

    Creates 8x8 pixel images with class-based patterns.
    Images are normalized to [0, 1] for rate encoding.

    Returns:
        X: (n_samples, n_features) flattened 8x8 images
        y: (n_samples,) class labels
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)

    for i in range(n_samples):
        label = y[i]
        img = np.zeros((8, 8))

        patterns = [
            lambda m: m.__setitem__((slice(2, 4), slice(1, 7)), 1),
            lambda m: m.__setitem__((slice(1, 3), slice(2, 6)), 1),
            lambda m: m.__setitem__((slice(3, 5), slice(3, 5)), 1),
            lambda m: m.__setitem__((slice(4, 6), slice(2, 6)), 1),
            lambda m: (m.__setitem__((slice(1, 3), slice(2, 4)), 1), m.__setitem__((slice(5, 7), slice(4, 6)), 1)),
            lambda m: (m.__setitem__((slice(3, 5), slice(1, 3)), 1), m.__setitem__((slice(3, 5), slice(5, 7)), 1)),
            lambda m: m.__setitem__((slice(1, 7), slice(3, 5)), 1),
            lambda m: (m.__setitem__((slice(1, 3), slice(2, 6)), 1), m.__setitem__((slice(5, 7), slice(2, 6)), 1)),
            lambda m: (m.__setitem__((slice(3, 5), slice(1, 7)), 1),),
            lambda m: (m.__setitem__((slice(2, 5), slice(1, 7)), 1),),
        ]

        if label < len(patterns):
            patterns[label](img)

        noise_level = 0.3
        img = img + rng.normal(0, noise_level, img.shape)
        img = np.clip(img, 0, 1)
        X[i] = img.flatten()

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
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
