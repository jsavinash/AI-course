"""Data loading and preprocessing for Diffusion-based image generation.

Generates synthetic 8x8 grayscale images for training the diffusion model.
"""

from pathlib import Path

import numpy as np

IMAGE_SIZE = 8
N_CHANNELS = 1
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE
DEFAULT_TIMESTEPS = 1000

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images for diffusion model training.

    Returns:
        X: (n_samples, N_FEATURES) image pixel arrays in [0, 1]
        y: (n_samples,) uniform labels (placeholder, self-supervised)
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
        cx, cy = rng.integers(2, IMAGE_SIZE - 2, size=2)
        r = rng.integers(1, 4)
        for gy in range(IMAGE_SIZE):
            for gx in range(IMAGE_SIZE):
                dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                if dist <= r:
                    grid[gy, gx] = 0.9
                elif dist <= r + 1:
                    grid[gy, gx] = 0.5
        X[i] = np.clip(grid.flatten() + rng.normal(0, noise_level, N_FEATURES), 0, 1)

    y = np.ones(n_samples, dtype=int)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
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
    """Reshape flattened images to (N, C, H, W) for CNN input."""
    return X.reshape(-1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
