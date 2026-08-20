"""Data loading and preprocessing for VAE-based data generation.

Generates synthetic feature data for training the Variational Autoencoder.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 32
LATENT_DIM = 16

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature data for VAE training.

    Returns:
        X: (n_samples, N_FEATURES) feature vectors in [0, 1]
        y: (n_samples,) uniform labels (placeholder, not used by unsupervised VAE)
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES), dtype=float)

    for i in range(n_samples):
        pattern = rng.integers(0, 4)
        if pattern == 0:
            X[i, :N_FEATURES // 2] = 0.8 + rng.normal(0, noise_level, N_FEATURES // 2)
        elif pattern == 1:
            X[i, N_FEATURES // 2:] = 0.8 + rng.normal(0, noise_level, N_FEATURES // 2)
        elif pattern == 2:
            idx = rng.choice(N_FEATURES, N_FEATURES // 4, replace=False)
            X[i, idx] = 0.9 + rng.normal(0, noise_level, N_FEATURES // 4)
        else:
            X[i, :] = rng.normal(0.4, 0.2, N_FEATURES)

        X[i, :] = np.clip(X[i, :], 0, 1)

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
