"""Data loading and preprocessing for Self-Organizing Maps."""

from pathlib import Path

import numpy as np

N_FEATURES = 32
GRID_HEIGHT = 5
GRID_WIDTH = 5

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature data for SOM training.

    Returns:
        X: (n_samples, N_FEATURES) feature vectors in [0, 1]
        y: (n_samples,) cluster labels (for evaluation only)
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, N_FEATURES), dtype=float)
    y = np.zeros(n_samples, dtype=int)

    n_per_cluster = n_samples // 5
    cluster_centers = np.array([
        np.full(N_FEATURES, 0.2),
        np.full(N_FEATURES, 0.4),
        np.full(N_FEATURES, 0.6),
        np.full(N_FEATURES, 0.8),
        np.zeros(N_FEATURES),
    ])
    cluster_centers[4, :N_FEATURES // 2] = 0.8
    cluster_centers[4, N_FEATURES // 2:] = 0.2

    for cluster_idx in range(5):
        start = cluster_idx * n_per_cluster
        end = start + n_per_cluster
        for i in range(start, min(end, n_samples)):
            X[i] = cluster_centers[cluster_idx] + rng.normal(0, noise_level, N_FEATURES)
            y[i] = cluster_idx

    X = np.clip(X, 0, 1)
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
