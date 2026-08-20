"""Data loading and preprocessing for PINN heat equation solver."""

from pathlib import Path

import numpy as np

N_FEATURES = 2
ALPHA = 0.01

DEFAULT_N_SAMPLES = 200


def heat_equation_solution(x: np.ndarray, t: float, alpha: float = ALPHA, n_terms: int = 50) -> np.ndarray:
    """Analytical solution to the 1D heat equation on [0, 1] with u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0.

    u(x,t) = sum_{n=1}^{inf} (2/(n*pi)) * (1 - exp(-n^2 * pi^2 * alpha * t)) * sin(n*pi*x)
    For initial condition sin(pi*x), only n=1 matters:
    u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t)
    """
    return np.sin(np.pi * x) * np.exp(-np.pi ** 2 * alpha * t)


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PDE solver data for the heat equation.

    Returns:
        X: (n_samples, 2) [x, t] coordinates
        u_true: (n_samples, 1) true temperature values
    """
    rng = np.random.default_rng(random_seed)
    X = np.zeros((n_samples, 2))
    u_true = np.zeros((n_samples, 1))

    for i in range(n_samples):
        x = rng.uniform(0, 1)
        t = rng.uniform(0, 0.5)
        X[i] = [x, t]
        u_true[i, 0] = heat_equation_solution(x, t, alpha=alpha)

    perm = rng.permutation(n_samples)
    return X[perm], u_true[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["u_true"]
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


def save_training_data(X: np.ndarray, u_true: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, u_true=u_true)
