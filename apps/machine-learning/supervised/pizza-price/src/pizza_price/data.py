"""Data loading and preprocessing for pizza price prediction."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_training_data(data_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load pizza training data from CSV or use built-in dataset.

    Expected CSV format:
        diameter,price
        6,7.0
        8,9.0
        ...
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df["diameter"].values.astype(float)
        y = df["price"].values.astype(float)
        return X, y

    # Built-in training data
    X = np.array([6, 8, 10, 14, 18], dtype=float)
    y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)
    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
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
    """Save training data to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"diameter": X, "price": y})
    df.to_csv(path, index=False)
