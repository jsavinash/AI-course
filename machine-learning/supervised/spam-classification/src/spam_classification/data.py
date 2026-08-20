"""Data loading and preprocessing for spam email classification."""

from pathlib import Path

import numpy as np
import pandas as pd

# Feature order MUST match what was used during training
FEATURE_NAMES = ["free", "win", "link", "!!!", "meeting"]


def load_training_data(data_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load spam training data from CSV or use built-in dataset.

    Expected CSV format:
        free,win,link,!!!,meeting,label
        1,1,1,1,0,1
        ...
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["label"].values.astype(int)
        return X, y

    # Built-in training data
    emails = np.array([
        [1, 1, 1, 1, 0],   # SPAM
        [0, 0, 0, 0, 1],   # NOT
        [1, 0, 1, 0, 0],   # SPAM
        [0, 0, 0, 0, 1],   # NOT
        [0, 1, 1, 1, 0],   # SPAM
        [0, 0, 0, 0, 1],   # NOT
        [1, 1, 1, 1, 0],   # SPAM
        [0, 0, 0, 0, 1],   # NOT
        [0, 1, 1, 0, 0],   # SPAM
        [0, 0, 0, 0, 1],   # NOT
    ], dtype=float)

    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    return emails, labels


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
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv(path, index=False)
