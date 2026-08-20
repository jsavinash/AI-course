"""Data loading and preprocessing for email spam detection."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "has_free",
    "has_win",
    "has_link",
    "has_exclamation",
    "has_meeting",
    "email_length",
    "has_caps",
    "has_money",
    "num_links",
    "num_exclamations",
    "has_urgent",
    "sender_reputation",
]

DEFAULT_N_SAMPLES = 1000


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic email features with spam/ham labels.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels (1=spam, 0=ham).
    """
    rng = np.random.default_rng(random_seed)

    n_spam = n_samples // 2
    n_ham = n_samples - n_spam

    spam_emails = _generate_spam(n_spam, rng)
    ham_emails = _generate_ham(n_ham, rng)

    X = np.vstack([spam_emails, ham_emails])
    y = np.concatenate([np.ones(n_spam, dtype=int), np.zeros(n_ham, dtype=int)])

    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def _generate_spam(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic spam email feature vectors."""
    return np.column_stack(
        [
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(5, 50, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(3, 15, n),
            rng.integers(5, 20, n),
            rng.integers(0, 2, n),
            rng.uniform(0.0, 0.3, n),
        ]
    ).astype(float)


def _generate_ham(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic legitimate email feature vectors."""
    return np.column_stack(
        [
            rng.choice(2, size=n, p=[0.7, 0.3]),
            rng.choice(2, size=n, p=[0.9, 0.1]),
            rng.choice(2, size=n, p=[0.6, 0.4]),
            rng.choice(2, size=n, p=[0.5, 0.5]),
            rng.choice(2, size=n, p=[0.3, 0.7]),
            rng.integers(20, 100, n),
            rng.choice(2, size=n, p=[0.3, 0.7]),
            rng.choice(2, size=n, p=[0.1, 0.9]),
            rng.integers(0, 5, n),
            rng.integers(0, 3, n),
            rng.choice(2, size=n, p=[0.9, 0.1]),
            rng.uniform(0.7, 1.0, n),
        ]
    ).astype(float)


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate email data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["is_spam"].values.astype(int)
        return X, y

    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["is_spam"] = y
    df.to_csv(path, index=False)


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
