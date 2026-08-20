"""Data loading and preprocessing for semi-supervised email classification.

Generates a realistic synthetic email dataset with:
- A small labeled subset (10-20% of data)
- A large unlabeled subset (80-90% of data)
- Email text features: keyword presence, length, special characters
- Binary labels: 1 = spam, 0 = ham

This demonstrates semi-supervised learning where the model leverages
both labeled and unlabeled data to improve classification accuracy.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Feature order MUST match what was used during training
FEATURE_NAMES = ["has_free", "has_win", "has_link", "has_exclamation", "has_meeting", "length_score", "has_caps"]

# Default labeled ratio (fraction of data that is labeled)
DEFAULT_LABELED_RATIO = 0.1
DEFAULT_N_SAMPLES = 1000


def _generate_synthetic_emails(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic email feature data with known spam/ham patterns.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels.
        All data is initially labeled.
    """
    rng = np.random.default_rng(random_seed)

    X = []
    y = []

    for _i in range(n_samples):
        is_spam = rng.random() < 0.4  # 40% spam, 60% ham

        if is_spam:
            features = [
                1 if rng.random() < 0.8 else 0,  # has_free
                1 if rng.random() < 0.7 else 0,  # has_win
                1 if rng.random() < 0.6 else 0,  # has_link
                1 if rng.random() < 0.75 else 0,  # has_exclamation
                0 if rng.random() < 0.7 else 1,  # has_meeting
                rng.integers(5, 10),  # length_score (spam tends to be longer)
                1 if rng.random() < 0.6 else 0,  # has_caps
            ]
        else:
            features = [
                0 if rng.random() < 0.7 else 1,  # has_free
                0 if rng.random() < 0.8 else 1,  # has_win
                0 if rng.random() < 0.8 else 1,  # has_link
                0 if rng.random() < 0.8 else 1,  # has_exclamation
                1 if rng.random() < 0.6 else 0,  # has_meeting
                rng.integers(1, 5),  # length_score (ham tends to be shorter)
                0 if rng.random() < 0.8 else 1,  # has_caps
            ]

        X.append(features)
        y.append(1 if is_spam else 0)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def load_training_data(
    data_path: Path | None = None,
    labeled_ratio: float = DEFAULT_LABELED_RATIO,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load semi-supervised email data with labeled and unlabeled subsets.

    Args:
        data_path: Optional path to CSV file. If provided, loads from CSV.
        labeled_ratio: Fraction of data to keep labeled (0.0 to 1.0).
        n_samples: Number of samples to generate if no data_path.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y, is_labeled) where:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Label vector of shape (n_samples,). Unlabeled samples have label -1.
        - is_labeled: Boolean mask of shape (n_samples,) indicating labeled samples.
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y_raw = df["label"].values.astype(int)

        # If CSV has "is_labeled" column, use it; otherwise treat all as labeled
        if "is_labeled" in df.columns:
            is_labeled = df["is_labeled"].values.astype(bool)
            y = np.where(is_labeled, y_raw, -1)
        else:
            is_labeled = np.ones(len(X), dtype=bool)
            y = y_raw

        return X, y, is_labeled

    # Generate synthetic data
    X, y_full = _generate_synthetic_emails(n_samples=n_samples, random_seed=random_seed)

    # Create labeled/unlabeled split
    rng = np.random.default_rng(random_seed)
    n_labeled = max(1, int(n_samples * labeled_ratio))
    labeled_indices = rng.choice(n_samples, size=n_labeled, replace=False)
    is_labeled = np.zeros(n_samples, dtype=bool)
    is_labeled[labeled_indices] = True

    # Unlabeled samples get label -1
    y = np.where(is_labeled, y_full, -1)

    return X, y, is_labeled


def get_labeled_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract only the labeled subset of the data.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,) with -1 for unlabeled

    Returns:
        Tuple of (X_labeled, y_labeled) with only labeled samples
    """
    mask = y != -1
    return X[mask], y[mask]


def get_unlabeled_data(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Extract only the unlabeled subset of the data.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,) with -1 for unlabeled

    Returns:
        X_unlabeled with only unlabeled samples
    """
    mask = y == -1
    return X[mask]


def save_training_data(X: np.ndarray, y: np.ndarray, is_labeled: np.ndarray, path: Path) -> None:
    """Save semi-supervised training data to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df["is_labeled"] = is_labeled.astype(int)
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
