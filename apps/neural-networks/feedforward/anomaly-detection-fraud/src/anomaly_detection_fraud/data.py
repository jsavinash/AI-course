"""Data generation and preprocessing for credit card fraud detection."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "time_since_last_transaction",
    "transaction_amount",
    "merchant_category",
    "merchant_risk_score",
    "cardholder_risk_score",
    "distance_from_home",
    "is_online",
    "is_foreign",
    "hour_of_day",
    "day_of_week",
    "account_age_days",
    "recent_transaction_count",
    "avg_transaction_amount_24h",
    "device_risk_score",
    "ip_risk_score",
]

DEFAULT_N_SAMPLES = 2000
DEFAULT_ANOMALY_FRACTION = 0.05


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic credit card transaction data with fraud labels.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels (1=fraud, 0=legit).
    """
    rng = np.random.default_rng(random_seed)

    n_fraud = max(1, int(n_samples * anomaly_fraction))
    n_normal = n_samples - n_fraud

    normal_data = _generate_normal_transactions(n_normal, rng)
    fraud_data = _generate_fraud_transactions(n_fraud, rng)

    X = np.vstack([normal_data, fraud_data])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def _generate_normal_transactions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate legitimate credit card transactions."""
    return np.column_stack(
        [
            rng.uniform(0, 1440, n),  # time_since_last_transaction (minutes)
            rng.uniform(5, 500, n),  # transaction_amount
            rng.integers(0, 12, n),  # merchant_category
            rng.uniform(0.1, 0.4, n),  # merchant_risk_score
            rng.uniform(0.1, 0.3, n),  # cardholder_risk_score
            rng.uniform(0, 5, n),  # distance_from_home (miles)
            rng.integers(0, 2, n),  # is_online
            rng.choice(2, size=n, p=[0.9, 0.1]),  # is_foreign
            rng.integers(8, 22, n),  # hour_of_day
            rng.integers(0, 7, n),  # day_of_week
            rng.integers(30, 3650, n),  # account_age_days
            rng.integers(0, 15, n),  # recent_transaction_count
            rng.uniform(20, 200, n),  # avg_transaction_amount_24h
            rng.uniform(0.05, 0.3, n),  # device_risk_score
            rng.uniform(0.05, 0.3, n),  # ip_risk_score
        ]
    ).astype(float)


def _generate_fraud_transactions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate fraudulent credit card transactions."""
    return np.column_stack(
        [
            rng.uniform(0, 30, n),  # time_since_last_transaction (unusual bursts)
            rng.uniform(300, 5000, n),  # transaction_amount (large)
            rng.integers(0, 12, n),  # merchant_category
            rng.uniform(0.6, 0.95, n),  # merchant_risk_score (high)
            rng.uniform(0.6, 0.95, n),  # cardholder_risk_score (high)
            rng.uniform(50, 500, n),  # distance_from_home (far)
            rng.choice(2, size=n, p=[0.3, 0.7]),  # is_online (more likely online)
            rng.choice(2, size=n, p=[0.3, 0.7]),  # is_foreign (more likely foreign)
            rng.integers(0, 6, n),  # hour_of_day (late night) or 0-5
            rng.integers(0, 7, n),  # day_of_week
            rng.integers(0, 30, n),  # account_age_days (new accounts)
            rng.integers(15, 50, n),  # recent_transaction_count (bursts)
            rng.uniform(500, 5000, n),  # avg_transaction_amount_24h (high)
            rng.uniform(0.7, 0.95, n),  # device_risk_score (high)
            rng.uniform(0.7, 0.95, n),  # ip_risk_score (high)
        ]
    ).astype(float)


def generate_normal_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate only normal transactions for unsupervised/anomaly training."""
    return _generate_normal_transactions(n_samples, np.random.default_rng(random_seed))


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate credit card transaction data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["is_fraud"].values.astype(int)
        return X, y

    return generate_synthetic_data(
        n_samples=n_samples,
        anomaly_fraction=anomaly_fraction,
        random_seed=random_seed,
    )


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["is_fraud"] = y
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
