"""Data loading and preprocessing for stock market prediction (RNN).

Generates synthetic financial time-series feature sequences for price prediction.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 5  # normalized open, high, low, close, volume
SEQ_LEN = 20

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stock price feature sequences.

    Each sample is a sequence of SEQ_LEN timesteps, each with 5 features
    (normalized open, high, low, close, volume). The target is the next
    closing price (derived from the sequence trend).

    Returns:
        X: (n_samples, SEQ_LEN, N_FEATURES) feature sequences
        y: (n_samples,) target prices
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Random walk base price
        base_price = rng.uniform(50, 200)
        prices = [base_price]
        for _ in range(SEQ_LEN):
            change = rng.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)

        # Normalize features to [0, 1] using rolling window
        window = prices[:SEQ_LEN]
        w_min = window.min()
        w_max = window.max()
        w_range = w_max - w_min if w_max - w_min > 1e-8 else 1.0

        for t in range(SEQ_LEN):
            o = prices[t] if t == 0 else prices[t - 1]
            h = max(prices[t], prices[t - 1] if t > 0 else prices[t])
            lo = min(prices[t], prices[t - 1] if t > 0 else prices[t])
            c = prices[t]
            v = rng.uniform(0.3, 0.9)  # normalized volume

            X[i, t, 0] = (o - w_min) / w_range  # open
            X[i, t, 1] = (h - w_min) / w_range  # high
            X[i, t, 2] = (lo - w_min) / w_range  # low
            X[i, t, 3] = (c - w_min) / w_range  # close
            X[i, t, 4] = v  # volume

        # Target: actual next closing price (next step after sequence)
        y[i] = prices[SEQ_LEN]

    # Shuffle
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
