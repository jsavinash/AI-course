"""Data loading and preprocessing for weather forecasting (RNN).

Generates synthetic weather time-series feature sequences for next-day forecasting.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 5  # temperature, humidity, pressure, wind_speed, precipitation
SEQ_LEN = 30  # days of history

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic weather feature sequences.

    Each sample is a sequence of SEQ_LEN days, each with 5 features:
    temperature, humidity, pressure, wind_speed, precipitation.

    The target is the weather vector for the next day, derived from
    temporal patterns (seasonal trend + autocorrelation + noise).

    Returns:
        X: (n_samples, SEQ_LEN, N_FEATURES) weather feature sequences
        y: (n_samples, N_FEATURES) next-day weather vector
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES))
    y = np.zeros((n_samples, N_FEATURES))

    for i in range(n_samples):
        # Random seasonal phase
        phase = rng.uniform(0, 2 * np.pi)

        # Generate base pattern for temperature (seasonal + trend)
        temps = np.zeros(SEQ_LEN + 1)
        for t in range(SEQ_LEN + 1):
            seasonal = np.sin(phase + t * 0.2) * 10 + 15
            trend = t * 0.05
            temps[t] = seasonal + trend + rng.normal(0, 2)

        # Humidity (anti-correlated with temperature)
        humidity = 80 - (temps[: SEQ_LEN + 1] - 10) * 0.5 + rng.normal(0, 3, SEQ_LEN + 1)
        humidity = np.clip(humidity, 0, 100)

        # Pressure (slowly varying)
        pressure_base = rng.uniform(990, 1030)
        pressure = pressure_base + np.cumsum(rng.normal(0, 0.3, SEQ_LEN + 1))

        # Wind speed
        wind = np.abs(rng.normal(8, 4, SEQ_LEN + 1)) + np.sin(np.arange(SEQ_LEN + 1) * 0.3) * 3

        # Precipitation (correlated with humidity)
        precip = np.where(
            humidity > 75, rng.uniform(0.5, 3.0, SEQ_LEN + 1), rng.uniform(0, 0.3, SEQ_LEN + 1)
        )

        # Fill X (first SEQ_LEN days)
        X[i, :, 0] = temps[:SEQ_LEN]  # temperature
        X[i, :, 1] = humidity[:SEQ_LEN]  # humidity
        X[i, :, 2] = pressure[:SEQ_LEN]  # pressure
        X[i, :, 3] = wind[:SEQ_LEN]  # wind speed
        X[i, :, 4] = precip[:SEQ_LEN]  # precipitation

        # Target: next day (day SEQ_LEN)
        y[i, 0] = temps[SEQ_LEN]
        y[i, 1] = humidity[SEQ_LEN]
        y[i, 2] = pressure[SEQ_LEN]
        y[i, 3] = wind[SEQ_LEN]
        y[i, 4] = precip[SEQ_LEN]

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
