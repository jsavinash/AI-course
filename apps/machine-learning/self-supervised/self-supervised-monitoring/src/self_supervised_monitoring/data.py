"""Data generation and preprocessing for self-supervised server monitoring.

The self-supervised task is denoising: given corrupted server metrics,
reconstruct the original values. Labels are generated from the data itself -
no human annotation required.

Normal server metrics follow correlated patterns (e.g., high CPU correlates
with high response time). Anomalies deviate from these patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "request_count",
    "bytes_per_request",
    "cpu_usage",
    "memory_usage",
    "disk_io",
    "network_in",
    "network_out",
    "error_rate",
    "connection_count",
    "response_time",
]

DEFAULT_N_SAMPLES = 2000
DEFAULT_ANOMALY_FRACTION = 0.05
DEFAULT_NOISE_RATE = 0.25  # Fraction of features to corrupt
DEFAULT_NOISE_SCALE = 0.15  # Relative noise scale


def _generate_normal_samples(n_samples: int, random_seed: int = 42) -> np.ndarray:
    """Generate synthetic normal server metrics with realistic correlations.

    Normal server behavior:
    - request_count and connection_count are correlated
    - cpu_usage correlates with request_count and response_time
    - memory_usage correlates with bytes_per_request
    - network_in/out correlate with request_count
    - error_rate is low for normal traffic
    - response_time correlates with cpu_usage
    """
    rng = np.random.default_rng(random_seed)

    # Base traffic level (0-1 scale) drives many correlated features
    traffic = rng.normal(50, 15, n_samples)
    traffic = np.clip(traffic, 0, 100)

    # Memory usage tends to correlate with traffic volume
    memory_usage = traffic * 0.6 + rng.normal(0, 10, n_samples)
    memory_usage = np.clip(memory_usage, 20, 95)

    # CPU usage correlates with traffic and has its own load factor
    cpu_load = traffic * 0.8 + rng.normal(0, 8, n_samples)
    cpu_usage = np.clip(cpu_load, 5, 95)

    # Response time correlates with CPU usage (higher CPU = slower responses)
    response_time = cpu_usage * 1.5 + rng.normal(0, 10, n_samples)
    response_time = np.clip(response_time, 5, 500)

    # Bytes per request is affected by memory pressure
    bytes_per_request = 8000 - memory_usage * 30 + rng.normal(0, 600, n_samples)
    bytes_per_request = np.clip(bytes_per_request, 100, 20000)

    # Disk I/O scales with request volume
    disk_io = traffic * 2.5 + rng.normal(0, 50, n_samples)
    disk_io = np.clip(disk_io, 0, 5000)

    # Network usage correlates with traffic
    network_in = traffic * 0.8 + rng.normal(0, 30, n_samples)
    network_in = np.clip(network_in, 0, 3000)
    network_out = traffic * 0.5 + rng.normal(0, 20, n_samples)
    network_out = np.clip(network_out, 0, 2000)

    # Error rate is low for normal traffic, slight correlation with high CPU
    error_rate = rng.normal(1.0, 0.5, n_samples)
    error_rate = np.clip(error_rate, 0, 5)

    # Connection count correlates with request count
    connection_count = traffic * 0.9 + rng.normal(0, 10, n_samples)
    connection_count = np.clip(connection_count, 0, 500)

    request_count = traffic + rng.normal(0, 10, n_samples)
    request_count = np.clip(request_count, 0, 200)

    data = np.column_stack(
        [
            request_count,
            bytes_per_request,
            cpu_usage,
            memory_usage,
            disk_io,
            network_in,
            network_out,
            error_rate,
            connection_count,
            response_time,
        ]
    )

    return data.astype(float)


def _generate_anomalous_samples(n_samples: int, random_seed: int = 99) -> np.ndarray:
    """Generate synthetic anomalous server metrics.

    Anomalies deviate from normal patterns:
    - Spikes in CPU/memory without corresponding traffic
    - Very high error rates
    - Unusual combinations (e.g., high CPU but low request count)
    """
    rng = np.random.default_rng(random_seed)

    # Normal base, then inject anomalies
    base = _generate_normal_samples(n_samples, random_seed=random_seed)
    anomaly_types = rng.integers(0, 5, size=n_samples)

    for i in range(n_samples):
        atype = anomaly_types[i]
        if atype == 0:
            # CPU spike without traffic increase
            base[i, 2] = rng.uniform(90, 100)
        elif atype == 1:
            # Memory leak pattern
            base[i, 3] = rng.uniform(90, 100)
        elif atype == 2:
            # Error storm
            base[i, 7] = rng.uniform(20, 80)
        elif atype == 3:
            # Network flood
            base[i, 5] = rng.uniform(2000, 5000)
            base[i, 6] = rng.uniform(1500, 5000)
        elif atype == 4:
            # Response time degradation without CPU increase
            base[i, 9] = rng.uniform(800, 2000)

    return base.astype(float)


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic server metrics with normal and anomalous samples.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels
        (0 = normal, 1 = anomaly).
    """
    rng = np.random.default_rng(random_seed)

    n_anomalies = max(1, int(n_samples * anomaly_fraction))
    n_normal = n_samples - n_anomalies

    normal_data = _generate_normal_samples(n_normal, random_seed=random_seed)
    anomaly_data = _generate_anomalous_samples(n_anomalies, random_seed=random_seed + 7)

    X = np.vstack([normal_data, anomaly_data])
    y = np.concatenate(
        [
            np.zeros(n_normal, dtype=int),
            np.ones(n_anomalies, dtype=int),
        ]
    )

    # Shuffle
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def generate_normal_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate only normal server metrics for self-supervised training.

    Self-supervised training uses only normal data - anomalies are
    detected at inference time via high reconstruction error.
    """
    return _generate_normal_samples(n_samples, random_seed=random_seed)


def corrupt_features(
    X: np.ndarray,
    noise_rate: float = DEFAULT_NOISE_RATE,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    random_seed: int = 42,
) -> np.ndarray:
    """Create corrupted version of input for self-supervised denoising task.

    Corruptions:
    - Zero out a random fraction of features (dropout-like)
    - Add Gaussian noise to remaining features

    Returns:
        Corrupted version of X.
    """
    rng = np.random.default_rng(random_seed)

    X_noisy = X.copy().astype(float)
    n_features = X.shape[1]
    n_to_corrupt = max(1, int(n_features * noise_rate))

    for i in range(len(X)):
        # Pick random features to corrupt
        mask = rng.choice(n_features, size=n_to_corrupt, replace=False)
        for j in mask:
            # Either zero out or add noise
            if rng.random() < 0.5:
                X_noisy[i, j] = 0.0
            else:
                X_noisy[i, j] += rng.normal(0, noise_scale * (abs(X[i, j]) + 1e-6))

    return X_noisy


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance.

    Returns:
        Tuple of (normalized_X, mean, std).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def denormalize(X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse normalization."""
    return X_norm * std + mean


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate server metrics data for training.

    Args:
        data_path: Optional path to CSV file.
        n_samples: Number of samples to generate if no data_path.
        anomaly_fraction: Fraction of anomalous samples.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels.
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["label"].values.astype(int)
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
    df["label"] = y
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
