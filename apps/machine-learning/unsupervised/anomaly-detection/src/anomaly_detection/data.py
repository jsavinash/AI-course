"""Data loading and preprocessing for PCA-based anomaly detection.

Generates a realistic synthetic server monitoring dataset with:

Normal traffic patterns:
   - Baseline load with diurnal patterns
   - Correlated metrics (CPU, memory, network, disk I/O)
   - Realistic bounded ranges for each metric

Anomalous patterns:
   - CPU spikes
   - Memory leaks
   - Network floods
   - Disk thrashing
   - Error rate bursts
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Feature order MUST match what was used during training
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

# Number of synthetic samples generated when no CSV is provided
DEFAULT_N_SAMPLES = 2000

# Ratio of anomalous samples in generated data
ANOMALY_RATIO = 0.05


def _generate_server_metrics(
    n_samples: int = DEFAULT_N_SAMPLES,
    anomaly_ratio: float = ANOMALY_RATIO,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic server monitoring metrics with injected anomalies.

    Normal traffic baselines are calibrated so that typical healthy-server
    values (e.g. request_count ~120, cpu_usage ~35) sit near the center of
    the normal cluster, while extreme spikes are clearly separated as anomalies.

    Returns:
        X: array of shape (n_samples, n_features) - server metrics
        y: array of shape (n_samples,) - 0 for normal, 1 for anomaly
    """
    rng = np.random.default_rng(random_seed)
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal

    # ---- Generate normal traffic ----
    # Baseline ranges for normal traffic (aligned with test expectations)
    t = np.linspace(0, 24 * np.pi, n_normal)
    diurnal = 0.5 * (1 + np.sin(t))  # diurnal pattern [0, 1]

    # Normal traffic centered around typical healthy server values
    req_base = 120 + 40 * diurnal + rng.normal(0, 15, n_normal)
    bpr_base = 4800 + 1200 * rng.random(n_normal)
    cpu_base = 35 + 12 * diurnal + rng.normal(0, 4, n_normal)
    mem_base = 55 + 10 * diurnal + rng.normal(0, 3, n_normal)
    disk_base = 950 + 200 * diurnal + rng.normal(0, 40, n_normal)
    net_in_base = 220 + 60 * diurnal + rng.normal(0, 15, n_normal)
    net_out_base = 180 + 50 * diurnal + rng.normal(0, 12, n_normal)
    err_base = 1.5 + 1.0 * rng.random(n_normal)
    conn_base = 480 + 120 * diurnal + rng.normal(0, 20, n_normal)
    rt_base = 95 + 25 * diurnal + rng.normal(0, 8, n_normal)

    normal = np.column_stack(
        [
            req_base,
            bpr_base,
            cpu_base,
            mem_base,
            disk_base,
            net_in_base,
            net_out_base,
            err_base,
            conn_base,
            rt_base,
        ]
    )
    normal = np.clip(normal, 0, None)

    # ---- Generate anomalies ----
    anomaly_type = rng.integers(0, 5, size=n_anomaly)
    anomaly = np.zeros((n_anomaly, len(FEATURE_NAMES)))

    for i, at in enumerate(anomaly_type):
        if at == 0:
            # CPU spike
            anomaly[i] = [
                rng.normal(520, 60),
                rng.normal(1400, 200),
                rng.normal(72, 5),
                rng.normal(80, 4),
                rng.normal(2000, 200),
                rng.normal(1700, 100),
                rng.normal(1000, 80),
                rng.normal(15, 3),
                rng.normal(2600, 100),
                rng.normal(280, 20),
            ]
        elif at == 1:
            # Memory leak
            anomaly[i] = [
                rng.normal(350, 50),
                rng.normal(6000, 500),
                rng.normal(55, 6),
                rng.normal(92, 2),
                rng.normal(1200, 100),
                rng.normal(400, 40),
                rng.normal(300, 30),
                rng.normal(4, 1),
                rng.normal(800, 50),
                rng.normal(160, 15),
            ]
        elif at == 2:
            # Network flood
            anomaly[i] = [
                rng.normal(4500, 300),
                rng.normal(200, 50),
                rng.normal(85, 5),
                rng.normal(60, 5),
                rng.normal(300, 50),
                rng.normal(4500, 200),
                rng.normal(4500, 200),
                rng.normal(2, 1),
                rng.normal(4000, 200),
                rng.normal(250, 30),
            ]
        elif at == 3:
            # Disk thrashing
            anomaly[i] = [
                rng.normal(600, 100),
                rng.normal(3000, 400),
                rng.normal(60, 10),
                rng.normal(50, 8),
                rng.normal(15000, 500),
                rng.normal(100, 30),
                rng.normal(150, 30),
                rng.normal(8, 2),
                rng.normal(300, 50),
                rng.normal(500, 60),
            ]
        else:
            # Error burst
            anomaly[i] = [
                rng.normal(3000, 300),
                rng.normal(1500, 200),
                rng.normal(90, 4),
                rng.normal(80, 4),
                rng.normal(1000, 150),
                rng.normal(600, 80),
                rng.normal(500, 70),
                rng.normal(45, 5),
                rng.normal(1200, 100),
                rng.normal(400, 40),
            ]

    anomaly = np.clip(anomaly, 0, None)

    X = np.vstack([normal, anomaly])
    y = np.concatenate(
        [
            np.zeros(n_normal, dtype=int),
            np.ones(n_anomaly, dtype=int),
        ]
    )

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    return X, y


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load server metrics from CSV or generate a synthetic dataset.

    Expected CSV format:
        request_count,bytes_per_request,cpu_usage,memory_usage,disk_io,network_in,network_out,error_rate,connection_count,response_time,is_anomaly
        120.3,4800.1,35.2,55.1,210.4,160.2,180.5,0.4,350.2,68.3,0
        ...

    Returns:
        X: array of shape (n_samples, n_features) with features
        y: array of shape (n_samples,) - 0 for normal, 1 for anomaly
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df.get("is_anomaly", np.zeros(len(df), dtype=int)).values.astype(int)
        return X, y

    return _generate_server_metrics(n_samples=n_samples, random_seed=random_seed)


def load_normal_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    """Load only normal (non-anomalous) server metrics for PCA training.

    Returns:
        X: array of shape (n_normal, n_features) - normal samples only
    """
    X, y = load_training_data(data_path, n_samples, random_seed)
    return X[y == 0]


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["is_anomaly"] = y
    df.to_csv(path, index=False)
