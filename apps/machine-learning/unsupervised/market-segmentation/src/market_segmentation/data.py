"""Data loading and preprocessing for market segmentation.

Generates a realistic synthetic customer dataset with distinct behavioural
segments, designed for unsupervised K-Means clustering:

Segments:
  1. Premium Shoppers    - high income, high spending
  2. Cautious High-Earners - high income, low spending
  3. Impulsive Shoppers  - low income, high spending
  4. Budget-Conscious    - low income, low spending
  5. Average Shoppers    - medium income, medium spending
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Feature order MUST match what was used during training
FEATURE_NAMES = ["annual_income", "spending_score"]

# Number of synthetic customers generated when no CSV is provided
DEFAULT_N_SAMPLES = 500


def _generate_customer_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic customer data with 5 distinct segments.

    Returns:
        X: array of shape (n_samples, 2) - [annual_income(k$), spending_score(0-100)]
        true_labels: ground-truth segment assignment for evaluation
    """
    rng = np.random.default_rng(random_seed)

    # Define the 5 segment centers (annual_income, spending_score)
    segments = [
        {"center": [70, 85], "std": [6.0, 7.0], "weight": 0.20},  # Premium Shoppers
        {"center": [80, 25], "std": [7.0, 7.0], "weight": 0.20},  # Cautious High-Earners
        {"center": [35, 80], "std": [7.0, 7.0], "weight": 0.20},  # Impulsive Shoppers
        {"center": [30, 30], "std": [6.0, 7.0], "weight": 0.20},  # Budget-Conscious
        {"center": [55, 55], "std": [8.0, 8.0], "weight": 0.20},  # Average Shoppers
    ]

    # Sample from a mixture of gaussians
    weights = np.array([s["weight"] for s in segments])
    n_per_segment = rng.multinomial(n_samples, weights)

    X_list: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for seg_idx, (seg, n_seg) in enumerate(zip(segments, n_per_segment, strict=False)):
        income = rng.normal(loc=seg["center"][0], scale=seg["std"][0], size=n_seg)
        spending = rng.normal(loc=seg["center"][1], scale=seg["std"][1], size=n_seg)

        # Clip to realistic ranges
        income = np.clip(income, 15, 120)
        spending = np.clip(spending, 1, 99)

        X_list.append(np.column_stack([income, spending]))
        labels.append(np.full(n_seg, seg_idx, dtype=int))

    X = np.vstack(X_list)
    true_labels = np.concatenate(labels)

    # Shuffle the data
    perm = rng.permutation(n_samples)
    X = X[perm]
    true_labels = true_labels[perm]

    return X, true_labels


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load customer data from CSV or generate a synthetic dataset.

    Expected CSV format:
        annual_income,spending_score
        75.4,82.3
        23.1,45.0
        ...

    Returns:
        X: array of shape (n_samples, 2) with features
        y: ground-truth segment labels (used only for evaluation in
           unsupervised learning; the model itself never sees them)
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df.get("segment", np.zeros(len(df), dtype=int)).values.astype(int)
        return X, y

    return _generate_customer_data(n_samples=n_samples, random_seed=random_seed)


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["segment"] = y
    df.to_csv(path, index=False)
