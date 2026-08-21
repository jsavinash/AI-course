"""Data generation and preprocessing for house price prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "location_score",
    "age",
    "garage",
    "lot_size",
    "year_built",
    "property_type",
    "school_rating",
]

# Location multipliers for synthetic data
LOCATIONS = ["downtown", "suburban", "riverside", "mountain", "beach"]
LOCATION_SCORES = {"downtown": 85, "suburban": 70, "riverside": 60, "mountain": 55, "beach": 90}

DEFAULT_N_SAMPLES = 1000


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic house data with features and prices.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is house prices.
    """
    rng = np.random.default_rng(random_seed)

    sqft = rng.integers(800, 5000, n_samples).astype(float)
    bedrooms = rng.integers(1, 7, n_samples).astype(float)
    bathrooms = rng.integers(1, 5, n_samples).astype(float)
    location_indices = rng.integers(0, len(LOCATIONS), n_samples)
    location_score = np.array([LOCATION_SCORES[LOCATIONS[i]] for i in location_indices]).astype(
        float
    )
    age = rng.integers(0, 80, n_samples).astype(float)
    garage = rng.integers(0, 4, n_samples).astype(float)
    lot_size = rng.integers(2000, 15000, n_samples).astype(float)
    year_built = 2024 - age
    property_type = rng.integers(0, 4, n_samples).astype(
        float
    )  # 0=single, 1=condo, 2=townhome, 3=villa
    school_rating = rng.uniform(4.0, 10.0, n_samples)

    # Price formula: base + sqft * factor + location premium + other features + noise
    price = (
        50000
        + sqft * rng.uniform(80, 150, n_samples)
        + bedrooms * 15000
        + bathrooms * 20000
        + location_score * 1500
        - age * 1500
        + garage * 8000
        + lot_size * 2
        + school_rating * 25000
        + property_type * 30000
        + rng.normal(0, 20000, n_samples)
    )
    price = np.round(price, 2)

    X = np.column_stack(
        [
            sqft,
            bedrooms,
            bathrooms,
            location_score,
            age,
            garage,
            lot_size,
            year_built,
            property_type,
            school_rating,
        ]
    )

    return X.astype(float), price


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate house data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values.astype(float)
        y = df["price"].values.astype(float)
        return X, y

    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["price"] = y
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
