"""Data validation and schema enforcement for MLOps pipelines."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataSchema:
    """Schema definition for validating input data."""

    feature_names: list[str]
    feature_types: dict[str, str] = field(default_factory=dict)  # "float", "int", "binary"
    required_columns: list[str] = field(default_factory=list)
    min_rows: int = 1
    max_rows: int | None = None
    value_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    categorical_values: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.required_columns:
            self.required_columns = self.feature_names


@dataclass
class ValidationResult:
    """Result of a data validation check."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class DataValidator:
    """Validate input data against a schema for training and inference."""

    def __init__(self, schema: DataSchema):
        self.schema = schema

    def validate(self, X: np.ndarray, y: np.ndarray | None = None) -> ValidationResult:
        """Validate feature matrix and optional labels."""
        result = ValidationResult(valid=True)

        # Check dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_samples < self.schema.min_rows:
            result.errors.append(
                f"Expected at least {self.schema.min_rows} samples, got {n_samples}"
            )

        if self.schema.max_rows and n_samples > self.schema.max_rows:
            result.errors.append(
                f"Expected at most {self.schema.max_rows} samples, got {n_samples}"
            )

        if n_features != len(self.schema.feature_names):
            result.errors.append(
                f"Expected {len(self.schema.feature_names)} features, got {n_features}"
            )

        # Check for NaN/Inf
        if np.isnan(X).any():
            result.errors.append("Input data contains NaN values")
        if np.isinf(X).any():
            result.errors.append("Input data contains infinite values")

        # Check value ranges
        for i, name in enumerate(self.schema.feature_names):
            if name in self.schema.value_ranges:
                lo, hi = self.schema.value_ranges[name]
                col = X[:, i]
                if col.min() < lo or col.max() > hi:
                    result.errors.append(
                        f"Feature '{name}' out of range [{lo}, {hi}]"
                    )

            # Check categorical values
            if name in self.schema.categorical_values:
                allowed = set(self.schema.categorical_values[name])
                col = X[:, i]
                if not set(np.unique(col)).issubset(allowed):
                    result.errors.append(
                        f"Feature '{name}' has values outside allowed set: {allowed}"
                    )

            # Check binary features
            if self.schema.feature_types.get(name) == "binary":
                col = X[:, i]
                if not set(np.unique(col)).issubset({0, 1}):
                    result.errors.append(f"Feature '{name}' must be binary (0/1)")

        # Validate labels if provided
        if y is not None:
            if len(y) != n_samples:
                result.errors.append(
                    f"Label count {len(y)} does not match sample count {n_samples}"
                )
            if np.isnan(y).any():
                result.errors.append("Labels contain NaN values")

        # Compute stats
        result.stats = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_means": {name: float(np.mean(X[:, i])) for i, name in enumerate(self.schema.feature_names)},
            "feature_stds": {name: float(np.std(X[:, i])) for i, name in enumerate(self.schema.feature_names)},
        }

        result.valid = len(result.errors) == 0
        return result

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a pandas DataFrame."""
        result = ValidationResult(valid=True)

        # Check required columns
        missing = [c for c in self.schema.required_columns if c not in df.columns]
        if missing:
            result.errors.append(f"Missing required columns: {missing}")

        if result.errors:
            result.valid = False
            return result

        # Convert to numpy and validate
        X = df[self.schema.feature_names].values
        return self.validate(X)


def create_pizza_schema() -> DataSchema:
    """Create the schema for pizza price prediction."""
    return DataSchema(
        feature_names=["diameter"],
        feature_types={"diameter": "float"},
        required_columns=["diameter"],
        min_rows=1,
        max_rows=10000,
        value_ranges={"diameter": (1.0, 50.0)},
    )


def create_spam_schema() -> DataSchema:
    """Create the schema for spam classification."""
    return DataSchema(
        feature_names=["free", "win", "link", "!!!", "meeting"],
        feature_types={f: "binary" for f in ["free", "win", "link", "!!!", "meeting"]},
        required_columns=["free", "win", "link", "!!!", "meeting"],
        min_rows=1,
        max_rows=100000,
        categorical_values={f: [0, 1] for f in ["free", "win", "link", "!!!", "meeting"]},
    )


def create_semi_supervised_email_schema() -> DataSchema:
    """Create the schema for semi-supervised email classification."""
    return DataSchema(
        feature_names=["has_free", "has_win", "has_link", "has_exclamation", "has_meeting", "length_score", "has_caps"],
        feature_types={
            "has_free": "binary",
            "has_win": "binary",
            "has_link": "binary",
            "has_exclamation": "binary",
            "has_meeting": "binary",
            "length_score": "int",
            "has_caps": "binary",
        },
        required_columns=["has_free", "has_win", "has_link", "has_exclamation", "has_meeting", "length_score", "has_caps"],
        min_rows=1,
        max_rows=100000,
        value_ranges={
            "length_score": (1.0, 10.0),
        },
        categorical_values={
            "has_free": [0, 1],
            "has_win": [0, 1],
            "has_link": [0, 1],
            "has_exclamation": [0, 1],
            "has_meeting": [0, 1],
            "has_caps": [0, 1],
        },
    )


def create_market_segmentation_schema() -> DataSchema:
    """Create the schema for market segmentation (unsupervised K-Means)."""
    return DataSchema(
        feature_names=["annual_income", "spending_score"],
        feature_types={"annual_income": "float", "spending_score": "float"},
        required_columns=["annual_income", "spending_score"],
        min_rows=1,
        max_rows=100000,
        value_ranges={
            "annual_income": (1.0, 200.0),
            "spending_score": (0.0, 100.0),
        },
    )


def create_recommendation_schema() -> DataSchema:
    """Create the schema for the recommendation engine (association rules)."""
    return DataSchema(
        feature_names=["item"],
        feature_types={"item": "categorical"},
        required_columns=["item"],
        min_rows=1,
        max_rows=100000,
    )


def create_anomaly_detection_schema() -> DataSchema:
    """Create the schema for PCA-based anomaly detection (server monitoring metrics)."""
    return DataSchema(
        feature_names=[
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
        ],
        feature_types={
            f: "float"
            for f in [
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
        },
        required_columns=[
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
        ],
        min_rows=1,
        max_rows=100000,
        value_ranges={
            "request_count": (0.0, 5000.0),
            "bytes_per_request": (0.0, 50000.0),
            "cpu_usage": (0.0, 100.0),
            "memory_usage": (0.0, 100.0),
            "disk_io": (0.0, 20000.0),
            "network_in": (0.0, 5000.0),
            "network_out": (0.0, 5000.0),
            "error_rate": (0.0, 100.0),
            "connection_count": (0.0, 5000.0),
            "response_time": (0.0, 5000.0),
        },
    )


def create_self_supervised_monitoring_schema() -> DataSchema:
    """Create the schema for self-supervised server monitoring anomaly detection."""
    feature_names = [
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
    return DataSchema(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        required_columns=feature_names,
        min_rows=1,
        max_rows=100000,
        value_ranges={
            "request_count": (0.0, 5000.0),
            "bytes_per_request": (0.0, 50000.0),
            "cpu_usage": (0.0, 100.0),
            "memory_usage": (0.0, 100.0),
            "disk_io": (0.0, 20000.0),
            "network_in": (0.0, 5000.0),
            "network_out": (0.0, 5000.0),
            "error_rate": (0.0, 100.0),
            "connection_count": (0.0, 5000.0),
            "response_time": (0.0, 5000.0),
        },
    )
