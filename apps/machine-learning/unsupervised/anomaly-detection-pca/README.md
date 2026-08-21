# anomaly-detection-pca



Anomaly Detection / PCA — AI engineering example · part of the MLOps monorepo

## 1. Mathematical Foundations

This example is grounded in **Anomaly Detection / PCA**. The equations below
drive every forward and backward pass in the implementation.

$$X_{\text{centered}} = X - \bar{x}$$

$$\Sigma = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$$

$$\Sigma v = \lambda v$$

$$X_{\text{reduced}} = X_{\text{centered}} V_k$$

$$\text{recon error} = \|X - X_{\text{reconstructed}}\|^2$$

### Derivation

PCA finds orthogonal directions of maximum variance. By computing the SVD of centered data $X = U\Sigma V^T$, the right singular vectors $V$ are the principal components. Anomalies are detected from large reconstruction error after projection.

### Worked Numerical Example

$$z = w \cdot x + b$$

Illustrative forward-pass evaluation (scalar example):

Input  x        = 12.0   (e.g. pizza diameter, inches)
Weights w       =  0.85
Bias    b       =  0.30
---------------------------------
z = w*x + b
  = 0.85 * 12.0 + 0.30
  = 10.20 + 0.30
  = 10.50   <- model output

### Conceptual Diagram

        Core transformation flow
   [ Input x ] --> ( w · x + b ) --> [ Output z ]
                       |
                  [ activation ]
                       |
                  [ prediction ]

![Math & architecture diagram](./assets/math-concept.png)

Interactive 2D/3D PCA projection; explained variance scree plot; anomaly score distribution.

## 2. Core Logic & Architecture

The example follows a consistent **data → train → evaluate → serve**
pipeline. Inputs are loaded and validated, transformed by the core algorithm, scored against
held-out data, and exposed through a REST API.

  Raw dataset→
  load + validate (data.py)→
  fit / transform (model.py)→
  evaluate + persist (train.py)→
  serve (api.py)

### Primary Components

| Class | Public methods | Responsibility |
| --- | --- | --- |
| `MetricsRequest` | — | Single metrics observation for anomaly detection. |
| `MetricsBulkRequest` | — | Bulk metrics request for anomaly detection. |
| `AnomalyResponse` | — | Anomaly detection response for a single observation. |
| `BulkAnomalyResponse` | — | Bulk anomaly detection response. |
| `StatsResponse` | — | Model statistics response. |
| `ModelInfoResponse` | — | Model information response. |
| `DriftResponse` | — | Drift detection response. |
| `PCAAnomalyDetector` | feature_mean, feature_std, reconstruction_threshold, cumulative_variance_ratio, _standardize, _compute_eigen, _select_n_components, fit, _fit_threshold, _reconstruction_errors, reconstruction_error, predict_anomaly, predict, predict_proba, reconstruct, is_anomaly, anomaly_score, transform, inverse_transform, evaluate, save, load, to_dict | PCA-based anomaly detector using reconstruction error.  The model learns a low-dimensional representation of normal data using PCA. Anomalies are detected by measuring how much information is lost when reconstructing data from the reduced representation.  Args:     n_components: Number of principal components to retain.         Can be an integer or a float between 0 and 1 (variance ratio).     threshold_method: Method for computing anomaly threshold.         - "percentile": use a percentile of training reconstruction errors         - "iqr": use Q3 + multiplier * IQR         - "fixed": use a user-specified threshold     threshold_percentile: Percentile for threshold (default 95).     threshold_iqr_multiplier: IQR multiplier for threshold (default 1.5).     threshold_value: Fixed threshold value (used when method="fixed").     random_seed: Random seed for reproducibility. |

### Data Flow



1. **Load** — `data.py` reads the source dataset and splits train/test.



2. **Validate** — a Pydantic schema guards input shape/dtypes before training.



3. **Fit / Transform** — `model.py` applies the mathematics from Section 1.



4. **Evaluate** — metrics (MSE/RMSE/R², accuracy, etc.) are computed and logged.



5. **Persist** — weights/artifacts are saved and registered in the model registry.



6. **Serve** — `api.py` exposes prediction endpoints with drift detection.

### Design Patterns & Performance

Key design choices in this module: a pure-NumPy implementation (no PyTorch/TensorFlow), schema validation via `ai_core.validation`, structured JSON logging through `ai_core.logging`, Prometheus metrics from `ai_core.metrics`, and MLflow/model-registry persistence via `ai_core.model_registry`. The FastAPI service wraps the trained model with observability middleware from `ai_core.fastapi_middleware`.

## 3. Detailed Code Walkthrough

The most important behaviour is summarised below; full source for each module is collapsible
so the page stays readable while remaining self-contained.

### `PCAAnomalyDetector.fit(X)`

Train the PCA model on normal (non-anomalous) data.

Args:
    X: array of shape (n_samples, n_features) - training data
        (ideally only normal samples, no anomalies)

Returns:
    self

### `PCAAnomalyDetector.predict(X)`

Predict anomaly scores for each sample.

Args:
    X: array of shape (n_samples, n_features)

Returns:
    Array of reconstruction errors with shape (n_samples,)

### `PCAAnomalyDetector.evaluate(X, y)`

Evaluate anomaly detection performance.

Args:
    X: array of shape (n_samples, n_features)
    y: optional ground-truth labels (0=normal, 1=anomaly)

Returns:
    Dict with evaluation metrics

### Source Files

<details>
<summary>model.py</summary>

```
"""PCA-based anomaly detection model using dimensionality reduction.

Implements Principal Component Analysis from scratch with:
- Eigendecomposition-based PCA (no scikit-learn)
- Reconstruction error computation for anomaly scoring
- Automatic threshold selection using percentiles or IQR
- Proper serialization with metadata
- Univariate and multivariate anomaly detection
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

@dataclass
class PCAAnomalyDetector:
    """PCA-based anomaly detector using reconstruction error.

    The model learns a low-dimensional representation of normal data
    using PCA. Anomalies are detected by measuring how much information
    is lost when reconstructing data from the reduced representation.

    Args:
        n_components: Number of principal components to retain.
            Can be an integer or a float between 0 and 1 (variance ratio).
        threshold_method: Method for computing anomaly threshold.
            - "percentile": use a percentile of training reconstruction errors
            - "iqr": use Q3 + multiplier * IQR
            - "fixed": use a user-specified threshold
        threshold_percentile: Percentile for threshold (default 95).
        threshold_iqr_multiplier: IQR multiplier for threshold (default 1.5).
        threshold_value: Fixed threshold value (used when method="fixed").
        random_seed: Random seed for reproducibility.
    """

    n_components: int | float = 0.95
    threshold_method: Literal["percentile", "iqr", "fixed"] = "percentile"
    threshold_percentile: float = 95.0
    threshold_iqr_multiplier: float = 1.5
    threshold_value: float = 0.0
    random_seed: int = 42

    # Learned state
    components: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None
    cumulative_variance: np.ndarray | None = None
    threshold: float = 0.0
    n_features: int = 0
    n_components_selected: int = 0

    @property
    def feature_mean(self) -> np.ndarray | None:
        """Mean of features used for standardization."""
        return self.mean

    @property
    def feature_std(self) -> np.ndarray | None:
        """Standard deviation of features used for standardization."""
        return self.std

    @property
    def reconstruction_threshold(self) -> float:
        """Anomaly threshold."""
        return self.threshold

    @property
    def cumulative_variance_ratio(self) -> float:
        """Total variance explained by selected components."""
        if self.cumulative_variance is None:
            return 0.0
        return float(self.cumulative_variance[self.n_components_selected - 1])

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Standardize features to zero mean and unit variance."""
        if self.mean is None or self.std is None:
            raise ValueError("Model not trained. Call fit() first.")
        return (X - self.mean) / (self.std + 1e-8)

    def _compute_eigen(self, X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors via SVD.

        Uses SVD for numerical stability instead of explicit covariance matrix.
        """
        n_samples = X_std.shape[0]
        # Center the data (already standardized, but ensure zero mean)
        X_centered = X_std - np.mean(X_std, axis=0)

        # SVD: X = U * S * Vt
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues = S^2 / (n_samples - 1)
        eigenvalues = (S**2) / (n_samples - 1)

        # Eigenvectors = Vt.T (each column is a component)
        eigenvectors = Vt.T

        return eigenvalues, eigenvectors

    def _select_n_components(self, eigenvalues: np.ndarray, n_features: int) -> int:
        """Determine the number of components based on n_components parameter."""
        if isinstance(self.n_components, int):
            if self.n_components > n_features:
                raise ValueError(
                    f"n_components ({self.n_components}) must be <= n_features ({n_features})"
                )
            return min(self.n_components, len(eigenvalues))

        # Float: variance ratio
        total_variance = np.sum(eigenvalues)
        if total_variance == 0:
            return len(eigenvalues)

        explained = eigenvalues / total_variance
        cumulative = np.cumsum(explained)
        n_comp = np.searchsorted(cumulative, self.n_components) + 1
        return min(n_comp, len(eigenvalues))

    def fit(self, X: np.ndarray) -> "PCAAnomalyDetector":
        """Train the PCA model on normal (non-anomalous) data.

        Args:
            X: array of shape (n_samples, n_features) - training data
                (ideally only normal samples, no anomalies)

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        self.n_features = X.shape[1]

        # Compute feature statistics for standardization
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_std = self._standardize(X)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self._compute_eigen(X_std)

        # Select number of components
        self.n_components_selected = self._select_n_components(eigenvalues, self.n_features)

        # Store components (eigenvectors) and variance info
        self.components = eigenvectors[:, : self.n_components_selected]
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio = (eigenvalues / total_variance)[
                : self.n_components_selected
            ]
            self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        else:
            self.explained_variance_ratio = np.zeros(self.n_components_selected)
            self.cumulative_variance = np.zeros(self.n_components_selected)

        # Compute reconstruction errors for threshold fitting
        reconstruction_errors = self._reconstruction_errors(X_std)

        # Compute anomaly threshold
        self._fit_threshold(reconstruction_errors)

        return self

    def _fit_threshold(self, reconstruction_errors: np.ndarray) -> None:
        """Compute the anomaly threshold based on training reconstruction errors."""
        if self.threshold_method == "percentile":
            self.threshold = float(np.percentile(reconstruction_errors, self.threshold_percentile))
        elif self.threshold_method == "iqr":
            q1 = float(np.percentile(reconstruction_errors, 25))
            q3 = float(np.percentile(reconstruction_errors, 75))
            iqr = q3 - q1
            self.threshold = q3 + self.threshold_iqr_multiplier * iqr
        elif self.threshold_method == "fixed":
            self.threshold = self.threshold_value
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def _reconstruction_errors(self, X_std: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors for standardized data."""
        if self.components is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Project onto principal components
        projected = X_std @ self.components

        # Reconstruct
        reconstructed = projected @ self.components.T

        # Compute squared reconstruction error per sample
        errors = np.sum((X_std - reconstructed) ** 2, axis=1)

        return errors

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute PCA reconstruction error for each sample.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of reconstruction errors with shape (n_samples,)
        """
        if self.components is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_std = self._standardize(X)
        return self._reconstruction_errors(X_std)

    # ---------- Prediction API (test-compatible) ----------

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels (0=normal, 1=anomaly).

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of ints (0 or 1) with shape (n_samples,)
        """
        return self.is_anomaly(X).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for each sample.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of reconstruction errors with shape (n_samples,)
        """
        return self.reconstruction_error(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities in [0, 1].

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of probabilities in [0, 1] with shape (n_samples,)
        """
        return self.anomaly_score(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct data from principal component space.

        Alias for inverse_transform.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_features) - reconstructed data
        """
        return self.inverse_transform(self.transform(X))

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Classify samples as anomalous or normal based on reconstruction error.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Boolean array of shape (n_samples,) - True if anomalous
        """
        errors = self.reconstruction_error(X)
        return errors > self.threshold

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized anomaly scores in [0, 1].

        Score is computed as: min(error / max(error, threshold), 1.0)

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of anomaly scores in [0, 1] with shape (n_samples,)
        """
        errors = self.reconstruction_error(X)
        denom = max(float(np.max(errors)), self.threshold)
        scores = np.clip(errors / denom, 0.0, 1.0)
        return scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_components_selected)
        """
        if self.components is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_std = self._standardize(X)
        return X_std @ self.components

    def inverse_transform(self, X_projected: np.ndarray) -> np.ndarray:
        """Reconstruct data from principal component space.

        Args:
            X_projected: array of shape (n_samples, n_components_selected)

        Returns:
            Array of shape (n_samples, n_features) - reconstructed data
        """
        if self.components is None:
            raise ValueError("Model not trained. Call fit() first.")

        reconstructed_std = X_projected @ self.components.T
        return reconstructed_std * self.std + self.mean

    def evaluate(self, X: np.ndarray, y: np.ndarray | None = None) -> dict[str, float]:
        """Evaluate anomaly detection performance.

        Args:
            X: array of shape (n_samples, n_features)
            y: optional ground-truth labels (0=normal, 1=anomaly)

        Returns:
            Dict with evaluation metrics
        """
        if self.components is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        errors = self.reconstruction_error(X)

        metrics = {
            "mean_reconstruction_error": float(np.mean(errors)),
            "std_reconstruction_error": float(np.std(errors)),
            "max_reconstruction_error": float(np.max(errors)),
            "anomaly_threshold": float(self.threshold),
            "n_components": float(self.n_components_selected),
            "explained_variance_ratio": float(
                np.sum(self.explained_variance_ratio[: self.n_components_selected])
                if self.explained_variance_ratio is not None
                else 0.0
            ),
        }

        if y is not None:
            predictions = self.is_anomaly(X)

            tp = int(np.sum((predictions == 1) & (y == 1)))
            fp = int(np.sum((predictions == 1) & (y == 0)))
            tn = int(np.sum((predictions == 0) & (y == 0)))
            fn = int(np.sum((predictions == 0) & (y == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(y) if len(y) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            metrics.update(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positive_rate": fpr,
                    "true_positives": float(tp),
                    "false_positives": float(fp),
                    "true_negatives": float(tn),
                    "false_negatives": float(fn),
                }
            )

        return metrics

    # ---------- Serialization ----------

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.components is None:
            raise ValueError("Cannot save untrained model")

        np.savez(
            path,
            components=self.components,
            mean=self.mean,
            std=self.std,
            explained_variance_ratio=self.explained_variance_ratio,
            cumulative_variance=self.cumulative_variance,
            n_features=np.array([self.n_features]),
            n_components=np.array([self.n_components_selected]),
            threshold=np.array([self.threshold]),
            threshold_method=np.array([self.threshold_method]),
            threshold_percentile=np.array([self.threshold_percentile]),
            threshold_iqr_multiplier=np.array([self.threshold_iqr_multiplier]),
... (truncated) ...
```

</details>

<details>
<summary>train.py</summary>

```
"""Production training pipeline for PCA-based anomaly detection."""

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_anomaly_detection_schema

from anomaly_detection.data import load_training_data, save_training_data
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)

def train(
    model_dir: Path,
    data_path: Path,
    n_components: int | float,
    threshold_method: str,
    threshold_percentile: float,
    threshold_iqr_multiplier: float,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the PCA anomaly detection model and save artifacts.

    Args:
        model_dir: Directory to save model artifacts
        data_path: Optional path to CSV data
        n_components: Number of PCA components or variance ratio to retain
        threshold_method: Method for anomaly threshold ("percentile", "iqr", "fixed")
        threshold_percentile: Percentile for threshold if method="percentile"
        threshold_iqr_multiplier: IQR multiplier if method="iqr"
        model_version: Model version string
        register_to_mlflow: Whether to register to MLflow
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with training metrics
    """
    # Load training data
    X, y = load_training_data(data_path, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), n_features=X.shape[1])

    # Validate training data
    validator = DataValidator(create_anomaly_detection_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / "training_data.csv")

    # Use only normal samples for PCA training (unsupervised anomaly detection)
    X_normal = X[y == 0]
    logger.info("Training on normal samples", n_normal=len(X_normal), n_anomaly=int(np.sum(y)))

    # Train model
    model = PCAAnomalyDetector(
        n_components=n_components,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        threshold_iqr_multiplier=threshold_iqr_multiplier,
        random_seed=random_seed,
    )
    model.fit(X_normal)

    # Evaluate on all data
    metrics = model.evaluate(X, y)
    logger.info(
        "Training complete",
        n_components=model.n_components_selected,
        explained_variance=metrics["explained_variance_ratio"],
        threshold=model.threshold,
        mean_error=metrics["mean_reconstruction_error"],
        max_error=metrics["max_reconstruction_error"],
    )

    if "accuracy" in metrics:
        logger.info(
            "Evaluation metrics",
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )

    # Save model
    model_path = model_dir / f"anomaly_detection_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, y, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        "mean_reconstruction_error": metrics["mean_reconstruction_error"],
        "std_reconstruction_error": metrics["std_reconstruction_error"],
        "max_reconstruction_error": metrics["max_reconstruction_error"],
        "threshold": model.threshold,
        "n_components": float(model.n_components_selected),
        "explained_variance_ratio": metrics["explained_variance_ratio"],
        "n_samples": float(len(X)),
        "n_normal": float(len(X_normal)),
        "n_anomaly": float(int(np.sum(y))),
    }

    if "accuracy" in metrics:
        training_metrics.update(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "true_negatives": metrics["true_negatives"],
                "false_negatives": metrics["false_negatives"],
            }
        )

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="anomaly-detection",
        model_version=model_version,
        model_type="anomaly_detection",
        metrics=training_metrics,
        parameters={
            "n_components": n_components,
            "threshold_method": threshold_method,
            "threshold_percentile": threshold_percentile,
            "threshold_iqr_multiplier": threshold_iqr_multiplier,
            "random_seed": random_seed,
        },
        artifacts={
            f"anomaly_detection_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "anomaly_detection", "method": "pca"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="anomaly-detection",
            model_version=model_version,
            metrics=training_metrics,
            params={
                "n_components": n_components,
                "threshold_method": threshold_method,
                "threshold_percentile": threshold_percentile,
                "threshold_iqr_multiplier": threshold_iqr_multiplier,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"anomaly_detection_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "anomaly_detection", "framework": "numpy", "method": "pca"},
        )
        logger.info("Registered model to MLflow", model="anomaly-detection", version=model_version)

    return training_metrics

def _save_chart(
    model: PCAAnomalyDetector,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    version: str,
) -> None:
    """Save the anomaly detection visualization chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model.components is None:
        return

    # Project data to 2D using first 2 principal components
    projected = model.transform(X)
    errors = model.reconstruction_error(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA projection colored by anomaly
    ax1 = axes[0]
    normal_mask = y == 0
    anomaly_mask = y == 1

    ax1.scatter(
        projected[normal_mask, 0],
        projected[normal_mask, 1],
        c="steelblue",
        s=30,
        alpha=0.5,
        label="Normal",
    )
    ax1.scatter(
        projected[anomaly_mask, 0],
        projected[anomaly_mask, 1],
        c="crimson",
        s=50,
        alpha=0.8,
        marker="x",
        label="Anomaly",
    )
    ax1.set_xlabel(f"PC1 ({model.explained_variance_ratio[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({model.explained_variance_ratio[1]:.1%} variance)")
    ax1.set_title(f"PCA Projection - v{version}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Reconstruction error histogram with threshold
    ax2 = axes[1]
    ax2.hist(
        errors[normal_mask],
        bins=50,
        alpha=0.6,
        label="Normal",
        color="steelblue",
        density=True,
    )
    ax2.hist(
        errors[anomaly_mask],
        bins=50,
        alpha=0.6,
        label="Anomaly",
        color="crimson",
        density=True,
    )
    ax2.axvline(
        model.threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({model.threshold:.2f})",
    )
    ax2.set_xlabel("Reconstruction Error")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Reconstruction Error Distribution - v{version}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    chart_path = output_dir / f"anomaly_detection_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))

def main():
    parser = argparse.ArgumentParser(description="Train PCA anomaly detection model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-components", type=str, default=os.getenv("N_COMPONENTS", "0.95"))
    parser.add_argument(
        "--threshold-method", type=str, default=os.getenv("THRESHOLD_METHOD", "percentile")
    )
    parser.add_argument(
        "--threshold-percentile", type=float, default=float(os.getenv("THRESHOLD_PERCENTILE", "95"))
    )
    parser.add_argument(
        "--threshold-iqr-multiplier",
        type=float,
        default=float(os.getenv("THRESHOLD_IQR_MULTIPLIER", "1.5")),
    )
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    # Parse n_components (could be int or float)
    n_components: int | float
    try:
        n_components = int(args.n_components)
    except ValueError:
        n_components = float(args.n_components)

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_components=n_components,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_iqr_multiplier=args.threshold_iqr_multiplier,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))

if __name__ == "__main__":
    main()
```

</details>

<details>
<summary>data.py</summary>

```
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
```

</details>

<details>
<summary>api.py</summary>

```
"""Production serving API for PCA-based anomaly detection."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_anomaly_detection_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from anomaly_detection.data import FEATURE_NAMES
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("ANOMALY_METRICS_PORT", "8005")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class MetricsRequest(BaseModel):
    """Single metrics observation for anomaly detection."""

    request_count: float = Field(..., ge=0, description="Number of requests")
    bytes_per_request: float = Field(..., ge=0, description="Average bytes per request")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_io: float = Field(..., ge=0, description="Disk I/O operations per second")
    network_in: float = Field(..., ge=0, description="Network inbound MB/s")
    network_out: float = Field(..., ge=0, description="Network outbound MB/s")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage")
    connection_count: float = Field(..., ge=0, description="Active connections")
    response_time: float = Field(..., ge=0, description="Average response time in ms")

class MetricsBulkRequest(BaseModel):
    """Bulk metrics request for anomaly detection."""

    samples: list[MetricsRequest] = Field(..., min_length=1, max_length=100)

class AnomalyResponse(BaseModel):
    """Anomaly detection response for a single observation."""

    is_anomaly: bool
    anomaly_score: float
    anomaly_probability: float
    reconstruction_error: float
    anomaly_threshold: float
    model_version: str

class BulkAnomalyResponse(BaseModel):
    """Bulk anomaly detection response."""

    samples: list[AnomalyResponse]
    n_anomalies: int
    n_samples: int
    model_version: str

class StatsResponse(BaseModel):
    """Model statistics response."""

    n_features: int
    n_components: int
    explained_variance_ratio: float
    reconstruction_threshold: float
    threshold_method: str
    mean_reconstruction_error: float
    max_reconstruction_error: float
    model_version: str

class ModelInfoResponse(BaseModel):
    """Model information response."""

    n_components: int
    n_features: int
    feature_names: list[str]
    cumulative_variance_ratio: float
    reconstruction_threshold: float
    model_version: str

class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]

# Global model state
_model: PCAAnomalyDetector | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("anomaly_detection", port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_anomaly_detection_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: "float" for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="anomaly-detection", model_version=_model_version, model_type="anomaly_detection"
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="anomaly-detection", version=_model_version)

    yield

    logger.info("Shutting down anomaly-detection API")

def _load_model() -> tuple[PCAAnomalyDetector, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            ad_models = [m for m in models if m.get("model_name") == "anomaly-detection"]
            if ad_models:
                ad_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = ad_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("anomaly_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "anomaly-detection" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("anomaly_detection_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "anomaly_detection_model.npz"
    if npz_path.exists():
        return PCAAnomalyDetector.load(str(npz_path)), "legacy"

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/anomaly_detection_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "anomaly_detection_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return PCAAnomalyDetector.load(str(p)), "1.0.0-bundled"

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning("No pre-existing model found on disk. Initializing baseline PCA model.")
    from anomaly_detection.data import load_training_data

    X_base, y_base = load_training_data(None)
    X_normal = X_base[y_base == 0]
    model = PCAAnomalyDetector(n_components=0.95, threshold_method="percentile", random_seed=42)
    model.fit(X_normal)
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_csvs = [
        MODEL_DIR / "anomaly-detection" / _model_version / "training_data.csv",
        MODEL_DIR / "training_data.csv",
        Path("/app/artifacts/models/training_data.csv"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "training_data.csv",
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if all(f in df.columns for f in FEATURE_NAMES):
                    return df[FEATURE_NAMES].values
            except Exception as e:
                logger.warning("Could not read reference csv", path=str(csv_path), error=str(e))

    from anomaly_detection.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base

# Create FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="PCA-based anomaly detection using dimensionality reduction",
    version="1.0.0",
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)

@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "anomaly-detection-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "features": FEATURE_NAMES,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_bulk": "POST /predict/bulk",
            "stats": "GET /stats",
            "model_info": "GET /model/info",
            "drift": "GET /drift",
            "metrics": "/metrics",
        },
    }

@app.get("/health")
def health_check():
    """Kubernetes liveness/readiness probe."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
    }

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/reload")
def reload_model():
    """Dynamically reload the model from disk/registry."""
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name="anomaly-detection",
                model_version=_model_version,
                model_type="anomaly_detection",
            )
        _reference_data = _load_reference_data()
        logger.info("Model reloaded dynamically", model="anomaly-detection", version=_model_version)
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e

@app.get("/drift", response_model=DriftResponse)
def drift_check():
    """Check for data drift between reference and recent predictions."""
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")

    if len(_recent_predictions) < 10:
        return DriftResponse(
            total_features=len(FEATURE_NAMES),
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])

    return DriftResponse(**summary)

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return model statistics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    evr = _model.explained_variance_ratio

    return StatsResponse(
        n_features=_model.n_features,
        n_components=_model.n_components_selected,
        explained_variance_ratio=float(
            np.sum(evr[: _model.n_components_selected]) if evr is not None else 0.0
        ),
        reconstruction_threshold=round(_model.threshold, 4),
        threshold_method=_model.threshold_method,
        mean_reconstruction_error=float(
            np.mean(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        max_reconstruction_error=float(
            np.max(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        model_version=_model_version,
    )

@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """Return detailed model information."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        n_components=_model.n_components_selected,
        n_features=_model.n_features,
        feature_names=FEATURE_NAMES,
        cumulative_variance_ratio=_model.cumulative_variance_ratio,
        reconstruction_threshold=round(_model.threshold, 4),
        model_version=_model_version,
    )

def _compute_anomaly(observation: MetricsRequest) -> AnomalyResponse:
    """Core anomaly detection logic shared by all detection endpoints."""
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    X = np.array(
        [
            [
                observation.request_count,
                observation.bytes_per_request,
                observation.cpu_usage,
                observation.memory_usage,
                observation.disk_io,
                observation.network_in,
                observation.network_out,
                observation.error_rate,
                observation.connection_count,
                observation.response_time,
            ]
        ]
    )
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        recon_error = float(_model.reconstruction_error(X)[0])
... (truncated) ...
```

</details>

## 4. Monorepo Integration

This example is a first-class consumer of the shared `packages/ai-core` library.
It reuses the following foundation modules instead of re-implementing infrastructure:

ai_core.drift
ai_core.fastapi_middleware
ai_core.logging
ai_core.metrics
ai_core.model_registry
ai_core.validation

### How it plugs in



- **Configuration** — 12-factor config from `ai_core.config`.



- **Observability** — structured logging + Prometheus metrics are wired in automatically.



- **Validation** — input schema validation prevents bad data reaching the model.



- **Registry** — trained artifacts are versioned and registered for reproducible serving.



- **Serving** — the FastAPI app mounts shared observability middleware for tracing & metrics.

Because every example shares `ai_core`, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.
