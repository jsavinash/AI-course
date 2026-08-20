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
        eigenvalues = (S ** 2) / (n_samples - 1)

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
            self.explained_variance_ratio = (eigenvalues / total_variance)[: self.n_components_selected]
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

            metrics.update({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
                "true_positives": float(tp),
                "false_positives": float(fp),
                "true_negatives": float(tn),
                "false_negatives": float(fn),
            })

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
            threshold_value=np.array([self.threshold_value]),
            random_seed=np.array([self.random_seed]),
            n_components_param=np.array([self.n_components]),
        )

    @classmethod
    def load(cls, path: str) -> "PCAAnomalyDetector":
        """Load model parameters from disk."""
        data = np.load(path, allow_pickle=True)

        model = cls(
            n_components=int(data["n_components_param"].item()),
            threshold_method=str(data["threshold_method"].item()),
            threshold_percentile=float(data["threshold_percentile"].item()),
            threshold_iqr_multiplier=float(data["threshold_iqr_multiplier"].item()),
            threshold_value=float(data["threshold_value"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        model.components = data["components"]
        model.mean = data["mean"]
        model.std = data["std"]
        model.explained_variance_ratio = data["explained_variance_ratio"]
        model.cumulative_variance = data["cumulative_variance"]
        model.n_features = int(data["n_features"].item())
        model.n_components_selected = int(data["n_components"].item())
        model.threshold = float(data["threshold"].item())
        return model

    def to_dict(self) -> dict:
        """Return model parameters as a dict."""
        return {
            "n_components": self.n_components,
            "n_components_selected": self.n_components_selected,
            "threshold": self.threshold,
            "threshold_method": self.threshold_method,
            "threshold_percentile": self.threshold_percentile,
            "threshold_iqr_multiplier": self.threshold_iqr_multiplier,
            "n_features": self.n_features,
            "explained_variance_ratio": float(
                np.sum(self.explained_variance_ratio[: self.n_components_selected])
                if self.explained_variance_ratio is not None
                else 0.0
            ),
            "random_seed": self.random_seed,
        }
