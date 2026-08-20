"""K-Means clustering model for market segmentation.

Implements a production-ready K-Means clustering model with:
- Standard K-Means training with multiple initializations
- Feature standardization
- Confidence scoring based on distance to nearest centroid
- Proper serialization with metadata
- Cluster profiling for business interpretation
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeans:
    """K-Means clustering for customer market segmentation.

    Uses Lloyd's algorithm with multiple random initializations (n_init)
    and picks the run with the lowest inertia.
    """

    n_clusters: int = 5
    max_iterations: int = 300
    n_init: int = 10
    random_seed: int = 42
    centroids: np.ndarray | None = None
    labels: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    inertia: float = 0.0
    n_iterations_used: int = 0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Standardize features to zero mean and unit variance."""
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Model not trained. Call fit() first.")
        return (X - self.feature_mean) / (self.feature_std + 1e-8)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances from each point to each centroid."""
        # X: (n_samples, n_features), centroids: (n_clusters, n_features)
        # Returns: (n_samples, n_clusters)
        return np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)

    def _fit_once(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Run K-Means once with a single random initialization."""
        n_samples, n_features = X.shape

        # Random initialization: pick k random points as initial centroids
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()

        labels = np.zeros(n_samples, dtype=int)
        inertia = 0.0
        n_iter = 0

        for _ in range(self.max_iterations):
            n_iter += 1
            # Assign points to nearest centroid
            distances = self._compute_distances(X, centroids)
            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            # Update centroids
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = np.mean(X[mask], axis=0)

            # Check centroid convergence
            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        # Compute final inertia
        distances = self._compute_distances(X, centroids)
        min_distances = np.min(distances, axis=1)
        inertia = float(np.sum(min_distances))

        return centroids, labels, inertia, n_iter

    def fit(self, X: np.ndarray) -> "KMeans":
        """Train the K-Means model on the given data.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        # Compute feature statistics for standardization
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        X_std = self._standardize(X)

        rng = np.random.default_rng(self.random_seed)

        best_centroids = None
        best_labels = None
        best_inertia = float("inf")
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_once(X_std, rng)
            if inertia < best_inertia:
                best_centroids = centroids
                best_labels = labels
                best_inertia = inertia
                best_n_iter = n_iter

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        self.n_iterations_used = best_n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest cluster centroid.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of cluster indices with shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_std = self._standardize(X)
        distances = self._compute_distances(X_std, self.centroids)
        return np.argmin(distances, axis=1)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Compute confidence scores for each prediction.

        Confidence is based on the relative distance to the nearest vs
        second-nearest centroid. A value of 1.0 means the point is very
        close to its assigned centroid relative to others.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Array of confidence scores in [0, 1] with shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_std = self._standardize(X)
        distances = self._compute_distances(X_std, self.centroids)

        # Sort distances to get nearest and second-nearest
        sorted_distances = np.sort(distances, axis=1)
        nearest = sorted_distances[:, 0]
        second_nearest = sorted_distances[:, 1]

        # Confidence: 1 - (nearest / second_nearest), clipped to [0, 1]
        # If second_nearest is 0 (duplicate points), confidence is 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(second_nearest > 0, nearest / second_nearest, 0.0)
        confidence = np.clip(1.0 - ratio, 0.0, 1.0)

        return confidence

    def cluster_profiles(self, X: np.ndarray) -> list[dict]:
        """Compute cluster profiles for business interpretation.

        Args:
            X: array of shape (n_samples, n_features) - typically reference data

        Returns:
            List of dicts with cluster statistics
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        labels = self.predict(X)

        profiles = []
        for k in range(self.n_clusters):
            mask = labels == k
            cluster_points = X[mask]
            n_members = int(np.sum(mask))

            if n_members > 0:
                profile = {
                    "cluster": k,
                    "n_members": n_members,
                    "percentage": round(float(n_members / len(X) * 100), 2),
                    "annual_income_mean": round(float(np.mean(cluster_points[:, 0])), 2),
                    "annual_income_std": round(float(np.std(cluster_points[:, 0])), 2),
                    "spending_score_mean": round(float(np.mean(cluster_points[:, 1])), 2),
                    "spending_score_std": round(float(np.std(cluster_points[:, 1])), 2),
                }
            else:
                profile = {
                    "cluster": k,
                    "n_members": 0,
                    "percentage": 0.0,
                    "annual_income_mean": 0.0,
                    "annual_income_std": 0.0,
                    "spending_score_mean": 0.0,
                    "spending_score_std": 0.0,
                }
            profiles.append(profile)

        return profiles

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        """Compute unsupervised clustering evaluation metrics.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            Dict with 'inertia', 'silhouette', and 'n_clusters'
        """
        if self.centroids is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        labels = self.predict(X)

        # Inertia (within-cluster sum of squares)
        X_std = self._standardize(X)
        distances = self._compute_distances(X_std, self.centroids)
        min_distances = np.min(distances, axis=1)
        inertia = float(np.sum(min_distances))

        # Silhouette score (simplified computation)
        silhouette = self._compute_silhouette(X_std, labels)

        return {
            "inertia": inertia,
            "silhouette": silhouette,
            "n_clusters": float(self.n_clusters),
        }

    def _compute_silhouette(self, X_std: np.ndarray, labels: np.ndarray) -> float:
        """Compute the average silhouette score."""
        n_samples = len(X_std)
        if n_samples < 2 or self.n_clusters < 2:
            return 0.0

        # Compute pairwise distances
        # For efficiency, use a simplified approach with cluster centroids
        silhouette_sum = 0.0
        n_valid = 0

        for i in range(n_samples):
            label = labels[i]
            point = X_std[i]

            # Compute mean distance to own cluster (a)
            own_mask = labels == label
            if np.sum(own_mask) <= 1:
                continue  # Singleton cluster
            own_points = X_std[own_mask]
            a = np.mean(np.sqrt(np.sum((own_points - point) ** 2, axis=1)))

            # Compute mean distance to nearest other cluster (b)
            b = float("inf")
            for k in range(self.n_clusters):
                if k == label:
                    continue
                other_mask = labels == k
                if np.sum(other_mask) == 0:
                    continue
                other_points = X_std[other_mask]
                dist = np.mean(np.sqrt(np.sum((other_points - point) ** 2, axis=1)))
                b = min(b, dist)

            if b == float("inf"):
                continue

            # Silhouette for this point
            s = (b - a) / max(a, b)
            silhouette_sum += s
            n_valid += 1

        if n_valid == 0:
            return 0.0

        return float(silhouette_sum / n_valid)

    # ---------- Serialization ----------

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        if self.centroids is None:
            raise ValueError("Cannot save untrained model")

        np.savez(
            path,
            centroids=self.centroids,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            n_clusters=self.n_clusters,
            max_iterations=self.max_iterations,
            n_init=self.n_init,
            random_seed=self.random_seed,
            inertia=self.inertia,
            n_iterations_used=self.n_iterations_used,
        )

    @classmethod
    def load(cls, path: str) -> "KMeans":
        """Load model parameters from disk."""
        data = np.load(path)
        model = cls(
            n_clusters=int(data["n_clusters"]),
            max_iterations=int(data["max_iterations"]),
            n_init=int(data["n_init"]),
            random_seed=int(data["random_seed"]),
        )
        model.centroids = data["centroids"]
        model.feature_mean = data["feature_mean"]
        model.feature_std = data["feature_std"]
        model.inertia = float(data["inertia"])
        model.n_iterations_used = int(data["n_iterations_used"])
        return model

    def to_dict(self) -> dict:
        """Return model parameters as a dict."""
        return {
            "n_clusters": self.n_clusters,
            "max_iterations": self.max_iterations,
            "n_init": self.n_init,
            "random_seed": self.random_seed,
            "inertia": self.inertia,
            "n_iterations_used": self.n_iterations_used,
        }
