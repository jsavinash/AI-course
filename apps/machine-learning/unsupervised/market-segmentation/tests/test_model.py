"""Unit tests for market-segmentation model."""

import numpy as np
import pytest
from market_segmentation.model import KMeans


class TestMarketSegmentationKMeans:
    """Tests for the market segmentation K-Means clustering model."""

    def _make_data(self, n_samples: int = 200) -> np.ndarray:
        """Generate simple separable clusters for testing."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=[30, 30], scale=3.0, size=(n_samples // 2, 2))
        cluster2 = rng.normal(loc=[80, 80], scale=3.0, size=(n_samples // 2, 2))
        return np.vstack([cluster1, cluster2])

    def test_fit_creates_centroids(self):
        """Test that fit produces the expected number of centroids."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        assert model.centroids is not None
        assert model.centroids.shape == (2, 2)
        assert model.labels is not None
        assert len(model.labels) == len(X)
        assert set(np.unique(model.labels)).issubset({0, 1})

    def test_clusters_are_separable(self):
        """Test that K-Means separates the two distinct clusters."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        # Points from cluster1 (low income) should mostly be in one cluster
        labels = model.labels
        low_income_labels = labels[: len(X) // 2]
        high_income_labels = labels[len(X) // 2 :]

        # The two groups should be assigned to different clusters
        assert set(low_income_labels) != set(high_income_labels)

    def test_predict_assigns_to_nearest_centroid(self):
        """Test that predict returns valid cluster indices."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        preds = model.predict(np.array([[35.0, 35.0], [75.0, 75.0]]))
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

    def test_predict_confidence_in_range(self):
        """Test that confidence scores are in [0, 1]."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        conf = model.predict_confidence(np.array([[35.0, 35.0], [75.0, 75.0]]))
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = KMeans.load(path)

        assert loaded.centroids is not None
        assert loaded.feature_mean is not None
        assert loaded.feature_std is not None
        np.testing.assert_allclose(loaded.centroids, model.centroids)
        np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
        assert loaded.n_clusters == model.n_clusters
        assert loaded.inertia == pytest.approx(model.inertia)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns unsupervised clustering metrics."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        metrics = model.evaluate(X)
        assert "inertia" in metrics
        assert "silhouette" in metrics
        assert "n_clusters" in metrics
        assert metrics["inertia"] > 0
        assert -1.0 <= metrics["silhouette"] <= 1.0

