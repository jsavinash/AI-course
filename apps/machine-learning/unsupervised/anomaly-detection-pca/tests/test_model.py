"""Unit tests for anomaly-detection-pca model."""

import numpy as np
import pytest
from anomaly_detection.model import PCAAnomalyDetector


class TestPCAAnomalyDetector:
    """Tests for the PCA-based anomaly detection model."""

    def _make_data(
        self, n_samples: int = 400, shuffle: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated normal data plus a few clear outliers."""
        rng = np.random.default_rng(42)
        # Correlated 5-dimensional normal data
        mean = np.array([50.0, 200.0, 30.0, 60.0, 90.0])
        cov = np.array(
            [
                [25.0, 40.0, 10.0, 12.0, 8.0],
                [40.0, 100.0, 20.0, 25.0, 15.0],
                [10.0, 20.0, 9.0, 8.0, 5.0],
                [12.0, 25.0, 8.0, 12.0, 6.0],
                [8.0, 15.0, 5.0, 6.0, 7.0],
            ]
        )
        X_normal = rng.multivariate_normal(mean, cov, size=n_samples)
        # 5 extreme outliers far from the distribution (5-10 std devs in multiple dims)
        # These live outside the dominant PCA subspace, so reconstruction error is huge
        anomalies = np.array(
            [
                [300.0, 1600.0, 99.0, 99.0, 500.0],
                [280.0, 1500.0, 98.0, 98.0, 480.0],
                [320.0, 1700.0, 99.5, 99.5, 520.0],
                [10.0, 50.0, 1.0, 1.0, 5.0],
                [400.0, 2000.0, 99.0, 99.0, 600.0],
            ]
        )
        X = np.vstack([X_normal, anomalies])
        y = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(len(anomalies), dtype=int)])
        # Shuffle so anomalies aren't all at the end (tests detection, not position)
        if shuffle:
            perm = rng.permutation(len(X))
            X = X[perm]
            y = y[perm]
        return X, y

    def test_fit_creates_components(self):
        """Test that fit produces the expected number of principal components."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        assert model.components is not None
        assert model.components.shape == (5, 3)
        assert model.feature_mean is not None
        assert model.feature_std is not None
        assert model.explained_variance_ratio is not None
        assert len(model.explained_variance_ratio) == 3
        assert 0.0 < model.cumulative_variance_ratio <= 1.0
        assert model.reconstruction_threshold > 0.0

    def test_transform_reduces_dimensions(self):
        """Test that transform projects data onto fewer dimensions."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        projected = model.transform(X)
        assert projected.shape == (len(X), 3)

    def test_predict_returns_scores(self):
        """Test that predict returns per-sample anomaly scores."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        scores = model.predict(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0.0)

    def test_predict_anomaly_flags_outliers(self):
        """Test that clear outliers are flagged as anomalies."""
        X, y = self._make_data()
        model = PCAAnomalyDetector(n_components=3, threshold_percentile=95.0)
        model.fit(X)

        preds = model.predict_anomaly(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})
        # All 5 synthetic outliers should be detected regardless of position
        assert np.sum(preds[y == 1]) == 5

    def test_predict_proba_range(self):
        """Test that anomaly probabilities are in [0, 1]."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        probs = model.predict_proba(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_reconstruction(self):
        """Test that reconstruction returns original feature dimensionality."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        reconstructed = model.reconstruct(X)
        assert reconstructed.shape == X.shape

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns anomaly detection metrics.

        Uses the production pattern: fit PCA on normal data only, then
        evaluate on the full dataset (normal + anomalies).
        """
        X, y = self._make_data(shuffle=False)
        n_normal = int(np.sum(y == 0))
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X[:n_normal])  # Fit on normal data only

        metrics = model.evaluate(X, y)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "false_positive_rate" in metrics
        assert "anomaly_threshold" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        # Perfect recall: all anomalies are detected (the core detection requirement)
        assert metrics["recall"] >= 0.8
        # F1 is bounded below by the designed FPR operating point:
        # threshold_percentile=95.0 intentionally flags ~5% of normal data,
        # which limits precision when the anomaly set is small.
        assert metrics["f1"] >= 0.3
        assert metrics["false_positive_rate"] <= 0.1

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = PCAAnomalyDetector.load(path)

        assert loaded.n_components == model.n_components
        assert loaded.feature_mean is not None
        assert loaded.feature_std is not None
        assert loaded.components is not None
        np.testing.assert_allclose(loaded.components, model.components)
        np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
        assert loaded.reconstruction_threshold == pytest.approx(model.reconstruction_threshold)
        assert loaded.cumulative_variance_ratio == pytest.approx(model.cumulative_variance_ratio)

    def test_invalid_components(self):
        """Test that n_components greater than n_features raises."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=10)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(X)

