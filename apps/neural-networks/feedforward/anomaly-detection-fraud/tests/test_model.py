"""Unit tests for anomaly-detection-fraud model."""

import numpy as np
import pytest
from anomaly_detection_fraud.model import FraudDetectionAutoencoder


class TestFraudDetectionAutoencoder:
    """Tests for the credit card fraud detection autoencoder."""

    def _make_data(self, n_samples: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        from anomaly_detection_fraud.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        """Test that training loss decreases."""
        X, y = self._make_data(n_samples=500)
        X_normal = X[y == 0]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=500, random_seed=42
        )
        model.fit(X_normal)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_labels(self):
        """Test that predictions are binary (0 or 1)."""
        X, y = self._make_data(n_samples=500)
        X_normal = X[y == 0]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=200, random_seed=42
        )
        model.fit(X_normal)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_reconstruction_error_shape(self):
        """Test that reconstruction error returns one value per sample."""
        X, y = self._make_data(n_samples=200)
        X_normal = X[y == 0]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=200, random_seed=42
        )
        model.fit(X_normal)

        errors = model.reconstruction_error(X)
        assert errors.shape == (len(X),)
        assert np.all(errors >= 0.0)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns anomaly detection metrics."""
        X, y = self._make_data(n_samples=500)
        X_normal = X[y == 0]
        X[y == 1]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=200, random_seed=42
        )
        model.fit(X_normal)

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "false_positive_rate" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y = self._make_data(n_samples=200)
        X_normal = X[y == 0]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=200, random_seed=42
        )
        model.fit(X_normal)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = FraudDetectionAutoencoder.load(path)

        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
        assert loaded.input_dim == model.input_dim
        assert loaded.threshold == pytest.approx(model.threshold)

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y = self._make_data(n_samples=100)
        X_normal = X[y == 0]

        model = FraudDetectionAutoencoder(
            hidden_dim=4, learning_rate=0.001, n_iterations=100, random_seed=42
        )
        model.fit(X_normal)

        metadata = model.to_dict()
        assert "input_dim" in metadata
        assert "hidden_dim" in metadata
        assert "training_mode" in metadata
        assert "threshold" in metadata
        assert metadata["training_mode"] == "supervised"

