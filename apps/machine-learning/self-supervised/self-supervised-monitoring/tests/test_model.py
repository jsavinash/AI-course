"""Unit tests for self-supervised-monitoring model."""

import numpy as np
from self_supervised_monitoring.data import generate_synthetic_data
from self_supervised_monitoring.model import DenoisingAutoencoder


class TestDenoisingAutoencoder:
    """Tests for the self-supervised DenoisingAutoencoder model."""

    def _make_data(self, n_samples: int = 500, seed: int = 42):
        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, _ = self._make_data(n_samples=500)
        X_normal = X[:400]
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=500, noise_rate=0.25, random_seed=42
        )
        model.fit(X_normal)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "self-supervised"

    def test_predict_proba_range(self):
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)
        probas = model.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=100)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X[y == 0])
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = DenoisingAutoencoder.load(path)
        assert loaded.input_dim == model.input_dim
        assert loaded.hidden_dim == model.hidden_dim
        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
