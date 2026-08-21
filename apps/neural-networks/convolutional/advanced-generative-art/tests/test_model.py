"""Unit tests for advanced-generative-art model."""

import numpy as np
from advanced_generative_art.model import *  # noqa: F403


class TestGenerativeArtDN:
    """Tests for the generative art deconvolutional network."""

    def _make_data(self, n_samples=20, seed=42):
        from advanced_generative_art.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_produces_loss_history(self):
        from advanced_generative_art.model import GenerativeArtDN

        X, y = self._make_data(n_samples=10)
        model = GenerativeArtDN(
            n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0

    def test_predict_returns_image(self):
        from advanced_generative_art.model import GenerativeArtDN

        X, y = self._make_data(n_samples=10)
        model = GenerativeArtDN(
            n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape[0] == 5

    def test_evaluate_returns_metrics(self):
        from advanced_generative_art.model import GenerativeArtDN

        X, y = self._make_data(n_samples=10)
        model = GenerativeArtDN(
            n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metrics = model.evaluate(X[:5], y[:5])
        assert "mse" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        from advanced_generative_art.model import GenerativeArtDN

        X, y = self._make_data(n_samples=10)
        model = GenerativeArtDN(
            n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = GenerativeArtDN.load(path)
        np.testing.assert_allclose(model.predict(X[:5]), loaded.predict(X[:5]), atol=1e-5)

