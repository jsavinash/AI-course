"""Unit tests for regression-house-price model."""

import numpy as np
from regression_house_price.model import HousePriceNN


class TestHousePriceNN:
    """Tests for the house price prediction feedforward neural network."""

    def _make_data(self, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        from regression_house_price.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        """Test that training loss decreases."""
        X, y = self._make_data(n_samples=200)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=500, random_seed=42)
        model.fit(X, y)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_prediction_shape(self):
        """Test that predictions return correct shape."""
        X, y = self._make_data(n_samples=200)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_returns_positive_prices(self):
        """Test that predicted prices are positive."""
        X, y = self._make_data(n_samples=200)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert np.all(preds > 0)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns regression metrics."""
        X, y = self._make_data(n_samples=200)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        metrics = model.evaluate(X, y)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] > 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y = self._make_data(n_samples=200)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = HousePriceNN.load(path)

        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
        assert loaded.input_dim == model.input_dim

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y = self._make_data(n_samples=100)
        model = HousePriceNN(hidden_dim=16, learning_rate=0.01, n_iterations=100, random_seed=42)
        model.fit(X, y)

        metadata = model.to_dict()
        assert "input_dim" in metadata
        assert "hidden_dim" in metadata
        assert "training_mode" in metadata
        assert "n_epochs_run" in metadata
        assert metadata["training_mode"] == "supervised"

