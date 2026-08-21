"""Unit tests for pizza-price model."""

import numpy as np
import pytest
from pizza_price.model import LinearRegression


class TestPizzaLinearRegression:
    """Tests for the pizza price Linear Regression model."""

    def test_training_converges(self):
        """Test that gradient descent converges to reasonable parameters."""
        X = np.array([6, 8, 10, 14, 18], dtype=float)
        y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)

        model = LinearRegression(learning_rate=0.001, n_iterations=2000)
        model.fit(X, y)

        # Weight should be positive (bigger pizza = more expensive)
        assert model.weight > 0
        # Sanity check on reasonable range
        assert 0.5 < model.weight < 2.0
        assert -2.0 < model.bias < 5.0

        # MSE should be low for this small dataset
        mse = model.mse(X, y)
        assert mse < 5.0

    def test_prediction_shape(self):
        """Test that predictions return correct shape."""
        model = LinearRegression()
        model.weight = 1.0
        model.bias = 0.5

        preds = model.predict(np.array([6.0, 8.0, 12.0]))
        assert preds.shape == (3,)
        np.testing.assert_allclose(preds, [6.5, 8.5, 12.5])

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X = np.array([6, 8, 10, 14, 18], dtype=float)
        y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)

        model = LinearRegression()
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = LinearRegression.load(path)

        assert loaded.weight == pytest.approx(model.weight)
        assert loaded.bias == pytest.approx(model.bias)
        assert len(loaded.loss_history) == len(model.loss_history)

