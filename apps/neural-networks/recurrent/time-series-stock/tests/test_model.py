"""Unit tests for time-series-stock model."""

import numpy as np
from time_series_stock.model import StockMarketRNN


class TestStockMarketRNN:
    """Tests for the stock market prediction RNN model."""

    def _make_data(self, n_samples=50, seed=42):
        from time_series_stock.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, y = self._make_data(n_samples=50)
        model = StockMarketRNN(
            n_features=5,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.01,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_prediction_shape(self):
        X, y = self._make_data(n_samples=50)
        model = StockMarketRNN(
            n_features=5,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.01,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=50)
        model = StockMarketRNN(
            n_features=5,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.01,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] > 0

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_data(n_samples=50)
        model = StockMarketRNN(
            n_features=5,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.01,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = StockMarketRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        np.testing.assert_allclose(loaded.model.W_hy, model.model.W_hy)
        assert loaded.n_features == model.n_features

    def test_to_dict_returns_metadata(self):
        X, y = self._make_data(n_samples=30)
        model = StockMarketRNN(
            n_features=5,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.01,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "n_features" in metadata
        assert "training_mode" in metadata

