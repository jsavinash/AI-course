"""Unit tests for spam-classification LogisticRegression model."""

import numpy as np
from spam_classification.model import LogisticRegression


class TestLogisticRegression:
    """Tests for the email spam detection logistic regression model."""

    def _make_data(self, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        from spam_classification.data import load_training_data

        X, y = load_training_data(None)
        rng = np.random.default_rng(seed)
        # Expand the small built-in dataset into a randomized synthetic set.
        idx = rng.integers(0, len(X), size=n_samples)
        return X[idx], y[idx]

    def test_training_converges(self):
        """Test that training loss decreases and predictions are produced."""
        X, y = self._make_data()
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.weights is not None

    def test_predict_proba_range(self):
        """Test that probabilities are in [0, 1]."""
        X, y = self._make_data()
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_returns_valid_labels(self):
        """Test that predictions are binary (0 or 1)."""
        X, y = self._make_data()
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics in valid ranges."""
        X, y = self._make_data()
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y = self._make_data()
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = LogisticRegression.load(path)

        np.testing.assert_allclose(loaded.weights, model.weights)
        assert loaded.bias == model.bias
        np.testing.assert_allclose(loaded.loss_history, model.loss_history)

    def test_untrained_raises(self):
        """Test that predicting before training raises."""
        model = LogisticRegression()
        with np.testing.assert_raises(ValueError):
            model.predict_proba(np.zeros((2, 5)))
