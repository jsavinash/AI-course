"""Unit tests for spam-classification model."""

import numpy as np
from spam_classification.model import SpamDetectionNN


class TestSpamDetectionNN:
    """Tests for the email spam detection feedforward neural network."""

    def _make_data(self, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        from classification_email_spam.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        """Test that training loss decreases."""
        X, y = self._make_data(n_samples=200)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=500, random_seed=42)
        model.fit(X, y)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_proba_range(self):
        """Test that probabilities are in [0, 1]."""
        X, y = self._make_data(n_samples=200)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_returns_valid_labels(self):
        """Test that predictions are binary (0 or 1)."""
        X, y = self._make_data(n_samples=200)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics."""
        X, y = self._make_data(n_samples=200)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y = self._make_data(n_samples=200)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=200, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = SpamDetectionNN.load(path)

        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
        assert loaded.input_dim == model.input_dim
        assert loaded.hidden_dim == model.hidden_dim

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y = self._make_data(n_samples=100)
        model = SpamDetectionNN(hidden_dim=8, learning_rate=0.01, n_iterations=100, random_seed=42)
        model.fit(X, y)

        metadata = model.to_dict()
        assert "input_dim" in metadata
        assert "hidden_dim" in metadata
        assert "training_mode" in metadata
        assert "n_epochs_run" in metadata
        assert metadata["training_mode"] == "supervised"

