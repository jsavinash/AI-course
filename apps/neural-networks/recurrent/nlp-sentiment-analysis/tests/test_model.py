"""Unit tests for nlp-sentiment-analysis model."""

import numpy as np
from nlp_sentiment_analysis.model import SentimentAnalysisRNN


class TestSentimentAnalysisRNN:
    """Tests for the sentiment analysis RNN model."""

    def _make_data(self, n_samples=100, seed=42):
        from nlp_sentiment_analysis.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, y = self._make_data(n_samples=50)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_labels(self):
        X, y = self._make_data(n_samples=50)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self):
        X, y = self._make_data(n_samples=50)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        probas = model.predict_proba(X[:10])
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=50)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_data(n_samples=50)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = SentimentAnalysisRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        np.testing.assert_allclose(loaded.model.W_hy, model.model.W_hy)
        assert loaded.vocab_size == model.vocab_size

    def test_to_dict_returns_metadata(self):
        X, y = self._make_data(n_samples=30)
        model = SentimentAnalysisRNN(
            vocab_size=50,
            seq_len=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "vocab_size" in metadata
        assert "training_mode" in metadata

