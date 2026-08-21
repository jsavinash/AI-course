"""Unit tests for nlp-language-translation model."""

import numpy as np
from nlp_language_translation.model import LanguageTranslationRNN


class TestLanguageTranslationRNN:
    """Tests for the language translation RNN model."""

    def _make_data(self, n_samples=200, seed=42):
        from nlp_language_translation.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, y = self._make_data(n_samples=50)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=80,
            random_seed=42,
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_token(self):
        X, y = self._make_data(n_samples=50)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        pred = model.predict(X[0])
        assert isinstance(pred, int)
        assert 0 <= pred < 40

    def test_predict_proba_shape(self):
        X, y = self._make_data(n_samples=50)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        probas = model.predict_proba(X[:5])
        assert probas.shape == (5, 40)
        assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=50)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_data(n_samples=50)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = LanguageTranslationRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        np.testing.assert_allclose(loaded.model.W_hy, model.model.W_hy)
        assert loaded.vocab_size == model.vocab_size

    def test_to_dict_returns_metadata(self):
        X, y = self._make_data(n_samples=30)
        model = LanguageTranslationRNN(
            vocab_size=40,
            seq_len=8,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "vocab_size" in metadata
        assert "hidden_dim" in metadata
        assert "training_mode" in metadata

