"""Unit tests for nlp-text-generation model."""

import numpy as np
from nlp_text_generation.model import TextGenerationRNN


class TestTextGenerationRNN:
    """Tests for the text generation RNN language model."""

    def _make_data(self, n_samples=50, seed=42):
        from nlp_text_generation.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=80,
            random_seed=42,
        )
        model.fit(X)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "self-supervised"

    def test_predict_returns_valid_tokens(self):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        preds = model.predict(X[0])
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset(set(range(26)))

    def test_predict_proba_shape(self):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        probas = model.predict_proba(X[0])
        assert probas.shape == (20, 26)
        assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    def test_generate_returns_sequence(self):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        generated = model.generate(X[0], n_tokens=10)
        assert len(generated) == 30
        assert set(np.unique(generated)).issubset(set(range(26)))

    def test_evaluate_returns_perplexity(self):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        metrics = model.evaluate(X)
        assert "perplexity" in metrics
        assert metrics["perplexity"] > 0

    def test_save_load_roundtrip(self, tmp_path):
        X = self._make_data(n_samples=50)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = TextGenerationRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        np.testing.assert_allclose(loaded.model.W_hy, model.model.W_hy)
        assert loaded.vocab_size == model.vocab_size

    def test_to_dict_returns_metadata(self):
        X = self._make_data(n_samples=30)
        model = TextGenerationRNN(
            vocab_size=26,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X)
        metadata = model.to_dict()
        assert "vocab_size" in metadata
        assert "training_mode" in metadata

