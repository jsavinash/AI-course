"""Unit tests for speech-audio-music model."""

import numpy as np
from speech_audio_music.model import MusicGenerationRNN


class TestMusicGenerationRNN:
    """Tests for the music generation RNN language model."""

    def _make_data(self, n_samples=50, seed=42):
        from speech_audio_music.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X = self._make_data(n_samples=50)
        model = MusicGenerationRNN(
            vocab_size=40,
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
        model = MusicGenerationRNN(
            vocab_size=40,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        preds = model.predict(X[0])
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset(set(range(40)))

    def test_generate_returns_sequence(self):
        X = self._make_data(n_samples=50)
        model = MusicGenerationRNN(
            vocab_size=40,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        generated = model.generate(X[0], n_tokens=10)
        assert len(generated) == 30
        assert set(np.unique(generated)).issubset(set(range(40)))

    def test_evaluate_returns_perplexity(self):
        X = self._make_data(n_samples=50)
        model = MusicGenerationRNN(
            vocab_size=40,
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
        model = MusicGenerationRNN(
            vocab_size=40,
            seq_len=20,
            hidden_dim=16,
            learning_rate=0.1,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = MusicGenerationRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        assert loaded.vocab_size == model.vocab_size

