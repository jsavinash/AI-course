"""Unit tests for speech-audio-recognition model."""

import numpy as np
from speech_audio_recognition.model import SpeechRecognitionRNN


class TestSpeechRecognitionRNN:
    """Tests for the speech recognition RNN model."""

    def _make_data(self, n_samples=50, seed=42):
        from speech_audio_recognition.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, y = self._make_data(n_samples=50)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_classes(self):
        X, y = self._make_data(n_samples=50)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset(set(range(10)))

    def test_predict_proba_shape(self):
        X, y = self._make_data(n_samples=50)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        probas = model.predict_proba(X[:10])
        assert probas.shape == (10, 10)
        assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=50)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
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

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_data(n_samples=50)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = SpeechRecognitionRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        assert loaded.n_classes == model.n_classes

    def test_to_dict_returns_metadata(self):
        X, y = self._make_data(n_samples=30)
        model = SpeechRecognitionRNN(
            n_features=16,
            seq_len=20,
            n_classes=10,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "n_features" in metadata
        assert "n_classes" in metadata
        assert "training_mode" in metadata

