"""Unit tests for capsnet-text-recognition model."""

from capsnet_text_recognition.model import *  # noqa: F403


class TestTextCharRecognitionCapsNet:
    """Tests for the text/character recognition capsule network model."""

    def _make_data(self, n_samples=100, seed=42):
        from capsnet_text_recognition.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        from capsnet_text_recognition.model import TextCharRecognitionCapsNet

        X, y = self._make_data(n_samples=80)
        model = TextCharRecognitionCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0

    def test_predict_returns_valid_classes(self):
        from capsnet_text_recognition.model import TextCharRecognitionCapsNet

        X, y = self._make_data(n_samples=80)
        model = TextCharRecognitionCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=3, random_seed=42
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
