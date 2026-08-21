"""Unit tests for vision-image-captioning model."""

import numpy as np
from vision_image_captioning.model import ImageCaptioningRNN


class TestImageCaptioningRNN:
    """Tests for the image captioning RNN model."""

    def _make_data(self, n_samples=50, seed=42):
        from vision_image_captioning.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        X, y = self._make_data(n_samples=50)
        model = ImageCaptioningRNN(
            n_pixels=64,
            vocab_size=20,
            caption_len=8,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=50,
            random_seed=42,
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_caption(self):
        X, y = self._make_data(n_samples=50)
        model = ImageCaptioningRNN(
            n_pixels=64,
            vocab_size=20,
            caption_len=8,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        captions = model.predict(X[:5])
        assert len(captions) == 5
        assert len(captions[0]) == 8

    def test_evaluate_returns_metrics(self):
        X, y = self._make_data(n_samples=50)
        model = ImageCaptioningRNN(
            n_pixels=64,
            vocab_size=20,
            caption_len=8,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "n_samples" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_data(n_samples=50)
        model = ImageCaptioningRNN(
            n_pixels=64,
            vocab_size=20,
            caption_len=8,
            hidden_dim=16,
            learning_rate=0.05,
            n_iterations=30,
            random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = ImageCaptioningRNN.load(path)
        np.testing.assert_allclose(loaded.model.W_xh, model.model.W_xh)
        assert loaded.vocab_size == model.vocab_size


# ============================================================
# CNN / DN / CapsNet Tests
# ============================================================

