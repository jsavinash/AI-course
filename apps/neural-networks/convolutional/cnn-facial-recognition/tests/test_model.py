"""Unit tests for cnn-facial-recognition model."""

import numpy as np
from cnn_facial_recognition.model import *  # noqa: F403


class TestFacialRecognitionCNN:
    """Tests for the facial recognition CNN model."""

    def _make_data(self, n_samples=60, seed=42):
        from cnn_facial_recognition.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        from cnn_facial_recognition.model import FacialRecognitionCNN

        X, y = self._make_data(n_samples=50)
        model = FacialRecognitionCNN(
            n_filters=4, hidden_dim=16, learning_rate=0.01, n_iterations=10, random_seed=42
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.training_mode == "supervised"

    def test_predict_returns_binary(self):
        from cnn_facial_recognition.model import FacialRecognitionCNN

        X, y = self._make_data(n_samples=50)
        model = FacialRecognitionCNN(
            n_filters=4, hidden_dim=16, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        preds = model.predict_class(X[:10])
        assert preds.shape == (10,)
        assert all(p in (0, 1) for p in preds)

    def test_evaluate_returns_metrics(self):
        from cnn_facial_recognition.model import FacialRecognitionCNN

        X, y = self._make_data(n_samples=50)
        model = FacialRecognitionCNN(
            n_filters=4, hidden_dim=16, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        from cnn_facial_recognition.model import FacialRecognitionCNN

        X, y = self._make_data(n_samples=50)
        model = FacialRecognitionCNN(
            n_filters=4, hidden_dim=16, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = FacialRecognitionCNN.load(path)
        np.testing.assert_allclose(model.predict_proba(X[:5]), loaded.predict_proba(X[:5]))

    def test_to_dict_returns_metadata(self):
        from cnn_facial_recognition.model import FacialRecognitionCNN

        X, y = self._make_data(n_samples=50)
        model = FacialRecognitionCNN(
            n_filters=4, hidden_dim=16, learning_rate=0.01, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "img_size" in metadata
        assert "training_mode" in metadata

