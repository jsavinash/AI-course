"""Unit tests for pattern-recognition-digits model."""

import numpy as np
from pattern_recognition_digits.data import generate_synthetic_data
from pattern_recognition_digits.model import DigitRecognitionNN


class TestDigitRecognitionNN:
    """Tests for the handwritten digit recognition feedforward neural network."""

    def _make_data(self, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        return generate_synthetic_data(n_samples=n_samples, noise_level=0.3, random_seed=seed)

    def test_training_converges(self):
        """Test that training loss decreases."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_classes(self):
        """Test that predictions are valid digit classes (0-9)."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset(set(range(10)))

    def test_predict_proba_shape(self):
        """Test that probability outputs have correct shape and sum to 1."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 10)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert "macro_f1" in metrics
        assert "per_class_precision" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(metrics["per_class_precision"]) == 10

    def test_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        cm = model.confusion_matrix(X, y)
        assert cm.shape == (10, 10)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y = self._make_data(n_samples=200)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=200, random_seed=42
        )
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = DigitRecognitionNN.load(path)

        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
        assert loaded.input_dim == model.input_dim
        assert loaded.n_classes == model.n_classes

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y = self._make_data(n_samples=100)
        model = DigitRecognitionNN(
            hidden_dim=32, learning_rate=0.1, n_iterations=100, random_seed=42
        )
        model.fit(X, y)

        metadata = model.to_dict()
        assert "input_dim" in metadata
        assert "hidden_dim" in metadata
        assert "n_classes" in metadata
        assert "training_mode" in metadata
        assert metadata["training_mode"] == "supervised"

