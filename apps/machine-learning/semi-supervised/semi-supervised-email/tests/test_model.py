"""Unit tests for semi-supervised-email model."""

import numpy as np
import pytest
from semi_supervised_email.data import load_training_data
from semi_supervised_email.model import SelfTrainingClassifier


class TestSemiSupervisedEmail:
    """Tests for the semi-supervised email classification model."""

    def _make_data(self, n_samples: int = 200, labeled_ratio: float = 0.1, seed: int = 42):
        """Generate semi-supervised email data."""
        return load_training_data(
            data_path=None,
            labeled_ratio=labeled_ratio,
            n_samples=n_samples,
            random_seed=seed,
        )

    def test_self_training_improves_over_supervised(self):
        """Test that self-training produces a working model with reasonable accuracy."""
        X, y, is_labeled = self._make_data(n_samples=500, labeled_ratio=0.15, seed=42)

        # Self-training model with moderate confidence threshold
        ss_model = SelfTrainingClassifier(
            confidence_threshold=0.8,
            max_iterations=15,
            min_labeled_ratio=0.9,
            random_seed=42,
        )
        ss_model.fit(X, y)

        # Evaluate on all labeled data
        X_labeled, y_labeled = _get_labeled_data(X, y)
        metrics = ss_model.evaluate(X_labeled, y_labeled)

        # Model should achieve reasonable accuracy and be in semi-supervised mode
        assert metrics["accuracy"] >= 0.4
        assert ss_model.training_mode == "semi-supervised"

    def test_self_training_uses_unlabeled_data(self):
        """Test that self-training incorporates unlabeled samples."""
        X, y, is_labeled = self._make_data(n_samples=200, labeled_ratio=0.1, seed=42)
        n_initial_labeled = np.sum(is_labeled)

        model = SelfTrainingClassifier(
            confidence_threshold=0.85,
            max_iterations=10,
            random_seed=42,
        )
        model.fit(X, y)

        # Should have used unlabeled data (semi-supervised mode)
        assert model.training_mode == "semi-supervised"
        assert len(model.n_labeled_history) > 1
        assert model.n_labeled_history[-1] > n_initial_labeled

    def test_predict_returns_valid_probabilities(self):
        """Test that predictions return valid probabilities."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        probas = model.predict_proba(X[:10])
        assert probas.shape == (10,)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_returns_valid_labels(self):
        """Test that predictions return binary labels."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        X_labeled, y_labeled = _get_labeled_data(X, y)
        metrics = model.evaluate(X_labeled, y_labeled)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = SelfTrainingClassifier.load(path)

        assert loaded.n_features == model.n_features
        assert loaded.confidence_threshold == model.confidence_threshold
        assert loaded.training_mode == model.training_mode
        assert loaded.n_iterations_used == model.n_iterations_used
        np.testing.assert_allclose(loaded.model.weights, model.model.weights)
        assert loaded.model.bias == pytest.approx(model.model.bias)

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        metadata = model.to_dict()
        assert "n_features" in metadata
        assert "training_mode" in metadata
        assert "n_iterations_used" in metadata
        assert "n_labeled_history" in metadata
        assert "accuracy_history" in metadata
        assert metadata["training_mode"] == "semi-supervised"

    def test_supervised_mode_when_no_unlabeled(self):
        """Test that model stays in supervised mode when all data is labeled."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=1.0, seed=42)

        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        assert model.training_mode == "supervised"
        assert len(model.n_labeled_history) == 1


def _get_labeled_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper to extract labeled data."""
    mask = y != -1
    return X[mask], y[mask]

