"""Unit tests for capsnet-autonomous-driving model."""

import numpy as np
from capsnet_autonomous_driving.model import *  # noqa: F403


class TestAutonomousDrivingCapsNet:
    """Tests for the autonomous driving capsule network model."""

    def _make_data(self, n_samples=80, seed=42):
        from capsnet_autonomous_driving.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        from capsnet_autonomous_driving.model import AutonomousDrivingCapsNet

        X, y = self._make_data(n_samples=50)
        model = AutonomousDrivingCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=10, random_seed=42
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0
        assert model.training_mode == "supervised"

    def test_predict_returns_valid_classes(self):
        from capsnet_autonomous_driving.model import AutonomousDrivingCapsNet

        X, y = self._make_data(n_samples=50)
        model = AutonomousDrivingCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_evaluate_returns_metrics(self):
        from capsnet_autonomous_driving.model import AutonomousDrivingCapsNet

        X, y = self._make_data(n_samples=50)
        model = AutonomousDrivingCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        from capsnet_autonomous_driving.model import AutonomousDrivingCapsNet

        X, y = self._make_data(n_samples=50)
        model = AutonomousDrivingCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = AutonomousDrivingCapsNet.load(path)
        np.testing.assert_allclose(model.predict_proba(X[:5]), loaded.predict_proba(X[:5]))

    def test_to_dict_returns_metadata(self):
        from capsnet_autonomous_driving.model import AutonomousDrivingCapsNet

        X, y = self._make_data(n_samples=50)
        model = AutonomousDrivingCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metadata = model.to_dict()
        assert "n_classes" in metadata
        assert "training_mode" in metadata

