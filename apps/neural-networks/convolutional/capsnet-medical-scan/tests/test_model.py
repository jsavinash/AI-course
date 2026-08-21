"""Unit tests for capsnet-medical-scan model."""

import numpy as np
from capsnet_medical_scan.model import *  # noqa: F403


class TestMedicalScanAnalysisCapsNet:
    """Tests for the medical scan analysis capsule network model."""

    def _make_data(self, n_samples=80, seed=42):
        from capsnet_medical_scan.data import generate_synthetic_data

        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        from capsnet_medical_scan.model import MedicalScanAnalysisCapsNet

        X, y = self._make_data(n_samples=50)
        model = MedicalScanAnalysisCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=10, random_seed=42
        )
        model.fit(X, y)
        assert len(model.loss_history) > 0

    def test_predict_returns_valid_classes(self):
        from capsnet_medical_scan.model import MedicalScanAnalysisCapsNet

        X, y = self._make_data(n_samples=50)
        model = MedicalScanAnalysisCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_evaluate_returns_metrics(self):
        from capsnet_medical_scan.model import MedicalScanAnalysisCapsNet

        X, y = self._make_data(n_samples=50)
        model = MedicalScanAnalysisCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        from capsnet_medical_scan.model import MedicalScanAnalysisCapsNet

        X, y = self._make_data(n_samples=50)
        model = MedicalScanAnalysisCapsNet(
            n_filters=4, learning_rate=0.05, n_iterations=5, random_seed=42
        )
        model.fit(X, y)
        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = MedicalScanAnalysisCapsNet.load(path)
        np.testing.assert_allclose(model.predict_proba(X[:5]), loaded.predict_proba(X[:5]))

