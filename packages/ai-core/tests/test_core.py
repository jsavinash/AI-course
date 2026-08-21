"""Production test suite for the shared ``ai_core`` MLOps library.

Exercises the core plumbing that every app depends on: config, validation,
model registry, drift detection, metrics, losses, optimizers and the training
loop. Raising coverage here protects the shared foundation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from unittest.mock import patch

from ai_core import config as config_mod
from ai_core.config import (
    BaseConfig,
    Config,
    MLflowConfig,
    load_config,
)
from ai_core.drift import DriftDetector, DriftResult
from ai_core.losses import bce_loss, cross_entropy_loss, mse_loss
from ai_core.metrics import MetricsCollector, PredictionMetrics
from ai_core.model_registry import (
    MODEL_STAGES,
    TRANSITIONS,
    ModelRegistry,
)
from ai_core.optim import adam_step, sgd_step
from ai_core.train_loop import Callback, TrainLoop
from ai_core.validation import (
    DataSchema,
    DataValidator,
    ValidationResult,
    create_pizza_schema,
    create_spam_schema,
)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


class TestLosses:
    def test_mse_loss_zero_for_equal(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse_loss(y, y) == 0.0

    def test_mse_loss_value(self):
        assert mse_loss(np.array([0.0, 0.0]), np.array([1.0, -1.0])) == 1.0

    def test_bce_loss_perfect(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        assert bce_loss(y_true, y_pred) < 1e-6

    def test_bce_loss_clips(self):
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        assert bce_loss(y_true, y_pred) < 1e-6

    def test_cross_entropy_perfect(self):
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        y_true = np.array([0, 1])
        assert cross_entropy_loss(logits, y_true) < 1e-3

    def test_cross_entropy_higher_for_wrong(self):
        correct = cross_entropy_loss(np.array([[10.0, 0.0]]), np.array([0]))
        wrong = cross_entropy_loss(np.array([[0.0, 10.0]]), np.array([0]))
        assert correct < wrong


# ---------------------------------------------------------------------------
# optim
# ---------------------------------------------------------------------------


class TestOptim:
    def test_sgd_step(self):
        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.5, 0.5])}
        out = sgd_step(params, grads, lr=0.1)
        np.testing.assert_allclose(out["w"], np.array([0.95, 1.95]))

    def test_adam_step_updates_state(self):
        params = {"w": np.array([1.0])}
        grads = {"w": np.array([0.1])}
        state: dict = {}
        new_params, new_state = adam_step(params, grads, state, lr=0.1)
        assert "_t" in new_state
        assert "w" in new_state
        assert new_params["w"][0] < 1.0

    def test_adam_converges(self):
        params = {"w": np.array([5.0])}
        state: dict = {}
        for _ in range(200):
            params, state = adam_step(params, {"w": np.array([params["w"][0]])}, state, lr=0.1)
        assert abs(params["w"][0]) < 0.1


# ---------------------------------------------------------------------------
# train_loop
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def test_runs_epochs_and_steps(self):
        calls = {"n": 0}

        def step(epoch, s):
            calls["n"] += 1
            return {"loss": 0.1 * calls["n"]}

        history = TrainLoop(epochs=2, steps_per_epoch=3, train_step=step).run()
        assert calls["n"] == 6
        assert len(history) == 2
        assert len(history[0]) == 3

    def test_callbacks_invoked(self):
        starts = []
        ends = []

        cb = Callback(
            on_epoch_start=lambda e: starts.append(e),
            on_epoch_end=lambda e, m: ends.append((e, len(m))),
        )
        TrainLoop(
            epochs=2, steps_per_epoch=1, train_step=lambda e, s: {}, callbacks=[cb]
        ).run()
        assert starts == [0, 1]
        assert ends == [(0, 1), (1, 1)]

    def test_silent_mode_no_output(self, capsys):
        TrainLoop(
            epochs=1, steps_per_epoch=1, train_step=lambda e, s: {}, verbose=False
        ).run()
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# logging / redaction
# ---------------------------------------------------------------------------


class TestLogging:
    def test_request_context_roundtrip(self):
        from ai_core.logging import (
            clear_request_context,
            generate_request_id,
            set_request_context,
        )

        rid = generate_request_id()
        assert len(rid) == 16
        set_request_context(request_id=rid, trace_id="t1")
        clear_request_context()

    def test_setup_logging_with_file(self, tmp_path):
        from ai_core.logging import get_logger, setup_logging

        log_file = tmp_path / "app.log"
        setup_logging(level="INFO", json_output=True, log_file=log_file)
        get_logger("test").info("hello", foo="bar")
        assert log_file.exists()

    def test_redaction_of_sensitive_keys(self):
        from ai_core.logging import _redact_event_dict

        event = {"password": "secret", "nested": {"api_key": "x"}, "ok": 1}
        out = _redact_event_dict(None, None, event)
        assert out["password"] == "***REDACTED***"
        assert out["nested"]["api_key"] == "***REDACTED***"
        assert out["ok"] == 1


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_from_dict_and_redaction(self):
        cfg = MLflowConfig(tracking_uri="http://mlflow:5000", s3_secret_key="topsecret")
        d = cfg.to_dict()
        assert d["s3_secret_key"] == "***REDACTED***"
        assert cfg.s3_secret_key == "topsecret"

    def test_coerce_value_types(self):
        assert config_mod._coerce_value("true", bool) is True
        assert config_mod._coerce_value("42", int) == 42
        assert config_mod._coerce_value("3.14", float) == 3.14
        assert config_mod._coerce_value("a,b,c", list[str]) == ["a", "b", "c"]
        assert config_mod._coerce_value("/tmp/x", __import__("pathlib").Path) == __import__("pathlib").Path("/tmp/x")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MYAPP_MODEL_NAME", "envmodel")
        monkeypatch.setenv("MYAPP_ENVIRONMENT", "production")
        cfg = Config.from_env(prefix="MYAPP")
        assert cfg.model_name == "envmodel"
        assert cfg.environment == "production"

    def test_load_from_yaml(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "model_name: fromyaml\nenvironment: staging\n"
            "mlflow:\n  experiment_name: xyz\n"
        )
        cfg = load_config(path=path)
        assert cfg.model_name == "fromyaml"
        assert cfg.mlflow.experiment_name == "xyz"

    def test_load_env_overrides_yaml(self, tmp_path, monkeypatch):
        path = tmp_path / "cfg.yaml"
        path.write_text("model_name: fromyaml\n")
        monkeypatch.setenv("MLOPS_MODEL_NAME", "overenv")
        cfg = load_config(path=path)
        assert cfg.model_name == "overenv"

    def test_validate_required(self):
        @dataclass
        class RequiredConfig(BaseConfig):
            name: str

        errs = RequiredConfig(name="x").validate()
        assert errs == []


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_matrix_passes(self):
        validator = DataValidator(create_pizza_schema())
        res = validator.validate(np.array([[10.0], [20.0], [30.0]]))
        assert res.valid
        assert res.stats["n_samples"] == 3

    def test_wrong_feature_count_fails(self):
        validator = DataValidator(create_pizza_schema())
        res = validator.validate(np.array([[1.0, 2.0]]))
        assert not res.valid
        assert any("features" in e for e in res.errors)

    def test_nan_fails(self):
        validator = DataValidator(create_pizza_schema())
        res = validator.validate(np.array([[np.nan]]))
        assert not res.valid

    def test_binary_constraint(self):
        validator = DataValidator(create_spam_schema())
        assert validator.validate(np.array([[1, 0, 1, 0, 1]])).valid
        assert not validator.validate(np.array([[2, 0, 1, 0, 1]])).valid

    def test_value_range(self):
        validator = DataValidator(create_pizza_schema())
        assert not validator.validate(np.array([[100.0]])).valid

    def test_min_rows(self):
        schema = DataSchema(feature_names=["x"], min_rows=5)
        res = DataValidator(schema).validate(np.array([[1.0]]))
        assert not res.valid

    def test_label_length_mismatch(self):
        validator = DataValidator(create_pizza_schema())
        res = validator.validate(np.array([[10.0]]), y=np.array([0, 1]))
        assert not res.valid

    def test_validate_dataframe(self):
        import pandas as pd

        validator = DataValidator(create_pizza_schema())
        df = pd.DataFrame({"diameter": [10.0, 20.0, 30.0]})
        assert validator.validate_dataframe(df).valid

    def test_missing_columns_dataframe(self):
        import pandas as pd

        validator = DataValidator(create_pizza_schema())
        df = pd.DataFrame({"other": [1.0]})
        assert not validator.validate_dataframe(df).valid

    def test_validation_result_to_dict(self):
        r = ValidationResult(valid=False, errors=["e1"])
        d = r.to_dict()
        assert d["valid"] is False
        assert d["errors"] == ["e1"]


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


def _make_artifact(tmp_path, name="model.npz"):
    p = tmp_path / name
    np.savez(p, w=np.array([1.0, 2.0]))
    return p


class TestModelRegistry:
    def test_stages_and_transitions_consistent(self):
        for stage, targets in TRANSITIONS.items():
            assert stage in MODEL_STAGES
            for t in targets:
                assert t in MODEL_STAGES

    def test_save_and_list(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        info = reg.save_model(
            model_name="m", model_version="1.0.0", model_type="classification",
            metrics={"acc": 0.9}, parameters={"lr": 0.1},
            artifacts={"model.npz": art}, status="staging",
        )
        assert info.model_name == "m"
        assert len(reg.list_models()) == 1

    def test_get_and_load(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        reg.save_model("m", "1.0.0", "regression", {"mse": 0.1}, {"lr": 0.1}, {"model.npz": art})
        assert reg.get_model("m", "1.0.0")["model_version"] == "1.0.0"
        loaded = reg.load_model("m", "1.0.0")
        assert "w" in loaded["weights"]

    def test_invalid_status_rejected(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        with pytest.raises(ValueError):
            reg.save_model("m", "1.0.0", "regression", {}, {}, {"model.npz": art},
                           status="not_a_stage")

    def test_transition_valid_and_invalid(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        reg.save_model("m", "1.0.0", "regression", {}, {}, {"model.npz": art}, status="staging")
        reg.transition("m", "1.0.0", "production")
        assert reg.get_model("m", "1.0.0")["status"] == "production"
        with pytest.raises(ValueError):
            reg.transition("m", "1.0.0", "candidate")

    def test_promote_to_production(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        reg.save_model("m", "1.0.0", "regression", {}, {}, {"model.npz": art}, status="staging")
        reg.save_model("m", "2.0.0", "regression", {}, {}, {"model.npz": art}, status="staging")
        reg.promote_to_production("m", "2.0.0")
        assert reg.get_latest("m", status="production")["model_version"] == "2.0.0"
        assert reg.get_model("m", "1.0.0")["status"] == "archived"

    def test_enforce_max_versions(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        art = _make_artifact(tmp_path)
        for i in range(7):
            reg.save_model("m", f"{i}.0.0", "regression", {}, {}, {"model.npz": art}, status="staging")
        versions = reg.list_versions("m")
        assert len(versions) == 7
        assert any(v["status"] == "archived" for v in versions)

    def test_get_latest_raises_when_empty(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.get_latest("missing")

    def test_log_to_mlflow_unavailable(self, tmp_path):
        reg = ModelRegistry(base_dir=tmp_path)
        with patch.dict("sys.modules", {"mlflow": None}):
            result = reg.log_to_mlflow("m", "1.0.0", {"acc": 1.0}, {"lr": 0.1}, {})
        assert result is None


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


class TestDrift:
    def test_no_drift_for_identical(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, size=(200, 3))
        det = DriftDetector(feature_names=["a", "b", "c"], psi_threshold=0.2)
        results = det.detect_drift(ref, ref.copy())
        summary = det.summarize(results)
        assert summary["total_features"] == 3
        assert summary["drifted_features"] < 3

    def test_drift_detected_on_shift(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, size=(200, 2))
        cur = rng.normal(5, 1, size=(200, 2))
        det = DriftDetector(feature_names=["a", "b"])
        results = det.detect_drift(ref, cur)
        assert det.summarize(results)["drifted_features"] >= 1

    def test_categorical_drift(self):
        ref = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        cur = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        det = DriftDetector(
            feature_names=["f1", "f2"],
            feature_types={"f1": "binary", "f2": "binary"},
        )
        results = det.detect_drift(ref, cur)
        assert all(isinstance(r, DriftResult) for r in results)
        assert det.summarize(results)["drift_ratio"] >= 0.0

    def test_drift_result_to_dict(self):
        r = DriftResult(
            feature_name="x", drift_score=0.5, p_value=0.01, is_drift=True,
            threshold=0.2, reference_mean=0.0, current_mean=1.0,
            reference_std=1.0, current_std=1.0,
        )
        d = r.to_dict()
        assert d["feature_name"] == "x"
        assert d["is_drift"] is True


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_singleton_per_service(self):
        a = MetricsCollector("svc_singleton")
        b = MetricsCollector("svc_singleton")
        assert a is b
        assert MetricsCollector("svc_other") is not a

    def test_record_prediction_and_error(self):
        m = MetricsCollector("svc_pred")
        m.record_prediction("1.0.0", 0.01)
        m.record_error("1.0.0", "boom")
        assert m.prediction_counter.labels(model_version="1.0.0")._value.get() == 1

    def test_record_request(self):
        m = MetricsCollector("svc_req")
        m.record_request("POST", "/predict", 200, 0.02)
        assert m.request_counter.labels(method="POST", endpoint="/predict", status="200")._value.get() == 1

    def test_set_model_version(self):
        m = MetricsCollector("svc_ver")
        m.set_model_version("1.2.3")
        assert m.model_version_gauge._value.get() == 1.2
        m.set_model_version("not-a-version")
        assert m.model_version_gauge._value.get() == 0.0

    def test_set_drift_ratio(self):
        m = MetricsCollector("svc_drift")
        m.set_drift_ratio(0.3)
        assert m.feature_drift_gauge._value.get() == 0.3

    def test_active_requests(self):
        m = MetricsCollector("svc_active")
        m.inc_active_requests()
        m.inc_active_requests()
        m.dec_active_requests()
        assert m.active_requests._value.get() == 1

    def test_prediction_metrics_container(self):
        pm = PredictionMetrics("m", "1.0.0")
        pm.add_prediction(0.01)
        pm.add_prediction(0.03)
        pm.add_error()
        assert pm.predictions == 2
        assert pm.errors == 1
        assert pm.error_rate == pytest.approx(100 * 1 / 3)
        assert pm.avg_latency == pytest.approx(0.02)
        assert "p95_latency_ms" in pm.to_dict()
