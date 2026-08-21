"""Serving API for Prompt Engineering."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from prompt_engineering.data import DEFAULT_VOCAB_SIZE
from prompt_engineering.model import PromptEngineeringModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("PROMPT_ENGINEERING_METRICS_PORT", "9022"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class GenerateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    technique: str = Field(default="zero-shot")
    input_text: str = Field(..., min_length=1)
    context: str | None = Field(default=None)
    examples: list[dict[str, str]] | None = Field(default=None)


class GenerateResponse(BaseModel):
    prompt: str
    template_id: str
    technique: str
    model_version: str


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    expected: str | None = Field(default=None)


class EvaluateResponse(BaseModel):
    scores: dict[str, float]
    average_score: float
    model_version: str


class OptimizeRequest(BaseModel):
    base_prompt: str = Field(..., min_length=1)
    responses: list[dict[str, str | None]] = Field(..., min_length=1)


class OptimizeResponse(BaseModel):
    optimized_prompt: str
    best_score: float
    optimization_history: list[dict[str, Any]]


class StatsResponse(BaseModel):
    model_id: str
    base_model_name: str
    n_templates: int
    n_techniques: int
    current_technique: str
    n_history_entries: int


OptimizeResponse.model_rebuild()

_model: PromptEngineeringModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("prompt_engineering", port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f"token_{i}" for i in range(DEFAULT_VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="prompt-engineering",
        model_version=_model_version,
        model_type="nlp",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="prompt-engineering", version=_model_version)

    yield
    logger.info("Shutting down prompt-engineering API")


def _load_model() -> tuple[PromptEngineeringModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            pe_models = [m for m in models if m.get("model_name") == "prompt-engineering"]
            if pe_models:
                pe_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = pe_models[0]
                model_dir = Path(latest["artifact_path"])
                json_files = list(model_dir.glob("prompt_engineering_v*.json")) + list(model_dir.glob("*.json"))
                if json_files:
                    return PromptEngineeringModel.load(str(json_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "prompt-engineering" / MODEL_VERSION
            if model_dir.exists():
                json_files = list(model_dir.glob("prompt_engineering_v*.json")) + list(model_dir.glob("*.json"))
                if json_files:
                    return PromptEngineeringModel.load(str(json_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    json_path = MODEL_DIR / "prompt_engineering.json"
    if json_path.exists():
        return PromptEngineeringModel.load(str(json_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/prompt_engineering_v1.0.0.json"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "prompt_engineering_v1.0.0.json",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return PromptEngineeringModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    model = PromptEngineeringModel(model_id="baseline", base_model_name="default")
    model._init()
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    from prompt_engineering.data import generate_synthetic_prompts
    X_base, _ = generate_synthetic_prompts(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title="Prompt Engineering API",
    description="Prompt Engineering service with various techniques (zero-shot, few-shot, chain-of-thought, self-ask, least-to-most, meta-prompting, context-amplification, iterative)",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "prompt_engineering-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "available_techniques": _model.get_available_techniques() if _model else [],
        "endpoints": {
            "health": "/health",
            "generate": "POST /generate",
            "evaluate": "POST /evaluate",
            "optimize": "POST /optimize",
            "stats": "GET /stats",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
        "model_id": _model.model_id if _model else "unknown",
    }


@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
def generate_prompt(body: GenerateRequest):
    """Generate a prompt using the specified template and technique."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        prompt = _model.generate_prompt(
            body.template_id,
            technique=body.technique,
            input_text=body.input_text,
            context=body.context,
            examples=body.examples,
        )

        response = GenerateResponse(
            prompt=prompt,
            template_id=body.template_id,
            technique=body.technique,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(len(body.input_text.split()))])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="generation")
        logger.exception("Prompt generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prompt generation failed") from e


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_prompt(body: EvaluateRequest):
    """Evaluate a prompt response against expected output."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        scores = _model.evaluate_prompt(body.prompt, body.response, body.expected)
        avg_score = float(np.mean(list(scores.values()))) if scores else 0.0

        response = EvaluateResponse(
            scores=scores,
            average_score=round(avg_score, 4),
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="evaluation")
        logger.exception("Prompt evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prompt evaluation failed") from e


@app.post("/optimize", response_model=OptimizeResponse)
def optimize_prompt(body: OptimizeRequest):
    """Optimize a prompt based on multiple response-evaluation pairs."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        responses = [(r["response"], r.get("expected")) for r in body.responses]
        optimized = _model.optimize_prompt(body.base_prompt, responses)
        best_score = _model.optimizer.get_best_score() if _model.optimizer else 0.0
        history = _model.optimizer.get_optimization_history() if _model.optimizer else []

        response = OptimizeResponse(
            optimized_prompt=optimized,
            best_score=round(best_score, 4),
            optimization_history=history,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="optimization")
        logger.exception("Prompt optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Prompt optimization failed") from e


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        model_id=info["model_id"],
        base_model_name=info["base_model_name"],
        n_templates=info["n_templates"],
        n_techniques=info["n_techniques"],
        current_technique=info["current_technique"],
        n_history_entries=info["n_history_entries"],
        model_version=_model_version,
    )
