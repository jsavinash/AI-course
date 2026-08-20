"""Serving API for Tool Use and Functional Calling."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from mlops_shared.drift import DriftDetector
from mlops_shared.fastapi_middleware import add_observability_middleware
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.metrics import MetricsCollector
from mlops_shared.model_registry import ModelRegistry
from pydantic import BaseModel, Field

from tool_use_and_functional_calling.data import (
    DEFAULT_N_SAMPLES,
    DEFAULT_VOCAB_SIZE,
    get_all_workflows,
    get_limitations,
    get_tool_by_name,
)
from tool_use_and_functional_calling.model import ToolCall, ToolResult, ToolUseModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("TOOL_USE_METRICS_PORT", "8015"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class InvokeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to process with function calling")
    workflow: str | None = Field(default=None, description="Optional predefined workflow name")


class InvokeResponse(BaseModel):
    query: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    final_response: str
    success: bool
    model_version: str


class RegisterToolRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(...)
    return_type: str = Field(default="any")
    keywords: list[str] = Field(default_factory=list)


class ExecuteToolRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(...)


class ExecuteToolResponse(BaseModel):
    tool_name: str
    output: Any
    success: bool
    error: str | None = None
    call_id: str


class StatsResponse(BaseModel):
    model_id: str
    base_model_name: str
    n_tools: int
    n_tool_calls: int
    n_tool_results: int
    n_executions: int
    model_version: str


_model: ToolUseModel | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("tool_use", port=METRICS_PORT)
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
        model_name="tool-use-and-functional-calling",
        model_version=_model_version,
        model_type="tool_use",
    )

    logger.info("Tool use model loaded", model="tool-use-and-functional-calling", version=_model_version)
    yield
    logger.info("Shutting down tool-use API")


def _load_model() -> tuple[ToolUseModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            tu_models = [m for m in models if m.get("model_name") == "tool-use-and-functional-calling"]
            if tu_models:
                tu_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = tu_models[0]
                model_dir = Path(latest["artifact_path"])
                json_files = list(model_dir.glob("tool_use_v*.json")) + list(model_dir.glob("*.json"))
                if json_files:
                    return ToolUseModel.load(str(json_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "tool-use-and-functional-calling" / MODEL_VERSION
            if model_dir.exists():
                json_files = list(model_dir.glob("tool_use_v*.json")) + list(model_dir.glob("*.json"))
                if json_files:
                    return ToolUseModel.load(str(json_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    json_path = MODEL_DIR / "tool_use.json"
    if json_path.exists():
        return ToolUseModel.load(str(json_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/tool_use_v1.0.0.json"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "tool_use_v1.0.0.json",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return ToolUseModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    model = ToolUseModel(model_id="baseline", base_model_name="tool-use-v1")
    model._init()
    return model, "1.0.0-baseline"


app = FastAPI(
    title="Tool Use and Functional Calling API",
    description="Function calling (tool use) in LLMs: architecture, workflow execution, tool management",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get("/")
def read_root():
    return {
        "service": "tool-use-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "n_tools": len(_model.tools) if _model else 0,
        "endpoints": {
            "health": "/health",
            "invoke": "POST /invoke",
            "tools_list": "GET /tools",
            "tool_detail": "GET /tools/{tool_name}",
            "register_tool": "POST /tools/register",
            "execute_tool": "POST /tools/execute",
            "workflows": "GET /workflows",
            "applications": "GET /applications",
            "advantages": "GET /advantages",
            "limitations": "GET /limitations",
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
        "n_tools": len(_model.tools) if _model else 0,
    }


@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/invoke", response_model=InvokeResponse)
def invoke_tool_use(body: InvokeRequest):
    """Run the full function-calling workflow for a user query."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        if body.workflow:
            result = _model.run_workflow(body.workflow)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Workflow not found: {body.workflow}")
        else:
            result = _model.invoke(body.query)

        response = InvokeResponse(
            query=result["query"],
            tool_call=result.get("tool_call"),
            tool_result=result.get("tool_result"),
            final_response=result["final_response"],
            success=result["success"],
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        _recent_predictions.append([float(len(body.query.split()))])
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)
        return response
    except HTTPException:
        raise
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="invoke")
        logger.exception("Tool use invoke failed", error=str(e))
        raise HTTPException(status_code=500, detail="Tool use invoke failed") from e


@app.get("/tools")
def list_tools():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"tools": _model.list_tools(), "count": len(_model.tools)}


@app.get("/tools/{tool_name}")
def get_tool_detail(tool_name: str):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tool = get_tool_by_name(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    return tool


@app.post("/tools/register")
def register_tool(body: RegisterToolRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tool = ToolUseModel(
        model_id=_model.model_id,
        description=body.description,
        parameters=body.parameters,
        return_type=body.return_type,
        keywords=body.keywords,
    )
    _model.register_tool(tool)
    return {"status": "registered", "tool": body.name}


@app.post("/tools/execute", response_model=ExecuteToolResponse)
def execute_tool(body: ExecuteToolRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tool_call = ToolCall(
        tool_name=body.tool_name,
        arguments=body.arguments,
    )
    result = _model.execute_tool(tool_call)
    return ExecuteToolResponse(
        tool_name=result.tool_name,
        output=result.output,
        success=result.success,
        error=result.error,
        call_id=result.call_id,
    )


@app.get("/workflows")
def list_workflows():
    return {"workflows": get_all_workflows(), "count": len(get_all_workflows())}


@app.get("/applications")
def list_applications():
    from tool_use_and_functional_calling.data import get_applications
    return {"applications": get_applications()}


@app.get("/advantages")
def list_advantages():
    from tool_use_and_functional_calling.data import get_advantages
    return {"advantages": get_advantages()}


@app.get("/limitations")
def list_limitations():
    return {"limitations": get_limitations()}


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        model_id=info["model_id"],
        base_model_name=info["base_model_name"],
        n_tools=info["n_tools"],
        n_tool_calls=info["n_tool_calls"],
        n_tool_results=info["n_tool_results"],
        n_executions=info["n_executions"],
        model_version=_model_version,
    )
