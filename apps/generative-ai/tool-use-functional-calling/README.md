<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tool-use-functional-calling - AI App Documentation</title>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMath()"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
/* CSS styles here */
</style>
</head>
<body>
<section id="math" class="section math-section">
<h2><span class="section-icon">∫</span> Mathematics &amp; Theory</h2>
<p class="section-subtitle">Tool Use and Functional Calling — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(\text{tool}, \text{args} | q) = \text{softmax}(W_t \cdot h_q)$$</div>
<div class="math-block">$$\text{result} = \text{execute}(\text{tool}, \text{args})$$</div>
<div class="math-block">$$\text{final} = \text{generate}(q, \text{result})$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Tool-augmented models decompose complex queries into executable function calls. A router network predicts which tool to invoke and with what arguments. The tool result is fed back into the language model for final response generation. This enables structured reasoning and access to external APIs.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive tool call graph; argument parsing explorer; multi-step reasoning trace.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  ToolSpec
  ToolCall
  ToolResult
  ToolUseModel</pre>
</div>
<div class="mermaid-wrapper">
<h3>Data Flow</h3>
<pre class="mermaid">graph TD
  A[Input Data] --> B[Preprocessing]
  B --> C[Model Training]
  C --> D[Evaluation]
  D --> E[Model Registry]
  E --> F[Serving API]</pre>
</div>
</section>
<section id="api" class="section api-section">
<h2><span class="section-icon">⚡</span> API Reference</h2>
<p class="section-subtitle">FastAPI endpoints and model interfaces</p>
<table class="api-table">
<thead><tr><th>Method</th><th>Endpoint</th></tr></thead>
<tbody><tr><td><code>GET</code></td><td><code>/</code></td></tr>
<tr><td><code>GET</code></td><td><code>/health</code></td></tr>
<tr><td><code>GET</code></td><td><code>/metrics</code></td></tr>
<tr><td><code>GET</code></td><td><code>/tools</code></td></tr>
<tr><td><code>GET</code></td><td><code>/tools/{tool_name}</code></td></tr>
<tr><td><code>POST</code></td><td><code>/tools/register</code></td></tr>
<tr><td><code>GET</code></td><td><code>/workflows</code></td></tr>
<tr><td><code>GET</code></td><td><code>/applications</code></td></tr>
<tr><td><code>GET</code></td><td><code>/advantages</code></td></tr>
<tr><td><code>GET</code></td><td><code>/limitations</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3411118387')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3411118387"><code class="language-python">&quot;&quot;&quot;Training pipeline for Tool Use and Functional Calling.

Demonstrates the complete 5-step function-calling workflow:
1. Register tools (tool definitions)
2. LLM tool decision (select tool + extract arguments)
3. Application-side execution
4. Result concatenation
5. Final generation
&quot;&quot;&quot;

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from tool_use_and_functional_calling.data import (
    DEFAULT_N_SAMPLES,
    DEFAULT_VOCAB_SIZE,
    load_tool_dataset,
    save_dataset,
    train_test_split,
)
from tool_use_and_functional_calling.model import ToolUseModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    n_tools: int = 5,
    model_id: str = &quot;tool-use-and-functional-calling-v1&quot;,
    base_model_name: str = &quot;tool-use-v1&quot;,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    logger.info(&quot;Loading tool-call dataset&quot;, n_samples=n_samples, n_tools=n_tools)
    X, y = load_tool_dataset(
        n_samples=n_samples,
        vocab_size=vocab_size,
        random_seed=random_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / &quot;tool_use_training_data.npz&quot;)

    model = ToolUseModel(model_id=model_id, base_model_name=base_model_name)
    model._init()

    for tool_name in list(model.tools.keys())[:n_tools]:
        spec = model.get_tool(tool_name)
        if spec:
            model.register_tool(spec)

    logger.info(&quot;Registered tools&quot;, n_tools=len(model.tools), tools=list(model.tools.keys()))

    test_queries = [
        &quot;What is the current TCS stock price?&quot;,
        &quot;What is the weather in Mumbai?&quot;,
        &quot;Calculate 25 + 37&quot;,
        &quot;What is the status of order ORD-12345?&quot;,
        &quot;Show me all users who signed up in the last 7 days&quot;,
        &quot;What is the weather in London?&quot;,
        &quot;Calculate 100 divided by 4&quot;,
        &quot;Track my order ORD-54321&quot;,
    ]

    logger.info(&quot;Running function-calling workflow on test queries&quot;)
    workflow_results = []
    for query in test_queries:
        result = model.invoke(query)
        workflow_results.append(result)
        logger.info(
            &quot;Workflow result&quot;,
            query=query[:50],
            tool_called=result.get(&quot;tool_call&quot;, {}).get(&quot;tool_name&quot;) if result.get(&quot;tool_call&quot;) else None,
            success=result.get(&quot;success&quot;),
            response=result.get(&quot;final_response&quot;, &quot;&quot;)[:60],
        )

    n_successful = sum(1 for r in workflow_results if r.get(&quot;success&quot;))
    n_with_tool_call = sum(1 for r in workflow_results if r.get(&quot;tool_call&quot;) is not None)

    logger.info(
        &quot;Workflow summary&quot;,
        total=len(workflow_results),
        successful=n_successful,
        with_tool_call=n_with_tool_call,
    )

    sample_queries = X_test[:5]
    predicted_tools = []
    for query_tokens in sample_queries:
        query_text = &quot; &quot;.join([str(int(t)) for t in query_tokens if int(t) &gt; 0])
        tool_call = model.decide_tool(query_text)
        predicted_tools.append(tool_call.tool_name if tool_call else &quot;none&quot;)

    model_path = model_dir / f&quot;tool_use_v{model_version}.json&quot;
    model.save(str(model_path))

    metrics = {
        &quot;n_samples&quot;: float(len(X)),
        &quot;n_train&quot;: float(len(X_train)),
        &quot;n_test&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(vocab_size),
        &quot;n_tools&quot;: float(len(model.tools)),
        &quot;n_workflow_queries&quot;: float(len(workflow_results)),
        &quot;n_successful_workflows&quot;: float(n_successful),
        &quot;n_tool_calls&quot;: float(n_with_tool_call),
        &quot;accuracy&quot;: float(n_with_tool_call) / max(len(workflow_results), 1),
        &quot;success_rate&quot;: float(n_successful) / max(len(workflow_results), 1),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;tool-use-and-functional-calling&quot;,
        model_version=model_version,
        model_type=&quot;tool_use&quot;,
        metrics=metrics,
        parameters={
            &quot;model_id&quot;: model_id,
            &quot;base_model_name&quot;: base_model_name,
            &quot;n_tools&quot;: n_tools,
            &quot;n_samples&quot;: n_samples,
            &quot;vocab_size&quot;: vocab_size,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;tool_use_v{model_version}.json&quot;: model_path,
            &quot;tool_use_training_data.npz&quot;: model_dir / &quot;tool_use_training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;tool_use_and_functional_calling&quot;, &quot;model_type&quot;: &quot;ToolUseModel&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;tool-use-and-functional-calling&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;model_id&quot;: model_id, &quot;n_tools&quot;: n_tools, &quot;n_samples&quot;: n_samples},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;tool_use&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Tool Use and Functional Calling Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, str(DEFAULT_N_SAMPLES))))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, str(DEFAULT_VOCAB_SIZE))))
    parser.add_argument(&quot;--n-tools&quot;, type=int, default=int(os.getenv(&quot;N_TOOLS&quot;, &quot;5&quot;)))
    parser.add_argument(&quot;--model-id&quot;, type=str, default=os.getenv(&quot;MODEL_ID&quot;, &quot;tool-use-and-functional-calling-v1&quot;))
    parser.add_argument(&quot;--base-model-name&quot;, type=str, default=os.getenv(&quot;BASE_MODEL_NAME&quot;, &quot;tool-use-v1&quot;))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        n_tools=args.n_tools,
        model_id=args.model_id,
        base_model_name=args.base_model_name,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3294175218')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3294175218"><code class="language-python">&quot;&quot;&quot;Serving API for Tool Use and Functional Calling.&quot;&quot;&quot;

from __future__ import annotations

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

from tool_use_and_functional_calling.data import (
    DEFAULT_VOCAB_SIZE,
    get_all_workflows,
    get_limitations,
    get_tool_by_name,
)
from tool_use_and_functional_calling.model import ToolCall, ToolSpec, ToolUseModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;TOOL_USE_METRICS_PORT&quot;, &quot;9025&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class InvokeRequest(BaseModel):
    query: str = Field(..., min_length=1, description=&quot;User query to process with function calling&quot;)
    workflow: str | None = Field(default=None, description=&quot;Optional predefined workflow name&quot;)


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
    return_type: str = Field(default=&quot;any&quot;)
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


InvokeResponse.model_rebuild()
RegisterToolRequest.model_rebuild()
ExecuteToolRequest.model_rebuild()
ExecuteToolResponse.model_rebuild()


_model: ToolUseModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;tool_use&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f&quot;token_{i}&quot; for i in range(DEFAULT_VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;tool-use-and-functional-calling&quot;,
        model_version=_model_version,
        model_type=&quot;tool_use&quot;,
    )

    logger.info(&quot;Tool use model loaded&quot;, model=&quot;tool-use-and-functional-calling&quot;, version=_model_version)
    yield
    logger.info(&quot;Shutting down tool-use API&quot;)


def _load_model() -&gt; tuple[ToolUseModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            tu_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;tool-use-and-functional-calling&quot;]
            if tu_models:
                tu_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = tu_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                json_files = list(model_dir.glob(&quot;tool_use_v*.json&quot;)) + list(model_dir.glob(&quot;*.json&quot;))
                if json_files:
                    return ToolUseModel.load(str(json_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;tool-use-and-functional-calling&quot; / MODEL_VERSION
            if model_dir.exists():
                json_files = list(model_dir.glob(&quot;tool_use_v*.json&quot;)) + list(model_dir.glob(&quot;*.json&quot;))
                if json_files:
                    return ToolUseModel.load(str(json_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    json_path = MODEL_DIR / &quot;tool_use.json&quot;
    if json_path.exists():
        return ToolUseModel.load(str(json_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/tool_use_v1.0.0.json&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;tool_use_v1.0.0.json&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return ToolUseModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    model = ToolUseModel(model_id=&quot;baseline&quot;, base_model_name=&quot;tool-use-v1&quot;)
    model._init()
    return model, &quot;1.0.0-baseline&quot;


app = FastAPI(
    title=&quot;Tool Use and Functional Calling API&quot;,
    description=&quot;Function calling (tool use) in LLMs: architecture, workflow execution, tool management&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;tool-use-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;n_tools&quot;: len(_model.tools) if _model else 0,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;invoke&quot;: &quot;POST /invoke&quot;,
            &quot;tools_list&quot;: &quot;GET /tools&quot;,
            &quot;tool_detail&quot;: &quot;GET /tools/{tool_name}&quot;,
            &quot;register_tool&quot;: &quot;POST /tools/register&quot;,
            &quot;execute_tool&quot;: &quot;POST /tools/execute&quot;,
            &quot;workflows&quot;: &quot;GET /workflows&quot;,
            &quot;applications&quot;: &quot;GET /applications&quot;,
            &quot;advantages&quot;: &quot;GET /advantages&quot;,
            &quot;limitations&quot;: &quot;GET /limitations&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;metrics&quot;: &quot;/metrics&quot;,
        },
    }


@app.get(&quot;/health&quot;)
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {
        &quot;status&quot;: &quot;healthy&quot;,
        &quot;model_loaded&quot;: True,
        &quot;model_version&quot;: _model_version,
        &quot;model_id&quot;: _model.model_id if _model else &quot;unknown&quot;,
        &quot;n_tools&quot;: len(_model.tools) if _model else 0,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/invoke&quot;, response_model=InvokeResponse)
def invoke_tool_use(body: InvokeRequest):
    &quot;&quot;&quot;Run the full function-calling workflow for a user query.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        if body.workflow:
            result = _model.run_workflow(body.workflow)
            if result is None:
                raise HTTPException(status_code=404, detail=f&quot;Workflow not found: {body.workflow}&quot;)
        else:
            result = _model.invoke(body.query)

        response = InvokeResponse(
            query=result[&quot;query&quot;],
            tool_call=result.get(&quot;tool_call&quot;),
            tool_result=result.get(&quot;tool_result&quot;),
            final_response=result[&quot;final_response&quot;],
            success=result[&quot;success&quot;],
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        _recent_predictions.append([float(len(body.query.split()))])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)
        return response
    except HTTPException:
        raise
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;invoke&quot;)
        logger.exception(&quot;Tool use invoke failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Tool use invoke failed&quot;) from e


@app.get(&quot;/tools&quot;)
def list_tools():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {&quot;tools&quot;: _model.list_tools(), &quot;count&quot;: len(_model.tools)}


@app.get(&quot;/tools/{tool_name}&quot;)
def get_tool_detail(tool_name: str):
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    tool = get_tool_by_name(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f&quot;Tool not found: {tool_name}&quot;)
    return tool


@app.post(&quot;/tools/register&quot;)
def register_tool(body: RegisterToolRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    tool = ToolSpec(
        name=body.name,
        description=body.description,
        parameters=body.parameters,
        return_type=body.return_type,
        keywords=body.keywords,
    )
    _model.register_tool(tool)
    return {&quot;status&quot;: &quot;registered&quot;, &quot;tool&quot;: body.name}


@app.post(&quot;/tools/execute&quot;, response_model=ExecuteToolResponse)
def execute_tool(body: ExecuteToolRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
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


@app.get(&quot;/workflows&quot;)
def list_workflows():
    return {&quot;workflows&quot;: get_all_workflows(), &quot;count&quot;: len(get_all_workflows())}


@app.get(&quot;/applications&quot;)
def list_applications():
    from tool_use_and_functional_calling.data import get_applications
    return {&quot;applications&quot;: get_applications()}


@app.get(&quot;/advantages&quot;)
def list_advantages():
    from tool_use_and_functional_calling.data import get_advantages
    return {&quot;advantages&quot;: get_advantages()}


@app.get(&quot;/limitations&quot;)
def list_limitations():
    return {&quot;limitations&quot;: get_limitations()}


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        model_id=info[&quot;model_id&quot;],
        base_model_name=info[&quot;base_model_name&quot;],
        n_tools=info[&quot;n_tools&quot;],
        n_tool_calls=info[&quot;n_tool_calls&quot;],
        n_tool_results=info[&quot;n_tool_results&quot;],
        n_executions=info[&quot;n_executions&quot;],
        model_version=_model_version,
    )</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-864071894')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-864071894"><code class="language-bash">uv run python -m tool_use_functional_calling.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>tool-use-functional-calling</strong></p>
</footer>
<script>
function copyCode(id) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.innerText);
}
function renderMath() {
  renderMathInElement(document.body, { delimiters: [{left: "$$", right: "$$", display: true}] });
}
</script>
</body>
</html>