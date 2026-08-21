<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>prompt-engineering - AI App Documentation</title>
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
<p class="section-subtitle">Prompt Engineering — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(y|x, p) = \prod_{t=1}^{|y|} P(y_t | x, p, y_{<t})$$</div>
<div class="math-block">$$\hat{p} = \arg\max_p \mathbb{E}_{x \sim \mathcal{D}} [\log P(y^* | x, p)]$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Prompt engineering reformulates downstream tasks as language modeling. Given a prompt $p$, the model generates output $y$ autoregressively. Prompt tuning optimizes $p$ to maximize task-specific likelihood. Soft prompts are continuous embeddings optimized via gradient descent.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive prompt comparison table; generation diversity vs prompt length; token probability explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  PromptTemplate
  PromptTechnique
  PromptExample
  PromptEvaluator
  PromptOptimizer
  PromptEngineeringModel</pre>
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
<tr><td><code>GET</code></td><td><code>/metrics</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3484462921')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3484462921"><code class="language-python">&quot;&quot;&quot;Training pipeline for Prompt Engineering.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from prompt_engineering.data import load_prompt_dataset, save_dataset, train_test_split
from prompt_engineering.model import (
    PromptEngineeringModel,
    PromptEvaluator,
    PromptTemplate,
)

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    model_id: str = &quot;prompt-engineering-v1&quot;,
    base_model_name: str = &quot;default&quot;,
    technique: str = &quot;zero-shot&quot;,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    logger.info(&quot;Loading prompt dataset&quot;, n_samples=n_samples, technique=technique)
    X, y = load_prompt_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / &quot;training_data.npz&quot;)

    model = PromptEngineeringModel(model_id=model_id, base_model_name=base_model_name)
    model._init()

    template = PromptTemplate(
        template_id=&quot;default&quot;,
        template_text=&quot;Analyze the following: {input_text}&quot;,
        placeholders=[&quot;input_text&quot;],
        technique=technique,
    )
    model.register_template(template)
    model.set_technique(technique)

    prompt = model.generate_prompt(&quot;default&quot;, input_text=&quot;Sample prompt for testing&quot;)
    logger.info(&quot;Generated prompt&quot;, prompt=prompt[:100])

    evaluator = PromptEvaluator()
    test_responses = [
        (&quot;Sample response 1&quot;, &quot;Expected output 1&quot;),
        (&quot;Sample response 2&quot;, &quot;Expected output 2&quot;),
        (&quot;Sample response 3&quot;, &quot;Expected output 3&quot;),
    ]
    for response, expected in test_responses:
        scores = model.evaluate_prompt(prompt, response, expected)
        logger.info(&quot;Evaluation scores&quot;, scores=scores)

    avg_scores = evaluator.get_average_scores()
    logger.info(&quot;Average evaluation scores&quot;, scores=avg_scores)

    if len(test_responses) &gt; 0:
        optimized_prompt = model.optimize_prompt(prompt, test_responses)
        logger.info(&quot;Optimized prompt&quot;, optimized=optimized_prompt[:100])

    model_path = model_dir / f&quot;prompt_engineering_v{model_version}.json&quot;
    model.save(str(model_path))

    metrics = {
        &quot;n_samples&quot;: float(len(X)),
        &quot;n_train&quot;: float(len(X_train)),
        &quot;n_test&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(vocab_size),
        &quot;technique&quot;: technique,
        &quot;n_templates&quot;: float(len(model.templates)),
        &quot;n_techniques&quot;: float(len(model.techniques)),
        &quot;avg_relevance&quot;: avg_scores.get(&quot;relevance&quot;, 0.0),
        &quot;avg_clarity&quot;: avg_scores.get(&quot;clarity&quot;, 0.0),
        &quot;avg_completeness&quot;: avg_scores.get(&quot;completeness&quot;, 0.0),
        &quot;avg_accuracy&quot;: avg_scores.get(&quot;accuracy&quot;, 0.0),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;prompt-engineering&quot;,
        model_version=model_version,
        model_type=&quot;nlp&quot;,
        metrics=metrics,
        parameters={
            &quot;model_id&quot;: model_id,
            &quot;base_model_name&quot;: base_model_name,
            &quot;technique&quot;: technique,
            &quot;n_samples&quot;: n_samples,
            &quot;vocab_size&quot;: vocab_size,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={f&quot;prompt_engineering_v{model_version}.json&quot;: model_path, &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;},
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;prompt_engineering&quot;, &quot;model_type&quot;: &quot;PromptEngineering&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;prompt-engineering&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;model_id&quot;: model_id, &quot;technique&quot;: technique, &quot;n_samples&quot;: n_samples},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;prompt_engineering&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Prompt Engineering Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, &quot;1000&quot;)))
    parser.add_argument(&quot;--model-id&quot;, type=str, default=os.getenv(&quot;MODEL_ID&quot;, &quot;prompt-engineering-v1&quot;))
    parser.add_argument(&quot;--base-model-name&quot;, type=str, default=os.getenv(&quot;BASE_MODEL_NAME&quot;, &quot;default&quot;))
    parser.add_argument(&quot;--technique&quot;, type=str, default=os.getenv(&quot;TECHNIQUE&quot;, &quot;zero-shot&quot;), choices=[&quot;zero-shot&quot;, &quot;few-shot&quot;, &quot;chain-of-thought&quot;, &quot;self-ask&quot;, &quot;least-to-most&quot;, &quot;meta-prompting&quot;, &quot;context-amplification&quot;, &quot;iterative&quot;])
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        model_id=args.model_id,
        base_model_name=args.base_model_name,
        technique=args.technique,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2022217226')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2022217226"><code class="language-python">&quot;&quot;&quot;Serving API for Prompt Engineering.&quot;&quot;&quot;

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

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;PROMPT_ENGINEERING_METRICS_PORT&quot;, &quot;9022&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class GenerateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    technique: str = Field(default=&quot;zero-shot&quot;)
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
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;prompt_engineering&quot;, port=METRICS_PORT)
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
        model_name=&quot;prompt-engineering&quot;,
        model_version=_model_version,
        model_type=&quot;nlp&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;prompt-engineering&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down prompt-engineering API&quot;)


def _load_model() -&gt; tuple[PromptEngineeringModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            pe_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;prompt-engineering&quot;]
            if pe_models:
                pe_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = pe_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                json_files = list(model_dir.glob(&quot;prompt_engineering_v*.json&quot;)) + list(model_dir.glob(&quot;*.json&quot;))
                if json_files:
                    return PromptEngineeringModel.load(str(json_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;prompt-engineering&quot; / MODEL_VERSION
            if model_dir.exists():
                json_files = list(model_dir.glob(&quot;prompt_engineering_v*.json&quot;)) + list(model_dir.glob(&quot;*.json&quot;))
                if json_files:
                    return PromptEngineeringModel.load(str(json_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    json_path = MODEL_DIR / &quot;prompt_engineering.json&quot;
    if json_path.exists():
        return PromptEngineeringModel.load(str(json_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/prompt_engineering_v1.0.0.json&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;prompt_engineering_v1.0.0.json&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return PromptEngineeringModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    model = PromptEngineeringModel(model_id=&quot;baseline&quot;, base_model_name=&quot;default&quot;)
    model._init()
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    from prompt_engineering.data import generate_synthetic_prompts
    X_base, _ = generate_synthetic_prompts(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title=&quot;Prompt Engineering API&quot;,
    description=&quot;Prompt Engineering service with various techniques (zero-shot, few-shot, chain-of-thought, self-ask, least-to-most, meta-prompting, context-amplification, iterative)&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;prompt_engineering-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;available_techniques&quot;: _model.get_available_techniques() if _model else [],
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;generate&quot;: &quot;POST /generate&quot;,
            &quot;evaluate&quot;: &quot;POST /evaluate&quot;,
            &quot;optimize&quot;: &quot;POST /optimize&quot;,
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
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/generate&quot;, response_model=GenerateResponse)
def generate_prompt(body: GenerateRequest):
    &quot;&quot;&quot;Generate a prompt using the specified template and technique.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

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
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;generation&quot;)
        logger.exception(&quot;Prompt generation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prompt generation failed&quot;) from e


@app.post(&quot;/evaluate&quot;, response_model=EvaluateResponse)
def evaluate_prompt(body: EvaluateRequest):
    &quot;&quot;&quot;Evaluate a prompt response against expected output.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

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
        _metrics.record_error(model_version=_model_version, error_type=&quot;evaluation&quot;)
        logger.exception(&quot;Prompt evaluation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prompt evaluation failed&quot;) from e


@app.post(&quot;/optimize&quot;, response_model=OptimizeResponse)
def optimize_prompt(body: OptimizeRequest):
    &quot;&quot;&quot;Optimize a prompt based on multiple response-evaluation pairs.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        responses = [(r[&quot;response&quot;], r.get(&quot;expected&quot;)) for r in body.responses]
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
        _metrics.record_error(model_version=_model_version, error_type=&quot;optimization&quot;)
        logger.exception(&quot;Prompt optimization failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prompt optimization failed&quot;) from e


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        model_id=info[&quot;model_id&quot;],
        base_model_name=info[&quot;base_model_name&quot;],
        n_templates=info[&quot;n_templates&quot;],
        n_techniques=info[&quot;n_techniques&quot;],
        current_technique=info[&quot;current_technique&quot;],
        n_history_entries=info[&quot;n_history_entries&quot;],
        model_version=_model_version,
    )</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2692160528')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2692160528"><code class="language-bash">uv run python -m prompt_engineering.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>prompt-engineering</strong></p>
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