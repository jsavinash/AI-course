<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>code-generation - AI App Documentation</title>
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
<p class="section-subtitle">Code Generation — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(c | p) = \prod_{t=1}^{|c|} P(c_t | p, c_{<t})$$</div>
<div class="math-block">$$\mathcal{L} = -\sum_{t=1}^{|c|} \log P(c_t | p, c_{<t}; \theta)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Code generation treats source code as a sequence modeled by a language model. The prompt $p$ provides context (docstring, imports, function signature). The model predicts tokens autoregressively, conditioned on previous predictions. Beam search and nucleus sampling improve output quality and diversity.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive code completion demo; token probability heatmap; beam search tree explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  CodeTokenizer
  MultiHeadAttention
  AddNorm
  FeedForward
  TransformerBlock
  BaseCodeModel
  CodeCompletionModel
  TextToCodeModel
  RefactoringModel
  TestingAndDebuggingModel
  CodeGenerationModel</pre>
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
<tr><td><code>POST</code></td><td><code>/reload</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1140404946')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1140404946"><code class="language-python">&quot;&quot;&quot;Training pipeline for Code Generation.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from code_generation.data import (
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    load_code_dataset,
    save_dataset,
    train_test_split,
)
from code_generation.model import CodeGenerationModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    seq_len: int = 128,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 2,
    d_ff: int = 1024,
    max_seq_len: int = 128,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = &quot;1.0.0&quot;,
    random_seed: int = 42,
    register_to_mlflow: bool = False,
) -&gt; dict:
    logger.info(&quot;Loading code dataset&quot;, n_samples=n_samples)
    X, y = load_code_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / &quot;training_data.npz&quot;)

    model = CodeGenerationModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    logger.info(&quot;Starting code generation training&quot;)
    model.fit(X_train, y_train, n_iterations=n_iterations)

    test_metrics = model.evaluate(X_test, y_test)
    logger.info(&quot;Training complete&quot;, final_loss=model.loss_history[-1], test_accuracy=test_metrics[&quot;accuracy&quot;])

    model_path = model_dir / f&quot;code_generation_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(vocab_size),
        &quot;d_model&quot;: float(d_model),
        &quot;n_layers&quot;: float(n_layers),
        &quot;d_ff&quot;: float(d_ff),
        &quot;max_seq_len&quot;: float(max_seq_len),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;code-generation&quot;,
        model_version=model_version,
        model_type=&quot;generation&quot;,
        metrics=metrics,
        parameters={
            &quot;vocab_size&quot;: vocab_size,
            &quot;d_model&quot;: d_model,
            &quot;n_heads&quot;: n_heads,
            &quot;n_layers&quot;: n_layers,
            &quot;d_ff&quot;: d_ff,
            &quot;max_seq_len&quot;: max_seq_len,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;code_generation_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;code_generation&quot;, &quot;model_type&quot;: &quot;CodeGeneration&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;code-generation&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;vocab_size&quot;: vocab_size, &quot;d_model&quot;: d_model, &quot;n_layers&quot;: n_layers, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;code_generation&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Code Generation Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, str(VOCAB_SIZE))))
    parser.add_argument(&quot;--seq-len&quot;, type=int, default=int(os.getenv(&quot;SEQ_LEN&quot;, str(MAX_SEQ_LEN))))
    parser.add_argument(&quot;--d-model&quot;, type=int, default=int(os.getenv(&quot;D_MODEL&quot;, &quot;256&quot;)))
    parser.add_argument(&quot;--n-heads&quot;, type=int, default=int(os.getenv(&quot;N_HEADS&quot;, &quot;8&quot;)))
    parser.add_argument(&quot;--n-layers&quot;, type=int, default=int(os.getenv(&quot;N_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--d-ff&quot;, type=int, default=int(os.getenv(&quot;D_FF&quot;, &quot;1024&quot;)))
    parser.add_argument(&quot;--max-seq-len&quot;, type=int, default=int(os.getenv(&quot;MAX_SEQ_LEN&quot;, str(MAX_SEQ_LEN))))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.01&quot;)))
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
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        random_seed=args.random_seed,
        register_to_mlflow=args.register_mlflow,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2129975769')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2129975769"><code class="language-python">&quot;&quot;&quot;Serving API for Code Generation.&quot;&quot;&quot;

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from code_generation.data import VOCAB_SIZE
from code_generation.model import CodeGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;CODE_GENERATION_METRICS_PORT&quot;, &quot;9020&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class CodeCompletionRequest(BaseModel):
    code_prefix: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(default=20, ge=1, le=200)


class CodeCompletionResponse(BaseModel):
    completed_code: str
    model_version: str


class TextToCodeRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(default=50, ge=1, le=200)


class TextToCodeResponse(BaseModel):
    generated_code: str
    model_version: str


class RefactorRequest(BaseModel):
    old_code: str = Field(..., min_length=1, max_length=500)
    target_language: str = Field(default=&quot;modern_python&quot;, max_length=50)


class RefactorResponse(BaseModel):
    refactored_code: str
    target_language: str
    model_version: str


class BugScanRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=500)


class BugScanResponse(BaseModel):
    bug_probability: float
    confidence: float
    suggested_fix: str
    model_version: str


class UnitTestRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(default=50, ge=1, le=200)


class UnitTestResponse(BaseModel):
    unit_tests: str
    model_version: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: CodeGenerationModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;code_generation&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f&quot;token_{i}&quot; for i in range(VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;code-generation&quot;,
        model_version=_model_version,
        model_type=&quot;generation&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;code-generation&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down code-generation API&quot;)


def _load_model() -&gt; tuple[CodeGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            cg_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;code-generation&quot;]
            if cg_models:
                cg_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = cg_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;code_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return CodeGenerationModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;code-generation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;code_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return CodeGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;code_generation.npz&quot;
    if npz_path.exists():
        return CodeGenerationModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/code_generation_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;code_generation_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return CodeGenerationModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    model = CodeGenerationModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=256,
        max_seq_len=32,
        learning_rate=0.001,
        n_iterations=10,
        random_seed=42,
    )
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    from code_generation.data import generate_synthetic_code_data
    X_base, _ = generate_synthetic_code_data(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title=&quot;Code Generation API&quot;,
    description=&quot;Generative AI code generation with capabilities for code completion, text-to-code, refactoring, testing, and debugging&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;code_generation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;capabilities&quot;: [&quot;code_completion&quot;, &quot;text_to_code&quot;, &quot;refactoring&quot;, &quot;testing_debugging&quot;],
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;complete&quot;: &quot;POST /complete&quot;,
            &quot;text_to_code&quot;: &quot;POST /text-to-code&quot;,
            &quot;refactor&quot;: &quot;POST /refactor&quot;,
            &quot;scan_bugs&quot;: &quot;POST /scan-bugs&quot;,
            &quot;generate_tests&quot;: &quot;POST /generate-tests&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;drift&quot;: &quot;GET /drift&quot;,
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
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;code-generation&quot;,
                model_version=_model_version,
                model_type=&quot;generation&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;code-generation&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)
    if len(_recent_predictions) &lt; 10:
        return {&quot;total_features&quot;: VOCAB_SIZE, &quot;drifted_features&quot;: 0, &quot;drift_ratio&quot;: 0.0, &quot;drifted&quot;: [], &quot;all_results&quot;: []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])
    return summary


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info[&quot;vocab_size&quot;],
        d_model=info[&quot;d_model&quot;],
        n_layers=info[&quot;n_layers&quot;],
        d_ff=info[&quot;d_ff&quot;],
        max_seq_len=info[&quot;max_seq_len&quot;],
        n_epochs_run=info[&quot;n_epochs_run&quot;],
        final_loss=info[&quot;final_loss&quot;],
        model_version=_model_version,
    )


@app.post(&quot;/complete&quot;, response_model=CodeCompletionResponse)
def complete_code(body: CodeCompletionRequest):
    &quot;&quot;&quot;Complete code given a prefix - predicts and auto-completes lines or full functions.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        completed = _model.complete_code(body.code_prefix, max_new_tokens=body.max_new_tokens)
        response = CodeCompletionResponse(
            completed_code=completed,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;code_completion&quot;)
        logger.exception(&quot;Code completion failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Code completion failed&quot;) from e


@app.post(&quot;/text-to-code&quot;, response_model=TextToCodeResponse)
def text_to_code(body: TextToCodeRequest):
    &quot;&quot;&quot;Translate plain English description into functional code blocks.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        generated = _model.text_to_code(body.description, max_new_tokens=body.max_new_tokens)
        response = TextToCodeResponse(
            generated_code=generated,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;text_to_code&quot;)
        logger.exception(&quot;Text-to-code generation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Text-to-code generation failed&quot;) from e


@app.post(&quot;/refactor&quot;, response_model=RefactorResponse)
def refactor_code(body: RefactorRequest):
    &quot;&quot;&quot;Upgrade older software frameworks, improve readability, translate code.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        refactored = _model.refactor_code(body.old_code, target_language=body.target_language)
        response = RefactorResponse(
            refactored_code=refactored,
            target_language=body.target_language,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;refactoring&quot;)
        logger.exception(&quot;Refactoring failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Refactoring failed&quot;) from e


@app.post(&quot;/scan-bugs&quot;, response_model=BugScanResponse)
def scan_bugs(body: BugScanRequest):
    &quot;&quot;&quot;Scan code for bugs and identify security vulnerabilities.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        result = _model.scan_for_bugs(body.code)
        response = BugScanResponse(
            bug_probability=result.get(&quot;bug_probability&quot;, 0.0),
            confidence=result.get(&quot;confidence&quot;, 0.0),
            suggested_fix=result.get(&quot;suggested_fix&quot;, &quot;&quot;),
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;bug_scan&quot;)
        logger.exception(&quot;Bug scan failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bug scan failed&quot;) from e


@app.post(&quot;/generate-tests&quot;, response_model=UnitTestResponse)
def generate_tests(body: UnitTestRequest):
    &quot;&quot;&quot;Auto-generate unit tests for given code.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        tests = _model.generate_unit_tests(body.code, max_new_tokens=body.max_new_tokens)
        response = UnitTestResponse(
            unit_tests=tests,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;test_generation&quot;)
        logger.exception(&quot;Test generation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Test generation failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2166420967')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2166420967"><code class="language-bash">uv run python -m code_generation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../image-generation/README.md">image-generation</a></li>
<li><a href="../retrieval-augmented-generation/README.md">retrieval-augmented-generation</a></li>
<li><a href="../text-generation/README.md">text-generation</a></li>
<li><a href="../video-generation/README.md">video-generation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>code-generation</strong></p>
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