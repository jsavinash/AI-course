<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>large-language-model - AI App Documentation</title>
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
<p class="section-subtitle">Transformer Architecture — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$</div>
<div class="math-block">$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$</div>
<div class="math-block">$$y = \text{softmax}(W_{proj} \cdot \text{LayerNorm}(x + \text{MultiHead}(x)))$$</div>
<div class="math-block">$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t}; \theta)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>The Transformer uses stacked encoder-decoder blocks. Each block applies multi-head self-attention followed by position-wise feed-forward networks, with residual connections and layer normalization. The decoder uses masked self-attention to prevent attending to future tokens during training.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive encoder-decoder diagram with attention head visualization and token probability explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  MultiHeadSelfAttention
  FeedForward
  TransformerBlock
  LargeLanguageModel</pre>
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
<button class="copy-btn" onclick="copyCode('code-1178553871')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1178553871"><code class="language-python">&quot;&quot;&quot;Training pipeline for Large Language Model (LLM).&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_large_language_model_schema

from large_language_model.data import (
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from large_language_model.model import LargeLanguageModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    max_seq_len: int = MAX_SEQ_LEN,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; float:
    X = generate_synthetic_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed)
    logger.info(&quot;Generated LLM training data&quot;, n_samples=n_samples, vocab_size=VOCAB_SIZE)

    validator = DataValidator(create_large_language_model_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)

    X_train, X_test = train_test_split(X, test_size=test_size, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, model_dir / &quot;training_data.npz&quot;)

    model = LargeLanguageModel(
        vocab_size=VOCAB_SIZE,
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
    model.fit(X_train)

    y_test = X_test
    test_metrics = model.evaluate(X_test, y_test)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;llm_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(VOCAB_SIZE),
        &quot;d_model&quot;: float(d_model),
        &quot;n_heads&quot;: float(n_heads),
        &quot;n_layers&quot;: float(n_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;large-language-model&quot;,
        model_version=model_version,
        model_type=&quot;classification&quot;,
        metrics=metrics,
        parameters={
            &quot;vocab_size&quot;: VOCAB_SIZE,
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
            f&quot;llm_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;large_language_model&quot;, &quot;model_type&quot;: &quot;LLM&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;large-language-model&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;vocab_size&quot;: VOCAB_SIZE, &quot;d_model&quot;: d_model, &quot;n_heads&quot;: n_heads, &quot;n_layers&quot;: n_layers, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;llm&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics[&quot;final_loss&quot;]


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Large Language Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--d-model&quot;, type=int, default=int(os.getenv(&quot;D_MODEL&quot;, &quot;128&quot;)))
    parser.add_argument(&quot;--n-heads&quot;, type=int, default=int(os.getenv(&quot;N_HEADS&quot;, &quot;4&quot;)))
    parser.add_argument(&quot;--n-layers&quot;, type=int, default=int(os.getenv(&quot;N_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--d-ff&quot;, type=int, default=int(os.getenv(&quot;D_FF&quot;, &quot;512&quot;)))
    parser.add_argument(&quot;--max-seq-len&quot;, type=int, default=int(os.getenv(&quot;MAX_SEQ_LEN&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--test-size&quot;, type=float, default=float(os.getenv(&quot;TEST_SIZE&quot;, &quot;0.2&quot;)))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics_loss = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, final_loss=metrics_loss, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2469418719')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2469418719"><code class="language-python">&quot;&quot;&quot;Serving API for Large Language Model (LLM).&quot;&quot;&quot;

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_large_language_model_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from large_language_model.data import MAX_SEQ_LEN, VOCAB_SIZE, generate_synthetic_data
from large_language_model.model import LargeLanguageModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;LLM_METRICS_PORT&quot;, &quot;8013&quot;))


class PredictRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=64)
    max_len: int = Field(default=10, ge=1, le=32)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=10, ge=1, le=100)


class PredictResponse(BaseModel):
    generated_tokens: list[int]
    next_token_probabilities: list[float]
    model_version: str
    training_mode: str


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: LargeLanguageModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_reference_data: np.ndarray | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;large_language_model&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_large_language_model_schema())

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;large-language-model&quot;,
        model_version=_model_version,
        model_type=&quot;classification&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;large-language-model&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down large-language-model API&quot;)


def _load_model() -&gt; tuple[LargeLanguageModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;large-language-model&quot;]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;llm_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return LargeLanguageModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;large-language-model&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;llm_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return LargeLanguageModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;llm_model.npz&quot;
    if npz_path.exists():
        return LargeLanguageModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/llm_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;llm_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return LargeLanguageModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base = generate_synthetic_data(n_samples=50, vocab_size=VOCAB_SIZE, random_seed=42)
    model = LargeLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=256,
        max_seq_len=MAX_SEQ_LEN,
        learning_rate=0.001,
        n_iterations=30,
        random_seed=42,
    )
    model.fit(X_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base = generate_synthetic_data(n_samples=50, vocab_size=VOCAB_SIZE, random_seed=42)
    return X_base.reshape(-1, 1)


app = FastAPI(
    title=&quot;Large Language Model API&quot;,
    description=&quot;Transformer-based LLM with self-attention, multi-head attention, positional encoding, and autoregressive decoding&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;large_language_model-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
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
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
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
                model_name=&quot;large-language-model&quot;,
                model_version=_model_version,
                model_type=&quot;classification&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;large-language-model&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.layers:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info[&quot;vocab_size&quot;],
        d_model=info[&quot;d_model&quot;],
        n_heads=info[&quot;n_heads&quot;],
        n_layers=info[&quot;n_layers&quot;],
        d_ff=info[&quot;d_ff&quot;],
        training_mode=info[&quot;training_mode&quot;],
        n_epochs_run=info[&quot;n_epochs_run&quot;],
        final_loss=info[&quot;final_loss&quot;],
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Generate next tokens using LLM with temperature and top-k sampling.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.tokens).reshape(1, -1)
    validation = _validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        generated = _model.predict(X, max_len=body.max_len, temperature=body.temperature, top_k=body.top_k)
        next_probs = _model.predict_proba(X)[0]

        probs_list = [float(p) for p in next_probs.flatten()]
        top_probs = probs_list[:10] + [0.0] * (10 - min(len(probs_list), 10))

        response = PredictResponse(
            generated_tokens=generated.tolist(),
            next_token_probabilities=top_probs,
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1037145862')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1037145862"><code class="language-bash">uv run python -m large_language_model.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../transformers-language-modeling/README.md">transformers-language-modeling</a></li>
<li><a href="../nlp-language-translation/README.md">nlp-language-translation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>large-language-model</strong></p>
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