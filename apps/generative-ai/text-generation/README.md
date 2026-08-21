<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>text-generation - AI App Documentation</title>
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
<p class="section-subtitle">Text Generation — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(w_t | w_{<t}) = \text{softmax}(W_h h_t + b_h)$$</div>
<div class="math-block">$$h_t = \text{LSTM}(x_t, h_{t-1})$$</div>
<div class="math-block">$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t})$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Text generation models learn to predict the next token given past context. Temperature scaling controls randomness: high temperature yields creative but incoherent text; low temperature yields repetitive but safe text. Top-k and nucleus sampling truncate the probability mass to improve diversity.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive temperature slider; generated text preview; perplexity vs context length.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  TextTokenizer
  MultiHeadAttention
  AddNorm
  FeedForward
  TransformerBlock
  BaseTextModel
  SamplingStrategy
  TextGenerationModel</pre>
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
<button class="copy-btn" onclick="copyCode('code-4080192439')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4080192439"><code class="language-python">&quot;&quot;&quot;Training pipeline for Text Generation.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from text_gen.data import load_text_dataset, save_dataset, train_test_split
from text_gen.model import TextGenerationModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    model_id: str = &quot;text-generation-v1&quot;,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    logger.info(&quot;Loading text dataset&quot;, n_samples=n_samples, temperature=temperature)
    X, y = load_text_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / &quot;training_data.npz&quot;)

    model = TextGenerationModel(
        model_id=model_id,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        random_seed=random_seed,
    )
    model._init()

    metrics = model.fit(X_train, y_train)
    logger.info(&quot;Training finished&quot;, metrics=metrics)

    eval_metrics = model.evaluate(X_test, y_test)
    logger.info(&quot;Evaluation metrics&quot;, metrics=eval_metrics)

    model_path = model_dir / f&quot;text_generation_v{model_version}.npz&quot;
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        &quot;temperature&quot;: temperature,
        &quot;top_k&quot;: float(top_k),
        &quot;top_p&quot;: top_p,
        &quot;n_samples&quot;: float(n_samples),
        &quot;vocab_size&quot;: float(vocab_size),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;text-generation&quot;,
        model_version=model_version,
        model_type=&quot;generative&quot;,
        metrics=combined_metrics,
        parameters={
            &quot;model_id&quot;: model_id,
            &quot;vocab_size&quot;: vocab_size,
            &quot;temperature&quot;: temperature,
            &quot;top_k&quot;: top_k,
            &quot;top_p&quot;: top_p,
            &quot;n_samples&quot;: n_samples,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={f&quot;text_generation_v{model_version}.npz&quot;: model_path, &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;},
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;text_generation&quot;, &quot;model_type&quot;: &quot;TextGeneration&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;text-generation&quot;,
            model_version=model_version,
            metrics=combined_metrics,
            params={&quot;model_id&quot;: model_id, &quot;temperature&quot;: temperature, &quot;top_k&quot;: top_k, &quot;top_p&quot;: top_p, &quot;n_samples&quot;: n_samples},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;text_generation&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Text Generation Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, &quot;1000&quot;)))
    parser.add_argument(&quot;--model-id&quot;, type=str, default=os.getenv(&quot;MODEL_ID&quot;, &quot;text-generation-v1&quot;))
    parser.add_argument(&quot;--temperature&quot;, type=float, default=float(os.getenv(&quot;TEMPERATURE&quot;, &quot;0.8&quot;)))
    parser.add_argument(&quot;--top-k&quot;, type=int, default=int(os.getenv(&quot;TOP_K&quot;, &quot;50&quot;)))
    parser.add_argument(&quot;--top-p&quot;, type=float, default=float(os.getenv(&quot;TOP_P&quot;, &quot;0.9&quot;)))
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
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-925971203')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-925971203"><code class="language-python">&quot;&quot;&quot;Serving API for Text Generation.&quot;&quot;&quot;

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
from text_gen.data import DEFAULT_VOCAB_SIZE
from text_gen.model import TextGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;TEXT_GENERATION_METRICS_PORT&quot;, &quot;9024&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    model_version: str


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    reference_text: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    score: float
    model_version: str


class StatsResponse(BaseModel):
    model_id: str
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int
    temperature: float
    top_k: int
    top_p: float
    model_version: str


_model: TextGenerationModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;text_gen_generative&quot;, port=METRICS_PORT)
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
        model_name=&quot;text-generation&quot;,
        model_version=_model_version,
        model_type=&quot;generative&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;text-generation&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down text-generation API&quot;)


def _load_model() -&gt; tuple[TextGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            tg_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;text-generation&quot;]
            if tg_models:
                tg_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = tg_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;text_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return TextGenerationModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;text-generation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;text_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return TextGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;text_generation.npz&quot;
    if npz_path.exists():
        return TextGenerationModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/text_generation_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;text_generation_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return TextGenerationModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    model = TextGenerationModel(model_id=&quot;baseline&quot;, vocab_size=DEFAULT_VOCAB_SIZE)
    model._init()
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    from nlp_text_generation.data import generate_synthetic_text
    X_base, _ = generate_synthetic_text(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title=&quot;Text Generation API&quot;,
    description=&quot;Transformer-based autoregressive text generation with temperature, top-k, and top-p sampling&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;text-generation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;generate&quot;: &quot;POST /generate&quot;,
            &quot;evaluate&quot;: &quot;POST /evaluate&quot;,
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
def generate_text(body: GenerateRequest):
    &quot;&quot;&quot;Generate text from a prompt using the transformer model.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        _model.temperature = body.temperature
        _model.top_k = body.top_k
        _model.top_p = body.top_p
        generated_text = _model.generate(body.prompt, max_new_tokens=body.max_new_tokens)

        response = GenerateResponse(
            generated_text=generated_text,
            prompt=body.prompt,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(len(body.prompt.split()))])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;generation&quot;)
        logger.exception(&quot;Text generation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Text generation failed&quot;) from e


@app.post(&quot;/evaluate&quot;, response_model=EvaluateResponse)
def evaluate_text(body: EvaluateRequest):
    &quot;&quot;&quot;Evaluate generated text against a reference.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        generated = _model.generate(body.prompt, max_new_tokens=50)
        gen_words = set(generated.lower().split())
        ref_words = set(body.reference_text.lower().split())
        score = len(gen_words.intersection(ref_words)) / max(len(ref_words), 1)

        response = EvaluateResponse(
            score=round(score, 4),
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;evaluation&quot;)
        logger.exception(&quot;Text evaluation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Text evaluation failed&quot;) from e


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        model_id=info.get(&quot;model_id&quot;, &quot;unknown&quot;),
        vocab_size=info.get(&quot;vocab_size&quot;, DEFAULT_VOCAB_SIZE),
        d_model=info.get(&quot;d_model&quot;, 256),
        n_layers=info.get(&quot;n_layers&quot;, 2),
        max_seq_len=info.get(&quot;max_seq_len&quot;, 128),
        temperature=info.get(&quot;temperature&quot;, 0.8),
        top_k=info.get(&quot;top_k&quot;, 50),
        top_p=info.get(&quot;top_p&quot;, 0.9),
        model_version=_model_version,
    )</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1335465644')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1335465644"><code class="language-bash">uv run python -m text_generation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../code-generation/README.md">code-generation</a></li>
<li><a href="../image-generation/README.md">image-generation</a></li>
<li><a href="../retrieval-augmented-generation/README.md">retrieval-augmented-generation</a></li>
<li><a href="../video-generation/README.md">video-generation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>text-generation</strong></p>
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