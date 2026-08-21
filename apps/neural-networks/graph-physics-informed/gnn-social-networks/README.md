<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>gnn-social-networks - AI App Documentation</title>
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
<p class="section-subtitle">Graph Neural Network (GNN) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$h_v^{(k+1)} = \sigma\left( W^{(k)} \cdot \text{AGGREGATE}_k \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$</div>
<div class="math-block">$$\text{AGGREGATE}_k = \text{mean} \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right)$$</div>
<div class="math-block">$$\text{GAT}: \alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_v]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(a^T [Wh_u \| Wh_k]))}$$</div>
<div class="math-block">$$h_v^{(k+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(k)} \right)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>GNNs generalize convolutions to graph-structured data. Each node updates its representation by aggregating messages from neighbors. After $K$ rounds of message passing, each node embeds its $K$-hop neighborhood. GATs introduce attention weights $\alpha_{uv}$ to prioritize important neighbors.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive graph with animated message passing; node embedding t-SNE projection; attention weight heatmap.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  GCNLayer
  GNNSocialNetworks</pre>
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
<button class="copy-btn" onclick="copyCode('code-2531042842')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2531042842"><code class="language-python">&quot;&quot;&quot;Training pipeline for GNN Social Network Analysis.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_gnn_social_networks_schema

from gnn_social_networks.data import (
    N_CLASSES,
    N_FEATURES,
    generate_synthetic_data,
    save_training_data,
)
from gnn_social_networks.model import GNNSocialNetworks

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_nodes: int = 20,
    hidden_dim: int = 16,
    learning_rate: float = 0.05,
    n_iterations: int = 200,
    weight_decay: float = 0.001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    X, A, y = generate_synthetic_data(
        n_samples=n_nodes, n_nodes=n_nodes, n_features=N_FEATURES, random_seed=random_seed
    )
    logger.info(&quot;Generated graph data&quot;, n_nodes=n_nodes, data_path=str(data_path))

    validator = DataValidator(create_gnn_social_networks_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, A, y, model_dir / &quot;training_data.npz&quot;)

    model = GNNSocialNetworks(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X, A, y)

    metrics = model.evaluate(X, A, y)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;gnn_model_v{model_version}.npz&quot;
    model.save(str(model_path))
    np.savez(model_dir / &quot;adjacency_matrix.npz&quot;, A=A)

    metrics_summary = {
        **metrics,
        &quot;training_mode&quot;: &quot;supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_nodes&quot;: float(n_nodes),
        &quot;hidden_dim&quot;: float(hidden_dim),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;gnn-social-networks&quot;,
        model_version=model_version,
        model_type=&quot;classification&quot;,
        metrics=metrics_summary,
        parameters={
            &quot;n_features&quot;: N_FEATURES,
            &quot;n_classes&quot;: N_CLASSES,
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;n_nodes&quot;: n_nodes,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;gnn_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
            &quot;adjacency_matrix.npz&quot;: model_dir / &quot;adjacency_matrix.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;gnn_social_networks&quot;, &quot;model_type&quot;: &quot;GNN&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;gnn-social-networks&quot;,
            model_version=model_version,
            metrics=metrics_summary,
            params={&quot;n_features&quot;: N_FEATURES, &quot;n_classes&quot;: N_CLASSES, &quot;hidden_dim&quot;: hidden_dim, &quot;learning_rate&quot;: learning_rate, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;gnn&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description=&quot;Train GNN Social Network model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-nodes&quot;, type=int, default=int(os.getenv(&quot;N_NODES&quot;, &quot;20&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;16&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.05&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;200&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.001&quot;)))
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
        n_nodes=args.n_nodes,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1782406336')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1782406336"><code class="language-python">&quot;&quot;&quot;Serving API for GNN Social Network Analysis.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_gnn_social_networks_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from gnn_social_networks.data import N_CLASSES, N_FEATURES, generate_synthetic_data
from gnn_social_networks.model import GNNSocialNetworks

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;GNN_METRICS_PORT&quot;, &quot;8029&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)
    adjacency_row: list[float] = Field(..., min_length=20, max_length=20)


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_probabilities: list[float]
    model_version: str
    training_mode: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    n_features: int
    n_classes: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: GNNSocialNetworks | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []
_adjacency: np.ndarray | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data, _adjacency

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;gnn_social_networks&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_gnn_social_networks_schema())
    feature_names = [f&quot;node_{i}&quot; for i in range(N_FEATURES)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version, _adjacency = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;gnn-social-networks&quot;,
        model_version=_model_version,
        model_type=&quot;classification&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;gnn-social-networks&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down gnn-social-networks API&quot;)


def _load_model() -&gt; tuple[GNNSocialNetworks, str, np.ndarray | None]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;gnn-social-networks&quot;]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;gnn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    adj_path = model_dir / &quot;adjacency_matrix.npz&quot;
                    adj = None
                    if adj_path.exists():
                        adj_data = np.load(adj_path)
                        adj = adj_data[&quot;A&quot;]
                    return GNNSocialNetworks.load(str(npz_files[0])), latest[&quot;model_version&quot;], adj
        else:
            model_dir = MODEL_DIR / &quot;gnn-social-networks&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;gnn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    adj_path = model_dir / &quot;adjacency_matrix.npz&quot;
                    adj = None
                    if adj_path.exists():
                        adj_data = np.load(adj_path)
                        adj = adj_data[&quot;A&quot;]
                    return GNNSocialNetworks.load(str(npz_files[0])), MODEL_VERSION, adj
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;gnn_model.npz&quot;
    if npz_path.exists():
        return GNNSocialNetworks.load(str(npz_path)), &quot;legacy&quot;, None

    candidate_paths = [
        Path(&quot;/app/artifacts/models/gnn_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;gnn_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            X_base, A_base, _ = generate_synthetic_data(n_samples=100, n_nodes=20, random_seed=42)
            return GNNSocialNetworks.load(str(p)), &quot;1.0.0-bundled&quot;, A_base

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, A_base, y_base = generate_synthetic_data(n_samples=100, n_nodes=20, random_seed=42)
    model = GNNSocialNetworks(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        hidden_dim=16,
        learning_rate=0.05,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base, A_base, y_base)
    return model, &quot;1.0.0-baseline&quot;, A_base


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _, _ = generate_synthetic_data(n_samples=100, n_nodes=20, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;GNN Social Network Analysis API&quot;,
    description=&quot;Processes graph-structured data using Graph Convolution to optimize directly on network topology&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;gnn_social_networks-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;n_features&quot;: N_FEATURES,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
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
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    global _model, _model_version, _reference_data, _adjacency
    try:
        _model, _model_version, _adjacency = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;gnn-social-networks&quot;,
                model_version=_model_version,
                model_type=&quot;classification&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;gnn-social-networks&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)
    if len(_recent_predictions) &lt; 10:
        return {&quot;total_features&quot;: N_FEATURES, &quot;drifted_features&quot;: 0, &quot;drift_ratio&quot;: 0.0, &quot;drifted&quot;: [], &quot;all_results&quot;: []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])
    return summary


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.layers:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return StatsResponse(
        n_features=_model.n_features,
        n_classes=_model.n_classes,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Classify a node using GNN with graph structure.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array([body.features]).reshape(1, -1)
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        A = np.eye(1) if _adjacency is None else _adjacency[:1, :1]

        probs = _model.predict_proba(X, A)[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

        response = PredictResponse(
            predicted_class=pred,
            confidence=round(confidence, 4),
            class_probabilities=probs.tolist(),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(body.features)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2443827444')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2443827444"><code class="language-bash">uv run python -m gnn_social_networks.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../deep-belief-networks/README.md">deep-belief-networks</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>gnn-social-networks</strong></p>
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