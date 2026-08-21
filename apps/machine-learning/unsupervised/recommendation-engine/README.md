<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>recommendation-engine - AI App Documentation</title>
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
<p class="section-subtitle">Recommendation Engine (Collaborative Filtering) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$</div>
<div class="math-block">$$\min_{b^*} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mu - b_u - b_i)^2 + \lambda(\|b_u\|^2 + \|b_i\|^2)$$</div>
<div class="math-block">$$\text{cosine}(u,v) = \frac{u \cdot v}{\|u\| \|v\|}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Matrix factorization decomposes the user-item interaction matrix into latent factors. Bias terms capture global mean and user/item-specific offsets. Regularization prevents overfitting. Similarity metrics enable neighborhood-based recommendations.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive embedding scatter plot; recommendation coverage vs diversity trade-off; top-k recall curve.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  AssociationRule
  Apriori</pre>
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
<button class="copy-btn" onclick="copyCode('code-331664408')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-331664408"><code class="language-python">&quot;&quot;&quot;Production training pipeline for the recommendation engine (Apriori).&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from recommendation_engine.data import load_training_data, save_training_data
from recommendation_engine.model import Apriori

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_itemset_size: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the recommendation engine Apriori model and save artifacts.

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Load training data
    transactions = load_training_data(data_path, random_seed=random_seed)
    logger.info(&quot;Loaded training data&quot;, n_transactions=len(transactions))

    # Save training data for reproducibility
    save_training_data(transactions, model_dir / &quot;training_data.csv&quot;)

    # Train model
    model = Apriori(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_itemset_size=max_itemset_size,
    )
    model.fit(transactions)

    # Evaluate model quality
    metrics = model.evaluate(transactions)
    logger.info(
        &quot;Training complete&quot;,
        n_rules=metrics[&quot;n_rules&quot;],
        n_frequent_itemsets=metrics[&quot;n_frequent_itemsets&quot;],
        coverage=metrics[&quot;coverage&quot;],
        avg_confidence=metrics[&quot;avg_confidence&quot;],
        avg_lift=metrics[&quot;avg_lift&quot;],
    )

    # Model validation - check rule quality
    if metrics[&quot;n_rules&quot;] == 0:
        logger.warning(
            &quot;No rules generated. Consider lowering min_support or min_confidence.&quot;,
            min_support=min_support,
            min_confidence=min_confidence,
        )

    # Save model
    model_path = model_dir / f&quot;recommendation_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, transactions, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        &quot;n_rules&quot;: metrics[&quot;n_rules&quot;],
        &quot;n_frequent_itemsets&quot;: metrics[&quot;n_frequent_itemsets&quot;],
        &quot;coverage&quot;: metrics[&quot;coverage&quot;],
        &quot;avg_confidence&quot;: metrics[&quot;avg_confidence&quot;],
        &quot;avg_lift&quot;: metrics[&quot;avg_lift&quot;],
        &quot;avg_support&quot;: metrics[&quot;avg_support&quot;],
        &quot;n_transactions&quot;: float(len(transactions)),
        &quot;n_products&quot;: float(len(model.products)),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;recommendation-engine&quot;,
        model_version=model_version,
        model_type=&quot;association_rules&quot;,
        metrics=training_metrics,
        parameters={
            &quot;min_support&quot;: min_support,
            &quot;min_confidence&quot;: min_confidence,
            &quot;min_lift&quot;: min_lift,
            &quot;max_itemset_size&quot;: max_itemset_size,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;recommendation_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;association_rules&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;recommendation-engine&quot;,
            model_version=model_version,
            metrics=training_metrics,
            params={
                &quot;min_support&quot;: min_support,
                &quot;min_confidence&quot;: min_confidence,
                &quot;min_lift&quot;: min_lift,
                &quot;max_itemset_size&quot;: max_itemset_size,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;recommendation_engine_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;association_rules&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;recommendation-engine&quot;, version=model_version
        )

    return training_metrics


def _save_chart(
    model: Apriori,
    transactions: list[list[str]],
    output_dir: Path,
    version: str,
) -&gt; None:
    &quot;&quot;&quot;Save the association rules chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.rules:
        return

    plt.figure(figsize=(12, 6))

    # Plot top rules by lift
    top_rules = model.rules[:15]
    labels = [
        f&quot;{'+'.join(sorted(r.antecedent))} -&gt; {'+'.join(sorted(r.consequent))}&quot; for r in top_rules
    ]
    lifts = [r.lift for r in top_rules]
    confidences = [r.confidence for r in top_rules]

    x = range(len(top_rules))
    bars = plt.bar(x, lifts, color=&quot;steelblue&quot;, alpha=0.7, label=&quot;Lift&quot;)

    # Add confidence as text on bars
    for _i, (bar, conf) in enumerate(zip(bars, confidences, strict=False)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f&quot;conf={conf:.2f}&quot;,
            ha=&quot;center&quot;,
            va=&quot;bottom&quot;,
            fontsize=8,
        )

    plt.xlabel(&quot;Association Rule&quot;)
    plt.ylabel(&quot;Lift&quot;)
    plt.title(f&quot;Top Association Rules by Lift - v{version}&quot;)
    plt.xticks(x, labels, rotation=45, ha=&quot;right&quot;, fontsize=8)
    plt.grid(True, alpha=0.3, axis=&quot;y&quot;)
    plt.legend()
    plt.tight_layout()

    chart_path = output_dir / f&quot;recommendation_engine_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train recommendation engine Apriori model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(
        &quot;--min-support&quot;, type=float, default=float(os.getenv(&quot;MIN_SUPPORT&quot;, &quot;0.05&quot;))
    )
    parser.add_argument(
        &quot;--min-confidence&quot;, type=float, default=float(os.getenv(&quot;MIN_CONFIDENCE&quot;, &quot;0.5&quot;))
    )
    parser.add_argument(&quot;--min-lift&quot;, type=float, default=float(os.getenv(&quot;MIN_LIFT&quot;, &quot;1.0&quot;)))
    parser.add_argument(
        &quot;--max-itemset-size&quot;, type=int, default=int(os.getenv(&quot;MAX_ITEMSET_SIZE&quot;, &quot;4&quot;))
    )
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(
        &quot;--register-mlflow&quot;,
        action=&quot;store_true&quot;,
        default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;,
    )
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_itemset_size=args.max_itemset_size,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-400491378')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-400491378"><code class="language-python">&quot;&quot;&quot;Production serving API for the recommendation engine (Apriori).&quot;&quot;&quot;

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

from recommendation_engine.data import PRODUCTS, load_training_data
from recommendation_engine.model import Apriori

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;RECOMMENDATION_METRICS_PORT&quot;, &quot;8004&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class RecommendRequest(BaseModel):
    &quot;&quot;&quot;Recommendation request with items in the basket.&quot;&quot;&quot;

    items: list[str] = Field(
        ..., min_length=1, max_length=50, description=&quot;Items in the user's basket&quot;
    )
    top_k: int = Field(5, ge=1, le=20, description=&quot;Number of recommendations to return&quot;)
    exclude_purchased: bool = Field(True, description=&quot;Exclude items already in the basket&quot;)


class RecommendResponse(BaseModel):
    &quot;&quot;&quot;Recommendation response.&quot;&quot;&quot;

    recommendations: list[dict]
    model_version: str


class RulesResponse(BaseModel):
    &quot;&quot;&quot;Association rules response.&quot;&quot;&quot;

    n_rules: int
    rules: list[dict]
    model_version: str


class RulesForItemResponse(BaseModel):
    &quot;&quot;&quot;Association rules for a specific item.&quot;&quot;&quot;

    item: str
    n_rules: int
    rules: list[dict]
    model_version: str


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

    n_transactions: int
    n_products: int
    n_frequent_itemsets: int
    n_rules: int
    min_support: float
    min_confidence: float
    min_lift: float
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: Apriori | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[str]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    &quot;&quot;&quot;Load model at startup and clean up at shutdown.&quot;&quot;&quot;
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;recommendation_engine&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=PRODUCTS,
        feature_types={p: &quot;binary&quot; for p in PRODUCTS},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;recommendation-engine&quot;,
        model_version=_model_version,
        model_type=&quot;association_rules&quot;,
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;recommendation-engine&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down recommendation-engine API&quot;)


def _load_model() -&gt; tuple[Apriori, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            rec_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;recommendation-engine&quot;]
            if rec_models:
                rec_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = rec_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;recommendation_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return Apriori.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;recommendation-engine&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;recommendation_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return Apriori.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;recommendation_model.npz&quot;
    if npz_path.exists():
        return Apriori.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/recommendation_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;recommendation_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return Apriori.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline Apriori model.&quot;)
    transactions = load_training_data(None)
    model = Apriori(min_support=0.05, min_confidence=0.5, min_lift=1.0, max_itemset_size=4)
    model.fit(transactions)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;recommendation-engine&quot; / _model_version / &quot;training_data.csv&quot;,
        MODEL_DIR / &quot;training_data.csv&quot;,
        Path(&quot;/app/artifacts/models/training_data.csv&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;training_data.csv&quot;,
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path, header=None)
                # Convert to one-hot matrix
                transactions = []
                for _, row in df.iterrows():
                    items = [
                        str(item).strip() for item in row.dropna().tolist() if str(item).strip()
                    ]
                    if items:
                        transactions.append(items)
                if transactions:
                    from recommendation_engine.data import transactions_to_onehot

                    X, _ = transactions_to_onehot(transactions, PRODUCTS)
                    return X
            except Exception as e:
                logger.warning(&quot;Could not read reference csv&quot;, path=str(csv_path), error=str(e))

    # Generate reference data
    transactions = load_training_data(None)
    from recommendation_engine.data import transactions_to_onehot

    X, _ = transactions_to_onehot(transactions, PRODUCTS)
    return X


# Create FastAPI app
app = FastAPI(
    title=&quot;Recommendation Engine API&quot;,
    description=&quot;Association Rule Learning with Apriori Algorithm for product recommendations&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;recommendation-engine-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;recommend&quot;: &quot;POST /recommend&quot;,
            &quot;rules&quot;: &quot;GET /rules&quot;,
            &quot;rules_for_item&quot;: &quot;GET /rules/{item}&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;drift&quot;: &quot;GET /drift&quot;,
            &quot;metrics&quot;: &quot;/metrics&quot;,
        },
    }


@app.get(&quot;/health&quot;)
def health_check():
    &quot;&quot;&quot;Kubernetes liveness/readiness probe.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {
        &quot;status&quot;: &quot;healthy&quot;,
        &quot;model_loaded&quot;: True,
        &quot;model_version&quot;: _model_version,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    &quot;&quot;&quot;Prometheus metrics endpoint.&quot;&quot;&quot;
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    &quot;&quot;&quot;Dynamically reload the model from disk/registry.&quot;&quot;&quot;
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;recommendation-engine&quot;,
                model_version=_model_version,
                model_type=&quot;association_rules&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;recommendation-engine&quot;, version=_model_version
        )
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    &quot;&quot;&quot;Check for data drift between reference and recent predictions.&quot;&quot;&quot;
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)

    if len(_recent_predictions) &lt; 10:
        return DriftResponse(
            total_features=len(PRODUCTS),
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    # Convert recent predictions to one-hot
    from recommendation_engine.data import transactions_to_onehot

    current, _ = transactions_to_onehot(_recent_predictions[-100:], PRODUCTS)
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])

    return DriftResponse(**summary)


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    &quot;&quot;&quot;Return model statistics.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return StatsResponse(
        n_transactions=_model.n_transactions,
        n_products=len(_model.products),
        n_frequent_itemsets=len(_model.frequent_itemsets),
        n_rules=len(_model.rules),
        min_support=_model.min_support,
        min_confidence=_model.min_confidence,
        min_lift=_model.min_lift,
        model_version=_model_version,
    )


@app.get(&quot;/rules&quot;, response_model=RulesResponse)
def get_rules(limit: int = 50):
    &quot;&quot;&quot;Return the top association rules.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    rules = [r.to_dict() for r in _model.rules[:limit]]
    return RulesResponse(
        n_rules=len(_model.rules),
        rules=rules,
        model_version=_model_version,
    )


@app.get(&quot;/rules/{item}&quot;, response_model=RulesForItemResponse)
def get_rules_for_item(item: str, limit: int = 10):
    &quot;&quot;&quot;Return association rules where the item appears in the consequent.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    rules = _model.get_rules_for_item(item, top_k=limit)
    return RulesForItemResponse(
        item=item,
        n_rules=len(rules),
        rules=rules,
        model_version=_model_version,
    )


@app.post(&quot;/recommend&quot;, response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    &quot;&quot;&quot;Generate product recommendations based on basket items.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate items exist in the product catalog
    unknown_items = [item for item in body.items if item not in _model.products]
    if unknown_items:
        raise HTTPException(
            status_code=422,
            detail=f&quot;Unknown items: {unknown_items}. Available products: {sorted(_model.products)}&quot;,
        )

    start = time.time()
    try:
        recommendations = _model.recommend(
            items=body.items,
            top_k=body.top_k,
            exclude_purchased=body.exclude_purchased,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(body.items)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return RecommendResponse(
            recommendations=recommendations,
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Recommendation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Recommendation failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-650404106')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-650404106"><code class="language-bash">uv run python -m recommendation_engine.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>recommendation-engine</strong></p>
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