<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>retrieval-augmented-generation - AI App Documentation</title>
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
<p class="section-subtitle">Retrieval-Augmented Generation (RAG) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(y | x) = \sum_{z \in \mathcal{Z}} P(y | x, z) P(z | x)$$</div>
<div class="math-block">$$\text{sim}(q, d) = \frac{q^T d}{\|q\| \|d\|}$$</div>
<div class="math-block">$$\text{top-}k = \arg\max_{d_i \in \mathcal{D}} \text{sim}(q, d_i)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>RAG combines retrieval with generation. Given a query $q$, the retriever finds top-$k$ documents $z$ from a knowledge base. The generator conditions on both the query and retrieved context. This allows the model to access up-to-date or domain-specific information without retraining.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive retrieval pipeline; relevance score distribution; context vs generation attention alignment.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  Document
  Chunk
  TextChunker
  TFIDFEmbedding
  VectorDatabase
  Retriever
  PromptAugmenter
  SimpleRAGGenerator
  RAGGenerator
  RAGModel</pre>
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
<button class="copy-btn" onclick="copyCode('code-3567154620')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3567154620"><code class="language-python">from __future__ import annotations

&quot;&quot;&quot;Training / indexing pipeline for Retrieval-Augmented Generation (RAG).

Covers the RAG training steps:
- Creating External Data
- Chunking and embedding
- Indexing into Vector Database
&quot;&quot;&quot;

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from retrieval_augmented_generation.data import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_N_DOCS,
    DEFAULT_OVERLAP,
    load_knowledge_base,
)
from retrieval_augmented_generation.model import Document, RAGModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_docs: int = DEFAULT_N_DOCS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    top_k: int = 5,
    model_id: str = &quot;rag-v1&quot;,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict[str, Any]:
    logger.info(
        &quot;Loading knowledge base&quot;,
        n_docs=n_docs,
        data_path=str(data_path) if data_path else &quot;synthetic&quot;,
    )
    documents_data = load_knowledge_base(data_path=data_path, n_docs=n_docs)
    documents = [
        Document(doc_id=d[&quot;id&quot;], title=d.get(&quot;title&quot;, &quot;&quot;), content=d.get(&quot;content&quot;, &quot;&quot;))
        for d in documents_data
    ]

    logger.info(&quot;Initializing RAG model&quot;)
    model = RAGModel(
        model_id=model_id,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        random_seed=random_seed,
    )

    logger.info(&quot;Indexing documents&quot;)
    model.index_documents(documents)

    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    chunks_path = model_dir / f&quot;rag_chunks_v{model_version}.json&quot;
    with open(chunks_path, &quot;w&quot;, encoding=&quot;utf-8&quot;) as f:
        json.dump(
            [
                {
                    &quot;chunk_id&quot;: c.chunk_id,
                    &quot;doc_id&quot;: c.doc_id,
                    &quot;title&quot;: c.title,
                    &quot;text&quot;: c.text,
                }
                for c in model.chunks
            ],
            f,
            indent=2,
        )
    artifacts[&quot;chunks&quot;] = chunks_path

    vocab_path = model_dir / f&quot;rag_vocab_v{model_version}.json&quot;
    with open(vocab_path, &quot;w&quot;, encoding=&quot;utf-8&quot;) as f:
        json.dump(model.embedding_model.vocab, f, indent=2)
    artifacts[&quot;vocab&quot;] = vocab_path

    stats = model.get_stats()
    logger.info(&quot;RAG indexing complete&quot;, stats=stats)

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;rag&quot;,
        model_version=model_version,
        model_type=&quot;retrieval-augmented-generation&quot;,
        metrics={
            &quot;documents_indexed&quot;: float(stats[&quot;documents_indexed&quot;]),
            &quot;chunks_indexed&quot;: float(stats[&quot;chunks_indexed&quot;]),
            &quot;vocab_size&quot;: float(stats[&quot;vocab_size&quot;]),
            &quot;vector_dim&quot;: float(stats[&quot;vector_dim&quot;]),
        },
        parameters={
            &quot;model_id&quot;: model_id,
            &quot;chunk_size&quot;: chunk_size,
            &quot;overlap&quot;: overlap,
            &quot;top_k&quot;: top_k,
            &quot;n_docs&quot;: n_docs,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts=artifacts,
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;rag&quot;, &quot;model_type&quot;: &quot;RetrievalAugmentedGeneration&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;rag&quot;,
            model_version=model_version,
            metrics={
                &quot;documents_indexed&quot;: float(stats[&quot;documents_indexed&quot;]),
                &quot;chunks_indexed&quot;: float(stats[&quot;chunks_indexed&quot;]),
            },
            params={&quot;model_id&quot;: model_id, &quot;chunk_size&quot;: chunk_size, &quot;top_k&quot;: top_k},
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            tags={&quot;model_type&quot;: &quot;rag&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return stats


def main() -&gt; None:
    parser = argparse.ArgumentParser(description=&quot;Train / index RAG model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-docs&quot;, type=int, default=int(os.getenv(&quot;N_DOCS&quot;, str(DEFAULT_N_DOCS))))
    parser.add_argument(
        &quot;--chunk-size&quot;, type=int, default=int(os.getenv(&quot;CHUNK_SIZE&quot;, str(DEFAULT_CHUNK_SIZE)))
    )
    parser.add_argument(
        &quot;--overlap&quot;, type=int, default=int(os.getenv(&quot;OVERLAP&quot;, str(DEFAULT_OVERLAP)))
    )
    parser.add_argument(&quot;--top-k&quot;, type=int, default=int(os.getenv(&quot;TOP_K&quot;, &quot;5&quot;)))
    parser.add_argument(&quot;--model-id&quot;, type=str, default=os.getenv(&quot;MODEL_ID&quot;, &quot;rag-v1&quot;))
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

    stats = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_docs=args.n_docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        model_id=args.model_id,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Indexing finished&quot;, stats=stats, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-644126295')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-644126295"><code class="language-python">from __future__ import annotations

&quot;&quot;&quot;Serving API for Retrieval-Augmented Generation (RAG).&quot;&quot;&quot;

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

from retrieval_augmented_generation.data import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_N_DOCS,
    DEFAULT_OVERLAP,
    generate_synthetic_knowledge_base,
    load_knowledge_base,
)
from retrieval_augmented_generation.model import Document, RAGModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
RAG_METRICS_PORT = int(os.getenv(&quot;RAG_METRICS_PORT&quot;, &quot;9023&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description=&quot;User question to answer via RAG&quot;)


class QueryResponse(BaseModel):
    query: str
    prompt: str
    answer: str
    retrieved_chunks: list[dict[str, Any]]
    model_version: str


class IndexRequest(BaseModel):
    documents: list[dict[str, str]]
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=50, le=1000)
    overlap: int = Field(default=DEFAULT_OVERLAP, ge=0, le=200)


class IndexResponse(BaseModel):
    indexed_docs: int
    indexed_chunks: int
    model_version: str


class StatsResponse(BaseModel):
    model_id: str
    documents_indexed: int
    chunks_indexed: int
    vocab_size: int
    vector_dim: int
    top_k: int
    chunk_size: int
    overlap: int
    model_version: str


class RefreshResponse(BaseModel):
    status: str
    total_chunks: int


_model: RAGModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_queries: np.ndarray | None = None
_recent_queries: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_queries

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;rag&quot;, port=RAG_METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f&quot;chunk_feat_{i}&quot; for i in range(64)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;rag&quot;,
        model_version=_model_version,
        model_type=&quot;retrieval-augmented-generation&quot;,
    )

    _reference_queries = _load_reference_queries()
    logger.info(&quot;RAG model loaded&quot;, model=&quot;rag&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down RAG API&quot;)


def _load_model() -&gt; tuple[RAGModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            rag_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;rag&quot;]
            if rag_models:
                rag_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = rag_models[0]
                model_version = latest[&quot;model_version&quot;]
                model = RAGModel(model_id=latest.get(&quot;parameters&quot;, {}).get(&quot;model_id&quot;, &quot;rag&quot;))
                documents_data = load_knowledge_base()
                documents = [
                    Document(doc_id=d[&quot;id&quot;], title=d.get(&quot;title&quot;, &quot;&quot;), content=d.get(&quot;content&quot;, &quot;&quot;))
                    for d in documents_data
                ]
                model.index_documents(documents)
                return model, model_version
        else:
            model_dir = MODEL_DIR / &quot;rag&quot; / MODEL_VERSION
            if model_dir.exists():
                model = RAGModel(model_id=f&quot;rag-{MODEL_VERSION}&quot;)
                documents_data = load_knowledge_base()
                documents = [
                    Document(doc_id=d[&quot;id&quot;], title=d.get(&quot;title&quot;, &quot;&quot;), content=d.get(&quot;content&quot;, &quot;&quot;))
                    for d in documents_data
                ]
                model.index_documents(documents)
                return model, MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    logger.warning(&quot;No pre-existing index found. Initializing baseline RAG model.&quot;)
    model = RAGModel(model_id=&quot;rag-baseline&quot;)
    documents = [
        Document(doc_id=d[&quot;id&quot;], title=d.get(&quot;title&quot;, &quot;&quot;), content=d.get(&quot;content&quot;, &quot;&quot;))
        for d in generate_synthetic_knowledge_base(n_docs=DEFAULT_N_DOCS)
    ]
    model.index_documents(documents)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_queries() -&gt; np.ndarray | None:
    try:
        docs = generate_synthetic_knowledge_base(n_docs=DEFAULT_N_DOCS)
        texts = [d[&quot;content&quot;] for d in docs]
        return np.array([[len(t.split())] for t in texts])
    except Exception:
        return None


app = FastAPI(
    title=&quot;Retrieval-Augmented Generation API&quot;,
    description=&quot;RAG system with chunking, TF-IDF embeddings, vector search, and grounded generation&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;rag-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;query&quot;: &quot;POST /query&quot;,
            &quot;index&quot;: &quot;POST /index&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;refresh&quot;: &quot;POST /refresh&quot;,
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
        &quot;documents_indexed&quot;: _model.get_stats().get(&quot;documents_indexed&quot;, 0) if _model else 0,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/query&quot;, response_model=QueryResponse)
def query_rag(body: QueryRequest):
    &quot;&quot;&quot;Answer a question using the RAG pipeline.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        result = _model.query(body.query)

        response = QueryResponse(
            query=result[&quot;query&quot;],
            prompt=result[&quot;prompt&quot;],
            answer=result[&quot;answer&quot;],
            retrieved_chunks=result[&quot;retrieved_chunks&quot;],
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_queries.append([float(len(body.query.split()))])
        if len(_recent_queries) &gt; 1000:
            _recent_queries.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;query&quot;)
        logger.exception(&quot;RAG query failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;RAG query failed&quot;) from e


@app.post(&quot;/index&quot;, response_model=IndexResponse)
def index_documents(body: IndexRequest):
    &quot;&quot;&quot;Index new documents into the RAG knowledge base.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        _model.chunk_size = body.chunk_size
        _model.overlap = body.overlap
        if _model.chunker is None:
            _model._init_components()
        for doc_data in body.documents:
            _model.add_document(
                Document(
                    doc_id=doc_data.get(&quot;id&quot;, f&quot;doc_{len(_model.documents)}&quot;),
                    title=doc_data.get(&quot;title&quot;, &quot;&quot;),
                    content=doc_data.get(&quot;content&quot;, &quot;&quot;),
                )
            )

        response = IndexResponse(
            indexed_docs=_model.get_stats()[&quot;documents_indexed&quot;],
            indexed_chunks=_model.get_stats()[&quot;chunks_indexed&quot;],
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)
        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;index&quot;)
        logger.exception(&quot;Document indexing failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Document indexing failed&quot;) from e


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.get_stats()
    return StatsResponse(
        model_id=_model.model_id if _model else &quot;unknown&quot;,
        documents_indexed=info.get(&quot;documents_indexed&quot;, 0),
        chunks_indexed=info.get(&quot;chunks_indexed&quot;, 0),
        vocab_size=info.get(&quot;vocab_size&quot;, 0),
        vector_dim=info.get(&quot;vector_dim&quot;, 0),
        top_k=info.get(&quot;top_k&quot;, 5),
        chunk_size=info.get(&quot;chunk_size&quot;, DEFAULT_CHUNK_SIZE),
        overlap=info.get(&quot;overlap&quot;, DEFAULT_OVERLAP),
        model_version=_model_version,
    )


@app.post(&quot;/refresh&quot;, response_model=RefreshResponse)
def refresh_index():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    result = _model.refresh()
    return RefreshResponse(status=result[&quot;status&quot;], total_chunks=result[&quot;total_chunks&quot;])</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-4280448814')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4280448814"><code class="language-bash">uv run python -m retrieval_augmented_generation.train --model-dir ./artifacts/models</code></pre>
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
<li><a href="../text-generation/README.md">text-generation</a></li>
<li><a href="../video-generation/README.md">video-generation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>retrieval-augmented-generation</strong></p>
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