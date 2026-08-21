from __future__ import annotations

"""Training / indexing pipeline for Retrieval-Augmented Generation (RAG).

Covers the RAG training steps:
- Creating External Data
- Chunking and embedding
- Indexing into Vector Database
"""

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
    model_id: str = "rag-v1",
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict[str, Any]:
    logger.info(
        "Loading knowledge base",
        n_docs=n_docs,
        data_path=str(data_path) if data_path else "synthetic",
    )
    documents_data = load_knowledge_base(data_path=data_path, n_docs=n_docs)
    documents = [
        Document(doc_id=d["id"], title=d.get("title", ""), content=d.get("content", ""))
        for d in documents_data
    ]

    logger.info("Initializing RAG model")
    model = RAGModel(
        model_id=model_id,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        random_seed=random_seed,
    )

    logger.info("Indexing documents")
    model.index_documents(documents)

    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    chunks_path = model_dir / f"rag_chunks_v{model_version}.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text,
                }
                for c in model.chunks
            ],
            f,
            indent=2,
        )
    artifacts["chunks"] = chunks_path

    vocab_path = model_dir / f"rag_vocab_v{model_version}.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(model.embedding_model.vocab, f, indent=2)
    artifacts["vocab"] = vocab_path

    stats = model.get_stats()
    logger.info("RAG indexing complete", stats=stats)

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="rag",
        model_version=model_version,
        model_type="retrieval-augmented-generation",
        metrics={
            "documents_indexed": float(stats["documents_indexed"]),
            "chunks_indexed": float(stats["chunks_indexed"]),
            "vocab_size": float(stats["vocab_size"]),
            "vector_dim": float(stats["vector_dim"]),
        },
        parameters={
            "model_id": model_id,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "n_docs": n_docs,
            "random_seed": random_seed,
        },
        artifacts=artifacts,
        tags={"framework": "numpy", "task": "rag", "model_type": "RetrievalAugmentedGeneration"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="rag",
            model_version=model_version,
            metrics={
                "documents_indexed": float(stats["documents_indexed"]),
                "chunks_indexed": float(stats["chunks_indexed"]),
            },
            params={"model_id": model_id, "chunk_size": chunk_size, "top_k": top_k},
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            tags={"model_type": "rag", "framework": "numpy"},
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / index RAG model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-docs", type=int, default=int(os.getenv("N_DOCS", str(DEFAULT_N_DOCS))))
    parser.add_argument(
        "--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    )
    parser.add_argument(
        "--overlap", type=int, default=int(os.getenv("OVERLAP", str(DEFAULT_OVERLAP)))
    )
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "rag-v1"))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
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
    logger.info("Indexing finished", stats=stats, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
