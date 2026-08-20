"""Retrieval-Augmented Generation (RAG) from scratch."""

from retrieval_augmented_generation.data import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    generate_synthetic_knowledge_base,
    load_knowledge_base,
    preprocess_text,
    save_knowledge_base,
)
from retrieval_augmented_generation.model import (
    Chunk,
    Document,
    PromptAugmenter,
    RAGModel,
    Retriever,
    SimpleRAGGenerator,
    TextChunker,
    TFIDFEmbedding,
    VectorDatabase,
)

__all__ = [
    "Document",
    "Chunk",
    "TextChunker",
    "TFIDFEmbedding",
    "VectorDatabase",
    "Retriever",
    "PromptAugmenter",
    "SimpleRAGGenerator",
    "RAGModel",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "generate_synthetic_knowledge_base",
    "load_knowledge_base",
    "save_knowledge_base",
    "preprocess_text",
]
