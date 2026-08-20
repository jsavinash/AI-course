"""Retrieval-Augmented Generation (RAG) model from scratch using NumPy.

Covers all major RAG topics from the GeeksforGeeks article:
- External Knowledge Source
- Text Chunking and Preprocessing
- Embedding Model (TF-IDF)
- Vector Database
- Query Encoder
- Retriever
- Prompt Augmentation Layer
- LLM / Generator
- Updater (optional refresh support)

Architecture:
    1. TextChunker: splits documents into overlapping chunks
    2. TFIDFEmbedding: builds vocabulary and encodes text as TF-IDF vectors
    3. VectorDatabase: stores chunk embeddings and retrieves by cosine similarity
    4. Retriever: queries the vector DB to find top-k relevant chunks
    5. PromptAugmenter: fuses the user query with retrieved context
    6. SimpleRAGGenerator: produces a grounded response from augmented prompt
    7. RAGModel: end-to-end orchestrator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retrieval_augmented_generation.data import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text Chunking and Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class TextChunker:
    chunk_size: int = 200
    overlap: int = 40
    random_seed: int = 42

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = document.content.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_text = " ".join(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{idx}",
                        doc_id=document.doc_id,
                        title=document.title,
                        text=chunk_text,
                        metadata={"doc_metadata": document.metadata},
                    )
                )
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# ---------------------------------------------------------------------------
# Embedding Model (TF-IDF)
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
    denom = norm_a * norm_b.T
    denom = np.where(denom == 0, 1e-12, denom)
    return (a @ b.T) / denom


@dataclass
class TFIDFEmbedding:
    vocab: dict[str, int] = field(default_factory=dict, repr=False)
    idf: np.ndarray | None = None
    vocab_size: int = 0
    random_seed: int = 42

    def fit(self, texts: list[str]) -> None:
        doc_freq: dict[str, int] = {}
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1

        n_docs = len(texts)
        self.vocab = {word: idx for idx, word in enumerate(doc_freq.keys())}
        self.vocab_size = len(self.vocab)
        self.idf = np.zeros(self.vocab_size, dtype=np.float64)
        for word, idx in self.vocab.items():
            self.idf[idx] = np.log((1 + n_docs) / (1 + doc_freq[word])) + 1.0

    def _text_to_vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        tf = np.zeros(self.vocab_size, dtype=np.float64)
        for word in words:
            idx = self.vocab.get(word)
            if idx is not None:
                tf[idx] += 1.0
        if tf.sum() > 0:
            tf = tf / tf.sum()
        return tf * self.idf

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._text_to_vector(t) for t in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._text_to_vector(query).reshape(1, -1)


# ---------------------------------------------------------------------------
# Vector Database
# ---------------------------------------------------------------------------


@dataclass
class VectorDatabase:
    embeddings: np.ndarray | None = None
    chunks: list[Chunk] = field(default_factory=list, repr=False)
    dim: int = 0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def index(self, chunks: list[Chunk]) -> None:
        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            raise ValueError("No chunks with embeddings found to index.")
        self.chunks = valid
        self.embeddings = np.stack([c.embedding for c in valid])
        self.dim = self.embeddings.shape[1]
        self._cache["indexed_at"] = len(valid)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        sims = _cosine_similarity(query_vector, self.embeddings).flatten()
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.chunks[int(i)], float(sims[int(i)])) for i in top_indices]

    def add(self, chunk: Chunk) -> None:
        if chunk.embedding is None:
            raise ValueError("Chunk must have an embedding before adding to the database.")
        self.chunks.append(chunk)
        if self.embeddings is None:
            self.embeddings = chunk.embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, chunk.embedding.reshape(1, -1)])
        self.dim = self.embeddings.shape[1]

    def refresh(self) -> None:
        self._cache["refreshed_at"] = len(self.chunks)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class Retriever:
    vector_db: VectorDatabase
    top_k: int = 5
    score_threshold: float = 0.0

    def retrieve(self, query_vector: np.ndarray) -> list[tuple[Chunk, float]]:
        results = self.vector_db.search(query_vector, top_k=self.top_k)
        return [(chunk, score) for chunk, score in results if score >= self.score_threshold]


# ---------------------------------------------------------------------------
# Prompt Augmentation Layer
# ---------------------------------------------------------------------------


@dataclass
class PromptAugmenter:
    max_context_chunks: int = 3
    context_separator: str = "\n\n---\n\n"

    def build_prompt(self, query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
        if not retrieved_chunks:
            return f"Question: {query}\n\nAnswer based on your knowledge:"
        context_parts = []
        for chunk, score in retrieved_chunks[: self.max_context_chunks]:
            context_parts.append(f"[Source: {chunk.title}] (relevance: {score:.2f})\n{chunk.text}")
        context = self.context_separator.join(context_parts)
        prompt = (
            f"Use the following context to answer the question accurately. "
            f"If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return prompt


# ---------------------------------------------------------------------------
# Generator (LLM simulation)
# ---------------------------------------------------------------------------


@dataclass
class SimpleRAGGenerator:
    temperature: float = 0.7
    max_response_tokens: int = 128
    random_seed: int = 42

    def generate(self, prompt: str, context: str) -> str:
        sentences = [s.strip() for s in context.replace("\n", " ").split(". ") if s.strip()]
        query_part = (
            prompt.split("Question:")[-1].split("Answer:")[0].strip()
            if "Question:" in prompt
            else prompt
        )
        query_words = set(query_part.lower().split())
        scored: list[tuple[float, str]] = []
        for sentence in sentences:
            s_words = set(sentence.lower().split())
            overlap = len(query_words & s_words)
            scored.append((overlap, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] == 0:
            return "I do not know based on the provided context."
        best = scored[0][1]
        if not best.endswith("."):
            best += "."
        return best


@dataclass
class RAGGenerator:
    generator: Any = None

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = SimpleRAGGenerator()

    def generate(self, prompt: str, context: str) -> str:
        return self.generator.generate(prompt, context)


# ---------------------------------------------------------------------------
# End-to-End RAG Model
# ---------------------------------------------------------------------------


@dataclass
class RAGModel:
    model_id: str = "rag"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    embedding_dim: int = 64
    top_k: int = 5
    temperature: float = 0.7
    random_seed: int = 42

    chunker: TextChunker | None = None
    embedding_model: TFIDFEmbedding | None = None
    vector_db: VectorDatabase = field(default_factory=VectorDatabase)
    retriever: Retriever | None = None
    augmenter: PromptAugmenter | None = None
    generator: RAGGenerator | None = None
    documents: list[Document] = field(default_factory=list, repr=False)
    chunks: list[Chunk] = field(default_factory=list, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def _init_components(self) -> None:
        self.chunker = TextChunker(
            chunk_size=self.chunk_size, overlap=self.overlap, random_seed=self.random_seed
        )
        self.embedding_model = TFIDFEmbedding(random_seed=self.random_seed)
        self.retriever = Retriever(vector_db=self.vector_db, top_k=self.top_k)
        self.augmenter = PromptAugmenter()
        self.generator = RAGGenerator()

    def index_documents(self, documents: list[Document]) -> None:
        if self.chunker is None:
            self._init_components()
        self.documents = documents
        self.chunks = self.chunker.chunk_documents(documents)
        texts = [c.text for c in self.chunks]
        self.embedding_model.fit(texts)
        embeddings = self.embedding_model.encode(texts)
        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb
        self.vector_db.index(self.chunks)
        self._cache["indexed_docs"] = len(documents)
        self._cache["indexed_chunks"] = len(self.chunks)

    def query(self, user_query: str) -> dict[str, Any]:
        if self.embedding_model is None or self.retriever is None:
            raise RuntimeError("Model not initialized. Call index_documents first.")
        query_vector = self.embedding_model.encode_query(user_query)
        retrieved = self.retriever.retrieve(query_vector)
        prompt = self.augmenter.build_prompt(user_query, retrieved)
        context = "\n".join([c.text for c, _ in retrieved])
        answer = self.generator.generate(prompt, context)
        return {
            "query": user_query,
            "prompt": prompt,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "text": c.text[:200] + "...",
                    "score": round(s, 4),
                }
                for c, s in retrieved
            ],
            "answer": answer,
        }

    def add_document(self, document: Document) -> None:
        if self.chunker is None:
            self._init_components()
        self.documents.append(document)
        new_chunks = self.chunker.chunk_document(document)
        texts = [c.text for c in new_chunks]
        if self.embedding_model is None or self.embedding_model.vocab_size == 0:
            self.embedding_model.fit(texts)
        else:
            self.embedding_model.fit([c.text for c in self.chunks] + texts)
        embeddings = self.embedding_model.encode(texts)
        for chunk, emb in zip(new_chunks, embeddings):
            chunk.embedding = emb
            self.chunks.append(chunk)
            self.vector_db.add(chunk)
        self._cache["indexed_docs"] = len(self.documents)
        self._cache["indexed_chunks"] = len(self.chunks)

    def refresh(self) -> dict[str, Any]:
        self.vector_db.refresh()
        return {"status": "refreshed", "total_chunks": len(self.chunks)}

    def get_stats(self) -> dict[str, Any]:
        return {
            "documents_indexed": len(self.documents),
            "chunks_indexed": len(self.chunks),
            "vocab_size": self.embedding_model.vocab_size if self.embedding_model else 0,
            "vector_dim": self.vector_db.dim,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }
