from __future__ import annotations

"""Data loading and preprocessing for Retrieval-Augmented Generation (RAG).

Covers:
- External Knowledge Source
- Text Chunking and Preprocessing
"""

from pathlib import Path

import numpy as np

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40
DEFAULT_N_DOCS = 50
DEFAULT_VOCAB_SIZE = 1000


def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = text.split()
    tokens = [t.strip(".,!?;:\"'()[]{}") for t in tokens if t.strip(".,!?;:\"'()[]{}")]
    return " ".join(tokens)


def _chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def generate_synthetic_knowledge_base(
    n_docs: int = DEFAULT_N_DOCS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    random_seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    topics = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "transformers",
        "attention mechanism",
        "retrieval augmented generation",
        "vector database",
        "embeddings",
        "fine tuning",
        "prompt engineering",
        "few shot learning",
        "zero shot learning",
        "diffusion models",
        "generative adversarial networks",
        "transfer learning",
        "self supervised learning",
        "supervised learning",
        "unsupervised learning",
        "semi supervised learning",
        "convolutional neural networks",
        "recurrent neural networks",
        "long short term memory",
        "gated recurrent unit",
        "word embeddings",
        "bert",
        "gpt",
        "t5",
        "rag pipeline",
        "knowledge graphs",
        "semantic search",
        "cosine similarity",
        "tf idf",
        "bag of words",
        "tokenization",
        "lemmatization",
        "stemming",
        "data preprocessing",
        "feature extraction",
        "model deployment",
        "mlops",
        "hyperparameter tuning",
        "regularization",
        "dropout",
        "batch normalization",
        "optimization algorithms",
        "gradient descent",
        "adam optimizer",
        "loss functions",
        "classification",
        "regression",
        "clustering",
        "dimensionality reduction",
        "principal component analysis",
        "autoencoders",
        "attention is all you need",
    ]
    templates = [
        "{topic} is a fundamental area in artificial intelligence. It involves building systems that can learn patterns from data and make predictions or decisions. Recent advances have significantly improved performance across many benchmarks.",
        "In {topic}, researchers focus on developing algorithms that can automatically improve through experience. Applications include image recognition, speech processing, and autonomous driving.",
        "The study of {topic} combines principles from statistics, optimization, and computer science. Modern approaches often use neural networks with millions of parameters trained on large datasets.",
        "{topic} has revolutionized how we approach complex problems in AI. By leveraging large-scale data and compute, models can now achieve human-level performance on specific tasks.",
        "Understanding {topic} requires knowledge of linear algebra, probability, and calculus. Key concepts include forward propagation, backpropagation, and gradient-based optimization.",
        "Practical {topic} involves data collection, preprocessing, model selection, training, evaluation, and deployment. Each step requires careful consideration of trade-offs between accuracy and efficiency.",
        "{topic} models are typically trained using stochastic gradient descent or its variants. Regularization techniques like dropout and weight decay help prevent overfitting.",
        "Evaluation of {topic} systems uses metrics such as accuracy, precision, recall, F1 score, and perplexity depending on the task. Proper validation strategies ensure generalization.",
        "Scaling {topic} often requires distributed computing and specialized hardware like GPUs or TPUs. Efficient architectures like transformers have enabled training on billions of parameters.",
        "Future directions in {topic} include improving interpretability, reducing data requirements, and building more robust systems that can handle out-of-distribution inputs.",
    ]
    documents = []
    for i in range(n_docs):
        topic = topics[rng.integers(0, len(topics))]
        template = templates[rng.integers(0, len(templates))]
        content = template.format(topic=topic)
        docs = []
        for _ in range(rng.integers(2, 6)):
            t = topics[rng.integers(0, len(topics))]
            tpl = templates[rng.integers(0, len(templates))]
            docs.append(tpl.format(topic=t))
        content = " ".join(docs)
        documents.append({"id": f"doc_{i}", "title": topic, "content": content})
    return documents


def build_chunks(
    documents: list[dict], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    chunks = []
    for doc in documents:
        text_chunks = _chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_chunk_{idx}",
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "text": chunk_text,
                }
            )
    return chunks


def load_knowledge_base(data_path: Path | None = None, n_docs: int = DEFAULT_N_DOCS) -> list[dict]:
    if data_path and Path(data_path).exists():
        import json

        with open(data_path, encoding="utf-8") as f:
            return json.load(f)
    return generate_synthetic_knowledge_base(n_docs=n_docs)


def save_knowledge_base(documents: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
