# Large Language Model (LLM)

Advanced AI system built on Transformer architecture designed to process, understand, and generate human-like text. LLMs learn patterns, grammar, and context from massive text corpora.

## Network Type
LLM (Large Language Model) - Transformer Decoder

## Architecture (from GeeksforGeeks article)
- **Input Embeddings**: token indices → dense vectors
- **Positional Encoding**: sinusoidal sin/cos adds sequence/order information
- **Self-Attention**: Q/K/V with scaled dot-product to understand word relationships
- **Multi-Head Attention**: parallel heads for diverse context reasoning
- **Feed-Forward Layers**: position-wise FFN with GELU activation
- **Residual + LayerNorm**: stable training via residual connections and normalization
- **Decoding**: autoregressive token-by-token generation with causal masking

## Features
- Zero-shot capability (no task-specific training)
- Few-shot inference (learn from examples in prompt)
- Temperature sampling (controls output randomness)
- Top-k sampling (limits to most likely tokens)
- Next-token prediction objective

## Training
```bash
large_language_model-train --model-dir ./artifacts/models --n-samples 500 --n-iterations 100
```

## Serving API
```bash
uvicorn large_language_model.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Next-token prediction with temperature/top-k sampling
- `GET /stats` - Model statistics
- `GET /metrics` - Prometheus metrics

## Input Format
- `tokens`: list of token indices (max 64)
- `max_len`: generation length (1-32)
- `temperature`: randomness control (0.1-2.0)
- `top_k`: sampling diversity limit (1-100)

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
