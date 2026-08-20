# Transformer Language Modeling

Next-token prediction using self-attention mechanisms that process data sequences in parallel, powering modern large language models.

## Network Type
Transformer (Attention)

## Architecture
- Input: 16 token IDs from vocabulary of 100
- Token Embedding + Positional Encoding
- Multi-Head Self-Attention (4 heads, 32-dim model)
- Feed-Forward layers (64 hidden units)
- Output: vocabulary logits (softmax)

## Training
```bash
transformer_language_modeling-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn transformer_language_modeling.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Single prediction (16 token IDs)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[0-99] token vocabulary

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
