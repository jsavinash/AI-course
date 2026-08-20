# Transformers Language Modeling

Transformer LLM that uses attention mechanisms to capture relationships between inputs, processes entire sequences at once, and overcomes RNN/LSTM limitations.

## Network Type
Transformer (Self-Attention)

## Architecture
- **Token Embeddings**: vocab_size -> d_model
- **Positional Encoding**: Sinusoidal sin/cos positional encodings
- **Self-Attention**: Q, K, V vectors with scaled dot-product attention
- **Multi-Head Attention**: h parallel attention heads with different pattern capture
- **Encoder-Decoder Attention**: Cross-attention between encoder and decoder
- **Feed-Forward Networks**: Position-wise FFN with GELU activation
- **Add & Norm**: Residual connections + Layer Normalization
- **Softmax Output**: Tied embedding weights for next-token prediction

## Training
```bash
transformers_language_modeling-train --model-dir ./artifacts/models --n-samples 500 --n-iterations 100
```

## Serving API
```bash
uvicorn transformers_language_modeling.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Next-token prediction (input tokens -> generated tokens)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format
- `tokens`: list of token indices (max 64)
- `max_len`: max generation length (1-32)

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
