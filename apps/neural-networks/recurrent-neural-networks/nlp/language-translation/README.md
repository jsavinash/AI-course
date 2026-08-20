# Language Translation — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** for many-to-one **language translation**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, vocab_size) → Hidden (hidden_dim, tanh) → Output (vocab_size, softmax)
```

- **Input**: One-hot encoded word/token index sequences
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Softmax over vocabulary (predicts translated token at final timestep)
- **Loss**: Cross-Entropy

## Quick Start

```bash
# Train
make train-language-translation-rnn

# Serve API (port 8012)
make serve-language-translation-rnn
```

Or via uv:

```bash
uv run python -m language_translation.train --model-dir ./artifacts/models --n-iterations 300
MODEL_DIR=./artifacts/models uv run uvicorn language_translation.api:app --host 0.0.0.0 --port 8012
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Translate a token sequence           |
| POST   | `/predict/bulk`   | Translate multiple sequences         |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8012/predict \
  -H "Content-Type: application/json" \
  -d '{"tokens": [0, 1, 2, 3, 4, 5, 6, 7]}'
```

Response:

```json
{
  "translated_token": 20,
  "translated_word": "token_20",
  "confidence": 0.8932,
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```

## Data

Synthetic data is generated on-the-fly with configurable vocabulary size and sequence length.
