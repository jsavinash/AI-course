# Sentiment Analysis — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** for many-to-one **binary sentiment classification**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, vocab_size) → Hidden (hidden_dim, tanh) → Output (1, sigmoid)
```

- **Input**: One-hot encoded word/token index sequences
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Sigmoid (positive/negative probability)
- **Loss**: Binary Cross-Entropy

## Quick Start

```bash
# Train
make train-sentiment-analysis-rnn

# Serve API (port 8013)
make serve-sentiment-analysis-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Classify sentiment of a token sequence |
| POST   | `/predict/bulk`   | Classify multiple sequences          |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8013/predict \
  -H "Content-Type: application/json" \
  -d '{"tokens": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]}'
```

Response:

```json
{
  "sentiment": "positive",
  "confidence": 0.8765,
  "positive_probability": 0.8765,
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```
