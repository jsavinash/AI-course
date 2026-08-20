# Text Generation — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** language model for **character-level text generation**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, vocab_size) → Hidden (hidden_dim, tanh) → Output (vocab_size, softmax)
```

- **Input**: One-hot encoded character index sequences
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Softmax over vocabulary (predicts next character at each timestep)
- **Loss**: Cross-Entropy (many-to-many language modeling)
- **Training mode**: self-supervised (next-token prediction)

## Quick Start

```bash
# Train
make train-text-generation-rnn

# Serve API (port 8014)
make serve-text-generation-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Generate text from a seed sequence   |
| POST   | `/predict/bulk`   | Generate text from multiple seeds    |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8014/predict \
  -H "Content-Type: application/json" \
  -d '{"tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "n_generate": 10}'
```

Response:

```json
{
  "generated_tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
  "generated_text": "abcdefghij...",
  "next_token": 20,
  "next_token_probability": 0.15,
  "perplexity": 12.34,
  "model_version": "1.0.0",
  "training_mode": "self-supervised"
}
```
