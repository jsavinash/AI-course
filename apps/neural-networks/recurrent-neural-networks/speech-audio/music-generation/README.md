# Music Generation — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** language model for **musical note sequence generation**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, vocab_size) → Hidden (hidden_dim, tanh) → Output (vocab_size, softmax)
```

- **Input**: One-hot encoded note index sequences (MIDI-style)
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Softmax over vocabulary (predicts next note at each timestep)
- **Loss**: Cross-Entropy (many-to-many language modeling)
- **Training mode**: self-supervised (next-token prediction)

## Quick Start

```bash
# Train
make train-music-generation-rnn

# Serve API (port 8016)
make serve-music-generation-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Generate music from a seed sequence  |
| POST   | `/predict/bulk`   | Generate from multiple seeds         |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8016/predict \
  -H "Content-Type: application/json" \
  -d '{"seed_notes": [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 20, 22, 24, 25, 26, 28, 29, 31], "n_generate": 10}'
```

Response:

```json
{
  "generated_notes": [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 20, 22, 24, 25, 26, 28, 29, 31, 32, 34, 36, ...],
  "generated_notes_str": ["C4", "D4", "E4", "F4", ...],
  "perplexity": 8.56,
  "n_generated": 10,
  "model_version": "1.0.0",
  "training_mode": "self-supervised"
}
```
