# Speech Recognition — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** for many-to-one **speech-to-text classification**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, n_mfcc_features) → Hidden (hidden_dim, tanh) → Output (n_words, softmax)
```

- **Input**: Sequences of acoustic feature vectors (simulated MFCC-like frames)
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Softmax over vocabulary of recognizable words
- **Loss**: Cross-Entropy

## Quick Start

```bash
# Train
make train-speech-recognition-rnn

# Serve API (port 8015)
make serve-speech-recognition-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Recognize speech from audio features |
| POST   | `/predict/bulk`   | Recognize multiple audio sequences   |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8015/predict \
  -H "Content-Type: application/json" \
  -d '{"audio_features": [[0.1, 0.2, ... 16 values], [0.05, 0.1, ...], ...]}'
```

Response:

```json
{
  "word": "hello",
  "word_index": 0,
  "confidence": 0.92,
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```
