# Image Captioning — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** with a dense image encoder for **image captioning**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Image (64 pixels) → Dense (hidden_dim) → RNN (hidden_dim, tanh) → Output (vocab_size, softmax)
```

- **Image Encoder**: Dense projection of pixel vector to a start token (one-hot-like)
- **RNN Decoder**: Many-to-many sequence model that generates word tokens
- **Output**: Softmax over vocabulary (predicts next word at each timestep)
- **Loss**: Cross-Entropy

## Quick Start

```bash
# Train
make train-image-captioning-rnn

# Serve API (port 8019)
make serve-image-captioning-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Generate caption from image pixels   |
| POST   | `/predict/bulk`   | Generate captions for multiple images|
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8019/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0.1, 0.2, 0.0, ... 64 values]}'
```

Response:

```json
{
  "caption_tokens": [0, 1, 3, 10, 2, 4, 5, 19],
  "caption": "start a object circle the bright blue end",
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```
