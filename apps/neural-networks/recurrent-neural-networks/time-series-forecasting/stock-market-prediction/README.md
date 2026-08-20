# Stock Market Prediction — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** for many-to-one **stock price regression**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, n_features) → Hidden (hidden_dim, tanh) → Output (1, linear)
```

- **Input**: Sequences of normalized financial features (open, high, low, close, volume)
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Linear (predicted future price)
- **Loss**: Mean Squared Error

## Quick Start

```bash
# Train
make train-stock-market-prediction-rnn

# Serve API (port 8017)
make serve-stock-market-prediction-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Predict stock price from features    |
| POST   | `/predict/bulk`   | Predict prices for multiple sequences|
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8017/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_sequences": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.12, 0.22, 0.32, 0.42, 0.52], ...]}'
```

Response:

```json
{
  "predicted_price": 215.75,
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```
