# Weather Forecasting — RNN (SimpleRNN)

A **SimpleRNN (Elman network)** for many-to-one **weather regression**, built from scratch with NumPy and trained via **Backpropagation Through Time (BPTT)**.

## Architecture

```
Input (seq_len, n_features) → Hidden (hidden_dim, tanh) → Output (n_features, linear)
```

- **Input**: Sequences of daily weather features (temperature, humidity, pressure, wind speed, precipitation)
- **Hidden**: Recurrent layer with tanh activation (internal memory)
- **Output**: Linear (next-day weather vector for all features)
- **Loss**: Mean Squared Error

## Quick Start

```bash
# Train
make train-weather-forecasting-rnn

# Serve API (port 8018)
make serve-weather-forecasting-rnn
```

## API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness/readiness probe             |
| POST   | `/predict`        | Forecast next-day weather            |
| POST   | `/predict/bulk`   | Forecast weather for multiple inputs |
| GET    | `/stats`          | Model statistics                     |
| GET    | `/drift`          | Data drift detection                 |
| GET    | `/metrics`        | Prometheus metrics                   |
| POST   | `/reload`         | Hot-reload model from disk           |

## Example

```bash
curl -X POST http://localhost:8018/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_sequences": [[20.0, 65.0, 1013.0, 8.0, 0.1], [20.5, 64.0, 1012.5, 8.5, 0.0], ...]}'
```

Response:

```json
{
  "predicted_weather": {
    "temperature": 21.5,
    "humidity": 63.2,
    "pressure": 1012.8,
    "wind_speed": 7.8,
    "precipitation": 0.05
  },
  "model_version": "1.0.0",
  "training_mode": "supervised"
}
```
