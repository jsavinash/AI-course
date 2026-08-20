# Self-Supervised Monitoring (Self-Supervised Learning)

Production-ready denoising autoencoder for server monitoring anomaly detection using self-supervised learning.

## Learning Type: Self-Supervised Learning

This example demonstrates **self-supervised learning** — the model generates its own labels from the data structure itself, requiring no human-annotated labels. The supervision signal comes from the data's intrinsic patterns.

## Problem

Detect anomalies in server monitoring metrics without any labeled anomaly examples. The model learns what "normal" server behavior looks like and flags deviations as potential issues.

## How It's Self-Supervised

The model is trained as a **denoising autoencoder**:

1. **Input**: Clean server metrics data (only normal data is used for training)
2. **Corruption**: A portion of features are randomly zeroed out and Gaussian noise is added
3. **Target**: The original uncorrupted input itself

The "label" is the original input — the model learns to reconstruct what was corrupted. This is self-supervised because the supervision comes from the data itself, not from external labels.

At inference time, **anomalies** are detected when the model produces high reconstruction errors — it has never seen anomalous data patterns during training, so it cannot reconstruct them accurately.

## Features

| Feature            | Description                       |
|--------------------|-----------------------------------|
| `request_count`    | Number of requests                |
| `bytes_per_request`| Average bytes per request         |
| `cpu_usage`        | CPU usage percentage (0-100)      |
| `memory_usage`     | Memory usage percentage (0-100)   |
| `disk_io`          | Disk I/O operations per second    |
| `network_in`       | Network inbound MB/s              |
| `network_out`      | Network outbound MB/s             |
| `error_rate`       | Error rate percentage (0-100)     |
| `connection_count` | Active connections                |
| `response_time`    | Average response time in ms       |

## Approach

- **Architecture**: Feedforward autoencoder (encoder + decoder) from scratch
- **Encoder**: Input → Hidden layer (ReLU activation)
- **Decoder**: Hidden layer → Output (linear activation)
- **Loss**: Mean Squared Error (MSE) between reconstruction and original
- **Corruption**: Input dropout + Gaussian noise during training
- **Optimization**: Full-batch gradient descent with He initialization
- **Early stopping**: Training stops when loss converges
- **Implementation**: Pure NumPy, no external ML libraries

## Architecture

```
apps/machine-learning/self-supervised/self-supervised-monitoring/
├── pyproject.toml              # Package configuration
├── src/self_supervised_monitoring/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Synthetic server metrics + corruption
│   ├── model.py                # DenoisingAutoencoder (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-self-supervised-monitoring
```

Or directly:

```bash
uv run python -m self_supervised_monitoring.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0
```

### Serving

```bash
MODEL_DIR=./artifacts/models uv run uvicorn self_supervised_monitoring.api:app --host 0.0.0.0 --port 8007
```

### API Usage

```bash
# Health check
curl http://localhost:8007/health

# Detect anomaly for single observation
curl -X POST http://localhost:8007/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_count": 50, "bytes_per_request": 5000,
    "cpu_usage": 35, "memory_usage": 55,
    "disk_io": 150, "network_in": 30, "network_out": 25,
    "error_rate": 1.0, "connection_count": 45,
    "response_time": 60
  }'

# Detect anomalies for multiple observations
curl -X POST http://localhost:8007/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"samples": [...]}'

# Get model statistics
curl http://localhost:8007/stats

# Get model information
curl http://localhost:8007/model/info
```

## API Endpoints

| Method | Path             | Description                |
|--------|------------------|----------------------------|
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Detect anomaly (single)    |
| POST   | `/predict/bulk`  | Detect anomalies (bulk)    |
| GET    | `/stats`         | Model statistics           |
| GET    | `/model/info`    | Model information          |
| GET    | `/drift`         | Drift detection            |
| GET    | `/metrics`       | Prometheus metrics         |
| POST   | `/reload`        | Reload model               |

## Model Parameters

| Parameter           | Default  | Description                              |
|---------------------|----------|------------------------------------------|
| `hidden_dim`        | `16`     | Size of hidden (encoding) layer          |
| `learning_rate`     | `0.01`   | Gradient descent step size               |
| `n_iterations`      | `5000`   | Maximum training iterations              |
| `hidden_activation` | `relu`   | Hidden layer activation: "relu" or "tanh"|
| `noise_rate`        | `0.25`   | Fraction of features to corrupt          |
| `threshold_percentile`| `95.0` | Percentile for anomaly threshold         |
| `random_seed`       | `42`     | Random seed for reproducibility          |

## How It Works

1. **Training**: The denoising autoencoder is trained on normal server metrics. During training:
   - The input is corrupted (features zeroed out + Gaussian noise added)
   - The model learns to reconstruct the original clean input from the corrupted version
   - This continues until reconstruction loss converges

2. **Threshold Setting**: The anomaly threshold is set at the 95th percentile of reconstruction errors on training data.

3. **Inference**: When a new metrics observation comes in:
   - The model reconstructs it
   - Reconstruction error (MSE) is computed
   - If error > threshold → anomaly detected

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy on test set
- **Precision**: Of predicted anomalies, how many were truly anomalous
- **Recall**: Of actual anomalies, how many were detected
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate

## Key Concepts Demonstrated

- Self-supervised representation learning
- Denoising autoencoders
- Anomaly detection via reconstruction error
- No-label training (only normal data needed)
- Production API serving with FastAPI
- Model versioning with registry
