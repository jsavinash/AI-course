# Anomaly Detection (Unsupervised Learning)

Production-ready PCA-based anomaly detection for server monitoring metrics.

## Learning Type: Unsupervised (Anomaly Detection)

This example demonstrates **unsupervised learning** — the model is trained on unlabeled data (presumed normal) and learns the normal pattern. Anomalies are detected by measuring how much information is lost during dimensionality reduction and reconstruction.

## Problem

Detect anomalous server behavior from monitoring metrics (CPU, memory, network, etc.) without requiring labeled anomaly data.

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

- **Algorithm**: Principal Component Analysis (PCA) from scratch
- **Eigendecomposition**: Covariance matrix eigenvalue decomposition
- **Reconstruction**: Project to lower-dimensional space and reconstruct
- **Anomaly Scoring**: Reconstruction error (MSE between original and reconstructed)
- **Thresholding**: Automatic threshold selection via percentile, IQR, or fixed value

## Architecture

```
apps/machine-learning/unsupervised/anomaly-detection/
├── pyproject.toml              # Package configuration
├── src/anomaly_detection/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Synthetic server metrics generation
│   ├── model.py                # PCAAnomalyDetector (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-anomaly-detection
```

Or directly:

```bash
uv run python -m anomaly_detection.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-anomaly-detection
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn anomaly_detection.api:app --host 0.0.0.0 --port 8004
```

### API Usage

```bash
# Health check
curl http://localhost:8004/health

# Detect anomaly for single observation
curl -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_count": 120,
    "bytes_per_request": 4800,
    "cpu_usage": 35,
    "memory_usage": 55,
    "disk_io": 950,
    "network_in": 220,
    "network_out": 180,
    "error_rate": 1.5,
    "connection_count": 480,
    "response_time": 95
  }'

# Detect anomalies for multiple observations
curl -X POST http://localhost:8004/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"samples": [...]}'

# Get model information
curl http://localhost:8004/model/info

# Get model statistics
curl http://localhost:8004/stats
```

## API Endpoints

| Method | Path             | Description                |
|--------|------------------|----------------------------|
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Detect anomaly (single)    |
| POST   | `/predict/bulk`  | Detect anomalies (bulk)    |
| GET    | `/model/info`    | Model information          |
| GET    | `/stats`         | Model statistics           |
| GET    | `/drift`         | Drift detection            |
| GET    | `/metrics`       | Prometheus metrics         |
| POST   | `/reload`        | Reload model               |

## Model Parameters

| Parameter                    | Default    | Description                              |
|------------------------------|------------|------------------------------------------|
| `n_components`               | `0.95`     | Number of PCA components (int or variance ratio) |
| `threshold_method`           | `percentile`| Threshold method: "percentile", "iqr", "fixed" |
| `threshold_percentile`       | `95.0`     | Percentile for "percentile" method       |
| `threshold_iqr_multiplier`   | `1.5`      | IQR multiplier for "iqr" method        |
| `threshold_value`            | `0.0`      | Fixed threshold for "fixed" method       |
| `random_seed`                | `42`       | Random seed for reproducibility          |

## How It Works

1. **Training**: PCA is fit on normal server metrics data. The covariance matrix is decomposed into eigenvalues/eigenvectors. Top-k components are retained.

2. **Reconstruction**: During inference, input data is projected onto the principal components and then reconstructed back to original dimension. The reconstruction error measures how well the input conforms to the learned normal pattern.

3. **Anomaly Detection**: High reconstruction error indicates the input doesn't fit the normal pattern → anomaly.

## Key Concepts Demonstrated

- Principal Component Analysis from scratch (eigendecomposition)
- Anomaly detection via reconstruction error
- Multiple threshold selection methods (percentile, IQR, fixed)
- Feature standardization
- Drift detection for monitoring
- Production API serving with FastAPI
- Model versioning with registry
