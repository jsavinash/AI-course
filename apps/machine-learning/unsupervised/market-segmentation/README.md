# Market Segmentation (Unsupervised Learning)

Production-ready K-Means clustering model for customer market segmentation.

## Learning Type: Unsupervised (Clustering)

This example demonstrates **unsupervised learning** — the model is trained on unlabeled data and discovers hidden patterns (customer clusters) in the data without any known labels.

## Problem

Segment customers into market segments based on annual income and spending score using K-Means clustering.

## Features

| Feature           | Description                         |
|-------------------|-------------------------------------|
| `annual_income`   | Annual income (in $1000s)           |
| `spending_score`  | Spending score (0-100 scale)        |

## Approach

- **Algorithm**: K-Means clustering with Lloyd's algorithm
- **Initialization**: Random centroid selection (with multiple random restarts)
- **Feature standardization**: Zero mean, unit variance
- **Multiple initializations**: `n_init` random restarts, best result retained
- **Evaluation**: Inertia (WCSS) and silhouette score

## Architecture

```
apps/machine-learning/unsupervised/market-segmentation/
├── pyproject.toml              # Package configuration
├── src/market_segmentation/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Training data generation
│   ├── model.py                # KMeans (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-market-segmentation
```

Or directly:

```bash
uv run python -m market_segmentation.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-market-segmentation
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn market_segmentation.api:app --host 0.0.0.0 --port 8002
```

### API Usage

```bash
# Health check
curl http://localhost:8002/health

# Segment a single customer
curl -X POST http://localhost:8002/segment \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 75.0, "spending_score": 85.0}'

# Segment multiple customers
curl -X POST http://localhost:8002/segment/bulk \
  -H "Content-Type: application/json" \
  -d '{"customers": [
    {"annual_income": 30.0, "spending_score": 30.0},
    {"annual_income": 80.0, "spending_score": 25.0}
  ]}'

# Get cluster profiles
curl http://localhost:8002/profiles

# Get model statistics
curl http://localhost:8002/stats
```

## API Endpoints

| Method | Path             | Description                      |
|--------|------------------|----------------------------------|
| GET    | `/health`        | Health check                     |
| POST   | `/segment`       | Segment single customer          |
| POST   | `/segment/bulk`  | Segment multiple customers       |
| GET    | `/profiles`      | Get cluster profiles             |
| GET    | `/stats`         | Model statistics                 |
| GET    | `/metrics`       | Prometheus metrics               |
| POST   | `/reload`        | Reload model                     |

## Model Parameters

| Parameter       | Default | Description                        |
|-----------------|---------|------------------------------------|
| `n_clusters`    | `5`     | Number of customer segments        |
| `max_iterations`| `300`   | Maximum K-Means iterations         |
| `n_init`        | `10`    | Number of random restarts          |
| `random_seed`   | `42`    | Random seed for reproducibility    |

## Cluster Profiles

The model provides interpretable cluster profiles including:

- Number of members per cluster
- Percentage of total customers
- Mean and std of income per cluster
- Mean and std of spending score per cluster

## Key Concepts Demonstrated

- K-Means clustering from scratch (Lloyd's algorithm)
- Multiple random initializations for robustness
- Feature standardization
- Confidence scoring based on distance to nearest vs. second-nearest centroid
- Silhouette score computation
- Cluster profiling for business interpretation
- Production API serving with FastAPI
- Model versioning with registry
