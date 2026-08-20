# Pizza Price Prediction (Supervised Learning)

Production-ready linear regression model for predicting pizza prices based on diameter.

## Learning Type: Supervised (Regression)

This example demonstrates **supervised learning** — the model is trained on labeled data where each input (pizza diameter) has a known target output (price).

## Problem

Predict the price of a pizza given its diameter using linear regression.

**Model**: `price = weight * diameter + bias`

## Approach

- **Algorithm**: Linear Regression trained via gradient descent on Mean Squared Error (MSE) loss
- **Implementation**: Pure NumPy, no scikit-learn dependency
- **Features**: Single feature — pizza diameter (inches)
- **Labels**: Pizza price (USD)

## Architecture

```
machine-learning/supervised/pizza-price/
├── pyproject.toml              # Package configuration
├── src/pizza_price/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Training data generation
│   ├── model.py                # LinearRegression (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-pizza
```

Or directly:

```bash
uv run python -m pizza_price.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-pizza
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn pizza_price.api:app --host 0.0.0.0 --port 8000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict single price
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"diameter": 12.0}'

# Predict bulk
curl -X POST http://localhost:8000/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"diameters": [7, 12, 16, 20]}'
```

## API Endpoints

| Method | Path             | Description                |
|--------|------------------|----------------------------|
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Predict single price       |
| POST   | `/predict/bulk`  | Predict multiple prices    |
| GET    | `/metrics`       | Prometheus metrics         |
| POST   | `/reload`        | Reload model               |

## Model Parameters

| Parameter        | Default   | Description                       |
|------------------|-----------|-----------------------------------|
| `learning_rate`  | `0.001`   | Gradient descent step size        |
| `n_iterations`   | `2000`    | Number of training iterations     |

## Evaluation Metrics

The model provides:

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Key Concepts Demonstrated

- Gradient descent optimization
- Loss function tracking and convergence
- Feature scaling considerations
- Model serialization/deserialization
- Production API serving with FastAPI
- Prometheus metrics integration
- Kubernetes health checks
- Model versioning with registry
