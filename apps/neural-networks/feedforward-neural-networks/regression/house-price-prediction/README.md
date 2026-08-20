# House Price Prediction (Feedforward Neural Network)

Production-ready feedforward neural network for house price prediction (regression), built from scratch with NumPy.

## Learning Type: Feedforward Neural Network - Regression

This example demonstrates a **feedforward neural network** (multi-layer perceptron) for predicting house prices from features like square footage, location, and property characteristics. The entire network is implemented from scratch using only NumPy.

## Problem

Predict continuous house prices based on property features including size, bedrooms, location quality, age, and school ratings.

## Features

| Feature             | Description                          |
|---------------------|--------------------------------------|
| `sqft`              | Square footage of the house          |
| `bedrooms`          | Number of bedrooms                   |
| `bathrooms`         | Number of bathrooms                  |
| `location_score`    | Location quality score (0-100)       |
| `age`               | Age of property in years             |
| `garage`            | Number of garage spaces              |
| `lot_size`          | Lot size in square feet              |
| `year_built`        | Year property was built              |
| `property_type`     | Property type code (0-3)             |
| `school_rating`     | School rating score (0-10)           |

## Approach

- **Architecture**: MLP with 1 hidden layer
  - **Input layer**: 10 features
  - **Hidden layer**: `hidden_dim` neurons with ReLU activation
  - **Output layer**: 1 neuron with linear activation (predicted price)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Batch Gradient Descent with He initialization
- **Regularization**: L2 (weight decay) to prevent overfitting
- **Early stopping**: Stops when loss converges
- **Validation tracking**: Monitors validation loss during training
- **Implementation**: Pure NumPy, no external ML libraries

## Architecture

```
apps/neural-networks/feedforward-neural-networks/regression/house-price-prediction/
├── pyproject.toml              # Package configuration
├── src/house_price_prediction/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Synthetic house feature data
│   ├── model.py                # HousePriceNN (from scratch)
│   ├── train.py                # Training pipeline (MSE loss + backprop)
│   └── api.py                  # FastAPI serving with observability
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-house-price-prediction-nn
```

Or directly:

```bash
uv run python -m house_price_prediction.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0
```

### Serving

```bash
MODEL_DIR=./artifacts/models uv run uvicorn house_price_prediction.api:app --host 0.0.0.0 --port 8009
```

### API Usage

```bash
# Health check
curl http://localhost:8009/health

# Predict house price (10 features)
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2000, 3, 2, 85, 10, 2, 5000, 2014, 0, 8.5]}'

# Predict house prices for multiple properties
curl -X POST http://localhost:8009/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"requests": [[2000,3,2,85,10,2,5000,2014,0,8.5]]}'

# Get model statistics
curl http://localhost:8009/stats
```

## API Endpoints

| Method | Path             | Description                  |
|--------|------------------|------------------------------|
| GET    | `/health`        | Health check                 |
| POST   | `/predict`       | Predict single house price   |
| POST   | `/predict/bulk`  | Predict multiple house prices|
| GET    | `/stats`         | Model statistics             |
| GET    | `/drift`         | Drift detection              |
| GET    | `/metrics`       | Prometheus metrics           |
| POST   | `/reload`        | Reload model                 |

## Model Parameters

| Parameter           | Default  | Description                              |
|---------------------|----------|------------------------------------------|
| `hidden_dim`        | `32`     | Neurons in hidden layer                  |
| `learning_rate`     | `0.001`  | Gradient descent step size               |
| `n_iterations`      | `2000`   | Maximum training iterations              |
| `weight_decay`      | `0.0001` | L2 regularization strength               |
| `hidden_activation` | `relu`   | Hidden layer activation                  |
| `random_seed`       | `42`     | Random seed for reproducibility          |

## How It Works

1. **Training**: The MLP is trained using supervised regression with MSE loss. Backpropagation computes gradients, and the optimizer updates weights via gradient descent.

2. **Prediction**: The network outputs a single continuous value — the predicted house price in USD.

3. **Feature Normalization**: Input features are standardized to zero mean and unit variance for stable training.

## Evaluation Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

## Key Concepts Demonstrated

- Feedforward neural network (MLP) from scratch
- Forward and backward propagation for regression
- He weight initialization
- Mean squared error loss
- L2 regularization (weight decay)
- Feature standardization
- Continuous output prediction
- Production API serving with FastAPI
