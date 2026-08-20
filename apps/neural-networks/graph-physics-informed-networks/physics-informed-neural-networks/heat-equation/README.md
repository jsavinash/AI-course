# PINN Heat Equation Solver

Physics-Informed Neural Networks trained to solve supervised learning tasks while respecting given laws of physics described by differential equations.

## Network Type
PINN (Physics-Informed Neural Network)

## Architecture
- Input: (batch, 2) [x, t coordinates]
- Hidden: Dense (32, tanh) -> Dense (32, tanh)
- Output: Dense (1, linear) = temperature u(x, t)

## Training
```bash
pinn_heat_equation-train --model-dir ./artifacts/models --n-samples 200 --n-iterations 500
```

## Serving API
```bash
uvicorn pinn_heat_equation.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Predict temperature at (x, t) coordinates
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format
- `x`: spatial coordinate [0, 1]
- `t`: time coordinate [0, 0.5]

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
