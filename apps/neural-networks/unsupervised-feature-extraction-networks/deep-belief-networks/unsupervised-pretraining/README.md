# Deep Belief Networks

Generative graphical models composed of multiple layers of latent variables, often used for unsupervised pre-training.

## Network Type
DBN (Deep Belief Network)

## Architecture
- Stack of Restricted Boltzmann Machines (RBMs) trained greedily layer by layer
- Layer 1: RBM(32 visible -> 16 hidden)
- Layer 2: RBM(16 visible -> 8 hidden)
- Training via Contrastive Divergence (CD-k)

## Training
```bash
deep_belief_networks-train --model-dir ./artifacts/models --n-epochs 100 --n-samples 500
```

## Serving API
```bash
uvicorn deep_belief_networks.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Encode features to latent representation (32 features)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[] (binary feature values [0, 1])

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
