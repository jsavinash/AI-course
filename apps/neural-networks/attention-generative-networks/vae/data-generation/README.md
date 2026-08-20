# VAE Data Generation

Variational Autoencoders that learn a probabilistic latent space to generate new data variations.

## Network Type
VAE (Variational Autoencoder)

## Architecture
- **Encoder**: Input (32 features) -> Dense (64, ReLU) -> Dense (16 = mean + 16 = log-variance)
- **Reparameterization**: z = mu + exp(0.5 * log_var) * epsilon (epsilon ~ N(0, 1))
- **Decoder**: Latent (16) -> Dense (64, ReLU) -> Dense (32, sigmoid)

## Training
```bash
vae_data_generation-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn vae_data_generation.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Reconstruct input data and return anomaly score (32 features)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[] (continuous feature values [0, 1])

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
