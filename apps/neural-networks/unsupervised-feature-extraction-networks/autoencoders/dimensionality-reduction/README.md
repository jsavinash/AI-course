# Autoencoders Dimensionality Reduction

Networks trained to compress input data into a lower-dimensional code and then reconstruct it, used for denoising and dimensionality reduction.

## Network Type
Autoencoder (AE)

## Architecture
- **Encoder**: Input (32 features) -> Dense (16, ReLU) -> Dense (8, ReLU) = latent code
- **Decoder**: Latent (8) -> Dense (16, ReLU) -> Dense (32, sigmoid)

## Training
```bash
autoencoders_dimensionality-reduction-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn autoencoders_dimensionality_reduction.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Reconstruct input (32 features) and return latent code
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
