# Diffusion Image Generation

Networks that generate data by systematically removing noise from a random starting state, widely used in modern AI art generators.

## Network Type
Diffusion

## Architecture
- **Forward Process**: Gradually adds Gaussian noise over 1000 timesteps
- **Reverse Process**: Denoise images iteratively using a CNN-based noise predictor
- **Denoiser**: Input (1 x 8x8) -> Conv2D (8, 3x3, ReLU) -> MaxPool -> Flatten -> Dense (32) -> Dense (1, linear)

## Training
```bash
diffusion_image_generation-train --model-dir ./artifacts/models --n-iterations 200 --n-samples 500
```

## Serving API
```bash
uvicorn diffusion_image_generation.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Generate image from random noise (optionally specify timesteps)
- `POST /predict/bulk` - Batch image generation (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[] (continuous pixel values [0, 1])

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
