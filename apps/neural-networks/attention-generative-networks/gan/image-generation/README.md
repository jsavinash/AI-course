# GAN Image Generation

Two networks (a generator and a discriminator) compete to create highly realistic synthetic images.

## Network Type
GAN (Generative Adversarial Network)

## Architecture
- **Generator**: Latent vector (16-dim) -> Dense (128) -> Dense (64) -> Reshape (1x8x8) -> Sigmoid
- **Discriminator**: Input (1x8x8) -> Conv2D (8, 3x3, ReLU) -> MaxPool -> Flatten -> Dense (16) -> Sigmoid

## Training
```bash
gan_image_generation-train --model-dir ./artifacts/models --n-iterations 200 --n-samples 500
```

## Serving API
```bash
uvicorn gan_image_generation.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Generate image from latent vector (16 values)
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
