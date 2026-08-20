# Medical Imaging Diagnosis

Detects tumors or fractures in synthetic medical X-ray, MRI, and CT scan images

## Network Type
CNN

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 3 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 3 output units (softmax)

## Training
```bash
medical_imaging-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn medical_imaging.api:app --host 0.0.0.0 --port 8006
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Single prediction (64 pixel values)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
['normal', 'benign', 'malignant']

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
