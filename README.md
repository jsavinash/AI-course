# MLOps Monorepo

A **production-ready MLOps monorepo** built with `uv` workspace, containing 47 AI/ML examples spanning supervised, unsupervised, reinforcement, self-supervised, semi-supervised, neural network, deep learning, and generative AI topics.

## Architecture

```
├── pyproject.toml                 # Root uv workspace + shared tool config
├── uv.lock                        # Locked, reproducible dependencies
├── Makefile                       # Unified task runner
├── README.md                      # This file
├── .gitignore
├── .env.example
│
├── packages/
│   └── ai-core/                   # Shared AI/ML foundation library
│       ├── pyproject.toml
│       ├── README.md
│       └── src/ai_core/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── metrics.py
│           ├── model_registry.py
│           ├── validation.py
│           ├── drift.py
│           ├── fastapi_middleware.py
│           └── nn_utils/
│
├── apps/
│   ├── machine-learning/
│   │   ├── supervised/
│   │   │   ├── pizza-price/
│   │   │   └── spam-classification/
│   │   ├── unsupervised/
│   │   │   ├── market-segmentation/
│   │   │   ├── recommendation-engine/
│   │   │   └── anomaly-detection-pca/
│   │   ├── reinforcement/
│   │   │   └── robot-maze-navigation/
│   │   ├── semi-supervised/
│   │   │   └── semi-supervised-email/
│   │   └── self-supervised/
│   │       └── self-supervised-monitoring/
│   │
│   ├── neural-networks/
│   │   ├── feedforward/
│   │   │   ├── classification-email-spam/
│   │   │   ├── regression-house-price/
│   │   │   ├── anomaly-detection-fraud/
│   │   │   └── pattern-recognition-digits/
│   │   ├── recurrent/
│   │   │   ├── nlp-language-translation/
│   │   │   ├── nlp-sentiment-analysis/
│   │   │   ├── nlp-text-generation/
│   │   │   ├── speech-audio-recognition/
│   │   │   ├── speech-audio-music/
│   │   │   ├── time-series-stock/
│   │   │   ├── time-series-weather/
│   │   │   └── vision-image-captioning/
│   │   ├── convolutional/
│   │   │   ├── cnn-medical-imaging/
│   │   │   ├── cnn-facial-recognition/
│   │   │   ├── cnn-video-surveillance/
│   │   │   ├── advanced-super-resolution/
│   │   │   ├── advanced-semantic-segmentation/
│   │   │   ├── advanced-generative-art/
│   │   │   ├── capsnet-autonomous-driving/
│   │   │   ├── capsnet-medical-scan/
│   │   │   └── capsnet-text-recognition/
│   │   ├── graph-physics-informed/
│   │   │   ├── gnn-social-networks/
│   │   │   ├── pinn-heat-equation/
│   │   │   └── snn-image-classification/
│   │   ├── attention-generative/
│   │   │   ├── diffusion/
│   │   │   ├── gan/
│   │   │   ├── transformers/
│   │   │   └── vae/
│   │   └── unsupervised/
│   │       ├── autoencoders-dimensionality-reduction/
│   │       ├── deep-belief-networks/
│   │       ├── restricted-boltzmann-machines/
│   │       └── self-organizing-maps/
│   │
│   ├── deep-learning/
│   │   ├── transformers-language-modeling/
│   │   ├── attention-mechanism/
│   │   ├── large-language-model/
│   │   ├── pre-training-fine-tuning/
│   │   ├── transfer-learning/
│   │   └── multimodal-llm/
│   │
│   └── generative-ai/
│       ├── prompt-engineering/
│       ├── code-generation/
│       ├── text-generation/
│       ├── image-generation/
│       ├── video-generation/
│       ├── retrieval-augmented-generation/
│       └── tool-use-functional-calling/
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── artifacts/
│   ├── models/
│   ├── experiments/
│   └── data/
│
├── docker/
│   ├── base/
│   ├── train.Dockerfile
│   └── serve.Dockerfile
│
├── k8s/
│   ├── base/
│   ├── overlays/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   └── scripts/
│
├── scripts/
│   ├── train_all.sh
│   ├── deploy.sh
│   └── setup.sh
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── cd.yml
        └── security.yml
```

## Quick Start

```bash
# Install dependencies
make install

# Train a model
make train-pizza

# Run API locally
make serve-pizza

# Run tests
make test

# Lint and typecheck
make lint
make typecheck
```

## Tech Stack

- **Package Manager**: uv (workspace monorepo)
- **ML**: NumPy, SciPy, scikit-learn, pandas
- **Deep Learning**: Pure NumPy implementations (no PyTorch/TensorFlow dependencies)
- **Serving**: FastAPI + Uvicorn
- **Observability**: Prometheus client, structured JSON logging
- **Tracking**: MLflow
- **Validation**: Pydantic, custom schema validation
- **Testing**: pytest + pytest-cov
- **Linting**: ruff
- **Type Checking**: mypy (strict mode)

## Makefile Commands

```bash
make install       # Install all dependencies
make lint          # Ruff linting
make format        # Ruff formatting
make typecheck     # Mypy type checking
make test          # Run pytest suite
make train-all     # Train all models
make serve-all     # Run all APIs locally
make docker-build  # Build Docker images
make k8s-apply     # Deploy to Kubernetes
```

## License

Educational purposes.
