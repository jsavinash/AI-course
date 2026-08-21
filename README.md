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
## AI Engineering Examples

Each example below ships a self-contained `README.md` (generated by `scripts/generate_docs.py`) covering **Mathematical Foundations**, **Core Logic & Architecture**, **Detailed Code Walkthrough**, and **Monorepo Integration**.

<details><summary>Full example index (53 examples)</summary>

| Category | Example | Documentation |
| --- | --- | --- |
| deep-learning | attention-mechanism | [README.md](apps/deep-learning/attention-mechanism/README.md) |
| deep-learning | large-language-model | [README.md](apps/deep-learning/large-language-model/README.md) |
| deep-learning | multimodal-llm | [README.md](apps/deep-learning/multimodal-llm/README.md) |
| deep-learning | pre-training-fine-tuning | [README.md](apps/deep-learning/pre-training-fine-tuning/README.md) |
| deep-learning | transfer-learning | [README.md](apps/deep-learning/transfer-learning/README.md) |
| deep-learning | transformers-language-modeling | [README.md](apps/deep-learning/transformers-language-modeling/README.md) |
| generative-ai | code-generation | [README.md](apps/generative-ai/code-generation/README.md) |
| generative-ai | image-generation | [README.md](apps/generative-ai/image-generation/README.md) |
| generative-ai | prompt-engineering | [README.md](apps/generative-ai/prompt-engineering/README.md) |
| generative-ai | retrieval-augmented-generation | [README.md](apps/generative-ai/retrieval-augmented-generation/README.md) |
| generative-ai | text-generation | [README.md](apps/generative-ai/text-generation/README.md) |
| generative-ai | tool-use-functional-calling | [README.md](apps/generative-ai/tool-use-functional-calling/README.md) |
| generative-ai | video-generation | [README.md](apps/generative-ai/video-generation/README.md) |
| machine-learning | reinforcement/robot-maze-navigation | [README.md](apps/machine-learning/reinforcement/robot-maze-navigation/README.md) |
| machine-learning | self-supervised/self-supervised-monitoring | [README.md](apps/machine-learning/self-supervised/self-supervised-monitoring/README.md) |
| machine-learning | semi-supervised/semi-supervised-email | [README.md](apps/machine-learning/semi-supervised/semi-supervised-email/README.md) |
| machine-learning | supervised/pizza-price | [README.md](apps/machine-learning/supervised/pizza-price/README.md) |
| machine-learning | supervised/spam-classification | [README.md](apps/machine-learning/supervised/spam-classification/README.md) |
| machine-learning | unsupervised/anomaly-detection-pca | [README.md](apps/machine-learning/unsupervised/anomaly-detection-pca/README.md) |
| machine-learning | unsupervised/market-segmentation | [README.md](apps/machine-learning/unsupervised/market-segmentation/README.md) |
| machine-learning | unsupervised/recommendation-engine | [README.md](apps/machine-learning/unsupervised/recommendation-engine/README.md) |
| neural-networks | attention-generative/diffusion | [README.md](apps/neural-networks/attention-generative/diffusion/README.md) |
| neural-networks | attention-generative/gan | [README.md](apps/neural-networks/attention-generative/gan/README.md) |
| neural-networks | attention-generative/transformers | [README.md](apps/neural-networks/attention-generative/transformers/README.md) |
| neural-networks | attention-generative/vae | [README.md](apps/neural-networks/attention-generative/vae/README.md) |
| neural-networks | convolutional/advanced-generative-art | [README.md](apps/neural-networks/convolutional/advanced-generative-art/README.md) |
| neural-networks | convolutional/advanced-semantic-segmentation | [README.md](apps/neural-networks/convolutional/advanced-semantic-segmentation/README.md) |
| neural-networks | convolutional/advanced-super-resolution | [README.md](apps/neural-networks/convolutional/advanced-super-resolution/README.md) |
| neural-networks | convolutional/capsnet-autonomous-driving | [README.md](apps/neural-networks/convolutional/capsnet-autonomous-driving/README.md) |
| neural-networks | convolutional/capsnet-medical-scan | [README.md](apps/neural-networks/convolutional/capsnet-medical-scan/README.md) |
| neural-networks | convolutional/capsnet-text-recognition | [README.md](apps/neural-networks/convolutional/capsnet-text-recognition/README.md) |
| neural-networks | convolutional/cnn-facial-recognition | [README.md](apps/neural-networks/convolutional/cnn-facial-recognition/README.md) |
| neural-networks | convolutional/cnn-medical-imaging | [README.md](apps/neural-networks/convolutional/cnn-medical-imaging/README.md) |
| neural-networks | convolutional/cnn-video-surveillance | [README.md](apps/neural-networks/convolutional/cnn-video-surveillance/README.md) |
| neural-networks | feedforward/anomaly-detection-fraud | [README.md](apps/neural-networks/feedforward/anomaly-detection-fraud/README.md) |
| neural-networks | feedforward/classification-email-spam | [README.md](apps/neural-networks/feedforward/classification-email-spam/README.md) |
| neural-networks | feedforward/pattern-recognition-digits | [README.md](apps/neural-networks/feedforward/pattern-recognition-digits/README.md) |
| neural-networks | feedforward/regression-house-price | [README.md](apps/neural-networks/feedforward/regression-house-price/README.md) |
| neural-networks | graph-physics-informed/gnn-social-networks | [README.md](apps/neural-networks/graph-physics-informed/gnn-social-networks/README.md) |
| neural-networks | graph-physics-informed/pinn-heat-equation | [README.md](apps/neural-networks/graph-physics-informed/pinn-heat-equation/README.md) |
| neural-networks | graph-physics-informed/snn-image-classification | [README.md](apps/neural-networks/graph-physics-informed/snn-image-classification/README.md) |
| neural-networks | recurrent/nlp-language-translation | [README.md](apps/neural-networks/recurrent/nlp-language-translation/README.md) |
| neural-networks | recurrent/nlp-sentiment-analysis | [README.md](apps/neural-networks/recurrent/nlp-sentiment-analysis/README.md) |
| neural-networks | recurrent/nlp-text-generation | [README.md](apps/neural-networks/recurrent/nlp-text-generation/README.md) |
| neural-networks | recurrent/speech-audio-music | [README.md](apps/neural-networks/recurrent/speech-audio-music/README.md) |
| neural-networks | recurrent/speech-audio-recognition | [README.md](apps/neural-networks/recurrent/speech-audio-recognition/README.md) |
| neural-networks | recurrent/time-series-stock | [README.md](apps/neural-networks/recurrent/time-series-stock/README.md) |
| neural-networks | recurrent/time-series-weather | [README.md](apps/neural-networks/recurrent/time-series-weather/README.md) |
| neural-networks | recurrent/vision-image-captioning | [README.md](apps/neural-networks/recurrent/vision-image-captioning/README.md) |
| neural-networks | unsupervised/autoencoders-dimensionality-reduction | [README.md](apps/neural-networks/unsupervised/autoencoders-dimensionality-reduction/README.md) |
| neural-networks | unsupervised/deep-belief-networks | [README.md](apps/neural-networks/unsupervised/deep-belief-networks/README.md) |
| neural-networks | unsupervised/restricted-boltzmann-machines | [README.md](apps/neural-networks/unsupervised/restricted-boltzmann-machines/README.md) |
| neural-networks | unsupervised/self-organizing-maps | [README.md](apps/neural-networks/unsupervised/self-organizing-maps/README.md) |

</details>

## Production Readiness

This monorepo already follows the target `uv` workspace layout: a shared
`packages/ai-core` library plus 53 kebab-case app members, each with
`src/<package>/` (`model.py`, `data.py`, `train.py`, `api.py`).

| Area | Status |
| --- | --- |
| Workspace / `uv` | ✅ 53 app members + `ai-core`, `uv.lock` present |
| Observability | ✅ 53/53 apps expose `/health` + `/metrics` with structured logging |
| Shared MLOps lib | ✅ config, logging, metrics, registry, validation, drift, FastAPI middleware; target namespaces `layers` / `api_base` / `losses` / `optim` / `train_loop` now re-exported |
| API hardening | ✅ all 53 apps: structured 422/500 error envelopes, security headers, CORS, request-size limit (413), opt-in API-key auth (401) + per-IP rate limit (429) via `add_observability_middleware` |
| CI/CD | ✅ `ci.yml` (lint/test), `cd.yml` (build/deploy), `security.yml` (dep + secret scan) |
| Docker | ✅ multi-stage, non-root, `HEALTHCHECK`; `.dockerignore` added |
| Kubernetes | 🟡 Helm `k8s/base` chart + `overlays/{dev,staging,prod}`; `cd.yml` deploys serving via `helm` (dry-run pending cluster) |
| Tests | 🟡 per-app smoke tests for all apps; `make test-cov` / `make test-app` added; >80% behavioral coverage gate pending |

See `ANALYSIS.md` for the deep assessment and `MIGRATION.md` for the change log.
