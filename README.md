# MLOps Monorepo

A **production-ready MLOps monorepo** built with `uv` workspace, containing 47 AI/ML examples spanning supervised, unsupervised, reinforcement, self-supervised, semi-supervised, neural network, deep learning, and generative AI topics.

## Architecture

```
├── pyproject.toml                 # Root uv workspace (no runtime deps) + shared tool config
├── uv.lock                        # Locked, reproducible dependencies (committed)
├── Makefile                       # Unified task runner (generic APP=<module> targets)
├── README.md                      # This file
├── .gitignore
├── .env.example                   # Config template (secrets are NEVER committed)
│
├── packages/                      # ── Shared libraries (workspace members) ──
│   └── ai-core/                   # Shared AI/ML foundation library
│       ├── pyproject.toml         # Common runtime deps (numpy, fastapi, mlflow, …) with upper bounds
│       ├── README.md
│       ├── tests/
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
├── apps/                          # ── Independent ML projects (one deployable unit each) ──
│   ├── machine-learning/          #   Each app: pyproject.toml (app-specific deps only),
│   ├── neural-networks/           #   src/<package>/{model,data,train,api}.py, tests/
│   ├── deep-learning/             #   Apps depend on `ai-core` via `tool.uv.sources (workspace = true)`
│   └── generative-ai/
│
├── artifacts/                     # Local-only training outputs (gitignored — registry/s3 is the source of truth)
│   ├── models/<model>/<version>/  # Canonical per-model registry layout
│   └── _archive_duplicates/       # Superseded duplicate artifacts (do not use)
│
├── docker/
│   ├── train.Dockerfile           # Parameterized: --build-arg APP_MODULE=<python_module>
│   └── serve.Dockerfile           # Parameterized: --build-arg APP_MODULE=<python_module>
│
├── k8s/
│   ├── base/
│   ├── overlays/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   └── scripts/
│
├── scripts/                       # Operational CLIs (codegen, validation, serving) — version-controlled
│
└── .github/
    └── workflows/                 # CI/CD pipelines — version-controlled
        ├── ci.yml                 # lint + typecheck + pytest (uv sync --all-packages)
        ├── cd.yml                 # build/push images, deploy serving per environment
        └── security.yml           # dependency audit + secret scanning
```

## Quick Start

```bash
# Install all workspace members (apps + shared packages) into one environment
make install          # = uv sync --all-packages

# Train a model (generic target, APP = python import module)
make train APP=spam_classification

# Run an API locally
make serve APP=spam_classification PORT=8001
# ...or start every API at once (fixed port map, see scripts/serve_all.py)
make serve-all

# Build parameterized images
make docker-train APP=spam_classification
make docker-serve APP=spam_classification

# Run tests
make test

# Lint and typecheck
make lint
make typecheck
```

## Tech Stack

- **Package Manager**: uv (workspace monorepo; all 54 members resolve from a single committed `uv.lock`)
- **ML**: NumPy, SciPy, scikit-learn, pandas
- **Deep Learning**: Pure NumPy implementations (no PyTorch/TensorFlow dependencies)
- **Serving**: FastAPI + Uvicorn
- **Observability**: Prometheus client, structured JSON logging
- **Tracking**: MLflow
- **Validation**: Pydantic, custom schema validation
- **Testing**: pytest + pytest-cov
- **Linting**: ruff
- **Type Checking**: mypy (strict mode)

## Dependency Strategy

The workspace uses a **two-tier dependency model** to avoid dependency drift across 54 members:

- **`packages/ai-core`** owns the common runtime dependencies (numpy, pandas, fastapi, pydantic, mlflow, structlog, …) pinned with **upper bounds** (e.g. `numpy>=1.26,<3`) so a single upstream major release cannot silently break every app.
- **Each app's `pyproject.toml`** declares only its app-specific extras and `ai-core` as a workspace dependency (`ai-core = { workspace = true }`).
- **The root `pyproject.toml`** is not a package: it defines the workspace and the shared `dev` tool group (pytest, ruff, mypy). `uv sync --all-packages` installs every member into one environment; `uv sync` (in CI/Docker) uses the committed `uv.lock` for reproducible builds.

## Makefile Commands

```bash
make install       # Install all dependencies (uv sync --all-packages)
make lint          # Ruff linting
make format        # Ruff formatting
make typecheck     # Mypy type checking
make test          # Run pytest suite
make test-cov      # Coverage report
make train APP=<module>       # Train one model (e.g. APP=spam_classification)
make serve APP=<module>       # Run one API locally
make docker-train APP=<module>  # Build training image for one app
make docker-serve APP=<module>  # Build serving image for one app
make train-all     # Train all models (per-app targets also available)
make serve-all     # Run all APIs locally
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

## Model Lifecycle

Every model follows one canonical pipeline (implemented end-to-end in
`apps/machine-learning/supervised/spam-classification`, with the reusable
machinery in `packages/ai-core/src/ai_core/export.py`):

```
train.py  ──►  MLflow  ──►  export.py  ──►  artifacts/  ──►  FastAPI  ──►  prediction
(learn)       (record it)   (package it)   (signed file)   (serve it)    (use it)
```

| Step | What happens | Where |
| --- | --- | --- |
| **train.py** (learn) | Validates data, trains, evaluates, saves `spam_model_v<ver>.npz` + `model_info.json` via `ModelRegistry`; with `--register-mlflow` records params/metrics/artifacts to MLflow and stamps the `mlflow_run_id` into the registry metadata | `<app>/src/<pkg>/train.py` |
| **MLflow** (record it) | One run per training: params, metrics, model/chart/data artifacts, tags — the provenance record | `ai_core.model_registry.log_to_mlflow` |
| **export.py** (package it) | Packages the trained `.npz` + metadata into a single **HMAC-SHA256-signed** `model.signed` file (payload hash, `mlflow_run_id`, `git_sha`, metrics, `exported_at` in the manifest) and can log the signed artifact back to MLflow (`--record-mlflow`) | `<app>/src/<pkg>/export.py` + `ai_core.export` |
| **artifacts/** (signed file) | `artifacts/models/<model>/<version>/model.signed` (+ human-readable `manifest.json`) — the only artifact shape the serving layer prefers to load | local dir today; S3/DVC remote is the production target |
| **FastAPI** (serve it) | At startup `_load_model()` prefers `model.signed`: `verify_signed_model()` checks the HMAC signature (key from `MODEL_SIGNING_KEY`) and the payload SHA-256 **before** weights are loaded; tampered/unverifiable artifacts are refused (set `REQUIRE_SIGNED_MODEL=false` only for local dev) | `<app>/src/<pkg>/api.py` |
| **prediction** (use it) | `/predict`, `/predict/email`, `/health`, `/metrics`, `/drift` — every response reports the signed artifact's `model_version` | `<app>/src/<pkg>/api.py` |

Run it locally:

```bash
# 1. learn
make train APP=spam_classification
# 2. record (requires an MLflow server: mlflow server --backend-store-uri ./mlruns)
uv run python -m spam_classification.train --model-dir ./artifacts/models --model-version 1.0.0 --register-mlflow
# 3. package (sign)
MODEL_SIGNING_KEY=<secret> uv run python -m spam_classification.export \
    --model-dir ./artifacts/models --model-version 1.0.0
# 4./5. serve + use (signature verified at startup)
MODEL_SIGNING_KEY=<secret> make serve APP=spam_classification PORT=8001
curl -X POST localhost:8001/predict/email -H 'Content-Type: application/json' \
    -d '{"text": "win a free prize now"}'
```

To roll this lifecycle out to another app, copy `export.py` from the
spam-classification app and add the "0. Signed model package" block from its
`api.py::_load_model` — the signing/verification itself lives entirely in
`ai_core.export`.

## Production Readiness

| Area | Status |
| --- | --- |
| Workspace / `uv` | ✅ 53 app members + `ai-core`; committed `uv.lock` for reproducible builds; root has zero runtime deps, common deps centralized in `ai-core` with upper version bounds |
| Secrets hygiene | ✅ `.env` untracked, `.env.example` template committed; workflows/Dockerfiles/scripts version-controlled (previously self-ignored by `.gitignore`) |
| Observability | ✅ 53/53 apps expose `/health` + `/metrics` with structured logging |
| Shared MLOps lib | ✅ config, logging, metrics, registry, validation, drift, FastAPI middleware; target namespaces `layers` / `api_base` / `losses` / `optim` / `train_loop` now re-exported |
| API hardening | ✅ all 53 apps: structured 422/500 error envelopes, security headers, CORS, request-size limit (413), opt-in API-key auth (401) + per-IP rate limit (429) via `add_observability_middleware` |
| CI/CD | ✅ `ci.yml` (lint/test), `cd.yml` (build/deploy), `security.yml` (dep + secret scan) — all tracked in git |
| Docker | ✅ multi-stage, non-root, `HEALTHCHECK`; **parameterized per app** via `--build-arg APP_MODULE`; installs all workspace members from the frozen lockfile; model artifacts come from registry/volumes, not baked into images |
| Kubernetes | 🟡 Helm `k8s/base` chart + `overlays/{dev,staging,prod}`; `cd.yml` deploys serving via `helm` (dry-run pending cluster) |
| Model/data lineage | 🟡 local registry layout `artifacts/models/<model>/<version>/` with `model_info.json`; migration to a hosted MLflow Model Registry + DVC/S3 data versioning is the next step |
| Tests | 🟡 330 passing tests (ai-core + per-app smoke tests); `make test-cov` / `make test-app` available; >80% behavioral coverage gate pending |

See `ANALYSIS.md` for the deep assessment and `MIGRATION.md` for the change log.
