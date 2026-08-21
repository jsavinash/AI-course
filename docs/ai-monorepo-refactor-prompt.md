# AI Monorepo Refactor & Restructure Prompt

## Context

You are working on a production-ready AI/ML monorepo that currently contains examples spanning:
- Machine Learning (supervised, unsupervised, reinforcement, semi-supervised, self-supervised)
- Neural Networks (feedforward, recurrent, convolutional, graph, physics-informed, attention/generative)
- Deep Learning (transformers, LLMs, attention mechanisms, pre-training/fine-tuning, transfer learning, multimodal)
- Generative AI (prompt engineering, code generation, RAG, text/image/video generation, tool use)

The repo uses `uv` as the package manager and is structured as a uv workspace monorepo. It already has shared MLOps utilities, FastAPI serving, MLflow integration, Kubernetes deployment configs, and a comprehensive Makefile.

## Your Mission

Perform a **deep analysis** of the current folder structure, identify production-readiness gaps, and then **refactor and restructure** the entire repository into a modern, production-ready `uv` framework monorepo while preserving all existing AI/ML code and functionality.

## Phase 1: Deep Analysis

### 1.1 Inventory the Current Structure
- Map every app, example, shared library, test, deployment config, and artifact.
- Document current directory hierarchy, package names, module boundaries, and dependency relationships.
- Identify which apps are fully implemented vs. stubs vs. missing content.
- Catalog all `pyproject.toml` files, their dependencies, entry points, and workspace relationships.
- Audit the `shared/mlops_shared/` package for completeness, duplication, and missing utilities.

### 1.2 Assess Production Readiness
Evaluate each component against production-grade criteria:
- **Package structure**: Are packages properly namespaced? Are `__init__.py` files correct? Do import paths work reliably in a workspace context?
- **Dependency management**: Are dependencies pinned? Are there version conflicts between workspace members? Are unnecessary dependencies exposed transitively?
- **Testing**: Is there per-app test coverage? Is there a unified test strategy? Are tests runnable in isolation and in aggregate?
- **Typing & linting**: Is mypy strict mode satisfied? Does ruff pass without ignores? Are type hints complete?
- **Observability**: Do all serving apps expose Prometheus metrics, structured logging, correlation IDs, and health checks?
- **Configuration**: Is config 12-factor compliant (env vars, YAML, validation)? Is there a shared config utility used consistently?
- **Model lifecycle**: Is MLflow integration consistent? Are model artifacts versioned? Is there a model registry?
- **CI/CD**: Are GitHub Actions workflows aligned with the current structure? Do they run lint, typecheck, test, build, and deploy for all workspace members?
- **Containerization**: Are Dockerfiles multi-stage, minimal, and consistent across apps? Is there a base image strategy?
- **Kubernetes**: Are K8s manifests Helm-compatible? Are there proper resource limits, health probes, and service accounts?

### 1.3 Identify Refactor Opportunities
- Consolidate duplicated code across apps (e.g., repeated training loops, API boilerplate).
- Flatten or standardize deeply nested or inconsistently named directories.
- Align package names with directory names and entry points.
- Extract common patterns into the shared library (e.g., base model classes, training utilities, API mixins).
- Identify orphaned files, stale configs, or unused dependencies.
- Check for circular workspace dependencies.

## Phase 2: Target Architecture Design

### 2.1 Proposed Directory Layout
Restructure into this production-ready layout:

```
.
├── pyproject.toml                 # Root uv workspace + shared tool config
├── uv.lock                        # Locked, reproducible dependencies
├── Makefile                       # Unified task runner
├── README.md                      # Architecture, quickstart, per-app index
├── .gitignore
├── .env.example
│
├── packages/
│   └── ai-core/                   # Shared AI/ML foundation library
│       ├── pyproject.toml
│       ├── src/
│       │   └── ai_core/
│       │       ├── __init__.py
│       │       ├── config.py          # 12-factor config loader
│       │       ├── logging.py         # Structured JSON logging
│       │       ├── metrics.py         # Prometheus metrics base
│       │       ├── registry.py        # MLflow model registry
│       │       ├── validation.py      # Schema validation
│       │       ├── drift.py           # Data/concept drift detection
│       │       ├── layers.py          # Neural network layer primitives
│       │       ├── losses.py          # Common loss functions
│       │       ├── optim.py           # Optimizer implementations
│       │       ├── train_loop.py      # Generic training loop with callbacks
│       │       └── api_base.py        # FastAPI base app with middleware
│       └── tests/
│
├── apps/
│   ├── machine-learning/
│   │   ├── supervised/
│   │   │   ├── pizza-price/
│   │   │   │   ├── pyproject.toml
│   │   │   │   ├── README.md
│   │   │   │   ├── src/
│   │   │   │   │   └── pizza_price/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       ├── model.py
│   │   │   │   │       ├── data.py
│   │   │   │   │       ├── train.py
│   │   │   │   │       └── api.py
│   │   │   │   └── tests/
│   │   │   └── spam-classification/
│   │   │       └── ... (same pattern)
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
│   │   └── attention-generative/
│   │       ├── diffusion/
│   │       ├── gan/
│   │       ├── transformers/
│   │       └── vae/
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
│   ├── conftest.py                 # Shared fixtures, test utilities
│   ├── unit/                       # Per-app unit tests
│   │   ├── test_pizza_price.py
│   │   ├── test_spam_classification.py
│   │   └── ...
│   ├── integration/                # Cross-app integration tests
│   └── e2e/                        # End-to-end API tests
│
├── artifacts/
│   ├── models/                     # Versioned model artifacts
│   ├── experiments/                # MLflow tracking data
│   └── data/                       # Cached datasets
│
├── docker/
│   ├── base/                       # Shared base image (Python + uv)
│   ├── train.Dockerfile            # Multi-stage training image
│   └── serve.Dockerfile            # Multi-stage serving image
│
├── k8s/
│   ├── base/                       # Base Helm chart
│   ├── overlays/                   # Environment-specific overlays
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   └── scripts/
│
├── scripts/
│   ├── train_all.sh                # Batch training orchestrator
│   ├── deploy.sh                   # Deployment helper
│   └── setup.sh                    # Initial environment setup
│
└── .github/
    └── workflows/
        ├── ci.yml                  # Lint, typecheck, test on PR
        ├── cd.yml                  # Build, push, deploy on merge
        └── security.yml            # Dependency scanning, secrets audit
```

### 2.2 Naming Conventions
- **Directory names**: kebab-case (e.g., `pizza-price`, `retrieval-augmented-generation`)
- **Package names**: snake_case, derived from directory (e.g., `pizza_price`, `retrieval_augmented_generation`)
- **Module files**: `model.py`, `data.py`, `train.py`, `api.py`, `utils.py` (if needed)
- **Class names**: PascalCase
- **Function/variable names**: snake_case
- **Workspace member names**: match the package name (e.g., `pizza-price` in root workspace maps to `pizza_price` package)

### 2.3 Dependency Strategy
- **Root `pyproject.toml`**: Defines workspace members, shared tool config (ruff, mypy, pytest), and common dev dependencies.
- **`packages/ai-core/pyproject.toml`**: Defines `ai-core` package with no external ML framework dependencies (pure NumPy/SciPy base) so it remains lightweight and testable.
- **App `pyproject.toml`**: Each app declares its specific dependencies (e.g., `torch`, `tensorflow`, `transformers`, `diffusers`, `opencv-python`) and depends on `ai-core` via workspace source.
- **No transitive bloat**: Each app only pulls in what it needs. Shared dependencies are declared in `ai-core` only if used by multiple apps.

### 2.4 uv Workspace Configuration
```toml
[tool.uv.workspace]
members = [
    "packages/ai-core",
    "apps/machine-learning/supervised/pizza-price",
    "apps/machine-learning/supervised/spam-classification",
    # ... all other apps
]

[tool.uv.sources]
ai-core = { workspace = true }
# No other workspace sources needed; uv resolves workspace deps automatically.
```

### 2.5 Testing Strategy
- **Per-app tests**: Each app has its own `tests/` directory with unit tests for model, data, and API.
- **Shared fixtures**: Root `tests/conftest.py` provides common fixtures (sample data, model instances, test clients).
- **Unified runner**: Root `pytest` config in `pyproject.toml` discovers all tests.
- **Coverage**: Enforce 80%+ coverage per app with `pytest-cov`.
- **Isolation**: Each app's tests can run independently via `uv run pytest apps/machine-learning/supervised/pizza-price`.

### 2.6 Linting & Type Checking
- **ruff**: Enforced at root level with per-app ignores only if absolutely necessary. Target `py311`, line-length 100.
- **mypy**: Strict mode at root. Per-app `mypy.ini` only if an app needs different settings.
- **Pre-commit hooks**: Optional but recommended for enforcing standards before commit.

### 2.7 Observability Standards
Every serving app (`api.py`) must include:
- `/health` endpoint returning `{"status": "ok"}`
- `/metrics` endpoint exposing Prometheus metrics
- Structured JSON logging via `ai-core.logging`
- Request correlation IDs via middleware
- Configurable via environment variables (port, model path, log level)

### 2.8 Model Lifecycle Standards
- All models trained via `python -m <package>.train` with CLI args:
  - `--model-dir` (default: `./artifacts/models`)
  - `--model-version` (default: `1.0.0`)
  - `--config` (optional YAML config path)
- Training saves artifacts to `artifacts/models/<app-name>/<version>/`
- MLflow tracking URI configurable via `MLFLOW_TRACKING_URI` env var
- Model registry in `ai-core.registry` provides load/save/list helpers

### 2.9 Docker Standards
- **Base image**: `python:3.11-slim` with `uv` installed
- **Multi-stage builds**: Separate `builder` (install deps) and `runtime` (copy artifacts) stages
- **Layer caching**: Copy `pyproject.toml` and `uv.lock` first, then `uv sync`, then source
- **Non-root user**: Run as non-root in final image
- **Health checks**: Include `HEALTHCHECK` in Dockerfile
- **Image variants**: `train` (with all ML deps) and `serve` (minimal runtime only)

### 2.10 Kubernetes Standards
- **Helm chart**: Single chart under `k8s/base/` with values files per environment
- **Resources**: Explicit CPU/memory requests and limits
- **Probes**: Liveness, readiness, and startup probes on `/health`
- **Secrets**: Injected via Kubernetes Secrets, never hardcoded
- **ConfigMaps**: For non-sensitive configuration
- **ServiceAccount**: Dedicated SA per app with minimal RBAC

## Phase 3: Execution Instructions

### 3.1 Refactor Steps
1. **Create new directory structure** as outlined in 2.1, moving files from old locations to new locations.
2. **Update all imports** across the entire codebase to reflect new package paths.
3. **Rewrite root `pyproject.toml`** with the new workspace member list and shared tool config.
4. **Rewrite each app `pyproject.toml`** with correct package name, dependencies, and entry points.
5. **Create `packages/ai-core`** by extracting shared code from `shared/mlops_shared/` and consolidating duplicated utilities from apps.
6. **Update `Makefile`** with new targets for the restructured layout.
7. **Update `README.md`** with the new architecture diagram, per-app quickstart, and development guide.
8. **Migrate tests** from root `tests/` into per-app `tests/` directories and root `tests/unit/`, `tests/integration/`, `tests/e2e/`.
9. **Update CI/CD workflows** in `.github/workflows/` to reflect the new structure.
10. **Update Dockerfiles** to use the new package paths.
11. **Update Kubernetes manifests** to reference new image names and config paths.
12. **Run full validation**: `make install`, `make lint`, `make typecheck`, `make test`, `make train-<app>`, `make serve-<app>` for every app.

### 3.2 Validation Checklist
- [ ] `uv sync` succeeds for the entire workspace
- [ ] `make lint` passes with zero errors
- [ ] `make typecheck` passes with zero errors
- [ ] `make test` passes with >80% coverage per app
- [ ] Every app's `train.py` runs successfully and produces artifacts in `artifacts/models/`
- [ ] Every app's `api.py` starts successfully on a random port and responds to `/health` and `/metrics`
- [ ] `docker build` succeeds for both train and serve images
- [ ] `kubectl apply --dry-run` succeeds for all K8s manifests
- [ ] No circular workspace dependencies
- [ ] All imports resolve correctly in IDE and at runtime

## Phase 4: Deliverables

1. **Restructured repository** with all code moved to new locations.
2. **Updated `pyproject.toml`** files (root + every package/app).
3. **Updated `Makefile`** with correct targets.
4. **Updated `README.md`** with new architecture and quickstart.
5. **Updated `.github/workflows/`** CI/CD pipelines.
6. **Updated `docker/`** and `k8s/`** configs.
7. **Migration report** (`MIGRATION.md`) documenting:
   - Old path → new path mapping for every file
   - Breaking changes (if any) and how to adapt
   - Rationale for each structural change
   - Validation results (test coverage, lint status, etc.)

## Constraints & Principles

- **Preserve all existing AI/ML code**: Do not rewrite model logic, training algorithms, or API endpoints unless they are broken. Focus on structure, not substance.
- **uv-first**: All package management, scripting, and task running must use `uv`.
- **Type-safe**: All code must pass mypy strict mode.
- **Lint-clean**: All code must pass ruff with minimal/no ignores.
- **Tested**: Every app must have passing unit tests.
- **Observable**: Every serving app must expose metrics and structured logs.
- **Documented**: Every app must have a README with description, how to train, how to serve, and sample API calls.
- **Consistent**: Naming, structure, and patterns must be uniform across all apps.

## Current State Snapshot

### Existing Directory Structure
```
/Users/avi/Documents/ai/
├── pyproject.toml                  # Root uv workspace
├── uv.lock                         # Lock file
├── Makefile                        # Task runner
├── README.md                       # Documentation
├── shared/
│   └── mlops_shared/               # Shared MLOps lib (config, logging, metrics, registry, validation, drift, middleware)
├── apps/
│   ├── machine-learning/           # ML examples
│   │   ├── supervised/
│   │   │   ├── pizza-price/
│   │   │   └── spam-classification-lr/
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
│   ├── neural-networks/            # NN examples
│   │   ├── feedforward-neural-networks/
│   │   ├── recurrent-neural-networks/
│   │   ├── convolutional-neural-networks/
│   │   ├── graph-physics-informed-networks/
│   │   ├── unsupervised-feature-extraction-networks/
│   │   └── attention-generative-networks/
│   ├── deep-learning/              # DL examples
│   │   ├── transformers-language-modeling/
│   │   ├── attention-mechanism/
│   │   ├── large-language-model/
│   │   ├── pre-training-fine-tuning/
│   │   ├── transfer-learning/
│   │   └── multimodal-llm/
│   └── generative-ai/              # GenAI examples
│       ├── prompt-engineering/
│       ├── code-generation/
│       ├── text-gen/
│       ├── image-generation/
│       ├── video-generation/
│       ├── retrieval-augmented-generation/
│       └── tool-use-and-functional-calling/
├── tests/
│   ├── test_apis.py
│   ├── test_models.py
│   └── test_models/                 # Per-app test modules
├── artifacts/
│   └── models/                      # Versioned model artifacts
├── docker/
├── k8s/
├── scripts/
└── .github/workflows/
```

### Key Observations
- Each app has its own `pyproject.toml`, `README.md`, and `src/<package>/` directory.
- Each app package contains `model.py`, `data.py`, `train.py`, and `api.py`.
- The `shared/mlops_shared/` package provides MLOps utilities consumed by apps.
- The root `pyproject.toml` defines the uv workspace with ~40 members.
- The project already uses `uv`, `ruff`, `mypy`, `pytest`, `fastapi`, `mlflow`, and `prometheus-client`.
- Some apps in `pyproject.toml` workspace members may not have corresponding directories or may be stubs.

## Expected Output

After completing this task, the repository should be a **gold-standard, production-ready uv monorepo** for AI/ML development that:
- Any engineer can clone, run `uv sync`, and immediately be productive.
- Any app can be trained, tested, and served with a single `make` command.
- CI/CD pipelines automatically validate every change.
- The structure is intuitive, consistent, and scalable for adding new AI topics.

Begin by performing the deep analysis, documenting findings, then executing the refactor step-by-step, validating at each stage.
