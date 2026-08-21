# Monorepo Deep Analysis & Production-Readiness Assessment

> **Note on the brief:** The "Current State Snapshot" in the refactor brief describes a repo with
> `shared/mlops_shared/`, `apps/.../spam-classification-lr/`, `feedforward-neural-networks/`, etc.
> **That snapshot does not match the actual repository on disk.** This analysis is grounded in the
> *real* filesystem state, gathered with direct inspection (Aug 2026).

## 1. Inventory (actual)

- **Workspace**: root `pyproject.toml` is a `uv` workspace with **53 app members** + `packages/ai-core`. `uv.lock` present (930 KB).
- **Shared library**: `packages/ai-core/src/ai_core/` with: `config.py`, `logging.py`, `metrics.py`, `model_registry.py`, `validation.py`, `drift.py`, `fastapi_middleware.py`, `nn_utils/` (`cnn.py`, `rnn.py`), `py.typed`, `__init__.py`.
- **Apps**: 53 example apps under `apps/{machine-learning,neural-networks,deep-learning,generative-ai}/...`, each with `pyproject.toml`, `src/<package>/`, and most with `tests/`.
- **Structure already matches target 2.1**: kebab-case dirs, snake_case packages, `model.py`/`data.py`/`train.py`/`api.py`, `src/` layout, `artifacts/`, `docker/`, `k8s/`, `scripts/`, `.github/workflows/`.
- **Observability**: **53/53** apps expose `/health` and `/metrics`.
- **Tests**: root `tests/` has `conftest.py`, `test_apis.py`, `test_models.py`, `test_models/`, `unit/`, `integration/`, `e2e/`. Per-app `tests/` exist for **27/53** apps.
- **CI**: single `.github/workflows/ci-cd.yml` (target wants `ci.yml` + `cd.yml` + `security.yml`).
- **Docker**: `docker/train.Dockerfile`, `docker/serve.Dockerfile` exist.
- **K8s**: flat manifests (`serving.yaml`, `training-jobs.yaml`, `namespace.yaml`, `secrets.yaml`, `storage.yaml`, `autoscaling.yaml`, `monitoring.yaml`, `mlflow.yaml`) — not the Helm `base/overlays/` structure target wants.
- **Makefile**: has `install`, `lint`, `format`, `typecheck`, `test`, `uv-sync`, and `train-<app>` for all 53 apps (and `serve-<app>` variants).

## 2. Production-readiness assessment (per criterion)

| Criterion | Status | Evidence |
|---|---|---|
| Package structure / namespacing | ✅ Good | `packages/ai-core` + 53 `src/<pkg>/` with `__init__.py`; workspace source deps correct |
| Dependency management | ✅ Good | Deps pinned with `>=`; `ai-core` is lightweight (numpy/scipy/pandas/mlflow/structlog/fastapi); apps depend on `ai-core` via `tool.uv.sources` workspace |
| Testing | ⚠️ Partial | Only 27/53 apps have per-app tests; no enforced coverage gate; no per-app `tests/` for 26 apps |
| Typing & linting | ⚠️ Unverified | `mypy`/`ruff` configured; full-workspace pass not confirmed in this pass (external tool run pending) |
| Observability | ✅ Good | 53/53 apps expose `/health` + `/metrics`, structured JSON logging, middleware (correlation ids in `fastapi_middleware.py`) |
| Configuration | ✅ Good | `ai_core.config` provides 12-factor `load_config` with typed dataclasses (MLflow, ModelRegistry, Monitoring, Data) |
| Model lifecycle | ✅ Good | `ai_core.model_registry` (ModelRegistry + stages/transitions) + MLflow; per-app `train.py` saves artifacts |
| CI/CD | ⚠️ Partial | Single `ci-cd.yml`; target wants split ci/cd/security; no security/dependency-scan job |
| Containerization | ⚠️ Need review | `train.Dockerfile`/`serve.Dockerfile` exist but not yet verified multi-stage/non-root/HEALTHCHECK |
| Kubernetes | ⚠️ Partial | Flat manifests present, no Helm `base/overlays/`, no explicit probes/limits in a single chart |

## 3. Real gaps (the actual "refactor" work)

1. **Test coverage**: 26 apps lack `tests/`. Target wants per-app tests + >80% coverage.
2. **CI decomposition**: split into `ci.yml` (lint/typecheck/test), `cd.yml` (build/push/deploy), `security.yml` (dep-scan/secrets).
3. **ai-core module naming**: target lists `layers.py`, `losses.py`, `optim.py`, `train_loop.py`, `api_base.py`. Actual has `nn_utils/` + `fastapi_middleware.py`. *Functionality is present*; only the file names differ. Renaming risks breaking 53 import sites — prefer thin re-export shims if alignment is required.
4. **K8s Helm structure**: introduce `k8s/base/` chart + `overlays/{dev,staging,prod}`; add probes/limits/SA/RBAC.
5. **Docker hardening**: verify/apply multi-stage, non-root, `HEALTHCHECK`, layer-caching.
6. **Docs**: README quickstart/per-app index exists; add production-readiness status + how-to.

## 4. Refactor opportunities (real)

- The workspace is already consolidated; no directory flattening needed.
- `ai-core` already centralizes config/logging/metrics/registry/validation/drift — no cross-app duplication of MLOps plumbing observed.
- No circular workspace deps detected (apps → `ai-core` only).
- Orphaned/duplicate risk: low, given the consistent per-app `src/<pkg>/` pattern.
