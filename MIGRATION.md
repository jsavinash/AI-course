# Migration Report — Production-Readiness Refactor

**Date:** 2026-08-21
**Author:** automated refactor pass (opencode)

## Executive summary

The brief's "Current State Snapshot" assumed a repo organized as
`shared/mlops_shared/`, `apps/.../spam-classification-lr/`,
`feedforward-neural-networks/`, etc. **That snapshot does not match the actual
repository.** On inspection the repo already conforms to the Phase 2.1 target
layout: a `uv` workspace with `packages/ai-core` plus 53 kebab-case app members,
each with `src/<package>/` containing `model.py`/`data.py`/`train.py`/`api.py`,
and shared `artifacts/`, `docker/`, `k8s/`, `scripts/`, `.github/workflows/`.

**Therefore no directory moves were required.** The refactor was executed as a
**production-readiness gap-closing pass** rather than a structural teardown, which
preserves all existing AI/ML code as required by the constraints.

## What changed (this pass)

| Area | Change | Breaking? |
|---|---|---|
| CI/CD | Split single `ci-cd.yml` into `ci.yml` (lint/typecheck/test+coverage), `cd.yml` (build/push/deploy), `security.yml` (pip-audit + TruffleHog) | No — additive |
| Testing | Added import-based smoke tests (`tests/test_smoke.py`) to the 25 apps that previously had no `tests/` dir | No — additive |
| ai-core naming | Added target-layout namespaces as re-export shims: `layers.py`, `api_base.py` (re-export `nn_utils` / `fastapi_middleware`), plus reference utilities `losses.py`, `optim.py`, `train_loop.py`; exported from `ai_core/__init__.py` | No — additive |
| API hardening | `add_observability_middleware` now adds production-grade cross-cutting behavior to **all 53 apps** (they all call it): raw-ASGI `ProductionMiddleware` (correlation-id logging, metrics, security headers — never buffers the body), global deps for request-size limiting (413), opt-in API-key auth (401, `API_KEY` env) and per-IP rate limiting (429, `RATE_LIMIT_PER_MINUTE` env), plus structured JSON error envelopes (no traceback leak) for 422/500/HTTPException. `ai_core/fastapi_middleware.py` rewritten; `api_base.py`/`__init__` re-exports trimmed. | No — additive, backward-compatible |
| Docker | Added `.dockerignore` (excludes `.venv`, `artifacts`, VCS, caches); Dockerfiles already met multi-stage / non-root / HEALTHCHECK criteria | No — additive |
| Kubernetes | Added Helm `k8s/base/` chart (Deployment+Service+ServiceAccount, probes, securityContext, resources) and `k8s/overlays/{dev,staging,prod}/values.yaml`; `cd.yml` deploys serving APIs via `helm upgrade` | No — additive |
| Coverage tooling | `pyproject.toml` `addopts` now measures `ai_core`; added `make test-cov` / `make test-app` targets; discovery broadened to `packages`+`apps` | No — additive |
| Docs | Added `ANALYSIS.md` (ground-truth deep analysis + gap assessment) | No |
| Docs | This `MIGRATION.md` | No |

No `pyproject.toml`, source file, import path, or directory was moved or renamed.
`uv sync`, app import paths, and `/health` + `/metrics` endpoints are untouched.

## Old → new path mapping

Not applicable — paths are unchanged. (If the brief's assumed layout had existed,
the mapping would be e.g. `apps/.../spam-classification-lr/` →
`apps/machine-learning/supervised/spam-classification/`, but those source paths do
not exist in this repo.)

## Remaining gaps (deferred — require broader effort / external runners)

1. **Coverage gate** — per-app `pytest-cov` enforcement at >80% is *configured*
   (`make test-cov`) but not yet a hard `fail_under` gate, because reaching 80% per
   app requires writing behavioral tests for all 53 apps (only smoke tests exist so
   far). This is the single largest outstanding item.
2. **CI decomposition execution** — new workflows are syntactically valid but have
   not yet run in CI (requires a push/PR to GitHub).
3. **Docker build** — `docker build -f docker/train.Dockerfile .` not executed here
   (needs Docker + image registry creds).
4. **K8s dry-run** — `helm template` / `kubectl apply --dry-run=server` not executed
   here (needs a cluster); chart is templated by hand but unverified against a live API.
5. **Full-workspace test** — `make test` across all 53 apps not run end-to-end here
   (several apps need heavy optional deps, e.g. torch); representative apps validated.

## Validation results (this pass)

- ✅ `uv run ruff check .` — **all checks passed** (20 auto-fixes applied to generator
  scripts + new modules).
- ✅ `uv run mypy packages/ai-core` — **no issues in 16 source files** (incl. shims).
- ✅ `uv run pytest` on a representative app (pre-training-fine-tuning) smoke suite —
  **5 passed**; pizza-price existing unit tests — **3 passed**.
- ✅ `uv run python -c "import ai_core"` plus all new shim symbols — importable & run.
- ✅ Confirmed 53/53 apps expose `/health` + `/metrics`; workspace has 53 members +
  `uv.lock`; root `pyproject.toml` already declares workspace + shared tool config.
- ✅ ai-core target namespaces (`layers`, `api_base`, `losses`, `optim`, `train_loop`)
  now importable without breaking any existing `nn_utils` / `fastapi_middleware` imports.
- ⏳ Docker build / K8s dry-run / full 53-app `make test` — pending external runners
  (environment-limited, not code-blocked).

## Rationale

Blindly executing the literal "move everything to the target layout" would have been
*destructive and redundant* given the repo already matches that layout. The
higher-integrity action was to (a) ground the analysis in the real filesystem, (b)
report the mismatch, and (c) close the genuine production-readiness gaps with
additive, non-breaking changes that can be validated incrementally.
