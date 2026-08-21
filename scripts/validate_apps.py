#!/usr/bin/env python3
"""Validate training + inference for every app in the monorepo.

For each app:
  1. Run its `train` module headlessly into a temporary MODEL_DIR.
  2. Serve the app's FastAPI app via TestClient (loads the trained model).
  3. Build a sample request body from the app's OpenAPI schema and POST it
     to the primary inference endpoint (/predict or first POST), asserting 200
     and that a numeric prediction is returned.

Outputs a per-app PASS / FAIL / SKIP table and a markdown report.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

APPS_ROOT = Path("apps")
TRAIN_TIMEOUT = int(os.getenv("VALIDATE_TRAIN_TIMEOUT", "180"))
SKIP = {"test-app"}

# env vars that most train scripts honor to shrink the workload
SMALL_ENV = {
    "MODEL_VERSION": "1.0.0",
    "N_ITERATIONS": "100",
    "EPOCHS": "2",
    "NUM_EPOCHS": "2",
    "NUM_ITERATIONS": "100",
    "BATCH_SIZE": "16",
}


def discover_apps() -> list[tuple[str, Path]]:
    apps: list[tuple[str, Path]] = []
    for train in sorted(APPS_ROOT.rglob("train.py")):
        pkg_dir = train.parent  # .../src/<pkg>
        pkg = pkg_dir.name
        if pkg in SKIP:
            continue
        apps.append((pkg, train))
    return apps


def _infer_scalar(schema: dict, t: str | None, name: str) -> object:
    lo = schema.get("minimum")
    hi = schema.get("maximum")
    if t == "integer":
        v = 1
        if lo is not None:
            v = max(v, int(lo))
        if hi is not None:
            v = min(v, int(hi))
        if lo is not None and hi is not None and not (lo <= v <= hi):
            v = int((lo + hi) / 2)
        return v
    if t == "number":
        v = 1.0
        if lo is not None:
            v = max(v, float(lo))
        if hi is not None:
            v = min(v, float(hi))
        if lo is not None and hi is not None and not (lo <= v <= hi):
            v = (lo + hi) / 2
        return v
    if t == "boolean":
        return True
    if t == "string":
        fmt = schema.get("format")
        if fmt == "date-time":
            return "2020-01-01T00:00:00"
        if fmt == "date":
            return "2020-01-01"
        return "x"
    return None


_STRINGY = {"context", "text", "prompt", "input", "description", "query",
            "question", "label", "name", "content", "message", "source",
            "target", "document", "title", "task"}
_LISTY = {"examples", "list", "items", "inputs", "candidates", "options"}


def build_sample(schema: dict, components: dict, depth: int = 0, name: str = "") -> object:
    if depth > 8:
        return None
    if "$ref" in schema:
        refname = schema["$ref"].split("/")[-1]
        schema = components.get("schemas", {}).get(refname, {})
    t = schema.get("type")
    if not t and "anyOf" in schema:
        # pick a concrete non-null option
        for opt in schema["anyOf"]:
            if opt.get("type") not in (None, "null"):
                t = opt.get("type")
                schema = opt
                break
    if not t and "properties" in schema:
        t = "object"
    if t == "object" or "properties" in schema:
        out = {}
        for k, v in schema.get("properties", {}).items():
            out[k] = build_sample(v, components, depth + 1, name=k)
        return out
    if t == "array":
        items = schema.get("items", {})
        val = build_sample(items, components, depth + 1)
        n = int(schema.get("minItems") or schema.get("min_length") or 1)
        n = max(1, min(n, 10000))
        return [val] * n
    if t in ("integer", "number", "boolean", "string"):
        return _infer_scalar(schema, t, name)
    # untyped: use name hints
    if name in _STRINGY:
        return "x"
    if name in _LISTY:
        return ["x"]
    return 1.0


def find_inference_endpoint(app) -> str | None:
    openapi = app.openapi()
    paths = openapi.get("paths", {})
    # prefer /predict (non-bulk), then any POST
    candidates = [p for p, m in paths.items() if "post" in m]
    for p in candidates:
        if p.rstrip("/").endswith("/predict") and not p.endswith("/bulk"):
            return p
    for p in candidates:
        if p not in ("/docs", "/openapi.json"):
            return p
    return None


def sample_for_endpoint(app, endpoint: str) -> object | None:
    openapi = app.openapi()
    op = openapi["paths"][endpoint]["post"]
    rb = op.get("requestBody", {})
    content = rb.get("content", {})
    schema = content.get("application/json", {}).get("schema")
    if not schema:
        return None
    components = openapi.get("components", {})
    return build_sample(schema, components)


def has_numeric_prediction(payload: dict) -> bool:
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                if walk(v):
                    return True
        elif isinstance(o, list):
            for v in o:
                if walk(v):
                    return True
        elif isinstance(o, (int, float)):
            return True
        return False
    return walk(payload)


def validate_one(pkg: str, train_path: Path) -> dict:
    result = {"pkg": pkg, "train": "skip", "infer": "skip", "detail": ""}
    tmp = tempfile.mkdtemp(prefix=f"val_{pkg}_")
    env = {**os.environ, "MODEL_DIR": tmp, **SMALL_ENV}
    train_args = []
    try:
        src = train_path.read_text(encoding="utf-8", errors="ignore")
        if '"--n-samples"' in src or "'--n-samples'" in src:
            train_args = ["--n-samples", "2000"]
    except Exception:  # noqa: BLE001
        pass
    # ---- TRAIN ----
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"{pkg}.train", *train_args],
            env=env,
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        if proc.returncode != 0:
            result["train"] = "fail"
            result["detail"] = (proc.stderr or proc.stdout)[-600:]
            return result
        result["train"] = "pass"
    except subprocess.TimeoutExpired:
        result["train"] = "fail"
        result["detail"] = f"training timed out after {TRAIN_TIMEOUT}s"
        return result
    except Exception as e:  # noqa: BLE001
        result["train"] = "fail"
        result["detail"] = f"{type(e).__name__}: {e}"
        return result

    # ---- INFER ----
    try:
        os.environ["MODEL_DIR"] = tmp
        mod = importlib.import_module(f"{pkg}.api")
        app = mod.app
        endpoint = find_inference_endpoint(app)
        if not endpoint:
            result["infer"] = "skip"
            result["detail"] = "no POST inference endpoint found"
            return result
        body = sample_for_endpoint(app, endpoint)
        if body is None:
            result["infer"] = "skip"
            result["detail"] = "could not build sample request body"
            return result
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            h = c.get("/health")
            if h.status_code != 200:
                result["infer"] = "fail"
                result["detail"] = f"/health -> {h.status_code}"
                return result
            r = c.post(endpoint, json=body)
            if r.status_code != 200:
                result["infer"] = "fail"
                result["detail"] = f"{endpoint} -> {r.status_code}: {r.text[:300]}"
                return result
            payload = r.json()
            if not has_numeric_prediction(payload):
                result["infer"] = "fail"
                result["detail"] = f"{endpoint} returned no numeric prediction: {str(payload)[:300]}"
                return result
            result["infer"] = "pass"
    except Exception as e:  # noqa: BLE001
        result["infer"] = "fail"
        result["detail"] = "".join(
            traceback.format_exception_only(type(e), e)
        ).strip()[:600]
    return result


def main() -> int:
    only = os.getenv("VALIDATE_ONLY")
    apps = discover_apps()
    if only:
        apps = [(p, t) for p, t in apps if p == only]
    results: list[dict] = []
    for pkg, train_path in apps:
        sys.stdout.write(f"validating {pkg} ... ")
        sys.stdout.flush()
        res = validate_one(pkg, train_path)
        results.append(res)
        print(f"{res['train']}/{res['infer']}" + (f" ({res['detail'][:80]})" if res['detail'] else ""))

    passed = sum(1 for r in results if r["train"] == "pass" and r["infer"] == "pass")
    trained = sum(1 for r in results if r["train"] == "pass")
    inferred = sum(1 for r in results if r["infer"] == "pass")
    print("\n================ SUMMARY ================")
    print(f"apps            : {len(results)}")
    print(f"train pass      : {trained}/{len(results)}")
    print(f"infer pass      : {inferred}/{len(results)}")
    print(f"full pass       : {passed}/{len(results)}")
    print("\n--- failures ---")
    for r in results:
        if r["train"] != "pass" or r["infer"] != "pass":
            print(f"* {r['pkg']:40s} train={r['train']} infer={r['infer']}")
            print(f"    {r['detail'][:500]}")

    report = Path("VALIDATION_REPORT.md")
    with report.open("w") as f:
        f.write("# Training + Inference Validation Report\n\n")
        f.write(f"- apps: {len(results)}\n")
        f.write(f"- train pass: {trained}/{len(results)}\n")
        f.write(f"- infer pass: {inferred}/{len(results)}\n")
        f.write(f"- full pass: {passed}/{len(results)}\n\n")
        f.write("| app | train | infer | detail |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['pkg']} | {r['train']} | {r['infer']} | {r['detail'][:120].replace('|','/')} |\n")
    print(f"\nreport written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
