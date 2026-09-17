#!/usr/bin/env python3
"""Generate comprehensive, self-contained HTML documentation for every AI example.

Each example under apps/<topic>/<name>/src/<module>/ gets a DOCUMENTATION.html
following the required 4-section structure:

  1. Mathematical Foundations
  2. Core Logic & Architecture
  3. Detailed Code Walkthrough
  4. Monorepo Integration

The generator analyses the real code (AST + import scan) and reuses the
MATH_TEMPLATES already defined in scripts/generate_readmes.py so the math is
accurate. Output HTML is self-contained (embedded CSS) and uses GitHub-safe
semantic tags (<details>, <summary>, <table>, <div>) plus MathJax for equation
rendering in a browser.
"""

import ast
import html
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_readmes import MATH_TEMPLATES, WORKED_EXAMPLES, DETAILED_EXAMPLES, RUNNABLE_DEMOS, _guess_math_key  # noqa: E402

# Priority-ordered keyword -> MATH_TEMPLATES key. Longer/more specific first.
MATH_KEYS = [
    "pizza", "spam", "house", "fraud", "digit", "anomaly", "market",
    "recommendation", "robot", "semi-supervised", "self-supervised", "self-organizing",
    "cnn", "capsnet", "rnn", "lstm", "transformer", "attention", "gan", "vae",
    "diffusion", "multimodal", "pre-training", "transfer", "prompt", "code",
    "text", "image", "video", "retrieval", "tool", "gnn", "pinn", "snn",
    "kmeans", "pca", "autoencoder", "random-forest", "q-learning", "self", "semi",
]


def match_math(name: str) -> dict:
    low = name.lower()
    for k in MATH_KEYS:
        if k in low:
            return MATH_TEMPLATES.get(k, MATH_TEMPLATES["default"])
    return MATH_TEMPLATES["default"]


def find_examples():
    out = []
    for model_file in sorted(ROOT.glob("apps/**/src/**/model.py")):
        if "__pycache__" in str(model_file):
            continue
        src_dir = model_file.parent
        # example folder = two levels up: apps/<topic>/<name>/
        example_dir = src_dir.parent.parent
        out.append((example_dir, src_dir))
    return out


# ---------------------------------------------------------------------------
# Code analysis
# ---------------------------------------------------------------------------

def analyze_module(src_dir: Path):
    info = {"classes": [], "functions": [], "docstrings": {}}
    for py in sorted(src_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in n.args.args if a.arg != "self"]
                        methods.append({"name": n.name, "args": args,
                                        "doc": ast.get_docstring(n)})
                info["classes"].append({"name": node.name, "methods": methods,
                                        "doc": ast.get_docstring(node)})
    return info


def read_source(src_dir: Path, filename: str, limit: int = 400):
    p = src_dir / filename
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) > limit:
        lines = lines[:limit] + ["... (truncated) ..."]
    return "\n".join(lines)


def analyze_integration(src_dir: Path):
    submodules = set()
    for py in src_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"from ai_core\.([a-z_]+)", text):
            submodules.add(m.group(1))
        for m in re.finditer(r"import ai_core\.([a-z_]+)", text):
            submodules.add(m.group(1))
    return sorted(submodules)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --border:#30363d;
  --text:#e6edf3; --muted:#9da7b3; --accent:#58a6ff; --accent2:#bc8cff;
  --green:#3fb950; --mono:'SFMono-Regular',Consolas,'Liberation Mono',monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  line-height:1.7;font-size:16px}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 80px}
header.hero{background:linear-gradient(135deg,#0d1117,#161b22);border-bottom:1px solid var(--border);
  padding:36px 20px}
header.hero .inner{max-width:980px;margin:0 auto}
header.hero h1{margin:0 0 8px;font-size:30px}
header.hero p{margin:0;color:var(--muted)}
h2{margin-top:40px;padding-bottom:8px;border-bottom:1px solid var(--border);color:var(--accent)}
h3{margin-top:28px;color:var(--accent2)}
p{color:var(--text)}
.muted{color:var(--muted)}
.equation{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:12px 16px;margin:12px 0;font-family:var(--mono);overflow-x:auto;color:#fff}
.ascii{background:#010409;border:1px solid var(--border);border-radius:8px;padding:16px;
  font-family:var(--mono);white-space:pre;overflow-x:auto;color:#7ee787;font-size:13px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:16px 20px;margin:16px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
th,td{border:1px solid var(--border);padding:8px 12px;text-align:left}
th{background:var(--panel2)}
code{background:#010409;border:1px solid var(--border);border-radius:4px;padding:1px 6px;
  font-family:var(--mono);font-size:13px;color:#ffa657}
pre{background:#010409;border:1px solid var(--border);border-radius:8px;padding:16px;
  overflow-x:auto}
pre code{background:none;border:none;padding:0;color:#c9d1d9;font-size:13px}
details{border:1px solid var(--border);border-radius:8px;margin:12px 0;background:var(--panel)}
summary{cursor:pointer;padding:12px 16px;font-weight:600;color:var(--accent)}
details[open] summary{border-bottom:1px solid var(--border)}
details pre{max-height:520px;overflow:auto;border:none;border-radius:0}
.badge{display:inline-block;background:var(--panel2);border:1px solid var(--border);
  border-radius:999px;padding:2px 10px;margin:2px;font-size:12px;color:var(--muted)}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:16px 0}
.step{background:var(--panel2);border:1px solid var(--border);border-radius:8px;
  padding:10px 14px;font-size:14px}
.arrow{color:var(--accent)}
img{border:1px solid var(--border);border-radius:8px;max-width:100%}
ul li{margin:6px 0}
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def render_math_section(math: dict, name: str, math_key: str = "default") -> str:
    eqs = math.get("equations", [])
    eq_html = "\n".join(f'<div class="equation">{esc(e)}</div>' for e in eqs)
    deriv = math.get("derivation", "")
    viz = math.get("visualization", "")
    title = math.get("title", name)
    # App-specific worked numerical example (correct for this algorithm).
    worked_text = WORKED_EXAMPLES.get(math_key, WORKED_EXAMPLES["default"])
    worked = (
        '<p class="muted">Concrete forward-pass / update evaluation using the '
        "algorithm's own equations:</p>\n"
        f'<div class="ascii">{esc(worked_text)}</div>'
    )
    detailed_text = DETAILED_EXAMPLES.get(math_key, DETAILED_EXAMPLES["default"])
    detailed = (
        '<p class="muted">A step-by-step, intuitive explanation with concrete '
        "data so the formal equations above become clear:</p>\n"
        f'<div class="ascii">{esc(detailed_text)}</div>'
    )
    runnable_code = RUNNABLE_DEMOS.get(math_key, RUNNABLE_DEMOS["default"])
    runnable = (
        '<p class="muted">Run this self-contained snippet in a Python shell to '
        "watch every step execute and print its value:</p>\n"
        f'<pre><code class="language-python">{esc(runnable_code)}</code></pre>'
    )
    diagram = """
<div class="ascii">   [ Input ] --> ( core transform ) --> [ Output ]
                        |
                  [ activation / loss ]
                        |
                  [ prediction ]</div>
"""
    return f"""
<h2>1. Mathematical Foundations</h2>
<p>This example is grounded in <strong>{esc(title)}</strong>. The equations below
drive every forward and backward pass in the implementation.</p>
{eq_html}
<h3>Derivation</h3>
<p>{esc(deriv)}</p>
<h3>Worked Numerical Example</h3>
{worked}
<h3>Detailed Walkthrough</h3>
{detailed}
<h3>Runnable Step-by-Step (execute me)</h3>
{runnable}
<img src="./assets/{esc(name)}.png" width="680" alt="{esc(title)} diagram">
<p class="muted">Plots of the execution above — left: the concept; right: the
step-by-step computation visualised. {esc(viz)}</p>
<h3>Conceptual Diagram</h3>
{diagram}
"""


def render_architecture(info: dict, name: str) -> str:
    classes = info.get("classes", [])
    cls_rows = []
    for c in classes:
        methods = ", ".join(m["name"] for m in c["methods"]) or "—"
        cls_rows.append(
            f"<tr><td><code>{esc(c['name'])}</code></td><td>{esc(methods)}</td>"
            f"<td>{esc(c['doc'] or '')}</td></tr>"
        )
    cls_table = "\n".join(cls_rows) or '<tr><td colspan="3" class="muted">No classes detected</td></tr>'
    flow = f"""
<div class="flow">
  <span class="step">Raw dataset</span><span class="arrow">→</span>
  <span class="step">load + validate (data.py)</span><span class="arrow">→</span>
  <span class="step">fit / transform (model.py)</span><span class="arrow">→</span>
  <span class="step">evaluate + persist (train.py)</span><span class="arrow">→</span>
  <span class="step">serve (api.py)</span>
</div>"""
    patterns = (
        "Key design choices in this module: a pure-NumPy implementation (no PyTorch/TensorFlow), "
        "schema validation via <code>ai_core.validation</code>, structured JSON logging through "
        "<code>ai_core.logging</code>, Prometheus metrics from <code>ai_core.metrics</code>, and "
        "MLflow/model-registry persistence via <code>ai_core.model_registry</code>. The FastAPI "
        "service wraps the trained model with observability middleware from "
        "<code>ai_core.fastapi_middleware</code>."
    )
    return f"""
<h2>2. Core Logic &amp; Architecture</h2>
<p>The example follows a consistent <strong>data &rarr; train &rarr; evaluate &rarr; serve</strong>
pipeline. Inputs are loaded and validated, transformed by the core algorithm, scored against
held-out data, and exposed through a REST API.</p>
{flow}
<h3>Primary Components</h3>
<table>
<thead><tr><th>Class</th><th>Public methods</th><th>Responsibility</th></tr></thead>
<tbody>{cls_table}</tbody>
</table>
<h3>Data Flow</h3>
<ol>
  <li><strong>Load</strong> — <code>data.py</code> reads the source dataset and splits train/test.</li>
  <li><strong>Validate</strong> — a Pydantic schema guards input shape/dtypes before training.</li>
  <li><strong>Fit / Transform</strong> — <code>model.py</code> applies the mathematics from Section 1.</li>
  <li><strong>Evaluate</strong> — metrics (MSE/RMSE/R², accuracy, etc.) are computed and logged.</li>
  <li><strong>Persist</strong> — weights/artifacts are saved and registered in the model registry.</li>
  <li><strong>Serve</strong> — <code>api.py</code> exposes prediction endpoints with drift detection.</li>
</ol>
<h3>Design Patterns &amp; Performance</h3>
<p>{patterns}</p>
"""


def render_code_walkthrough(src_dir: Path, info: dict) -> str:
    files = ["model.py", "train.py", "data.py", "api.py"]
    blocks = []
    for fn in files:
        src = read_source(src_dir, fn)
        if not src:
            continue
        blocks.append(
            f'<details><summary>{esc(fn)}</summary>\n'
            f'<pre><code>{esc(src)}</code></pre></details>'
        )
    code_blocks = "\n".join(blocks)
    # Highlight a couple of key methods generically.
    highlights = []
    for c in info.get("classes", []):
        for m in c["methods"]:
            if m["name"] in ("fit", "predict", "forward", "train", "evaluate") and m["doc"]:
                highlights.append(
                    f'<div class="card"><h3><code>{esc(c["name"])}.{esc(m["name"])}'
                    f'({esc(", ".join(m["args"]))})</code></h3><p>{esc(m["doc"])}</p></div>'
                )
    hl_html = "\n".join(highlights) or '<p class="muted">No docstring-annotated key methods.</p>'
    return f"""
<h2>3. Detailed Code Walkthrough</h2>
<p>The most important behaviour is summarised below; full source for each module is collapsible
so the page stays readable while remaining self-contained.</p>
{hl_html}
<h3>Source Files</h3>
{code_blocks}
"""


def render_integration(submodules: list) -> str:
    badges = "\n".join(f'<span class="badge">ai_core.{esc(s)}</span>' for s in submodules) or \
        '<span class="badge muted">none detected</span>'
    return f"""
<h2>4. Monorepo Integration</h2>
<p>This example is a first-class consumer of the shared <code>packages/ai-core</code> library.
It reuses the following foundation modules instead of re-implementing infrastructure:</p>
<div>{badges}</div>
<h3>How it plugs in</h3>
<ul>
  <li><strong>Configuration</strong> — 12-factor config from <code>ai_core.config</code>.</li>
  <li><strong>Observability</strong> — structured logging + Prometheus metrics are wired in automatically.</li>
  <li><strong>Validation</strong> — input schema validation prevents bad data reaching the model.</li>
  <li><strong>Registry</strong> — trained artifacts are versioned and registered for reproducible serving.</li>
  <li><strong>Serving</strong> — the FastAPI app mounts shared observability middleware for tracing &amp; metrics.</li>
</ul>
<p class="muted">Because every example shares <code>ai_core</code>, cross-cutting concerns (drift detection,
logging, metrics, model registry) behave identically across the 47 examples in this monorepo.</p>
"""


def render_page(example_dir: Path, src_dir: Path):
    name = example_dir.name
    math = match_math(name)
    math_key = _guess_math_key(name, "")
    info = analyze_module(src_dir)
    submodules = analyze_integration(src_dir)
    title = math.get("title", name)
    body = (
        render_math_section(math, name, math_key)
        + render_architecture(info, name)
        + render_code_walkthrough(src_dir, info)
        + render_integration(submodules)
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(name)} — AI Example Documentation</title>
<script>
  window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }},
    svg: {{ fontCache: 'global' }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>{CSS}</style>
</head>
<body>
<header class="hero"><div class="inner">
  <h1>{esc(name)}</h1>
  <p>{esc(title)} — AI engineering example &middot; part of the MLOps monorepo</p>
</div></header>
<div class="wrap">
{body}
</div>
</body>
</html>
"""
    return html_doc


def main():
    examples = find_examples()
    print(f"Found {len(examples)} example modules")
    for example_dir, src_dir in examples:
        out = render_page(example_dir, src_dir)
        out_path = example_dir / "DOCUMENTATION.html"
        out_path.write_text(out, encoding="utf-8")
        print(f"  wrote {out_path.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
