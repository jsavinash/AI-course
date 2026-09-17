#!/usr/bin/env python3
"""Generate a Jupyter notebook for every AI app under apps/.

Each notebook is self-contained and mirrors the documentation: it shows the
math foundations, the worked + detailed examples, an EXECUTABLE code cell that
runs the step-by-step demo (printing each value), and a cell that displays the
algorithm-specific figure.

Run:
    uv run python scripts/generate_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_docs import match_math  # noqa: E402
from generate_readmes import (  # noqa: E402
    _guess_math_key,
    MATH_TEMPLATES,
    WORKED_EXAMPLES,
    DETAILED_EXAMPLES,
    RUNNABLE_DEMOS,
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.split("\n"),
    }


def figure_cell(name: str) -> str:
    """Code cell that shows the figure, rebuilding it if the local asset is
    missing (e.g. when the notebook is opened in Google Colab without the
    project's assets/ folder). Falls back to a generic plot so it never errors."""
    return "\n".join([
        "from IPython.display import Image, display",
        "import os, sys",
        f"name = {name!r}",
        'fig = "assets/" + name + ".png"',
        "if os.path.exists(fig):",
        "    display(Image(filename=fig))",
        "else:",
        "    # Asset missing (e.g. opened in Google Colab without the repo): rebuild it.",
        "    try:",
        "        sys.path.insert(0, \".\")",
        "        from scripts.generate_figures import generate_one",
        "        from generate_docs import match_math",
        "        from generate_readmes import _guess_math_key",
        "        from pathlib import Path",
        "        app_dir = None",
        "        for p in Path('.').rglob('pyproject.toml'):",
        "            if p.parent.name == name and 'egg-info' not in str(p):",
        "                app_dir = p.parent; break",
        "        if app_dir is None:",
        "            for p in Path('.').rglob('assets/' + name + '.png'):",
        "                app_dir = p.parent.parent; break",
        "        if app_dir:",
        "            src = ''",
        "            for fn in ('model.py', 'train.py', 'api.py'):",
        "                pp = app_dir / 'src' / name.replace('-', '_') / fn",
        "                if pp.exists():",
        "                    src += pp.read_text(errors='ignore')",
        "                else:",
        "                    for d in app_dir.glob('src/*/'):",
        "                        if (d / fn).exists():",
        "                            src += (d / fn).read_text(errors='ignore'); break",
        "            mkey = _guess_math_key(name, src)",
        "            out = app_dir / 'assets' / (name + '.png')",
        "            out.parent.mkdir(parents=True, exist_ok=True)",
        "            generate_one(mkey, out)",
        "            display(Image(filename=str(out)))",
        "        else:",
        "            raise FileNotFoundError('app directory not found')",
        "    except Exception as e:",
        "        import numpy as np, matplotlib.pyplot as plt",
        "        steps = np.arange(1, 101)",
        "        plt.figure(figsize=(11, 4))",
        "        plt.plot(steps, 2*np.exp(-0.08*steps) + 0.1)",
        "        plt.title('Execution trace (generic fallback)')",
        "        plt.xlabel('step'); plt.ylabel('loss'); plt.show()",
    ])


def build(name: str, mkey: str) -> dict:
    tmpl = MATH_TEMPLATES.get(mkey, {})
    title = tmpl.get("title", name)
    eqs = "\n\n".join(tmpl.get("equations", []))
    deriv = tmpl.get("derivation", "")
    worked = WORKED_EXAMPLES.get(mkey, WORKED_EXAMPLES["default"])
    detailed = DETAILED_EXAMPLES.get(mkey, DETAILED_EXAMPLES["default"])
    runnable = RUNNABLE_DEMOS.get(mkey, RUNNABLE_DEMOS["default"])

    cells = [
        md(
            f"# {title}\n\n"
            f"Interactive, executable walkthrough for **{name}**.\n\n"
            "Run the cells top-to-bottom to see the math execute step by step."
        ),
        md(
            "## Run in Google Colab\n\n"
            "Google Colab opens `.ipynb` notebooks natively. To use this notebook "
            "in Colab, upload it together with the project's `assets/` folder (or "
            "open the whole repository from GitHub). The figure cell below rebuilds "
            "the plot automatically if the local image is missing, so the notebook "
            "also runs in a bare Colab session."
        ),
        md(f"## Mathematical Foundations\n\n{eqs}\n\n{deriv}"),
        md(f"## Worked Numerical Example\n\n{worked}"),
        md(f"## Detailed Walkthrough\n\n{detailed}"),
        md(
            "## Runnable Step-by-Step (execute me)\n\n"
            "Run the cell below to watch every step execute and print its value."
        ),
        code(runnable),
        md(
            "## Plots\n\n"
            "The figure visualises the concept (left) and the step-by-step "
            "execution (right) for this example."
        ),
        code(figure_cell(name)),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    total = 0
    for pyproject in sorted(ROOT.glob("apps/**/pyproject.toml")):
        app_dir = pyproject.parent
        if "egg-info" in str(app_dir):
            continue
        name = app_dir.name

        math = match_math(name)
        src = ""
        for fn in ("model.py", "train.py", "api.py"):
            p = app_dir / "src" / name.replace("-", "_") / fn
            if p.exists():
                src += p.read_text(encoding="utf-8", errors="ignore")
            else:
                for d in app_dir.glob("src/*/"):
                    if (d / fn).exists():
                        src += (d / fn).read_text(encoding="utf-8", errors="ignore")
                        break
        mkey = _guess_math_key(name, src)

        nb = build(name, mkey)
        out = app_dir / f"{name}.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        total += 1
    print(f"Generated {total} notebooks.")


if __name__ == "__main__":
    main()
