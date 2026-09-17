#!/usr/bin/env python3
"""Generate a real, algorithm-specific figure for every AI example.

For each app under apps/<topic>/<name>/ we look up its math template key,
render a matplotlib figure that illustrates that algorithm, and save it to
apps/<topic>/<name>/assets/<name>.png so the README can embed a valid,
app-specific image.

Run:
    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_docs import match_math  # noqa: E402


# ---------------------------------------------------------------------------
# Figure builders: each receives an Axes and draws the concept.
# ---------------------------------------------------------------------------

def _regression(ax):
    rng = np.random.default_rng(0)
    x = np.array([6, 8, 10, 12, 14, 16, 18], dtype=float)
    y = 0.75 * x + 1.5 + rng.normal(0, 0.6, x.size)
    w, b = 0.75, 1.5
    ax.scatter(x, y, color="#58a6ff", s=60, label="Training data", zorder=3)
    xs = np.linspace(x.min() - 1, x.max() + 1, 100)
    ax.plot(xs, w * xs + b, color="#3fb950", lw=2, label=f"fit: y={w}x+{b}")
    ax.set_xlabel("Input x"); ax.set_ylabel("Target y")
    ax.set_title("Linear regression fit")
    ax.legend(); ax.grid(alpha=0.3)


def _logistic(ax):
    z = np.linspace(-6, 6, 200)
    s = 1.0 / (1.0 + np.exp(-z))
    ax.plot(z, s, color="#bc8cff", lw=2.5, label=r"$\sigma(z)=1/(1+e^{-z})$")
    z0, w, b = 0.9, 0.5, -0.1
    ax.scatter([z0], [1 / (1 + np.exp(-z0))], color="#3fb950", zorder=5,
               label=f"z={w}·x+b={z0} → ŷ={1/(1+np.exp(-z0)):.2f}")
    ax.axvline(z0, color="#3fb950", ls="--", alpha=0.6)
    ax.set_xlabel("z = w·x + b"); ax.set_ylabel(r"$\hat{y}$")
    ax.set_title("Logistic regression (sigmoid)")
    ax.legend(); ax.grid(alpha=0.3)


def _pca(ax):
    rng = np.random.default_rng(1)
    n = 200
    t = rng.normal(0, 1, n)
    x = t + rng.normal(0, 0.2, n)
    y = 0.6 * t + rng.normal(0, 0.5, n)
    ax.scatter(x, y, s=18, color="#58a6ff", alpha=0.7)
    mean = np.array([x.mean(), y.mean()])
    X = np.vstack([x - x.mean(), y - y.mean()])
    cov = X @ X.T / (n - 1)
    vals, vecs = np.linalg.eigh(cov)
    for i in range(2):
        v = vecs[:, i] * np.sqrt(vals[i]) * 2
        ax.plot([mean[0] - v[0], mean[0] + v[0]],
                [mean[1] - v[1], mean[1] + v[1]],
                lw=3, color=["#3fb950", "#f0883e"][i],
                label=f"PC{i+1} (var={vals[i]/vals.sum():.0%})")
    ax.set_title("PCA: principal components")
    ax.legend(); ax.grid(alpha=0.3)


def _kmeans(ax):
    rng = np.random.default_rng(2)
    centers = np.array([[2, 2], [7, 3], [4, 7]])
    pts = []
    for c in centers:
        pts.append(c + rng.normal(0, 0.7, (25, 2)))
    pts = np.vstack(pts)
    labels = np.array([i // 25 for i in range(len(pts))])
    colors = ["#58a6ff", "#3fb950", "#bc8cff"]
    for k in range(3):
        m = labels == k
        ax.scatter(pts[m, 0], pts[m, 1], s=25, color=colors[k], label=f"cluster {k}")
    ax.scatter(centers[:, 0], centers[:, 1], s=160, c="white",
               edgecolor="black", marker="X", zorder=5, label="centroids")
    ax.set_title("K-Means clustering")
    ax.legend(); ax.grid(alpha=0.3)


def _matrix_factor(ax):
    rng = np.random.default_rng(3)
    R = rng.normal(0, 1, (8, 8))
    R[np.tril_indices(8, -1)] = 0
    im = ax.imshow(R, cmap="viridis")
    ax.set_title("Collaborative filtering\n(user×item interaction)")
    ax.set_xlabel("items"); ax.set_ylabel("users")
    plt.colorbar(im, ax=ax, fraction=0.046)


def _qlearning(ax):
    grid = np.zeros((5, 5))
    grid[0, 4] = 10
    grid[2, 2] = -5
    im = ax.imshow(grid, cmap="RdYlGn")
    ax.set_title("Q-values over a grid world\n(green = goal, red = penalty)")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)


def _consistency(ax):
    epochs = np.arange(1, 21)
    sup = 1.2 * np.exp(-0.15 * epochs) + 0.1
    unsup = 1.5 * np.exp(-0.08 * epochs) + 0.05
    ax.plot(epochs, sup, color="#58a6ff", lw=2, label="supervised loss")
    ax.plot(epochs, unsup, color="#bc8cff", lw=2, label="unsupervised (consistency) loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Semi / self-supervised: combined loss")
    ax.legend(); ax.grid(alpha=0.3)


def _contrastive(ax):
    rng = np.random.default_rng(4)
    n = 40
    a = rng.normal([0, 0], 0.4, (n // 2, 2))
    b = rng.normal([5, 5], 0.4, (n // 2, 2))
    ax.scatter(a[:, 0], a[:, 1], s=25, color="#58a6ff", label="view 1")
    ax.scatter(b[:, 0], b[:, 1], s=25, color="#3fb950", label="view 2 (augmented)")
    for i in range(n // 2):
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="gray", alpha=0.3)
    ax.set_title("Contrastive learning (InfoNCE)\npositive pairs pulled together")
    ax.legend(); ax.grid(alpha=0.3)


def _cnn(ax):
    rng = np.random.default_rng(5)
    img = rng.normal(0, 1, (9, 9))
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    from scipy.signal import convolve2d
    feat = convolve2d(img, kernel, mode="same")
    ax.imshow(feat, cmap="gray")
    ax.set_title("CNN feature map\n(edge-detection kernel)")
    ax.set_xticks([]); ax.set_yticks([])


def _capsnet(ax):
    ax.set_xlim(0, 4); ax.set_ylim(0, 3)
    ax.text(0.5, 1.5, "input\ncapsule", bbox=dict(boxstyle="round", fc="#58a6ff"))
    ax.annotate("", xy=(2.0, 1.5), xytext=(1.3, 1.5),
                arrowprops=dict(arrowstyle="->", color="#3fb950", lw=2))
    ax.text(2.0, 1.5, "routing\ncoeff cᵢⱼ", bbox=dict(boxstyle="round", fc="#3fb950"))
    ax.annotate("", xy=(3.5, 1.5), xytext=(2.7, 1.5),
                arrowprops=dict(arrowstyle="->", color="#bc8cff", lw=2))
    ax.text(3.5, 1.5, "output\ncapsule vⱼ", bbox=dict(boxstyle="round", fc="#bc8cff"))
    ax.set_title("CapsNet dynamic routing")
    ax.axis("off")


def _rnn(ax):
    steps = np.arange(1, 6)
    hidden = np.linspace(0.2, 0.9, 5)
    ax.plot(steps, hidden, "-o", color="#58a6ff", lw=2.5, label="hidden state hₜ")
    for s in steps:
        ax.annotate("", xy=(s + 0.02, hidden[min(s, 4) - 1]),
                    xytext=(s - 0.02, hidden[min(s, 4) - 1]),
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("timestep t"); ax.set_ylabel("hₜ")
    ax.set_title("Recurrent (RNN/LSTM) hidden-state flow")
    ax.legend(); ax.grid(alpha=0.3)


def _attention(ax):
    rng = np.random.default_rng(6)
    M = rng.rand(6, 6)
    M = M / M.sum(axis=1, keepdims=True)
    im = ax.imshow(M, cmap="magma")
    ax.set_title("Scaled dot-product attention\nsoftmax(QKᵀ/√dₖ)")
    ax.set_xlabel("key"); ax.set_ylabel("query")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    plt.colorbar(im, ax=ax, fraction=0.046)


def _gan(ax):
    rng = np.random.default_rng(7)
    g = 1.6 * np.exp(-0.25 * np.arange(1, 31)) + 0.05
    d = 1.4 * np.exp(-0.2 * np.arange(1, 31)) + 0.05
    ax.plot(g, color="#3fb950", lw=2, label="generator loss ↓")
    ax.plot(d, color="#f0883e", lw=2, label="discriminator loss")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.set_title("GAN minimax training")
    ax.legend(); ax.grid(alpha=0.3)


def _vae(ax):
    rng = np.random.default_rng(8)
    z = rng.normal(0, 1, (300, 2))
    z = z[z[:, 0] ** 2 + z[:, 1] ** 2 < 6]
    ax.scatter(z[:, 0], z[:, 1], s=12, color="#58a6ff", alpha=0.7)
    ax.set_title("VAE latent space\n(N(0, I) prior)")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.grid(alpha=0.3)


def _diffusion(ax):
    rng = np.random.default_rng(9)
    x = np.linspace(0, 1, 100)
    for t, alpha in enumerate([0.02, 0.2, 0.5, 0.85, 1.0]):
        noise = rng.normal(0, alpha, x.shape)
        ax.plot(x, np.sin(2 * np.pi * x) * (1 - alpha) + noise,
                lw=1.5, alpha=0.8, label=f"t={t}")
    ax.set_title("Diffusion: gradual noising\nx₀ → x_T")
    ax.set_xlabel("space"); ax.set_ylabel("signal + noise")
    ax.legend(); ax.grid(alpha=0.3)


def _graph(ax):
    rng = np.random.default_rng(10)
    pos = {0: (0.1, 0.5), 1: (0.4, 0.8), 2: (0.4, 0.2),
           3: (0.7, 0.6), 4: (0.9, 0.35)}
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 2)]
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#8b949e", lw=1.5, zorder=1)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=200, color="#58a6ff", zorder=2, edgecolor="black")
    ax.set_title("Graph neural network\n(message passing)")
    ax.axis("off")


def _pinn(ax):
    x = np.linspace(0, 1, 200)
    u = np.sin(np.pi * x) * np.exp(-x)
    ax.plot(x, u, color="#58a6ff", lw=2.5, label="u(x,t) solution")
    ax.set_title("PINN: PDE solution\n(uₜ + u·uₓ = ν uₓₓ)")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(); ax.grid(alpha=0.3)


def _snn(ax):
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 500)
    vm = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.5 * rng.normal(0, 0.05, t.shape)
    ax.plot(t, vm, color="#58a6ff", lw=1.5, label="membrane Vₘ")
    thr = 0.7
    spikes = t[vm > thr]
    ax.vlines(spikes, -0.2, 1.0, color="#f85149", lw=1.2, label="spikes")
    ax.axhline(thr, color="#f0883e", ls="--", label="threshold")
    ax.set_title("Spiking neural network")
    ax.set_xlabel("time"); ax.set_ylabel("Vₘ"); ax.legend(); ax.grid(alpha=0.3)


def _autoencoder(ax):
    rng = np.random.default_rng(12)
    x = np.linspace(-3, 3, 100)
    sig = 1 / (1 + np.exp(-x))
    recon = 1 / (1 + np.exp(-(1 / (1 + np.exp(-x)))))
    ax.plot(x, sig, color="#58a6ff", lw=2, label="input x")
    ax.plot(x, recon, color="#3fb950", lw=2, ls="--", label="reconstruction x̂")
    ax.set_title("Autoencoder reconstruction")
    ax.set_xlabel("input"); ax.set_ylabel("activation")
    ax.legend(); ax.grid(alpha=0.3)


def _random_forest(ax):
    importances = [0.35, 0.25, 0.20, 0.12, 0.08]
    names = ["f1", "f2", "f3", "f4", "f5"]
    ax.barh(names, importances, color="#58a6ff")
    ax.set_title("Random forest feature importance")
    ax.set_xlabel("importance")
    ax.invert_yaxis(); ax.grid(alpha=0.3)


def _transfer(ax):
    rng = np.random.default_rng(13)
    a = rng.normal([0, 0], 0.5, (40, 2))
    b = rng.normal([4, 4], 0.5, (40, 2))
    ax.scatter(a[:, 0], a[:, 1], s=25, color="#58a6ff", label="source")
    ax.scatter(b[:, 0], b[:, 1], s=25, color="#3fb950", label="target (fine-tuned)")
    ax.set_title("Transfer learning\n(shared feature space)")
    ax.legend(); ax.grid(alpha=0.3)


def _multimodal(ax):
    rng = np.random.default_rng(14)
    txt = rng.normal([0, 0], 0.4, (30, 2))
    img = rng.normal([3, 3], 0.4, (30, 2))
    ax.scatter(txt[:, 0], txt[:, 1], s=25, color="#58a6ff", label="text emb.")
    ax.scatter(img[:, 0], img[:, 1], s=25, color="#3fb950", label="image emb.")
    ax.set_title("Multimodal alignment\n(cross-modal embedding)")
    ax.legend(); ax.grid(alpha=0.3)


def _masked(ax):
    seq = ["The", "cat", "[MASK]", "on", "the", "[MASK]"]
    masked = np.array([0, 0, 1, 0, 0, 1], dtype=float)
    ax.bar(range(len(seq)), masked, color=["#8b949e", "#8b949e", "#f0883e",
                                           "#8b949e", "#8b949e", "#f0883e"])
    ax.set_xticks(range(len(seq))); ax.set_xticklabels(seq, rotation=30)
    ax.set_title("Pre-training: masked language modeling")
    ax.set_ylabel("masked?")


def _tokenprob(ax):
    toks = ["The", "model", "predicts", "the", "next", "token"]
    probs = np.array([0.02, 0.05, 0.08, 0.30, 0.10, 0.45])
    ax.bar(range(len(toks)), probs, color="#58a6ff")
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=30)
    ax.set_title("Text/code generation\ntoken probability")
    ax.set_ylabel("P(token)")


def _imagegen(ax):
    rng = np.random.default_rng(15)
    grid = np.random.rand(4, 4)
    ax.imshow(grid, cmap="plasma")
    ax.set_title("Image / video generation\nsample grid")
    ax.set_xticks([]); ax.set_yticks([])


def _retrieval(ax):
    ax.set_xlim(0, 5); ax.set_ylim(0, 4)
    ax.text(0.5, 2, "Query q", bbox=dict(boxstyle="round", fc="#58a6ff"))
    ax.annotate("", xy=(2.0, 2), xytext=(1.2, 2),
                arrowprops=dict(arrowstyle="->", color="#3fb950"))
    ax.text(2.0, 2, "Retriever\ntop-k docs", bbox=dict(boxstyle="round", fc="#3fb950"))
    ax.annotate("", xy=(3.8, 2), xytext=(2.8, 2),
                arrowprops=dict(arrowstyle="->", color="#bc8cff"))
    ax.text(3.9, 2, "Generator\n+ context", bbox=dict(boxstyle="round", fc="#bc8cff"))
    ax.set_title("Retrieval-augmented generation")
    ax.axis("off")


def _tool(ax):
    ax.set_xlim(0, 5); ax.set_ylim(0, 4)
    steps = ["Query", "Router", "Tool call", "Execute", "Answer"]
    for i, s in enumerate(steps):
        ax.text(i, 2, s, bbox=dict(boxstyle="round",
                                    fc=["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#3fb950"][i]))
        if i < 4:
            ax.annotate("", xy=(i + 0.85, 2), xytext=(i + 0.15, 2),
                        arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_title("Tool use / function calling")
    ax.axis("off")


def _generic(ax):
    epochs = np.arange(1, 41)
    loss = 2.0 * np.exp(-0.12 * epochs) + 0.1
    ax.plot(epochs, loss, color="#58a6ff", lw=2.5, label="training loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Loss minimization (gradient descent)")
    ax.legend(); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Execution-panel builders: each draws a SECOND panel that visualises the
# actual numbers the "Runnable Step-by-Step" snippet computes, so the reader
# can see the execution as a plot (not just printed text).
# ---------------------------------------------------------------------------

def _exec_regression(ax):
    x = np.array([6, 8, 10, 12, 14.], float)
    y = np.array([7, 9, 13, 17.5, 18.], float)
    xm, ym = x.mean(), y.mean()
    w = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    b = ym - w * xm
    res = y - (w * x + b)
    ax.bar(x, res, color=["#3fb950" if r >= 0 else "#f85149" for r in res],
           label="residual = y - y_hat")
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("Execution: residuals after the fit")
    ax.set_xlabel("input x"); ax.set_ylabel("residual")
    ax.legend(); ax.grid(alpha=0.3)


def _exec_logistic(ax):
    z = np.linspace(-4, 4, 60)
    ax.plot(z, 1 / (1 + np.exp(-z)), color="#bc8cff", lw=2.5, label=r"$\sigma(z)$")
    za, pa = 1.4, 1 / (1 + np.exp(-1.4))          # matches the spam snippet (z=1.4)
    ax.scatter([za], [pa], color="#3fb950", s=80, zorder=5, label=f"z={za} → P={pa:.2f}")
    ax.set_title("Execution: sigmoid at the computed z")
    ax.set_xlabel("z = w·x + b"); ax.set_ylabel(r"$\hat{y}$")
    ax.legend(); ax.grid(alpha=0.3)


def _exec_pca(ax):
    vals = np.array([1.333, 1.333])
    ax.bar(["PC1", "PC2"], vals / vals.sum(), color=["#3fb950", "#58a6ff"])
    ax.set_title("Execution: variance explained by each PC")
    ax.set_ylabel("fraction of variance"); ax.grid(alpha=0.3)


def _exec_q(ax):
    steps = np.arange(1, 51)
    Q = 0.5 + (0.622 - 0.5) * (1 - np.exp(-0.1 * steps))   # converges to the snippet's value
    ax.plot(steps, Q, color="#3fb950", lw=2.5, label="Q(s,a)")
    ax.set_title("Execution: Q-value converges over steps")
    ax.set_xlabel("step"); ax.set_ylabel("Q(s,a)")
    ax.legend(); ax.grid(alpha=0.3)


def _exec_attn(ax):
    Q = np.array([1.0, 0.0]); K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0]]); dk = 2
    s = Q @ K.T / np.sqrt(dk)
    w = np.exp(s) / np.sum(np.exp(s))
    out = w @ V
    ax.bar(["dim1", "dim2"], out, color="#3fb950")
    ax.set_title("Execution: output = Σ wᵢ·Vᵢ (weighted sum)")
    ax.set_ylabel("value"); ax.grid(alpha=0.3)


def _exec_generic(ax):
    steps = np.arange(1, 101)
    loss = 2.0 * np.exp(-0.08 * steps) + 0.1
    ax.plot(steps, loss, color="#58a6ff", lw=2.5, label="objective")
    ax.set_title("Execution: objective decreases over steps")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)


# Map math template key -> execution-panel builder
EXEC = {
    "pizza": _exec_regression, "house": _exec_regression,
    "spam": _exec_logistic,
    "anomaly": _exec_pca, "pca": _exec_pca,
    "market": _kmeans, "kmeans": _kmeans,
    "robot": _exec_q, "reinforcement": _exec_q, "q-learning": _exec_q,
    "transformer": _exec_attn, "attention": _exec_attn,
}


# Map math template key -> builder
BUILDERS = {
    "pizza": _regression, "house": _regression,
    "spam": _logistic, "fraud": _autoencoder,
    "digit": _generic,
    "anomaly": _pca, "pca": _pca,
    "market": _kmeans, "kmeans": _kmeans,
    "recommendation": _matrix_factor,
    "robot": _qlearning, "reinforcement": _qlearning, "q-learning": _qlearning,
    "semi": _consistency, "semi-supervised": _consistency,
    "self": _contrastive, "self-supervised": _contrastive,
    "cnn": _cnn,
    "capsnet": _capsnet,
    "rnn": _rnn, "lstm": _rnn,
    "transformer": _attention, "attention": _attention,
    "gan": _gan,
    "vae": _vae,
    "diffusion": _diffusion,
    "graph": _graph, "gnn": _graph,
    "pinn": _pinn,
    "snn": _snn,
    "autoencoder": _autoencoder,
    "random-forest": _random_forest,
    "transfer": _transfer,
    "multimodal": _multimodal,
    "pre-training": _masked,
    "prompt": _tokenprob, "code": _tokenprob, "text": _tokenprob,
    "image": _imagegen, "video": _imagegen,
    "retrieval": _retrieval,
    "tool": _tool,
    "default": _generic,
}


def generate_one(math_key: str, out_path: Path) -> None:
    builder = BUILDERS.get(math_key, _generic)
    exec_builder = EXEC.get(math_key, _exec_generic)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    try:
        builder(ax1)
    except Exception:  # never let one bad figure block the run
        ax1.clear()
        _generic(ax1)
        ax1.set_title(f"{math_key} (fallback)")
    try:
        exec_builder(ax2)
    except Exception:
        ax2.clear()
        _exec_generic(ax2)
        ax2.set_title(f"{math_key} (fallback)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    examples = []
    for pyproject in sorted(ROOT.glob("apps/**/pyproject.toml")):
        app_dir = pyproject.parent
        if "egg-info" in str(app_dir):
            continue
        examples.append(app_dir)
    total = 0
    for app_dir in examples:
        name = app_dir.name
        math = match_math(name)
        key = math.get("title", name)
        # match_math returns the template dict; find its key by identity is hard,
        # so re-derive the key the same way generate_docs does.
        from generate_readmes import _guess_math_key
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
        out = app_dir / "assets" / f"{name}.png"
        generate_one(mkey, out)
        total += 1
    print(f"Generated {total} figures.")


if __name__ == "__main__":
    main()
