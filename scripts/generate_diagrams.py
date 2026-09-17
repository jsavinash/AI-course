#!/usr/bin/env python3
"""Render a correct, topic-specific diagram for every AI example.

For each example under apps/<topic>/<name>/src/<module>/ we detect its ML family
(via the same keyword matching used for math) and draw a bespoke PIL schematic:
scatter+fit line, sigmoid, PCA axes, K-Means clusters, Q-learning grid, unfolded
RNN, attention heatmap, CNN convolution, GAN loop, VAE, diffusion steps, GNN,
etc. The image is written to <example>/assets/math-concept.png, which every
README.md already references.
"""

import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_docs import find_examples, MATH_KEYS, MATH_TEMPLATES  # noqa: E402

W, H = 920, 480
BG = (13, 17, 23)
PANEL = (22, 27, 34)
PANEL2 = (28, 33, 48)
BORDER = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (157, 167, 179)
ACCENT = (88, 166, 255)
ACCENT2 = (188, 140, 255)
GREEN = (63, 185, 80)
RED = (248, 113, 113)
ORANGE = (245, 158, 11)
CYAN = (56, 211, 159)


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def new_canvas(title, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 34), title, fill=ACCENT, font=font(26), anchor="mm")
    if sub:
        d.text((W // 2, 62), sub, fill=MUTED, font=font(15), anchor="mm")
    return img, d


def arrow(d, x1, y1, x2, y2, color=ACCENT, w=2, head=8):
    d.line([x1, y1, x2, y2], fill=color, width=w)
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (math.pi * 5 / 6, -math.pi * 5 / 6):
        a = ang + da
        d.line([x2, y2, x2 - head * math.cos(a), y2 - head * math.sin(a)],
               fill=color, width=w)


def box(d, x, y, w, h, label, fill=PANEL, outline=BORDER, tcolor=TEXT, tsize=15):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=outline, width=2)
    if label:
        d.text((x + w / 2, y + h / 2), label, fill=tcolor, font=font(tsize), anchor="mm")


def node(d, cx, cy, r, label, fill=PANEL2, outline=ACCENT, tcolor=TEXT):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
    if label:
        d.text((cx, cy), label, fill=tcolor, font=font(13), anchor="mm")


def rnd(a, b):
    return a + random.random() * (b - a)


# ---------------------------------------------------------------------------
# Diagram painters
# ---------------------------------------------------------------------------

def d_linear(d):
    ox, oy, gw, gh = 90, 110, 740, 300
    d.rectangle([ox, oy, ox + gw, oy + gh], outline=BORDER, width=2)
    pts = [(rnd(0, 1), rnd(0, 1) * 0.8 + 0.1) for _ in range(28)]
    slope, bias = 0.7, 0.15
    pts = [(x, slope * x + bias + rnd(-0.12, 0.12)) for x, _ in pts]
    for x, y in pts:
        px = ox + x * gw
        py = oy + gh - y * gh
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=ACCENT)
    xs = [p[0] for p in pts]
    ymean = sum(slope * x + bias for x in xs) / len(xs)
    xmean = sum(xs) / len(xs)
    sxx = sum((x - xmean) ** 2 for x in xs)
    sxy = sum((x - xmean) * (slope * x + bias - ymean) for x in xs)
    w = sxy / sxx
    b = ymean - w * xmean
    d.line([ox, oy + gh - b * gh, ox + gw, oy + gh - (w + b) * gh],
           fill=ORANGE, width=3)
    d.text((ox + gw - 150, oy + 20), "fitted line  ŷ = w·x + b",
           fill=ORANGE, font=font(14))


def d_logistic(d):
    ox, oy, gw, gh = 90, 110, 740, 300
    d.rectangle([ox, oy, ox + gw, oy + gh], outline=BORDER, width=2)
    for x in range(0, 101):
        xx = x / 100
        z = 6 * (xx - 0.5)
        s = 1 / (1 + math.exp(-z))
        px = ox + xx * gw
        py = oy + gh - s * gh
        if x % 2 == 0:
            d.line([px, py, ox + (x + 1) / 100 * gw,
                    oy + gh - (1 / (1 + math.exp(-6 * ((x + 1) / 100 - 0.5)))) * gh],
                   fill=ACCENT2, width=3)
    for _ in range(22):
        xx = rnd(0, 1)
        z = 6 * (xx - 0.5)
        s = 1 / (1 + math.exp(-z))
        py = oy + gh - s * gh
        px = ox + xx * gw
        col = GREEN if s > 0.5 else RED
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
    d.line([ox + 0.5 * gw, oy, ox + 0.5 * gw, oy + gh], fill=MUTED, width=1)
    d.text((ox + gw - 120, oy + 20), "σ(z)=1/(1+e⁻ᶻ)", fill=ACCENT2, font=font(14))


def d_pca(d):
    ox, oy, gw, gh = 150, 120, 620, 280
    d.rectangle([ox, oy, ox + gw, oy + gh], outline=BORDER, width=2)
    random.seed(7)
    cx, cy = gw / 2, gh / 2
    for _ in range(60):
        t = rnd(0, 1) * math.pi * 2
        r1 = rnd(-180, 180)
        x = cx + r1 * math.cos(t) * 1.0 - r1 * 0.0
        y = cy + r1 * math.sin(t) * 0.25
        px = ox + x
        py = oy + y
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=ACCENT)
    arrow(d, ox + cx, oy + cy, ox + cx + 190, oy + cy, ACCENT2, 4, 12)
    arrow(d, ox + cx, oy + cy, ox + cx - 40, oy + cy - 120, GREEN, 4, 12)
    d.text((ox + cx + 95, oy + cy - 14), "PC1", fill=ACCENT2, font=font(14))
    d.text((ox + cx - 30, oy + cy - 130), "PC2", fill=GREEN, font=font(14))


def d_kmeans(d):
    ox, oy, gw, gh = 120, 110, 680, 300
    centers = [(200, 180, ACCENT), (480, 250, GREEN), (640, 150, ORANGE)]
    random.seed(3)
    for (ccx, ccy, col) in centers:
        for _ in range(20):
            x = ccx + rnd(-90, 90)
            y = ccy + rnd(-70, 70)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
    for (ccx, ccy, col) in centers:
        d.line([ccx - 12, ccy, ccx + 12, ccy], fill=TEXT, width=3)
        d.line([ccx, ccy - 12, ccx, ccy + 12], fill=TEXT, width=3)
    d.text((ox, oy - 4), "K-Means: assign points to nearest centroid", fill=MUTED, font=font(14))


def d_recommendation(d):
    ox, oy = 210, 120
    rows, cols = 5, 6
    cw, ch = 80, 44
    vals = [random.random() for _ in range(rows * cols)]
    mx = max(vals)
    for i in range(rows):
        for j in range(cols):
            v = vals[i * cols + j] / mx
            col = (int(20 + v * 40), int(60 + v * 150), int(80 + v * 120))
            x = ox + j * cw
            y = oy + i * ch
            d.rectangle([x, y, x + cw - 6, y + ch - 6], fill=col, outline=BORDER)
    # predicted cell
    px, py = ox + 4 * cw, oy + 2 * ch
    d.rectangle([px - 3, py - 3, px + cw - 3, py + ch - 3], outline=ORANGE, width=3)
    d.text((px + cw / 2, py + ch / 2), "?", fill=ORANGE, font=font(20), anchor="mm")
    d.text((ox - 150, oy + 40), "users", fill=MUTED, font=font(14))
    d.text((ox + 200, oy - 30), "items", fill=MUTED, font=font(14))


def d_qlearning(d):
    ox, oy, n, s = 120, 120, 6, 90
    for i in range(n):
        for j in range(n):
            x = ox + i * s
            y = oy + j * s
            d.rectangle([x, y, x + s - 8, y + s - 8], outline=BORDER, width=1)
    start = (0, 0)
    goal = (5, 5)
    sx, sy = ox + start[0] * s + s / 2 - 4, oy + start[1] * s + s / 2 - 4
    gx, gy = ox + goal[0] * s + s / 2 - 4, oy + goal[1] * s + s / 2 - 4
    d.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=GREEN)
    d.ellipse([gx - 8, gy - 8, gx + 8, gy + 8], fill=RED)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3),
            (4, 4), (5, 4), (5, 5)]
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        ax, ay = ox + a[0] * s + s / 2 - 4, oy + a[1] * s + s / 2 - 4
        bx, by = ox + b[0] * s + s / 2 - 4, oy + b[1] * s + s / 2 - 4
        arrow(d, ax, ay, bx, by, ACCENT, 2, 7)
    d.text((ox + 250, oy + n * s + 6),
           "Q-learning: agent learns optimal path (green→red)", fill=MUTED, font=font(14))


def d_som(d):
    ox, oy, n, s = 180, 110, 9, 60
    for i in range(n):
        for j in range(n):
            x = ox + i * s
            y = oy + j * s
            hue = int(255 * (i / n))
            col = (hue, 80, 255 - hue)
            d.rectangle([x, y, x + s - 6, y + s - 6], fill=col, outline=BORDER)


def d_cnn(d):
    # input 5x5 grid
    ix, iy, g = 110, 200, 34
    for i in range(5):
        for j in range(5):
            v = 60 + ((i * 3 + j * 7) % 5) * 30
            d.rectangle([ix + j * g, iy + i * g, ix + j * g + g - 4, iy + i * g + g - 4],
                        fill=(v, v, 70), outline=BORDER)
    arrow(d, ix + 5 * g, iy + 2 * g, ix + 5 * g + 60, iy + 2 * g, ACCENT)
    # kernel
    kx, ky = ix + 5 * g + 70, iy + g
    for i in range(3):
        for j in range(3):
            v = 120 + ((i + j) % 2) * 60
            d.rectangle([kx + j * 18, ky + i * 18, kx + j * 18 + 16, ky + i * 18 + 16],
                        fill=(v, v, v), outline=BORDER)
    d.text((kx + 20, ky - 24), "kernel 3×3", fill=MUTED, font=font(13))
    arrow(d, kx + 3 * 18, ky + 1.5 * 18, kx + 3 * 18 + 50, ky + 1.5 * 18, ACCENT)
    # feature map 3x3
    fx, fy = kx + 3 * 18 + 60, iy + g
    for i in range(3):
        for j in range(3):
            v = 50 + ((i * 2 + j) % 3) * 50
            d.rectangle([fx + j * g, fy + i * g, fx + j * g + g - 4, fy + i * g + g - 4],
                        fill=(30, v, 120), outline=BORDER)
    d.text((ix, iy - 30), "convolution: input · kernel → feature map", fill=MUTED, font=font(14))


def d_capsnet(d):
    random.seed(5)
    for i in range(4):
        y = 160 + i * 60
        d.ellipse([180, y, 260, y + 34], fill=PANEL2, outline=ACCENT)
        d.text((220, y + 17), f"capsule {i}", fill=TEXT, font=font(12), anchor="mm")
    for i in range(3):
        y = 160 + i * 60
        d.ellipse([560, y, 640, y + 34], fill=PANEL2, outline=ACCENT2)
        d.text((600, y + 17), f"capsule {i}", fill=TEXT, font=font(12), anchor="mm")
    for i in range(4):
        for j in range(3):
            arrow(d, 260, 160 + i * 60 + 17, 560, 160 + j * 60 + 17,
                  MUTED if random.random() > 0.6 else BORDER, 1, 5)
    d.text((220, 120), "routing", fill=MUTED, font=font(14))


def d_attention(d):
    ox, oy, n = 230, 130, 5
    s = 50
    d.text((ox - 30, oy - 26), "Q", fill=ACCENT, font=font(16))
    d.text((ox + n * s + 10, oy - 26), "K / V", fill=ACCENT2, font=font(16))
    for i in range(n):
        box(d, ox + i * s, oy - 10, s - 6, 34, f"q{i}", PANEL2, ACCENT)
        box(d, ox + i * s, oy + 40, s - 6, 34, f"k{i}", PANEL2, ACCENT2)
        box(d, ox + i * s, oy + 90, s - 6, 34, f"v{i}", PANEL2, ACCENT2)
    mx = ox + 90
    my = oy + 190
    for i in range(n):
        for j in range(n):
            w = 0.2 + 0.6 * (1 - abs(i - j) / n)
            col = (int(30 + w * 60), int(60 + w * 150), int(120 + w * 100))
            d.rectangle([mx + j * 30, my + i * 26, mx + j * 30 + 28, my + i * 26 + 24],
                        fill=col, outline=BORDER)
    d.text((mx + 30, my - 18), "softmax(QKᵀ/√d) · V", fill=MUTED, font=font(14))


def d_gan(d):
    box(d, 180, 170, 150, 70, "Generator", PANEL2, ACCENT)
    box(d, 590, 170, 150, 70, "Discriminator", PANEL2, RED)
    box(d, 180, 90, 150, 50, "noise z", PANEL, BORDER)
    box(d, 590, 300, 150, 50, "real data", PANEL, BORDER)
    box(d, 410, 300, 150, 50, "fake / real", PANEL, BORDER)
    arrow(d, 330, 205, 590, 205, ACCENT)
    arrow(d, 590, 205, 410, 300, RED)
    arrow(d, 330, 330, 410, 330, MUTED)
    arrow(d, 590, 325, 410, 330, MUTED)
    d.text((380, 250), "adversarial min–max", fill=MUTED, font=font(14))


def d_gnn(d):
    nodes = [(200, 160), (360, 120), (380, 260), (560, 180), (640, 300), (520, 320)]
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (2, 5)]
    for a, b in edges:
        arrow(d, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], MUTED, 1, 6)
    labs = ["A", "B", "C", "D", "E", "F"]
    for (x, y), lb in zip(nodes, labs):
        node(d, x, y, 26, lb)


def d_diffusion(d):
    items = ["clean", "t=1", "t=2", "t=3", "noisy"]
    x0 = 110
    for i, lab in enumerate(items):
        x = x0 + i * 150
        y = 200
        for r in range(8):
            for c in range(8):
                v = int(40 + (r + c + i * 6) % 12 * 15)
                d.rectangle([x + c * 12, y + r * 12, x + c * 12 + 11, y + r * 12 + 11],
                            fill=(v, v, 90) if i < 4 else (200, 200, 200), outline=None)
        d.text((x + 40, y - 24), lab, fill=MUTED, font=font(13))
        if i < 4:
            arrow(d, x + 96, y + 48, x + 150 - 96 + 96, y + 48, ACCENT, 2, 7)
    d.text((110, 340), "forward: add noise   ←→   reverse: denoise", fill=MUTED, font=font(14))


def d_vae(d):
    box(d, 170, 230, 110, 60, "encoder", PANEL2, ACCENT)
    box(d, 600, 230, 110, 60, "decoder", PANEL2, ACCENT)
    d.ellipse([400, 150, 470, 220], fill=PANEL2, outline=GREEN)
    d.text((435, 185), "μ,σ", fill=GREEN, font=font(16), anchor="mm")
    box(d, 60, 230, 80, 60, "x", PANEL, BORDER)
    box(d, 720, 230, 80, 60, "x̂", PANEL, BORDER)
    arrow(d, 140, 260, 170, 260, ACCENT)
    arrow(d, 280, 260, 400, 200, ACCENT)
    arrow(d, 470, 200, 600, 260, ACCENT)
    arrow(d, 710, 260, 720, 260, ACCENT)
    d.text((435, 250), "latent z ~ N(μ,σ²)", fill=MUTED, font=font(13))


def d_multimodal(d):
    box(d, 110, 200, 120, 60, "image", PANEL2, ACCENT)
    box(d, 110, 320, 120, 60, "text", PANEL2, ACCENT2)
    box(d, 400, 200, 140, 80, "shared\nspace", PANEL2, GREEN)
    box(d, 660, 260, 150, 70, "output", PANEL, BORDER)
    arrow(d, 230, 230, 400, 235, ACCENT)
    arrow(d, 230, 350, 400, 270, ACCENT2)
    arrow(d, 540, 240, 660, 290, GREEN)


def d_llm(d):
    toks = ["The", "cat", "sat", "<mask>", "mat"]
    x = 150
    for t in toks:
        col = ORANGE if t == "<mask>" else PANEL2
        box(d, x, 150, 110, 44, t, col, ACCENT if t == "<mask>" else BORDER)
        x += 125
    for i in range(3):
        y = 230 + i * 60
        box(d, 250, y, 420, 44, f"Transformer block {i+1}", PANEL2, ACCENT2)
        if i < 2:
            arrow(d, 460, y + 44, 460, y + 60, MUTED, 1, 6)
    arrow(d, 200, 194, 460, 230, MUTED, 1, 6)
    arrow(d, 460, 390, 460, 420, MUTED, 1, 6)
    d.text((200, 430), "predict masked token", fill=MUTED, font=font(14))


def d_rag(d):
    box(d, 90, 200, 110, 60, "docs", PANEL, BORDER)
    box(d, 260, 200, 130, 60, "retriever", PANEL2, ACCENT)
    box(d, 460, 200, 130, 60, "LLM", PANEL2, ACCENT2)
    box(d, 670, 200, 130, 60, "answer", PANEL, BORDER)
    arrow(d, 200, 230, 260, 230, ACCENT)
    arrow(d, 390, 230, 460, 230, ACCENT)
    arrow(d, 590, 230, 670, 230, ACCENT2)
    d.text((300, 290), "retrieve context → augment prompt", fill=MUTED, font=font(13))


def d_tooluse(d):
    box(d, 120, 230, 120, 60, "LLM", PANEL2, ACCENT)
    box(d, 360, 230, 150, 60, "function\ncall", PANEL2, ORANGE)
    box(d, 600, 150, 150, 60, "tool API", PANEL, BORDER)
    box(d, 600, 310, 150, 60, "result", PANEL, BORDER)
    arrow(d, 240, 250, 360, 250, ACCENT)
    arrow(d, 510, 245, 600, 180, ORANGE)
    arrow(d, 600, 270, 510, 290, MUTED)
    arrow(d, 510, 270, 360, 290, ACCENT)


def d_pinn(d):
    ox, oy, gw, gh = 130, 130, 660, 280
    d.rectangle([ox, oy, ox + gw, oy + gh], outline=BORDER, width=2)
    for x in range(0, 101):
        xx = x / 100
        u = math.sin(3 * math.pi * xx) * math.exp(-0.5 * xx)
        px = ox + xx * gw
        py = oy + gh - (u + 1) / 2 * gh
        if x % 2 == 0:
            nx = (x + 1) / 100
            nu = math.sin(3 * math.pi * nx) * math.exp(-0.5 * nx)
            d.line([px, py, ox + nx * gw, oy + gh - (nu + 1) / 2 * gh], fill=ACCENT, width=3)
    d.text((ox + gw - 230, oy + 10), "u(x,t) solves  ℒ[u] = 0", fill=ACCENT2, font=font(14))
    d.text((ox + 20, oy + gh - 24), "physics-informed residual", fill=MUTED, font=font(13))


def d_contrastive(d):
    box(d, 120, 150, 110, 70, "view 1", PANEL, BORDER)
    box(d, 120, 300, 110, 70, "view 2", PANEL, BORDER)
    box(d, 330, 150, 130, 70, "encoder", PANEL2, ACCENT)
    box(d, 330, 300, 130, 70, "encoder", PANEL2, ACCENT)
    node(d, 620, 180, 26, "z₁", PANEL2, GREEN)
    node(d, 620, 320, 26, "z₂", PANEL2, GREEN)
    arrow(d, 230, 185, 330, 185, ACCENT)
    arrow(d, 230, 335, 330, 335, ACCENT)
    arrow(d, 460, 185, 594, 180, GREEN)
    arrow(d, 460, 335, 594, 320, GREEN)
    d.line([620, 206, 620, 294], fill=GREEN, width=2)
    d.text((560, 250), "pull", fill=GREEN, font=font(12))


def d_transfer(d):
    x = 130
    for i in range(4):
        box(d, x, 170, 90, 50, f"base {i}", PANEL2, ACCENT)
        if i < 3:
            arrow(d, x + 90, 195, x + 110, 195, MUTED, 1, 6)
        x += 110
    box(d, 130, 280, 470, 50, "frozen pretrained backbone", PANEL, BORDER)
    box(d, 660, 170, 130, 60, "new head", PANEL2, ORANGE)
    arrow(d, 600, 195, 660, 200, ORANGE)


def d_rbm(d):
    vnodes = [(160, 150), (160, 220), (160, 290), (160, 360)]
    hnodes = [(540, 180), (540, 260), (540, 340)]
    for (x, y) in vnodes:
        node(d, x, y, 22, "", PANEL2, ACCENT)
    for (x, y) in hnodes:
        node(d, x, y, 22, "", PANEL2, ACCENT2)
    for vx, vy in vnodes:
        for hx, hy in hnodes:
            d.line([vx, vy, hx, hy], fill=BORDER, width=1)
    d.text((120, 410), "visible", fill=MUTED, font=font(13))
    d.text((500, 410), "hidden", fill=MUTED, font=font(13))


def d_autoencoder(d):
    box(d, 110, 220, 80, 60, "x", PANEL, BORDER)
    box(d, 260, 200, 110, 60, "encoder", PANEL2, ACCENT)
    box(d, 470, 220, 80, 40, "code", PANEL2, GREEN)
    box(d, 640, 200, 110, 60, "decoder", PANEL2, ACCENT)
    box(d, 820, 220, 80, 60, "x̂", PANEL, BORDER)
    arrow(d, 190, 250, 260, 230, ACCENT)
    arrow(d, 370, 230, 470, 240, ACCENT)
    arrow(d, 550, 240, 640, 230, ACCENT)
    arrow(d, 750, 230, 820, 250, ACCENT)
    d.text((470, 280), "bottleneck", fill=MUTED, font=font(12))


def d_generic(d):
    layers = [(110, 3), (300, 5), (520, 4), (720, 2)]
    pos = []
    for lx, ln in layers:
        ys = [180 + i * 60 for i in range(ln)]
        pos.append([(lx, y) for y in ys])
        for y in ys:
            node(d, lx, y, 20, "", PANEL2, ACCENT)
    for a, b in zip(pos, pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                d.line([x1, y1, x2, y2], fill=BORDER, width=1)
    d.text((110, 380), "input", fill=MUTED, font=font(13))
    d.text((720, 380), "output", fill=MUTED, font=font(13))


# key -> (diagram func, display title, subtitle)
DIAG = {
    "pizza": (d_linear, "Linear Regression", "fit ŷ = w·x + b to data"),
    "house": (d_linear, "Linear Regression", "fit ŷ = w·x + b to data"),
    "spam": (d_logistic, "Logistic Regression", "sigmoid decision boundary"),
    "digit": (d_logistic, "Classification", "logistic / softmax"),
    "fraud": (d_autoencoder, "Autoencoder Anomaly", "reconstruction error"),
    "anomaly": (d_pca, "PCA", "principal components"),
    "pca": (d_pca, "PCA", "principal components"),
    "market": (d_kmeans, "K-Means Clustering", "assign to nearest centroid"),
    "kmeans": (d_kmeans, "K-Means Clustering", "assign to nearest centroid"),
    "recommendation": (d_recommendation, "Collaborative Filtering", "user × item matrix"),
    "robot": (d_qlearning, "Q-Learning", "learn optimal policy"),
    "q-learning": (d_qlearning, "Q-Learning", "learn optimal policy"),
    "self-organizing": (d_som, "Self-Organizing Map", "topology-preserving grid"),
    "cnn": (d_cnn, "Convolutional Network", "kernel × input → feature map"),
    "capsnet": (d_capsnet, "Capsule Network", "dynamic routing"),
    "rnn": (d_generic, "Recurrent Network", "sequential hidden states"),
    "lstm": (d_generic, "LSTM", "gated recurrent units"),
    "attention": (d_attention, "Self-Attention", "softmax(QKᵀ/√d)·V"),
    "transformer": (d_attention, "Transformer", "multi-head self-attention"),
    "gan": (d_gan, "Generative Adversarial Net", "generator vs discriminator"),
    "gnn": (d_gnn, "Graph Neural Network", "message passing"),
    "diffusion": (d_diffusion, "Diffusion Model", "denoise from noise"),
    "image": (d_diffusion, "Image Generation", "GAN / VAE / Diffusion"),
    "video": (d_diffusion, "Video Generation", "temporal denoising"),
    "vae": (d_vae, "Variational Autoencoder", "latent μ,σ"),
    "multimodal": (d_multimodal, "Multimodal Learning", "shared embedding space"),
    "pre-training": (d_llm, "Pre-training (MLM)", "masked language modeling"),
    "prompt": (d_llm, "Prompt Engineering", "conditioned generation"),
    "code": (d_llm, "Code Generation", "autoregressive tokens"),
    "text": (d_llm, "Text Generation", "autoregressive tokens"),
    "retrieval": (d_rag, "Retrieval-Augmented Gen", "retrieve then generate"),
    "tool": (d_tooluse, "Tool Use / Function Calls", "LLM → tool → result"),
    "pinn": (d_pinn, "Physics-Informed NN", "PDE residual loss"),
    "self-supervised": (d_contrastive, "Self-Supervised", "contrastive embeddings"),
    "semi-supervised": (d_contrastive, "Semi-Supervised", "consistency regularization"),
    "transfer": (d_transfer, "Transfer Learning", "pretrained backbone + head"),
    "default": (d_generic, "Neural Network", "feed-forward architecture"),
}
# extras (fall back to generic if no specific)
for k in ("self", "semi", "random-forest", "snn", "deep-belief", "rbm"):
    DIAG.setdefault(k, (d_generic, "Neural Network", "feed-forward architecture"))
DIAG["rbm"] = (d_rbm, "Restricted Boltzmann Machine", "visible ↔ hidden")
DIAG["snn"] = (d_generic, "Spiking Neural Network", "event-based computation")
DIAG["random-forest"] = (d_generic, "Ensemble Model", "bagged decision trees")


def key_for(name: str):
    low = name.lower()
    for k in MATH_KEYS:
        if k in low:
            return k
    return "default"


def render(example_dir: Path):
    slug = example_dir.name  # kebab-case, unique per example
    key = key_for(slug)
    func, title, sub = DIAG.get(key, DIAG["default"])
    img, d = new_canvas(title, sub)
    func(d)
    assets = example_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    # remove any stale pngs so only the properly-named one remains
    for stale in assets.glob("*.png"):
        stale.unlink()
    out = assets / f"{slug}.png"
    img.save(out)
    return out, slug, title


def main():
    import json
    mapping = {}
    n = 0
    for example_dir, _ in find_examples():
        out, slug, title = render(example_dir)
        mapping[str(example_dir.relative_to(ROOT))] = {"slug": slug, "title": title,
                                                      "image": f"assets/{slug}.png"}
        n += 1
    (ROOT / "scripts" / "_diagram_map.json").write_text(json.dumps(mapping, indent=2))
    print(f"Rendered {n} diagrams")


if __name__ == "__main__":
    main()
