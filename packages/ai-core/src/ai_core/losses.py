"""Reference loss functions (framework-agnostic, NumPy-backed).

These are lightweight reference implementations intended for teaching and
quick experimentation. Production apps in this monorepo implement their own
domain-specific training; import these only where a simple, dependency-free
loss is convenient.
"""

from __future__ import annotations

import numpy as np


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def bce_loss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0 - eps)
    return float(
        -np.mean(
            y_true * np.log(y_pred)
            + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
    )


def cross_entropy_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    rows = np.arange(len(y_true))
    return float(-np.mean(np.log(probs[rows, y_true] + 1e-12)))
