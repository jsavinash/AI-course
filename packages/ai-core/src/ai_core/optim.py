"""Reference optimizers (framework-agnostic, NumPy-backed).

Lightweight SGD helper for teaching and dependency-free experiments. Real
apps use their own optimizers; this is a convenience building block.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def sgd_step(
    params: Mapping[str, np.ndarray],
    grads: Mapping[str, np.ndarray],
    lr: float,
) -> dict:
    return {
        k: np.asarray(v) - lr * np.asarray(grads[k])
        for k, v in params.items()
    }


def adam_step(
    params: Mapping[str, np.ndarray],
    grads: Mapping[str, np.ndarray],
    state: Mapping[str, dict],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple:
    new_params: dict = {}
    new_state: dict = {}
    t = state.get("_t", 0) + 1
    for k, v in params.items():
        g = np.asarray(grads[k])
        s = state.get(k, {"m": np.zeros_like(v), "v": np.zeros_like(v)})
        m = beta1 * s["m"] + (1 - beta1) * g
        v = beta2 * s["v"] + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        new_params[k] = np.asarray(v) - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_state[k] = {"m": m, "v": v}
    new_state["_t"] = t
    return new_params, new_state
