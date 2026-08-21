"""Neural network layer primitives.

Re-exported from :mod:`ai_core.nn_utils` so the package exposes a stable
``ai_core.layers`` namespace (target layout) without duplicating code.
"""

from ai_core.nn_utils import (
    Activation,
    Deconv2D,
    SimpleCNN,
    SimpleRNN,
)

__all__ = ["SimpleCNN", "Activation", "Deconv2D", "SimpleRNN"]
