"""Self-supervised learning for server monitoring anomaly detection.

Uses a denoising autoencoder trained on normal server metrics.
The self-supervised signal comes from reconstructing the original (uncorrupted)
input from a corrupted version - no human labels required.
"""

from self_supervised_monitoring.model import DenoisingAutoencoder

__all__ = ["DenoisingAutoencoder"]
