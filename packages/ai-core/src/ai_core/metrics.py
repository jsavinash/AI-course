"""Production-grade Prometheus metrics collection for MLOps pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

logger = logging.getLogger(__name__)

_instances: dict[str, MetricsCollector] = {}


class MetricsCollector:
    """Collect and expose Prometheus metrics for ML services.

    Uses a singleton pattern per service_name to avoid duplicate
    metric registration in Prometheus.
    """

    def __new__(cls, service_name: str, port: int = 8000) -> MetricsCollector:
        if service_name not in _instances:
            instance = super().__new__(cls)
            _instances[service_name] = instance
        return _instances[service_name]

    def __init__(self, service_name: str, port: int = 8000):
        if not hasattr(self, "_initialized"):
            self.service_name = service_name
            self.port = port

            # Prediction metrics
            self.prediction_counter = Counter(
                f"{service_name}_predictions_total",
                f"Total number of predictions for {service_name}",
                ["model_version"],
            )
            self.prediction_duration = Histogram(
                f"{service_name}_prediction_duration_seconds",
                f"Prediction latency for {service_name}",
                ["model_version"],
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            )
            self.prediction_error_counter = Counter(
                f"{service_name}_prediction_errors_total",
                f"Total number of prediction errors for {service_name}",
                ["model_version", "error_type"],
            )
            self.model_version_gauge = Gauge(
                f"{service_name}_model_version",
                f"Current model version for {service_name}",
            )
            self.model_info = Info(
                f"{service_name}_model_info",
                f"Model metadata for {service_name}",
            )
            self.request_counter = Counter(
                f"{service_name}_requests_total",
                f"Total HTTP requests for {service_name}",
                ["method", "endpoint", "status"],
            )
            self.request_duration = Histogram(
                f"{service_name}_request_duration_seconds",
                f"HTTP request latency for {service_name}",
                ["method", "endpoint"],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self.active_requests = Gauge(
                f"{service_name}_active_requests",
                f"Number of in-flight requests for {service_name}",
            )
            self.feature_drift_gauge = Gauge(
                f"{service_name}_feature_drift_ratio",
                f"Data drift ratio for {service_name}",
            )
            self.data_count = Gauge(
                f"{service_name}_data_count",
                f"Number of data samples observed for {service_name}",
            )

            self._initialized = True

    def start_metrics_server(self) -> bool:
        """Start the Prometheus metrics HTTP server."""
        try:
            start_http_server(self.port)
            return True
        except OSError as e:
            logger.warning(
                f"Metrics server on port {self.port} already in use: {e}. "
                "Metrics will be available at the /metrics endpoint instead."
            )
            return False

    def record_prediction(self, model_version: str, duration: float) -> None:
        """Record a successful prediction."""
        self.prediction_counter.labels(model_version=model_version).inc()
        self.prediction_duration.labels(model_version=model_version).observe(duration)

    def record_error(self, model_version: str, error_type: str = "unknown") -> None:
        """Record a prediction error."""
        self.prediction_error_counter.labels(
            model_version=model_version, error_type=error_type
        ).inc()

    def record_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Record an HTTP request."""
        status_label = str(status)
        self.request_counter.labels(method=method, endpoint=endpoint, status=status_label).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def set_model_version(self, version: str) -> None:
        """Set the current model version.

        Parses a semantic version string (e.g. ``"1.2.3"``) into a numeric
        gauge value using the major and minor components (``1.2``).
        """
        parts = [p for p in version.split(".") if p.isdigit()]
        if not parts:
            self.model_version_gauge.set(0.0)
            return
        numeric = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        try:
            self.model_version_gauge.set(float(numeric))
        except ValueError:
            self.model_version_gauge.set(0.0)

    def set_model_info(self, **kwargs) -> None:
        """Set model metadata info."""
        self.model_info.info(kwargs)

    def set_drift_ratio(self, ratio: float) -> None:
        """Set the current data drift ratio."""
        self.feature_drift_gauge.set(ratio)

    def inc_active_requests(self) -> None:
        """Increment active request count."""
        self.active_requests.inc()

    def dec_active_requests(self) -> None:
        """Decrement active request count."""
        self.active_requests.dec()

    def observe_data(self, n_samples: int) -> None:
        """Record number of data samples observed."""
        self.data_count.set(n_samples)


@dataclass
class PredictionMetrics:
    """Container for prediction metrics."""

    model_name: str
    model_version: str
    predictions: int = 0
    errors: int = 0
    total_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add_prediction(self, latency: float) -> None:
        """Add a prediction latency measurement."""
        self.predictions += 1
        self.total_latency += latency
        self.latencies.append(latency)

    def add_error(self) -> None:
        """Record a prediction error."""
        self.errors += 1

    @property
    def avg_latency(self) -> float:
        """Average prediction latency."""
        if self.predictions == 0:
            return 0.0
        return self.total_latency / self.predictions

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        total = self.predictions + self.errors
        if total == 0:
            return 0.0
        return (self.errors / total) * 100

    def _percentile(self, percentile: float) -> float:
        """Calculate a percentile from recorded latencies."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for reporting."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predictions": self.predictions,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency * 1000,
            "p95_latency_ms": self._percentile(95) * 1000 if self.latencies else 0,
            "p99_latency_ms": self._percentile(99) * 1000 if self.latencies else 0,
        }
