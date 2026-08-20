"""Data drift and model drift detection for MLOps pipelines."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    feature_name: str
    drift_score: float
    p_value: float
    is_drift: bool
    threshold: float
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "is_drift": self.is_drift,
            "threshold": self.threshold,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "reference_std": self.reference_std,
            "current_std": self.current_std,
        }


class DriftDetector:
    """Detect data drift between reference and current distributions.

    Uses:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-squared test for categorical/binary features
    - Population Stability Index (PSI) as a robust drift metric
    """

    def __init__(
        self,
        feature_names: list[str],
        feature_types: dict[str, str] | None = None,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
    ):
        self.feature_names = feature_names
        self.feature_types = feature_types or {}
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold

    def detect_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> list[DriftResult]:
        """Detect drift between reference and current data.

        Args:
            reference: Reference (training) data, shape (n_samples, n_features)
            current: Current (inference) data, shape (n_samples, n_features)

        Returns:
            List of DriftResult for each feature
        """
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)

        results = []
        for i, name in enumerate(self.feature_names):
            ref_col = reference[:, i]
            cur_col = current[:, i]

            ftype = self.feature_types.get(name, "float")

            if ftype == "binary" or ftype == "categorical":
                result = self._detect_categorical_drift(name, ref_col, cur_col)
            else:
                result = self._detect_continuous_drift(name, ref_col, cur_col)

            results.append(result)

        return results

    def _detect_continuous_drift(
        self, name: str, ref: np.ndarray, cur: np.ndarray
    ) -> DriftResult:
        """Detect drift for continuous features using KS test and PSI."""
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)

        # Population Stability Index
        psi = self._compute_psi(ref, cur)

        # Combined drift score (weighted)
        drift_score = max(ks_stat, psi)

        is_drift = ks_pvalue < self.ks_threshold or psi > self.psi_threshold

        return DriftResult(
            feature_name=name,
            drift_score=float(drift_score),
            p_value=float(ks_pvalue),
            is_drift=bool(is_drift),
            threshold=self.ks_threshold,
            reference_mean=float(np.mean(ref)),
            current_mean=float(np.mean(cur)),
            reference_std=float(np.std(ref)),
            current_std=float(np.std(cur)),
        )

    def _detect_categorical_drift(
        self, name: str, ref: np.ndarray, cur: np.ndarray
    ) -> DriftResult:
        """Detect drift for categorical/binary features using chi-squared test."""
        # Get unique values
        all_values = np.unique(np.concatenate([ref, cur]))
        if len(all_values) < 2:
            # Only one category - check proportion change
            ref_prop = np.mean(ref == all_values[0])
            cur_prop = np.mean(cur == all_values[0])
            drift_score = abs(ref_prop - cur_prop)
            is_drift = drift_score > 0.2
            return DriftResult(
                feature_name=name,
                drift_score=float(drift_score),
                p_value=1.0,
                is_drift=bool(is_drift),
                threshold=0.2,
                reference_mean=float(ref_prop),
                current_mean=float(cur_prop),
                reference_std=0.0,
                current_std=0.0,
            )

        # Chi-squared test
        ref_counts = np.array([np.sum(ref == v) for v in all_values])
        cur_counts = np.array([np.sum(cur == v) for v in all_values])

        # Normalize to proportions
        ref_props = ref_counts / len(ref)
        cur_props = cur_counts / len(cur)

        # Chi-squared test
        chi2_stat, chi2_pvalue = stats.chisquare(cur_counts, f_exp=ref_counts)

        # PSI for categorical
        psi = self._compute_psi_categorical(ref_props, cur_props)

        drift_score = max(chi2_stat / 100, psi)
        is_drift = chi2_pvalue < self.ks_threshold or psi > self.psi_threshold

        return DriftResult(
            feature_name=name,
            drift_score=float(drift_score),
            p_value=float(chi2_pvalue),
            is_drift=bool(is_drift),
            threshold=self.ks_threshold,
            reference_mean=float(np.mean(ref)),
            current_mean=float(np.mean(cur)),
            reference_std=float(np.std(ref)),
            current_std=float(np.std(cur)),
        )

    def _compute_psi(self, ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index for continuous data."""
        # Create bins based on reference distribution
        bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_hist, _ = np.histogram(ref, bins=bins)
        cur_hist, _ = np.histogram(cur, bins=bins)

        # Convert to proportions
        ref_props = ref_hist / len(ref)
        cur_props = cur_hist / len(cur)

        return self._compute_psi_categorical(ref_props, cur_props)

    def _compute_psi_categorical(self, ref_props: np.ndarray, cur_props: np.ndarray) -> float:
        """Compute PSI for categorical distributions."""
        # Avoid division by zero
        ref_props = np.clip(ref_props, 1e-6, 1.0)
        cur_props = np.clip(cur_props, 1e-6, 1.0)

        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi)

    def summarize(self, results: list[DriftResult]) -> dict[str, Any]:
        """Summarize drift detection results."""
        drifted = [r for r in results if r.is_drift]
        return {
            "total_features": len(results),
            "drifted_features": len(drifted),
            "drift_ratio": len(drifted) / len(results) if results else 0.0,
            "drifted": [r.to_dict() for r in drifted],
            "all_results": [r.to_dict() for r in results],
        }
