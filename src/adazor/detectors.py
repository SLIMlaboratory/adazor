"""Drift detectors operating on a window of binary outlier indicators.

Every detector follows the same protocol:

* ``reset(outliers)`` is called after each (re)training with the indicators of
  the training window, which form the reference sample;
* ``detect(outliers)`` is called at every step with the indicators of the
  current window and returns ``True`` when a drift alert must be raised.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from scipy.stats import norm

__all__ = [
    "DriftDetector",
    "NoDriftDetector",
    "ThresholdDriftDetector",
    "ZTestDriftDetector",
    "effective_level",
    "min_detectable_shift",
    "window_size_for_shift",
]


def _as_bool_array(outliers: Iterable[bool]) -> np.ndarray:
    return np.fromiter((bool(o) for o in outliers), dtype=bool)


class DriftDetector(ABC):
    """Base class for detectors working on windows of outlier indicators."""

    @abstractmethod
    def reset(self, outliers: Iterable[bool]) -> None:
        """Store the reference sample (indicators of the training window)."""

    @abstractmethod
    def detect(self, outliers: Iterable[bool]) -> bool:
        """Return ``True`` if the current window signals a drift."""

    @abstractmethod
    def message(self) -> str:
        """Short human-readable description (e.g. for plot labels)."""


class NoDriftDetector(DriftDetector):
    """Detector that never raises an alert (no-adaptation baseline)."""

    def reset(self, outliers: Iterable[bool]) -> None:
        return None

    def detect(self, outliers: Iterable[bool]) -> bool:
        return False

    def message(self) -> str:
        return "No drift detection"


class ThresholdDriftDetector(DriftDetector):
    """OCDD decision rule: alert when the outlier rate in the window exceeds ``f``."""

    def __init__(self, f: float = 0.1) -> None:
        if not 0.0 <= f <= 1.0:
            raise ValueError("f must lie in [0, 1]")
        self.f = float(f)

    def reset(self, outliers: Iterable[bool]) -> None:
        return None

    def detect(self, outliers: Iterable[bool]) -> bool:
        outs = _as_bool_array(outliers)
        if outs.size == 0:
            return False
        return bool(outs.mean() > self.f)

    def message(self) -> str:
        return f"Threshold drift detection (f={self.f})"


class ZTestDriftDetector(DriftDetector):
    """ADAZOR decision rule: two-sided two-proportion test against the reference.

    Parameters
    ----------
    alpha:
        Nominal significance level of each test.
    variance:
        ``"conservative"`` (default) bounds the variance of the difference of
        the two rates through ``p(1-p) <= 1/4``: an alert is raised when
        ``|p_t - p_0| > z_{1-alpha/2} * sqrt(1/(4 n_0) + 1/(4 n_t))``, which for
        windows of equal size ``n`` reduces to ``z_{1-alpha/2} / sqrt(2n)``.
        This is the rule used in the paper; its actual level is
        ``effective_level(alpha, p) <= alpha``.
        ``"pooled"`` uses the standard pooled estimate of the variance.
    """

    VARIANCES = ("conservative", "pooled")

    def __init__(self, alpha: float = 0.05, variance: str = "conservative") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if variance not in self.VARIANCES:
            raise ValueError(f"variance must be one of {self.VARIANCES}")
        self.alpha = float(alpha)
        self.variance = variance
        self._z = float(norm.ppf(1 - self.alpha / 2))
        self.is_built = False
        self.tot_orig = 0
        self.n_orig = 0

    def reset(self, outliers: Iterable[bool]) -> None:
        outs = _as_bool_array(outliers)
        if outs.size == 0:
            raise ValueError("reset requires a non-empty reference window")
        self.tot_orig = int(outs.sum())
        self.n_orig = int(outs.size)
        self.is_built = True

    @property
    def baseline_rate(self) -> float:
        if not self.is_built:
            raise RuntimeError("ZTestDriftDetector has not been reset.")
        return self.tot_orig / self.n_orig

    def threshold(self, n: int, p_current: float | None = None) -> float:
        """Smallest absolute difference of rates that triggers an alert."""
        if self.variance == "conservative":
            return self._z * math.sqrt(0.25 / self.n_orig + 0.25 / n)
        p_bar = (self.tot_orig + p_current * n) / (self.n_orig + n)
        return self._z * math.sqrt(p_bar * (1 - p_bar) * (1 / self.n_orig + 1 / n))

    def detect(self, outliers: Iterable[bool]) -> bool:
        if not self.is_built:
            raise RuntimeError("ZTestDriftDetector has not been reset.")
        outs = _as_bool_array(outliers)
        n = int(outs.size)
        if n == 0:
            return False
        p_current = outs.sum() / n
        diff = abs(p_current - self.baseline_rate)
        threshold = self.threshold(n, p_current)
        if threshold == 0.0:
            # pooled variance with both samples constant: drift iff they differ
            return bool(diff > 0)
        return bool(diff > threshold)

    def message(self) -> str:
        return rf"Z-test drift detection ($\alpha={self.alpha}$, {self.variance})"


# ---------------------------------------------------------------------------
# Closed-form properties of the conservative rule (Sect. 4.3 of the paper)
# ---------------------------------------------------------------------------

def effective_level(alpha: float, p: float) -> float:
    """Actual per-test level of the conservative rule when the outlier rate is ``p``.

    Assumes two independent windows of equal size under the null hypothesis and
    the normal approximation; it does not depend on the window size.
    """
    if not 0.0 < p < 1.0:
        return 0.0
    z = norm.ppf(1 - alpha / 2)
    return float(2 * (1 - norm.cdf(z / (2 * math.sqrt(p * (1 - p))))))


def min_detectable_shift(n: int, alpha: float, p: float | None = None) -> float:
    """Smallest difference of outlier rates triggering an alert with windows of size ``n``.

    With ``p=None`` the conservative bound is used, otherwise the exact variance
    at outlier rate ``p``.
    """
    z = norm.ppf(1 - alpha / 2)
    if p is None:
        return float(z / math.sqrt(2 * n))
    return float(z * math.sqrt(2 * p * (1 - p) / n))


def window_size_for_shift(delta: float, alpha: float, p: float | None = None) -> int:
    """Smallest window size for which a shift ``delta`` triggers an alert."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    z = norm.ppf(1 - alpha / 2)
    var = 0.25 if p is None else p * (1 - p)
    return math.ceil(2 * var * z**2 / delta**2)
