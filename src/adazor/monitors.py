"""Monitors: label-free drift detectors observing the inputs of the stream.

A monitor is told the content of the window at every (re)training through
``fit(X)``, then sees every input entering the window through ``update(x)``,
which returns ``True`` when a drift alert must be raised.

:class:`OutlierRateMonitor` implements both ADAZOR and OCDD: a one-class model
maps inputs to outlier indicators, and a decision rule from
:mod:`adazor.detectors` is applied to the indicators in the window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

import numpy as np

from .detectors import DriftDetector
from .outliers import OCSVMOutlierDetector

__all__ = ["Monitor", "NullMonitor", "OutlierRateMonitor"]


class Monitor(ABC):
    """Label-free drift monitor."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Start monitoring with ``X`` (the window just used for training) as reference."""

    @abstractmethod
    def update(self, x: np.ndarray) -> bool:
        """Observe the input entering the window; return ``True`` on a drift alert."""

    def describe(self) -> str:
        return type(self).__name__


class NullMonitor(Monitor):
    """Never raises an alert (no-adaptation baseline)."""

    def fit(self, X: np.ndarray) -> None:
        return None

    def update(self, x: np.ndarray) -> bool:
        return False


class OutlierRateMonitor(Monitor):
    """Outlier indicators of a one-class model, tested by a decision rule.

    Parameters
    ----------
    rule:
        ``ZTestDriftDetector`` for ADAZOR, ``ThresholdDriftDetector`` for OCDD.
    outlier_factory:
        Returns a fresh, unfitted outlier detector.
    baseline:
        ``"in_sample"`` (as in the paper): the outlier detector is fitted on the
        whole window and the reference indicators are those of the same points.
        ``"holdout"``: the detector is fitted on a random part of the window and
        the reference indicators are computed on the remaining
        ``holdout_fraction`` of the points, which removes the optimistic bias of
        in-sample indicators.
    seed:
        Seed for the random split used by ``baseline="holdout"``.
    """

    BASELINES = ("in_sample", "holdout")

    def __init__(self, rule: DriftDetector,
                 outlier_factory: Callable[[], OCSVMOutlierDetector] = OCSVMOutlierDetector,
                 baseline: str = "in_sample", holdout_fraction: float = 0.5,
                 seed: int | None = None) -> None:
        if baseline not in self.BASELINES:
            raise ValueError(f"baseline must be one of {self.BASELINES}")
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must lie in (0, 1)")
        self.rule = rule
        self.outlier_factory = outlier_factory
        self.baseline = baseline
        self.holdout_fraction = holdout_fraction
        self.rng = np.random.default_rng(seed)
        self.outlier = None
        self.flags: deque = deque()

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if self.baseline == "in_sample":
            self.outlier = self.outlier_factory().fit(X)
            reference = self.outlier.flags(X)
        else:
            idx = self.rng.permutation(len(X))
            n_hold = max(1, round(self.holdout_fraction * len(X)))
            hold, train = idx[:n_hold], idx[n_hold:]
            self.outlier = self.outlier_factory().fit(X[train])
            reference = self.outlier.flags(X[hold])
        self.rule.reset(reference)
        self.reference_rate = float(np.mean(reference))
        self.flags = deque(self.outlier.flags(X).tolist(), maxlen=len(X))

    def update(self, x: np.ndarray) -> bool:
        if self.outlier is None:
            raise RuntimeError("monitor not fitted")
        self.flags.append(bool(self.outlier.flags(x)[0]))
        return self.rule.detect(self.flags)

    @property
    def outlier_rate(self) -> float:
        return float(np.mean(self.flags)) if self.flags else float("nan")

    def describe(self) -> str:
        return f"{self.rule.message()}, baseline={self.baseline}"
