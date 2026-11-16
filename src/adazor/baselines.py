"""Label-free competitors of ADAZOR, implemented as monitors.

* :class:`KSWINMonitor`: one Kolmogorov--Smirnov windowing detector (River's
  ``KSWIN``) per feature, with a Bonferroni correction of the significance
  level; an alert is raised when any feature signals a change.
* :class:`D3Monitor`: a discriminative two-sample detector in the spirit of
  D3 (Gozüaçık et al., CIKM 2019): a classifier is trained to tell the
  reference window from the most recent inputs, and a drift is declared when
  its cross-validated AUC exceeds a threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .monitors import Monitor

__all__ = ["D3Monitor", "KSWINMonitor"]


class KSWINMonitor(Monitor):
    """Per-feature KSWIN with Bonferroni correction.

    Each feature has its own detector, initialized with that feature's values
    in the reference window; ``alpha`` is the overall level, divided by the
    number of features.
    """

    def __init__(self, alpha: float = 0.005, stat_size: int = 30, seed: int | None = None) -> None:
        self.alpha = alpha
        self.stat_size = stat_size
        self.seed = seed
        self.detectors: list = []

    def fit(self, X: np.ndarray) -> None:
        from river.drift import KSWIN

        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if n <= 2 * self.stat_size:
            raise ValueError("the window must contain more than 2 * stat_size examples")
        self.detectors = [
            KSWIN(alpha=self.alpha / d, window_size=n, stat_size=self.stat_size,
                  seed=None if self.seed is None else self.seed + j, window=X[:, j].tolist())
            for j in range(d)
        ]

    def update(self, x: np.ndarray) -> bool:
        if not self.detectors:
            raise RuntimeError("monitor not fitted")
        drift = False
        for det, v in zip(self.detectors, np.asarray(x, dtype=float)):
            det.update(float(v))
            drift = drift or det.drift_detected
        return drift

    def describe(self) -> str:
        return f"KSWIN per feature (alpha={self.alpha}, Bonferroni)"


class D3Monitor(Monitor):
    """Discriminative drift detector.

    Inputs entering the window are buffered; every time ``m = round(rho * n)``
    new inputs have been collected, a classifier is cross-validated on the task
    "reference window (label 0) vs. buffer (label 1)". If the AUC exceeds
    ``tau`` an alert is raised; otherwise the reference slides forward to the
    ``n`` most recent inputs and the buffer is emptied.

    Parameters
    ----------
    rho:
        Size of the buffer relative to the window.
    tau:
        AUC threshold.
    classifier:
        ``"logreg"`` (standardized logistic regression) or ``"tree"``.
    folds:
        Number of cross-validation folds.
    """

    def __init__(self, rho: float = 0.2, tau: float = 0.7, classifier: str = "logreg",
                 folds: int = 5, seed: int | None = None) -> None:
        if classifier not in ("logreg", "tree"):
            raise ValueError("classifier must be 'logreg' or 'tree'")
        self.rho, self.tau, self.classifier, self.folds, self.seed = rho, tau, classifier, folds, seed
        self.reference: np.ndarray | None = None
        self.buffer: list = []
        self.m = 0
        self.last_auc = float("nan")

    def _model(self):
        if self.classifier == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return DecisionTreeClassifier(max_depth=5, random_state=self.seed)

    def fit(self, X: np.ndarray) -> None:
        self.reference = np.asarray(X, dtype=float)
        self.m = max(self.folds, round(self.rho * len(self.reference)))
        self.buffer = []

    def update(self, x: np.ndarray) -> bool:
        if self.reference is None:
            raise RuntimeError("monitor not fitted")
        self.buffer.append(np.asarray(x, dtype=float))
        if len(self.buffer) < self.m:
            return False
        new = np.asarray(self.buffer)
        Z = np.vstack([self.reference, new])
        labels = np.r_[np.zeros(len(self.reference)), np.ones(len(new))]
        cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        scores = cross_val_predict(self._model(), Z, labels, cv=cv, method="predict_proba")[:, 1]
        self.last_auc = float(roc_auc_score(labels, scores))
        if self.last_auc > self.tau:
            return True
        self.reference = Z[-len(self.reference):]
        self.buffer = []
        return False

    def describe(self) -> str:
        return f"D3 (rho={self.rho}, tau={self.tau}, {self.classifier})"
