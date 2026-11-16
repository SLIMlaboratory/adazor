"""Outlier detectors mapping inputs to binary indicators."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

__all__ = ["OCSVMOutlierDetector"]


class OCSVMOutlierDetector:
    """One-class SVM with Gaussian kernel, optionally preceded by standardization.

    The scaler is fitted on the same window as the OCSVM, so that no
    information from future examples is used.

    Parameters
    ----------
    nu:
        Upper bound on the fraction of training outliers (scikit-learn ``nu``).
    gamma:
        Kernel parameter, a positive float or ``"scale"`` / ``"auto"`` as in
        scikit-learn. A fixed numeric ``gamma`` is meaningful only when the
        features are on a known scale: use ``standardize=True`` otherwise.
    standardize:
        Whether to standardize the features (on the training window) before
        fitting the OCSVM. Without it, features with large ranges make the
        Gaussian kernel degenerate and almost every new point is flagged as an
        outlier.
    """

    def __init__(self, nu: float = 0.1, gamma: float | str = "auto",
                 standardize: bool = True) -> None:
        self.nu = nu
        self.gamma = gamma
        self.standardize = standardize
        self._model = None

    def fit(self, X: np.ndarray) -> OCSVMOutlierDetector:
        ocsvm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self._model = make_pipeline(StandardScaler(), ocsvm) if self.standardize else ocsvm
        self._model.fit(np.asarray(X, dtype=float))
        return self

    def flags(self, X: np.ndarray) -> np.ndarray:
        """Boolean array, ``True`` for the points regarded as outliers."""
        if self._model is None:
            raise RuntimeError("the outlier detector has not been fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return self._model.predict(X) == -1
