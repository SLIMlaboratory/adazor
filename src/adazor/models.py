"""Predictors with a common interface, refitted from scratch on each window.

Every model exposes ``fit(X, y)`` (which discards any previous state),
``predict(X)`` and ``predict_one(x)`` on ``numpy`` arrays. Models are created
through :func:`make_model`, so that every run and every retraining starts from
a fresh instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["MODEL_NAMES", "Model", "RiverModel", "SklearnModel", "make_model"]


class Model(ABC):
    """Binary classifier refitted from scratch on a window of examples."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        """Fit on ``(X, y)``, discarding any previous state."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels of the rows of ``X``."""

    def predict_one(self, x: np.ndarray) -> Any:
        return self.predict(np.asarray(x, dtype=float)[None, :])[0]


def _to_dict(x: np.ndarray) -> dict[str, float]:
    return {f"x_{i}": float(v) for i, v in enumerate(x)}


class RiverModel(Model):
    """Wrapper for an incremental ``river`` classifier, trained by a single
    pass over the window."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.estimator = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RiverModel:
        self.estimator = self.factory()
        for xi, yi in zip(np.asarray(X, dtype=float), np.asarray(y)):
            self.estimator.learn_one(_to_dict(xi), yi.item() if hasattr(yi, "item") else yi)
        return self

    def predict_one(self, x: np.ndarray) -> Any:
        if self.estimator is None:
            raise RuntimeError("model not fitted")
        return self.estimator.predict_one(_to_dict(np.asarray(x, dtype=float)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(xi) for xi in np.asarray(X, dtype=float)], dtype=object)


class SklearnModel(Model):
    """Wrapper for a batch ``scikit-learn`` classifier.

    Windows containing a single class (which most scikit-learn classifiers
    refuse) yield a constant predictor.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.estimator = None
        self.constant = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnModel:
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) == 1:
            self.estimator, self.constant = None, classes[0]
        else:
            self.estimator, self.constant = self.factory().fit(np.asarray(X, dtype=float), y), None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.constant is not None:
            return np.full(len(X), self.constant, dtype=object)
        if self.estimator is None:
            raise RuntimeError("model not fitted")
        return self.estimator.predict(X)


def _river(name: str, seed: int | None) -> Callable[[], Any]:
    from river import forest, linear_model, naive_bayes, tree
    factories = {
        "ht": lambda: tree.HoeffdingTreeClassifier(),
        "gnb": lambda: naive_bayes.GaussianNB(),
        "pa": lambda: linear_model.PAClassifier(C=1.0, mode=1),
        "arf": lambda: forest.ARFClassifier(n_models=10, seed=seed),
        "lr": lambda: linear_model.LogisticRegression(),
    }
    return factories[name]


def _sklearn(name: str, seed: int | None) -> Callable[[], Any]:
    from sklearn.svm import SVC, LinearSVC
    factories = {
        "lsv": lambda: LinearSVC(C=1.0, max_iter=5000, random_state=seed),
        "gsv": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
    }
    return factories[name]


MODEL_NAMES = {
    "ht": "Hoeffding tree (river)",
    "gnb": "Gaussian naive Bayes (river)",
    "pa": "Passive-aggressive classifier (river)",
    "arf": "Adaptive random forest (river)",
    "lr": "Logistic regression (river)",
    "lsv": "Linear support vector classifier (scikit-learn)",
    "gsv": "Gaussian-kernel support vector classifier (scikit-learn)",
}


def make_model(name: str, seed: int | None = None) -> Model:
    """Create a fresh, unfitted model given its short name (see ``MODEL_NAMES``)."""
    if name in ("ht", "gnb", "pa", "arf", "lr"):
        return RiverModel(_river(name, seed))
    if name in ("lsv", "gsv"):
        return SklearnModel(_sklearn(name, seed))
    raise ValueError(f"unknown model '{name}'; choose among {sorted(MODEL_NAMES)}")
