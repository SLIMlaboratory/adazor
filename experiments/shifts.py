"""Multivariate shifts with a known position (third review, M1 and M3).

Two kinds of streams:

* :class:`DatasetShiftStream` draws the rows of a real dataset with
  replacement and, from a given position on, transforms them according to one
  of the shift families of Rabanser et al. (NeurIPS 2019):

  - ``decorrelate``: every feature of a shifted example comes from a different
    random row, so that each feature keeps its distribution while the
    dependence between features is destroyed;
  - ``noise``: Gaussian noise with standard deviation ``severity`` times that
    of each feature;
  - ``translate``: a translation of norm ``severity`` (in units of the
    feature standard deviations) along a random direction;
  - ``knockout``: a random fraction ``severity`` of the features is replaced
    by its mean;
  - ``subpopulation``: one cluster of the data (k-means, chosen at random
    among those with at least 5% of the rows) is absent before the shift and
    makes up a fraction ``severity`` of the examples afterwards.

* :class:`CorrelationFlipStream` simulates ``d`` standard normal features in
  pairs with correlation ``rho``, which becomes ``-rho`` after the shift: every
  feature keeps the same N(0, 1) distribution, while the joint distribution
  moves to a region of low density for the reference.

The label of a transformed example is that of the original row (the labeling
rule does not change). With ``width > 0`` the probability that an example is
shifted grows linearly from 0 to 1 (to ``severity`` for ``subpopulation``)
over ``width`` examples.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["FAMILIES", "CorrelationFlipStream", "DatasetShiftStream"]

FAMILIES = ("decorrelate", "noise", "translate", "knockout", "subpopulation")


def _events(position: int, width: int) -> list[tuple[int, int]]:
    return [(position + 1, position + width + 1)]


def _ramp(t: int, position: int, width: int) -> float:
    if t < position:
        return 0.0
    if width == 0 or t >= position + width:
        return 1.0
    return (t - position) / width


class DatasetShiftStream:
    """Rows of ``X, y`` drawn with replacement, shifted from ``position`` on."""

    def __init__(self, X: np.ndarray, y: np.ndarray, family: str, severity: float,
                 position: int, width: int = 0, length: int = 8000,
                 seed: int | None = None) -> None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y)
        self.family, self.severity = family, float(severity)
        self.position, self.width, self.length = position, width, length
        self.rng = np.random.default_rng(seed)
        self.t = 0
        n, d = self.X.shape
        self.mean, self.sd = self.X.mean(axis=0), self.X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.pre = np.arange(n)
        if family == "translate":
            u = self.rng.normal(size=d)
            self.offset = self.severity * self.sd * u / np.linalg.norm(u)
        elif family == "knockout":
            k = max(1, round(self.severity * d))
            self.knocked = self.rng.choice(d, size=k, replace=False)
        elif family == "subpopulation":
            from sklearn.cluster import KMeans
            z = (self.X - self.mean) / self.sd
            labels = KMeans(n_clusters=5, n_init=4, random_state=0).fit_predict(z)
            sizes = np.bincount(labels, minlength=5)
            eligible = np.flatnonzero(sizes >= 0.05 * n)
            c = self.rng.choice(eligible)
            self.pre = np.flatnonzero(labels != c)
            self.cluster = np.flatnonzero(labels == c)

    @property
    def events(self) -> list[tuple[int, int]]:
        return _events(self.position, self.width)

    def _shifted(self) -> tuple[np.ndarray, object]:
        f = self.family
        if f == "subpopulation":
            i = self.rng.choice(self.cluster) if self.rng.random() < self.severity \
                else self.rng.choice(self.pre)
            return self.X[i], self.y[i]
        i = self.rng.choice(self.pre)
        x = self.X[i].copy()
        if f == "decorrelate":
            x = self.X[self.rng.integers(0, len(self.X), size=x.size), np.arange(x.size)]
        elif f == "noise":
            x = x + self.rng.normal(size=x.size) * self.severity * self.sd
        elif f == "translate":
            x = x + self.offset
        elif f == "knockout":
            x[self.knocked] = self.mean[self.knocked]
        return x, self.y[i]

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.t >= self.length:
            raise StopIteration
        p = _ramp(self.t, self.position, self.width)
        self.t += 1
        if self.rng.random() < p:
            x, y = self._shifted()
        else:
            i = self.rng.choice(self.pre)
            x, y = self.X[i], self.y[i]
        return x, (y.item() if hasattr(y, "item") else y)


class CorrelationFlipStream:
    """Pairs of N(0, 1) features with correlation ``rho``, flipped to ``-rho``.

    The label is 1 when the sum of the first features of the pairs is
    positive, a rule that does not change with the shift. ``flip=False`` gives
    a stationary stream.
    """

    def __init__(self, d: int, rho: float = 0.9, position: int = 2000, width: int = 0,
                 length: int = 8000, flip: bool = True, seed: int | None = None) -> None:
        if d < 2 or d % 2:
            raise ValueError("d must be an even number >= 2")
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")
        self.d, self.rho, self.flip = d, rho, flip
        self.position, self.width, self.length = position, width, length
        self.rng = np.random.default_rng(seed)
        self.t = 0

    @property
    def events(self) -> list[tuple[int, int]]:
        return _events(self.position, self.width) if self.flip else []

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.t >= self.length:
            raise StopIteration
        shifted = self.flip and self.rng.random() < _ramp(self.t, self.position, self.width)
        self.t += 1
        r = -self.rho if shifted else self.rho
        a = self.rng.normal(size=self.d // 2)
        b = r * a + np.sqrt(1 - r * r) * self.rng.normal(size=self.d // 2)
        x = np.empty(self.d)
        x[0::2], x[1::2] = a, b
        return x, int(a.sum() > 0)
