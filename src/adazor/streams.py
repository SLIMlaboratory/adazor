"""Data streams: synthetic Gaussian-mixture streams with controlled drift, and
streams read from arrays or CSV files.

All streams are iterators yielding ``(x, y)`` pairs, where ``x`` is a 1-D
``numpy`` array of floats and ``y`` is a label. They keep their position
between calls, so that a stream can be consumed in several chunks (e.g. by
``itertools.islice``) without losing or repeating examples.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = ["ArrayStream", "DriftingMixtureStream", "GaussianMixture", "load_csv"]

Example = tuple[np.ndarray, int]
DriftPoint = int | tuple[int, int]


class GaussianMixture:
    """Mixture of Gaussians with diagonal covariance matrices.

    Parameters
    ----------
    mu:
        Mean vector of each component.
    sigma:
        Diagonal of the covariance matrix (i.e. the **variances**) of each
        component.
    class_:
        Label associated with each component.
    p:
        Mixing probabilities, summing to one.
    """

    def __init__(self, mu: Sequence[Sequence[float]], sigma: Sequence[Sequence[float]],
                 class_: Sequence[int], p: Sequence[float]) -> None:
        if not len(mu) == len(sigma) == len(class_) == len(p):
            raise ValueError("mu, sigma, class_ and p must have the same length")
        if not np.isclose(sum(p), 1.0):
            raise ValueError("mixing probabilities must sum to one")
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        if np.any(self.sigma < 0):
            raise ValueError("variances must be non-negative")
        self.class_ = list(class_)
        self.p = np.asarray(p, dtype=float)

    def draw(self, rng: np.random.Generator) -> Example:
        """Draw one labeled example."""
        i = rng.choice(len(self.p), p=self.p)
        x = rng.normal(self.mu[i], np.sqrt(self.sigma[i]))
        return x, self.class_[i]

    @staticmethod
    def interpolate(a: GaussianMixture, b: GaussianMixture, lam: float) -> GaussianMixture:
        """Mixture whose parameters are ``(1 - lam) * a + lam * b``."""
        if a.class_ != b.class_:
            raise ValueError("mixtures must share the component labels")
        p = (1 - lam) * a.p + lam * b.p
        return GaussianMixture((1 - lam) * a.mu + lam * b.mu,
                               (1 - lam) * a.sigma + lam * b.sigma,
                               a.class_, p / p.sum())


class DriftingMixtureStream:
    """Stream moving through a sequence of Gaussian mixtures.

    Parameters
    ----------
    mixtures:
        The concepts ``P_0, P_1, ...`` visited by the stream.
    drift_points:
        One entry per transition between consecutive mixtures: an integer
        ``k`` for an abrupt drift (examples with index ``t >= k`` follow the
        next mixture), or a pair ``(start, end)`` for a gradual drift, during
        which the parameters are linearly interpolated with weight
        ``(t - start) / (end - start)`` for ``start <= t <= end``.
        Indices ``t`` start from 0.
    seed:
        Seed of the random generator.
    length:
        Optional number of examples, after which the stream stops.
    """

    def __init__(self, mixtures: Sequence[GaussianMixture], drift_points: Sequence[DriftPoint],
                 seed: int | None = None, length: int | None = None) -> None:
        if len(mixtures) < 2:
            raise ValueError("at least two mixtures are needed")
        if len(drift_points) != len(mixtures) - 1:
            raise ValueError("drift_points must have len(mixtures) - 1 entries")
        last = -1
        for d in drift_points:
            start, end = (d, d) if isinstance(d, int) else d
            if start > end or start <= last:
                raise ValueError("drift points must be increasing and non-overlapping")
            last = end
        self.mixtures = list(mixtures)
        self.drift_points = list(drift_points)
        self.rng = np.random.default_rng(seed)
        self.length = length
        self.t = 0

    def mixture_at(self, t: int) -> GaussianMixture:
        """The mixture generating the example with index ``t``."""
        for i, d in enumerate(self.drift_points):
            if isinstance(d, int):
                if t < d:
                    return self.mixtures[i]
            else:
                start, end = d
                if t < start:
                    return self.mixtures[i]
                if t <= end:
                    lam = (t - start) / (end - start) if end > start else 1.0
                    return GaussianMixture.interpolate(self.mixtures[i], self.mixtures[i + 1], lam)
        return self.mixtures[-1]

    def __iter__(self) -> Iterator[Example]:
        return self

    def __next__(self) -> Example:
        if self.length is not None and self.t >= self.length:
            raise StopIteration
        example = self.mixture_at(self.t).draw(self.rng)
        self.t += 1
        return example


class ArrayStream:
    """Stream over the rows of ``X`` and the entries of ``y``, in their order.

    ``shuffle=True`` randomly permutes the examples. This **destroys the
    temporal order of the data and any drift it contains**: use it only to
    build a stationary control stream.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, shuffle: bool = False,
                 seed: int | None = None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if shuffle:
            idx = np.random.default_rng(seed).permutation(len(y))
            X, y = X[idx], y[idx]
        self.X, self.y = X, y
        self.t = 0

    def __len__(self) -> int:
        return len(self.y)

    def __iter__(self) -> Iterator[Example]:
        return self

    def __next__(self) -> Example:
        if self.t >= len(self.y):
            raise StopIteration
        example = self.X[self.t], self.y[self.t].item()
        self.t += 1
        return example


def load_csv(path: str, target: str = "target") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a CSV file with one row per example and a target column.

    Non-numeric feature columns are replaced by integer codes assigned in
    order of first appearance, so that the encoding of an example never depends
    on later examples. The row order of the file is preserved.
    """
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"column '{target}' not found in {path}")
    y = df[target].to_numpy()
    features = df.drop(columns=[target])
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            features[col] = pd.factorize(features[col])[0]
    return features.to_numpy(dtype=float), y, list(features.columns)
