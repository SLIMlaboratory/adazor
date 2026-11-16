"""Streams used in the experiments, each with its known drift events.

Drift events are intervals ``(start, end)`` expressed, like the alerts of
:func:`adazor.run_stream`, as numbers of examples read: an abrupt drift
before the example with index ``k`` (0-based) is the event ``(k + 1, k + 1)``.
Streams whose drifts are unknown have ``events=None``; stationary streams
have ``events=[]``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from adazor import DriftingMixtureStream, GaussianMixture

__all__ = ["SCENARIOS", "InjectedShiftStream", "mixture_scenario"]


def _gm(mu, sigma, class_=(0, 1), p=(0.5, 0.5)) -> GaussianMixture:
    return GaussianMixture(mu=mu, sigma=sigma, class_=list(class_), p=list(p))


# sigma entries are variances
_S1_P0 = _gm([[0, -1], [1, 1]], [[0.01, 0.2], [0.2, 0.01]])
_S1_P1 = _gm([[0, -1], [-1, -2]], [[0.01, 0.2], [0.2, 0.01]])
_S2_P0 = _gm([[0, 1], [0, 0]], [[0.4, 0.4], [0.4, 0.4]])
_S2_P1 = _gm([[0, 0], [1, 1]], [[0.2, 0.2], [0.2, 0.2]])
# virtual drift: both classes move parallel to the Bayes boundary x0 = 0
_V_P0 = _gm([[-1, 0], [1, 0]], [[0.2, 0.2], [0.2, 0.2]])
_V_P1 = _gm([[-1, 3], [1, 3]], [[0.2, 0.2], [0.2, 0.2]])
# real drift only: the two components swap labels, P(X) is unchanged
_L_P1 = _gm([[1, 0], [-1, 0]], [[0.2, 0.2], [0.2, 0.2]])

GRADUAL = [(2000, 3000), (5000, 6000)]
ABRUPT = [2500, 5500]

SCENARIOS = {
    # name: (P0, P1, drift points, description)
    "setting1": (_S1_P0, _S1_P1, GRADUAL, "paper Setting I, gradual"),
    "setting2": (_S2_P0, _S2_P1, GRADUAL, "paper Setting II, gradual"),
    "setting1_abrupt": (_S1_P0, _S1_P1, ABRUPT, "Setting I concepts, abrupt"),
    "setting2_abrupt": (_S2_P0, _S2_P1, ABRUPT, "Setting II concepts, abrupt"),
    "virtual": (_V_P0, _V_P1, GRADUAL, "inputs move, labeling rule unchanged, gradual"),
    "virtual_abrupt": (_V_P0, _V_P1, ABRUPT, "inputs move, labeling rule unchanged, abrupt"),
    "label_swap_abrupt": (_V_P0, _L_P1, ABRUPT, "labeling rule changes, inputs unchanged, abrupt"),
    "stationary": (_V_P0, _V_P0, None, "no drift"),
}


def _events(drift_points) -> list[tuple[int, int]]:
    out = []
    for d in drift_points:
        start, end = (d, d) if isinstance(d, int) else d
        out.append((start + 1, end + 1))
    return out


def mixture_scenario(name: str, seed: int, length: int = 8000):
    """Stream and drift events of a Gaussian-mixture scenario (P0 -> P1 -> P0)."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}'; choose among {sorted(SCENARIOS)}")
    p0, p1, points, _ = SCENARIOS[name]
    if points is None:
        return DriftingMixtureStream([p0, p0], [length], seed=seed, length=length), []
    stream = DriftingMixtureStream([p0, p1, p0], points, seed=seed, length=length)
    return stream, [e for e in _events(points) if e[0] <= length]


class InjectedShiftStream:
    """Real data with an input shift injected at a known position.

    The examples are split into two halves by the median of one feature. Up to
    ``position`` examples are drawn from the two halves with equal probability
    (i.e. at random from the whole dataset, a stationary stream); afterwards
    the probability of drawing from the upper half grows linearly to
    ``strength`` over ``width`` examples (``width=0``: abrupt shift) and then
    stays constant. Examples are drawn without replacement; the stream stops
    when one of the halves is exhausted or after ``length`` examples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, feature: int, position: int,
                 width: int = 0, strength: float = 0.9, seed: int | None = None,
                 length: int | None = None) -> None:
        X = np.asarray(X, dtype=float)
        if not 0 <= feature < X.shape[1]:
            raise ValueError("feature index out of range")
        if not 0.5 <= strength <= 1.0:
            raise ValueError("strength must lie in [0.5, 1]")
        self.X, self.y = X, np.asarray(y)
        self.rng = np.random.default_rng(seed)
        order = np.argsort(X[:, feature], kind="stable")
        half = len(order) // 2
        self.pools = [list(self.rng.permutation(order[:half])), list(self.rng.permutation(order[half:]))]
        self.position, self.width, self.strength, self.length = position, width, strength, length
        self.t = 0

    @property
    def events(self) -> list[tuple[int, int]]:
        return [(self.position + 1, self.position + self.width + 1)]

    def p_high(self, t: int) -> float:
        if t < self.position:
            return 0.5
        if self.width == 0 or t >= self.position + self.width:
            return self.strength
        return 0.5 + (self.strength - 0.5) * (t - self.position) / self.width

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.length is not None and self.t >= self.length:
            raise StopIteration
        side = int(self.rng.random() < self.p_high(self.t))
        if not self.pools[side]:
            raise StopIteration
        i = self.pools[side].pop()
        self.t += 1
        return self.X[i], self.y[i].item()
