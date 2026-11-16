"""Stream processing loop shared by ADAZOR and OCDD (Algorithm 1 of the paper).

1. The first ``n`` examples form the window ``W``; the predictor is fitted
   on it and the monitor is told the inputs of ``W`` (``monitor.fit``). For
   ADAZOR and OCDD (:class:`adazor.monitors.OutlierRateMonitor`) this fits the
   outlier detector and computes the reference indicators.
2. Each new example enters ``W`` (the oldest one leaves) and its input is
   passed to ``monitor.update``, which decides whether to raise an alert.
3. On an alert, only the ``floor(r * n)`` most recent items of ``W`` are kept,
   ``s = n - floor(r * n)`` new examples are read to fill it again, the
   predictor is refitted on ``W`` from scratch and ``monitor.fit`` is called
   again.

Between alerts the predictor is not updated.

Two accuracy measures are recorded:

* *prequential accuracy*: each example read after the initial window is
  predicted by the model in force **before** being used for anything else
  (including the examples read while refilling the window, which are
  predicted by the model about to be replaced);
* *window accuracy*: accuracy of the model in force on the examples currently
  in ``W``, as in the paper. Right after a retraining this is computed on the
  training data, hence it is optimistic for the following ``n`` steps. It is
  recorded every ``eval_every`` steps (``0`` disables it) and, at each alert,
  just before retraining.

Time ``t`` is the number of examples read from the stream so far, so that the
example read at time ``t`` is the ``t``-th example of the stream.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import Model
from .monitors import Monitor

__all__ = ["RunResult", "run_stream"]


@dataclass
class RunResult:
    n_read: int = 0
    alerts: list[int] = field(default_factory=list)
    prequential_correct: int = 0
    prequential_total: int = 0
    window_accuracy: list[tuple[int, float]] = field(default_factory=list)
    alert_window_accuracy: list[tuple[int, float]] = field(default_factory=list)
    truncated_refill: bool = False

    @property
    def n_alerts(self) -> int:
        return len(self.alerts)

    @property
    def prequential_accuracy(self) -> float:
        return self.prequential_correct / self.prequential_total if self.prequential_total else float("nan")

    @property
    def mean_window_accuracy(self) -> float:
        return float(np.mean([a for _, a in self.window_accuracy])) if self.window_accuracy else float("nan")

    def to_dict(self) -> dict:
        return {
            "n_read": self.n_read,
            "alerts": self.alerts,
            "n_alerts": self.n_alerts,
            "prequential_accuracy": self.prequential_accuracy,
            "mean_window_accuracy": self.mean_window_accuracy,
            "window_accuracy": self.window_accuracy,
            "alert_window_accuracy": self.alert_window_accuracy,
            "truncated_refill": self.truncated_refill,
        }


def _accuracy(model: Model, X: deque, y: deque) -> float:
    pred = model.predict(np.asarray(X))
    return float(np.mean([p == t for p, t in zip(pred, y)]))


def run_stream(
    stream: Iterable[tuple[np.ndarray, Any]],
    *,
    n: int,
    r: float,
    monitor: Monitor,
    model_factory: Callable[[], Model],
    max_examples: int | None = None,
    eval_every: int = 1,
) -> RunResult:
    """Process a stream according to Algorithm 1 and return what happened.

    ``model_factory`` is called at the beginning and at every retraining, so
    that each fit starts from a fresh model; the monitor is refitted at the
    same times.
    """
    if n < 2:
        raise ValueError("the window must contain at least two examples")
    if not 0.0 <= r < 1.0:
        raise ValueError("r must lie in [0, 1)")
    keep = math.floor(r * n)
    s = n - keep

    source = iter(stream)
    res = RunResult()

    def read() -> tuple[np.ndarray, Any] | None:
        if max_examples is not None and res.n_read >= max_examples:
            return None
        try:
            x, y = next(source)
        except StopIteration:
            return None
        res.n_read += 1
        return np.asarray(x, dtype=float), y

    X: deque = deque(maxlen=n)
    Y: deque = deque(maxlen=n)
    for _ in range(n):
        item = read()
        if item is None:
            raise ValueError("the stream is shorter than the window")
        X.append(item[0])
        Y.append(item[1])

    def refit() -> Model:
        model = model_factory().fit(np.asarray(X), np.asarray(Y))
        monitor.fit(np.asarray(X))
        return model

    model = refit()
    if eval_every:
        res.window_accuracy.append((res.n_read, _accuracy(model, X, Y)))

    while True:
        item = read()
        if item is None:
            break
        x, y = item
        res.prequential_correct += bool(model.predict_one(x) == y)
        res.prequential_total += 1
        X.append(x)
        Y.append(y)

        if monitor.update(x):
            t_alert = res.n_read
            res.alerts.append(t_alert)
            res.alert_window_accuracy.append((t_alert, _accuracy(model, X, Y)))
            kept = list(zip(X, Y))[n - keep:] if keep else []
            fresh = []
            for _ in range(s):
                item = read()
                if item is None:
                    break
                x, y = item
                res.prequential_correct += bool(model.predict_one(x) == y)
                res.prequential_total += 1
                fresh.append((x, y))
            if len(fresh) < s:          # the stream ended while refilling
                res.truncated_refill = True
                break
            X.clear()
            Y.clear()
            for x, y in kept + fresh:
                X.append(x)
                Y.append(y)
            model = refit()
            if eval_every:
                res.window_accuracy.append((res.n_read, _accuracy(model, X, Y)))
        elif eval_every and (res.n_read - n) % eval_every == 0:
            res.window_accuracy.append((res.n_read, _accuracy(model, X, Y)))

    return res
