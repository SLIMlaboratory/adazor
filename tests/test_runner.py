import math
from typing import ClassVar

import numpy as np
import pytest

from adazor import (
    ArrayStream,
    NullMonitor,
    OutlierRateMonitor,
    ThresholdDriftDetector,
    make_model,
    run_stream,
)
from adazor.models import Model
from adazor.monitors import Monitor


class FireAt(Monitor):
    """Fires at the given stream positions (number of examples read)."""

    def __init__(self, counter, times):
        self.counter, self.times = counter, set(times)
        self.fits = 0

    def fit(self, X):
        self.fits += 1

    def update(self, x):
        return self.counter() in self.times


class Recorder(Model):
    """Constant model remembering what it was trained on."""

    fits: ClassVar[list] = []

    def fit(self, X, y):
        Recorder.fits.append(np.asarray(X)[:, 0].tolist())
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def stream(length):
    X = np.column_stack([np.arange(length, dtype=float), np.zeros(length)])
    return ArrayStream(X, np.zeros(length, dtype=int))


@pytest.mark.parametrize("n,r", [(10, 0.1), (10, 0.35), (7, 0.5), (10, 0.0)])
def test_refill_uses_n_minus_floor_rn(n, r):
    Recorder.fits = []
    src = stream(200)
    alert_at = 30
    res = run_stream(src, n=n, r=r, monitor=FireAt(lambda: src.t, [alert_at]),
                     model_factory=Recorder, eval_every=0)
    keep = math.floor(r * n)
    s = n - keep
    assert res.alerts == [alert_at]
    # second fit: the keep most recent examples at the alert, then s new ones
    expected = list(range(alert_at - keep, alert_at)) + list(range(alert_at, alert_at + s))
    assert Recorder.fits[1] == [float(v) for v in expected]
    assert len(Recorder.fits[1]) == n


def test_every_example_is_read_once_and_counted():
    src = stream(500)
    monitor = FireAt(lambda: src.t, [50, 120, 121, 300])
    res = run_stream(src, n=20, r=0.1, monitor=monitor, model_factory=Recorder, eval_every=0)
    assert res.n_read == 500
    assert res.prequential_total == 500 - 20
    # an alert cannot fire while the window is being refilled (121 is skipped)
    assert res.alerts == [50, 120, 300]
    assert monitor.fits == 4                 # initial fit + one per alert


def test_truncated_refill_is_reported():
    src = stream(60)
    res = run_stream(src, n=20, r=0.1, monitor=FireAt(lambda: src.t, [50]),
                     model_factory=Recorder, eval_every=0)
    assert res.truncated_refill and res.n_read == 60


def test_fresh_model_at_each_retraining():
    created = []

    def factory():
        m = make_model("ht")
        created.append(m)
        return m

    src = stream(200)
    run_stream(src, n=20, r=0.1, monitor=FireAt(lambda: src.t, [40, 100]),
               model_factory=factory, eval_every=0)
    assert len(created) == 3 and len({id(m) for m in created}) == 3


def test_stationary_stream_with_threshold_equal_nu_fires_often():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 2))
    y = (X[:, 0] > 0).astype(int)
    res = run_stream(ArrayStream(X, y), n=250, r=0.1,
                     monitor=OutlierRateMonitor(ThresholdDriftDetector(0.1)),
                     model_factory=lambda: make_model("gnb"), eval_every=0)
    assert res.n_alerts >= 3
    none = run_stream(ArrayStream(X, y), n=250, r=0.1, monitor=NullMonitor(),
                      model_factory=lambda: make_model("gnb"), eval_every=100)
    assert none.n_alerts == 0 and none.prequential_accuracy > 0.9
    assert none.window_accuracy[0][0] == 250


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        run_stream(stream(10), n=20, r=0.1, monitor=NullMonitor(), model_factory=Recorder)
    with pytest.raises(ValueError):
        run_stream(stream(100), n=20, r=1.0, monitor=NullMonitor(), model_factory=Recorder)


def test_recent_accuracy_is_sliding_prequential():
    # Recorder always predicts 0; labels switch from 0 to 1 after 100 examples
    length, n = 200, 20
    X = np.column_stack([np.arange(length, dtype=float), np.zeros(length)])
    y = (np.arange(length) >= 100).astype(int)
    res = run_stream(ArrayStream(X, y), n=n, r=0.1, monitor=NullMonitor(),
                     model_factory=Recorder, eval_every=5)
    acc = dict(res.recent_accuracy)
    assert min(acc) == 2 * n                 # first value once n predictions exist
    assert acc[100] == 1.0 and acc[110] == 0.5 and acc[120] == 0.0
    assert all(t % 5 == 0 for t in acc)


def test_records_fit_times_and_reference_rates():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 2))
    y = (X[:, 0] > 0).astype(int)
    monitor = OutlierRateMonitor(ThresholdDriftDetector(0.1), baseline="holdout", seed=0)
    res = run_stream(ArrayStream(X, y), n=250, r=0.1, monitor=monitor,
                     model_factory=lambda: make_model("gnb"), eval_every=0)
    assert res.fits[0] == 250 and len(res.fits) == res.n_alerts + 1 - res.truncated_refill
    assert [t + 225 for t in res.alerts[:len(res.fits) - 1]] == res.fits[1:]
    assert len(res.reference_rates) == len(res.fits)
    assert all(0.0 <= p <= 1.0 for p in res.reference_rates)
    assert res.reference_rates[-1] == monitor.reference_rate
    none = run_stream(ArrayStream(X, y), n=250, r=0.1, monitor=NullMonitor(),
                      model_factory=lambda: make_model("gnb"), eval_every=0)
    assert none.fits == [250] and none.reference_rates == []
