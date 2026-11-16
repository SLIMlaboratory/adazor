import math
from typing import ClassVar

import numpy as np
import pytest

from adazor import (
    ArrayStream,
    NoDriftDetector,
    OCSVMOutlierDetector,
    ThresholdDriftDetector,
    make_model,
    run_stream,
)
from adazor.detectors import DriftDetector
from adazor.models import Model


class FireAt(DriftDetector):
    """Fires at the given stream positions (number of examples read)."""

    def __init__(self, counter, times):
        self.counter, self.times = counter, set(times)

    def reset(self, outliers):
        pass

    def detect(self, outliers):
        return self.counter() in self.times

    def message(self):
        return "test"


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
    res = run_stream(src, n=n, r=r, detector=FireAt(lambda: src.t, [alert_at]),
                     model_factory=Recorder, outlier_factory=OCSVMOutlierDetector, eval_every=0)
    keep = math.floor(r * n)
    s = n - keep
    assert res.alerts == [alert_at]
    # second fit: the keep most recent examples at the alert, then s new ones
    expected = list(range(alert_at - keep, alert_at)) + list(range(alert_at, alert_at + s))
    assert Recorder.fits[1] == [float(v) for v in expected]
    assert len(Recorder.fits[1]) == n


def test_every_example_is_read_once_and_counted():
    src = stream(500)
    res = run_stream(src, n=20, r=0.1, detector=FireAt(lambda: src.t, [50, 120, 121, 300]),
                     model_factory=Recorder, outlier_factory=OCSVMOutlierDetector, eval_every=0)
    assert res.n_read == 500
    assert res.prequential_total == 500 - 20
    # an alert cannot fire while the window is being refilled (121 is skipped)
    assert res.alerts == [50, 120, 300]


def test_truncated_refill_is_reported():
    src = stream(60)
    res = run_stream(src, n=20, r=0.1, detector=FireAt(lambda: src.t, [50]),
                     model_factory=Recorder, outlier_factory=OCSVMOutlierDetector, eval_every=0)
    assert res.truncated_refill and res.n_read == 60


def test_fresh_model_at_each_retraining():
    created = []

    def factory():
        m = make_model("ht")
        created.append(m)
        return m

    src = stream(200)
    run_stream(src, n=20, r=0.1, detector=FireAt(lambda: src.t, [40, 100]),
               model_factory=factory, outlier_factory=OCSVMOutlierDetector, eval_every=0)
    assert len(created) == 3 and len({id(m) for m in created}) == 3


def test_stationary_stream_with_threshold_equal_nu_fires_often():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 2))
    y = (X[:, 0] > 0).astype(int)
    res = run_stream(ArrayStream(X, y), n=250, r=0.1, detector=ThresholdDriftDetector(0.1),
                     model_factory=lambda: make_model("gnb"), outlier_factory=OCSVMOutlierDetector,
                     eval_every=0)
    assert res.n_alerts >= 3
    none = run_stream(ArrayStream(X, y), n=250, r=0.1, detector=NoDriftDetector(),
                      model_factory=lambda: make_model("gnb"), outlier_factory=OCSVMOutlierDetector,
                      eval_every=100)
    assert none.n_alerts == 0 and none.prequential_accuracy > 0.9
    assert none.window_accuracy[0][0] == 250


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        run_stream(stream(10), n=20, r=0.1, detector=NoDriftDetector(), model_factory=Recorder,
                   outlier_factory=OCSVMOutlierDetector)
    with pytest.raises(ValueError):
        run_stream(stream(100), n=20, r=1.0, detector=NoDriftDetector(), model_factory=Recorder,
                   outlier_factory=OCSVMOutlierDetector)
