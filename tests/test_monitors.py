import numpy as np
import pytest

from adazor import (
    D3Monitor,
    KSWINMonitor,
    OCSVMOutlierDetector,
    OutlierRateMonitor,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    make_model,
)


def feed(monitor, X):
    """Index of the first alert while feeding the rows of X, or None."""
    for i, x in enumerate(X):
        if monitor.update(x):
            return i
    return None


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(250, 3))
    same = rng.normal(size=(1000, 3))
    shifted = rng.normal(size=(1000, 3)) + np.array([4.0, 0.0, 0.0])
    return ref, same, shifted


def test_outlier_rate_monitor_tracks_window(data):
    ref, _, shifted = data
    mon = OutlierRateMonitor(ThresholdDriftDetector(0.5))
    mon.fit(ref)
    assert len(mon.flags) == 250 and mon.outlier_rate < 0.2
    for x in shifted[:250]:                       # keep feeding after alerts
        mon.update(x)
    expected = mon.outlier.flags(shifted[:250]).mean()
    assert mon.outlier_rate == pytest.approx(expected) and expected > 0.3


@pytest.mark.parametrize("baseline", ["in_sample", "holdout"])
def test_adazor_detects_shift_and_not_stationarity(data, baseline):
    ref, same, shifted = data
    mon = OutlierRateMonitor(ZTestDriftDetector(0.025), baseline=baseline, seed=0)
    mon.fit(ref)
    assert feed(mon, same) is None
    mon.fit(ref)
    assert feed(mon, shifted) is not None


def test_holdout_baseline_is_out_of_sample():
    rng = np.random.default_rng(1)
    ins, hold = [], []
    for _ in range(30):
        X = rng.normal(size=(250, 2))
        a = OutlierRateMonitor(ZTestDriftDetector(0.025), baseline="in_sample")
        b = OutlierRateMonitor(ZTestDriftDetector(0.025), baseline="holdout", seed=0)
        a.fit(X)
        b.fit(X)
        ins.append(a.rule.baseline_rate)
        hold.append(b.rule.baseline_rate)
    assert abs(np.mean(ins) - 0.1) < 0.01          # nu-property
    assert np.mean(hold) > np.mean(ins) + 0.01      # unseen points: more outliers


def test_outlier_monitor_validates_arguments():
    with pytest.raises(ValueError):
        OutlierRateMonitor(ZTestDriftDetector(), baseline="nope")
    with pytest.raises(RuntimeError):
        OutlierRateMonitor(ZTestDriftDetector(), OCSVMOutlierDetector).update(np.zeros(2))


def test_kswin_monitor(data):
    ref, same, shifted = data
    mon = KSWINMonitor(alpha=0.005, seed=0)
    mon.fit(ref)
    assert len(mon.detectors) == 3
    assert feed(mon, shifted[:300]) is not None
    mon.fit(ref)
    first = feed(mon, same[:300])
    assert first is None or first > 100             # rare false alarms only


def test_d3_monitor(data):
    ref, same, shifted = data
    mon = D3Monitor(rho=0.2, tau=0.7, seed=0)
    mon.fit(ref)
    assert feed(mon, same[:500]) is None
    assert mon.last_auc < 0.7
    mon.fit(ref)
    assert feed(mon, shifted[:100]) == 49            # first check after m = 50 inputs
    assert mon.last_auc > 0.9


def test_standardized_model_is_scale_invariant():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 1] > 0).astype(int)              # the label depends on the tiny feature
    Xs = X * np.array([1e4, 1e-3])
    plain = make_model("pa", seed=0).fit(Xs[:300], y[:300])
    std = make_model("pa", seed=0, standardize=True).fit(Xs[:300], y[:300])
    acc = lambda m: np.mean([p == t for p, t in zip(m.predict(Xs[300:]), y[300:])])
    assert acc(std) > 0.9
    assert acc(std) > acc(plain) + 0.2
    assert std.predict_one(Xs[300]) in (0, 1)


def test_unseen_current_window_excludes_fitted_points(data):
    ref, same, _ = data
    mon = OutlierRateMonitor(ZTestDriftDetector(0.025, variance="pooled"), baseline="holdout",
                             current="unseen", seed=0)
    mon.fit(ref)
    assert len(mon.flags) == 250 and len(mon._tested()) == 125
    # right after the fit, the tested points are exactly the reference ones
    assert mon.outlier_rate == pytest.approx(mon.rule.baseline_rate)
    unseen = [f is not None for f in mon.flags]      # mask of the window at the fit
    for x in same[:300]:
        mon.update(x)
        unseen = (unseen + [True])[-250:]            # the loop's window slides by one
        assert len(mon.flags) == 250
        assert len(mon._tested()) == sum(unseen)
    assert len(mon._tested()) == 250                 # fitted points have left
    assert mon.outlier_rate == pytest.approx(mon.outlier.flags(same[50:300]).mean())


def test_unseen_removes_post_refit_bias():
    # high-dimensional stationary data: the in-sample flags of the fitted half
    # lower the current rate right after a refit, unless they are excluded
    rng = np.random.default_rng(1)
    gaps = {"window": [], "unseen": []}
    for rep in range(30):
        X = rng.normal(size=(250, 40))
        for current, values in gaps.items():
            mon = OutlierRateMonitor(ZTestDriftDetector(0.025, variance="pooled"),
                                     baseline="holdout", current=current, seed=rep)
            mon.fit(X)
            values.append(mon.outlier_rate - mon.rule.baseline_rate)
    assert np.mean(gaps["window"]) < -0.05
    assert np.mean(gaps["unseen"]) == pytest.approx(0.0, abs=1e-12)


def test_unseen_requires_holdout():
    with pytest.raises(ValueError):
        OutlierRateMonitor(ZTestDriftDetector(), baseline="in_sample", current="unseen")
    with pytest.raises(ValueError):
        OutlierRateMonitor(ZTestDriftDetector(), baseline="holdout", current="nope")
