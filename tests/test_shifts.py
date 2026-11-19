import numpy as np
import pytest

from adazor import MMDMonitor
from experiments.shifts import FAMILIES, CorrelationFlipStream, DatasetShiftStream


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    X = np.column_stack([a, a + 0.3 * rng.normal(size=4000), rng.normal(size=4000)])
    return X, (X[:, 0] > 0).astype(int)


def take(stream):
    xs, ys = zip(*stream)
    return np.array(xs), np.array(ys)


@pytest.mark.parametrize("family,severity", [(f, 0.5) for f in FAMILIES])
def test_every_family_runs_and_reports_its_event(dataset, family, severity):
    X, y = dataset
    s = DatasetShiftStream(X, y, family, severity, position=1000, width=200, length=3000, seed=1)
    xs, ys = take(s)
    assert xs.shape == (3000, 3) and set(ys) <= {0, 1}
    assert s.events == [(1001, 1201)]


def test_before_the_shift_rows_come_from_the_data(dataset):
    X, y = dataset
    xs, _ = take(DatasetShiftStream(X, y, "noise", 1.0, position=1000, length=1000, seed=2))
    rows = {tuple(r) for r in X}
    assert all(tuple(r) in rows for r in xs)


def test_decorrelate_keeps_marginals_and_breaks_dependence(dataset):
    X, y = dataset
    xs, _ = take(DatasetShiftStream(X, y, "decorrelate", 0, position=0, length=4000, seed=3))
    assert np.allclose(xs.mean(0), X.mean(0), atol=0.06)
    assert np.allclose(xs.std(0), X.std(0), atol=0.06)
    assert np.corrcoef(X[:, 0], X[:, 1])[0, 1] > 0.9
    assert abs(np.corrcoef(xs[:, 0], xs[:, 1])[0, 1]) < 0.06


def test_translate_moves_by_the_requested_norm(dataset):
    X, y = dataset
    s = DatasetShiftStream(X, y, "translate", 2.0, position=0, length=10, seed=4)
    assert np.linalg.norm(s.offset / s.sd) == pytest.approx(2.0)


def test_subpopulation_is_absent_before_the_shift(dataset):
    X, y = dataset
    s = DatasetShiftStream(X, y, "subpopulation", 0.3, position=2000, length=4000, seed=5)
    xs, _ = take(s)
    cluster = {tuple(r) for r in X[s.cluster]}
    before = np.mean([tuple(r) in cluster for r in xs[:2000]])
    after = np.mean([tuple(r) in cluster for r in xs[2000:]])
    assert before == 0.0 and after == pytest.approx(0.3, abs=0.05)


def test_correlation_flip_keeps_marginals():
    s = CorrelationFlipStream(4, rho=0.9, position=3000, length=6000, seed=6)
    xs, _ = take(s)
    pre, post = xs[:3000], xs[3000:]
    for part in (pre, post):
        assert np.allclose(part.mean(0), 0, atol=0.08) and np.allclose(part.std(0), 1, atol=0.08)
    assert np.corrcoef(pre[:, 0], pre[:, 1])[0, 1] > 0.85
    assert np.corrcoef(post[:, 0], post[:, 1])[0, 1] < -0.85
    assert s.events == [(3001, 3001)]
    assert CorrelationFlipStream(2, flip=False).events == []
    with pytest.raises(ValueError):
        CorrelationFlipStream(3)


def test_mmd_detects_a_shift_and_not_stationarity():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(250, 3))
    mon = MMDMonitor(alpha=0.01, seed=0)
    mon.fit(ref)
    assert not any(mon.update(x) for x in rng.normal(size=(500, 3)))
    mon.fit(ref)
    assert any(mon.update(x) for x in rng.normal(size=(100, 3)) + 1.5)
    assert 0 < mon.last_p <= 1
