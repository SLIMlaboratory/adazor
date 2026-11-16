import itertools as it

import numpy as np
import pytest

from adazor.streams import ArrayStream, DriftingMixtureStream, GaussianMixture, load_csv


def mixtures():
    a = GaussianMixture(mu=[[0, 0], [1, 1]], sigma=[[0.2, 0.2], [0.2, 0.2]], class_=[0, 1], p=[0.5, 0.5])
    b = GaussianMixture(mu=[[0, 1], [0, 0]], sigma=[[0.4, 0.4], [0.4, 0.4]], class_=[0, 1], p=[0.5, 0.5])
    return a, b


def test_sigma_are_variances():
    g = GaussianMixture(mu=[[0.0]], sigma=[[4.0]], class_=[0], p=[1.0])
    rng = np.random.default_rng(0)
    x = np.array([g.draw(rng)[0][0] for _ in range(20000)])
    assert abs(x.var() - 4.0) < 0.15


def test_stream_keeps_position_across_chunks():
    a, b = mixtures()
    s = DriftingMixtureStream([a, b], [100], seed=0)
    list(it.islice(s, 10))
    list(it.islice(s, 10))
    assert s.t == 20                     # no example lost between chunks


def test_same_seed_same_stream():
    a, b = mixtures()
    x1 = [x for x, _ in it.islice(DriftingMixtureStream([a, b], [5], seed=3), 10)]
    x2 = [x for x, _ in it.islice(DriftingMixtureStream([a, b], [5], seed=3), 10)]
    assert all(np.array_equal(u, v) for u, v in zip(x1, x2))


def test_gradual_interpolation_endpoints():
    a, b = mixtures()
    s = DriftingMixtureStream([a, b, a], [(10, 20), (30, 40)], seed=0)
    assert np.allclose(s.mixture_at(9).mu, a.mu)
    assert np.allclose(s.mixture_at(15).mu, (a.mu + b.mu) / 2)
    assert np.allclose(s.mixture_at(20).mu, b.mu)
    assert np.allclose(s.mixture_at(25).mu, b.mu)
    assert np.allclose(s.mixture_at(35).sigma, (a.sigma + b.sigma) / 2)
    assert np.allclose(s.mixture_at(41).mu, a.mu)


def test_abrupt_drift():
    a, b = mixtures()
    s = DriftingMixtureStream([a, b], [50], seed=0)
    assert s.mixture_at(49) is a and s.mixture_at(50) is b


def test_invalid_drift_points():
    a, b = mixtures()
    with pytest.raises(ValueError):
        DriftingMixtureStream([a, b, a], [(10, 20), (15, 30)])


def test_length_limits_stream():
    a, b = mixtures()
    assert len(list(DriftingMixtureStream([a, b], [5], seed=0, length=12))) == 12


def test_array_stream_preserves_order_unless_shuffled():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    assert [lab for _, lab in ArrayStream(X, y)] == list(range(10))
    shuffled = [lab for _, lab in ArrayStream(X, y, shuffle=True, seed=0)]
    assert sorted(shuffled) == list(range(10)) and shuffled != list(range(10))


def test_load_csv_encodes_text_columns(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b,target\n1.5,x,0\n2.5,y,1\n3.5,x,1\n")
    X, y, names = load_csv(str(path))
    assert names == ["a", "b"]
    assert X.tolist() == [[1.5, 0.0], [2.5, 1.0], [3.5, 0.0]]
    assert y.tolist() == [0, 1, 1]
