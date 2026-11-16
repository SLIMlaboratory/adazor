import numpy as np
import pytest

import adazor
from adazor.models import MODEL_NAMES, make_model


@pytest.mark.parametrize("name", sorted(MODEL_NAMES))
def test_models_fit_and_predict(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = make_model(name, seed=0).fit(X[:150], y[:150])
    pred = m.predict(X[150:])
    assert len(pred) == 50
    assert np.mean([p == t for p, t in zip(pred, y[150:])]) > 0.7
    assert m.predict_one(X[150]) in (0, 1)


def test_single_class_window():
    X = np.random.default_rng(0).normal(size=(20, 2))
    m = make_model("lsv").fit(X, np.ones(20, dtype=int))
    assert all(p == 1 for p in m.predict(X))


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("nope")


def test_public_api_star_import():
    ns = {}
    exec("from adazor import *", ns)  # noqa: S102
    for name in adazor.__all__:
        assert name in ns
