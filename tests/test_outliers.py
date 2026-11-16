import numpy as np

from adazor.outliers import OCSVMOutlierDetector


def test_standardization_makes_detector_scale_invariant():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(750, 4)) * np.array([1000.0, 1.0, 50.0, 0.01])
    det = OCSVMOutlierDetector(nu=0.1, gamma="auto", standardize=True).fit(X[:250])
    assert det.flags(X[250:]).mean() < 0.25
    raw = OCSVMOutlierDetector(nu=0.1, gamma=0.5, standardize=False).fit(X[:250])
    assert raw.flags(X[250:]).mean() > 0.9   # the failure mode of the old experiments


def test_flags_accepts_single_point():
    X = np.random.default_rng(1).normal(size=(100, 3))
    det = OCSVMOutlierDetector().fit(X)
    assert det.flags(X[0]).shape == (1,)
