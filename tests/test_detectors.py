import math

import numpy as np
import pytest

from adazor.detectors import (
    NoDriftDetector,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    effective_level,
    min_detectable_shift,
    window_size_for_shift,
)


def test_no_drift_detector_never_signals():
    det = NoDriftDetector()
    det.reset([True] * 10)
    assert det.detect([True] * 100) is False


def test_threshold_detector_is_strict():
    det = ThresholdDriftDetector(f=0.1)
    assert det.detect([True] * 10 + [False] * 90) is False   # exactly 10%
    assert det.detect([True] * 11 + [False] * 89) is True


def test_ztest_requires_reset():
    with pytest.raises(RuntimeError):
        ZTestDriftDetector().detect([False] * 10)


def test_conservative_threshold_matches_paper():
    det = ZTestDriftDetector(alpha=0.025)
    det.reset([True] * 25 + [False] * 225)              # p0 = 0.1, n = 250
    assert math.isclose(det.threshold(250), 0.10024, abs_tol=1e-4)
    assert det.detect([True] * 50 + [False] * 200) is False   # shift 0.100
    assert det.detect([True] * 51 + [False] * 199) is True    # shift 0.104


def test_ztest_is_two_sided():
    det = ZTestDriftDetector(alpha=0.025)
    det.reset([True] * 125 + [False] * 125)             # p0 = 0.5
    assert det.detect([True] * 90 + [False] * 160) is True   # p = 0.36


def test_pooled_variant_is_less_conservative():
    cons = ZTestDriftDetector(alpha=0.025)
    pooled = ZTestDriftDetector(alpha=0.025, variance="pooled")
    ref = [True] * 25 + [False] * 225
    cur = [True] * 45 + [False] * 205                   # shift 0.08
    cons.reset(ref)
    pooled.reset(ref)
    assert cons.detect(cur) is False
    assert pooled.detect(cur) is True


def test_pooled_variant_with_constant_samples():
    det = ZTestDriftDetector(alpha=0.05, variance="pooled")
    det.reset([False] * 50)
    assert det.detect([False] * 50) is False


def test_empirical_level_close_to_effective_level():
    rng = np.random.default_rng(0)
    det = ZTestDriftDetector(alpha=0.5)                 # large alpha: measurable rate
    p, n, reps, fired = 0.3, 250, 20000, 0
    for _ in range(reps):
        det.reset(rng.random(n) < p)
        fired += det.detect(rng.random(n) < p)
    assert abs(fired / reps - effective_level(0.5, p)) < 0.01


def test_closed_form_values_quoted_in_paper():
    assert math.isclose(effective_level(0.025, 0.1), 1.87e-4, rel_tol=0.01)
    assert math.isclose(effective_level(0.025, 0.5), 0.025, rel_tol=1e-9)
    assert math.isclose(min_detectable_shift(250, 0.025), 0.1002, abs_tol=1e-4)
    assert math.isclose(min_detectable_shift(250, 0.025, p=0.1), 0.0601, abs_tol=1e-4)
    assert window_size_for_shift(0.05, 0.025) == 1005
    assert window_size_for_shift(0.05, 0.025, p=0.1) == 362
