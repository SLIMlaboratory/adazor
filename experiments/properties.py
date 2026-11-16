"""Checks of the statistical claims of Sect. 4.3 (configurations with ``kind = "properties"``).

Tasks:

* ``level``: empirical firing rate of the two-proportion rules under the null
  hypothesis (two independent Bernoulli samples of size ``n`` with parameter
  ``p``), compared with :func:`adazor.effective_level` for the conservative
  rule and with ``alpha`` for the pooled one.
* ``ocsvm_rates``: outlier rate of an OCSVM on the points it was fitted on vs.
  on fresh points from the same distribution (a mixture concept or random
  rows of a CSV dataset); firing rate of the fixed threshold ``f`` and of the
  conservative rule on such stationary windows.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from adazor import OCSVMOutlierDetector, effective_level, min_detectable_shift

from .registry import _csv, _gamma, data_root
from .streams import SCENARIOS


def _level(task: dict, rng: np.random.Generator) -> dict:
    n, p, alpha, reps = task["n"], task["p"], task["alpha"], task.get("reps", 200_000)
    a = rng.binomial(n, p, size=reps) / n
    b = rng.binomial(n, p, size=reps) / n
    diff = np.abs(a - b)
    conservative = float(np.mean(diff > min_detectable_shift(n, alpha)))
    pbar = (a + b) / 2
    se = np.sqrt(2 * pbar * (1 - pbar) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = float(np.mean(np.where(se > 0, diff / se > norm.ppf(1 - alpha / 2), diff > 0)))
    return {"empirical_conservative": conservative, "effective_level": effective_level(alpha, p),
            "empirical_pooled": pooled, "nominal": alpha}


def _sampler(source: dict, rng: np.random.Generator):
    if source["kind"] == "mixture":
        p0, p1, _, _ = SCENARIOS[source["scenario"]]
        concept = p0 if source.get("concept", 0) == 0 else p1
        return lambda k: np.array([concept.draw(rng)[0] for _ in range(k)])
    X, _, _ = _csv(str(data_root() / source["file"]))
    return lambda k: X[rng.choice(len(X), size=k, replace=False)]


def _ocsvm_rates(task: dict, rng: np.random.Generator) -> dict:
    n, reps, f, alpha = task["n"], task.get("reps", 200), task.get("f", 0.1), task.get("alpha", 0.025)
    draw = _sampler(task["source"], rng)
    ins, outs = [], []
    for _ in range(reps):
        X = draw(2 * n)
        det = OCSVMOutlierDetector(nu=task.get("nu", 0.1), gamma=_gamma(task.get("gamma", "auto")),
                                   standardize=task.get("standardize", True)).fit(X[:n])
        ins.append(det.flags(X[:n]).mean())
        outs.append(det.flags(X[n:]).mean())
    ins, outs = np.array(ins), np.array(outs)
    delta = min_detectable_shift(n, alpha)
    return {
        "in_sample_mean": float(ins.mean()), "in_sample_std": float(ins.std(ddof=1)),
        "out_of_sample_mean": float(outs.mean()), "out_of_sample_std": float(outs.std(ddof=1)),
        "threshold_firing_rate": float(np.mean(outs > f)),
        "adazor_firing_rate": float(np.mean(np.abs(outs - ins) > delta)),
    }


def run_properties(spec: dict) -> dict:
    rng = np.random.default_rng(spec["seed"])
    task = spec["task"]
    if task == "level":
        return _level(spec, rng)
    if task == "ocsvm_rates":
        return _ocsvm_rates(spec, rng)
    raise ValueError(f"unknown properties task '{task}'")
