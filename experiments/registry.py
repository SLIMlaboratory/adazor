"""Build streams, monitors and predictors from the specifications of a job."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from adazor import (
    ArrayStream,
    D3Monitor,
    KSWINMonitor,
    NullMonitor,
    OCSVMOutlierDetector,
    OutlierRateMonitor,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    load_csv,
    make_model,
)

from .streams import InjectedShiftStream, mixture_scenario

DEFAULT_DATA_ROOT = "~/repo/ricerca/occ-drift-detection/data"


def data_root() -> Path:
    return Path(os.path.expanduser(os.environ.get("OCC_DATA", DEFAULT_DATA_ROOT)))


@lru_cache(maxsize=4)
def _csv(path: str):
    return load_csv(path)


def build_stream(spec: dict, seed: int):
    """Return ``(stream, events)``; ``events`` is ``None`` when unknown."""
    kind = spec["kind"]
    if kind == "mixture":
        return mixture_scenario(spec["scenario"], seed, spec.get("length", 8000))
    if kind in ("csv", "injected"):
        X, y, _ = _csv(str(data_root() / spec["file"]))
        if kind == "csv":
            shuffle = spec.get("shuffle", False)
            # a shuffled stream is a stationary control: no drift by construction
            return ArrayStream(X, y, shuffle=shuffle, seed=seed), ([] if shuffle else None)
        feature = spec.get("feature", "auto")
        if feature == "auto":   # the feature with the most distinct values
            feature = int(np.argmax([len(np.unique(X[:, j])) for j in range(X.shape[1])]))
        stream = InjectedShiftStream(X, y, feature=feature, position=spec["position"],
                                     width=spec.get("width", 0), strength=spec.get("strength", 0.9),
                                     seed=seed, length=spec.get("length"))
        return stream, stream.events
    raise ValueError(f"unknown stream kind '{kind}'")


def _gamma(value):
    return value if isinstance(value, str) else float(value)


def build_monitor(spec: dict, seed: int):
    kind = spec["kind"]
    if kind == "none":
        return NullMonitor()
    if kind in ("adazor", "ocdd"):
        o = spec.get("outlier", {})

        def outlier_factory():
            return OCSVMOutlierDetector(nu=o.get("nu", 0.1), gamma=_gamma(o.get("gamma", "auto")),
                                        standardize=o.get("standardize", True))

        if kind == "ocdd":
            rule = ThresholdDriftDetector(spec.get("f", 0.1))
        else:
            rule = ZTestDriftDetector(spec.get("alpha", 0.025), variance=spec.get("variance", "conservative"))
        return OutlierRateMonitor(rule, outlier_factory, baseline=spec.get("baseline", "in_sample"),
                                  holdout_fraction=spec.get("holdout_fraction", 0.5), seed=seed)
    if kind == "kswin":
        return KSWINMonitor(alpha=spec.get("alpha", 0.005), stat_size=spec.get("stat_size", 30), seed=seed)
    if kind == "d3":
        return D3Monitor(rho=spec.get("rho", 0.2), tau=spec.get("tau", 0.7),
                         classifier=spec.get("classifier", "logreg"), folds=spec.get("folds", 5), seed=seed)
    raise ValueError(f"unknown monitor kind '{kind}'")


def build_model_factory(spec: dict, seed: int):
    name, standardize = spec["name"], spec.get("standardize", False)
    return lambda: make_model(name, seed=seed, standardize=standardize)


def label(spec: dict, what: str) -> str:
    """Short human-readable label of a stream, monitor or model specification."""
    if "label" in spec:
        return spec["label"]
    if what == "stream":
        if spec["kind"] == "mixture":
            return spec["scenario"]
        stem = Path(spec["file"]).stem
        if spec["kind"] == "injected":
            return f"{stem} (shift, w={spec.get('width', 0)})"
        return f"{stem} (shuffled)" if spec.get("shuffle") else stem
    if what == "monitor":
        k = spec["kind"]
        if k == "adazor":
            extra = [spec.get("variance", "conservative"), spec.get("baseline", "in_sample")]
            return f"ADAZOR a={spec.get('alpha', 0.025)} " + "/".join(extra)
        if k == "ocdd":
            return f"OCDD f={spec.get('f', 0.1)}"
        return {"none": "no detection", "kswin": "KSWIN", "d3": "D3"}.get(k, k)
    if what == "model":
        return spec["name"] + (" (std)" if spec.get("standardize") else "")
    raise ValueError(what)
