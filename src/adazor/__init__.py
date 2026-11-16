"""ADAZOR: Automatic Drift Alerting using Z-test on Outlier Rate.

Label-free concept drift detection: a one-class SVM maps each input to an
outlier indicator, and a two-proportion test compares the outlier rate in a
sliding window with the rate observed at the last retraining.
"""

from .detectors import (
    DriftDetector,
    NoDriftDetector,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    effective_level,
    min_detectable_shift,
    window_size_for_shift,
)
from .models import MODEL_NAMES, Model, RiverModel, SklearnModel, make_model
from .outliers import OCSVMOutlierDetector
from .runner import RunResult, run_stream
from .streams import ArrayStream, DriftingMixtureStream, GaussianMixture, load_csv

__version__ = "0.2.0"

__all__ = [
    "MODEL_NAMES",
    "ArrayStream",
    "DriftDetector",
    "DriftingMixtureStream",
    "GaussianMixture",
    "Model",
    "NoDriftDetector",
    "OCSVMOutlierDetector",
    "RiverModel",
    "RunResult",
    "SklearnModel",
    "ThresholdDriftDetector",
    "ZTestDriftDetector",
    "effective_level",
    "load_csv",
    "make_model",
    "min_detectable_shift",
    "run_stream",
    "window_size_for_shift",
]
