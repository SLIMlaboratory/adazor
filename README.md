# ADAZOR

**A**utomatic **D**rift **A**lerting using **Z**-test on **O**utlier **R**ate:
label-free concept drift detection for classifiers deployed on data streams.

A one-class SVM fitted on the current window maps every input to a binary
outlier indicator; a two-proportion test compares the outlier rate in a
sliding window with the rate observed at the last retraining, and raises an
alert when the two differ significantly. Labels are used only to retrain the
predictor, never to decide when to retrain.

## Installation

Python 3.11 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments,dev]"     # package, plotting, tests
pytest                                  # run the test suite
```

`requirements.txt` lists the exact versions used for the paper.

## Usage

```python
from adazor import (ArrayStream, OCSVMOutlierDetector, OutlierRateMonitor,
                    ZTestDriftDetector, load_csv, make_model, run_stream)

X, y, _ = load_csv("weather.csv")            # rows in temporal order, 'target' column
monitor = OutlierRateMonitor(                # ADAZOR = outlier indicators + Z-test
    ZTestDriftDetector(alpha=0.025),         # conservative variance, as in the paper
    lambda: OCSVMOutlierDetector(nu=0.1, gamma="auto", standardize=True),
    baseline="in_sample",                    # or "holdout"
)
result = run_stream(
    ArrayStream(X, y),
    n=250,                                   # window size
    r=0.1,                                   # fraction of the window kept at retraining
    monitor=monitor,
    model_factory=lambda: make_model("pa", standardize=True),
)
print(result.n_alerts, result.alerts, result.prequential_accuracy)
```

`run_stream` implements Algorithm 1 of the paper: after an alert it keeps the
`floor(r*n)` most recent examples of the window, reads `s = n - floor(r*n)` new
ones, refits the predictor from scratch and refits the monitor on the new
window. See the docstring of `adazor.runner` for the exact definitions of time
and of the two accuracy measures that are recorded (prequential and window
accuracy).

A *monitor* is any label-free detector with `fit(X)` (called at every
(re)training with the inputs of the window) and `update(x) -> bool` (called for
each input entering the window). OCDD is `OutlierRateMonitor` with a
`ThresholdDriftDetector`; `KSWINMonitor` and `D3Monitor` are the label-free
baselines used in the paper.

The package also exposes the closed-form properties of the conservative test:
`effective_level`, `min_detectable_shift` and `window_size_for_shift`.

| module | content |
|---|---|
| `adazor.detectors` | decision rules on outlier indicators: `ZTestDriftDetector` (ADAZOR, `variance="conservative"` or `"pooled"`), `ThresholdDriftDetector` (OCDD), `NoDriftDetector`; closed-form properties |
| `adazor.outliers` | `OCSVMOutlierDetector`: standardization (fitted on the window) + one-class SVM |
| `adazor.monitors` | `Monitor` interface, `OutlierRateMonitor` (ADAZOR / OCDD, in-sample or hold-out baseline), `NullMonitor` |
| `adazor.baselines` | `KSWINMonitor` (per-feature KSWIN with Bonferroni correction), `D3Monitor` (discriminative detector) |
| `adazor.streams` | `GaussianMixture`, `DriftingMixtureStream` (abrupt/gradual drift), `ArrayStream`, `load_csv` |
| `adazor.models` | predictors refitted from scratch on each window (`make_model`: `ht`, `gnb`, `pa`, `arf`, `lr`, `lsv`, `gsv`; optional standardization) |
| `adazor.runner` | `run_stream`, the processing loop, and `RunResult` |

## Reproducing the experiments

The experiments of the paper are described by the TOML files in `configs/`
and run by the `experiments` package (from the repository root):

```bash
python -m experiments run configs/e2_simulations.toml --workers 4   # resumable
python -m experiments status configs/e2_simulations.toml
python -m experiments analyze configs/e2_simulations.toml --out-dir figures/
```

Each configuration expands into a grid of jobs (streams x monitors x
predictors x run parameters x seeds, possibly in several blocks); every job
writes one JSON record to `results/<experiment>/<job_id>.json` with its full
specification, the outcome, the drift events when known, and provenance (package
version, git commit, library versions). Jobs already done are skipped, so an
interrupted run can simply be restarted. `analyze` writes the tables (CSV and
LaTeX) and the PDF figures declared in the `[analysis]` section of the
configuration.

| config | content |
|---|---|
| `smoke.toml` | end-to-end check of the pipeline (under a minute) |
| `e1_properties.toml` | statistical claims: effective level of the test, in-sample vs. out-of-sample outlier rates, firing rates without drift |
| `e2_simulations.toml` | Gaussian-mixture streams with known drift: abrupt/gradual, inputs moving, labeling rule changing, stationary |
| `e3_injected.toml` | real datasets with an input shift injected at a known position |
| `e4_benchmarks.toml` | benchmark streams in their original order, and shuffled as stationary controls |
| `e5_sensitivity.toml` | one parameter at a time: window size, alpha, threshold f, retained fraction r |
| `e6_comparison.toml` | ADAZOR variants, OCDD with several thresholds, KSWIN and D3 |

The benchmark streams are the CSV files of the
[concept-drift-datasets-scikit-multiflow](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow)
collection (not redistributed here). Their location is given by the
environment variable `OCC_DATA` (default:
`~/repo/ricerca/occ-drift-detection/data`).

## Changes with respect to the `experiments_adazor` branch

This branch replaces the code of `experiments_adazor`, which was used for
the conference version of the paper. The following problems were fixed.

**Experiments (these invalidate the benchmark results obtained with
`notebooks/legacy/adazor_experiment.ipynb`):**

- The notebook **shuffled every dataset** before streaming it, which removes
  the temporal order and any drift. `ArrayStream` now preserves the order;
  shuffling is available only as an explicit option (`--shuffle`) for
  stationary control experiments.
- The OCSVM was used with `gamma=0.5` on **unscaled features**, whereas the
  paper states `gamma='auto'`. On most benchmark streams (Weather, Airlines,
  Spam, Phishing, SEA) the kernel becomes degenerate: almost every new point is
  flagged as an outlier, so both detectors fire at a fixed pace that does not
  depend on the data. Features are now standardized on each training window,
  and `gamma` defaults to `"auto"`.
- The Hoeffding tree of the second simulated setting was **shared across
  runs** (state learned in one repetition carried over to the next). Models
  are now created by a factory for every run and every retraining.

**Processing loop:**

- The loop (Algorithm 1) existed only in notebooks, in two slightly different
  copies; it is now `adazor.runner.run_stream`, covered by tests.
- The number of examples read after an alert is `n - floor(r*n)`; the time
  counter was advanced by `int(n*(1-r))` instead, which differs when `r*n` is
  not an integer. Time is now the number of examples actually read.
- `DriftingMixtureStream` lost one time step every time a new generator was
  created on it (i.e. at every alert) and could yield the same time index
  twice; it is now a stateful iterator.
- The window accuracy recorded at an alert referred to the old model in one
  notebook and to the new one in the other; it now consistently refers to the
  model in force before retraining. A prequential (test-then-train) accuracy is
  also recorded.

**Package:**

- The OCSVM, previously created inside the notebooks, is part of the package
  (`OCSVMOutlierDetector`).
- `ZTestDriftDetector` gained the pooled-variance variant described in the
  paper; the conservative rule remains the default.
- `GaussianMixture`: `sigma` holds variances (the docstring said standard
  deviations); the stream now draws from independent normals instead of
  `multivariate_normal`, so random sequences differ from the old ones.
- A single `pyproject.toml` with a `src/` layout (the root one pointed to a
  directory that no longer existed); consistent authors and license.
- `__all__` exported a non-existent name (`sklearnBatchAdapter`), which broke
  `from adazor import *`.
- The unit tests, deleted from the previous branch, were restored and extended.
- The plots generated by the legacy notebook were removed, since they were
  obtained from shuffled data.

## License

Apache License 2.0, see `LICENSE`.
