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
from adazor import (ArrayStream, OCSVMOutlierDetector, ZTestDriftDetector,
                    load_csv, make_model, run_stream)

X, y, _ = load_csv("weather.csv")            # rows in temporal order, 'target' column
result = run_stream(
    ArrayStream(X, y),
    n=250,                                    # window size
    r=0.1,                                    # fraction of the window kept at retraining
    detector=ZTestDriftDetector(alpha=0.025), # ADAZOR (conservative variance, as in the paper)
    model_factory=lambda: make_model("pa"),
    outlier_factory=lambda: OCSVMOutlierDetector(nu=0.1, gamma="auto", standardize=True),
)
print(result.n_alerts, result.alerts, result.prequential_accuracy)
```

`run_stream` implements Algorithm 1 of the paper: after an alert it keeps the
`floor(r*n)` most recent examples of the window, reads `s = n - floor(r*n)` new
ones, and refits the predictor and the outlier detector from scratch.
See the docstring of `adazor.runner` for the exact definitions of time and of
the two accuracy measures that are recorded (prequential and window accuracy).

The package also exposes the closed-form properties of the conservative test
used in the paper: `effective_level`, `min_detectable_shift` and
`window_size_for_shift`.

| module | content |
|---|---|
| `adazor.detectors` | `ZTestDriftDetector` (ADAZOR, `variance="conservative"` or `"pooled"`), `ThresholdDriftDetector` (OCDD rule), `NoDriftDetector`, closed-form properties |
| `adazor.outliers` | `OCSVMOutlierDetector`: standardization (fitted on the window) + one-class SVM |
| `adazor.streams` | `GaussianMixture`, `DriftingMixtureStream` (abrupt/gradual drift), `ArrayStream`, `load_csv` |
| `adazor.models` | predictors refitted from scratch on each window (`make_model`: `ht`, `gnb`, `pa`, `arf`, `lr`, `lsv`, `gsv`) |
| `adazor.runner` | `run_stream`, the processing loop, and `RunResult` |

## Reproducing the experiments

Controlled simulations of gradual drift (two settings, ten seeds each):

```bash
python experiments/simulations.py --setting 1 --out results/sim1.json --plot results/sim1.pdf
python experiments/simulations.py --setting 2 --out results/sim2.json --plot results/sim2.pdf
```

Benchmark streams: the CSV files of the
[concept-drift-datasets-scikit-multiflow](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow)
collection (not redistributed here), e.g.

```bash
python experiments/benchmarks.py data/real-world/*.csv data/artificial/*.csv \
    --models ht gnb pa arf lr lsv gsv --f 0.1 0.15 0.2 0.25 --out results/benchmarks.jsonl
```

Results are appended to a JSON-lines file, one record per
(dataset, predictor, detector, seed).

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
