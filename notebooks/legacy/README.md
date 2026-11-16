# Legacy notebook — do not use for results

`adazor_experiment.ipynb` is the notebook used for the benchmark experiments
of the conference version of the paper. It is kept only for reference. Its
results are not valid:

- cell 7 shuffles every dataset before streaming it, which removes any drift;
- the OCSVM uses `gamma=0.5` on unscaled features, which makes it flag almost
  every new point as an outlier on most datasets, so that alerts fire at a
  fixed pace regardless of the data.

Use `experiments/benchmarks.py` instead (see the main README).
