"""Benchmark streams read from CSV files (Sect. 5.3 of the paper).

Each CSV file must contain one example per row, in temporal order, and a
``target`` column. The order of the rows is preserved: ``--shuffle`` permutes
the examples and therefore removes any drift; use it only to obtain a
stationary control stream.

Features are standardized on each training window before fitting the OCSVM
(disable with ``--no-standardize``): without standardization, features with
large ranges make the Gaussian kernel degenerate, almost every new point is
flagged as an outlier, and both detectors fire at a fixed pace regardless of
the data.

Example:
    python experiments/benchmarks.py data/real-world/weather.csv \
        --models pa ht --out results/weather.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adazor import (
    MODEL_NAMES,
    ArrayStream,
    OCSVMOutlierDetector,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    load_csv,
    make_model,
    run_stream,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--models", nargs="+", default=sorted(MODEL_NAMES), choices=sorted(MODEL_NAMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="seeds of the randomized predictors (the streams are deterministic)")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--gamma", default="auto", help="OCSVM gamma: a number, 'auto' or 'scale'")
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--f", type=float, nargs="+", default=[0.1],
                    help="threshold(s) of the fixed-threshold detector")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--variance", choices=ZTestDriftDetector.VARIANCES, default="conservative")
    ap.add_argument("--shuffle", action="store_true",
                    help="permute the examples (stationary control: removes any drift)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="record the window accuracy every k steps (0: never, it is costly)")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--out", required=True, help="JSON-lines file; results are appended")
    args = ap.parse_args()

    gamma = args.gamma if args.gamma in ("auto", "scale") else float(args.gamma)
    detectors = {f"threshold_f{f:g}": (lambda f=f: ThresholdDriftDetector(f)) for f in args.f}
    detectors["adazor"] = lambda: ZTestDriftDetector(args.alpha, variance=args.variance)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in args.csv:
        X, y, _ = load_csv(path)
        dataset = Path(path).stem
        for model in args.models:
            for det_name, make_detector in detectors.items():
                for seed in args.seeds:
                    start = time.time()
                    res = run_stream(
                        ArrayStream(X, y, shuffle=args.shuffle, seed=seed),
                        n=args.n, r=args.r, detector=make_detector(),
                        model_factory=lambda model=model, seed=seed: make_model(model, seed=seed),
                        outlier_factory=lambda: OCSVMOutlierDetector(args.nu, gamma, args.standardize),
                        max_examples=args.max_examples, eval_every=args.eval_every)
                    record = {"dataset": dataset, "model": model, "detector": det_name, "seed": seed,
                              "params": {k: v for k, v in vars(args).items() if k not in ("csv", "out")},
                              "seconds": time.time() - start, **res.to_dict()}
                    with out.open("a") as fh:
                        fh.write(json.dumps(record) + "\n")
                    print(f"{dataset} {model} {det_name} seed {seed}: {res.n_alerts} alerts, "
                          f"prequential accuracy {res.prequential_accuracy:.4f} "
                          f"({record['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
