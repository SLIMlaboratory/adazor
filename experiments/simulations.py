"""Controlled simulations of gradual drift (Sect. 5.2 of the paper).

Two settings: an initial concept P0 turns gradually into P1 between examples
2000 and 3000, stays there, and returns to P0 between 5000 and 6000; streams
contain 8000 examples. Every run uses a fresh predictor and a fresh stream.

Example:
    python experiments/simulations.py --setting 1 --out results/sim1.json --plot results/sim1.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adazor import (
    DriftingMixtureStream,
    GaussianMixture,
    NoDriftDetector,
    OCSVMOutlierDetector,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    make_model,
    run_stream,
)

DRIFT_POINTS = [(2000, 3000), (5000, 6000)]
LENGTH = 8000

# sigma entries are variances (diagonals of the covariance matrices)
SETTINGS = {
    1: (GaussianMixture(mu=[[0, -1], [1, 1]], sigma=[[0.01, 0.2], [0.2, 0.01]],
                        class_=[0, 1], p=[0.5, 0.5]),
        GaussianMixture(mu=[[0, -1], [-1, -2]], sigma=[[0.01, 0.2], [0.2, 0.01]],
                        class_=[0, 1], p=[0.5, 0.5])),
    2: (GaussianMixture(mu=[[0, 1], [0, 0]], sigma=[[0.4, 0.4], [0.4, 0.4]],
                        class_=[0, 1], p=[0.5, 0.5]),
        GaussianMixture(mu=[[0, 0], [1, 1]], sigma=[[0.2, 0.2], [0.2, 0.2]],
                        class_=[0, 1], p=[0.5, 0.5])),
}


def detectors(f: float, alpha: float, variance: str):
    return {
        "none": NoDriftDetector,
        "threshold": lambda: ThresholdDriftDetector(f),
        "adazor": lambda: ZTestDriftDetector(alpha, variance=variance),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", type=int, choices=sorted(SETTINGS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 142, 10)))
    ap.add_argument("--model", default="ht")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--gamma", default="0.5", help="OCSVM gamma: a number, 'auto' or 'scale'")
    ap.add_argument("--standardize", action="store_true",
                    help="standardize features before the OCSVM (the paper's simulations do not)")
    ap.add_argument("--f", type=float, default=0.1, help="threshold of the fixed-threshold detector")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--variance", choices=ZTestDriftDetector.VARIANCES, default="conservative")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", help="optional path of a figure with the accuracy trajectories")
    args = ap.parse_args()

    gamma = args.gamma if args.gamma in ("auto", "scale") else float(args.gamma)
    p0, p1 = SETTINGS[args.setting]
    results = {"args": vars(args), "drift_points": DRIFT_POINTS, "runs": {}}
    for name, make_detector in detectors(args.f, args.alpha, args.variance).items():
        runs = []
        for seed in args.seeds:
            stream = DriftingMixtureStream([p0, p1, p0], DRIFT_POINTS, seed=seed, length=LENGTH)
            res = run_stream(
                stream, n=args.n, r=args.r, detector=make_detector(),
                model_factory=lambda seed=seed: make_model(args.model, seed=seed),
                outlier_factory=lambda: OCSVMOutlierDetector(args.nu, gamma, args.standardize),
                eval_every=args.eval_every)
            runs.append({"seed": seed, **res.to_dict()})
            print(f"setting {args.setting} {name:9s} seed {seed}: {res.n_alerts:3d} alerts, "
                  f"prequential accuracy {res.prequential_accuracy:.3f}", flush=True)
        counts = [r["n_alerts"] for r in runs]
        print(f"setting {args.setting} {name:9s} alerts: {np.mean(counts):.2f} ± {np.std(counts, ddof=1):.2f} "
              "(mean ± sample std)")
        results["runs"][name] = runs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results))
    if args.plot:
        from plots import trajectories
        trajectories(results, args.plot)


if __name__ == "__main__":
    main()
