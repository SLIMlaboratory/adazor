"""Command-line interface: ``python -m experiments {run,status,analyze} CONFIG``."""

from __future__ import annotations

import os

# one BLAS/OpenMP thread per process: parallelism comes from running several jobs
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m experiments", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for cmd in ("run", "status", "analyze"):
        p = sub.add_parser(cmd)
        p.add_argument("config")
        p.add_argument("--results", default="results", help="root folder of the result files")
        if cmd == "run":
            p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        if cmd == "analyze":
            p.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.results)

    if args.command == "run":
        from .run import run_config
        sys.exit(1 if run_config(args.config, args.workers, root) else 0)
    if args.command == "status":
        from .run import pending
        jobs, todo = pending(args.config, root)
        print(f"{args.config}: {len(jobs) - len(todo)}/{len(jobs)} jobs done")
        return
    from .analysis import analyze
    for path in analyze(args.config, args.out_dir, root):
        print(path)


if __name__ == "__main__":
    main()
