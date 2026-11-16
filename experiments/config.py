"""Experiment configurations: TOML files expanded into grids of jobs.

A configuration contains::

    name = "e2_simulations"      # results go to results/<name>/
    kind = "stream"              # or "properties" (see experiments.properties)
    seeds = [42, 52]

    [run_defaults]               # n, r, eval_every, max_examples
    n = 250

    [outlier]                    # defaults for the OCSVM of ADAZOR/OCDD monitors
    nu = 0.1

    [[streams]]                  # one table per stream
    kind = "mixture"
    scenario = "setting1"

    [[monitors]]                 # one table per monitor
    kind = "adazor"

    [[models]]                   # one table per predictor
    name = "ht"

    [[runs]]                     # optional: one table per set of run parameters
    n = 500

The jobs are the Cartesian product streams x monitors x models x runs x seeds.
``[[blocks]]`` tables can override any of these lists; the jobs are then the
union of the products of the blocks (useful for one-at-a-time sweeps).
Any table may have a ``label`` used in tables and figures.
"""

from __future__ import annotations

import hashlib
import itertools as it
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUN = {"n": 250, "r": 0.1, "eval_every": 0, "max_examples": None}
OUTLIER_MONITORS = ("adazor", "ocdd")
AXES = ("streams", "monitors", "models", "runs", "seeds")


@dataclass(frozen=True)
class Job:
    experiment: str
    kind: str
    spec: dict

    @property
    def job_id(self) -> str:
        blob = json.dumps({"kind": self.kind, "spec": self.spec}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def load(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    if "name" not in cfg:
        raise ValueError(f"{path}: missing 'name'")
    return cfg


def _with_outlier_defaults(monitor: dict, defaults: dict) -> dict:
    if monitor.get("kind") not in OUTLIER_MONITORS:
        return monitor
    return {**monitor, "outlier": {**defaults, **monitor.get("outlier", {})}}


def expand(cfg: dict) -> list[Job]:
    """All the jobs of a configuration, without duplicates, in a stable order."""
    name, kind = cfg["name"], cfg.get("kind", "stream")
    if kind == "properties":
        seeds = cfg.get("seeds", [0])
        jobs = [Job(name, kind, {**task, "seed": s}) for task in cfg["tasks"] for s in seeds]
    elif kind == "stream":
        jobs = []
        for block in cfg.get("blocks", [{}]):
            axes = {a: block.get(a, cfg.get(a)) for a in AXES}
            axes["models"] = axes["models"] or [{"name": "ht"}]
            axes["runs"] = axes["runs"] or [{}]
            axes["seeds"] = axes["seeds"] or [42]
            for a in ("streams", "monitors"):
                if not axes[a]:
                    raise ValueError(f"{name}: no {a} defined")
            outlier = cfg.get("outlier", {})
            run_defaults = {**DEFAULT_RUN, **cfg.get("run_defaults", {})}
            for stream, monitor, model, run, seed in it.product(*(axes[a] for a in AXES)):
                spec = {
                    "stream": stream,
                    "monitor": _with_outlier_defaults(monitor, outlier),
                    "model": model,
                    "run": {**run_defaults, **run},
                    "seed": seed,
                }
                jobs.append(Job(name, kind, spec))
    else:
        raise ValueError(f"{name}: unknown kind '{kind}'")
    unique = {}
    for job in jobs:
        unique.setdefault(job.job_id, job)
    return list(unique.values())
