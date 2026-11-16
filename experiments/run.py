"""Execute the jobs of a configuration in parallel, one result file per job."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import adazor

from .config import Job, expand, load
from .properties import run_properties
from .registry import build_model_factory, build_monitor, build_stream

RESULTS_ROOT = Path("results")


def provenance() -> dict:
    root = Path(adazor.__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import river
    import sklearn
    return {"adazor": adazor.__version__, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "river": river.__version__,
            "sklearn": sklearn.__version__, "python": platform.python_version(),
            "host": platform.node()}


def run_job(job: Job) -> dict:
    start = time.time()
    if job.kind == "properties":
        outcome = run_properties(job.spec)
    else:
        spec = job.spec
        stream, events = build_stream(spec["stream"], spec["seed"])
        run = spec["run"]
        res = adazor.run_stream(
            stream, n=run["n"], r=run["r"],
            monitor=build_monitor(spec["monitor"], spec["seed"]),
            model_factory=build_model_factory(spec["model"], spec["seed"]),
            max_examples=run.get("max_examples"), eval_every=run.get("eval_every", 0))
        outcome = {**res.to_dict(), "events": events}
    return {"experiment": job.experiment, "job_id": job.job_id, "kind": job.kind, "spec": job.spec,
            "result": outcome, "seconds": time.time() - start}


def result_path(job: Job, root: Path = RESULTS_ROOT) -> Path:
    return root / job.experiment / f"{job.job_id}.json"


def _write(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)


def _worker(job: Job, root: Path, prov: dict) -> tuple[str, float, str | None]:
    try:
        record = run_job(job)
    except Exception as exc:  # noqa: BLE001 - report and go on with the other jobs
        return job.job_id, 0.0, f"{type(exc).__name__}: {exc}"
    record["provenance"] = prov
    _write(record, result_path(job, root))
    return job.job_id, record["seconds"], None


def pending(config_path: str, root: Path = RESULTS_ROOT) -> tuple[list[Job], list[Job]]:
    jobs = expand(load(config_path))
    todo = [j for j in jobs if not result_path(j, root).exists()]
    return jobs, todo


def run_config(config_path: str, workers: int = 1, root: Path = RESULTS_ROOT) -> int:
    """Run the jobs not done yet; return the number of failed jobs."""
    jobs, todo = pending(config_path, root)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"with {workers} worker(s)", flush=True)
    prov = provenance()
    if prov["git_dirty"]:
        print("warning: the adazor working tree has uncommitted changes", flush=True)
    failures = 0
    if workers <= 1:
        results = (_worker(j, root, prov) for j in todo)
        for k, (job_id, secs, err) in enumerate(results, 1):
            failures += err is not None
            print(f"[{k}/{len(todo)}] {job_id} " + (f"FAILED {err}" if err else f"{secs:.1f}s"), flush=True)
        return failures
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, j, root, prov) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            job_id, secs, err = fut.result()
            failures += err is not None
            print(f"[{k}/{len(todo)}] {job_id} " + (f"FAILED {err}" if err else f"{secs:.1f}s"), flush=True)
    return failures
