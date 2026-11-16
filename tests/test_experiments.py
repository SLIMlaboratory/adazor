"""Tests of the experiment pipeline (configs, streams, metrics, runner)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.config import expand
from experiments.metrics import detection_metrics
from experiments.run import pending, run_config
from experiments.streams import InjectedShiftStream, mixture_scenario


def test_metrics_on_hand_made_alerts():
    m = detection_metrics(alerts=[300, 2100, 2150, 2900, 6100], events=[(2001, 3001)], n_read=8000,
                          start=250, tol=250)
    assert m["n_detected"] == 1 and m["n_missed"] == 0
    assert m["delays"] == [99]
    assert m["n_redundant"] == 2              # 2150 and 2900 fall in the same event window
    assert m["n_false_alarms"] == 2           # 300 and 6100
    stationary = 8000 - 250 - (3251 - 2001 + 1)
    assert m["false_alarms_per_1000"] == pytest.approx(2000 / stationary)


def test_metrics_missed_and_stationary():
    m = detection_metrics(alerts=[], events=[(2501, 2501)], n_read=8000, start=250, tol=250)
    assert m["n_missed"] == 1 and math.isnan(m["mean_delay"])
    s = detection_metrics(alerts=[1000], events=[], n_read=1250, start=250, tol=250)
    assert s["n_false_alarms"] == 1 and s["false_alarms_per_1000"] == 1.0


def test_scenarios_have_events_in_reading_units():
    stream, events = mixture_scenario("setting1", seed=0)
    assert events == [(2001, 3001), (5001, 6001)]
    assert len(list(stream)) == 8000
    _, ev = mixture_scenario("label_swap_abrupt", seed=0)
    assert ev == [(2501, 2501), (5501, 5501)]
    assert mixture_scenario("stationary", seed=0)[1] == []


def test_label_swap_keeps_inputs():
    from experiments.streams import SCENARIOS
    p0, p1, _, _ = SCENARIOS["label_swap_abrupt"]
    assert sorted(map(tuple, p0.mu)) == sorted(map(tuple, p1.mu))
    assert not np.allclose(p0.mu, p1.mu)


def test_injected_shift_stream():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    y = np.zeros(4000, dtype=int)
    s = InjectedShiftStream(X, y, feature=0, position=1000, width=0, strength=1.0, seed=0, length=1800)
    xs = np.array([x for x, _ in s])
    median = np.median(X[:, 0])
    assert len(xs) == 1800 and s.events == [(1001, 1001)]
    assert 0.4 < np.mean(xs[:1000, 0] > median) < 0.6
    assert np.all(xs[1000:, 0] >= median)
    assert len({tuple(x) for x in xs}) == len(xs)          # drawn without replacement


def write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "tiny.toml"
    cfg.write_text("""
name = "tiny"
seeds = [1, 2]
[run_defaults]
n = 50
eval_every = 0
[[streams]]
kind = "mixture"
scenario = "virtual_abrupt"
length = 3000
[[monitors]]
kind = "adazor"
[[monitors]]
kind = "ocdd"
f = 0.1
[[models]]
name = "gnb"
""")
    return cfg


def test_expand_dedups_and_ids_are_stable(tmp_path):
    from experiments.config import load
    cfg = load(write_config(tmp_path))
    jobs = expand(cfg)
    assert len(jobs) == 4
    assert [j.job_id for j in jobs] == [j.job_id for j in expand(cfg)]
    cfg["blocks"] = [{}, {}]                     # identical blocks -> same jobs
    assert len(expand(cfg)) == 4
    assert jobs[0].spec["monitor"]["outlier"] == {}   # outlier defaults attached to OCSVM monitors


def test_run_resume_and_records(tmp_path):
    cfg = write_config(tmp_path)
    root = tmp_path / "results"
    assert run_config(str(cfg), workers=1, root=root) == 0
    jobs, todo = pending(str(cfg), root)
    assert len(jobs) == 4 and todo == []
    rec = json.loads(next((root / "tiny").glob("*.json")).read_text())
    assert rec["result"]["events"] == [[2501, 2501]]
    assert rec["provenance"]["adazor"]
    assert run_config(str(cfg), workers=1, root=root) == 0   # nothing left to do
