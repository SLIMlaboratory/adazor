"""Tests of the code that turns results into the paper's numbers.

Covers ``experiments.analysis`` (rows, tables, figures), ``experiments.properties``
(the checks of Sect. 4.3) and ``experiments.registry`` (from specifications to
streams, monitors and labels).
"""

import math

import numpy as np
import pandas as pd
import pytest

from adazor import (
    D3Monitor,
    KSWINMonitor,
    NullMonitor,
    OutlierRateMonitor,
    ThresholdDriftDetector,
    ZTestDriftDetector,
    effective_level,
)
from experiments.analysis import analyze, rows, table
from experiments.properties import run_properties
from experiments.registry import build_model_factory, build_monitor, build_stream, data_root, label
from experiments.run import run_config

# --------------------------------------------------------------------------- analysis

def record(monitor, alerts, events, seed=1, n=500, n_read=8000, acc=0.9):
    return {
        "experiment": "x", "job_id": f"{monitor['kind']}-{seed}", "kind": "stream", "seconds": 1.0,
        "spec": {"stream": {"kind": "mixture", "scenario": "virtual_abrupt"}, "monitor": monitor,
                 "model": {"name": "ht"}, "seed": seed, "run": {"n": n, "r": 0.1}},
        "result": {"n_alerts": len(alerts), "n_read": n_read, "alerts": alerts, "events": events,
                   "prequential_accuracy": acc, "mean_window_accuracy": acc},
    }


ADAZOR = {"kind": "adazor", "alpha": 0.025, "baseline": "holdout"}
OCDD = {"kind": "ocdd", "f": 0.1}


def test_rows_labels_parameters_and_metrics():
    recs = [record(ADAZOR, alerts=[2600, 7000], events=[(2501, 2501)]),
            record(OCDD, alerts=[1000, 2550, 2600], events=[(2501, 2501)])]
    df = rows(recs, tol=250).set_index("monitor")
    a, o = df.loc["ADAZOR a=0.025 conservative/holdout"], df.loc["OCDD f=0.1"]
    assert a["stream"] == "virtual_abrupt" and a["model"] == "ht"
    assert a["alpha"] == 0.025 and math.isnan(a["f"]) and o["f"] == 0.1
    assert (a["n"], a["r"], a["seed"]) == (500, 0.1, 1)
    # ADAZOR: detection after 99 examples, one false alarm at 7000
    assert (a["n_detected"], a["n_missed"], a["n_false_alarms"]) == (1, 0, 1)
    assert a["mean_delay"] == 99
    # OCDD: the first alert in the event window detects, the second is redundant
    assert (o["n_detected"], o["n_redundant"], o["n_false_alarms"]) == (1, 1, 1)
    assert o["mean_delay"] == 49


def test_rows_tolerance_defaults_to_window_size():
    # an alert 400 examples after the event: missed with tol=250, detected with tol=n=500
    rec = record(ADAZOR, alerts=[2901], events=[(2501, 2501)], n=500)
    assert rows([rec], tol=250).loc[0, "n_missed"] == 1
    assert rows([rec], tol=None).loc[0, "n_detected"] == 1


def test_rows_without_ground_truth_and_properties():
    rec = record(OCDD, alerts=[1000], events=None)
    row = rows([rec]).loc[0]
    assert row["n_alerts"] == 1 and "n_detected" not in row.index
    prop = {"kind": "properties", "spec": {"task": "level", "seed": 3, "n": 250},
            "result": {"empirical_conservative": 0.01}}
    prow = rows([prop]).loc[0]
    assert (prow["task"], prow["seed"], prow["n"], prow["source"]) == ("level", 3, 250, "")
    assert prow["empirical_conservative"] == 0.01


def test_table_reports_mean_and_std_per_group(tmp_path):
    df = pd.DataFrame({"stream": ["s", "s", "t"], "monitor": ["m", "m", "m"],
                       "n_alerts": [2, 4, 5], "seed": [1, 2, 1]})
    base = table(df, {"name": "alerts", "group_by": ["stream", "monitor"], "metrics": ["n_alerts"],
                      "digits": 1}, tmp_path, "exp")
    tex = base.with_suffix(".tex").read_text()
    assert "s & m & $3.0\\pm1.4$\\\\" in tex          # mean 3, sample std sqrt(2)
    assert "t & m & $5.0$\\\\" in tex                 # a single run has no std
    csv = pd.read_csv(base.with_suffix(".csv"))
    assert csv.loc[0, "n_alerts_mean"] == 3.0


def test_table_filter(tmp_path):
    df = pd.DataFrame({"stream": ["s", "t"], "n_alerts": [2, 5]})
    base = table(df, {"name": "f", "group_by": ["stream"], "metrics": ["n_alerts"],
                      "where": {"stream": "t"}}, tmp_path, "exp")
    tex = base.with_suffix(".tex").read_text()
    assert "t & $5.00$" in tex and "s &" not in tex


def test_analyze_end_to_end(tmp_path):
    cfg = tmp_path / "tiny.toml"
    cfg.write_text("""
name = "tiny"
seeds = [1, 2]
[run_defaults]
n = 50
eval_every = 10
[[streams]]
kind = "mixture"
scenario = "virtual_abrupt"
length = 1500
[[monitors]]
kind = "adazor"
label = "ADAZOR"
[[monitors]]
kind = "ocdd"
label = "OCDD"
[[models]]
name = "gnb"
[analysis]
tol = 50
[[analysis.tables]]
name = "alerts"
group_by = ["stream", "monitor"]
metrics = ["n_alerts", "n_missed"]
[[analysis.figures]]
type = "trajectories"
name = "traj"
streams = ["virtual_abrupt"]
monitors = ["ADAZOR", "OCDD"]
[[analysis.figures]]
type = "alerts"
name = "alerts"
streams = ["virtual_abrupt"]
monitors = ["ADAZOR", "OCDD"]
seed = 1
[[analysis.figures]]
type = "sweep"
name = "sweep"
x = "seed"
y = "n_alerts"
""")
    root = tmp_path / "results"
    with pytest.raises(SystemExit):                    # nothing run yet
        analyze(str(cfg), str(tmp_path / "out"), root=root)
    assert run_config(str(cfg), workers=1, root=root) == 0
    produced = analyze(str(cfg), str(tmp_path / "out"), root=root)
    names = {p.name for p in produced}
    assert names == {"tiny_rows.csv", "tiny_alerts.tex", "tiny_traj.pdf", "tiny_alerts.pdf", "tiny_sweep.pdf"}
    assert all(p.exists() and p.stat().st_size > 0 for p in produced)
    df = pd.read_csv(tmp_path / "out" / "tiny_rows.csv")
    assert len(df) == 4 and set(df.monitor) == {"ADAZOR", "OCDD"}


# --------------------------------------------------------------------------- properties

def test_level_task_matches_effective_level():
    res = run_properties({"task": "level", "seed": 0, "n": 250, "p": 0.3, "alpha": 0.05, "reps": 100_000})
    assert res["empirical_conservative"] <= res["nominal"]
    assert res["empirical_conservative"] == pytest.approx(effective_level(0.05, 0.3), abs=0.005)
    assert res["empirical_pooled"] == pytest.approx(0.05, abs=0.01)


def test_ocsvm_rates_on_mixture():
    res = run_properties({"task": "ocsvm_rates", "seed": 0, "n": 100, "reps": 20,
                          "source": {"kind": "mixture", "scenario": "stationary"}})
    assert set(res) == {"in_sample_mean", "in_sample_std", "out_of_sample_mean", "out_of_sample_std",
                        "threshold_firing_rate", "adazor_firing_rate"}
    # the OCSVM sees its own training points as more typical than new ones
    assert res["out_of_sample_mean"] > res["in_sample_mean"]
    assert 0.0 <= res["threshold_firing_rate"] <= 1.0 and 0.0 <= res["adazor_firing_rate"] <= 1.0


@pytest.fixture
def csv_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"few": rng.integers(0, 3, 2000), "many": rng.normal(size=2000),
                       "target": rng.integers(0, 2, 2000)})
    df.to_csv(tmp_path / "toy.csv", index=False)
    monkeypatch.setenv("OCC_DATA", str(tmp_path))
    return tmp_path


def test_ocsvm_rates_on_csv(csv_root):
    res = run_properties({"task": "ocsvm_rates", "seed": 0, "n": 100, "reps": 5,
                          "source": {"kind": "csv", "file": "toy.csv"}})
    assert 0.0 <= res["in_sample_mean"] <= 1.0


def test_unknown_properties_task():
    with pytest.raises(ValueError):
        run_properties({"task": "nope", "seed": 0})


# --------------------------------------------------------------------------- registry

def test_data_root_follows_environment(csv_root):
    assert data_root() == csv_root


def test_build_monitor_settings():
    m = build_monitor({"kind": "adazor", "alpha": 0.01, "variance": "pooled", "baseline": "holdout",
                       "outlier": {"nu": 0.2, "gamma": 0.5, "standardize": False}}, seed=1)
    assert isinstance(m, OutlierRateMonitor) and m.baseline == "holdout"
    assert isinstance(m.rule, ZTestDriftDetector) and (m.rule.alpha, m.rule.variance) == (0.01, "pooled")
    o = m.outlier_factory()
    assert (o.nu, o.gamma, o.standardize) == (0.2, 0.5, False)
    o = build_monitor({"kind": "ocdd", "f": 0.2}, seed=1)
    assert isinstance(o.rule, ThresholdDriftDetector) and o.rule.f == 0.2
    assert o.baseline == "in_sample" and o.outlier_factory().gamma == "auto"
    k = build_monitor({"kind": "kswin", "alpha": 0.001}, seed=1)
    assert isinstance(k, KSWINMonitor) and k.alpha == 0.001 and k.stat_size == 30
    d = build_monitor({"kind": "d3", "tau": 0.8}, seed=1)
    assert isinstance(d, D3Monitor) and (d.tau, d.rho) == (0.8, 0.2)
    assert isinstance(build_monitor({"kind": "none"}, seed=1), NullMonitor)
    with pytest.raises(ValueError):
        build_monitor({"kind": "nope"}, seed=1)


def test_build_stream_csv_and_shuffled_control(csv_root):
    s, events = build_stream({"kind": "csv", "file": "toy.csv"}, seed=1)
    assert events is None                              # real data: drifts unknown
    first = next(iter(s))[0]
    X = pd.read_csv(csv_root / "toy.csv").drop(columns="target").to_numpy(dtype=float)
    assert np.array_equal(first, X[0])                 # original order preserved
    _, events = build_stream({"kind": "csv", "file": "toy.csv", "shuffle": True}, seed=1)
    assert events == []                                # shuffled copy: stationary by construction
    with pytest.raises(ValueError):
        build_stream({"kind": "nope"}, seed=1)


def test_build_stream_injected_uses_feature_with_most_values(csv_root):
    s, events = build_stream({"kind": "injected", "file": "toy.csv", "position": 500, "length": 1500},
                             seed=1)
    assert events == [(501, 501)]
    xs = np.array([x for x, _ in s])
    median = np.median(pd.read_csv(csv_root / "toy.csv")["many"])
    assert np.mean(xs[500:, 1] > median) > 0.8         # shift on 'many', not on 'few'
    assert 0.35 < np.mean(xs[:500, 1] > median) < 0.65


def test_model_factory_gives_fresh_models():
    f = build_model_factory({"name": "gnb", "standardize": True}, seed=1)
    assert f() is not f()


@pytest.mark.parametrize("spec,what,expected", [
    ({"kind": "adazor"}, "monitor", "ADAZOR a=0.025 conservative/in_sample"),
    ({"kind": "adazor", "alpha": 0.05, "variance": "pooled", "baseline": "holdout"}, "monitor",
     "ADAZOR a=0.05 pooled/holdout"),
    ({"kind": "ocdd", "f": 0.25}, "monitor", "OCDD f=0.25"),
    ({"kind": "none"}, "monitor", "no detection"),
    ({"kind": "kswin"}, "monitor", "KSWIN"),
    ({"kind": "d3"}, "monitor", "D3"),
    ({"kind": "adazor", "label": "custom"}, "monitor", "custom"),
    ({"kind": "mixture", "scenario": "setting1"}, "stream", "setting1"),
    ({"kind": "csv", "file": "real/elec.csv"}, "stream", "elec"),
    ({"kind": "csv", "file": "real/elec.csv", "shuffle": True}, "stream", "elec (shuffled)"),
    ({"kind": "injected", "file": "real/elec.csv", "width": 1000}, "stream", "elec (shift, w=1000)"),
    ({"name": "ht", "standardize": True}, "model", "ht (std)"),
    ({"name": "gnb"}, "model", "gnb"),
])
def test_labels(spec, what, expected):
    assert label(spec, what) == expected


def test_label_rejects_unknown_kind():
    with pytest.raises(ValueError):
        label({"kind": "adazor"}, "nope")
