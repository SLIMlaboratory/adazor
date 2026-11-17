"""Tables and figures from the stored results of a configuration.

The ``[analysis]`` section of a configuration lists what to produce::

    [analysis]
    tol = 250                          # tolerance of the ground-truth metrics

    [[analysis.tables]]
    name = "alerts"                    # -> <out-dir>/<experiment>_alerts.{csv,tex}
    group_by = ["stream", "monitor"]
    metrics = ["n_alerts", "prequential_accuracy", "n_missed", "mean_delay"]
    where = {model = "ht"}             # optional filter on row values

    [[analysis.figures]]
    type = "trajectories"              # or "alerts", "sweep"
    name = "setting1"
    streams = ["setting1"]
    monitors = ["no detection", "OCDD f=0.1", "ADAZOR a=0.025 conservative/in_sample"]

Rows are built from the result records: one row per job, with the labels of
stream, monitor and model, the run parameters, the seed, the counts and
accuracies of :class:`adazor.RunResult`, and the ground-truth metrics of
:func:`experiments.metrics.detection_metrics` when the drift events are known.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import expand, load
from .metrics import detection_metrics
from .registry import label
from .run import RESULTS_ROOT, result_path


def load_records(config_path: str, root: Path = RESULTS_ROOT) -> tuple[dict, list[dict]]:
    cfg = load(config_path)
    records = []
    for job in expand(cfg):
        path = result_path(job, root)
        if path.exists():
            records.append(json.loads(path.read_text()))
    return cfg, records


def rows(records: list[dict], tol: int | None = None) -> pd.DataFrame:
    out = []
    for rec in records:
        spec, res = rec["spec"], rec["result"]
        if rec["kind"] == "properties":
            src = spec.get("source", {})
            row = {"task": spec["task"], "seed": spec["seed"],
                   "source": label(src, "stream") if src else "",
                   **{k: v for k, v in spec.items() if k not in ("task", "seed", "source")}, **res}
            out.append(row)
            continue
        run = spec["run"]
        row = {
            "stream": label(spec["stream"], "stream"),
            "monitor": label(spec["monitor"], "monitor"),
            "model": label(spec["model"], "model"),
            "seed": spec["seed"], "n": run["n"], "r": run["r"],
            "alpha": spec["monitor"].get("alpha"), "f": spec["monitor"].get("f"),
            "n_alerts": res["n_alerts"], "n_read": res["n_read"],
            "prequential_accuracy": res["prequential_accuracy"],
            "mean_window_accuracy": res["mean_window_accuracy"],
            "seconds": rec["seconds"],
        }
        if res.get("events") is not None:
            m = detection_metrics(res["alerts"], [tuple(e) for e in res["events"]], res["n_read"],
                                  start=run["n"], tol=run["n"] if tol is None else tol)
            row.update({k: v for k, v in m.items() if k != "delays"})
        out.append(row)
    return pd.DataFrame(out)


def _filter(df: pd.DataFrame, where: dict | None) -> pd.DataFrame:
    for k, v in (where or {}).items():
        df = df[df[k].isin(v if isinstance(v, list) else [v])]
    return df


def _fmt(mean: float, std: float, digits: int) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or math.isnan(std) or std == 0:
        return f"${mean:.{digits}f}$"
    return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"


def table(df: pd.DataFrame, spec: dict, out_dir: Path, experiment: str) -> Path:
    df = _filter(df, spec.get("where"))
    keys, metrics, digits = spec["group_by"], spec["metrics"], spec.get("digits", 2)
    grouped = df.groupby(keys, sort=False)[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    summary = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    base = out_dir / f"{experiment}_{spec['name']}"
    summary.to_csv(base.with_suffix(".csv"))
    lines = ["\\begin{tabular}{@{}" + "l" * len(keys) + "r" * len(metrics) + "@{}}", "\\toprule",
             " & ".join(keys + [m.replace("_", " ") for m in metrics]) + "\\\\ \\colrule"]
    for idx in mean.index:
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(i) for i in idx]
        cells += [_fmt(mean.loc[idx, m] if len(idx) > 1 else mean.loc[idx[0], m],
                       std.loc[idx, m] if len(idx) > 1 else std.loc[idx[0], m], digits) for m in metrics]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\botrule", "\\end{tabular}"]
    base.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    return base


# --------------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _shade(ax, events):
    for b, e in events or []:
        if e > b:
            ax.axvspan(b, e, color="0.88", zorder=0)
        else:
            ax.axvline(b, color="0.6", lw=0.8, zorder=0)


def _trajectories(records, spec, path):
    plt = _style()
    streams, monitors = spec["streams"], spec["monitors"]
    fig, axes = plt.subplots(len(streams), len(monitors), sharey=True, squeeze=False,
                             figsize=(2.3 * len(monitors), 1.7 * len(streams)), constrained_layout=True)
    for i, s in enumerate(streams):
        for j, m in enumerate(monitors):
            ax = axes[i][j]
            sel = [r for r in records if r["kind"] == "stream" and label(r["spec"]["stream"], "stream") == s
                   and label(r["spec"]["monitor"], "monitor") == m
                   and (spec.get("model") is None or r["spec"]["model"]["name"] == spec["model"])]
            counts = []
            for r in sel:
                res = r["result"]
                counts.append(res["n_alerts"])
                if res["window_accuracy"]:
                    t, a = zip(*res["window_accuracy"])
                    ax.step(t, a, where="post", color="tab:blue", alpha=0.35, lw=0.7)
                if res["alert_window_accuracy"]:
                    t, a = zip(*res["alert_window_accuracy"])
                    ax.scatter(t, a, color="tab:red", s=5, zorder=3)
            if sel:
                _shade(ax, sel[0]["result"].get("events"))
            title = f"{m}: {np.mean(counts):.1f} alerts" if counts else m
            ax.set_title(title if i == 0 else f"{np.mean(counts):.1f} alerts" if counts else "", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{s}\nwindow accuracy")
            if i == len(streams) - 1:
                ax.set_xlabel("examples read")
    fig.savefig(path)
    plt.close(fig)


def _alerts(records, spec, path):
    plt = _style()
    streams, monitors = spec["streams"], spec["monitors"]
    seed = spec.get("seed")
    fig, axes = plt.subplots(len(streams), 1, squeeze=False, figsize=(6.6, 0.25 + 0.22 * len(monitors) * len(streams)),
                             constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, s in enumerate(streams):
        ax = axes[i][0]
        for j, m in enumerate(monitors):
            sel = [r for r in records if r["kind"] == "stream" and label(r["spec"]["stream"], "stream") == s
                   and label(r["spec"]["monitor"], "monitor") == m
                   and (seed is None or r["spec"]["seed"] == seed)
                   and (spec.get("model") is None or r["spec"]["model"]["name"] == spec["model"])]
            if not sel:
                continue
            res = sel[0]["result"]
            ax.scatter(res["alerts"], [j] * len(res["alerts"]), marker="|", s=40, color=colors[j % len(colors)])
            if j == 0:
                _shade(ax, res.get("events"))
        ax.set_yticks(range(len(monitors)), monitors)
        ax.set_ylim(-0.6, len(monitors) - 0.4)
        ax.set_title(s, fontsize=8, loc="left")
    axes[-1][0].set_xlabel("examples read")
    fig.savefig(path)
    plt.close(fig)


def _sweep(df, spec, path):
    plt = _style()
    df = _filter(df, spec.get("where"))
    x, y, series = spec["x"], spec["y"], spec.get("series", "monitor")
    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    for name, g in df.groupby(series, sort=False):
        stats = g.groupby(x)[y].agg(["mean", "std"]).sort_index()
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", ms=3, capsize=2, lw=1, label=name)
    ax.set_xlabel(x)
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(path)
    plt.close(fig)


def analyze(config_path: str, out_dir: str, root: Path = RESULTS_ROOT) -> list[Path]:
    cfg, records = load_records(config_path, root)
    if not records:
        raise SystemExit("no results yet: run the configuration first")
    acfg = cfg.get("analysis", {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = rows(records, acfg.get("tol"))
    produced = []
    df.to_csv(out / f"{cfg['name']}_rows.csv", index=False)
    produced.append(out / f"{cfg['name']}_rows.csv")
    for spec in acfg.get("tables", []):
        produced.append(table(df, spec, out, cfg["name"]).with_suffix(".tex"))
    for spec in acfg.get("figures", []):
        path = out / f"{cfg['name']}_{spec['name']}.pdf"
        if spec["type"] == "trajectories":
            _trajectories(records, spec, path)
        elif spec["type"] == "alerts":
            _alerts(records, spec, path)
        elif spec["type"] == "sweep":
            _sweep(df, spec, path)
        else:
            raise ValueError(f"unknown figure type '{spec['type']}'")
        produced.append(path)
    return produced
