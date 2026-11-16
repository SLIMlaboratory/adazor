"""Plotting helpers for the experiment scripts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"none": "No drift detection", "threshold": "Fixed threshold", "adazor": "ADAZOR"}


def trajectories(results: dict, path: str) -> None:
    """One panel per detector: window accuracy of every run, alerts as dots,
    drift intervals shaded."""
    names = list(results["runs"])
    fig, axes = plt.subplots(1, len(names), sharey=True, figsize=(4.5 * len(names), 3.2),
                             constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        runs = results["runs"][name]
        for run in runs:
            if run["window_accuracy"]:
                t, a = zip(*run["window_accuracy"])
                ax.step(t, a, where="post", color="tab:blue", alpha=0.3, lw=0.8)
            if run["alert_window_accuracy"]:
                t, a = zip(*run["alert_window_accuracy"])
                ax.scatter(t, a, color="tab:red", s=8, zorder=3)
        for start, end in results.get("drift_points", []):
            ax.axvspan(start, end, color="0.85", zorder=0)
        counts = [r["n_alerts"] for r in runs]
        ax.set_title(f"{LABELS.get(name, name)}: {np.mean(counts):.2f} ± {np.std(counts, ddof=1):.2f} alerts",
                     fontsize=9)
        ax.set_xlabel("examples read")
    np.atleast_1d(axes)[0].set_ylabel("window accuracy")
    fig.savefig(path)
    plt.close(fig)
