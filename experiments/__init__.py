"""Config-driven experiments for the ADAZOR paper.

Usage (from the repository root)::

    python -m experiments run configs/e2_simulations.toml --workers 4
    python -m experiments status configs/e2_simulations.toml
    python -m experiments analyze configs/e2_simulations.toml --out-dir figures/

Each configuration expands into a grid of jobs; every job writes one JSON
record to ``results/<experiment>/<job_id>.json``, so that interrupted runs can
be resumed and finished jobs are never repeated.
"""
