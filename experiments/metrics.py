"""Ground-truth metrics computed from alert times and known drift events."""

from __future__ import annotations

import math

__all__ = ["detection_metrics"]


def detection_metrics(alerts: list[int], events: list[tuple[int, int]], n_read: int,
                      start: int, tol: int) -> dict:
    """Score the alerts of one run against known drift events.

    Parameters
    ----------
    alerts:
        Times of the alerts (numbers of examples read).
    events:
        Drift intervals ``(begin, end)`` in the same units; an empty list for a
        stationary stream.
    n_read:
        Length of the processed stream.
    start:
        Time from which alerts are possible (the window size).
    tol:
        An event is detected by the first alert in ``[begin, end + tol]``;
        further alerts in that interval are *redundant*, alerts outside every
        such interval are *false alarms*.

    Returns
    -------
    A dictionary with the number of events, detected and missed events, the
    detection delays (``alert - begin``), false and redundant alerts, and
    false alarms per 1000 examples of the stationary part of the stream.
    """
    windows = [(b, min(e + tol, n_read)) for b, e in sorted(events)]
    detected, delays, redundant, false = 0, [], 0, 0
    first_seen = [None] * len(windows)
    for a in sorted(alerts):
        hit = next((i for i, (b, e) in enumerate(windows) if b <= a <= e), None)
        if hit is None:
            false += 1
        elif first_seen[hit] is None:
            first_seen[hit] = a
            detected += 1
            delays.append(a - windows[hit][0])
        else:
            redundant += 1
    covered = sum(max(0, min(e, n_read) - max(b, start) + 1) for b, e in windows)
    stationary = max(0, n_read - start - covered)
    return {
        "n_events": len(windows),
        "n_detected": detected,
        "n_missed": len(windows) - detected,
        "delays": delays,
        "mean_delay": sum(delays) / len(delays) if delays else math.nan,
        "n_false_alarms": false,
        "n_redundant": redundant,
        "false_alarms_per_1000": 1000 * false / stationary if stationary else math.nan,
    }
