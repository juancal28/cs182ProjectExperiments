"""Quantities read from the training logs of finished runs: the two clocks and their stand-ins.

Used by e1.paper. Everything here is a pure function of numbers already on disk.

    t_reg     first logged step at which new-task error is within 10% of ridge's (the proposal's
              regression clock, here read from pooled runs instead of fresh-task runs)
    phi       how far seen-task error has moved from ridge's toward dMMSE's:
              0 = ridge, 1 = dMMSE. Positive values need knowledge of the pool.
    t_onset   first logged step from which phi stays at or above a level; a stand-in for the
              proposal's memorization clock t_mem, which no run reached in 500,000 steps
    rebound   how far the distance to ridge on new tasks has grown back from its minimum
"""
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

LOG_KEYS = ["step", "gap_ridge_new", "gap_ridge_seen", "gap_dmmse_seen", "mse_model_new", "mse_model_seen"]


def load_runs(root) -> dict:
    """{(seed, opt, M, batch): {"final": final.json, <log key>: array over logged steps}}"""
    runs = {}
    for d in sorted(Path(root).glob("seed_*/*")):
        if not (d / "final.json").exists():
            continue
        final = json.loads((d / "final.json").read_text(encoding="utf-8"))
        if final["M"] == "inf":
            continue
        recs = [json.loads(line) for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        run = {"final": final}
        for k in LOG_KEYS:
            run[k] = np.array([r[k] for r in recs], dtype=float)
        runs[(final["seed"], final["opt"], final["M"], final["batch"])] = run
    return runs


def first_step_within(steps, mse_model, mse_ref: float, tol: float = 0.10) -> Optional[int]:
    """First logged step at which mse_model <= (1 + tol) * mse_ref."""
    hit = np.nonzero(np.asarray(mse_model) <= (1 + tol) * mse_ref)[0]
    return int(steps[hit[0]]) if len(hit) else None


def memorized_fraction(mse_model_seen, mse_ridge_seen: float, mse_dmmse_seen: float):
    """phi = (ridge error - model error) / (ridge error - dMMSE error) on seen tasks."""
    return (mse_ridge_seen - np.asarray(mse_model_seen)) / (mse_ridge_seen - mse_dmmse_seen)


def onset_step(steps, phi, level: float = 0.05, hold: int = 5) -> Optional[int]:
    """First logged step (after 0) that starts `hold` consecutive evaluations with phi >= level."""
    ok = np.asarray(phi) >= level
    for i in range(1, len(ok) - hold + 1):
        if ok[i:i + hold].all():
            return int(steps[i])
    return None


def rebound(steps, gap) -> dict:
    """Minimum of the distance to ridge during training and its relative growth by the end."""
    gap = np.asarray(gap)
    i = int(gap.argmin())
    return {"step_min": int(steps[i]), "min": float(gap[i]), "final": float(gap[-1]),
            "rise": float(gap[-1] / gap[i] - 1.0)}


def level_crossing(log2m, y, tau: float) -> Optional[float]:
    """Largest log2 M at which y falls through tau, interpolating log y linearly in log2 M.

    Beyond the returned pool size every grid point is below tau. None if the largest pool is
    still above tau or no point is above it.
    """
    pts = sorted(zip(log2m, y))
    if pts[-1][1] > tau:
        return None
    for (xa, ya), (xb, yb) in reversed(list(zip(pts, pts[1:]))):
        if ya > tau >= yb:
            return xa + (xb - xa) * (math.log(ya) - math.log(tau)) / (math.log(ya) - math.log(yb))
    return None


def power_law(ms, ts) -> Optional[dict]:
    """Least-squares fit of t = c * M^nu in log-log space. Needs three or more points."""
    ms, ts = np.asarray(ms, dtype=float), np.asarray(ts, dtype=float)
    if len(ms) < 3 or len(set(ms.tolist())) < 2:
        return None
    nu, logc = np.polyfit(np.log2(ms), np.log2(ts), 1)
    return {"nu": float(nu), "log2_c": float(logc), "n": int(len(ms))}
