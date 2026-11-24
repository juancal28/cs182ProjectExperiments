"""Turn per-seed summary tables into the threshold M* for each optimizer.

    python -m e1.aggregate ../results/e1/full                 # every seed found under the folder
    python -m e1.aggregate seedA/summary.csv seedB/summary.csv --out ../results/e1/aggregate

Reads only summary.csv files, so a table from any runner with the same columns works.

Crossover statistic (the proposal's definition of M*):
    s(M) = gap_ridge_new at batch 512  -  gap_ridge_new at batch 256, paired by seed.
    s > 0: more data per task moved the model away from ridge (below threshold).
    M* is where the seed-averaged s crosses from positive to negative, interpolated in log2 M.
Single-run readout (per batch size):
    r(M) = gap_dmmse_seen - gap_ridge_seen; M where r crosses from negative to positive.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1.config import GRID_BATCH, GRID_OPT

NUMERIC = ["seed", "M", "batch", "steps", "gap_ridge_new", "gap_dmmse_new", "gap_ridge_seen", "gap_dmmse_seen"]


def load_rows(paths) -> list[dict]:
    """Read summary.csv files (or folders searched recursively for them). Fresh-task rows are skipped."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("summary.csv")) if p.is_dir() else [p]
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["M"] == "inf":
                    continue
                for k in NUMERIC:
                    r[k] = float(r[k]) if k.startswith("gap") else int(float(r[k]))
                r["source"] = str(f)
                rows.append(r)
    return rows


def crossover(log2m, y, direction: str = "pos_to_neg") -> dict:
    """Zero crossings of y along the grid, by linear interpolation between adjacent points.

    status: "ok" (exactly one crossing in the wanted direction), "multiple" (more than one;
    x_star is the lowest), or "not_bracketed" (none). An exact zero counts as the far side.
    """
    sign = 1.0 if direction == "pos_to_neg" else -1.0
    pts = sorted((x, sign * v) for x, v in zip(log2m, y) if v is not None and math.isfinite(v))
    found = []
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if ya > 0 and yb <= 0:
            found.append(xa + (xb - xa) * ya / (ya - yb))
    status = "ok" if len(found) == 1 else ("multiple" if found else "not_bracketed")
    return {"status": status, "x_star": found[0] if found else None, "crossings": found}


def s_by_seed(rows, opt) -> dict:
    """{seed: {M: s(M)}} for cells where the seed has both batch sizes."""
    cell = {(r["seed"], r["M"], r["batch"]): r["gap_ridge_new"] for r in rows if r["opt"] == opt}
    out = defaultdict(dict)
    for (seed, m, batch), gap in cell.items():
        if batch == 512 and (seed, m, 256) in cell:
            out[seed][m] = gap - cell[(seed, m, 256)]
    return dict(out)


def readout_by_seed(rows, opt, batch) -> dict:
    """{seed: {M: gap_dmmse_seen - gap_ridge_seen}} at one batch size."""
    out = defaultdict(dict)
    for r in rows:
        if r["opt"] == opt and r["batch"] == batch:
            out[r["seed"]][r["M"]] = r["gap_dmmse_seen"] - r["gap_ridge_seen"]
    return dict(out)


def mean_curve(by_seed: dict, seeds=None):
    """Seed-averaged curve over the Ms present; returns (log2 M list, mean list)."""
    seeds = list(by_seed) if seeds is None else seeds
    acc = defaultdict(list)
    for sd in seeds:
        for m, v in by_seed[sd].items():
            acc[m].append(v)
    ms = sorted(acc)
    return [math.log2(m) for m in ms], [float(np.mean(acc[m])) for m in ms]


def bootstrap_interval(by_seed: dict, direction: str, n_boot: int = 2000, rng_seed: int = 0):
    """Percentile interval for the crossing, resampling seeds with replacement.

    Returns (lo, hi, fraction of resamples that had a crossing), or None with fewer than two seeds.
    """
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return None
    rng = np.random.default_rng(rng_seed)
    stars = []
    for _ in range(n_boot):
        pick = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
        res = crossover(*mean_curve(by_seed, pick), direction=direction)
        if res["x_star"] is not None:
            stars.append(res["x_star"])
    if not stars:
        return (None, None, 0.0)
    lo, hi = np.percentile(stars, [2.5, 97.5])
    return (float(lo), float(hi), len(stars) / n_boot)


def missing_cells(rows) -> list[str]:
    """Cells absent for some seed, optimizer or batch size, over the pool sizes that appear in any table.

    The pool sizes are read from the tables, not from the default grid, because a seed may be run with others.
    """
    have = {(r["seed"], r["opt"], r["M"], r["batch"]) for r in rows}
    out = []
    for seed in sorted({r["seed"] for r in rows}):
        for opt in GRID_OPT:
            for m in sorted({r["M"] for r in rows}):
                for b in GRID_BATCH:
                    if (seed, opt, m, b) not in have:
                        out.append(f"seed {seed} {opt} M={m} batch={b}")
    return out


def analyze(rows) -> dict:
    """Thresholds per optimizer, per-seed outcomes and the curves behind them."""
    groups = {(r["scale"], r["steps"]) for r in rows}
    if len(groups) > 1:
        raise ValueError(f"refusing to mix runs of different length: {sorted(groups)}. Pass one scale at a time.")
    result = {"scale": next(iter(groups)) if groups else None, "missing": missing_cells(rows),
              "thresholds": [], "per_seed": [], "curves": {}}
    fmt = lambda v: "" if v is None else round(v, 3)
    for opt in GRID_OPT:
        opt_rows = [r for r in rows if r["opt"] == opt]
        if not opt_rows:
            continue
        s = s_by_seed(rows, opt)
        x, y = mean_curve(s) if s else ([], [])
        main = crossover(x, y)
        ci = bootstrap_interval(s, "pos_to_neg") if s else None
        ro = {b: readout_by_seed(rows, opt, b) for b in GRID_BATCH}
        ro_star = {b: crossover(*mean_curve(ro[b]), direction="neg_to_pos") if ro[b] else {"x_star": None, "status": "no_data"}
                   for b in GRID_BATCH}
        result["curves"][opt] = {"s": s, "s_mean": (x, y), "readout": ro,
                                 "readout_mean": {b: mean_curve(ro[b]) if ro[b] else ([], []) for b in GRID_BATCH}}
        result["thresholds"].append({
            "opt": opt, "n_seeds": len({r["seed"] for r in opt_rows}), "n_seeds_paired": len(s),
            "log2_Mstar": fmt(main["x_star"]), "Mstar": "" if main["x_star"] is None else round(2 ** main["x_star"]),
            "status": main["status"], "all_crossings_log2": " ".join(f"{c:.3f}" for c in main["crossings"]),
            "ci95_lo": fmt(ci[0]) if ci else "", "ci95_hi": fmt(ci[1]) if ci else "",
            "boot_bracketed_frac": fmt(ci[2]) if ci else "",
            "readout_log2_b256": fmt(ro_star[256]["x_star"]), "readout_log2_b512": fmt(ro_star[512]["x_star"]),
            "tuned": "/".join(sorted({r["tuned"] for r in opt_rows})),
        })
        for seed in sorted({r["seed"] for r in opt_rows}):
            one = crossover(*mean_curve({seed: s[seed]})) if seed in s else {"x_star": None, "status": "no_pairs"}
            row = {"seed": seed, "opt": opt, "n_runs": sum(r["seed"] == seed for r in opt_rows),
                   "log2_Mstar": fmt(one["x_star"]), "status": one["status"]}
            for b in GRID_BATCH:
                r1 = crossover(*mean_curve({seed: ro[b][seed]}), direction="neg_to_pos") if seed in ro[b] else {"x_star": None}
                row[f"readout_log2_b{b}"] = fmt(r1["x_star"])
            result["per_seed"].append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="summary.csv files, or folders searched for them")
    p.add_argument("--out", type=Path, help="output folder (default: 'aggregate' next to the first path)")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    rows = load_rows(a.paths)
    if not rows:
        print("no summary rows found", file=sys.stderr)
        return 1
    res = analyze(rows)
    first = Path(a.paths[0])
    out = a.out or ((first if first.is_dir() else first.parent) / "aggregate")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "thresholds.csv", res["thresholds"])
    write_csv(out / "per_seed.csv", res["per_seed"])

    scale, steps = res["scale"]
    print(f"{len(rows)} runs, scale '{scale}' ({steps} steps), seeds {sorted({r['seed'] for r in rows})}")
    for t in res["thresholds"]:
        star = f"log2 M* = {t['log2_Mstar']} (M* ~ {t['Mstar']})" if t["log2_Mstar"] != "" else "M* not bracketed by the grid"
        ci = f", 95% interval [{t['ci95_lo']}, {t['ci95_hi']}]" if t["ci95_lo"] != "" else ""
        flag = "  [several sign changes: " + t["all_crossings_log2"] + "]" if t["status"] == "multiple" else ""
        print(f"  {t['opt']:<6} {star}{ci}{flag}; single-run readout log2 M: "
              f"batch 256 = {t['readout_log2_b256'] or 'n/a'}, batch 512 = {t['readout_log2_b512'] or 'n/a'}; "
              f"seeds {t['n_seeds_paired']}; learning rate {t['tuned']}")
    if res["missing"]:
        print(f"  {len(res['missing'])} grid cells missing, e.g. {res['missing'][0]}")
        (out / "missing.txt").write_text("\n".join(res["missing"]) + "\n", encoding="utf-8")
    if not a.no_plots:
        from e1.plots import save_figures
        for f in save_figures(res, out):
            print(f"  figure: {f}")
    print(f"tables: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
