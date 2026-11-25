"""Figures that read the runs against the measured thresholds M*.

    python -m e1.mstar_figures ../results/e1/full        # writes into ../results/e1/full/aggregate

  mstar_summary.png              M* per optimizer: pooled estimate, bootstrap interval, each seed
  training_by_side_of_mstar.png  distance to ridge through training, runs grouped by side of M*
  memorization_vs_pool.png       error on training-pool tasks against ridge and dMMSE, M* marked

M* and its interval come from e1.aggregate (the batch-size crossover); nothing is re-estimated here.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from e1.aggregate import analyze, load_rows
from e1.config import GRID_BATCH, GRID_OPT
from e1.plots import AXIS, COLOR, GRID, INK, INK2, LABEL, MUTED, SURFACE

ABOVE = "#898781"          # neutral for runs above the threshold interval


def _num(v):
    return None if v in ("", None) else float(v)


def _pow(x: float) -> str:
    """Label for a pool size given its log2; grid sizes are half powers of two (1448 is 2^10.5)."""
    half = round(x * 2) / 2
    return f"2^{half:g}" if abs(x - half) < 0.01 else f"2^{x:.2f}"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def _head(fig, title, subtitle, top):
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=13, color=INK)
    fig.text(0.012, top, subtitle, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.4)


def _thresholds(res) -> dict:
    """{opt: {"star", "lo", "hi", "seeds": {seed: x or None}}} in log2 M."""
    out = {}
    for t in res["thresholds"]:
        out[t["opt"]] = {"star": _num(t["log2_Mstar"]), "lo": _num(t["ci95_lo"]), "hi": _num(t["ci95_hi"]), "seeds": {}}
    for r in res["per_seed"]:
        out[r["opt"]]["seeds"][r["seed"]] = _num(r["log2_Mstar"])
    return out


def _side(log2m: float, th: dict) -> str:
    """below / inside / above the threshold interval (the pooled estimate when there is no interval)."""
    lo = th["lo"] if th["lo"] is not None else th["star"]
    hi = th["hi"] if th["hi"] is not None else th["star"]
    if lo is None:
        return "inside"
    return "below" if log2m < lo - 0.01 else ("above" if log2m > hi + 0.01 else "inside")


def fig_summary(res, rows, out: Path) -> Path:
    th = _thresholds(res)
    opts = [o for o in GRID_OPT if o in th]
    grid = sorted({math.log2(r["M"]) for r in rows})
    fig, ax = plt.subplots(figsize=(9.6, 1.5 + 0.95 * len(opts)))
    scale, steps = res["scale"]
    seeds = sorted({r["seed"] for r in rows})
    _head(fig, "Task-diversity threshold M* by optimizer",
          "Pool size at which a larger batch stops pushing the model away from ridge.\n"
          f"{scale.capitalize()} scale, {steps:,} steps per run, seeds {', '.join(map(str, seeds))}.", 0.895)
    _style(ax)
    for g in grid:
        ax.axvline(g, color=GRID, lw=0.6, zorder=0)
    ax.axvline(11.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(11.5, len(opts) - 0.3, " published for Adam, 4 layers: about 2^11.5", fontsize=8, color=MUTED, va="center", ha="left")
    for i, opt in enumerate(opts):
        y = len(opts) - 1 - i
        t = th[opt]
        if t["lo"] is not None:
            ax.plot([t["lo"], t["hi"]], [y, y], color=COLOR[opt], lw=9, alpha=0.28, solid_capstyle="round", zorder=2)
        for seed, x in sorted(t["seeds"].items()):
            if x is not None:
                ax.plot(x, y, "o", ms=7, mfc=SURFACE, mec=COLOR[opt], mew=1.8, zorder=3)
        if t["star"] is not None:
            ax.plot(t["star"], y, "o", ms=10, color=COLOR[opt], mec=SURFACE, mew=2, zorder=4)
            note = f"M* = 2^{t['star']:.2f}  (about {round(2 ** t['star']):,} tasks)"
        else:
            xs = [f"2^{x:.2f}" for x in t["seeds"].values() if x is not None]
            note = "not pinned down: seeds give " + " and ".join(xs)
        if t["lo"] is not None:
            note += f";  95% interval 2^{t['lo']:.2f} to 2^{t['hi']:.2f}"
        box = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
        ax.text(grid[0] - 0.25, y + 0.3, LABEL[opt], fontsize=11, color=INK, va="center", ha="left", bbox=box, zorder=5)
        ax.text(grid[0] + 0.75, y + 0.3, note, fontsize=9, color=INK2, va="center", ha="left", bbox=box, zorder=5)
    ax.set_ylim(-0.75, len(opts) - 0.1)
    ax.set_xlim(grid[0] - 0.3, grid[-1] + 0.3)
    ax.set_yticks([])
    ax.set_xticks(grid)
    ax.set_xticklabels([_pow(g) for g in grid])
    ax.set_xlabel("pool size M (number of distinct training tasks)", fontsize=9, color=INK2)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=INK2, mec=SURFACE, mew=1.5, label="both seeds pooled"),
               Line2D([], [], marker="o", ls="", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.6, label="one seed"),
               Line2D([], [], color=INK2, lw=7, alpha=0.28, solid_capstyle="round", label="95% bootstrap interval")]
    leg = ax.legend(handles=handles, loc="lower right", ncol=3, frameon=False, fontsize=8, handletextpad=0.5, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    path = out / "mstar_summary.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def _histories(rows, key: str) -> dict:
    """{(opt, M, batch): (steps, seed-mean of `key`)} from each run's log.jsonl."""
    acc = defaultdict(list)
    steps = None
    for r in rows:
        log = Path(r["source"]).parent / f"{r['opt']}_M{r['M']:05d}_b{r['batch']}" / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        steps = [x["step"] for x in recs]
        acc[(r["opt"], r["M"], r["batch"])].append([x[key] for x in recs])
    return {k: (steps, np.mean(np.array(v), axis=0)) for k, v in acc.items() if len({len(a) for a in v}) == 1}


def _spread(values, gap):
    """Move label positions apart (in the given units) so none sit closer than `gap`; keeps order."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = (pos - np.array(values, dtype=float)[order]).mean()
    pos -= shift
    out = np.empty_like(pos)
    out[order] = pos
    return out


def fig_training(res, rows, out: Path) -> Path:
    th = _thresholds(res)
    opts = [o for o in GRID_OPT if o in th]
    hist = _histories(rows, "gap_ridge_new")
    batches = [b for b in GRID_BATCH if any(k[2] == b for k in hist)]
    fig, axes = plt.subplots(len(batches), len(opts), figsize=(4.5 * len(opts), 3.5 * len(batches) + 1.1),
                             sharex=True, sharey=True, squeeze=False)
    scale, steps_total = res["scale"]
    _head(fig, "Distance to ridge through training, by side of M*",
          "Distance between the model's predictions and ridge's on new tasks (log scale), averaged over seeds. One line per pool size.\n"
          "Coloured: pool sizes below the M* interval.  Dashed: inside it.  Grey: above it.", 0.955)
    for row, batch in enumerate(batches):
        for col, opt in enumerate(opts):
            ax = axes[row][col]
            _style(ax)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ms = sorted(m for (o, m, b) in hist if o == opt and b == batch)
            ends = []
            for m in ms:
                steps, y = hist[(opt, m, batch)]
                x, y = np.array(steps[1:]), np.array(y[1:])
                side = _side(math.log2(m), th[opt])
                kw = {"below": dict(color=COLOR[opt], lw=1.7, zorder=3),
                      "inside": dict(color=COLOR[opt], lw=1.4, ls=(0, (4, 2)), alpha=0.75, zorder=2),
                      "above": dict(color=ABOVE, lw=1.1, alpha=0.85, zorder=1)}[side]
                ax.plot(x, y, **kw)
                ends.append((m, float(y[-1]), side))
            ax.set_yscale("log")
            # Direct labels at the right edge: every pool size below or inside the interval, and the largest.
            show = [e for e in ends if e[2] != "above"] + [e for e in ends if e[2] == "above"][-1:]
            if show:
                pos = _spread([math.log10(e[1]) for e in show], 0.115)
                for (m, yv, side), p in zip(show, pos):
                    ax.annotate(_pow(math.log2(m)), (steps_total, 10 ** p), xytext=(5, 0), textcoords="offset points",
                                fontsize=8, va="center", ha="left", color=INK2 if side != "above" else MUTED,
                                annotation_clip=False)
            t = th[opt]
            where = (f"M* interval 2^{t['lo']:.2f} to 2^{t['hi']:.2f}" if t["lo"] is not None
                     else (f"M* = 2^{t['star']:.2f}" if t["star"] is not None else "M* not estimated"))
            if row == 0:
                ax.set_title(f"{LABEL[opt]}: {where}", loc="left", fontsize=10.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(f"batch {batch}\ndistance to ridge, new tasks", fontsize=9, color=INK2)
            if row == len(batches) - 1:
                ax.set_xlabel("training step", fontsize=9, color=INK2)
            ax.set_xlim(0, steps_total)
            ax.xaxis.set_major_formatter(lambda v, _: f"{int(v / 1000)}k" if v else "0")
    handles = [Line2D([], [], color=INK2, lw=1.7, label="below the M* interval (optimizer colour)"),
               Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="inside the interval"),
               Line2D([], [], color=ABOVE, lw=1.1, label="above the interval")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.tight_layout(rect=(0, 0.035, 0.985, 0.925), w_pad=3.0)
    path = out / "training_by_side_of_mstar.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_memorization(res, rows, out: Path) -> Path:
    th = _thresholds(res)
    opts = [o for o in GRID_OPT if o in th]
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(5.6 * len(batches), 4.7), sharey=True, squeeze=False)
    scale, steps = res["scale"]
    _head(fig, "How far toward memorization each optimizer gets",
          "Error on prompts drawn from the training pool, averaged over seeds. A model that had memorized the pool would sit on\n"
          f"the dMMSE line; one that learned regression sits on ridge. Triangles mark each optimizer's M*. {scale.capitalize()} scale, {steps:,} steps.",
          0.93)
    mean = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))
    for ax, batch in zip(axes[0], batches):
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        sub = [r for r in rows if r["batch"] == batch]
        ms = sorted({r["M"] for r in sub})
        x = [math.log2(m) for m in ms]
        ridge = [mean([r for r in sub if r["M"] == m], "mse_ridge_seen") for m in ms]
        dmmse = [mean([r for r in sub if r["M"] == m], "mse_dmmse_seen") for m in ms]
        ax.plot(x, ridge, color=INK2, lw=1.2, zorder=1)
        ax.plot(x, dmmse, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(x[-1], ridge[-1] + 0.006, "ridge", fontsize=8.5, color=INK2, ha="right", va="bottom")
        ax.text(x[-1], dmmse[-1] - 0.045, "dMMSE (pool memorized)", fontsize=8.5, color=INK2, ha="right", va="top")
        ends = []
        for opt in opts:
            y = [mean([r for r in sub if r["M"] == m and r["opt"] == opt], "mse_model_seen") for m in ms]
            ax.plot(x, y, color=COLOR[opt], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=3)
            ends.append((opt, y[0]))
        pos = _spread([e[1] for e in ends], 0.0115)
        for (opt, _), p in zip(ends, pos):
            ax.text(x[0] - 0.12, p, LABEL[opt], fontsize=9, color=INK, ha="right", va="center")
        lo = min(dmmse) - 0.028
        mark = lo + 0.008
        for opt in opts:
            t = th[opt]
            if t["star"] is not None:
                ax.plot(t["star"], mark, marker="^", ms=8, color=COLOR[opt], mec=SURFACE, mew=1.2, zorder=4)
            elif t["lo"] is not None:
                ax.plot([t["lo"], t["hi"]], [mark, mark], color=COLOR[opt], lw=5, alpha=0.4, solid_capstyle="round", zorder=4)
        ax.set_ylim(lo, max(ridge) + 0.02)
        ax.set_xlim(x[0] - 1.15, x[-1] + 0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([_pow(v) if abs(v - round(v)) < 0.01 else "" for v in x], fontsize=8.5)
        ax.set_title(f"batch {batch}", loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("pool size M", fontsize=9, color=INK2)
    axes[0][0].set_ylabel("error on training-pool tasks", fontsize=9, color=INK2)
    handles = [Line2D([], [], color=COLOR[o], lw=2, marker="o", ms=5, mec=SURFACE, label=LABEL[o]) for o in opts]
    handles += [Line2D([], [], color=INK2, lw=1.2, label="ridge"), Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label="dMMSE"),
                Line2D([], [], marker="^", ls="", ms=7, color=INK2, label="M* (bar: interval when not pinned down)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    path = out / "memorization_vs_pool.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="summary.csv files, or folders searched for them")
    p.add_argument("--out", type=Path, help="output folder (default: 'aggregate' next to the first path)")
    a = p.parse_args(argv)
    rows = load_rows(a.paths)
    if not rows:
        print("no summary rows found")
        return 1
    res = analyze(rows)
    first = Path(a.paths[0])
    out = a.out or ((first if first.is_dir() else first.parent) / "aggregate")
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_summary(res, rows, out), fig_training(res, rows, out), fig_memorization(res, rows, out)):
        print(f"figure: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
