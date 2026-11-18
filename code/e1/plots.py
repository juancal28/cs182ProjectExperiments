"""Figures for the aggregator: one panel per optimizer, thin lines per seed, bold seed mean."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e1.config import GRID_OPT

# Fixed colour per optimizer (categorical slots 1-3 of the default palette), never by rank.
COLOR = {"adamw": "#2a78d6", "muon": "#eb6834", "sgd": "#1baf7a"}
LABEL = {"adamw": "AdamW", "muon": "Muon", "sgd": "SGD"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _panel(ax, title, per_seed, mean_xy, star, above, below):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    for seed, curve in sorted(per_seed.items()):
        ms = sorted(curve)
        ax.plot([math.log2(m) for m in ms], [curve[m] for m in ms], color=title[1], lw=1, alpha=0.35, zorder=2)
    x, y = mean_xy
    ax.plot(x, y, color=title[1], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=3)
    if star is not None:
        ax.axvline(star, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    crossing = "no zero crossing" if star is None else f"crosses zero at 2^{star:.2f}"
    ax.set_title(f"{title[0]}: {crossing}", loc="left", fontsize=10.5, color=INK, pad=8)
    ax.text(0.99, 0.97, above, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    ax.text(0.99, 0.03, below, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.margins(y=0.22)                      # keep the corner notes clear of the lines
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel("pool size, log2 M", fontsize=9, color=INK2)


def _figure(res, out: Path, name, suptitle, subtitle, ylabel, get_seed, get_mean, get_star, above, below) -> Path:
    opts = [o for o in GRID_OPT if o in res["curves"]]
    fig, axes = plt.subplots(1, len(opts), figsize=(max(4.4 * len(opts), 8.8), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    stars = {t["opt"]: t for t in res["thresholds"]}
    for ax, opt in zip(axes[0], opts):
        n = len(get_seed(opt))
        _panel(ax, (f"{LABEL[opt]}, {n} seed{'s' if n != 1 else ''}", COLOR[opt]),
               get_seed(opt), get_mean(opt), get_star(stars[opt]), above, below)
    axes[0][0].set_ylabel(ylabel, fontsize=9, color=INK2)
    scale, steps = res["scale"]
    fig.suptitle(suptitle, x=0.012, y=0.98, ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.012, 0.915, f"{subtitle}. {scale.capitalize()} scale, {steps:,} steps per run.",
             ha="left", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path = out / name
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def save_figures(res, out: Path) -> list[Path]:
    num = lambda v: None if v == "" else float(v)
    c = res["curves"]
    return [
        _figure(res, out, "crossover.png", "Crossover statistic s(M)",
                "Distance to ridge on new tasks, batch 512 minus batch 256",
                "s(M)", lambda o: c[o]["s"], lambda o: c[o]["s_mean"], lambda t: num(t["log2_Mstar"]),
                "larger batch moves away from ridge", "larger batch moves toward ridge"),
        _figure(res, out, "readout_b256.png", "Single-run readout at batch 256",
                "Distance to dMMSE minus distance to ridge, on seen tasks",
                "gap to dMMSE - gap to ridge", lambda o: c[o]["readout"][256], lambda o: c[o]["readout_mean"][256],
                lambda t: num(t["readout_log2_b256"]), "closer to ridge", "closer to dMMSE"),
    ]
