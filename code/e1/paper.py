"""Numbers, tables and figures for the paper draft, all computed from the results folder.

    python -m e1.probes ../results/e1/full          # once: off-distribution probes (needs the weights)
    python -m e1.paper ../results/e1/full ../paper  # writes ../paper/generated and ../paper/figures

Outputs:
    generated/numbers.json   every number the draft quotes, with the per-run values behind it
    generated/numbers.tex    the same numbers as LaTeX macros
    generated/tab_*.tex      table bodies
    figures/fig_*.pdf        figures

Rerun after adding seeds; the draft picks the new values up when it is compiled again.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used as matplotlib.ticker)
import numpy as np

from e1 import aggregate, dynamics
from e1.config import GRID_OPT
from e1.plots import AXIS, COLOR, GRID, INK, INK2, LABEL, MUTED

LEVEL = 0.005        # distance to ridge on new tasks used for the level-crossing threshold
REBOUND = 0.10       # a run "left ridge" if the distance grew back by more than this fraction
ONSET = 0.05         # memorized fraction that defines the onset of memorization
MACRO = {"adamw": "Adamw", "muon": "Muon", "sgd": "Sgd"}
BATCHES = (256, 512)


# ---------------------------------------------------------------- numbers

def gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def compute(root: Path) -> dict:
    runs = dynamics.load_runs(root)
    res = aggregate.analyze(aggregate.load_rows([root]))
    seeds = sorted({k[0] for k in runs})
    ms = sorted({k[2] for k in runs})
    opts = [o for o in GRID_OPT if any(k[1] == o for k in runs)]
    top = ms[-1]
    log2m = [math.log2(m) for m in ms]
    out = {"seeds": seeds, "Ms": ms, "opts": opts, "n_runs": len(runs), "steps": res["scale"][1],
           "level": LEVEL, "rebound": REBOUND, "onset": ONSET, "opt": {}}

    def phi(r):
        return dynamics.memorized_fraction(r["mse_model_seen"], r["final"]["mse_ridge_seen"],
                                           r["final"]["mse_dmmse_seen"])

    th = {t["opt"]: t for t in res["thresholds"]}
    per_seed = {(p["opt"], p["seed"]): p for p in res["per_seed"]}
    num = lambda v: None if v in ("", None) else float(v)
    for o in opts:
        s = res["curves"][o]["s"]
        e = {"tuned": th[o]["tuned"], "lr": runs[(seeds[0], o, top, 256)]["final"]["lr"],
             "lr_aux": runs[(seeds[0], o, top, 256)]["final"]["lr_aux"] or None,
             "mstar": num(th[o]["log2_Mstar"]), "mstar_status": th[o]["status"],
             "mstar_lo": num(th[o]["ci95_lo"]), "mstar_hi": num(th[o]["ci95_hi"]),
             "mstar_seed": {sd: num(per_seed[(o, sd)]["log2_Mstar"]) for sd in seeds},
             "mstar_seed_status": {sd: per_seed[(o, sd)]["status"] for sd in seeds},
             "s": {sd: [s[sd][m] for m in ms] for sd in seeds},
             "s_mean": list(res["curves"][o]["s_mean"][1])}
        hi = [m for m in ms if m >= 4096]
        lo = [m for m in ms if m <= 2048]
        e["s_neg_high"] = sum(s[sd][m] < 0 for sd in seeds for m in hi)
        e["s_n_high"] = len(hi) * len(seeds)
        e["s_pos_low"] = sum(s[sd][m] > 0 for sd in seeds for m in lo)
        e["s_n_low"] = len(lo) * len(seeds)
        for b in BATCHES:
            fin = lambda sd, m, key: runs[(sd, o, m, b)]["final"][key]
            gap = {sd: [fin(sd, m, "gap_ridge_new") for m in ms] for sd in seeds}
            ph = {sd: [float(phi(runs[(sd, o, m, b)])[-1]) for m in ms] for sd in seeds}
            reb = {sd: [dynamics.rebound(runs[(sd, o, m, b)]["step"], runs[(sd, o, m, b)]["gap_ridge_new"])
                        for m in ms] for sd in seeds}
            left = {sd: max([m for m, r in zip(ms, reb[sd]) if r["rise"] > REBOUND], default=None) for sd in seeds}
            t_reg = {sd: [dynamics.first_step_within(runs[(sd, o, m, b)]["step"], runs[(sd, o, m, b)]["mse_model_new"],
                                                     fin(sd, m, "mse_ridge_new")) for m in ms] for sd in seeds}
            t_on = {sd: [dynamics.onset_step(runs[(sd, o, m, b)]["step"], phi(runs[(sd, o, m, b)]), ONSET)
                         for m in ms] for sd in seeds}
            fit_ms = []
            for i, m in enumerate(ms):                      # smallest pools first, while every seed reached onset
                if any(t_on[sd][i] is None for sd in seeds):
                    break
                fit_ms.append(i)
            fit = dynamics.power_law([ms[i] for i in fit_ms for _ in seeds],
                                     [t_on[sd][i] for i in fit_ms for sd in seeds])
            e[b] = {"gap": gap, "gap_mean": [float(np.mean([gap[sd][i] for sd in seeds])) for i in range(len(ms))],
                    "phi": ph, "phi_mean": [float(np.mean([ph[sd][i] for sd in seeds])) for i in range(len(ms))],
                    "rise": {sd: [r["rise"] for r in reb[sd]] for sd in seeds},
                    "step_min": {sd: [r["step_min"] for r in reb[sd]] for sd in seeds},
                    "left_ridge_max_M": left, "step_min_small": [reb[sd][0]["step_min"] for sd in seeds],
                    "level": {sd: dynamics.level_crossing(log2m, gap[sd], LEVEL) for sd in seeds},
                    "t_reg": t_reg, "t_reg_top": [t_reg[sd][-1] for sd in seeds],
                    "t_on": t_on, "t_on_small": [t_on[sd][0] for sd in seeds],
                    "fit": fit, "fit_Ms": [ms[i] for i in fit_ms],
                    "t_mem_reached": sum(fin(sd, m, "t_mem") not in ("", None) for sd in seeds for m in ms),
                    "id_excess_top": [fin(sd, top, "mse_model_new") / fin(sd, top, "mse_ridge_new") - 1 for sd in seeds]}
        out["opt"][o] = e

    # Two-clock comparison against AdamW: log2 ratios of the regression clock (largest pool) and of
    # the memorization onset (smallest pool), and the shift they imply if the exponent is shared.
    out["clock"] = {}
    for b in BATCHES:
        ref = out["opt"]["adamw"][b]
        nu_src = out["opt"]["muon"][b]["fit"]
        cmp_ = {}
        for o in opts:
            e = out["opt"][o][b]
            if None in e["t_reg_top"] or None in e["t_on_small"] or None in ref["t_on_small"]:
                continue
            d_reg = math.log2(gmean(e["t_reg_top"]) / gmean(ref["t_reg_top"]))
            d_mem = math.log2(gmean(e["t_on_small"]) / gmean(ref["t_on_small"]))
            cmp_[o] = {"t_reg": gmean(e["t_reg_top"]), "t_on": gmean(e["t_on_small"]), "d_reg": d_reg, "d_mem": d_mem,
                       "pred_shift": None if (nu_src is None or o == "adamw") else (d_reg - d_mem) / nu_src["nu"]}
        out["clock"][b] = cmp_
    out["run_hours"] = sum(r["final"]["wall_seconds"] for r in runs.values()) / 3600
    out["shift"] = {o: {"crossover_seed": {sd: (None if None in (out["opt"][o]["mstar_seed"][sd], out["opt"]["adamw"]["mstar_seed"][sd])
                                                else out["opt"][o]["mstar_seed"][sd] - out["opt"]["adamw"]["mstar_seed"][sd]) for sd in seeds},
                        "crossover_pooled": (None if None in (out["opt"][o]["mstar"], out["opt"]["adamw"]["mstar"])
                                             else out["opt"][o]["mstar"] - out["opt"]["adamw"]["mstar"]),
                        "level_seed": {sd: (None if None in (out["opt"][o][256]["level"][sd], out["opt"]["adamw"][256]["level"][sd])
                                            else out["opt"][o][256]["level"][sd] - out["opt"]["adamw"][256]["level"][sd]) for sd in seeds}}
                    for o in opts if o != "adamw"}

    probes_csv = root / "aggregate" / "probes.csv"
    out["probes"] = None
    if probes_csv.exists():
        acc = defaultdict(list)
        with open(probes_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["M"] != str(top):
                    continue
                acc[(r["probe"], float(r["value"]), r["opt"])].append(
                    {"seed": int(r["seed"]), "batch": int(r["batch"]),
                     "ratio": float(r["mse_model_last"]) / float(r["mse_ridge_last"]),
                     "mse_ridge_last": float(r["mse_ridge_last"])})
        out["probes"] = {}
        for (probe, value, o), v in sorted(acc.items()):
            ratios = [x["ratio"] for x in v]
            out["probes"].setdefault(probe, {}).setdefault(f"{value:.6g}", {})[o] = {
                "value": value, "runs": v, "gmean": gmean(ratios), "min": min(ratios), "max": max(ratios)}
    return out


# ---------------------------------------------------------------- LaTeX

def _f(v, nd=2, none="--"):
    return none if v is None else f"{v:.{nd}f}"


def _pow(m: int) -> str:
    x = math.log2(m)
    half = round(x * 2) / 2
    return f"2^{{{half:g}}}" if abs(x - half) < 0.01 else f"2^{{{x:.2f}}}"


def _k(step) -> str:
    return "--" if step is None else f"{step / 1000:.0f}"


def write_macros(n: dict, path: Path) -> None:
    sa, sb = n["seeds"][0], n["seeds"][-1]
    m = {"NRuns": str(n["n_runs"]), "NSeeds": str(len(n["seeds"])), "SeedA": str(sa), "SeedB": str(sb),
         "LevelTau": f"{n['level']:g}", "ReboundPct": f"{100 * n['rebound']:.0f}", "OnsetPct": f"{100 * n['onset']:.0f}"}
    for o in n["opts"]:
        e, k = n["opt"][o], MACRO[o]
        m[f"Mstar{k}"] = _f(e["mstar"], none="not bracketed")
        m[f"Mstar{k}Lo"], m[f"Mstar{k}Hi"] = _f(e["mstar_lo"]), _f(e["mstar_hi"])
        m[f"Mstar{k}SeedA"], m[f"Mstar{k}SeedB"] = _f(e["mstar_seed"][sa]), _f(e["mstar_seed"][sb])
        m[f"SNegHigh{k}"], m[f"SNHigh{k}"] = str(e["s_neg_high"]), str(e["s_n_high"])
        m[f"SPosLow{k}"], m[f"SNLow{k}"] = str(e["s_pos_low"]), str(e["s_n_low"])
        b = e[256]
        m[f"Level{k}SeedA"], m[f"Level{k}SeedB"] = _f(b["level"][sa]), _f(b["level"][sb])
        m[f"GapSmall{k}"] = f"{b['gap_mean'][0]:.3f}"
        m[f"GapTop{k}"] = f"{1e3 * b['gap_mean'][-1]:.1f}"
        m[f"PhiSmall{k}"] = f"{100 * b['phi_mean'][0]:.0f}"
        m[f"PhiSmallLarge{k}"] = f"{100 * e[512]['phi_mean'][0]:.0f}"
        m[f"StepMin{k}"] = "/".join(_k(t) for t in b["step_min_small"])
        m[f"Treg{k}"] = "/".join(_k(t) for t in b["t_reg_top"])
        m[f"Ton{k}"] = "/".join(_k(t) for t in b["t_on_small"])
        m[f"IdExcess{k}"] = "--".join(f"{100 * v:.1f}" for v in (min(e[bb]["id_excess_top"][i] for bb in BATCHES for i in range(len(n['seeds']))),
                                                                  max(e[bb]["id_excess_top"][i] for bb in BATCHES for i in range(len(n['seeds'])))))
        for bb, tag in ((256, ""), (512, "Large")):
            fit = e[bb]["fit"]
            m[f"Nu{k}{tag}"] = _f(None if fit is None else fit["nu"])
        if o != "adamw":
            c = n["clock"][256].get(o)
            if c:
                m[f"RegSpeed{k}"] = f"{2 ** -c['d_reg']:.1f}" if c["d_reg"] < 0 else f"{2 ** c['d_reg']:.1f}"
                m[f"MemSpeed{k}"] = f"{2 ** -c['d_mem']:.1f}" if c["d_mem"] < 0 else f"{2 ** c['d_mem']:.1f}"
                m[f"PredShift{k}"] = "--" if c["pred_shift"] is None else f"{c['pred_shift']:+.1f}"
            c2 = n["clock"][512].get(o)
            if c2:
                m[f"PredShiftLarge{k}"] = "--" if c2["pred_shift"] is None else f"{c2['pred_shift']:+.1f}"
            sh = n["shift"][o]
            m[f"ShiftPooled{k}"] = "--" if sh["crossover_pooled"] is None else f"{sh['crossover_pooled']:+.2f}"
            m[f"ShiftSeedA{k}"] = "--" if sh["crossover_seed"][sa] is None else f"{sh['crossover_seed'][sa]:+.2f}"
            m[f"ShiftSeedB{k}"] = "--" if sh["crossover_seed"][sb] is None else f"{sh['crossover_seed'][sb]:+.2f}"
            m[f"LevelShiftSeedA{k}"] = "--" if sh["level_seed"][sa] is None else f"{sh['level_seed'][sa]:+.2f}"
            m[f"LevelShiftSeedB{k}"] = "--" if sh["level_seed"][sb] is None else f"{sh['level_seed'][sb]:+.2f}"
    if n["probes"]:
        rng = lambda c: f"{c['min']:.2f}--{c['max']:.2f}"
        for o in n["opts"]:
            k = MACRO[o]
            m[f"ProbeId{k}"] = f"{n['probes']['query']['1'][o]['gmean']:.2f}"
            m[f"ProbeQuerySixteen{k}"] = f"{n['probes']['query']['16'][o]['gmean']:.0f}"
            m[f"ProbeQueryFour{k}"] = rng(n["probes"]["query"]["4"][o])
            m[f"ProbeInputThird{k}"] = rng(n["probes"]["input"][f"{1 / 3:.6g}"][o])
            m[f"ProbeInputThree{k}"] = rng(n["probes"]["input"]["3"][o])
            m[f"ProbeSkew{k}"] = rng(n["probes"]["skew"]["1"][o])
        # A predictor that always outputs 0 has error E[y^2] / D = 1 in these units.
        base = [r["mse_ridge_last"] for o in n["opts"] for r in n["probes"]["query"]["1"][o]["runs"]]
        m["ProbeZero"] = f"{1 / float(np.mean(base)):.0f}"
    m["RunHours"] = f"{n['run_hours']:.0f}"
    m["TmemReached"] = str(sum(n["opt"][o][b]["t_mem_reached"] for o in n["opts"] for b in BATCHES))
    lines = ["% Generated by `python -m e1.paper`. Do not edit; rerun instead."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(n: dict, out: Path) -> None:
    sa, sb = n["seeds"][0], n["seeds"][-1]
    ms = n["Ms"]
    lr_txt = {"reference": "reference", "quick-sweep": "quick sweep", "untuned": "untuned"}

    def star(e, sd):
        mark = "$^\\dagger$" if e["mstar_seed_status"][sd] == "multiple" else ""
        return _f(e["mstar_seed"][sd]) + mark

    rows = []
    for o in n["opts"]:
        e = n["opt"][o]
        b = e[256]
        pooled = (_f(e["mstar"]) if e["mstar"] is not None else "none") + f" ({_f(e['mstar_lo'])}--{_f(e['mstar_hi'])})"
        left = " / ".join("none" if b["left_ridge_max_M"][sd] is None else f"${_pow(b['left_ridge_max_M'][sd])}$" for sd in (sa, sb))
        rows.append(f"{LABEL[o]} & {lr_txt.get(e['tuned'], e['tuned'])} & {star(e, sa)} & {star(e, sb)} & {pooled} & "
                    f"{_f(b['level'][sa])} & {_f(b['level'][sb])} & {left} \\\\")
    (out / "tab_thresholds.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for o in n["opts"]:
        for bb in BATCHES:
            b = n["opt"][o][bb]
            c = n["clock"][bb].get(o)
            fit = b["fit"]
            ratio = lambda d: "1" if o == "adamw" else (f"{2 ** d:.2f}" if c else "--")
            rows.append(
                f"{LABEL[o] if bb == 256 else ''} & {bb} & {' / '.join(_k(t) for t in b['t_reg_top'])} & {ratio(c['d_reg']) if c else '--'} & "
                f"{' / '.join(_k(t) for t in b['t_on_small'])} & {ratio(c['d_mem']) if c else '--'} & "
                f"{100 * b['phi_mean'][0]:.0f} & "
                + ("--" if fit is None else f"{fit['nu']:.2f} ({len(b['fit_Ms'])})") + " \\\\")
        if o != n["opts"][-1]:
            rows.append("\\addlinespace")
    (out / "tab_clocks.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for o in n["opts"]:
        for sd in (sa, sb):
            rows.append(f"{LABEL[o] if sd == sa else ''} & {sd} & " + " & ".join(f"${1e3 * v:+.2f}$" for v in n["opt"][o]["s"][sd]) + " \\\\")
        if o != n["opts"][-1]:
            rows.append("\\addlinespace")
    head = "Optimizer & seed & " + " & ".join(f"${_pow(m)}$" for m in ms) + " \\\\"
    (out / "tab_s.tex").write_text(head + "\n\\midrule\n" + "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for o in n["opts"]:
        for bb in BATCHES:
            rows.append(f"{LABEL[o] if bb == 256 else ''} & {bb} & " + " & ".join(f"{1e3 * v:.2f}" for v in n["opt"][o][bb]["gap_mean"]) + " \\\\")
        if o != n["opts"][-1]:
            rows.append("\\addlinespace")
    head = "Optimizer & batch & " + " & ".join(f"${_pow(m)}$" for m in ms) + " \\\\"
    (out / "tab_gap.tex").write_text(head + "\n\\midrule\n" + "\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for o in n["opts"]:
        for bb in BATCHES:
            rows.append(f"{LABEL[o] if bb == 256 else ''} & {bb} & " + " & ".join(f"${100 * v:+.1f}$" for v in n["opt"][o][bb]["phi_mean"]) + " \\\\")
        if o != n["opts"][-1]:
            rows.append("\\addlinespace")
    (out / "tab_phi.tex").write_text(head + "\n\\midrule\n" + "\n".join(rows) + "\n", encoding="utf-8")

    if n["probes"]:
        def frac(v):
            for d in (2, 3, 4, 8, 16):
                if abs(v * d - 1) < 1e-6:
                    return f"1/{d}"
            return f"{v:g}"
        pick = [("query", v) for v in (1 / 16, 1 / 4, 1.0, 4.0, 16.0)] + [("input", v) for v in (1 / 3, 1 / 2, 2.0, 3.0)] + [("skew", 1.0)]
        name = {"query": "Query scaled by", "input": "All inputs scaled by", "skew": "Skewed covariance"}
        rows, last = [], None
        for probe, v in pick:
            cell = n["probes"][probe][f"{v:.6g}"]
            label = name[probe] + ("" if probe == "skew" else f" ${frac(v)}$")
            if probe == "query" and v == 1.0:
                label = "None (training distribution)"
            if last and last != probe:
                rows.append("\\addlinespace")
            last = probe
            rows.append(label + " & " + " & ".join(
                f"{cell[o]['gmean']:.2f} ({cell[o]['min']:.2f}--{cell[o]['max']:.2f})" for o in n["opts"]) + " \\\\")
        (out / "tab_probes.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- figures

def _rc():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _ax(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)
    ax.minorticks_off()


def _save(fig, path: Path) -> None:
    for ax in fig.axes:                                  # log axes: no labels on minor ticks
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(path)
    plt.close(fig)


def _mticks(ax, ms, every=1.0):
    lo, hi = math.log2(ms[0]), math.log2(ms[-1])
    ticks = [x for x in np.arange(math.ceil(lo), math.floor(hi) + 1) if (x - math.ceil(lo)) % every == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{int(t)}}}$" for t in ticks])
    ax.set_xlabel("pool size $M$")


def fig_crossover(n, out: Path) -> Path:
    ms = n["Ms"]
    x = [math.log2(m) for m in ms]
    sa, sb = n["seeds"][0], n["seeds"][-1]
    fig, axes = plt.subplots(1, len(n["opts"]), figsize=(6.5, 2.2), sharey=True)
    for ax, o in zip(axes, n["opts"]):
        e = n["opt"][o]
        _ax(ax)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        if e["mstar_lo"] is not None:
            ax.axvspan(e["mstar_lo"], e["mstar_hi"], color=COLOR[o], alpha=0.10, lw=0, zorder=0)
        if e["mstar"] is not None:
            ax.axvline(e["mstar"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for sd, mk in ((sa, "o"), (sb, "s")):
            ax.plot(x, [1e3 * v for v in e["s"][sd]], color=COLOR[o], lw=0.7, alpha=0.75, marker=mk, ms=3,
                    mfc="white", mew=0.7, zorder=2, label=f"seed {sd}")
        ax.plot(x, [1e3 * v for v in e["s_mean"]], color=COLOR[o], lw=1.7, zorder=3, label="mean of seeds")
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_yticks([-1, 0, 1, 10])
        ax.set_yticklabels(["$-1$", "0", "1", "10"])
        ax.set_ylim(-4.5, 30)
        note = "no crossing in the mean" if e["mstar"] is None else f"mean crosses zero at $2^{{{e['mstar']:.2f}}}$"
        ax.set_title(f"{LABEL[o]}: {note}", loc="left", color=INK)
        _mticks(ax, ms)
        if o == n["opts"][0]:
            ax.set_ylabel("$s(M) \\times 10^{3}$")
            ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    path = out / "fig_crossover.pdf"
    _save(fig, path)
    return path


def fig_final_state(n, out: Path) -> Path:
    ms = n["Ms"]
    x = [math.log2(m) for m in ms]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4))
    for ax in (a, b):
        _ax(ax)
        _mticks(ax, ms)
    for o in n["opts"]:
        e = n["opt"][o][256]
        for sd in n["seeds"]:
            a.plot(x, e["gap"][sd], color=COLOR[o], lw=0, marker="o", ms=2.2, alpha=0.45, mew=0)
            b.plot(x, [100 * v for v in e["phi"][sd]], color=COLOR[o], lw=0, marker="o", ms=2.2, alpha=0.45, mew=0)
        a.plot(x, e["gap_mean"], color=COLOR[o], lw=1.5, label=LABEL[o])
        b.plot(x, [100 * v for v in e["phi_mean"]], color=COLOR[o], lw=1.5, label=LABEL[o])
    a.set_yscale("log")
    a.axhline(n["level"], color=MUTED, lw=0.7, ls=(0, (1, 2)))
    a.text(x[-1], n["level"] * 1.12, f"$\\tau = {n['level']:g}$", ha="right", va="bottom", fontsize=7, color=INK2)
    a.set_ylabel("distance to ridge, new tasks ($\\Delta_{\\mathrm{ridge}}$)")
    a.set_title("(a) Distance to ridge at the end of training", loc="left", color=INK)
    a.legend(frameon=False, loc="upper right")
    b.axhline(0, color=INK2, lw=0.6)
    b.text(x[-1], 0.8, "ridge", ha="right", va="bottom", fontsize=7, color=INK2)
    b.set_ylabel("memorized fraction $\\varphi$ (%)")
    b.set_title("(b) Progress from ridge toward dMMSE on pool tasks", loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    path = out / "fig_final_state.pdf"
    _save(fig, path)
    return path


def fig_dynamics(n, runs, out: Path) -> Path:
    ms = n["Ms"]
    want = [m for m in (512, 1448, 4096, 32768) if m in ms] or ms[:4]
    fig, axes = plt.subplots(1, len(want), figsize=(6.5, 2.05), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), want):
        _ax(ax)
        for o in n["opts"]:
            curves = [runs[(sd, o, m, 256)] for sd in n["seeds"]]
            steps = curves[0]["step"] / 1000
            mean = np.exp(np.mean([np.log(c["gap_ridge_new"]) for c in curves], axis=0))
            ax.plot(steps[1:], mean[1:], color=COLOR[o], lw=1.1, label=LABEL[o])
        ax.set_yscale("log")
        ax.set_ylim(6e-4, 0.3)
        ax.set_xticks([0, 250, 500])
        ax.set_xlabel("step (thousands)")
        ax.set_title(f"$M = {_pow(m)}$", loc="left", color=INK)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("distance to ridge, new tasks")
    np.atleast_1d(axes)[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=0.6)
    path = out / "fig_dynamics.pdf"
    _save(fig, path)
    return path


def fig_clocks(n, out: Path) -> Path:
    ms = n["Ms"]
    x = [math.log2(m) for m in ms]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for ax in (a, b):
        _ax(ax)
        ax.set_yscale("log")
    for o in n["opts"]:
        e = n["opt"][o][256]
        for sd in n["seeds"]:
            pts = [(xx, t) for xx, t in zip(x, e["t_reg"][sd]) if t is not None]
            a.plot([p[0] for p in pts], [p[1] / 1000 for p in pts], color=COLOR[o], lw=0, marker="o", ms=2.6, alpha=0.5, mew=0)
        mean = [gmean([e["t_reg"][sd][i] for sd in n["seeds"]]) / 1000 if all(e["t_reg"][sd][i] for sd in n["seeds"]) else np.nan
                for i in range(len(ms))]
        a.plot(x, mean, color=COLOR[o], lw=1.4, label=LABEL[o])
    _mticks(a, ms)
    a.set_ylabel("step (thousands)")
    a.set_yticks([10, 20, 50, 100])
    a.set_yticklabels(["10", "20", "50", "100"])
    a.set_title("(a) Regression clock $t_{\\mathrm{reg}}$, batch 256", loc="left", color=INK)
    a.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.4)
    a.set_ylim(7, 330)

    for o in n["opts"]:
        for bb, fill, ls in ((256, COLOR[o], "-"), (512, "white", (0, (3, 2)))):
            e = n["opt"][o][bb]
            for sd in n["seeds"]:
                pts = [(xx, t) for xx, t in zip(x, e["t_on"][sd]) if t is not None]
                b.plot([p[0] for p in pts], [p[1] / 1000 for p in pts], color=COLOR[o], lw=0, marker="o", ms=3.2,
                       mfc=fill, mew=0.8, alpha=0.9)
            if e["fit"]:
                xs = np.array([math.log2(e["fit_Ms"][0]) - 0.15, math.log2(e["fit_Ms"][-1]) + 0.15])
                b.plot(xs, 2 ** (e["fit"]["log2_c"] + e["fit"]["nu"] * xs) / 1000, color=COLOR[o], lw=0.9, ls=ls)
    b.axhline(n["steps"] / 1000, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    b.text(8.75, n["steps"] / 1000 * 1.06, "end of training", ha="left", va="bottom", fontsize=7, color=INK2)
    b.plot([], [], color=INK2, lw=0, marker="o", ms=3.2, mew=0.8, label="batch 256")
    b.plot([], [], color=INK2, lw=0, marker="o", ms=3.2, mfc="white", mew=0.8, label="batch 512")
    b.legend(frameon=False, loc="lower right")
    b.set_xlim(8.7, 12.5)
    b.set_xticks([9, 10, 11, 12])
    b.set_xticklabels([f"$2^{{{t}}}$" for t in (9, 10, 11, 12)])
    b.set_xlabel("pool size $M$")
    b.set_yticks([50, 100, 200, 500])
    b.set_yticklabels(["50", "100", "200", "500"])
    b.set_ylim(30, 760)
    b.set_ylabel("step (thousands)")
    b.set_title("(b) Onset of memorization $t_{\\mathrm{on}}$", loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    path = out / "fig_clocks.pdf"
    _save(fig, path)
    return path


def fig_probes(n, out: Path) -> Path:
    pr = n["probes"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    for ax, probe, title in ((a, "query", "(a) Query scaled"), (b, "input", "(b) All inputs scaled")):
        _ax(ax)
        keys = sorted(pr[probe], key=float)
        xs = [math.log2(float(k)) for k in keys]
        for o in n["opts"]:
            ax.fill_between(xs, [pr[probe][k][o]["min"] for k in keys], [pr[probe][k][o]["max"] for k in keys],
                            color=COLOR[o], alpha=0.13, lw=0)
            ax.plot(xs, [pr[probe][k][o]["gmean"] for k in keys], color=COLOR[o], lw=1.3, marker="o", ms=2.6, mew=0, label=LABEL[o])
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        lab = lambda v: f"{v:g}" if v >= 1 else f"1/{round(1 / v)}"
        ax.set_xticklabels([lab(float(k)) for k in keys])
        ax.set_xlabel("scale factor")
        ax.set_title(title, loc="left", color=INK)
    a.set_ylabel("error relative to ridge")
    a.legend(frameon=False, loc="upper center")
    _ax(c)
    for i, o in enumerate(n["opts"]):
        cell = pr["skew"]["1"][o]
        for j, r in enumerate(sorted(cell["runs"], key=lambda r: (r["batch"], r["seed"]))):
            c.plot(i + (j - 1.5) * 0.09, r["ratio"], color=COLOR[o], lw=0, marker="o", ms=3.4,
                   mfc=COLOR[o] if r["batch"] == 256 else "white", mew=0.8)
        c.plot([i - 0.28, i + 0.28], [cell["gmean"]] * 2, color=COLOR[o], lw=1.4)
    c.axhline(1, color=INK2, lw=0.6)
    c.set_yscale("log")
    c.set_xticks(range(len(n["opts"])))
    c.set_xticklabels([LABEL[o] for o in n["opts"]])
    c.set_xlim(-0.6, len(n["opts"]) - 0.4)
    c.set_title("(c) Skewed covariance", loc="left", color=INK)
    fig.tight_layout(w_pad=1.0)
    path = out / "fig_probes.pdf"
    _save(fig, path)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", help="results folder, for example ../results/e1/full")
    p.add_argument("paper", help="paper folder; generated/ and figures/ are written inside it")
    a = p.parse_args(argv)
    root, paper = Path(a.results), Path(a.paper)
    gen, figs = paper / "generated", paper / "figures"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    n = compute(root)
    (gen / "numbers.json").write_text(json.dumps(n, indent=1, default=float), encoding="utf-8")
    write_macros(n, gen / "numbers.tex")
    write_tables(n, gen)
    _rc()
    runs = dynamics.load_runs(root)
    made = [fig_crossover(n, figs), fig_final_state(n, figs), fig_dynamics(n, runs, figs), fig_clocks(n, figs)]
    if n["probes"]:
        made.append(fig_probes(n, figs))
    else:
        print("no aggregate/probes.csv: run `python -m e1.probes` first for the probe table and figure", file=sys.stderr)
    for path in made:
        print(path)
    print(gen / "numbers.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
