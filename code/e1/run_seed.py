"""Run the E1 grid for one or more seeds, skipping runs that are already finished.

    python -m e1.run_seed --seed 7                     # 54 runs at full length
    python -m e1.run_seed --seed 0 1 2 3               # several seeds, one after the other
    python -m e1.run_seed --seed 7 --M 600 3000 inf    # your own pool sizes instead of the default nine
    python -m e1.run_seed --seed 7 --scale smoke       # minutes-long check of the whole pipeline
    python -m e1.run_seed --seed 7 --opt adamw         # filters: --opt, --batch
    python -m e1.run_seed --seed 7 --workers 4         # runs in parallel per GPU (or in total on CPU)
    python -m e1.run_seed --seed 7 --no-wandb          # train without logging to Weights & Biases

Stopping it (Ctrl+C, reboot, spot interruption) loses at most the steps since the last
checkpoint; run the same command again to continue.
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

from e1 import tracking
from e1.config import GRID_OPT, MAX_POOL, grid, run_name
from e1.run_one import DEFAULT_OUT, parse_m, seed_dir

SUMMARY_COLUMNS = [
    "seed", "opt", "M", "log2_M", "batch", "lr", "lr_aux", "tuned", "steps", "scale",
    "gap_ridge_new", "gap_dmmse_new", "gap_ridge_seen", "gap_dmmse_seen",
    "mse_model_new", "mse_model_seen", "mse_ridge_new", "mse_dmmse_new", "mse_ridge_seen", "mse_dmmse_seen",
    "t_mem", "settings_hash", "git_commit", "torch_version", "device", "wall_seconds",
]


def write_summary(sdir: Path) -> int:
    """Rebuild summary.csv from every final.json under the seed folder. Returns the row count."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sdir.glob("*/final.json"))]
    rows.sort(key=lambda r: (GRID_OPT.index(r["opt"]), float("inf") if r["M"] == "inf" else r["M"], r["batch"]))
    tmp = sdir / "summary.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(sdir / "summary.csv")
    return len(rows)


def _worker(device: str, threads: int, todo, done, wandb_target):
    import torch
    from e1.train import run_one
    torch.set_num_threads(threads)
    while True:
        item = todo.get()
        if item is None:
            return
        s, sdir = item
        label = f"seed {s.seed} {run_name(s)}"
        try:
            run_one(s, sdir, device, verbose=False, wandb_target=wandb_target)
            done.put((label, sdir, None))
        except Exception:
            done.put((label, sdir, traceback.format_exc()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int, required=True,
                   help="one or more seeds; each gets its own results folder")
    p.add_argument("--scale", choices=["full", "prelim", "smoke"], default="full")
    p.add_argument("--opt", nargs="+", choices=GRID_OPT)
    p.add_argument("--batch", nargs="+", type=int)
    p.add_argument("--M", nargs="+", type=parse_m,
                   help=f"pool sizes to run instead of the default grid: any of 1..{MAX_POOL}, "
                        "or 'inf' for a fresh task per sequence")
    p.add_argument("--workers", type=int, default=1, help="parallel runs per GPU, or in total on CPU")
    p.add_argument("--device", default=None, help="'cpu' forces CPU; default uses every GPU if there is one")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--allow-gpu-sharing", action="store_true",
                   help="permit --workers > 1 per GPU on Windows (it crashed the graphics driver on an RTX 4070 Ti)")
    tracking.add_arguments(p)
    a = p.parse_args(argv)

    import torch
    if (a.workers > 1 and a.device != "cpu" and torch.cuda.is_available()
            and sys.platform == "win32" and not a.allow_gpu_sharing):
        p.error("several runs sharing one GPU crashed Windows during testing (bugcheck 0x119). "
                "Use --workers 1 here, or pass --allow-gpu-sharing to try anyway.")
    seeds = list(dict.fromkeys(a.seeds))
    try:
        runs = [s for seed in seeds for s in grid(seed, a.scale, a.opt, a.batch, a.M)]
        target = tracking.resolve(a.wandb, a.scale, a.wandb_entity, a.wandb_project)
    except (ValueError, RuntimeError) as e:
        p.error(str(e))
    sdirs = {seed: seed_dir(a.out, a.scale, seed) for seed in seeds}
    for sdir in sdirs.values():
        sdir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in runs if not (sdirs[s.seed] / run_name(s) / "final.json").exists()]
    n_gpu = 0 if a.device == "cpu" else torch.cuda.device_count()
    devices = [f"cuda:{g}" for g in range(n_gpu) for _ in range(a.workers)] or ["cpu"] * a.workers
    threads = 1 if n_gpu else max(1, (torch.get_num_threads() or 1) // len(devices))
    print(f"seed(s) {' '.join(map(str, seeds))}, scale {a.scale}: {len(runs)} runs selected, "
          f"{len(runs) - len(pending)} already done, {len(pending)} to run on {len(devices)} worker(s) "
          f"[{devices[0]}{' ...' if len(devices) > 1 else ''}]", flush=True)
    for sdir in sdirs.values():
        print(f"results: {sdir}", flush=True)
    print(f"W&B: {target.entity}/{target.project}" if target else "W&B: off", flush=True)

    failures = []
    if pending:
        ctx = mp.get_context("spawn")
        todo, done = ctx.Queue(), ctx.Queue()
        for s in pending:
            todo.put((s, str(sdirs[s.seed])))
        procs = []
        for dev in devices[: len(pending)]:
            todo.put(None)
            proc = ctx.Process(target=_worker, args=(dev, threads, todo, done, target), daemon=True)
            proc.start()
            procs.append(proc)
        t0, finished = time.time(), 0
        while finished < len(pending):
            try:
                label, sdir, err = done.get(timeout=30)
            except Exception:
                if not any(pr.is_alive() for pr in procs):
                    failures.append(("worker", "all workers exited before finishing; rerun to continue"))
                    break
                continue
            finished += 1
            if err:
                failures.append((label, err))
            write_summary(Path(sdir))
            print(f"[{finished}/{len(pending)}] {'FAILED ' if err else ''}{label}  "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        for proc in procs:
            proc.join(timeout=10)

    for seed, sdir in sdirs.items():
        n = write_summary(sdir)
        selected = sum(s.seed == seed for s in runs)
        print(f"seed {seed}: summary.csv has {n} rows ({selected} runs selected): {sdir / 'summary.csv'}", flush=True)
    for label, err in failures:
        print(f"\nFAILED {label}:\n{err}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
