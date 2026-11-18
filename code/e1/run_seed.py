"""Run the whole E1 grid for one seed, skipping runs that are already finished.

    python -m e1.run_seed --seed 7                     # 54 runs at full length
    python -m e1.run_seed --seed 7 --scale smoke       # minutes-long check of the whole pipeline
    python -m e1.run_seed --seed 7 --opt adamw         # filters: --opt, --batch, --M
    python -m e1.run_seed --seed 7 --workers 4         # runs in parallel per GPU (or in total on CPU)

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

from e1.config import GRID_OPT, grid, run_name
from e1.run_one import DEFAULT_OUT, seed_dir

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


def _worker(device: str, threads: int, todo, done, sdir: str):
    import torch
    from e1.train import run_one
    torch.set_num_threads(threads)
    while True:
        s = todo.get()
        if s is None:
            return
        try:
            run_one(s, sdir, device, verbose=False)
            done.put((run_name(s), None))
        except Exception:
            done.put((run_name(s), traceback.format_exc()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--scale", choices=["full", "prelim", "smoke"], default="full")
    p.add_argument("--opt", nargs="+", choices=GRID_OPT)
    p.add_argument("--batch", nargs="+", type=int)
    p.add_argument("--M", nargs="+", type=int)
    p.add_argument("--workers", type=int, default=1, help="parallel runs per GPU, or in total on CPU")
    p.add_argument("--device", default=None, help="'cpu' forces CPU; default uses every GPU if there is one")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--allow-gpu-sharing", action="store_true",
                   help="permit --workers > 1 per GPU on Windows (it crashed the graphics driver on an RTX 4070 Ti)")
    a = p.parse_args(argv)

    import torch
    if (a.workers > 1 and a.device != "cpu" and torch.cuda.is_available()
            and sys.platform == "win32" and not a.allow_gpu_sharing):
        p.error("several runs sharing one GPU crashed Windows during testing (bugcheck 0x119). "
                "Use --workers 1 here, or pass --allow-gpu-sharing to try anyway.")
    sdir = seed_dir(a.out, a.scale, a.seed)
    sdir.mkdir(parents=True, exist_ok=True)
    runs = grid(a.seed, a.scale, a.opt, a.batch, a.M)
    pending = [s for s in runs if not (sdir / run_name(s) / "final.json").exists()]
    n_gpu = 0 if a.device == "cpu" else torch.cuda.device_count()
    devices = [f"cuda:{g}" for g in range(n_gpu) for _ in range(a.workers)] or ["cpu"] * a.workers
    threads = 1 if n_gpu else max(1, (torch.get_num_threads() or 1) // len(devices))
    print(f"seed {a.seed}, scale {a.scale}: {len(runs)} runs selected, {len(runs) - len(pending)} already done, "
          f"{len(pending)} to run on {len(devices)} worker(s) [{devices[0]}{' ...' if len(devices) > 1 else ''}]", flush=True)
    print(f"results: {sdir}", flush=True)

    failures = []
    if pending:
        ctx = mp.get_context("spawn")
        todo, done = ctx.Queue(), ctx.Queue()
        for s in pending:
            todo.put(s)
        procs = []
        for dev in devices[: len(pending)]:
            todo.put(None)
            proc = ctx.Process(target=_worker, args=(dev, threads, todo, done, str(sdir)), daemon=True)
            proc.start()
            procs.append(proc)
        t0, finished = time.time(), 0
        while finished < len(pending):
            try:
                name, err = done.get(timeout=30)
            except Exception:
                if not any(pr.is_alive() for pr in procs):
                    failures.append(("worker", "all workers exited before finishing; rerun to continue"))
                    break
                continue
            finished += 1
            if err:
                failures.append((name, err))
            write_summary(sdir)
            print(f"[{finished}/{len(pending)}] {'FAILED ' if err else ''}{name}  "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        for proc in procs:
            proc.join(timeout=10)

    n = write_summary(sdir)
    print(f"summary.csv has {n} of {len(runs)} selected runs: {sdir / 'summary.csv'}", flush=True)
    for name, err in failures:
        print(f"\nFAILED {name}:\n{err}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
