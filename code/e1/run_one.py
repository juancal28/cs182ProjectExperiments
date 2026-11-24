"""Train a single run.

    python -m e1.run_one --seed 7 --M 2896 --opt muon --batch 256
    python -m e1.run_one --seed 7 --M inf --opt sgd --batch 256 --lr 0.05 --steps 20000 --tag pilot
"""
import argparse
from pathlib import Path

from e1 import tracking
from e1.config import make_settings
from e1.train import run_one

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "e1"


def seed_dir(out: Path, scale: str, seed: int, tag: str = "") -> Path:
    return Path(out) / (f"{scale}_{tag}" if tag else scale) / f"seed_{seed:04d}"


def parse_m(text: str):
    return None if text.lower() == "inf" else int(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--M", type=parse_m, required=True, help="pool size, or 'inf' for a fresh task per sequence")
    p.add_argument("--opt", choices=["adamw", "muon", "sgd"], required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--scale", choices=["full", "prelim", "smoke"], default="full")
    p.add_argument("--steps", type=int, help="override the scale's number of steps")
    p.add_argument("--lr", type=float, help="override the peak learning rate")
    p.add_argument("--lr-aux", type=float, help="override the AdamW-part learning rate (muon only)")
    p.add_argument("--tag", default="", help="suffix for the output folder, so overrides never mix with grid runs")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default=None)
    tracking.add_arguments(p)
    a = p.parse_args(argv)
    if (a.steps or a.lr or a.lr_aux) and not a.tag:
        p.error("--steps/--lr/--lr-aux change the experiment; give --tag so results go to their own folder")
    s = make_settings(a.seed, a.M, a.opt, a.batch, a.scale, steps=a.steps, lr=a.lr, lr_aux=a.lr_aux)
    out = seed_dir(a.out, a.scale, a.seed, a.tag)
    if a.lr or a.lr_aux:   # several learning rates of the same cell must not share a folder
        out = out / (f"lr{s.lr:g}" + (f"_aux{s.lr_aux:g}" if s.lr_aux else ""))
    try:
        target = tracking.resolve(a.wandb, a.scale, a.wandb_entity, a.wandb_project, a.tag)
    except RuntimeError as e:
        p.error(str(e))
    summary = run_one(s, out, a.device, wandb_target=target)
    print({k: summary[k] for k in summary if k.startswith(("gap_", "mse_", "t_mem", "wall"))})


if __name__ == "__main__":
    main()
