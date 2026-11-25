"""Off-distribution probes (the proposal's E4) on the final weights of finished runs.

    python -m e1.probes ../results/e1/full            # every finished run found there
    python -m e1.probes ../results/e1/full --M 32768  # only these pool sizes

Evaluation only; nothing is trained. Writes <folder>/aggregate/probes.csv with one row per
run and probe setting. Prompts follow Garg et al. (2022), with the label noise of training:

    query   only the final input x_K is multiplied by c (in-context examples untouched)
    input   every input is multiplied by c
    skew    inputs drawn from N(0, Sigma): random eigenbasis, i-th eigenvalue proportional
            to 1/i^2, scaled so that the expected squared norm equals that of training

Tasks are new (w ~ N(0, I)), so ridge with lambda = sigma^2 is the Bayes predictor on every
shifted prompt, and errors are reported next to ridge's on the same prompts. `last` columns
are the squared error at the final position (K - 1 in-context examples); `all` columns
average over the K positions. Errors are against the noiseless target and divided by D.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from e1.baselines import ridge_preds
from e1.config import Settings
from e1.model import ICLTransformer
from e1.seeding import make_gen

QUERY_SCALES = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0]
INPUT_SCALES = [1 / 3, 1 / 2, 1.0, 2.0, 3.0]
N_PROMPTS = 8192
FIELDS = ["seed", "opt", "M", "batch", "probe", "value", "mse_model_last", "mse_ridge_last",
          "gap_ridge_last", "mse_model_all", "mse_ridge_all", "gap_ridge_all"]


def skew_transform(seed: int, d: int) -> torch.Tensor:
    """T with x = z @ T ~ N(0, Sigma): eigenvalues of Sigma proportional to 1/i^2, trace(Sigma) = d."""
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=make_gen(seed, "probe", "skew_basis")))
    eig = 1.0 / torch.arange(1, d + 1, dtype=torch.float32) ** 2
    eig = eig * d / eig.sum()
    return q @ torch.diag(eig.sqrt()) @ q.T


def probe_prompts(s: Settings, probe: str, value: float, n: int = N_PROMPTS):
    """xs [n, K, D], noisy ys [n, K], noiseless y_true [n, K] for one probe setting.

    The base draws depend only on the seed, so every run of a seed sees the same prompts and
    the scale probes are the same prompts rescaled.
    """
    gen = make_gen(s.seed, "probe", "base")
    xs = torch.randn(n, s.k, s.d, generator=gen)
    noise = torch.randn(n, s.k, generator=gen) * s.noise_std
    w = torch.randn(n, s.d, generator=gen)
    if probe == "query":
        xs = xs.clone()
        xs[:, -1] *= value
    elif probe == "input":
        xs = xs * value
    elif probe == "skew":
        xs = xs @ skew_transform(s.seed, s.d)
    else:
        raise ValueError(probe)
    y_true = (xs * w.unsqueeze(1)).sum(-1)
    return xs, y_true + noise, y_true


def settings_list():
    return ([("query", c) for c in QUERY_SCALES] + [("input", c) for c in INPUT_SCALES] + [("skew", 1.0)])


@torch.no_grad()
def probe_run(run_dir: Path, device) -> list[dict]:
    """All probe rows for one finished run folder."""
    s = Settings(**json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"])
    model = ICLTransformer(s.d, s.k, s.n_layer, s.n_embd, s.n_head)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    rows = []
    for probe, value in settings_list():
        xs, ys, y_true = (t.to(device) for t in probe_prompts(s, probe, value))
        pred, ridge = model(xs, ys), ridge_preds(xs, ys)

        def mse(a, b, last):
            e = (a - b) ** 2
            return (e[:, -1] if last else e).mean().item() / s.d

        rows.append({
            "seed": s.seed, "opt": s.opt, "M": "inf" if s.M is None else s.M, "batch": s.batch,
            "probe": probe, "value": value,
            "mse_model_last": mse(pred, y_true, True), "mse_ridge_last": mse(ridge, y_true, True),
            "gap_ridge_last": mse(pred, ridge, True),
            "mse_model_all": mse(pred, y_true, False), "mse_ridge_all": mse(ridge, y_true, False),
            "gap_ridge_all": mse(pred, ridge, False),
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("folder", help="results folder holding seed_*/<run>/model.pt")
    p.add_argument("--M", nargs="*", type=int, default=None, help="only these pool sizes")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(a.folder)
    rows = []
    for run_dir in sorted(root.glob("seed_*/*")):
        if not (run_dir / "final.json").exists() or not (run_dir / "model.pt").exists():
            continue
        m = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]["M"]
        if a.M and m not in a.M:
            continue
        rows += probe_run(run_dir, device)
    if not rows:
        print("no finished runs found", file=sys.stderr)
        return 1
    out = root / "aggregate" / "probes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} rows from {len(rows) // len(settings_list())} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
