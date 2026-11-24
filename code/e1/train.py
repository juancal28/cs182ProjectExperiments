"""One training run: resumable, logged through training, summarized in final.json."""
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must precede CUDA init

import torch

from e1 import data, tracking
from e1.baselines import dmmse_preds, ridge_preds
from e1.config import Settings, run_name, settings_hash
from e1.model import ICLTransformer, init_weights
from e1.optim import build_optimizers


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _gap(a: torch.Tensor, b: torch.Tensor, d: int) -> float:
    """Mean squared gap over prompts and positions, divided by D (the reference's normalization)."""
    return ((a - b) ** 2).mean().item() / d


class EvalSets:
    """Fixed evaluation prompts with cached ridge and dMMSE predictions."""

    def __init__(self, s: Settings, pool_m: Optional[torch.Tensor], device):
        self.d = s.d
        self.sets = {}
        kinds = ["new"] + (["seen"] if pool_m is not None else [])
        for kind in kinds:
            xs, ys, y_true = (t.to(device) for t in data.eval_set(s.seed, kind, s.n_eval, pool_m))
            entry = {"xs": xs, "ys": ys, "y_true": y_true, "ridge": ridge_preds(xs, ys)}
            if pool_m is not None:
                entry["dmmse"] = dmmse_preds(xs, ys, pool_m.to(device))
            self.sets[kind] = entry

    def baseline_errors(self) -> dict:
        out = {}
        for kind, e in self.sets.items():
            out[f"mse_ridge_{kind}"] = _gap(e["ridge"], e["y_true"], self.d)
            if "dmmse" in e:
                out[f"mse_dmmse_{kind}"] = _gap(e["dmmse"], e["y_true"], self.d)
        return out

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        model.eval()
        out = {}
        for kind, e in self.sets.items():
            preds = model(e["xs"], e["ys"])
            out[f"gap_ridge_{kind}"] = _gap(preds, e["ridge"], self.d)
            if "dmmse" in e:
                out[f"gap_dmmse_{kind}"] = _gap(preds, e["dmmse"], self.d)
            out[f"mse_model_{kind}"] = _gap(preds, e["y_true"], self.d)
        model.train()
        return out


def first_memorized_step(records: list) -> Optional[int]:
    """First logged step (after 0) at which seen-task predictions are closer to dMMSE than to ridge."""
    for r in records:
        if r["step"] > 0 and "gap_dmmse_seen" in r and r["gap_dmmse_seen"] < r["gap_ridge_seen"]:
            return r["step"]
    return None


def run_one(s: Settings, out_root, device=None, stop_after: Optional[int] = None, verbose: bool = True,
            wandb_target: Optional[tracking.WandbTarget] = None):
    """Train one run, resuming from its checkpoint if there is one.

    Returns the summary dict, or None if `stop_after` interrupted it. Results live in
    out_root / run_name(s). A finished run is never retrained. With `wandb_target`, every
    log record also goes to Weights & Biases.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_root) / run_name(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, final_path = out_dir / "config.json", out_dir / "final.json"
    ckpt_path, log_path = out_dir / "checkpoint.pt", out_dir / "log.jsonl"
    h = settings_hash(s)

    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8")).get("settings_hash")
        if old != h:
            raise RuntimeError(f"{out_dir} holds a run with different settings (hash {old}, now {h}). "
                               "Use a different output folder or delete that run.")
    else:
        _atomic_write_text(cfg_path, json.dumps({"settings": s.as_dict(), "settings_hash": h}, indent=2))
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))

    pool_m = None if s.M is None else data.task_pool(s.seed)[: s.M]
    pool_dev = None if pool_m is None else pool_m.to(device)
    evals = EvalSets(s, pool_m, device)

    model = ICLTransformer(s.d, s.k, s.n_layer, s.n_embd, s.n_head)
    init_weights(model, s.seed)
    model.to(device).train()
    opts, scheds = build_optimizers(model, s)

    start, records, elapsed = 0, [], 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ck["settings_hash"] != h:
            raise RuntimeError(f"checkpoint in {out_dir} was written with different settings")
        model.load_state_dict(ck["model"])
        for o, sd in zip(opts, ck["opts"]):
            o.load_state_dict(sd)
        for sc, sd in zip(scheds, ck["scheds"]):
            sc.load_state_dict(sd)
        start, records, elapsed = ck["step"], ck["records"], ck["elapsed"]
    if start == 0:
        records = [{"step": 0, "lr": 0.0, "train_loss": None, **evals.evaluate(model)}]
    _atomic_write_text(log_path, "".join(json.dumps(r) + "\n" for r in records))

    with tracking.start(s, wandb_target, quiet=not verbose) as tracker:
        for r in records:
            tracker.log(r)
        t0 = time.time()
        loss_sum, loss_n = torch.zeros((), device=device), 0
        bx = by = None
        for step in range(start, s.steps):
            j = step % s.block_steps
            if j == 0 or bx is None:
                xs, u, noise, w = data.train_block(s.seed, s.batch, step // s.block_steps, s.block_steps,
                                                   fresh_tasks=s.M is None)
                bx, noise = xs.to(device), noise.to(device)
                w = w.to(device) if s.M is None else pool_dev[data.pick_index(u, s.M).to(device)]
                by = data.targets(bx, w, noise)
            loss = ((model(bx[j], by[j]) - by[j]) ** 2).mean()
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
            for sc in scheds:
                sc.step()
            loss_sum += loss.detach()
            loss_n += 1
            done = step + 1

            if done % s.eval_every == 0 or done == s.steps:
                rec = {"step": done, "lr": opts[0].param_groups[0]["lr"],
                       "train_loss": loss_sum.item() / loss_n, **evals.evaluate(model)}
                if not math.isfinite(rec["train_loss"]):
                    raise FloatingPointError(f"{run_name(s)}: training loss is not finite at step {done}")
                records.append(rec)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                tracker.log(rec)
                loss_sum.zero_()
                loss_n = 0
                if verbose:
                    rate = (done - start) / max(time.time() - t0, 1e-9)
                    print(f"[{run_name(s)}] step {done:>7}/{s.steps}  loss {rec['train_loss']:.4f}  "
                          f"gap_ridge_new {rec['gap_ridge_new']:.4f}  {rate:.0f} steps/s", flush=True)
            if done % s.ckpt_every == 0 and done < s.steps:
                _atomic_torch_save({"settings_hash": h, "step": done, "records": records,
                                    "elapsed": elapsed + time.time() - t0,
                                    "model": model.state_dict(),
                                    "opts": [o.state_dict() for o in opts],
                                    "scheds": [sc.state_dict() for sc in scheds]}, ckpt_path)
            if stop_after is not None and done >= stop_after and done < s.steps:
                return None

        _atomic_torch_save(model.state_dict(), out_dir / "model.pt")
        last = records[-1]
        summary = {
            "seed": s.seed, "opt": s.opt, "M": "inf" if s.M is None else s.M,
            "log2_M": "" if s.M is None else round(math.log2(s.M), 4),
            "batch": s.batch, "lr": s.lr, "lr_aux": "" if s.lr_aux is None else s.lr_aux,
            "tuned": s.tuned, "steps": s.steps, "scale": s.scale,
            **{k: v for k, v in last.items() if k.startswith(("gap_", "mse_"))},
            **evals.baseline_errors(),
            "t_mem": first_memorized_step(records) or "",
            "settings_hash": h, "git_commit": _git_commit(), "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "wall_seconds": round(elapsed + time.time() - t0, 1),
        }
        _atomic_write_text(final_path, json.dumps(summary, indent=2))
        if ckpt_path.exists():
            ckpt_path.unlink()
        tracker.summarize(summary)
    return summary
