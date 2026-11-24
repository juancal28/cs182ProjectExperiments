"""Weights & Biases logging for training runs.

Logging is deliberately not part of Settings: turning it on or off must not change a run's
settings hash or its numbers. Every run goes to one entity and project (WANDB_ENTITY and
WANDB_PROJECT in e1.config), so all seeds from all machines land in the same place.
"""
import argparse
import dataclasses
from typing import Optional

from e1 import config
from e1.config import Settings, run_name, settings_hash


@dataclasses.dataclass(frozen=True)
class WandbTarget:
    entity: str
    project: str
    tag: str = ""         # the output-folder suffix of run_one --tag, so pilots get their own group


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                   help="log to Weights & Biases (default: on, except at smoke scale)")
    p.add_argument("--wandb-entity", help="override WANDB_ENTITY from e1/config.py")
    p.add_argument("--wandb-project", help="override WANDB_PROJECT from e1/config.py")


def resolve(enabled: Optional[bool], scale: str, entity: Optional[str] = None,
            project: Optional[str] = None, tag: str = "") -> Optional[WandbTarget]:
    """Where a command logs to, or None when logging is off.

    Raises RuntimeError when logging is on but cannot work, so that is found before any training.
    """
    if enabled is None:
        enabled = scale != "smoke"
    if not enabled:
        return None
    entity = entity or config.WANDB_ENTITY
    if not entity:
        raise RuntimeError("no W&B entity: set WANDB_ENTITY in e1/config.py to the team's entity, "
                           "pass --wandb-entity, or use --no-wandb")
    try:
        import wandb
    except ImportError:
        raise RuntimeError("wandb is not installed: pip install wandb, or use --no-wandb") from None
    wandb.login()         # asks for the API key here, in the foreground, not inside a worker process
    return WandbTarget(entity, project or config.WANDB_PROJECT, tag)


def group_name(s: Settings, tag: str = "") -> str:
    """One W&B group per seed folder, named like the folder: "prelim/seed_0007"."""
    return f"{s.scale}_{tag}/seed_{s.seed:04d}" if tag else f"{s.scale}/seed_{s.seed:04d}"


class Tracker:
    """Sends a run's log records to W&B. Does nothing when `run` is None."""

    def __init__(self, run=None):
        self.run = run
        # A resumed W&B run already holds the records logged before the interruption.
        self.last = run.summary.get("step", -1) if run is not None and run.resumed else -1

    def log(self, rec: dict) -> None:
        if self.run is not None and rec["step"] > self.last:
            self.run.log({k: v for k, v in rec.items() if v is not None})
            self.last = rec["step"]

    def summarize(self, summary: dict) -> None:
        if self.run is not None:
            self.run.summary.update({k: v for k, v in summary.items() if v != ""})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            self.run.finish(exit_code=0 if exc_type is None else 1)


def start(s: Settings, target: Optional[WandbTarget], quiet: bool = False) -> Tracker:
    """Open (or resume) the W&B run for these settings.

    The run id is derived from the settings hash, so a training run that is stopped and
    continued keeps logging to the same W&B run.
    """
    if target is None:
        return Tracker()
    import wandb
    h = settings_hash(s)
    run = wandb.init(entity=target.entity, project=target.project, group=group_name(s, target.tag),
                     job_type=s.opt, name=f"{run_name(s)}_s{s.seed}", id=f"{run_name(s)}-s{s.seed}-{h}",
                     resume="allow", config={**s.as_dict(), "settings_hash": h},
                     tags=[s.scale, s.opt, s.tuned], settings=wandb.Settings(silent=quiet))
    # Plot against the training step. W&B's own step counter cannot go back, which a resume
    # from an older checkpoint needs.
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    return Tracker(run)
