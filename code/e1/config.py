"""Settings for one training run, the E1 grid, and the scale presets."""
import dataclasses
import hashlib
import json
from typing import Optional

D = 8                # input dimension
K = 16               # in-context examples per prompt
NOISE_STD = 0.5      # label noise standard deviation (variance 0.25)
RIDGE_LAM = NOISE_STD**2
MAX_POOL = 32768     # one pool of this size per seed; a run with pool size M uses the first M

GRID_M = [512, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 32768]   # 2^9 ... 2^13, 2^15
GRID_OPT = ["adamw", "muon", "sgd"]                                # run order
GRID_BATCH = [256, 512]

# Peak learning rates. Only AdamW has a reference value (Raventos et al.: Adam, 1e-3).
# "tuned" is carried into every result row so untuned numbers are never mistaken for final ones.
LEARNING_RATES = {
    "adamw": {"lr": 1e-3, "lr_aux": None, "tuned": "reference"},
    # lr: Muon matrices; lr_aux: AdamW part. Best of {0.005, 0.02, 0.05} in a 20,000-step
    # fresh-task check on seed 0 (0.05 failed to learn). Not a full pilot.
    "muon": {"lr": 5e-3, "lr_aux": 1e-3, "tuned": "quick-sweep"},
    # SGD did not learn in that check at 0.003-0.1 and diverged at 0.3 and 1.0.
    "sgd": {"lr": 1e-1, "lr_aux": None, "tuned": "untuned"},
}

# steps, eval_every, n_eval, ckpt_every
SCALES = {
    "full": dict(steps=500_000, eval_every=1_000, n_eval=8_192, ckpt_every=10_000),
    "prelim": dict(steps=20_000, eval_every=500, n_eval=8_192, ckpt_every=5_000),
    "smoke": dict(steps=300, eval_every=100, n_eval=256, ckpt_every=100),
}
SMOKE_M = [512, 32768]


@dataclasses.dataclass(frozen=True)
class Settings:
    seed: int
    M: Optional[int]          # pool size; None means a fresh task for every sequence
    opt: str                  # adamw | muon | sgd
    batch: int
    steps: int = 500_000
    lr: float = 1e-3
    lr_aux: Optional[float] = None
    tuned: str = "untuned"
    scale: str = "full"
    eval_every: int = 1_000
    n_eval: int = 8_192
    ckpt_every: int = 10_000
    block_steps: int = 500    # training data is generated in blocks of this many steps
    d: int = D
    k: int = K
    noise_std: float = NOISE_STD
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 2
    sgd_momentum: float = 0.9
    muon_momentum: float = 0.95

    def __post_init__(self):
        if self.opt not in LEARNING_RATES:
            raise ValueError(f"unknown optimizer {self.opt!r}")
        if self.M is not None and not (1 <= self.M <= MAX_POOL):
            raise ValueError(f"M must be in 1..{MAX_POOL} or None, got {self.M}")
        for name in ("eval_every", "ckpt_every"):
            if getattr(self, name) % self.block_steps:
                raise ValueError(f"{name} must be a multiple of block_steps={self.block_steps}")
        if self.steps % 2:
            raise ValueError("steps must be even (the schedule peaks at steps / 2)")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def settings_hash(s: Settings) -> str:
    return hashlib.sha256(json.dumps(s.as_dict(), sort_keys=True).encode()).hexdigest()[:12]


def run_name(s: Settings) -> str:
    m = "Minf" if s.M is None else f"M{s.M:05d}"
    return f"{s.opt}_{m}_b{s.batch}"


def make_settings(seed: int, M: Optional[int], opt: str, batch: int, scale: str = "full", **overrides) -> Settings:
    """Settings for one grid cell at a scale preset; explicit overrides win."""
    fields = dict(SCALES[scale])
    fields.update(LEARNING_RATES[opt])
    if scale == "smoke":
        fields["block_steps"] = 100
    explicit_lr = overrides.get("lr") is not None or overrides.get("lr_aux") is not None
    fields.update({k: v for k, v in overrides.items() if v is not None})
    if explicit_lr and "tuned" not in overrides:
        fields["tuned"] = "override"
    return Settings(seed=seed, M=M, opt=opt, batch=batch, scale=scale, **fields)


def grid(seed: int, scale: str = "full", opts=None, batches=None, Ms=None, **overrides) -> list[Settings]:
    """The E1 grid for one seed, in run order: optimizer, then M, then batch size."""
    default_m = SMOKE_M if scale == "smoke" else GRID_M
    out = []
    for opt in GRID_OPT:
        if opts and opt not in opts:
            continue
        for M in default_m:
            if Ms and M not in Ms:
                continue
            for batch in GRID_BATCH:
                if batches and batch not in batches:
                    continue
                out.append(make_settings(seed, M, opt, batch, scale, **overrides))
    return out
