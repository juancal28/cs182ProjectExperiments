"""Optimizers and the shared triangle learning-rate schedule."""
import torch

from e1.config import Settings


def triangle(i: int, total: int) -> float:
    """Multiplier for update number i (0-based): 0 -> 1 over the first half, 1 -> 0 over the second.

    Matches the reference: optax.join_schedules of two linear schedules with warmup = total / 2.
    """
    warmup = total // 2
    if i < warmup:
        return i / warmup
    return max(0.0, 1.0 - (i - warmup) / (total - warmup))


def muon_param_names(model: torch.nn.Module) -> list[str]:
    """Muon gets the 2-D weight matrices inside the transformer blocks (attention and MLP)."""
    return [n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]


def build_optimizers(model: torch.nn.Module, s: Settings):
    """Returns (optimizers, schedulers). Weight decay is zero everywhere."""
    named = dict(model.named_parameters())
    if s.opt == "adamw":
        opts = [torch.optim.AdamW(named.values(), lr=s.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)]
    elif s.opt == "sgd":
        opts = [torch.optim.SGD(named.values(), lr=s.lr, momentum=s.sgd_momentum, weight_decay=0.0)]
    elif s.opt == "muon":
        if s.lr_aux is None:
            raise ValueError("muon needs lr_aux for the parameters AdamW handles")
        muon_names = set(muon_param_names(model))
        opts = [
            torch.optim.Muon([named[n] for n in named if n in muon_names], lr=s.lr,
                             weight_decay=0.0, momentum=s.muon_momentum, nesterov=True),
            torch.optim.AdamW([named[n] for n in named if n not in muon_names], lr=s.lr_aux,
                              betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        ]
    else:
        raise ValueError(s.opt)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda i: triangle(i, s.steps)) for o in opts]
    return opts, scheds
