"""Task pool, training data and evaluation sets. Everything is generated on the CPU.

Streams (see e1.seeding):
  pool         depends on seed only; a run with pool size M uses the first M rows
  train        depends on seed, batch size and block index; never on M or the optimizer
  eval         depends on seed only
"""
from typing import Optional

import torch

from e1.config import D, K, MAX_POOL, NOISE_STD
from e1.seeding import make_gen


def task_pool(seed: int) -> torch.Tensor:
    """[MAX_POOL, D] task vectors for this seed."""
    return torch.randn(MAX_POOL, D, generator=make_gen(seed, "pool"))


def pick_index(u: torch.Tensor, M: int) -> torch.Tensor:
    """Map uniforms in [0, 1) to task indices in [0, M)."""
    return (u.double() * M).long().clamp_(max=M - 1)


def train_block(seed: int, batch: int, block_idx: int, block_steps: int, fresh_tasks: bool = False):
    """Data for steps [block_idx * block_steps, (block_idx + 1) * block_steps).

    Returns xs [S, B, K, D], u [S, B] uniforms for the task pick, noise [S, B, K] (already
    scaled), and w [S, B, D] fresh tasks or None.
    """
    gen = make_gen(seed, "train", batch, block_steps, block_idx)
    xs = torch.randn(block_steps, batch, K, D, generator=gen)
    u = torch.rand(block_steps, batch, generator=gen)
    noise = torch.randn(block_steps, batch, K, generator=gen) * NOISE_STD
    w = None
    if fresh_tasks:
        w = torch.randn(block_steps, batch, D, generator=make_gen(seed, "train_w", batch, block_steps, block_idx))
    return xs, u, noise, w


def targets(xs: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y = w . x (+ noise). xs [..., K, D], w [..., D] -> [..., K]."""
    ys = (xs * w.unsqueeze(-2)).sum(-1)
    return ys if noise is None else ys + noise


def eval_set(seed: int, kind: str, n: int, pool_m: Optional[torch.Tensor] = None):
    """Fixed evaluation prompts. kind is "new" (fresh Gaussian tasks) or "seen" (tasks from pool_m).

    Returns xs [n, K, D], ys [n, K] noisy targets, y_true [n, K] noiseless targets.
    The inputs, noise and task picks depend only on the seed; pool_m only maps picks to tasks.
    """
    if kind not in ("new", "seen"):
        raise ValueError(kind)
    gen = make_gen(seed, "eval", kind)
    xs = torch.randn(n, K, D, generator=gen)
    noise = torch.randn(n, K, generator=gen) * NOISE_STD
    if kind == "new":
        w = torch.randn(n, D, generator=gen)
    else:
        if pool_m is None:
            raise ValueError("seen-task evaluation needs the run's pool")
        w = pool_m[pick_index(torch.rand(n, generator=gen), pool_m.shape[0])]
    y_true = targets(xs, w)
    return xs, y_true + noise, y_true
