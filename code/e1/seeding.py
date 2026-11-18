"""All randomness in the runner comes from here.

Every random tensor is drawn from a CPU generator whose seed is a stable hash of
(user seed, stream name, index...). Nothing uses the global torch seed, so the
inputs of a run are a pure function of its settings on any machine.
"""
import hashlib

import torch


def stream_seed(seed: int, *parts) -> int:
    """Stable 63-bit seed for one named stream. Python's hash() is salted, so use SHA-256."""
    digest = hashlib.sha256(repr((int(seed),) + tuple(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def make_gen(seed: int, *parts) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(stream_seed(seed, *parts))
    return gen
