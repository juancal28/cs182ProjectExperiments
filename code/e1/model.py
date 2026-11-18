"""GPT-2-style transformer for in-context regression, mirroring icl/gpt2.py and icl/models.py
of the reference code (nanoGPT layout): pre-LayerNorm blocks with biases, learned positional
embedding, final LayerNorm, read-in and read-out without bias.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from e1.seeding import make_gen


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        shape = (b, t, self.n_head, c // self.n_head)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        # Written out as in the reference. The fused attention kernel is avoided because its
        # backward pass is non-deterministic on CUDA, which would break same-seed repeatability.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        att = torch.softmax(att.masked_fill(~mask, torch.finfo(att.dtype).min), dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).reshape(b, t, c))


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = SelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ICLTransformer(nn.Module):
    def __init__(self, d: int = 8, k: int = 16, n_layer: int = 4, n_embd: int = 64, n_head: int = 2):
        super().__init__()
        self.d, self.k, self.n_layer = d, k, n_layer
        self.read_in = nn.Linear(d + 1, n_embd, bias=False)
        self.wpe = nn.Parameter(torch.zeros(2 * k, n_embd))
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5)
        self.read_out = nn.Linear(n_embd, 1, bias=False)

    @staticmethod
    def to_seq(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Interleave tokens: x-token = [0, x], y-token = [y, 0, ..., 0]. -> [B, 2K, D+1]"""
        b, k, d = xs.shape
        x_tok = torch.cat([xs.new_zeros(b, k, 1), xs], dim=2)
        y_tok = torch.cat([ys.unsqueeze(-1), xs.new_zeros(b, k, d)], dim=2)
        return torch.stack([x_tok, y_tok], dim=2).reshape(b, 2 * k, d + 1)

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """xs [B, K, D], ys [B, K] -> prediction for every y_k, read at the x_k token."""
        h = self.read_in(self.to_seq(xs, ys)) + self.wpe[: 2 * xs.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.read_out(self.ln_f(h))[:, ::2, 0]


def init_weights(model: ICLTransformer, seed: int) -> None:
    """Reference init, drawn from the seed's "init" stream in parameter-definition order.

    Weights and positional embedding: normal(0, 0.02). Residual output projections (both
    c_proj weights): normal(0, 0.02 / sqrt(2 * n_layer)). Biases 0, LayerNorm scale 1.
    """
    gen = make_gen(seed, "init")
    scaled = 0.02 / math.sqrt(2 * model.n_layer)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".ln_" in name or name.startswith("ln_f"):
                p.fill_(1.0 if name.endswith("weight") else 0.0)
            elif name.endswith("bias"):
                p.zero_()
            else:
                std = scaled if name.endswith("c_proj.weight") else 0.02
                p.copy_(torch.randn(p.shape, generator=gen) * std)
