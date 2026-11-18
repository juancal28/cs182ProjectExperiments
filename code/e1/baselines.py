"""Closed-form reference predictors: ridge and the discrete-prior Bayes estimator (dMMSE).

Both return a prediction for every position k that uses only examples before k, matching
icl/models.py in the reference code. Computed in float64, returned as float32.
"""
import torch

from e1.config import NOISE_STD, RIDGE_LAM


@torch.no_grad()
def ridge_preds(xs: torch.Tensor, ys: torch.Tensor, lam: float = RIDGE_LAM) -> torch.Tensor:
    """xs [N, K, D], ys [N, K] -> [N, K]. Position 0 predicts 0."""
    x, y = xs.double(), ys.double()
    n, k, d = x.shape
    # Running sums over the first i examples: A_i = X_i^T X_i, b_i = X_i^T y_i.
    a = torch.cumsum(x.unsqueeze(-1) * x.unsqueeze(-2), dim=1)          # [N, K, D, D]
    b = torch.cumsum(x * y.unsqueeze(-1), dim=1)                        # [N, K, D]
    eye = lam * torch.eye(d, dtype=x.dtype, device=x.device)
    w = torch.linalg.solve(a[:, :-1] + eye, b[:, :-1].unsqueeze(-1)).squeeze(-1)   # [N, K-1, D]
    preds = torch.zeros(n, k, dtype=x.dtype, device=x.device)
    preds[:, 1:] = (x[:, 1:] * w).sum(-1)
    return preds.float()


@torch.no_grad()
def dmmse_preds(xs: torch.Tensor, ys: torch.Tensor, pool: torch.Tensor,
                sigma: float = NOISE_STD, chunk: int = 0) -> torch.Tensor:
    """xs [N, K, D], ys [N, K], pool [M, D] -> [N, K].

    Posterior mean over the pool: weights are a softmax of -sum_j (y_j - w.x_j)^2 / (2 sigma^2)
    over the examples before position k. Position 0 uses the pool mean. `chunk` is the number
    of prompts processed at once (0 picks one that keeps the work tensor near 256 MB).
    """
    x, y, w = xs.double(), ys.double(), pool.double()
    n, k, _ = x.shape
    m = w.shape[0]
    if chunk <= 0:
        chunk = max(1, int(256e6 // (8 * k * m)))
    out = torch.empty(n, k, dtype=x.dtype, device=x.device)
    for s in range(0, n, chunk):
        xc, yc = x[s:s + chunk], y[s:s + chunk]
        fit = xc @ w.T                                                   # [n, K, M] = w_m . x_j
        sq = (yc.unsqueeze(-1) - fit) ** 2
        log_w = -torch.cumsum(sq, dim=1) / (2 * sigma**2)                # after j+1 examples
        log_w = torch.cat([torch.zeros_like(log_w[:, :1]), log_w[:, :-1]], dim=1)   # before position k
        post = torch.softmax(log_w, dim=-1)                              # [n, K, M]
        out[s:s + chunk] = (post * fit).sum(-1)                          # E[w | context] . x_k
    return out.float()
