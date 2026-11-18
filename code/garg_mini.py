"""Minimal recreation of Garg et al. 2022 (linear regression, in-context).

Trains a small GPT-2-style transformer from scratch on prompts
(x_1, y_1, ..., x_n, y_n) with y_i = w.x_i, a fresh w per prompt, then plots
squared error vs number of in-context examples against least squares,
in-distribution and under two of Garg's shifts (scaled inputs, skewed covariance).

Usage:  python garg_mini.py                      # CPU-sized: d=5, about 10 min
        python garg_mini.py --d 20 --n 41 --embd 256 --layers 12 --heads 8 \
               --steps 500000 --lr 1e-4 --curriculum   # the paper's setting (GPU)
"""
import argparse, math, time
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=5)        # input dimension
p.add_argument("--n", type=int, default=11)       # in-context examples per prompt
p.add_argument("--embd", type=int, default=64)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--curriculum", action="store_true")  # Garg's schedule: grow d and n
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="garg_mini")
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D, N = args.d, args.n


# ---------- data: one new regression task per prompt ----------
def sample(batch, n=N, d_active=D, x_transform=None):
    xs = torch.randn(batch, n, D, device=dev)
    if x_transform is not None:              # distribution shift on inputs
        xs = xs @ x_transform
    w = torch.randn(batch, D, device=dev)
    xs[:, :, d_active:] = 0                  # curriculum: unused dims are zero
    w[:, d_active:] = 0
    ys = (xs * w[:, None, :]).sum(-1)        # y_i = w . x_i   (noiseless)
    return xs, ys


# ---------- model: decoder-only transformer on interleaved (x, y) tokens ----------
class ICLTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.read_in = nn.Linear(D, args.embd)
        self.pos = nn.Parameter(torch.zeros(2 * N, args.embd))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(args.embd, args.heads, 4 * args.embd,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, args.layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(args.embd)
        self.read_out = nn.Linear(args.embd, 1)

    def forward(self, xs, ys):
        B, n, _ = xs.shape
        y_tok = torch.zeros_like(xs)
        y_tok[:, :, 0] = ys                                  # y padded to D dims
        z = torch.stack((xs, y_tok), dim=2).view(B, 2 * n, D)  # x1,y1,x2,y2,...
        h = self.read_in(z) + self.pos[: 2 * n]
        causal = torch.triu(torch.ones(2 * n, 2 * n, dtype=torch.bool, device=z.device), 1)
        h = self.ln(self.blocks(h, mask=causal))
        return self.read_out(h)[:, ::2, 0]   # prediction for y_i, read at the x_i token


# ---------- baseline: least squares on the first k examples ----------
def least_squares(xs, ys):
    preds = torch.zeros_like(ys)             # k = 0 -> predict 0
    for k in range(1, xs.shape[1]):
        w_hat = torch.linalg.pinv(xs[:, :k]) @ ys[:, :k, None]   # min-norm solution
        preds[:, k] = (xs[:, k : k + 1] @ w_hat)[:, 0, 0]
    return preds


@torch.no_grad()
def evaluate(model, x_transform=None, batch=2000):
    xs, ys = sample(batch, x_transform=x_transform)
    scale = ys.pow(2).mean()                 # = error of always predicting 0
    err = lambda pred: ((pred - ys) ** 2).mean(0).cpu() / scale.cpu()
    return err(model(xs, ys)), err(least_squares(xs, ys))


def skewed_covariance():                      # Garg's "skewed": eigenvalues 1/i
    eig = 1 / (torch.arange(D) + 1.0)
    U, _, _ = torch.linalg.svd(torch.randn(D, D))
    t = U @ torch.diag(eig) @ U.T
    return (t * math.sqrt(D / (eig ** 2).sum())).to(dev)


# ---------- train ----------
model = ICLTransformer().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
print(f"device={dev}  params={sum(q.numel() for q in model.parameters()):,}")
t0 = time.time()
for step in range(args.steps + 1):
    if args.curriculum:                       # d: 5 -> D, n: 11 -> N, every 2000 steps
        k = step // 2000
        d_act, n_act = min(D, 5 + k), min(N, 11 + 2 * k)
    else:
        d_act, n_act = D, N
    xs, ys = sample(args.batch, n=n_act, d_active=d_act)
    loss = ((model(xs, ys) - ys) ** 2).mean()   # squared error at every position
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        tf, ls = evaluate(model)
        print(f"step {step:6d}  train {loss.item():7.3f}  "
              f"err@last: transformer {tf[-1]:.4f}  least-squares {ls[-1]:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
torch.save(model.state_dict(), f"{args.out}.pt")

# ---------- evaluate + plot ----------
settings = {"in-distribution": None,
            "inputs x 3": 3.0 * torch.eye(D, device=dev),
            "inputs x 1/3": torch.eye(D, device=dev) / 3,
            "skewed covariance": skewed_covariance()}
fig, axes = plt.subplots(1, len(settings), figsize=(4 * len(settings), 3.4), sharey=True)
for ax, (name, t) in zip(axes, settings.items()):
    tf, ls = evaluate(model, t)
    ax.plot(tf, label="Transformer", lw=2)
    ax.plot(ls, label="Least squares", lw=2, ls="--")
    ax.axvline(D, color="gray", lw=0.8, ls=":")       # n = d: w becomes identifiable
    ax.set_title(name); ax.set_xlabel("in-context examples")
    print(f"{name:20s} transformer {[round(v, 3) for v in tf.tolist()]}")
    print(f"{'':20s} least sq.   {[round(v, 3) for v in ls.tolist()]}")
axes[0].set_ylabel("squared error / error of predicting 0"); axes[0].legend()
plt.tight_layout(); plt.savefig(f"{args.out}.png", dpi=150)
print(f"saved {args.out}.png and {args.out}.pt")
