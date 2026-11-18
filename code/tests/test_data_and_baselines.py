import numpy as np
import torch

from e1 import data
from e1.baselines import dmmse_preds, ridge_preds
from e1.config import D, K, MAX_POOL, NOISE_STD, RIDGE_LAM
from e1.seeding import make_gen, stream_seed


# ---------- seeding ----------

def test_stream_seed_is_stable_and_separates_streams():
    assert stream_seed(7, "pool") == stream_seed(7, "pool")
    assert stream_seed(7, "pool") != stream_seed(8, "pool")
    assert stream_seed(7, "pool") != stream_seed(7, "init")
    assert stream_seed(7, "train", 256, 0) != stream_seed(7, "train", 256, 1)
    assert 0 <= stream_seed(7, "pool") < 2**63


def test_generators_do_not_touch_global_seed():
    torch.manual_seed(123)
    a = torch.randn(3)
    torch.manual_seed(123)
    torch.randn(5, generator=make_gen(1, "x"))
    assert torch.equal(a, torch.randn(3))


# ---------- pool and training data ----------

def test_pool_is_repeatable_and_seed_dependent():
    p = data.task_pool(3)
    assert p.shape == (MAX_POOL, D)
    assert torch.equal(p, data.task_pool(3))
    assert not torch.equal(p, data.task_pool(4))
    assert abs(p.std().item() - 1.0) < 0.01


def test_train_block_repeatable_and_keyed_by_seed_batch_block():
    a = data.train_block(3, 16, 0, 10)
    b = data.train_block(3, 16, 0, 10)
    assert all(torch.equal(x, y) for x, y in zip(a[:3], b[:3]))
    assert a[0].shape == (10, 16, K, D) and a[1].shape == (10, 16) and a[2].shape == (10, 16, K)
    assert a[3] is None
    for other in (data.train_block(4, 16, 0, 10), data.train_block(3, 16, 1, 10), data.train_block(3, 32, 0, 10)):
        assert not torch.equal(a[0][:, :16], other[0][:, :16])


def test_fresh_tasks_do_not_change_other_streams():
    a = data.train_block(3, 16, 0, 10)
    b = data.train_block(3, 16, 0, 10, fresh_tasks=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert b[3].shape == (10, 16, D)


def test_pick_index_in_range_and_nested_across_M():
    u = torch.rand(10000, generator=make_gen(0, "t"))
    for M in (1, 512, 1448, MAX_POOL):
        idx = data.pick_index(u, M)
        assert idx.min() >= 0 and idx.max() < M
    assert data.pick_index(torch.tensor([0.999999999]), 512).item() == 511


def test_eval_sets_repeatable_and_share_inputs_across_M():
    pool = data.task_pool(5)
    xs_a, ys_a, yt_a = data.eval_set(5, "seen", 64, pool[:512])
    xs_b, ys_b, yt_b = data.eval_set(5, "seen", 64, pool[:2048])
    assert torch.equal(xs_a, xs_b)                       # inputs depend on the seed only
    assert not torch.equal(yt_a, yt_b)                   # tasks depend on the pool
    xs_n, ys_n, yt_n = data.eval_set(5, "new", 64)
    assert torch.equal(xs_n, data.eval_set(5, "new", 64)[0])
    assert not torch.equal(xs_n, xs_a)
    assert abs((ys_n - yt_n).std().item() - NOISE_STD) < 0.05


def test_seen_eval_tasks_come_from_the_pool():
    pool = data.task_pool(5)[:8]
    xs, _, y_true = data.eval_set(5, "seen", 32, pool)
    w_hat = torch.linalg.lstsq(xs.double(), y_true.double().unsqueeze(-1)).solution.squeeze(-1)   # [32, D]
    dist = torch.cdist(w_hat.float(), pool).min(dim=1).values
    assert dist.max() < 1e-3


# ---------- baselines ----------

def _prompts(n=40, seed=0):
    g = make_gen(seed, "test")
    xs = torch.randn(n, K, D, generator=g)
    w = torch.randn(n, D, generator=g)
    ys = data.targets(xs, w, torch.randn(n, K, generator=g) * NOISE_STD)
    return xs, ys, w


def test_ridge_matches_independent_solve():
    xs, ys, _ = _prompts()
    preds = ridge_preds(xs, ys)
    assert torch.all(preds[:, 0] == 0)
    x, y = xs.double().numpy(), ys.double().numpy()
    for i in (1, 5, D, K - 1):
        for n in range(5):
            X, Y = x[n, :i], y[n, :i]
            w = np.linalg.solve(X.T @ X + RIDGE_LAM * np.eye(D), X.T @ Y)
            assert abs(preds[n, i].item() - x[n, i] @ w) < 1e-4


def test_dmmse_single_task_is_that_task():
    xs, ys, _ = _prompts()
    w = torch.randn(1, D, generator=make_gen(1, "w"))
    assert torch.allclose(dmmse_preds(xs, ys, w), (xs * w).sum(-1), atol=1e-5)


def test_dmmse_first_position_is_pool_mean_and_chunking_is_irrelevant():
    xs, ys, _ = _prompts()
    pool = data.task_pool(2)[:300]
    full = dmmse_preds(xs, ys, pool, chunk=1000)
    assert torch.allclose(full[:, 0], xs[:, 0] @ pool.mean(0), atol=1e-5)
    for chunk in (1, 7, 40):
        assert torch.allclose(full, dmmse_preds(xs, ys, pool, chunk=chunk), atol=1e-6)


def test_baselines_are_causal():
    xs, ys, _ = _prompts()
    pool = data.task_pool(2)[:64]
    xs2, ys2 = xs.clone(), ys.clone()
    xs2[:, 9:] += 1.0
    ys2[:, 9:] -= 2.0
    for f in (ridge_preds, lambda a, b: dmmse_preds(a, b, pool)):
        assert torch.allclose(f(xs, ys)[:, :9], f(xs2, ys2)[:, :9], atol=1e-6)
        # position 9 may use x_9 itself but not y_9
        ys3 = ys.clone()
        ys3[:, 9] += 5.0
        assert torch.allclose(f(xs, ys)[:, :10], f(xs, ys3)[:, :10], atol=1e-6)


def test_dmmse_wins_on_seen_tasks_and_ridge_on_new_tasks():
    pool = data.task_pool(11)[:64]
    err = {}
    for kind in ("seen", "new"):
        xs, ys, y_true = data.eval_set(11, kind, 2000, pool)
        err[kind] = {"ridge": ((ridge_preds(xs, ys) - y_true) ** 2).mean().item(),
                     "dmmse": ((dmmse_preds(xs, ys, pool) - y_true) ** 2).mean().item()}
    assert err["seen"]["dmmse"] < err["seen"]["ridge"]
    assert err["new"]["ridge"] < err["new"]["dmmse"]
