import math

import pytest
import torch

from e1.config import D, K, make_settings
from e1.model import ICLTransformer, init_weights
from e1.optim import build_optimizers, muon_param_names, triangle


def _model(seed=0):
    m = ICLTransformer()
    init_weights(m, seed)
    return m


def test_token_layout_matches_reference():
    xs = torch.arange(2 * 3 * D, dtype=torch.float32).reshape(2, 3, D) + 1
    ys = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    seq = ICLTransformer.to_seq(xs, ys)
    assert seq.shape == (2, 6, D + 1)
    assert torch.equal(seq[0, 0], torch.cat([torch.zeros(1), xs[0, 0]]))        # x-token = [0, x]
    assert torch.equal(seq[0, 1], torch.cat([ys[0, :1], torch.zeros(D)]))       # y-token = [y, 0...]
    assert torch.equal(seq[1, 4], torch.cat([torch.zeros(1), xs[1, 2]]))


def test_forward_shape_and_causality():
    m = _model().eval()
    xs, ys = torch.randn(5, K, D), torch.randn(5, K)
    out = m(xs, ys)
    assert out.shape == (5, K)
    xs2, ys2 = xs.clone(), ys.clone()
    xs2[:, 7:] += 3.0
    ys2[:, 6:] -= 3.0          # y_6 sits after the x_6 token, so prediction 6 must not see it
    assert torch.allclose(out[:, :7], m(xs2, ys2)[:, :7], atol=1e-5)
    assert not torch.allclose(out[:, 7:], m(xs2, ys2)[:, 7:], atol=1e-5)


def test_init_is_a_function_of_the_seed():
    a, b, c = _model(1), _model(1), _model(2)
    for (n, p), q, r in zip(a.named_parameters(), b.parameters(), c.parameters()):
        assert torch.equal(p, q), n
        if p.ndim == 2:
            assert not torch.equal(p, r), n


def test_init_statistics_and_parameter_count():
    m = _model()
    p = dict(m.named_parameters())
    assert abs(p["blocks.0.mlp.c_fc.weight"].std().item() - 0.02) < 0.002
    scaled = 0.02 / math.sqrt(2 * 4)
    assert abs(p["blocks.0.mlp.c_proj.weight"].std().item() - scaled) < 0.001
    assert abs(p["blocks.3.attn.c_proj.weight"].std().item() - scaled) < 0.001
    assert torch.all(p["blocks.0.attn.c_attn.bias"] == 0)
    assert torch.all(p["ln_f.weight"] == 1) and torch.all(p["ln_f.bias"] == 0)
    assert "read_in.bias" not in p and "read_out.bias" not in p
    e = 64
    per_block = (3 * e * e + 3 * e) + (e * e + e) + (4 * e * e + 4 * e) + (4 * e * e + e) + 4 * e
    expected = (D + 1) * e + 2 * K * e + 4 * per_block + 2 * e + e
    assert sum(q.numel() for q in m.parameters()) == expected


def test_triangle_schedule():
    total = 1000
    assert triangle(0, total) == 0.0
    assert triangle(250, total) == pytest.approx(0.5)
    assert triangle(500, total) == pytest.approx(1.0)
    assert triangle(750, total) == pytest.approx(0.5)
    assert triangle(1000, total) == 0.0


def test_muon_gets_block_matrices_and_adamw_the_rest():
    m = _model()
    names = muon_param_names(m)
    assert len(names) == 4 * 4
    assert all(n.startswith("blocks.") and n.endswith(".weight") and ".ln_" not in n for n in names)
    s = make_settings(0, 512, "muon", 256, "smoke")
    opts, _ = build_optimizers(m, s)
    assert isinstance(opts[0], torch.optim.Muon) and isinstance(opts[1], torch.optim.AdamW)
    n_muon = sum(p.numel() for g in opts[0].param_groups for p in g["params"])
    n_adam = sum(p.numel() for g in opts[1].param_groups for p in g["params"])
    assert n_muon == sum(dict(m.named_parameters())[n].numel() for n in names)
    assert n_muon + n_adam == sum(p.numel() for p in m.parameters())
    assert all(p.ndim == 2 for g in opts[0].param_groups for p in g["params"])


@pytest.mark.parametrize("opt", ["adamw", "muon", "sgd"])
def test_no_weight_decay_and_schedule_starts_at_zero(opt):
    s = make_settings(0, 512, opt, 256, "smoke")
    opts, scheds = build_optimizers(_model(), s)
    for o in opts:
        for g in o.param_groups:
            assert g["weight_decay"] == 0.0
            assert g["lr"] == 0.0                     # first update of the triangle
    if opt == "adamw":
        assert opts[0].param_groups[0]["betas"] == (0.9, 0.999) and opts[0].param_groups[0]["eps"] == 1e-8
    for o in opts:
        o.step()
    for sc in scheds:
        sc.step()
    assert opts[0].param_groups[0]["lr"] == pytest.approx(s.lr / (s.steps // 2))
