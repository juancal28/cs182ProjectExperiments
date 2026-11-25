"""Log-derived clocks (e1.dynamics) and the off-distribution prompts (e1.probes)."""
import numpy as np
import torch

from e1 import dynamics as dy
from e1 import probes
from e1.config import make_settings


def test_first_step_within_uses_tolerance():
    steps = [0, 1000, 2000, 3000]
    assert dy.first_step_within(steps, [5.0, 1.2, 1.09, 1.0], 1.0, tol=0.10) == 2000
    assert dy.first_step_within(steps, [5.0, 4.0, 3.0, 2.0], 1.0) is None


def test_memorized_fraction_endpoints():
    phi = dy.memorized_fraction([0.35, 0.20, 0.275], mse_ridge_seen=0.35, mse_dmmse_seen=0.20)
    assert np.allclose(phi, [0.0, 1.0, 0.5])


def test_onset_needs_a_sustained_run_and_skips_step_zero():
    steps = list(range(0, 9000, 1000))
    phi = [0.9, 0.0, 0.06, 0.0, 0.06, 0.07, 0.08, 0.02, 0.0]
    assert dy.onset_step(steps, phi, level=0.05, hold=3) == 4000
    assert dy.onset_step(steps, phi, level=0.05, hold=4) is None


def test_rebound_reports_growth_from_the_minimum():
    r = dy.rebound([0, 1000, 2000, 3000], [1.0, 0.1, 0.05, 0.08])
    assert r["step_min"] == 2000 and abs(r["rise"] - 0.6) < 1e-12
    assert dy.rebound([0, 1000], [1.0, 0.5])["rise"] == 0.0


def test_level_crossing_interpolates_in_log_and_takes_the_last_crossing():
    assert abs(dy.level_crossing([9, 10], [0.04, 0.01], 0.02) - 9.5) < 1e-9
    # falls below, comes back above, falls again: the last descent counts
    assert abs(dy.level_crossing([9, 10, 11, 12], [0.04, 0.01, 0.04, 0.01], 0.02) - 11.5) < 1e-9
    assert dy.level_crossing([9, 10], [0.04, 0.03], 0.02) is None      # never gets below
    assert dy.level_crossing([9, 10], [0.01, 0.005], 0.02) is None     # never above


def test_power_law_recovers_exponent():
    ms = np.array([512, 1024, 2048, 4096])
    fit = dy.power_law(ms, 3.0 * ms ** 1.5)
    assert abs(fit["nu"] - 1.5) < 1e-9 and abs(2 ** fit["log2_c"] - 3.0) < 1e-6
    assert dy.power_law([512, 1024], [1.0, 2.0]) is None


def test_skew_transform_has_the_stated_spectrum():
    d = 8
    t = probes.skew_transform(3, d)
    cov = t.T @ t
    eig = torch.linalg.eigvalsh(cov).flip(0)
    want = 1.0 / torch.arange(1, d + 1) ** 2
    assert torch.allclose(eig, want * d / want.sum(), atol=1e-5)
    assert abs(cov.trace().item() - d) < 1e-4


def test_probe_prompts_share_base_draws_and_scale_the_right_inputs():
    s = make_settings(5, 512, "adamw", 256, scale="smoke")
    base_x, base_y, base_t = probes.probe_prompts(s, "query", 1.0, n=64)
    qx, qy, qt = probes.probe_prompts(s, "query", 4.0, n=64)
    assert torch.equal(qx[:, :-1], base_x[:, :-1]) and torch.allclose(qx[:, -1], 4.0 * base_x[:, -1])
    assert torch.equal(qy[:, :-1], base_y[:, :-1]) and torch.allclose(qt[:, -1], 4.0 * base_t[:, -1], atol=1e-5)
    ix, iy, it = probes.probe_prompts(s, "input", 0.5, n=64)
    assert torch.allclose(ix, 0.5 * base_x) and torch.allclose(it, 0.5 * base_t, atol=1e-6)
    assert torch.allclose(iy - it, base_y - base_t, atol=1e-5)          # label noise is not rescaled
    sx, _, _ = probes.probe_prompts(s, "skew", 1.0, n=4096)
    assert abs(sx.pow(2).sum(-1).mean().item() - s.d) < 0.4             # same expected squared norm
