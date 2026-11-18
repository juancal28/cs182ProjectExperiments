import json

import pytest
import torch

from e1.config import Settings, make_settings, run_name, settings_hash
from e1.train import run_one


def tiny(opt="adamw", M=16, seed=0, **kw):
    base = dict(seed=seed, M=M, opt=opt, batch=8, steps=60, lr=1e-3,
                lr_aux=1e-3 if opt == "muon" else None, scale="test",
                eval_every=20, n_eval=32, ckpt_every=20, block_steps=10)
    base.update(kw)
    return Settings(**base)


def read_log(out, s):
    return [json.loads(line) for line in (out / run_name(s) / "log.jsonl").read_text().splitlines()]


def test_run_writes_expected_files_and_log(tmp_path):
    s = tiny()
    summary = run_one(s, tmp_path, "cpu", verbose=False)
    d = tmp_path / run_name(s)
    assert (d / "final.json").exists() and (d / "model.pt").exists() and (d / "config.json").exists()
    assert not (d / "checkpoint.pt").exists()
    log = read_log(tmp_path, s)
    assert [r["step"] for r in log] == [0, 20, 40, 60]
    assert log[-1]["lr"] == 0.0                                    # the triangle has returned to zero
    for key in ("gap_ridge_new", "gap_dmmse_new", "gap_ridge_seen", "gap_dmmse_seen", "mse_model_new"):
        assert key in log[-1] and log[-1][key] >= 0
    assert summary["settings_hash"] == settings_hash(s)
    assert summary["gap_ridge_new"] == log[-1]["gap_ridge_new"]
    assert summary["mse_dmmse_seen"] < summary["mse_ridge_seen"]


@pytest.mark.parametrize("opt", ["adamw", "muon", "sgd"])
def test_same_settings_give_identical_logs(tmp_path, opt):
    s = tiny(opt)
    run_one(s, tmp_path / "a", "cpu", verbose=False)
    run_one(s, tmp_path / "b", "cpu", verbose=False)
    assert read_log(tmp_path / "a", s) == read_log(tmp_path / "b", s)


@pytest.mark.parametrize("opt", ["adamw", "muon", "sgd"])
def test_resume_equals_uninterrupted(tmp_path, opt):
    s = tiny(opt)
    run_one(s, tmp_path / "whole", "cpu", verbose=False)
    assert run_one(s, tmp_path / "parts", "cpu", stop_after=25, verbose=False) is None
    d = tmp_path / "parts" / run_name(s)
    assert (d / "checkpoint.pt").exists() and not (d / "final.json").exists()
    run_one(s, tmp_path / "parts", "cpu", verbose=False)
    assert read_log(tmp_path / "whole", s) == read_log(tmp_path / "parts", s)
    a = torch.load(tmp_path / "whole" / run_name(s) / "model.pt", weights_only=True)
    b = torch.load(d / "model.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_finished_run_is_not_retrained(tmp_path):
    s = tiny()
    first = run_one(s, tmp_path, "cpu", verbose=False)
    log_path = tmp_path / run_name(s) / "log.jsonl"
    before = log_path.stat().st_mtime_ns
    assert run_one(s, tmp_path, "cpu", verbose=False) == first
    assert log_path.stat().st_mtime_ns == before


def test_optimizers_and_batch_sizes_share_initial_weights_and_pool(tmp_path):
    step0 = []
    for i, (opt, batch) in enumerate([("adamw", 8), ("sgd", 8), ("muon", 16)]):
        s = tiny(opt, batch=batch)
        run_one(s, tmp_path / str(i), "cpu", verbose=False)
        step0.append({k: v for k, v in read_log(tmp_path / str(i), s)[0].items() if k.startswith(("gap", "mse"))})
    assert step0[0] == step0[1] == step0[2]          # same model, same pool, same evaluation prompts
    other = tiny("adamw", seed=1)
    run_one(other, tmp_path / "other", "cpu", verbose=False)
    assert read_log(tmp_path / "other", other)[0]["gap_ridge_new"] != step0[0]["gap_ridge_new"]


def test_changed_settings_refuse_to_reuse_a_folder(tmp_path):
    run_one(tiny(), tmp_path, "cpu", verbose=False)
    with pytest.raises(RuntimeError, match="different settings"):
        run_one(tiny(lr=2e-3), tmp_path, "cpu", verbose=False)


def test_fresh_tasks_run_has_no_seen_set(tmp_path):
    s = tiny(M=None)
    summary = run_one(s, tmp_path, "cpu", verbose=False)
    assert summary["M"] == "inf" and "gap_ridge_new" in summary and "gap_dmmse_seen" not in summary


def test_training_reduces_loss(tmp_path):
    # Two tasks are quick to memorize, so the error on those tasks must fall well below
    # the error of predicting zero (about 1 after dividing by D).
    s = tiny(M=2, steps=1500, eval_every=500, ckpt_every=500, lr=3e-3, batch=64)
    run_one(s, tmp_path, "cpu", verbose=False)
    log = read_log(tmp_path, s)
    assert log[-1]["mse_model_seen"] < 0.5 * log[0]["mse_model_seen"]
    assert log[-1]["train_loss"] < log[1]["train_loss"]


def test_make_settings_marks_overrides():
    assert make_settings(0, 512, "adamw", 256).tuned == "reference"
    assert make_settings(0, 512, "sgd", 256).tuned == "untuned"
    assert make_settings(0, 512, "sgd", 256, lr=0.05).tuned == "override"
    assert make_settings(0, 512, "adamw", 256, "prelim").steps == 20_000
