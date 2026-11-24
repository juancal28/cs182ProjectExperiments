import pytest

from e1 import config, tracking
from e1.config import make_settings


class FakeRun:
    def __init__(self, resumed=False, summary=None):
        self.resumed, self.summary, self.logged, self.exit_code = resumed, dict(summary or {}), [], None

    def log(self, rec):
        self.logged.append(rec)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


def test_logging_is_off_at_smoke_scale_and_when_disabled():
    assert tracking.resolve(None, "smoke") is None
    assert tracking.resolve(False, "full") is None


def test_missing_entity_is_refused_before_training(monkeypatch):
    monkeypatch.setattr(config, "WANDB_ENTITY", None)
    with pytest.raises(RuntimeError, match="WANDB_ENTITY"):
        tracking.resolve(None, "full")


def test_tracker_without_a_run_does_nothing():
    with tracking.start(make_settings(0, 512, "adamw", 256, "smoke"), None) as t:
        t.log({"step": 0, "train_loss": None})
        t.summarize({"gap_ridge_new": 1.0})


def test_tracker_drops_none_values_and_closes_the_run():
    run = FakeRun()
    with tracking.Tracker(run) as t:
        t.log({"step": 0, "train_loss": None, "gap_ridge_new": 2.0})
        t.summarize({"t_mem": "", "wall_seconds": 3.0})
    assert run.logged == [{"step": 0, "gap_ridge_new": 2.0}]
    assert run.summary == {"wall_seconds": 3.0} and run.exit_code == 0


def test_resumed_run_skips_records_it_already_holds():
    run = FakeRun(resumed=True, summary={"step": 200})
    t = tracking.Tracker(run)
    for step in (0, 100, 200, 300):
        t.log({"step": step})
    assert [r["step"] for r in run.logged] == [300]


def test_failed_training_marks_the_run_failed():
    run = FakeRun()
    with pytest.raises(FloatingPointError):
        with tracking.Tracker(run):
            raise FloatingPointError("loss is not finite")
    assert run.exit_code == 1


def test_group_matches_the_results_folder():
    s = make_settings(7, 512, "adamw", 256, "prelim")
    assert tracking.group_name(s) == "prelim/seed_0007"
    assert tracking.group_name(s, "pilot20k") == "prelim_pilot20k/seed_0007"
