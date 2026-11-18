import csv
import math

import pytest

from e1.aggregate import analyze, crossover, load_rows, main
from e1.config import GRID_M

COLS = ["seed", "opt", "M", "batch", "steps", "scale", "tuned",
        "gap_ridge_new", "gap_dmmse_new", "gap_ridge_seen", "gap_dmmse_seen"]


def synth(seed, opt, star, steps=500, flip_last=False):
    """Rows whose s(M) and readout are linear in log2 M with a zero at `star`."""
    rows = []
    for m in GRID_M:
        x = math.log2(m)
        s = star - x                                    # positive below the threshold
        if flip_last and m == GRID_M[-1]:
            s = 0.5                                     # noise far above the threshold
        for batch in (256, 512):
            rows.append({"seed": seed, "opt": opt, "M": m, "batch": batch, "steps": steps, "scale": "test",
                         "tuned": "reference", "gap_ridge_new": 1.0 + (s if batch == 512 else 0.0),
                         "gap_dmmse_new": 0.5, "gap_ridge_seen": 1.0, "gap_dmmse_seen": 1.0 + (x - star)})
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_crossover_interpolates_in_log2():
    res = crossover([9, 10, 11], [2.0, 1.0, -1.0])
    assert res["status"] == "ok" and res["x_star"] == pytest.approx(10.5)
    assert crossover([9, 10, 11], [-2.0, -1.0, 1.0], "neg_to_pos")["x_star"] == pytest.approx(10.5)


def test_crossover_not_bracketed_and_multiple():
    assert crossover([9, 10, 11], [3.0, 2.0, 1.0]) == {"status": "not_bracketed", "x_star": None, "crossings": []}
    assert crossover([9, 10], [-1.0, 1.0])["status"] == "not_bracketed"     # wrong direction only
    res = crossover([9, 10, 11, 12, 13], [1.0, -1.0, 1.0, -1.0, -2.0])
    assert res["status"] == "multiple" and res["x_star"] == pytest.approx(9.5) and len(res["crossings"]) == 2


def test_known_threshold_recovered_with_interval(tmp_path):
    files = [write(tmp_path / f"s{seed}.csv", synth(seed, "adamw", star)) for seed, star in [(0, 11.3), (1, 11.7), (2, 11.5)]]
    res = analyze(load_rows(files))
    t = res["thresholds"][0]
    assert t["opt"] == "adamw" and t["status"] == "ok" and t["n_seeds_paired"] == 3
    assert t["log2_Mstar"] == pytest.approx(11.5, abs=0.01)
    assert t["readout_log2_b256"] == pytest.approx(11.5, abs=0.01)
    assert 11.3 <= t["ci95_lo"] <= 11.5 <= t["ci95_hi"] <= 11.7
    assert [r["log2_Mstar"] for r in res["per_seed"]] == pytest.approx([11.3, 11.7, 11.5], abs=0.01)
    assert res["missing"] and all("adamw" not in m for m in res["missing"])   # muon and sgd cells are absent


def test_single_seed_gives_crossing_without_interval(tmp_path):
    t = analyze(load_rows([write(tmp_path / "a.csv", synth(0, "muon", 12.2))]))["thresholds"][0]
    assert t["log2_Mstar"] == pytest.approx(12.2, abs=0.01) and t["ci95_lo"] == ""


def test_multiple_sign_changes_are_flagged(tmp_path):
    t = analyze(load_rows([write(tmp_path / "a.csv", synth(0, "adamw", 11.5, flip_last=True))]))["thresholds"][0]
    assert t["log2_Mstar"] == pytest.approx(11.5, abs=0.01)
    assert t["status"] in ("ok", "multiple")           # the late flip is negative-to-positive, so not a second crossing
    rows = synth(0, "adamw", 11.5)
    for r in rows:                                      # an early dip creates a second positive-to-negative crossing
        if r["M"] == 1024 and r["batch"] == 512:
            r["gap_ridge_new"] = 0.5
    t2 = analyze(load_rows([write(tmp_path / "b.csv", rows)]))["thresholds"][0]
    assert t2["status"] == "multiple" and len(t2["all_crossings_log2"].split()) == 2


def test_incomplete_seed_is_reported_not_fatal(tmp_path):
    rows = [r for r in synth(0, "adamw", 11.5) if not (r["M"] == 2048 and r["batch"] == 512)]
    res = analyze(load_rows([write(tmp_path / "a.csv", rows)]))
    assert "seed 0 adamw M=2048 batch=512" in res["missing"]
    assert res["thresholds"][0]["log2_Mstar"] == pytest.approx(11.5, abs=0.01)


def test_mixed_run_lengths_are_refused(tmp_path):
    a = write(tmp_path / "a.csv", synth(0, "adamw", 11.5, steps=500))
    b = write(tmp_path / "b.csv", synth(1, "adamw", 11.5, steps=900))
    with pytest.raises(ValueError, match="different length"):
        analyze(load_rows([a, b]))


def test_cli_writes_tables_and_figures(tmp_path):
    for seed, star in [(0, 11.4), (1, 11.6)]:
        d = tmp_path / f"seed_{seed:04d}"
        d.mkdir()
        write(d / "summary.csv", synth(seed, "adamw", star) + synth(seed, "sgd", star - 1))
    assert main([str(tmp_path)]) == 0
    out = tmp_path / "aggregate"
    for name in ("thresholds.csv", "per_seed.csv", "crossover.png", "readout_b256.png", "missing.txt"):
        assert (out / name).exists() and (out / name).stat().st_size > 0
