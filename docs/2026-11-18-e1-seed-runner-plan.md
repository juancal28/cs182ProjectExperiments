# E1 Seed Runner Implementation Plan

**Goal:** A PyTorch runner that takes a seed and produces that seed's E1 threshold grid, plus an aggregator that turns seed tables into M\*.

**Architecture:** A small package `e1` under `project/code/`. Every random input is a pure function of (seed, stream name, index), generated on the CPU. One training run is a resumable function that writes a folder; a driver hands runs to one worker process per GPU slot; an aggregator reads only `summary.csv` files.

**Tech stack:** Python 3.11, PyTorch 2.15 (CUDA 13.2 build, in `project/code/.venv`), numpy, matplotlib, pytest.

**Spec:** `project/docs/2026-11-18-e1-seed-runner-design.md`

## Global constraints

- D = 8, K = 16, noise standard deviation 0.5, ridge penalty 0.25.
- Model: 4 layers, width 64, 2 heads, token dimension 9, x-token `[0, x]`, y-token `[y, 0…]`, pre-LayerNorm, biases in blocks, no bias on read-in and read-out, LayerNorm eps 1e-5, tanh GELU, init std 0.02 and 0.02 / sqrt(2 × layers) on both `c_proj` weights.
- Schedule: update number i uses peak × i / warmup for i < warmup, then peak × (1 − (i − warmup) / (total − warmup)), with warmup = total / 2.
- Full scale: 500,000 steps, evaluation every 1,000 steps on 8,192 prompts per set, checkpoint every 10,000 steps.
- Grid: M ∈ {512, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 32768} × {adamw, muon, sgd} × batch {256, 512}.
- Muon weight decay must be 0 explicitly. No randomness outside `e1.seeding`. float32 training, float64 baselines.
- No git commits unless Juan asks.

## Review focus

1. A run killed mid-write: a half-written checkpoint or `final.json` must never be read as valid (atomic rename; test resume).
2. M = 32,768 dMMSE memory: must be chunked and must not depend on chunk size (test).
3. A seed's results at one scale must never mix with another scale (separate folders per scale; settings hash in every row).
4. `s(M)` with no sign change or several: the aggregator reports "not bracketed" or flags it instead of crashing (test).
5. A summary from fewer than all 54 runs, or from one seed: the aggregator still runs and says what is missing (test).

## Files

| File | Responsibility |
|---|---|
| `e1/config.py` | `Settings` dataclass, grid, learning rates, scale presets, `settings_hash`, `run_name` |
| `e1/seeding.py` | `stream_seed(seed, *parts) -> int`, `make_gen(seed, *parts) -> torch.Generator` |
| `e1/data.py` | `task_pool(seed)`, `train_block(...)`, `eval_set(...)` |
| `e1/baselines.py` | `ridge_preds(xs, ys, lam)`, `dmmse_preds(xs, ys, pool, sigma, chunk)` |
| `e1/model.py` | `ICLTransformer`, `init_weights(model, seed)` |
| `e1/optim.py` | `triangle(i, total)`, `build_optimizers(model, settings)` |
| `e1/train.py` | `run_one(settings, out_root, device, stop_after=None) -> dict` |
| `e1/run_one.py`, `e1/run_seed.py`, `e1/aggregate.py` | Command-line entry points |
| `tests/test_*.py` | One test file per module |
| `README.md`, `requirements.txt` | Local and AWS usage |

## Tasks

Each task: write the tests, see them fail, implement, see them pass.

### Task 1: seeding and data

- `stream_seed` is SHA-256 of `repr((seed, *parts))`, reduced mod 2^63.
- `task_pool(seed) -> Tensor[32768, 8]`; a run uses `pool[:M]`.
- `train_block(seed, batch, block_idx, block_steps, fresh_tasks=False) -> (xs[S,B,K,D], u[S,B], noise[S,B,K], w or None)`; task index is `floor(u * M)`.
- `eval_set(seed, kind, n, pool_M or None) -> (xs, ys, y_true)` for `kind` in `{"seen", "new"}`.
- Tests: repeatable; a different seed changes everything; pools nested; the block does not depend on M or optimizer; indices in range.

### Task 2: baselines

- Ridge position 0 predicts 0; dMMSE position 0 predicts x · mean(pool).
- Tests: ridge equals an independent numpy solve; dMMSE with M = 1 equals x · w; chunk size does not change the result; changing later examples does not change earlier predictions; on seen tasks dMMSE error < ridge error, on new tasks the reverse (M = 64, 2,000 prompts).

### Task 3: model

- `ICLTransformer(d=8, k=16, n_layer=4, n_embd=64, n_head=2)`, `forward(xs[B,K,D], ys[B,K]) -> preds[B,K]`.
- Tests: output shape; causality; same seed gives identical weights, a different seed does not; init standard deviations; parameter count equals the formula.

### Task 4: optimizers and schedule

- `build_optimizers(model, settings) -> (optimizers, schedulers)`.
- Muon takes 2-D weights whose names start with `blocks.`; AdamW takes the rest.
- Tests: schedule values at 0, warmup / 2, warmup, total; Muon group membership by name; weight decay 0 everywhere; AdamW betas and eps.

### Task 5: one training run

- Folder layout and files as in the spec. `stop_after` exists so tests can interrupt a run.
- Log line: step, learning rate, mean training loss since the last evaluation, and per set the gaps to ridge and dMMSE and the model's error against the noiseless target.
- Tests (tiny settings): same settings twice give identical logs; interrupted and resumed equals uninterrupted (log and weights); different optimizers share initial weights; `final.json` present only after completion; `M = None` (fresh tasks) runs without a seen set.

### Task 6: seed driver

- `run_seed --seed S [--scale full|prelim|smoke] [--opt ...] [--batch ...] [--M ...] [--workers-per-gpu N] [--out DIR]`.
- Results go to `<out>/<scale>/seed_<S>/`. `summary.csv` is rebuilt from the `final.json` files after every finished run.
- Order: adamw, then muon, then sgd; within an optimizer by M, batch 256 before 512.
- Test: smoke scale on CPU finishes and yields a 12-row summary; a second invocation does no training.

### Task 7: aggregation

- `aggregate <dirs...> [--out DIR]` writes `thresholds.csv`, `per_seed.csv`, and figures.
- Tests on synthetic tables: known crossing recovered by log2 interpolation; no crossing gives "not bracketed"; two crossings flagged; a single seed gives no interval but still a crossing; missing rows are listed.

### Task 8: README and verification on this machine

- Full test suite, smoke run on CPU and on the GPU, then a speed measurement to size the preliminary runs.
