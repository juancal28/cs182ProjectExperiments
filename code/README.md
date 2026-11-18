# E1 seed runner

Runs experiment E1 of the proposal (the task-diversity threshold M\* under AdamW, Muon and SGD) for one seed, and turns seed tables into M\*. It is a PyTorch port of the Raventós et al. recipe (`github.com/mansheej/icl-task-diversity`). Design: `../docs/2026-11-18-e1-seed-runner-design.md`.

## Use

Run everything from this folder (`project/code`).

```
python -m e1.run_seed --seed 7                 # the full grid: 54 runs of 500,000 steps
python -m e1.run_seed --seed 7 --scale prelim  # same grid, 20,000 steps per run (about 3.5 hours on an RTX 4070 Ti)
python -m e1.run_seed --seed 7 --scale smoke   # 12 tiny runs; checks the whole pipeline in minutes
python -m e1.run_seed --seed 7 --opt adamw     # filters: --opt, --batch, --M
python -m e1.aggregate ../results/e1/full      # M* per optimizer from every seed found there
python -m pytest tests -q                      # the test suite (CPU, about a minute)
```

- **Stopping and resuming:** stop it any time; the same command continues from the last checkpoint and skips finished runs.
- **Results:** `../results/e1/<scale>/seed_<NNNN>/summary.csv` is the table to share (one row per run). Each run folder also holds `log.jsonl` (metrics through training), `config.json` and the final weights.
- **One run with overrides** (for a learning-rate pilot or a fresh-task run): `python -m e1.run_one --seed 7 --M inf --opt sgd --lr 0.03 --steps 20000 --tag pilot`.

## What the seed controls

The seed fixes the task pool, the initial weights, the training data stream and the evaluation prompts. Within a seed, all 54 runs share the pool (a run with pool size M uses the first M of 32,768 tasks) and the initial weights. Inputs are generated on the CPU, so they are the same on every machine. The same seed on the same machine and software reproduces the same numbers; across different GPUs, results agree only up to floating-point drift.

## Learning rates

They live in `e1/config.py` (`LEARNING_RATES`). Only AdamW has a reference value (1e-3). Muon's value (0.005 for the matrices, 1e-3 for the rest) is the best of three tried in a 20,000-step check and is marked `quick-sweep`. SGD is marked `untuned`: in that check it learned nothing at 0.003 to 0.1 and diverged at 0.3 and 1.0. The mark is written into every result row. Change the values there once the team agrees on values; a changed value produces a different settings hash, and the runner refuses to mix it into a folder of older runs.

## Differences from the reference code

- PyTorch instead of JAX; random streams therefore differ from the original, by design of this runner.
- 4 layers, width 64 (the proposal's small model) instead of the repository's 8 layers, width 128.
- 8,192 fixed evaluation prompts per set instead of 1,048,576.
- One pool of 32,768 tasks per seed, with smaller pools as prefixes of it.
- dMMSE is also evaluated on new tasks.

## Setup on Windows (this machine)

```
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m e1.run_seed --seed 7 --scale smoke
```

- Windows Smart App Control blocked the PyTorch 2.15.0 CUDA 13.2 build here; the 2.14.1 CUDA 12.6 build is allowed.
- Run one training process per GPU on Windows. Six runs sharing the RTX 4070 Ti crashed the graphics driver (bugcheck 0x119), so `--workers` above 1 on a GPU is refused unless `--allow-gpu-sharing` is given.
- Measured speed on the RTX 4070 Ti, one run at a time: about 105 steps per second for AdamW and 60 for Muon, at either batch size.

## Setup on a Linux GPU machine (for example AWS)

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
tmux new -s e1
python -m e1.run_seed --seed 7        # uses every GPU it finds, one worker per GPU
```

Copy `../results/e1/full/seed_0007/` back when it finishes. `--workers N` runs N processes per GPU; that has not been tested on Linux.
