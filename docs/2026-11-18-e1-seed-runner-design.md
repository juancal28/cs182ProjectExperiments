# E1 seed runner: design

Date: 2026-11-18. Status: awaiting Juan's approval.

Items marked **(my call)** were chosen by Claude, not decided by Juan or stated in the proposal. Each can be overruled.

## Intent

Experiment E1 of the proposal (`project/proposal/cs182_proposal.pdf`) measures the task-diversity threshold M\* under SGD, AdamW and Muon. Each of the four teammates runs E1 for one seed, on a runner they built themselves; the team will then compare runners and standardize on one.

This is Juan's runner. It takes a seed and produces that seed's complete copy of the E1 grid. It must be good enough to be the team standard: faithful to the reference recipe, checkable, and easy for someone else to run.

## Goals

1. One command with a seed runs the whole E1 grid for that seed: 9 pool sizes × 3 optimizers × 2 batch sizes = 54 training runs.
2. The seed fully determines every random input. The same seed on the same machine and software gives identical results.
3. Within a seed, the task pool and the initial weights are shared across optimizers and batch sizes, so those are the only things that differ between runs.
4. It runs unchanged on Juan's Windows machine (one RTX 4070 Ti) and on a multi-GPU AWS Linux instance, using every GPU it finds.
5. It can be stopped and restarted at any point without losing or repeating finished work.
6. A smoke mode exercises the entire pipeline in under 5 minutes on this machine's CPU.
7. It writes one small results table per seed in a fixed format, and an aggregation step turns any number of such tables into M\* per optimizer.
8. It matches the Raventós et al. recipe closely enough to reproduce their Adam threshold, and that is checked before any optimizer comparison is trusted.

## Non-goals

- E2's fresh-task runs, E3 controls, E4 probes, and a learning-rate tuning harness. The single-run command accepts overrides that make those possible later, but nothing is built for them.
- JAX, or bit-identical results across different hardware.
- Automating AWS provisioning, or coordinating one seed across several machines at once.
- Experiment tracking services such as wandb.

## Anti-goals

- Any randomness that the seed does not control.
- Silent departures from the reference recipe. Every deliberate difference is listed in this document and in the README.
- Results that cannot be traced to the exact settings that produced them.
- A runner only Juan can operate.

## Constraints

- This machine: Windows 11, Python 3.11, RTX 4070 Ti (12 GB). The installed PyTorch 2.15 is the CPU-only build, so a CUDA build is needed. Install it in a virtual environment under `project/code/` rather than changing the system Python **(my call)**.
- PyTorch 2.15 includes `torch.optim.Muon`. It accepts only 2-D parameters and its default weight decay is 0.1, so weight decay must be set to 0 explicitly.
- By the proposal's budget one seed is about 61 A100-hours. That figure was estimated for the JAX code; PyTorch speed is unmeasured.
- Each optimizer's learning rate comes from a pilot the team has not confirmed. Only AdamW has a reference value (1e-3).

## Approaches considered

**A. PyTorch runner copying the Raventós recipe (chosen).** Runs on this machine after one install and unchanged on AWS. Muon is built in. An independent implementation agreeing with a teammate's JAX fork is stronger evidence than several forks of one codebase. Cost: it departs from the proposal's "we build on the JAX code of [2]", and it inherits no validation, so it must reproduce the Adam threshold itself.

**B. Fork the Raventós JAX code.** It is what the proposal promised and reuses the authors' model, data and baselines. Cost: the repo pins 2023 versions (jax 0.4.8, optax 0.1.4, which predates Muon), so the dependency stack must be upgraded; JAX has no GPU support on native Windows and this machine has no Linux distribution under WSL.

Juan chose A, with the runner built so one command fans out over all available GPUs, because AWS can then be used for speed.

## The recipe to match

Source: `github.com/mansheej/icl-task-diversity` (Apache-2.0). The builder should read `icl/gpt2.py`, `icl/models.py`, `icl/tasks.py`, `icl/optim.py` and `icl/evaluate.py` and match them, including details not listed here such as which layers have biases.

| Part | Value |
|---|---|
| Data | D = 8, K = 16 points, noise standard deviation 0.5, inputs and tasks standard Gaussian |
| Task pool | M vectors drawn once; each training sequence picks one uniformly with replacement |
| Tokens | Dimension D + 1. x-token is `[0, x]`, y-token is `[y, 0, …, 0]`, interleaved to length 2K (checked against `icl/utils.py`) |
| Model | Linear read-in without bias, learned positional embedding, pre-LayerNorm GPT-2 blocks with biases on the attention and MLP layers, causal mask, MLP 4× width with tanh-approximate GELU, no dropout, final LayerNorm (eps 1e-5), linear read-out to 1 without bias, predictions read at x-token positions |
| Size | 4 layers, width 64, 2 heads (the proposal's small model; the repo only ships 8 layers, width 128) |
| Init | Normal with standard deviation 0.02; residual output projections 0.02 / sqrt(2 × layers) |
| Loss | Mean squared error over all K positions |
| Schedule | Triangle: linear 0 to peak over 250K steps, linear back to 0 over 250K steps |
| Length | 500K steps, float32 |
| Baseline optimizer | Adam, peak 1e-3, betas (0.9, 0.999), eps 1e-8, no weight decay |

The model is written from scratch in the nanoGPT style, not with `nn.TransformerEncoderLayer`, so parameter names, biases and init can mirror the reference.

## The grid

- M ∈ {512, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 32768}, the proposal's powers of two with half-powers rounded to the nearest integer.
- Optimizers: SGD, AdamW, Muon.
- Batch sizes 256 and 512, both for 500K steps, both at the same learning rate **(my call**; the proposal says only "the same number of steps").

## Optimizers

- **AdamW:** `torch.optim.AdamW`, weight decay 0, otherwise the baseline row above. With zero decay this is Adam.
- **SGD:** momentum 0.9, no Nesterov, weight decay 0 **(my call)**.
- **Muon:** `torch.optim.Muon` with weight decay 0 and library defaults otherwise, on the 2-D weight matrices inside the transformer blocks (attention and MLP). AdamW handles everything else: read-in, positional embedding, read-out, LayerNorm parameters and biases. The combined query/key/value matrix is treated as one matrix, as in the reference layout **(my call)**.
- All three use the same triangle schedule shape.
- Learning rates live in one settings file. Muon has two: one for its matrices and one for the AdamW part. Values without a pilot behind them are marked untuned, and that mark is carried into every result row.

## What the seed controls

Every random number comes from a generator whose seed is a stable hash (SHA-256, not Python's `hash`) of the user's seed, a stream name, and where relevant a step or block index. All generation happens on the CPU so the inputs are identical on any machine.

| Stream | Depends on | Shared across |
|---|---|---|
| Task pool | seed | Everything. One pool of 32,768 vectors per seed; a run with pool size M uses the first M, so smaller pools are subsets of larger ones **(my call)** |
| Initial weights | seed | All 54 runs |
| Training inputs, noise, task picks | seed, batch size, step | Optimizers and pool sizes. The pool size enters only when a task pick is mapped to an index |
| Evaluation prompts | seed | All 54 runs |

Because training data is a pure function of the step, resuming needs no saved generator state. The builder may generate data in blocks for speed as long as this property holds.

What is promised: identical inputs everywhere, and identical results on the same machine and software. Across different GPUs, results agree only up to floating-point drift.

## Evaluation and logging

- Two fixed evaluation sets of 8,192 prompts each **(my call)**: seen tasks (drawn from the run's pool) and new tasks (fresh Gaussian tasks).
- Ridge (penalty 0.25) and dMMSE (over the run's pool) predictions are computed once per run on both sets and cached. dMMSE must be chunked to fit memory at M = 32,768.
- Every 1,000 steps, and at the final step, log for both sets: the squared gap between model and ridge, between model and dMMSE, and the model's, ridge's and dMMSE's error against the true targets. All are averaged over positions and divided by D, as in the reference.
- Logging through training is required because E2 reads memorization time from these runs.
- Reported values are those at the final step, where the learning rate has reached zero.

## Commands

Names are illustrative; the builder picks the final spelling.

```
run_seed --seed 7                      # all 54 runs, skipping finished ones
run_seed --seed 7 --scale smoke        # minutes-long version of the same pipeline
run_seed --seed 7 --opt adamw          # filters: --opt, --batch, --M
run_seed --seed 7 --workers-per-gpu 2  # several runs per GPU
run_one  --seed 7 --M 2896 --opt muon --batch 256 [--lr ...] [--steps ...]
aggregate <results folder(s)>          # any number of seeds
```

- `run_seed` starts one driver that hands unfinished runs to one worker per GPU (or per requested slot). On a machine without a GPU it uses the CPU.
- AdamW runs go first, so the replication check below is available earliest **(my call)**.
- `run_one` accepts `--M inf` for a fresh task per sequence, which is what the pilot and E2 need **(my call)**.

## Outputs

```
project/results/e1/seed_0007/
  summary.csv                    # 54 rows; the file sent to teammates
  adamw_M02896_b256/
    config.json                  # every setting, plus a hash of them
    log.jsonl                    # one line per evaluation
    checkpoint.pt                # for resuming; overwritten every 10K steps
    model.pt                     # final weights
    final.json                   # written last; its presence means "done"
```

- Files are written atomically (write to a temporary name, then rename).
- `summary.csv` columns: seed, optimizer, M, log2 M, batch size, learning rate(s), tuned/untuned, steps, the four final gaps (ridge and dMMSE on seen and new tasks), the final errors, the first step at which the seen-task gap to dMMSE fell below the gap to ridge, settings hash, git commit, PyTorch version, device name, wall-clock time.
- A resumed run must give the same log as an uninterrupted one on the same machine.
- Checkpoints and weights are excluded from git; summaries and logs are small enough to keep.

## Aggregation

For each optimizer, from all seeds supplied:

- s(M) = (gap to ridge on new tasks at batch 512) − (same at batch 256), paired by seed.
- M\* is where the seed-averaged s crosses from positive to negative, by linear interpolation in log2 M between adjacent grid points.
- Bootstrap interval by resampling seeds, when there are at least two.
- The single-run readout: the M at which the seen-task gaps to dMMSE and to ridge are equal, per batch size.
- A per-seed table of outcomes, since results near the threshold may be bimodal.
- Figures of s(M) and of the gaps against log2 M, with individual seeds visible.

Known weakness of the proposal's statistic: far above the threshold both models sit at ridge, so s is near zero and its sign is noise. The aggregator reports every sign change, flags more than one, and says "not bracketed" when there is none.

It reads only `summary.csv`, so a teammate's runner can be compared by producing the same columns.

## How we know it is right

1. **Baselines:** ridge against an independent solve; dMMSE with M = 1 returns that task's prediction; dMMSE beats ridge on seen tasks and loses on new ones; no prediction depends on later examples (also tested for the model).
2. **Seed behavior:** the same seed twice gives identical pool, weights, data and logs; different optimizers and batch sizes share pool and weights; a different seed changes all of them; pools are nested; stop-and-resume equals uninterrupted.
3. **Smoke run:** the full pipeline, including aggregation, at toy scale.
4. **Timing run:** one full AdamW run at M = 32,768, batch 256. It passes if the model ends closer to ridge than to dMMSE on new tasks **(my call)**. It also gives the real hours per run on this GPU.
5. **Bracket:** AdamW at M = 512 passes if the model ends closer to dMMSE than to ridge on seen tasks **(my call)**.
6. **Replication gate:** the AdamW block of the seed (18 runs) passes if s(M) changes sign from positive to negative between 2^11 and 2^12 **(my call)**. The design reviewer reports that Raventós et al. give that range for this model size; Claude has not checked the paper itself.

One seed cannot tell an implementation error from seed noise, and outcomes near the threshold may be bimodal. So a failed gate is reported, not automatically treated as a bug: the builder stops, shows Juan the per-run numbers, and Juan and the team decide whether to debug, add AdamW seeds near the threshold, or proceed. No extra compute is planned for that in advance **(my call)**.

## Scope of the build

- **The build is done** when the code, the tests and the smoke run pass on this machine, and the README covers local and AWS use.
- **Real compute is Juan's to launch.** The timing run, the AdamW block and the rest of the seed each start only on his say-so, and only after he has accepted or overruled the "(my call)" items, because changing those later invalidates finished runs.
- **Local or AWS:** if the timing run implies the full seed would take more than about three days of continuous running here, use AWS; otherwise run it here **(my call)**. A deadline from the team overrides this.
- **Throughput work** (several runs per GPU, compilation, block data generation) is done after the timing run and only if the measured speed calls for it. No deadline has been given; a tight one would move this work into the first build.
- **SGD and Muon blocks** wait for both the replication gate and agreed learning rates.

## Order of work for Juan's seed

1. Set up the environment with a CUDA build of PyTorch.
2. Tests and smoke run on this machine.
3. Timing run; decide local or AWS.
4. AdamW block, then the replication gate.
5. Muon and SGD blocks, once their learning rates are agreed.
6. Send `summary.csv` to the team.

On AWS the steps are: start a GPU instance, copy the code, install requirements, run the same `run_seed` command in a session that survives disconnects, and copy the seed folder back. The README documents this; nothing is automated.

## Build notes (2026-11-18)

Things found while building that change statements above:

- **PyTorch version:** Windows Smart App Control blocked the 2.15.0 CUDA 13.2 build, so the environment uses 2.14.1 for CUDA 12.6.
- **Several runs per GPU is off on this machine:** six training processes sharing the RTX 4070 Ti crashed the graphics driver (bugcheck 0x119). The driver refuses more than one worker per GPU on Windows unless told otherwise.
- **Measured speed:** about 106 steps per second for AdamW at batch 256, one run at a time. A full-length run is about 80 minutes and the full seed about three days here.
- **AWS:** Juan's project has $100 in credits and a GPU quota of zero in us-east-2, so it cannot run this until a quota increase is approved.
- **Preliminary scale:** a `prelim` preset runs the same grid at 20,000 steps per run. Its thresholds are not comparable to the full experiment's.
- **Attention is written out by hand:** PyTorch's fused attention has a non-deterministic backward pass on CUDA.
- **SGD at learning rate 0.1 diverges;** the placeholder is now 0.01, still untuned.

## Left to the builder

Module layout, test framework, speed levers (block data generation, compilation, runs per GPU), the exact smoke-mode numbers, plotting details, and the bootstrap implementation.
