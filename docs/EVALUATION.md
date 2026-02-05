# Evaluation Suite

How to run unit tests, eval-by-difficulty, checkpoint × difficulty matrix, synthetic benchmark suite, and the comparison protocol. Where results live and how they map to "Option A" vs "Option B" report tables.

## 1. Unit tests (checkpoint load and forward)

- **Purpose:** Sanity check: load checkpoint, build model from config, run one batch, assert shapes and delta consistency (no NaN/Inf, deltas = counterfactuals − baseline).
- **Run:**
  ```bash
  # With a checkpoint available (skips if unset or missing):
  CHECKPOINT_PATH=checkpoints/checkpoint_step_5000.pt pytest tests/test_checkpoint.py -v

  # Without checkpoint (test is skipped):
  pytest tests/test_checkpoint.py -v
  ```
- **Files:** [tests/test_checkpoint.py](../tests/test_checkpoint.py)

## 2. Single-checkpoint eval by difficulty

- **Purpose:** Evaluate one or more checkpoints on fixed eval sets per curriculum difficulty (stage_0–like … stage_3–like). Reports full metrics: baseline RMSE/MAE/R², CF RMSE/MAE, delta RMSE/MAE/**delta_correlation**, ATE MAE, calibration.
- **Note on PEHE:** Many treatment-effect papers report **PEHE** (mean squared error of individual treatment effects). In our outputs, `delta_rmse` is the **RMSE of individual deltas**, so \(\\text{delta\\_rmse} = \\sqrt{\\text{PEHE}}\\). We also output `pehe = delta_rmse**2` for drop-in comparability.
- **Run:**
  ```bash
  python -m experiments.eval.run_checkpoint_by_difficulty \
    --checkpoint checkpoints/checkpoint_step_20000.pt checkpoints/final_model.pt \
    --output-dir results \
    --seeds 42 43 44 \
    --num-batches 200
  ```
- **Output:** `results/<checkpoint_stem>_<stage_name>.json` per (checkpoint, difficulty).
- **Output (multi-seed):** `results/<checkpoint_stem>_<stage_name>_seed<seed>.json` plus `results/<checkpoint_stem>_<stage_name>_summary.json` with mean±std across seeds.
- **Files:** [experiments/eval/run_checkpoint_by_difficulty.py](../experiments/eval/run_checkpoint_by_difficulty.py)

## 3. Checkpoint × difficulty matrix

- **Purpose:** Run evaluation for all checkpoints (in a directory or list) × all curriculum difficulties; aggregate into one table (CSV + JSON).
- **Run:**
  ```bash
  # Scan checkpoints/ for *.pt:
  python -m experiments.eval.run_all_checkpoints_matrix \
    --checkpoint-dir checkpoints \
    --output-dir results \
    --seeds 42 43 44 \
    --num-batches 200

  # Or explicit checkpoint list:
  python -m experiments.eval.run_all_checkpoints_matrix \
    --checkpoint checkpoints/checkpoint_step_5000.pt checkpoints/final_model.pt \
    --output-dir results
  ```
- **Output:** `results/checkpoint_difficulty_matrix.csv`, `results/checkpoint_difficulty_matrix.json`. Rows = (checkpoint, difficulty); columns = delta_correlation, delta_mae, baseline_mae, etc.
- **Output (multi-seed):** Same files, but include a `seed` column with per-seed rows plus `seed=summary` rows containing `*_mean` and `*_std` columns.
- **Files:** [experiments/eval/run_all_checkpoints_matrix.py](../experiments/eval/run_all_checkpoints_matrix.py)

## 4. Synthetic benchmark suite (SCM families)

- **Purpose:** Evaluate one checkpoint on multiple SCM types (linear_gaussian, nonlinear_additive, multiplicative, etc.). Reports baseline, CF, delta, **delta_correlation**, ATE.
- **Run:**
  ```bash
  python -m experiments.benchmarks.synthetic_suite \
    --checkpoint checkpoints/final_model.pt \
    --output benchmark_results.json
  ```
- **Output:** `benchmark_results.json` (or path given by `--output`).
- **Files:** [experiments/benchmarks/synthetic_suite.py](../experiments/benchmarks/synthetic_suite.py)

## 5. Comparison protocol (ours vs baseline)

- **Purpose:** Same eval data and same metrics for "our model" and a baseline (stub or Do-PFN). One JSON per method so you can compare fairly.
- **Run:**
  ```bash
  python -m experiments.compare.run_protocol \
    --checkpoint checkpoints/final_model.pt \
    --stage stage_1_basic \
    --output-dir results \
    --seeds 42 43 44 \
    --num-batches 200
  ```
- **Output:** `results/method_ours_seed<seed>.json`, `results/method_baseline_seed<seed>.json` plus `results/method_ours_summary.json` (and `results/method_baseline_summary.json`).
- **Files:** [experiments/compare/run_protocol.py](../experiments/compare/run_protocol.py), [experiments/compare/protocol.py](../experiments/compare/protocol.py), [experiments/baselines/stub.py](../experiments/baselines/stub.py)

## Where results live

- **Eval-by-difficulty:** `results/<checkpoint_stem>_<stage_name>.json` (default `--output-dir results`).
- **Matrix:** `results/checkpoint_difficulty_matrix.csv`, `results/checkpoint_difficulty_matrix.json`.
- **Benchmark suite:** Path given by `--output` (e.g. `benchmark_results.json`).
- **Comparison protocol:** `results/method_ours.json`, `results/method_baseline.json`.

## Mapping to report tables

- **Option A (same category):** Use comparison protocol outputs. Table: "Causal in-context methods on [protocol name]" — columns: Method (Ours, Do-PFN, …), Baseline R², CF RMSE, Δ_corr, ATE MAE. Fill from `method_ours.json` and `method_baseline.json` (or Do-PFN JSON when added).
- **Option B (new category):** Use checkpoint × difficulty matrix. Table: "Many-intervention regime (K=4,6,8)" — columns: Method, K=4 Δ_corr, K=6 Δ_corr, K=8 Δ_corr, One-call? (Y/N). Fill from `checkpoint_difficulty_matrix.csv` (rows for your checkpoint(s)); baseline/Do-PFN can be N/A or added later.

Use the same metric names in scripts and report so tables are reproducible from this evaluation suite.
