# Evaluation Suite

How to run unit tests, eval-by-difficulty, checkpoint × difficulty matrix, synthetic benchmark suite, and the comparison protocol. Where results live and how they map to "Option A" vs "Option B" report tables.

## 0. Full comparison (one command)

- **Purpose:** Run unit tests, eval matrix, comparison protocol with **all methods** (ours, mean_stub, outcome, dr, bart) on SCM and IHDP, ablations (no-cross-world, single-world), and write a summary table. Optional: checkpoint sanity tests per checkpoint; per-checkpoint × per-stage comparison with significance.
- **Run:**
  ```bash
  # Default: checkpoints/ scanned, results under results/full_comparison_<timestamp>
  uv run python code/scripts/run_full_comparison.py

  # Custom output dir
  uv run python code/scripts/run_full_comparison.py --output-dir results/my_run

  # With IHDP data (optional)
  uv run python code/scripts/run_full_comparison.py --ihdp-data data/ihdp.csv

  # Full eval: checkpoint tests + per-checkpoint comparison with significance
  uv run python code/scripts/run_full_comparison.py --checkpoint-tests --compare-all-checkpoints --stages all

  # Quick run (fewer batches, one seed, skips IHDP/ablations)
  uv run python code/scripts/run_full_comparison.py --quick
  ```
- **Options:** `--device cuda`, `--seeds 42 43 44`, `--num-batches 200`, `--stages stage_1_basic` (or `all`). `--checkpoint-tests` runs pytest test_checkpoint per checkpoint. `--compare-all-checkpoints` runs protocol per checkpoint × stage with significance. `--include-ihdp` enables IHDP (default when not `--quick`).
- **Output:** Under `--output-dir`: `unit_tests.log`, `eval_matrix/`, `protocol_scm/`, `protocol_ihdp/` (if run), `ablations/`, `compare/<checkpoint>/` (if `--compare-all-checkpoints`), **full_comparison_summary.json** and **full_comparison_summary.md**.
- **Files:** [code/scripts/run_full_comparison.py](../code/scripts/run_full_comparison.py). One-page guide: [REPRODUCE.md](REPRODUCE.md).

## 1. Unit tests (checkpoint load and forward)

- **Purpose:** Sanity check: load checkpoint, build model from config, run one batch, assert shapes and delta consistency (no NaN/Inf, deltas = counterfactuals − baseline).
- **Run:**
  ```bash
  # With a checkpoint available (skips if unset or missing):
  CHECKPOINT_PATH=checkpoints/checkpoint_step_5000.pt pytest test/test_checkpoint.py -v

  # Without checkpoint (test is skipped):
  pytest test/test_checkpoint.py -v
  ```
- **Files:** [test/test_checkpoint.py](../test/test_checkpoint.py)

## 2. Single-checkpoint eval by difficulty

- **Purpose:** Evaluate one or more checkpoints on fixed eval sets per curriculum difficulty (stage_0–like … stage_3–like). Reports full metrics: baseline RMSE/MAE/R², CF RMSE/MAE, delta RMSE/MAE/**delta_correlation**, ATE MAE, calibration.
- **Note on PEHE:** Many treatment-effect papers report **PEHE** (mean squared error of individual treatment effects). In our outputs, `delta_rmse` is the **RMSE of individual deltas**, so \(\\text{delta\\_rmse} = \\sqrt{\\text{PEHE}}\\). We also output `pehe = delta_rmse**2` for drop-in comparability.
- **Richer metrics (delta calibration, sign accuracy):** Scripts that use [train/metrics.py](../code/train/metrics.py) also report:
  - **delta_slope**, **delta_intercept**: regression of true deltas on predicted deltas (ideal: slope=1, intercept=0).
  - **delta_r2**: R² of that regression.
  - **sign_accuracy**: fraction of (batch, world, query) where sign(predicted_delta) == sign(true_delta). Useful when ranking interventions by effect sign.
  These appear in eval-by-difficulty JSON, checkpoint matrix, and comparison protocol outputs. Interpret slope/intercept as calibration of effect magnitude; sign_accuracy as a simple measure of correct effect direction.
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
- **Files:** [experiments/eval/run_checkpoint_by_difficulty.py](../code/experiments/eval/run_checkpoint_by_difficulty.py)

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
- **Files:** [experiments/eval/run_all_checkpoints_matrix.py](../code/experiments/eval/run_all_checkpoints_matrix.py)

## 4. Synthetic benchmark suite (SCM families)

- **Purpose:** Evaluate one checkpoint on multiple SCM types (linear_gaussian, nonlinear_additive, multiplicative, etc.). Reports baseline, CF, delta, **delta_correlation**, ATE.
- **Run:**
  ```bash
  python -m experiments.benchmarks.synthetic_suite \
    --checkpoint checkpoints/final_model.pt \
    --output benchmark_results.json
  ```
- **Output:** `benchmark_results.json` (or path given by `--output`).
- **Files:** [experiments/benchmarks/synthetic_suite.py](../code/experiments/benchmarks/synthetic_suite.py)

## 4b. IHDP (semi-synthetic benchmark)

- **Purpose:** Run the model on a standard benchmark (IHDP: 25 covariates, binary treatment, continuous outcome) so results are comparable to the literature. IHDP is **single-treatment** (W=2: do(T=0), do(T=1)); we run in this reduced setup.
- **Data:** [experiments/benchmarks/ihdp_data.py](../code/experiments/benchmarks/ihdp_data.py) loads from a CSV (path or URL). Expected columns: covariates (e.g. x1–x25 or 25 numeric columns), treatment (`t` or `treatment`), outcome (`y` or `y_factual`). Optional `y0`/`y1` or `mu0`/`mu1` (potential outcomes) for PEHE. **Kaggle IHDP** (e.g. konradb/ihdp-data) is supported: use columns `treatment`, `y_factual`, `mu0`, `mu1`; see [data/README.md](../data/README.md). Train/test split 80/20 by default.
- **Run (single run, one JSON):**
  ```bash
  python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/final_model.pt --data path/to/ihdp.csv --output results/ihdp_results.json
  ```
- **Run (protocol-style, all methods, multiple seeds):**
  ```bash
  python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/final_model.pt --data path/to/ihdp.csv --output-dir results/ihdp_protocol --methods ours,mean_stub,outcome,dr,bart --seeds 42 43 44
  ```
- **Output:** delta_correlation, delta_rmse (= √PEHE), pehe, and (if y0/y1 or mu0/mu1 present) pehe_ite; optionally outcome baseline. With `--output-dir`: `method_<name>_seed<N>.json` and `method_<name>_summary.json` per method.
- **Caveats:** IHDP is single-treatment; our model is trained for many interventions. Interpret as conditional effect estimates in a two-world setup. See PROJECT_REPORT limitations.
- **Files:** [experiments/benchmarks/run_ihdp.py](../code/experiments/benchmarks/run_ihdp.py), [experiments/benchmarks/ihdp_data.py](../code/experiments/benchmarks/ihdp_data.py).

## 4c. ACIC (semi-synthetic benchmark, optional)

- **Purpose:** Run on ACIC 2016-style data (binary treatment, continuous outcome, many covariates) for an extra benchmark comparable to the literature.
- **Data:** [experiments/benchmarks/acic_data.py](../code/experiments/benchmarks/acic_data.py). Path can be (1) a **single CSV** with columns: covariates (x1..xk or numeric), `z` (treatment), `y` (outcome), `mu0`, `mu1` (optional potential outcomes), or (2) a **directory** with `x.csv` (covariates) and `zymu_*.csv` (z, y, mu0, mu1).
- **Run:**
  ```bash
  python -m experiments.benchmarks.run_acic --checkpoint checkpoints/final_model.pt --data path/to/acic.csv --output results/acic_results.json
  # Or directory with x.csv + zymu_*.csv:
  python -m experiments.benchmarks.run_acic --checkpoint checkpoints/final_model.pt --data path/to/acic_dir --output results/acic_results.json
  ```
- **Output:** Same metrics as IHDP (delta_correlation, delta_rmse, pehe, pehe_ite if mu0/mu1 present); optionally outcome baseline.
- **Files:** [experiments/benchmarks/run_acic.py](../code/experiments/benchmarks/run_acic.py), [experiments/benchmarks/acic_data.py](../code/experiments/benchmarks/acic_data.py).

## 5. Comparison protocol (ours vs baselines)

- **Purpose:** Same eval data and same metrics for our model and one or more baselines. One JSON per method per seed so you can compare fairly. Methods share the interface: (support_x, support_y, query_x, …) → prediction [B,W,Nq], log_var [B,W,Nq].
- **Methods:**
  - **ours** — Parallel-universe transformer (requires `--checkpoint`).
  - **mean_stub** — Predicts mean(support_y) for every query; no use of covariates ([experiments/baselines/stub.py](../code/experiments/baselines/stub.py)).
  - **outcome** — Ridge regression fit on (support_x, support_y) per batch; predicts E[Y|X] at each world’s query covariates ([experiments/baselines/outcome_baseline.py](../code/experiments/baselines/outcome_baseline.py)).
  - **dr** — Linear doubly-robust style: outcome model per treatment on support; predict per world at query ([experiments/baselines/dr_baseline.py](../code/experiments/baselines/dr_baseline.py)).
  - **bart** — Tree ensemble (GradientBoostingRegressor) per treatment on support; predict per world at query ([experiments/baselines/bart_baseline.py](../code/experiments/baselines/bart_baseline.py)).
- **Run:**
  ```bash
  python -m experiments.compare.run_protocol \
    --checkpoint checkpoints/final_model.pt \
    --methods ours,mean_stub,outcome,dr,bart \
    --stage stage_1_basic \
    --output-dir results \
    --seeds 42 43 44 \
    --num-batches 200
  ```
- **Output:** For each method: `results/method_<name>_seed<seed>.json` and `results/method_<name>_summary.json` (mean±std across seeds).
- **Files:** [experiments/compare/run_protocol.py](../code/experiments/compare/run_protocol.py), [experiments/baselines/stub.py](../code/experiments/baselines/stub.py), [experiments/baselines/outcome_baseline.py](../code/experiments/baselines/outcome_baseline.py), [experiments/baselines/dr_baseline.py](../code/experiments/baselines/dr_baseline.py), [experiments/baselines/bart_baseline.py](../code/experiments/baselines/bart_baseline.py).

## 6. Ablations (no cross-world, single-world)

- **Purpose:** Isolate the benefit of cross-world attention and of batching multiple worlds in one forward.
- **No cross-world:** Evaluate the same checkpoint with cross-world attention disabled at eval time (override `cross_world_layers=[]` when loading). Alternatively, train a checkpoint with `cross_world_layers=[]` (e.g. via [experiments/ablations/run_ablations.py](../code/experiments/ablations/run_ablations.py) ablation `cross_attention`) and run the same eval matrix on it.
  - **Run (eval-time override):**
    ```bash
    python -m experiments.eval.run_checkpoint_by_difficulty --checkpoint checkpoints/final_model.pt --ablate-no-cross-world --output-dir results/ablation_no_cross_world
    python -m experiments.compare.run_protocol --checkpoint checkpoints/final_model.pt --methods ours --ablate-no-cross-world --output-dir results/ablation_no_cross_world
    ```
  - **Output:** Same JSON layout with `_no_cross_world` suffix (e.g. `ckpt_stage_1_basic_no_cross_world_seed42.json`). Compare delta_correlation (and delta_slope, sign_accuracy) to the full model.
- **Single-world per intervention:** Run the model once per intervention (W=2 each time: baseline + one intervention), then aggregate metrics. Same total number of forward passes as the batched run; shows whether batching multiple worlds in one forward helps beyond compute.
  - **Run:**
    ```bash
    python -m experiments.eval.run_checkpoint_by_difficulty --checkpoint checkpoints/final_model.pt --single-world --output-dir results/ablation_single_world
    ```
  - **Output:** JSON with `_single_world` suffix (e.g. `ckpt_stage_1_basic_single_world_seed42.json`). Compare to the same checkpoint without `--single-world`.
- **Files:** [experiments/eval/run_checkpoint_by_difficulty.py](../code/experiments/eval/run_checkpoint_by_difficulty.py), [experiments/compare/run_protocol.py](../code/experiments/compare/run_protocol.py), [model/backbone.py](../code/model/backbone.py) (`cross_world_layers=[]`), [experiments/ablations/run_ablations.py](../code/experiments/ablations/run_ablations.py).

## 7. Optional: long-run curriculum

- **Purpose:** Spend more training steps on harder stages (stage_2_moderate, stage_3_final) to improve delta_correlation on those stages without changing the default curriculum for quick runs.
- **Config:** In [episodes/config.py](../episodes/config.py), `CurriculumConfig.get_long_run_curriculum()` returns stages with increased `min_steps` for stage_2 (40k) and stage_3 (60k). Use it by setting `long_run_curriculum=True` in [train/config.py](../train/config.py) (or when constructing `TrainingConfig`). The trainer then uses this curriculum instead of `get_default_curriculum()`.
- **When to use:** When you want a single longer training run focused on harder stages; compare eval matrix delta_correlation for the same checkpoint with default vs long-run curriculum.

## Where results live

- **Eval-by-difficulty:** `results/<checkpoint_stem>_<stage_name>.json` (default `--output-dir results`).
- **Matrix:** `results/checkpoint_difficulty_matrix.csv`, `results/checkpoint_difficulty_matrix.json`.
- **Benchmark suite:** Path given by `--output` (e.g. `benchmark_results.json`).
- **Comparison protocol:** `results/method_<name>_seed<seed>.json`, `results/method_<name>_summary.json` (e.g. ours, mean_stub, outcome, dr, bart).
- **Full comparison:** `results/full_comparison_<timestamp>/full_comparison_summary.json` and `full_comparison_summary.md` (single table across methods and datasets).

## Mapping to report tables

- **Option A (same category):** Use comparison protocol or full comparison outputs. Table: "Causal in-context methods on [protocol name]" — columns: Method (Ours, mean_stub, outcome, dr, bart), Dataset (stage_1_basic, IHDP), Δ_corr, sqrt(PEHE), delta_slope, sign_acc, ATE MAE, calibration_ratio. Fill from `method_*_summary.json` or **full_comparison_summary.json** / **full_comparison_summary.md**.
- **Option B (new category):** Use checkpoint × difficulty matrix. Table: "Many-intervention regime (K=4,6,8)" — columns: Method, K=4 Δ_corr, K=6 Δ_corr, K=8 Δ_corr, One-call? (Y/N). Fill from `checkpoint_difficulty_matrix.csv` (rows for your checkpoint(s)); baseline/Do-PFN can be N/A or added later.

Use the same metric names in scripts and report so tables are reproducible from this evaluation suite.
