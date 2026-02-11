# Full Evaluation Report

**Date:** February 2026  
**Scope:** Tests run, datasets prepared, existing results summarized, and recommendations for full evaluation (more datasets, more baselines, runtimes).

---

## 1. What Was Run (This Session)

### 1.1 Unit tests

- **Command:** `uv run pytest tests/test_scm.py tests/test_model.py -v`
- **Result:** **9 passed** in ~0.8s.
- **Tests:** Schema generation, SCM sampling, interventions, counterfactuals; continuous/categorical encoders, tokenizer, transformer encoder, full model forward.
- **Checkpoint test:** `tests/test_checkpoint.py` was run; **skipped** (no checkpoint file when run in this environment). When `CHECKPOINT_PATH` points to a valid `.pt` file, it loads the model and runs one batch.

### 1.2 Full test and eval script (existing runs)

The script `scripts/run_full_comparison.py` was **not** re-run to completion in this session because the `checkpoints/` directory was empty in the current workspace. Previous runs (with checkpoints present) produced:

- **quick_run** (1 seed, 10 batches, stage_1_basic): Unit tests passed; 19 checkpoints passed sanity tests; eval matrix and comparison (ours vs mean_stub) completed; significance JSON per checkpoint.
- **full_test_20260205_132948** (3 seeds 42/43/44, 200 batches, stage_1_basic): Same structure; full eval matrix with mean±std across seeds in CSV/JSON.

So the **full test and eval pipeline is in place** and has been run successfully when checkpoints exist.

### 1.3 Datasets prepared

- **Synthetic SCM data:** No download. Episodes are generated on-the-fly by the curriculum dataloader (`episodes/dataset.py`) for all evaluation and training. No separate dataset file.
- **IHDP-style data:** A **synthetic IHDP-style CSV** was created so the IHDP pipeline can be run without external files:
  - **Path:** `data/ihdp_synthetic_sample.csv`
  - **Contents:** 500 rows, 25 covariates (x1–x25), binary treatment `t`, outcome `y`, and potential outcomes `y0`, `y1` (for PEHE). Generated with a simple linear model so the loader and `run_ihdp` can be tested.
  - **Usage:** When you have a checkpoint:
    ```bash
    uv run python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/final_model.pt --data data/ihdp_synthetic_sample.csv --output results/ihdp_results.json
    ```
- **Real IHDP:** For results comparable to the literature, you need real IHDP data (e.g. NPCI 1000 replications). Common sources:
  - CEVAE repo: `https://github.com/AMLab-Amsterdam/CEVAE` (datasets/IHDP/csv/).
  - Or search “IHDP dataset causal inference” for CSV/archives. Place CSVs in `data/` and pass `--data data/ihdp_npci_1.csv` (or similar) to `run_ihdp`.

The **IHDP loader** was verified: it correctly loads `data/ihdp_synthetic_sample.csv`, returns support/query shapes and potential outcomes.

---

## 2. Existing Results (From Your Earlier Runs)

These come from `results/quick_run` and `results/full_test_20260205_132948` when 19 checkpoints were present.

### 2.1 Checkpoint sanity

- All **19 checkpoints** (checkpoint_step_5000 through 85000, final_model) **passed** the checkpoint load-and-forward test (no NaN/Inf, deltas = counterfactuals − baseline).

### 2.2 Eval matrix (checkpoint × difficulty)

- **quick_run:** 19 checkpoints × 4 stages (stage_0_warmup … stage_3_final), 1 seed, 10 batches. Delta_correlation and other metrics in `results/quick_run/eval_matrix/checkpoint_difficulty_matrix.csv`.
- **full_test_20260205_132948:** Same checkpoints × 4 stages, **3 seeds** (42, 43, 44), **200 batches**; summary rows with mean±std in the same CSV/JSON.

**Representative delta_correlation (from full run summary rows):**

| Checkpoint        | stage_0_warmup | stage_1_basic | stage_2_moderate | stage_3_final |
|-------------------|----------------|---------------|------------------|---------------|
| checkpoint_10000  | 0.18           | 0.13          | 0.04             | 0.05          |
| checkpoint_15000  | 0.19           | 0.15          | 0.05             | 0.04          |
| checkpoint_20000  | 0.18           | 0.08          | 0.05             | -0.03         |
| final_model       | ~0.09          | ~-0.01        | ~0.06            | ~0.02         |

Easier stages (0–1) get higher delta_correlation; stage_3 is noisier. Best stage_1_basic in this run is around **checkpoint_15000** (delta_corr ~0.15).

### 2.3 Comparison protocol (ours vs mean_stub)

- For each checkpoint, `run_protocol` produced `method_ours_seed42.json` and `method_mean_stub_seed42.json` (and summaries). Mean_stub delta_correlation is 0 by construction.
- **Significance:** Bootstrap 95% CI for (ours − mean_stub) on delta_correlation; with one seed the interval is degenerate (e.g. mean=0.13, p=1). With 3+ seeds, run `compare_significance` to get meaningful CIs.

**Example (checkpoint_step_15000, stage_1_basic, seed 42):**

- delta_correlation: 0.1315  
- delta_slope: 0.37, delta_intercept: 0.06, sign_accuracy: 0.52  
- pehe: 2.85, calibration_ratio: 0.85, coverage_95: 0.93  

### 2.4 Richer metrics

- The new metrics (delta_slope, delta_intercept, sign_accuracy) are present in the comparison JSONs and in the eval-by-difficulty outputs. They are reported whenever the eval scripts and protocol run.

---

## 3. What You Should Run for a Full Evaluation (With Checkpoints)

With checkpoints in `checkpoints/`, run in this order:

1. **Full test and eval (one command)**  
   ```bash
   uv run python scripts/run_full_comparison.py --output-dir results/full_eval --seeds 42 43 44 --num-batches 200
   ```  
   Approx. time: ~30–60+ min depending on GPU and number of checkpoints (19 × 4 stages × 200 batches × 3 seeds plus comparison).

2. **Full comparison (one command: tests, matrix, protocol all methods, IHDP, ablations, summary table)**  
   ```bash
   uv run python scripts/run_full_comparison.py --output-dir results/full_comparison --ihdp-data data/ihdp_synthetic_sample.csv
   ```  
   Runs unit tests, eval matrix, protocol with **ours, mean_stub, outcome, dr, bart** on SCM stage_1_basic and on IHDP, ablations (no-cross-world, single-world), and writes **full_comparison_summary.json** and **full_comparison_summary.md**. See [EVALUATION.md](EVALUATION.md) §0b.

3. **Comparison protocol (all methods)**  
   ```bash
   uv run python -m experiments.compare.run_protocol --checkpoint checkpoints/checkpoint_step_15000.pt --methods ours,mean_stub,outcome,dr,bart --stage stage_1_basic --output-dir results/compare_full --seeds 42 43 44 --num-batches 200
   ```  
   Produces `method_ours_*`, `method_mean_stub_*`, `method_outcome_*`, `method_dr_*`, `method_bart_*` and summaries.

4. **IHDP**  
   ```bash
   uv run python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/checkpoint_step_15000.pt --data data/ihdp_synthetic_sample.csv --output results/ihdp_results.json
   ```  
   Protocol-style (all methods, multiple seeds): use `--output-dir results/ihdp_protocol --methods ours,mean_stub,outcome,dr,bart --seeds 42 43 44`. For **Kaggle IHDP** (konradb/ihdp-data) see [data/README.md](../data/README.md).

5. **ACIC (optional)**  
   ```bash
   uv run python -m experiments.benchmarks.run_acic --checkpoint checkpoints/checkpoint_step_15000.pt --data path/to/acic.csv --output results/acic_results.json
   ```  
   Data: single CSV (covariates + z, y, mu0, mu1) or directory with `x.csv` and `zymu_*.csv`. See [EVALUATION.md](EVALUATION.md) §4c.

6. **Ablations (optional)**  
   - No cross-world: add `--ablate-no-cross-world` to `run_checkpoint_by_difficulty` or `run_protocol`.  
   - Single-world: add `--single-world` to `run_checkpoint_by_difficulty`.  
   Run on one checkpoint and compare metrics to the same checkpoint without flags.

---

## 4. Do We Need More Datasets?

**Short answer:** For a solid paper and comparability with the literature, **yes** — more benchmarks and more baselines.

### 4.1 Current coverage

- **In-house:** Synthetic SCM episodes (curriculum stages 0–3). No download; infinite variety via seed.
- **Semi-synthetic:** IHDP-style pipeline is implemented; you have a synthetic sample and can add real IHDP.

### 4.2 Datasets supported

| Dataset   | Type            | Treatment | Status |
|----------|-----------------|-----------|--------|
| **IHDP** | Semi-synthetic  | Binary    | Supported; Kaggle (konradb/ihdp-data) with column mapping (y_factual, mu0, mu1). See [data/README.md](../data/README.md). |
| **ACIC** | Semi-synthetic | Binary    | Supported; [experiments/benchmarks/acic_data.py](../experiments/benchmarks/acic_data.py) and [run_acic.py](../experiments/benchmarks/run_acic.py). CSV or directory (x.csv + zymu_*.csv). |
| **Twins**| Observational   | Binary    | Not yet; would require loader producing support/query with W=2. |
| **News** | Observational   | Continuous| Not yet; would require binning or continuous-treatment adapter. |

**Practical order:** (1) Use **real IHDP** or Kaggle IHDP with `run_ihdp`; (2) use **ACIC** with `run_acic` for a second benchmark; (3) Twins/News if you add loaders later.

---

## 5. Do We Need More Models to Compare?

**Short answer:** **Yes.** We now compare to **mean_stub**, **outcome** (ridge), **dr** (doubly-robust style), and **bart** (tree ensemble). For credibility you can add Do-PFN if code is available.

### 5.1 Current baselines

- **mean_stub:** Predicts mean(support_y); delta_corr = 0. Trivial.
- **outcome:** Ridge regression on support (x,y); predicts E[Y|X] per world. Non-trivial but not causal.
- **dr:** Linear doubly-robust style: outcome model per treatment on support; predict per world at query. [experiments/baselines/dr_baseline.py](../experiments/baselines/dr_baseline.py).
- **bart:** Tree ensemble (GradientBoostingRegressor) per treatment on support; predict per world at query. [experiments/baselines/bart_baseline.py](../experiments/baselines/bart_baseline.py).

All are registered in `run_protocol`; use `--methods ours,mean_stub,outcome,dr,bart`.

### 5.2 Optional future baselines

| Method   | Type              | Effort | Why |
|----------|-------------------|--------|-----|
| **Do-PFN** | Causal in-context | High  | Same “in-context” family; direct competitor. |
| **CFR** / **TARNet** | Representation learning | Medium | Often used in causal ML benchmarks. |

**Practical order:** Use **dr** and **bart** for comparison now; add Do-PFN adapter if external code is available.

---

## 6. Approximate Runtimes

Rough estimates on a single GPU (e.g. one mid-range CUDA device), 19 checkpoints:

| Task | Config | Approx. time |
|------|--------|--------------|
| Unit tests (SCM + model) | default | &lt; 1 min |
| Checkpoint tests | 19 checkpoints | ~2–5 min |
| Eval matrix | 19 × 4 stages, 1 seed, 20 batches | ~15–30 min |
| Eval matrix | 19 × 4 stages, 3 seeds, 200 batches | ~1–3 h |
| Comparison (ours + mean_stub) | 19 checkpoints, 1 seed, 10 batches | ~10–20 min |
| Comparison (ours + mean_stub + outcome) | 1 checkpoint, 3 seeds, 200 batches | ~5–10 min |
| IHDP run_ihdp | 1 checkpoint, 1 CSV | &lt; 1 min |
| Full script (all phases) | 19 ckpts, 3 seeds, 200 batches | ~1.5–4 h |
| Full comparison script | unit tests + matrix + protocol (all methods) + IHDP + ablations | ~2–5 h |

Times scale roughly linearly with num_batches and number of checkpoints; multi-GPU is not used by the current eval scripts.

---

## 7. Summary

- **Tests:** Unit tests (SCM + model) **passed**. Checkpoint tests and full test script have been run successfully in the past with 19 checkpoints; they were not re-run here because `checkpoints/` was empty.
- **Datasets:** Synthetic SCM is built-in. **IHDP-style** data: `data/ihdp_synthetic_sample.csv` created and IHDP loader verified. For full evaluation, add **real IHDP** (and optionally ACIC/Twins/News).
- **Results:** Existing results (quick_run and full_test_20260205_132948) show all checkpoints passing, best stage_1_basic delta_correlation around **checkpoint_15000** (~0.15), and richer metrics (slope, intercept, sign_accuracy) available in the JSON outputs.
- **Next steps for full evaluation:** Run **scripts/run_full_comparison.py** for a single-command full comparison (unit tests, eval matrix, protocol with ours/mean_stub/outcome/dr/bart on SCM + IHDP, ablations, summary table). Alternatively run the full test script and comparison with `--methods ours,mean_stub,outcome,dr,bart`; run IHDP (and optionally ACIC) with real or synthetic data. Runtimes are on the order of **1–4 hours** for the full suite with 19 checkpoints and 3 seeds; full comparison script about **2–5 hours** on one GPU.
