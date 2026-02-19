# Benchmark Implementation Check & Presentation Framing

## 1. Implementation verification

### 1.1 Metrics and definitions (correct)

- **PEHE** (Precision in Estimation of Heterogeneous Effect): mean squared error of individual treatment effects, i.e. \(\mathrm{PEHE} = \mathbb{E}[(\hat{\tau}(X) - \tau(X))^2]\). In this codebase:
  - **delta_rmse** = RMSE of (predicted delta vs true delta) = **sqrt(PEHE)** when both are defined on the same units.
  - **pehe** is set as `delta_rmse**2` in protocol/run_ihdp/run_checkpoint_by_difficulty for compatibility with papers that report PEHE.
- **ATE error**: in `train/metrics.py`, **ate_mae** = |mean(predicted delta) − mean(true delta)|; in `real_world_suite.py` it is **ATE_Err** = |mean(pred_cate) − mean(true_cate)|. Both are correct ATE absolute error.
- IHDP in `real_world_suite`: 80/20 train–test, support = train (X with T, Y), query = test with two worlds (T=0, T=1). PEHE and ATE are computed on the test set. This matches standard IHDP evaluation.

### 1.2 What was fixed

- **test/run_tests.sh**: Uses `uv run python` so dependencies (numpy, torch) are available; added step 3 for checkpoint test (skipped if `CHECKPOINT_PATH` unset).
- **IHDP CSV**: `real_world_suite.IHDPDataset` expects raw CEVAE file (no header) in `data/ihdp/ihdp_npci_1.csv`. The download script writes to `data/ihdp_npci_1.csv` with a header. For the benchmark suite, either use its built-in download (writes to `data/ihdp/`) or ensure the file in `data/ihdp/` is raw (no header). No code change was required for correctness; the suite downloads its own copy from the same CEVAE URL.

### 1.3 Notes for reproducibility

- **Checkpoint**: Your reported “Ours” PEHE (4.40 on IHDP) uses `final_model.pt`. Results can vary with checkpoint (e.g. `checkpoint_step_15000.pt` often does better on synthetic in prior runs). For presentations, state the exact checkpoint and, if possible, report multiple checkpoints or a small sensitivity check.
- **Seeds**: IHDP 80/20 split uses a fixed seed (42) in `real_world_suite`; protocol/run_ihdp use configurable seeds. Use the same seed(s) when comparing methods.
- **Baselines**: TabPFN, TransTEE, Dragonnet, Linear-T/S, GB-T/S, DR-* live in `code/experiments/baselines/` and are invoked from `real_world_suite` and `synthetic_suite`. Ensure the same versions and configs (e.g. device, no extra tuning) when comparing.

---

## 2. Test results (run from repo root)

Commands used:

```bash
uv run python test/test_scm.py
uv run python test/test_model.py
uv run python test/test_checkpoint.py   # skipped if CHECKPOINT_PATH not set
```

**Results:**

| Test suite        | Result  | Notes                                      |
|-------------------|---------|--------------------------------------------|
| SCM components    | **PASS** | Schema, sampling, interventions, counterfactuals |
| Model components  | **PASS** | Encoders, tokenizer, transformer, full forward   |
| Checkpoint        | **SKIP** | Run with `CHECKPOINT_PATH=checkpoints/final_model.pt` to execute |

Or run all at once:

```bash
./test/run_tests.sh
```

---

## 3. How to frame the work for presentations

Your results show that the Parallel Universe Transformer is **weaker** than TabPFN and several classical/ML baselines on IHDP and on some SCMs (e.g. heteroskedastic). Below are concise ways to frame this honestly and constructively.

### 3.1 One-line positioning

- **Option A (honest, research):** “We present an in-context causal model that matches or lags behind strong baselines on standard benchmarks; we analyze failure modes and outline paths to close the gap.”
- **Option B (method focus):** “We introduce a parallel-universe transformer for CATE that uses support/query batching and cross-world attention; we benchmark it against T/S-learners, DR, TabPFN, and TransTEE and discuss when and why it underperforms.”

### 3.2 Slide structure suggestion

1. **Problem**  
   CATE estimation with limited data; need methods that use context (support set) and scale to multiple interventions.

2. **Method**  
   Parallel Universe Transformer: support (X, Y), query (multiple worlds), cross-world attention, single forward pass for all worlds. Emphasize: in-context, no retraining per task.

3. **Experiments**  
   - Synthetic: SCM types (linear, nonlinear, multiplicative, heteroskedastic).  
   - Real: IHDP (standard benchmark).  
   - Baselines: Linear-T/S, GB-T/S, DR-Linear/GB, TabPFN, TransTEE, Dragonnet.  
   - **Multi-intervention**: K=3–15, stable performance; unique capability vs baselines.

4. **Results (your numbers)**  
   - **Synthetic:** Competitive or tied on some SCMs; **heteroskedastic:** ours 2.25 vs TabPFN 1.09 → model struggles with complex noise or needs more tuning.  
   - **IHDP:** Ours 4.40 PEHE vs TabPFN 0.39, TransTEE 0.57, GB-S 0.60 → clear gap; in-context zero-shot does not yet match task-specific or pretrained baselines.

5. **Discussion / Limitations**  
   - Zero-shot in-context may not match distribution of IHDP (different from training SCMs).  
   - Possible next steps: input normalization, feature-type handling, few-shot or light fine-tuning on IHDP-like data, checkpoint selection (e.g. best on validation IHDP or synthetic).

6. **Conclusion**  
   “We proposed a parallel-universe transformer for CATE and ran a comprehensive benchmark. It underperforms TabPFN and strong classical baselines on IHDP and on heteroskedastic SCMs. We identify failure modes and suggest directions (normalization, adaptation, checkpoint selection) for future work.”

### 3.3 Tables for slides (from your numbers)

**Synthetic (PEHE, lower is better):**

| SCM type        | Ours | Linear-T | GB-S  | TabPFN |
|-----------------|------|----------|-------|--------|
| Linear Gaussian | 1.43 | 1.43     | **1.15** | N/A  |
| Nonlinear Add.  | 1.18 | 1.18     | **1.17** | N/A  |
| Multiplicative  | 1.18 | 1.18     | 1.18  | N/A    |
| Heteroskedastic | 2.25 | 1.18     | 1.18  | **1.09** |

**IHDP (PEHE / ATE error):**

| Model   | PEHE  | ATE Err |
|---------|-------|---------|
| **TabPFN** | **0.39** | 0.20  |
| TransTEE | 0.57  | 0.36  |
| GB-S     | 0.60  | 0.15  |
| …        | …     | …     |
| Ours     | 4.40  | 4.26  |

### 3.4 Messaging do’s and don’ts

- **Do:** Say clearly that your model currently underperforms on IHDP and on heteroskedastic SCMs; stress that the benchmark is comprehensive and reproducible.  
- **Do:** Highlight the **design** (parallel worlds, in-context, batching) and the **evaluation** (many baselines, synthetic + real).  
- **Do:** Propose concrete next steps (normalization, fine-tuning, checkpoint selection, ablation of cross-world layers).  
- **Don’t:** Claim SOTA or hide the gap; reviewers and audiences will check.  
- **Don’t:** Blame only “wrong checkpoint” without showing that another checkpoint clearly improves results.

### 3.5 Possible “future work” slide

- Input normalization and feature-type handling for IHDP.  
- Few-shot or light fine-tuning on IHDP (or similar) to bridge distribution shift.  
- Systematic checkpoint selection (e.g. by validation PEHE or by synthetic difficulty).  
- Ablations: no cross-world vs full model on IHDP to see if in-context helps once other factors are fixed.

---

## 4. Multi-Intervention Benchmark

The many-K benchmark evaluates our model on K interventions in one forward pass (K = 3, 5, 8, 10, 15). Baselines (TabPFN, GB-S) are binary CATE only.

**Results (representative):**

| K  | Delta RMSE (linear_gaussian) |
|----|------------------------------|
| 3  | ~2.6                         |
| 5  | ~3.1                         |
| 8  | ~2.7                         |
| 10 | ~2.6                         |
| 15 | ~2.6                         |

No degradation with K; stable Delta RMSE. **Framing**: We trade slight CATE gap for multi-intervention capability: one forward pass for K interventions.

**How to run:** `scripts/run_many_k_synthetic.py --checkpoint <path> --k-list 3 5 8 10 15`

---

## 5. Quick reference: run benchmarks and tests

```bash
# From repo root; ensure code/ is the implementation you care about
cd /path/to/parallel-universe-transformers

# Unit tests
./test/run_tests.sh

# IHDP benchmark (all baselines + Ours)
uv run python -m experiments.benchmarks.real_world_suite --checkpoint checkpoints/final_model.pt --output ihdp_results.json

# Synthetic SCM benchmark
uv run python -m experiments.benchmarks.synthetic_suite --checkpoint checkpoints/final_model.pt --output synthetic_results.json
```

Use the same checkpoint and seeds when comparing or reporting numbers.
