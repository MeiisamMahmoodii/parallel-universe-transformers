# Parallel Universe Transformers: Project Report

**Technical Report — Parallel Universe Transformers**

This project implements a **black-box tabular meta-learner** for causal effect estimation. From observational data and **without a causal graph at inference**, it predicts an **observational baseline**, **conditional interventional responses** per intervention, and **conditional effect (delta)** estimates for **many interventions per sample** in one forward pass, using a **parallel-world** transformer trained on synthetic Structural Causal Models (SCMs). This is **amortized causal estimation under a meta-learned prior over SCMs**: the model outputs interventional predictions from observational samples because it has been trained on synthetic SCM episodes and learns regularities linking observational patterns to intervention outcomes within that SCM family. Out-of-family generalization is not guaranteed; causal validity depends on how well the training SCM distribution matches the deployment domain. We show that the model passes all correctness checks, consistently beats a trivial baseline, and achieves non-trivial delta correlation on synthetic curriculum evaluation (strongest on easier stages, with room to improve on harder ones). See [Model and goal](MODEL_AND_GOAL.md) for full technical detail and [Evaluation](EVALUATION.md) for how to reproduce.

---

## 1. What the project is and what it does exactly

**Inputs:**

- An **observational support set**: (x, y) pairs from the same domain (e.g. tabular features and outcome).
- **Query units**: rows x for which we want predictions.
- A set of **K interventions** (e.g. do(X_j = v_j) for specific features and values).

**Outputs (per query):**

- **Baseline (observational reference):** prediction of the observational conditional mean μ(x) = E[Y | X = x]. We use it as a reference for deltas; it is not necessarily a causal “no-intervention” baseline unless additional identification assumptions hold.
- **K interventional responses:** for each intervention k, prediction of a conditional interventional mean μ_k(x_{-j}) = E[Y | do(X_j = v_k), X_{-j} = x_{-j}] (with the obvious generalization when multiple features are intervened on). The model outputs one prediction per world (one baseline + K interventional worlds).
- **K deltas (conditional effects):** for each intervention k, Δ_k(x) = μ_k(x_{-j}) − μ(x) — the difference between interventional response and observational baseline. These are **conditional effect** (CATE-like) estimates per unit features, not guaranteed unit-level counterfactuals at inference (see §2).
- **Uncertainty** (optional): e.g. log-variance per world for prediction intervals.

**Black-box:** No causal graph or SCM structure is required at test time. The model is trained on synthetic SCM episodes where ground truth is known; at inference it conditions on the support set and query and outputs the above.

End-to-end: tabular input → tokenizer → parallel worlds → shared backbone + cross-world attention → heads → baseline, interventional responses, deltas, uncertainty. See [MODEL_AND_GOAL.md](MODEL_AND_GOAL.md) §7 for pipeline and backbone diagrams.

---

## 2. What we are estimating

We estimate the following quantities (conditional interventional formulation, Option A):

- **Baseline (observational reference):** μ(x) = E[Y | X = x]. The model’s “world 0” predicts this observational conditional mean.

- **Interventional response (per intervention k):** μ_k(x_{-j}) = E[Y | do(X_j = v_k), X_{-j} = x_{-j}]. For intervention k that sets some subset of features (e.g. X_j) to values (e.g. v_k), we condition on the rest of the covariates X_{-j} = x_{-j}. This avoids conditioning on an impossible event (e.g. X_j = x_j and do(X_j = v_k) with x_j ≠ v_k). The model outputs one prediction per world (one baseline + K interventional worlds).

- **Delta (conditional effect):** Δ_k(x) = μ_k(x_{-j}) − μ(x). We estimate these **directly** in the architecture (Δ̂_k = ŷ_do,k − ŷ_base) and train with a delta-consistency loss.

**Synthetic vs real deployment:** On synthetic SCM episodes we can compute **true unit-level counterfactual deltas** because counterfactuals share the same exogenous noise per unit. Our delta-consistency loss is well-defined in that regime. In **real observational deployment**, we do not observe the unit’s latent noise; without additional structure, the best-defined estimand is a conditional interventional mean difference (CATE-like). Therefore our outputs should be interpreted as **conditional interventional mean differences under the learned SCM prior**, not guaranteed individual counterfactuals.

**Evaluation metrics:** We report **delta_correlation** (correlation between predicted and true deltas; higher = better alignment), **delta_rmse** (RMSE of predicted vs true deltas; delta_rmse = √PEHE), and pehe = delta_rmse² for comparability with the treatment-effect literature.

---

## 3. Why we have a "parallel universe" (design and motivation)

**Problem:** Real-world decisions often require many “what-if” questions per unit (e.g. many features and values). Building and using explicit causal graphs is costly. In-context causal methods (e.g. Do-PFN) pack support, query, and intervention(s) into **one long sequence**, so the **comparison** between baseline and interventional worlds is only implicit and scaling to many interventions can mean one forward per intervention or a very long context.

**Idea:** Treat each intervention as its own **world**. Run **1 baseline world + K interventional worlds** in parallel (batch dimension B×W). Use a **shared transformer** over all worlds and, at selected layers, **inter-world cross-attention** so each world can attend to the others’ query tokens. Define **deltas in the architecture**: Δ̂_k = ŷ_do,k − ŷ_base, and train with regression loss on outcomes plus a **delta-consistency loss** so that Δ̂_k matches the true interventional-minus-baseline effect from synthetic SCMs.

**Why “parallel universe”:** We literally process **parallel streams** (one sequence per world) in one batch, with **explicit comparison** across worlds via cross-attention and first-class deltas — hence “parallel universes.” The model learns to estimate effects by comparing these worlds, without ever seeing a graph at test time. See [MODEL_AND_GOAL.md](MODEL_AND_GOAL.md) §3–4 and §7.3 for story and backbone/cross-world diagrams.

---

## 4. How we implement parallel worlds in the transformer architecture

We implement the parallel-world design as follows.

**Input layout.** The model receives **query features per world**: `query_x` has shape [B, W, Nq, d] — batch B, W = 1+K worlds (one baseline, K interventions), Nq query units, d features. For the baseline world (index 0), query features are the original unit; for each intervention world, query features are the **intervention-modified** inputs (e.g. after applying do(X_j = v_j)). The **support set** (x, y) is shared and has shape [B, Ns, d] for x and [B, Ns] for y.

**Tokenization and world IDs.** The tabular tokenizer turns support and query into sequences of tokens (feature values + feature IDs + role + optional Y token). Crucially, each **query token gets a world embedding**: world 0 for baseline, world 1..K for each intervention. So the model sees which "world" each token belongs to. Support tokens are not world-specific (they are shared context). Query tokens are produced per world, so we get **W separate query sequences** (one per world), each with the same length (Nq × (d+1) tokens after flattening).

**One batch of world-sequences.** Support tokens are expanded to be identical for every world ([B, Ns*(d+1), d_model] → [B, W, Ns*(d+1), d_model]). They are then concatenated with the per-world query tokens: for each world w we have [support tokens | query tokens for world w]. This gives a tensor [B, W, Ns*(d+1) + Nq*(d+1), d_model]. We then **flatten the world dimension into the batch dimension**: [B×W, Ns*(d+1) + Nq*(d+1), d_model]. So the transformer encoder effectively processes B×W sequences in one batch, each sequence being "support + one world's query." The **shared backbone** is a standard transformer stack (self-attention + feedforward per layer) over these B×W sequences; within each sequence, support and query tokens attend to each other normally.

**Cross-world attention at selected layers.** At a subset of layers (e.g. layers 3 and 5 in the default config), we insert **cross-world attention** after the self-attention block. We reshape the hidden states from [B×W, N, d_model] back to [B, W, N, d_model] so we can separate worlds. We take only the **query part** of the sequence (the tokens that come after the support set). For each world w, we run **cross-attention**: the query tokens of world w attend to the query tokens of the **other** worlds (or optionally only to the baseline world). The key/value are the other worlds' query hidden states; the query is the current world's query hidden states. After the cross-attention update (with residual and layer norm), we put the updated query back next to the unchanged support tokens and flatten again to [B×W, N, d_model] for the next transformer block. So within the same layer, each world's query representation is updated using information from the other worlds' query representations — that is the "communication" between parallel universes.

**Outputs and deltas.** After the full encoder, we take the hidden states for the **Y (outcome) token** of each query unit in each world — one position per (world, query index). A small prediction head maps these to a scalar prediction per (world, query). We reshape to [B, W, Nq], so we have a baseline prediction (world 0) and K interventional predictions (worlds 1..K). **Deltas** are computed in the model as prediction[:, 1:, :] − prediction[:, 0:1, :], i.e. each interventional prediction minus the baseline. No extra parameters: deltas are just the difference of the head outputs, which are trained with both a regression loss on the per-world predictions and a delta-consistency loss so that this difference matches the true effect from the SCM.

**Summary.** Parallel worlds are implemented as **W parallel sequences** (support + query per world) stacked in the batch dimension (B×W), with **world embeddings** in the tokenizer so the model knows which world each query token belongs to, and **cross-world attention** at selected layers so each world's query tokens can attend to the other worlds' query tokens. One shared transformer backbone and one shared head produce baseline and interventional predictions; deltas are the explicit difference and are trained directly.

**Design notes:** Cross-world attention is permitted because the task **explicitly provides multiple interventions jointly**; we are not modeling a sequential decision process. Intervention ordering (world index 1..K) is fixed per forward pass; randomizing intervention order during training could be added to reduce dependence on world index.

---

## 5. Use cases

- **Policy / decision support:** “For this unit, what’s the effect of changing feature A vs B vs C?” in one model call.
- **Sensitivity / attribution:** Many do-interventions (e.g. do(X_i = x_i)) to see which variables drive the outcome.
- **Fairness / recourse:** “What would the outcome be under different protected-attribute or recourse interventions?” for many interventions per individual.
- **Synthetic SCM benchmarks:** Pre-trained on synthetic SCMs; evaluate on held-out SCM families or (with appropriate caveats) real data.

**Caveat:** We treat feature-wise interventions as well-defined manipulations in the SCM. In real tabular data, **intervenability and causal meaning of features must be specified**; otherwise “do(feature = value)” may not correspond to any feasible action (e.g. some columns may be post-treatment, proxies, colliders, or not manipulable).

---

## 6. Evaluation results (what we tested and what we found)

**What we test:**

1. **Unit and checkpoint sanity tests:** SCM and model unit tests; for each checkpoint, load model, run one batch, and assert shapes and **delta consistency** (deltas = interventional predictions − baseline, no NaN/Inf).
2. **Eval matrix:** All checkpoints × curriculum difficulties (stage_0_warmup through stage_3_final) × multiple seeds. Main metric: **delta_correlation** (and delta_mae, baseline_mae, etc.) on 200 batches per (checkpoint, stage, seed).
3. **Comparison protocol:** Same data and metrics for our model vs a **mean baseline stub** (predicts mean(support_y) for all worlds → deltas = 0). Multi-seed with bootstrap CI for delta_correlation.

**Results (representative full run: 20 checkpoints, 4 stages, 3 seeds):**

- **Correctness:** All unit and checkpoint sanity tests pass; delta consistency holds for every checkpoint.
- **Delta correlation by difficulty:**
  - **stage_0_warmup** (easiest): ~0.15–0.21
  - **stage_1_basic:** ~0.09–0.15
  - **stage_2_moderate:** ~0.04–0.07
  - **stage_3_final** (hardest): ~0.03–0.07
- **Ours vs baseline:** Our model attains **positive** delta_correlation across stages and checkpoints; the mean baseline stub has delta_correlation **0** by construction. Beating this baseline is a **sanity check** (the stub predicts constant across worlds), not evidence of strong causal learning; stronger baselines are needed (see §8).

**Takeaway:** The model behaves correctly and produces conditional effect estimates with non-trivial delta_correlation on synthetic SCM evaluation. Performance is **difficulty-dependent**: stronger on easier curriculum stages, with clear room to improve on harder ones. **Metric caveat:** delta_correlation can look decent even when calibration is poor (wrong scale or biased mean); it can also be unstable when true deltas have low variance. Results are written under a timestamped output dir (e.g. `results/full_test_<timestamp>/`); see [EVALUATION.md](EVALUATION.md) for layout and how to run `scripts/run_full_test_and_eval.py`.

---

## 7. How and why we can compete with existing models

**Design advantages (why we can compete):**

1. **Many interventions in one call:** One forward pass for 1 + K worlds (baseline + K interventional worlds). Do-PFN-style methods typically encode interventions in one long sequence or use one call per intervention; we scale in the batch dimension instead.
2. **Explicit deltas and structural bias:** Deltas are **first-class** outputs (Δ̂_k = ŷ_do,k − ŷ_base) with cross-world attention trained to compare worlds; Do-PFN has no such built-in bias for “effect” as a direct object.
3. **Same metric language:** We report delta_correlation, PEHE (via delta_rmse), and related metrics, so we can be compared on the same benchmarks and tables.

**Current evidence:** We beat the in-protocol trivial baseline (mean stub). We have **not yet** run Do-PFN or other causal in-context baselines on the same protocol.

**What is needed to claim competitiveness:**

1. Run Do-PFN (or another published causal in-context method) on the **same** curriculum stages, seeds, and metrics (delta_correlation, PEHE).
2. Optionally run on standard benchmarks (e.g. IHDP-style, ACIC) if applicable.
3. Report a comparison table: Method | Stage/dataset | delta_corr | PEHE | One-call many interventions (Y/N).

**Summary:** We are **positioned** to compete in many-intervention, in-context causal effect settings. Head-to-head and benchmark results on the same metrics will complete the picture. See [MODEL_AND_GOAL.md](MODEL_AND_GOAL.md) §4 for a side-by-side comparison with Do-PFN.

---

## 8. Limitations, future work, and reproducibility

**Limitations:**

1. Evaluation is on **synthetic SCMs** (same family as training); real-world transfer is untested.
2. Harder curriculum stages show modest delta_correlation (~0.04–0.07).
3. Current comparison is only vs a trivial baseline (mean stub); no Do-PFN or other strong baselines yet.
4. **Metric limitations:** delta_correlation alone does not guarantee good calibration or scale; reporting slope/intercept of (true deltas vs predicted deltas) and sign accuracy would strengthen claims.

**Future work:**

1. Add Do-PFN (or equivalent) to the comparison protocol and report the same metrics.
2. Semi-synthetic or real benchmarks where applicable.
3. Improvements for high-difficulty stages (architecture, curriculum, or data).
4. **Stronger evaluation:** (a) Calibration/scale: regress true deltas on predicted deltas (slope, intercept). (b) Sign accuracy or top-k intervention ranking accuracy. (c) Ablations: no-cross-world attention (same backbone) to isolate the parallel-world mechanism; single-world-per-intervention with same compute to show batching is not the only advantage. (d) Where possible, a simple doubly-robust baseline on synthetic data with known adjustment set.
5. Consider **randomizing intervention ordering** during training to reduce dependence on world index.

**Reproducibility:** Run the full test and eval suite with:

```bash
uv run python scripts/run_full_test_and_eval.py
```

Results are written under `results/full_test_<timestamp>/` (unit_tests.log, pytest_checkpoint_*.log, eval_matrix/, compare/, summary.json). Default: seeds 42, 43, 44; 200 batches per stage. Use `--quick` for a fast smoke run. See [EVALUATION.md](EVALUATION.md) for all options and output layout.

---

## 9. Glossary and references

**Glossary (short):**

- **SCM:** Structural Causal Model; a set of equations and noise distributions that define how variables and outcomes are generated.
- **Do-operator:** Intervention that sets a variable to a value (e.g. do(X = x)); we predict outcomes under such interventions.
- **Counterfactual (strict):** Outcome for the same unit under a different intervention, defined in an SCM via shared exogenous noise (abduction–action–prediction). Our model does not perform abduction at inference; we output conditional interventional means, not guaranteed unit-level counterfactuals on real observational data.
- **ITE:** Individual Treatment Effect; the effect for a specific unit (true ITE requires unit-level counterfactuals). On synthetic SCMs we evaluate true ITE; at deployment our deltas are conditional effect (CATE-like) estimates under the learned prior.
- **PEHE:** Precision in Estimation of Heterogeneous Effect; mean squared error of individual effects. We report delta_rmse = √PEHE and pehe = delta_rmse².
- **delta_correlation:** Correlation between predicted and true deltas. Does not alone guarantee calibration or correct scale.
- **Support / query:** In meta-learning, the support set is the (x, y) context the model conditions on; the query is the unit(s) we predict for.

**Related work:** Do-PFN and causal in-context / meta-learning methods; see [MODEL_AND_GOAL.md](MODEL_AND_GOAL.md) §4 for the comparison table.

**Citation:** If you use this code in your research, please cite:

```bibtex
@software{parallel_universe_transformers,
  title={Parallel Universe Transformers: Black-box Meta-learning for Causal Effect Estimation},
  author={Meisam},
  year={2026},
  url={https://github.com/meisam/parallel-universe-transformers}
}
```

(Also in [README.md](../README.md).)
