# Multi-Intervention Evaluation

## What We Test

Our model predicts **K interventions** (SET, SHIFT, RANDOMIZE) on different features in **one forward pass**. Each intervention can target a different feature with a different type:

- **SET**: do(X_j = v) — hard set to value
- **SHIFT**: do(X_j = X_j + δ) — additive shift
- **RANDOMIZE**: do(X_j ~ marginal) — resample from marginal

The model returns baseline predictions and K counterfactual predictions (and deltas) in a single call, with chunking for memory when K > 8.

## Results Summary

The **many-K benchmark** (K = 3, 5, 8, 10, 15) on synthetic SCMs (e.g. linear_gaussian) shows:

- **No degradation with K** — Delta RMSE remains stable as K increases
- **Stable performance** — One forward pass for K interventions scales without quality loss

## How to Run

```bash
# From repo root
PYTHONPATH=code uv run python scripts/run_many_k_synthetic.py \
  --checkpoint experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt \
  --k-list 3 5 8 10 15 \
  --output-dir results
```

Results are saved to `results/many_k_synthetic.json`.

## Positioning

- **Our model** predicts K interventions in one pass; baselines (TabPFN, GB-S) are binary CATE only (T=0 vs T=1).
- **Unique capability**: One forward pass for K interventions; no need to run K separate models.
- **Trade-off**: We trade slight CATE gap on standard benchmarks for multi-intervention capability.
