# Training and Evaluation Guide

## 1. What Changed (Phase 1 & 2)

### Phase 1: Model Improvements

- **Model chunking bug fix**: When K > chunk_size (8), the model now returns exactly K counterfactuals (was K+1). Fixed in `model/model.py` and root `model/model.py`.
- **Outcome normalization fix (exp_13)**: Eval scaler now fits on train (70%) only to match finetuning protocol; previously fit on train_val (80%), causing PEHE regression.
- **New experiment configs**:
  - `exp_14_ihdp_longer.json`: max_steps=6000, weight_decay=0.01 (same as exp_10 but 2× steps)
  - `exp_15_ihdp_cosine_lr.json`: max_steps=4000, weight_decay=0.01 (cosine LR, Trainer default)
- **Default checkpoint**: `run_full_benchmark.py` now defaults to `experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt`.

### Phase 2: Multi-Intervention Framing

- **doc/MULTI_INTERVENTION_EVALUATION.md**: Describes many-K benchmark, how to run, and positioning.
- **doc/BENCHMARK_AND_PRESENTATION.md**: Added section 4 (Multi-Intervention Benchmark), updated slide structure, framing.

---

## 2. How to Train

### Pretrained model

```bash
PYTHONPATH=code uv run python code/train_model.py
```

### Finetune on IHDP

```bash
PYTHONPATH=code uv run python scripts/finetune_ihdp.py \
  --resume-from checkpoints/checkpoint_step_40000.pt \
  --max-steps 3000 \
  --lr 0.00001 \
  --weight-decay 0.01 \
  --output checkpoints/finetuned_ihdp.pt
```

### Run experiment by config

```bash
PYTHONPATH=code uv run python experiments/run_experiment.py exp_10
PYTHONPATH=code uv run python experiments/run_experiment.py exp_14
PYTHONPATH=code uv run python experiments/run_experiment.py exp_15
```

---

## 3. How to Evaluate

### Full benchmark (IHDP, ACIC, Twins, Synthetic)

```bash
PYTHONPATH=code uv run python scripts/run_full_benchmark.py \
  --checkpoint experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt \
  --output-dir results
```

### Many-K synthetic benchmark

```bash
PYTHONPATH=code uv run python scripts/run_many_k_synthetic.py \
  --checkpoint experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt \
  --k-list 3 5 8 10 15 \
  --output-dir results
```

### Per-dataset

- **IHDP**: Via `run_full_benchmark.py --datasets ihdp` or `real_world_suite.evaluate_ihdp()`
- **ACIC**: Via `run_full_benchmark.py --datasets acic --acic-data <path>`
- **Twins**: Via `run_full_benchmark.py --datasets twins`
- **Synthetic**: Via `run_full_benchmark.py` (if checkpoint provided) or `run_many_k_synthetic.py`

---

## 4. Checkpoint Selection

- **Default**: `experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt`
- **Best IHDP seed (exp_10)**: seed 456 has PEHE ~2.56 vs 2.64 for seed 42. Use `finetuned_seed456.pt` if available.
- **Scale-outcome checkpoints (exp_13)**: Use `--scale-outcome` when evaluating models trained with outcome normalization.

---

## 5. Reproducibility

- **Seeds**: IHDP uses seed 42 by default; configurable via `--seed`.
- **Splits**: `get_finetune_indices` (70/10/20) for finetuning; `get_benchmark_indices` (80/20) for eval. Test set is the same.
- **Commands**: All commands assume `PYTHONPATH=code` and run from repo root.
