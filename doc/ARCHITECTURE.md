# Architecture Overview

Parallel Universe Transformers: a black-box meta-learner for causal effect estimation trained on synthetic SCM data.

## Components

### 1. Synthetic SCM Engine (`scm/`)
- `schema.py`, `mechanisms.py`, `noise.py`, `sample.py`, `intervene.py`, `counterfactual.py`
- Generates training episodes with ground-truth counterfactuals
- Mixed continuous/categorical features, multiple mechanism types, 3 intervention types

### 2. Episode Pipeline (`episodes/`)
- `config.py`: Curriculum (4 stages), `packer.py`: tensor packing, `dataset.py`: IterableDataset
- Support/query format with parallel worlds (baseline + K interventions)
- **Per-episode input normalization**: Continuous features are z-scored per episode (support stats applied to support and all query worlds); toggle via `CurriculumStage.normalize_continuous`

### 3. Model (`model/`)
- `tokenizer.py`: Tabular tokenization (continuous: MLP+Fourier, categorical: embeddings)
- `backbone.py`: Transformer encoder with cross-world attention at layers 3, 5
- `heads.py`: Prediction + uncertainty (Gaussian NLL, optional quantiles) + **explicit delta head** (effect from baseline vs intervention hidden states)
- Default: d_model=256, 6 layers, 8 heads

### 4. Training (`train/`)
- `losses.py`: Gaussian NLL + delta consistency (from outcome head or explicit delta head) + **calibration loss** (location + scale of deltas)
- `metrics.py`: Baseline, CF, delta, ATE, calibration
- `trainer.py`: Curriculum, mixed precision, gradient checkpointing

### 5. Inference (`inference/`)
- `api.py`: `ParallelUniverseModel.from_pretrained()`, `predict_interventions()`
- DataFrame interface, automatic schema inference

### 6. Experiments (`experiments/`)
- `compare/`: Protocol for ours vs baselines (mean_stub, outcome, dr, bart)
- `eval/`: Checkpoint evaluation by difficulty
- `benchmarks/`: IHDP, ACIC
- `ablations/`: Cross-attention, delta loss

## Key Design

- **Loss**: L_total = L_pred + λ_delta * L_delta + λ_cal * L_cal (L_cal = calibration on deltas; config `lambda_cal`).
- **Delta head**: Deltas are produced by an explicit `DeltaHead` (baseline vs intervention hidden states → effect); training uses these for L_delta and L_cal.
- **World processing**: Fold into batch with world embeddings; cross-attend at layers 3, 5
- **Curriculum**: 4 stages (warmup → basic → moderate → final)

## Tensor Shapes

```
support_x: [B, Ns, d], query_x: [B, W, Nq, d]
→ prediction: [B, W, Nq], log_var: [B, W, Nq], deltas: [B, W-1, Nq]
```
