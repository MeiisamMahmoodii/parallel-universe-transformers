# Parallel Universe Transformers

A black-box meta-learner for causal effect estimation that predicts counterfactual outcomes and intervention effects from tabular data without requiring explicit causal graphs.

## Overview

This project implements a **world-parallel transformer** architecture that processes K+1 parallel "universes" (baseline + K interventions) with inter-world cross-attention. The model is trained on synthetic Structural Causal Model (SCM) episodes where ground-truth counterfactuals are known, enabling it to learn to predict intervention effects in a meta-learning fashion.

## Key Features

- **Dataset-conditioned meta-learning**: Conditions on support set (x,y) pairs to adapt to new domains
- **Parallel world processing**: Efficiently handles multiple interventions simultaneously
- **Inter-world cross-attention**: Enables information sharing between baseline and intervention worlds
- **Uncertainty quantification**: Provides calibrated uncertainty estimates for predictions and deltas
- **Synthetic SCM training**: Learns from diverse synthetic causal mechanisms for generalization

## Architecture

The system consists of:

1. **Synthetic SCM Engine**: Generates training episodes with ground-truth counterfactuals
2. **Tabular Tokenizer**: Converts mixed-type tabular data to transformer tokens
3. **Parallel-World Transformer**: Shared backbone with inter-world cross-attention
4. **Prediction Heads**: Outputs baseline, counterfactual, and uncertainty estimates

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or install in editable mode with dev dependencies
uv sync --all-extras
```

**Common uv commands:**
```bash
uv sync                    # Install dependencies (creates .venv, uses pyproject.toml)
uv sync --all-extras       # Include dev + logging extras (pytest, wandb, etc.)
uv lock                    # Generate uv.lock for reproducible installs
uv run python script.py    # Run script in project environment
uv run pytest tests/       # Run tests
```

With pip:
```bash
pip install -e ".[dev]"  # or pip install -e .
```

## Quick Start

```python
from inference.engine import ParallelUniverseModel
from inference.api import Intervention

# Load model
model = ParallelUniverseModel.from_pretrained("checkpoints/model.pt")

# Define interventions
interventions = [
    Intervention(feature="age", type="set", value=30),
    Intervention(feature="income", type="shift", value=10000),
]

# Predict
results = model.predict_interventions(
    data=support_data,  # Observational data
    query=query_data,   # Rows to predict for
    interventions=interventions
)

print(f"Baseline: {results.baseline}")
print(f"Counterfactuals: {results.counterfactuals}")
print(f"Deltas: {results.deltas}")
print(f"Uncertainty: {results.uncertainty}")
```

## Project Structure

```
parallel-universe-transformers/
├── scm/                    # Synthetic SCM engine
│   ├── schema.py          # Feature schema sampler
│   ├── mechanisms.py      # Structural equation functions
│   ├── noise.py           # Noise distributions
│   ├── sample.py          # Observational sampling
│   ├── intervene.py       # Do-operator
│   └── counterfactual.py  # Counterfactual computation
├── episodes/              # Episode generation
│   ├── packer.py         # Tensor packing
│   ├── dataset.py        # PyTorch dataset
│   └── config.py         # Curriculum config
├── model/                 # Model architecture
│   ├── embeddings.py     # Tabular tokenization
│   ├── tokenizer.py      # Full tokenization pipeline
│   ├── backbone.py       # Transformer encoder
│   ├── cross_world.py    # Inter-world attention
│   ├── attention.py      # Attention primitives
│   └── heads.py          # Prediction heads
├── train/                 # Training system
│   ├── losses.py         # Loss functions
│   ├── trainer.py        # Training loop
│   ├── metrics.py        # Evaluation metrics
│   └── config.py         # Training config
├── inference/             # Inference engine
│   ├── engine.py         # Inference API
│   ├── chunking.py       # Multi-intervention batching
│   └── api.py            # Public interface
└── experiments/           # Experiments
    ├── ablations/        # Ablation studies
    ├── baselines/        # Baseline comparisons
    └── benchmarks/       # Synthetic benchmarks
```

## Training

**Single-GPU:**
```bash
uv run python train_model.py --mixed-precision --batch-size 32 --gradient-accumulation 4 --max-steps 100000
```

**Multi-GPU (DistributedDataParallel):** Use `torchrun` with `--nproc_per_node` equal to the number of GPUs. Each process uses one GPU; data is sharded across ranks. Checkpoints and logging are on rank 0 only.
```bash
torchrun --nproc_per_node=2 train_model.py --mixed-precision --batch-size 32 --gradient-accumulation 4 --max-steps 100000
```
Single-GPU is the same script with one process: `torchrun --nproc_per_node=1 train_model.py ...`

**Memory and curriculum:** Training uses a curriculum (more features and interventions in later stages). Attention memory scales as batch × worlds × sequence², so later stages can hit CUDA OOM. Two mitigations are applied by default: (1) **gradient checkpointing** is on (disable with `--no-gradient-checkpointing` if you have plenty of VRAM). (2) **Per-stage batch size** is reduced with a safety margin (e.g. stage 1 runs with per-GPU batch size 1 while accumulation keeps the effective batch large). If you still see OOM, try lowering `--batch-size` manually or set `PYTORCH_ALLOC_CONF=expandable_segments:True`.

## Evaluation

```bash
python -m experiments.benchmarks.synthetic_suite --checkpoint checkpoints/model.pt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{parallel_universe_transformers,
  title={Parallel Universe Transformers: Black-box Meta-learning for Causal Effect Estimation},
  author={Meisam},
  year={2026},
  url={https://github.com/meisam/parallel-universe-transformers}
}
```

## License

MIT License
