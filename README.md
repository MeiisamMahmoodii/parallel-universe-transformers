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

**Important:** All Python commands (training, evaluation, tests) must be run with **`PYTHONPATH=code`** so that imports resolve to the `code/` package. Example: `PYTHONPATH=code uv run python code/train_model.py`.

**Common uv commands:**
```bash
uv sync                    # Install dependencies (creates .venv, uses pyproject.toml)
uv sync --all-extras       # Include dev + logging extras (pytest, wandb, etc.)
uv lock                    # Generate uv.lock for reproducible installs
PYTHONPATH=code uv run python script.py   # Run script (always set PYTHONPATH=code)
PYTHONPATH=code uv run pytest test/       # Run tests
```

With pip:
```bash
pip install -e ".[dev]"  # or pip install -e .
```

## Quick Start

Run with `PYTHONPATH=code` so imports resolve to `code/`:

```python
from inference.api import ParallelUniverseModel, Intervention

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
├── code/                   # Source code
│   ├── scm/               # Synthetic SCM engine
│   │   ├── schema.py      # Feature schema sampler
│   │   ├── mechanisms.py  # Structural equation functions
│   │   └── ...
│   ├── episodes/          # Episode generation
│   │   └── ...
│   ├── model/             # Model architecture
│   │   └── ...
│   ├── train/             # Training system
│   │   └── ...
│   ├── inference/         # Inference API
│   ├── experiments/       # Experiments
│   │   └── ...
│   ├── utils/             # Shared utilities
│   ├── examples/          # Usage examples
│   └── scripts/           # Run scripts
├── data/                  # Data files
├── doc/                   # Documentation
└── test/                  # Tests
```

## Training

**Single-GPU:**
```bash
PYTHONPATH=code uv run python code/train_model.py --mixed-precision --batch-size 32 --gradient-accumulation 4 --max-steps 100000
```

**Multi-GPU (DistributedDataParallel):** Use `torchrun` with `--nproc_per_node` equal to the number of GPUs. Each process uses one GPU; data is sharded across ranks. Checkpoints and logging are on rank 0 only.
```bash
PYTHONPATH=code torchrun --nproc_per_node=2 code/train_model.py --mixed-precision --batch-size 32 --gradient-accumulation 4 --max-steps 100000
```
Single-GPU is the same script with one process: `PYTHONPATH=code torchrun --nproc_per_node=1 code/train_model.py ...`

**Memory and curriculum:** Training uses a curriculum (more features and interventions in later stages). Attention memory scales as batch × worlds × sequence², so later stages can hit CUDA OOM. Two mitigations are applied by default: (1) **gradient checkpointing** is on (disable with `--no-gradient-checkpointing` if you have plenty of VRAM). (2) **Per-stage batch size** is reduced with a safety margin (e.g. stage 1 runs with per-GPU batch size 1 while accumulation keeps the effective batch large). If you still see OOM, try lowering `--batch-size` manually or set `PYTORCH_ALLOC_CONF=expandable_segments:True`.

**Long-run curriculum (optional):** To spend more steps on harder stages (stage_2, stage_3) and potentially improve delta_correlation there, set `long_run_curriculum=True` in [train/config.py](code/train/config.py) or pass it when constructing `TrainingConfig` (e.g. in a custom training script). See [episodes/config.py](code/episodes/config.py) `get_long_run_curriculum()` and [doc/EVALUATION.md](doc/EVALUATION.md) for details.

## Usage: when to use (and when not to)

- **Use this model when:** You have (or can simulate) **many interventions** per unit, want **conditional effect estimates** (CATE-like) from tabular support + query, and your setting is **synthetic or SCM-like** or you accept a learned-prior interpretation. One forward pass gives baseline, interventional predictions, and deltas for all interventions. See [doc/PROJECT_REPORT.md](doc/PROJECT_REPORT.md) for the technical scope and [doc/EVALUATION.md](doc/EVALUATION.md) for reproduction and benchmarks.
- **Do not rely on it when:** There is **no clear intervenability** (e.g. immutable covariates only), you need **guaranteed unit-level ITE** without SCM assumptions, or you use **out-of-family data** without caveats. The model is trained on synthetic SCMs; generalization to real data is not guaranteed.

For architecture details, see [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md).

**Single-call prediction:** From DataFrames and a checkpoint, use the helper in [code/inference/api.py](code/inference/api.py): `predict(support_df, query_df, interventions_list, checkpoint_path, device="cuda")`. See [code/examples/](code/examples/) for usage examples.

## Reproduce results

To run the full benchmark (IHDP + synthetic) with the best checkpoint by default:

```bash
PYTHONPATH=code uv run python scripts/run_full_benchmark.py --output-dir results
```

To find the best checkpoint on IHDP (PEHE): `PYTHONPATH=code uv run python scripts/eval_all_checkpoints_ihdp.py --checkpoint-dir checkpoints --output-dir results`.

To reproduce the full test and evaluation suite (unit tests, checkpoint tests, eval matrix, comparison protocol, IHDP, ablations):

```bash
PYTHONPATH=code uv run python code/scripts/run_full_comparison.py
```

Results are written under `results/full_comparison_<timestamp>/`. Use `--quick` for a fast smoke run. Use `--checkpoint-tests` to run sanity checks per checkpoint, `--compare-all-checkpoints` for per-checkpoint comparison with significance, and `--stages all` for all curriculum stages. See [doc/EVALUATION.md](doc/EVALUATION.md) and [doc/REPRODUCE.md](doc/REPRODUCE.md).

## Evaluation

```bash
PYTHONPATH=code uv run python -m experiments.benchmarks.synthetic_suite --checkpoint checkpoints/model.pt
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
