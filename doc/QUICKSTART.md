# Parallel Universe Transformers - Quick Start Guide

This guide will help you get started with training and using the Parallel Universe Transformer for causal effect estimation.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
cd parallel-universe-transformers

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# With dev dependencies (pytest, black, etc.)
uv sync --all-extras
```

Run commands inside the uv environment:
```bash
uv run python code/train_model.py --max-steps 1000   # run script
uv run pytest test/                             # run tests
```

Alternative with pip:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Test

Run the test suite to verify everything is working:

```bash
./test/run_tests.sh
```

Or run individual tests:

```bash
python test/test_scm.py
python test/test_model.py
```

## Training a Model

### Option 1: Quick Training (Small Model)

For quick experiments and testing:

```bash
python code/train_model.py \
    --d-model 128 \
    --n-layers 4 \
    --n-heads 4 \
    --batch-size 16 \
    --max-steps 10000 \
    --checkpoint-dir checkpoints/quick_test
```

### Option 2: Default Training

Train with default configuration:

```bash
python code/train_model.py
```

### Option 3: Full Training with W&B Logging

Train with Weights & Biases logging:

```bash
python code/train_model.py \
    --wandb \
    --wandb-project my-project \
    --wandb-run-name experiment-1 \
    --max-steps 100000
```

### Training Options

Key arguments:
- `--d-model`: Model dimension (default: 256)
- `--n-layers`: Number of transformer layers (default: 6)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--max-steps`: Maximum training steps (default: 100000)
- `--lambda-delta`: Weight for delta loss (default: 1.0)
- `--mixed-precision`: Enable mixed precision training
- `--gradient-checkpointing`: Enable gradient checkpointing (saves memory)

## Using a Trained Model

### Basic Usage

```python
from inference.api import ParallelUniverseModel, Intervention
import pandas as pd

# Load model
model = ParallelUniverseModel.from_pretrained('checkpoints/final_model.pt')

# Prepare data
support_data = pd.DataFrame({...})  # Your observational data
query_data = pd.DataFrame({...})    # Rows to predict for

# Define interventions
interventions = [
    Intervention(feature='age', type='set', value=30),
    Intervention(feature='income', type='shift', value=10000),
]

# Predict
results = model.predict_interventions(
    data=support_data,
    query=query_data,
    interventions=interventions
)

# Access results
print(f"Baseline: {results.baseline}")
print(f"Counterfactuals: {results.counterfactuals}")
print(f"Deltas (effects): {results.deltas}")
print(f"Uncertainty: {results.uncertainty}")
```

### Running Examples

Basic example:
```bash
python examples/basic_usage.py
```

Advanced example with custom SCM:
```bash
python examples/advanced_usage.py
```

## Running Experiments

### Ablation Studies

Run all ablations:
```bash
python -m experiments.ablations.run_ablations --ablation all
```

Run specific ablation:
```bash
python -m experiments.ablations.run_ablations --ablation cross_attention
python -m experiments.ablations.run_ablations --ablation delta_loss
```

### Benchmark Suite

Evaluate on synthetic benchmark:
```bash
python -m experiments.benchmarks.synthetic_suite \
    --checkpoint checkpoints/final_model.pt \
    --output benchmark_results.json
```

## Project Structure

```
parallel-universe-transformers/
├── scm/                    # Synthetic SCM engine
├── episodes/              # Episode generation
├── model/                 # Model architecture
├── train/                 # Training system
├── inference/             # Inference API
├── experiments/           # Experiments
├── examples/              # Usage examples
├── test/                  # Test suite
├── train_model.py         # Main training script
└── README.md             # Full documentation
```

## Next Steps

1. **Train a model**: Start with a quick test run
2. **Evaluate**: Run the benchmark suite
3. **Experiment**: Try ablations to understand the model
4. **Apply**: Use the trained model on your data

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Enable `--gradient-checkpointing`
- Reduce `--d-model` or `--n-layers`

### Slow Training

- Enable `--mixed-precision`
- Increase `--batch-size` if you have memory
- Use more `--num-workers` for data loading

### Poor Performance

- Increase `--max-steps`
- Tune `--lambda-delta` (try 0.5, 1.0, 2.0)
- Check if cross-world attention is enabled (default: yes)

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

## Support

For issues and questions, please open an issue on GitHub.
