# Experiment Harness

Self-contained experiment harness for running and comparing finetuning experiments. Each experiment is defined by a JSON config and can be run independently.

## Structure

```
experiments/
├── README.md           # This file
├── configs/            # Experiment configs (exp_01 through exp_09)
├── results/            # Output per experiment (created at runtime)
│   ├── exp_01_baseline/
│   │   ├── metrics.json
│   │   ├── per_seed.json
│   │   ├── comparison.json
│   │   └── checkpoints/
│   └── ...
├── run_experiment.py   # Run one experiment by name
├── run_all.py         # Run all experiments
└── compare_results.py # Aggregate and print comparison table
```

## How to Run

### Single experiment

```bash
PYTHONPATH=code uv run python experiments/run_experiment.py exp_01
PYTHONPATH=code uv run python experiments/run_experiment.py exp_01_baseline
```

Accepts `exp_01` or `exp_01_baseline` (match by prefix).

### All experiments

```bash
PYTHONPATH=code uv run python experiments/run_all.py
```

Optionally run in parallel:

```bash
PYTHONPATH=code uv run python experiments/run_all.py --parallel 2
```

### Compare results

```bash
PYTHONPATH=code uv run python experiments/compare_results.py
```

With JSON and CSV output:

```bash
PYTHONPATH=code uv run python experiments/compare_results.py --output-json --output-csv
```

## Experiment List

| ID | Name | What it varies |
|----|------|----------------|
| exp_01 | baseline | Multi-dataset, lr=5e-6, weight_decay=0.1, 1000 steps |
| exp_02 | lambda_delta_4 | lambda_delta=4.0 instead of 2.0 |
| exp_03 | ihdp_only | Single dataset IHDP, 1500 steps, lr=1e-5 |
| exp_04 | multi_dataset | Same as baseline |
| exp_05 | higher_lr | lr=1e-5 instead of 5e-6 |
| exp_06 | longer_finetune | 2000 steps |
| exp_07 | ihdp_heavy_mix | mix_ratio [0.6, 0.2, 0.2] |
| exp_08 | synthetic_ihdp_mix | synthetic-mix-ratio 0.3, IHDP only |
| exp_09 | stronger_reg | weight_decay=0.2 |

## Prerequisites

- Pretrained checkpoint at `checkpoints/checkpoint_step_40000.pt`
- ACIC data at `data/acic_sample.csv` (for multi-dataset experiments)
- Run from repository root with `PYTHONPATH=code`
