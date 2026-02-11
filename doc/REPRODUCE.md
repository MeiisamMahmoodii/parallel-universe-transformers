# Reproduce results

One-page guide to reproduce the evaluation and comparison results from the Parallel Universe Transformers codebase.

## Single command (full run)

From the project root:

```bash
uv run python code/scripts/run_full_comparison.py
```

- **What it does:** Runs unit tests (SCM + model), eval matrix (all checkpoints × curriculum difficulties), comparison protocol with all methods (ours, mean_stub, outcome, dr, bart) on SCM and IHDP, and ablations.
- **Output:** All results under `results/full_comparison_<timestamp>/`:
  - `unit_tests.log`, `eval_matrix/`, `protocol_scm/`, `protocol_ihdp/`, `ablations/`
  - `full_comparison_summary.json`, `full_comparison_summary.md`
- **Defaults:** Device cuda, seeds 42 43 44, 200 batches. Use `--quick` for a fast smoke run.
- **Extra options:** `--checkpoint-tests` (sanity tests per checkpoint), `--compare-all-checkpoints` (protocol per checkpoint with significance), `--stages all`.

## Options

- `--checkpoint-dir DIR` — Directory to scan for checkpoints (default: checkpoints).
- `--output-dir DIR` — Output root (default: results/full_comparison_<timestamp>).
- `--device cuda|cpu`
- `--seeds 42 43 44` — Seeds for eval and comparison.
- `--num-batches N` — Batches per stage (default: 200).
- `--stages stage_1_basic` or `all` — Curriculum stage(s) for protocol.
- `--checkpoint-tests` — Run pytest test_checkpoint per checkpoint.
- `--compare-all-checkpoints` — Run protocol per checkpoint × stage with significance.
- `--include-ihdp` — Run IHDP protocol (default when not `--quick`).
- `--quick` — Fewer seeds/batches, skip IHDP/ablations.

## Citation

If you use this code or report these results, please cite:

```bibtex
@software{parallel_universe_transformers,
  title={Parallel Universe Transformers: Black-box Meta-learning for Causal Effect Estimation},
  author={Meisam},
  year={2026},
  url={https://github.com/meisam/parallel-universe-transformers}
}
```

Same BibTeX appears in [README.md](../README.md) and [PROJECT_REPORT.md](PROJECT_REPORT.md).

## More detail

- Full evaluation suite (eval-by-difficulty, matrix, comparison protocol, IHDP, ablations): [EVALUATION.md](EVALUATION.md).
- Technical scope and limitations: [PROJECT_REPORT.md](PROJECT_REPORT.md).
