#!/usr/bin/env python3
"""Run full benchmark: IHDP (all models) + optional Synthetic (Ours only, if checkpoint provided).

What we test:
- IHDP: Infant Health and Development Program, 80/20 train-test, PEHE and ATE error.
  Models: Ours (if checkpoint), Linear-T, Linear-S, GB-T, GB-S, DR-Linear, DR-GB,
  TabPFN (if installed), TransTEE (in-repo, trains 50 epochs on support),
  Dragonnet (in-repo, trains 50 epochs), Ridge.
- Synthetic (only if checkpoint): SCM types linear_gaussian, nonlinear_additive,
  multiplicative, heteroskedastic, heavy_tailed, high_dimensional. Ours only.

Usage:
  cd code && uv run python -m experiments.benchmarks.real_world_suite --output ../results/ihdp_results.json
  cd code && uv run python -m experiments.benchmarks.synthetic_suite --checkpoint ../checkpoints/final_model.pt --output ../results/synthetic_results.json
Or run this script from repo root (uses best checkpoint by default; synthetic only if checkpoint exists):
  PYTHONPATH=code uv run python scripts/run_full_benchmark.py --output-dir results
  PYTHONPATH=code uv run python scripts/run_full_benchmark.py --checkpoint checkpoints/final_model.pt --output-dir results
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite (IHDP + optional Synthetic)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_step_40000.pt", help="Path to our model .pt (default: best-by-PEHE from eval_all_checkpoints_ihdp)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for result JSONs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-synthetic", action="store_true", help="Do not run synthetic benchmark even if checkpoint exists")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we can import from code
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    os.chdir(repo_root)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "checkpoint_exists": args.checkpoint is not None and Path(args.checkpoint).exists(),
        "ihdp": {},
        "synthetic": {},
        "what_was_tested": {
            "ihdp": "IHDP dataset, 80/20 split, PEHE and ATE error. Models: Ours (if ckpt), Linear-T/S, GB-T/S, DR-Linear/GB, TabPFN, TransTEE, Dragonnet, Ridge.",
            "synthetic": "Synthetic SCMs (linear_gaussian, nonlinear_additive, multiplicative, heteroskedastic, heavy_tailed, high_dimensional). Ours only; requires checkpoint."
        }
    }

    # 1. IHDP (all baselines; Ours if checkpoint)
    print("\n" + "=" * 60)
    print("1. IHDP BENCHMARK (all models)")
    print("=" * 60)
    try:
        from experiments.benchmarks.real_world_suite import RealWorldBenchmark
        benchmark = RealWorldBenchmark(args.checkpoint, device=args.device)
        ihdp_results = benchmark.evaluate_ihdp()
        benchmark.print_summary(ihdp_results)
        summary["ihdp"] = ihdp_results
        ihdp_path = out_dir / "ihdp_results.json"
        with open(ihdp_path, "w") as f:
            json.dump(ihdp_results, f, indent=2)
        print(f"IHDP results saved to {ihdp_path}")
    except Exception as e:
        summary["ihdp"] = {"error": str(e)}
        print(f"IHDP benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Synthetic (only if checkpoint and not --skip-synthetic)
    if summary["checkpoint_exists"] and not args.skip_synthetic:
        print("\n" + "=" * 60)
        print("2. SYNTHETIC BENCHMARK (Ours)")
        print("=" * 60)
        try:
            from experiments.benchmarks.synthetic_suite import SyntheticBenchmark
            syn_bench = SyntheticBenchmark(args.checkpoint)
            syn_results = syn_bench.run_full_benchmark()
            summary["synthetic"] = syn_results
            syn_path = out_dir / "synthetic_results.json"
            with open(syn_path, "w") as f:
                json.dump(syn_results, f, indent=2)
            print(f"Synthetic results saved to {syn_path}")
        except Exception as e:
            summary["synthetic"] = {"error": str(e)}
            print(f"Synthetic benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not summary["checkpoint_exists"]:
            print("\n2. SYNTHETIC: skipped (no checkpoint).")
        else:
            print("\n2. SYNTHETIC: skipped (--skip-synthetic).")

    summary_path = out_dir / "full_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
