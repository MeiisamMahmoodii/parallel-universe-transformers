#!/usr/bin/env python3
"""Run synthetic benchmark with K=5, 10, 15 interventions (many-K counterfactual evaluation).

Usage:
  PYTHONPATH=code uv run python scripts/run_many_k_synthetic.py --checkpoint checkpoints/checkpoint_step_40000.pt --output-dir results
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
code_dir = repo_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))


def main():
    parser = argparse.ArgumentParser(description="Many-K synthetic benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 15], help="List of K values")
    parser.add_argument("--scm-types", type=str, nargs="*", default=None, help="SCM types (default: all)")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from experiments.benchmarks.synthetic_suite import SyntheticBenchmark
    benchmark = SyntheticBenchmark(args.checkpoint)
    print("Running many-K benchmark (counterfactual representation at scale)...")
    results = benchmark.run_many_k_benchmark(k_list=args.k_list, scm_types=args.scm_types)
    path = out_dir / "many_k_synthetic.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
