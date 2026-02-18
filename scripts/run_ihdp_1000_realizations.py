#!/usr/bin/env python3
"""Run IHDP evaluation over 1000 realizations (seeds 0..999).

Reports mean ± std PEHE and ATE per model for comparison with papers that use
the standard 1000-realization IHDP protocol.

Usage:
  PYTHONPATH=code uv run python scripts/run_ihdp_1000_realizations.py --checkpoint checkpoints/checkpoint_step_40000.pt
  PYTHONPATH=code uv run python scripts/run_ihdp_1000_realizations.py --checkpoint ... --n-realizations 100 --models Ours TabPFN GB-S
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="IHDP 1000-realization benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_step_40000.pt", help="Path to our model")
    parser.add_argument("--output", type=str, default="results/ihdp_1000_realizations.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-realizations", type=int, default=1000, help="Number of seeds (default: 1000)")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Models to run (default: all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from experiments.benchmarks.real_world_suite import RealWorldBenchmark

    benchmark = RealWorldBenchmark(args.checkpoint, device=device)
    if args.models:
        benchmark.baselines = {k: v for k, v in benchmark.baselines.items() if k in args.models}

    pehe_by_model = {name: [] for name in benchmark.baselines}
    ate_by_model = {name: [] for name in benchmark.baselines}

    n = args.n_realizations
    for seed in range(n):
        if (seed + 1) % 50 == 0 or seed == 0:
            print(f"Realization {seed + 1}/{n}...")
        try:
            results = benchmark.evaluate_ihdp(seed=seed)
            for name, m in results.items():
                if "error" in m:
                    continue
                pehe_by_model[name].append(m["PEHE"])
                ate_by_model[name].append(m["ATE_Err"])
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    summary = {}
    for name in benchmark.baselines:
        pehe_list = pehe_by_model[name]
        ate_list = ate_by_model[name]
        if pehe_list:
            summary[name] = {
                "pehe_mean": float(np.mean(pehe_list)),
                "pehe_std": float(np.std(pehe_list)),
                "ate_mean": float(np.mean(ate_list)),
                "ate_std": float(np.std(ate_list)),
                "n_realizations": len(pehe_list),
            }
        else:
            summary[name] = {"error": "No successful runs"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"IHDP {n} REALIZATIONS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} | {'PEHE (mean ± std)':<25} | {'ATE (mean ± std)':<25}")
    print("-" * 75)
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:<20} | {s['error']}")
        else:
            pehe = f"{s['pehe_mean']:.4f} ± {s['pehe_std']:.4f}"
            ate = f"{s['ate_mean']:.4f} ± {s['ate_std']:.4f}"
            print(f"{name:<20} | {pehe:<25} | {ate:<25}")
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
