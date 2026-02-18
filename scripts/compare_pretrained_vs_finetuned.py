#!/usr/bin/env python3
"""Compare pretrained vs finetuned checkpoint on IHDP, ACIC, Twins.

Runs both checkpoints over multiple seeds and reports mean ± std for PEHE and ATE.
Useful to see if pretrained is preferable for ACIC/Twins (which may forget over IHDP finetuning).

Usage:
  PYTHONPATH=code uv run python scripts/compare_pretrained_vs_finetuned.py \
    --pretrained checkpoints/checkpoint_step_40000.pt \
    --finetuned checkpoints/finetuned_ihdp.pt \
    --seeds 42 123 456 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained vs finetuned on IHDP, ACIC, Twins")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained checkpoint path")
    parser.add_argument("--finetuned", type=str, required=True, help="Finetuned checkpoint path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--acic-data", type=str, default=None, help="ACIC path (default: data/acic_sample.csv)")
    parser.add_argument("--max-samples", type=int, default=500, help="Subsample Twins/ACIC (0=full)")
    parser.add_argument("--output", type=str, default="results/pretrained_vs_finetuned_comparison.json")
    args = parser.parse_args()

    if not Path(args.pretrained).exists():
        print(f"Pretrained checkpoint not found: {args.pretrained}")
        return 1
    if not Path(args.finetuned).exists():
        print(f"Finetuned checkpoint not found: {args.finetuned}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    acic_path = args.acic_data or str(repo_root / "data" / "acic_sample.csv")
    max_s = args.max_samples if args.max_samples > 0 else None

    from experiments.benchmarks.real_world_suite import RealWorldBenchmark

    results = {"pretrained": {}, "finetuned": {}}
    for name, ckpt in [("pretrained", args.pretrained), ("finetuned", args.finetuned)]:
        benchmark = RealWorldBenchmark(ckpt, device=args.device)
        pehe_ihdp, ate_ihdp = [], []
        pehe_acic, ate_acic = [], []
        pehe_twins, ate_twins = [], []

        for seed in args.seeds:
            # IHDP
            r = benchmark.evaluate_ihdp(seed=seed)
            if "Ours" in r and "error" not in r["Ours"]:
                if r["Ours"].get("PEHE") is not None:
                    pehe_ihdp.append(r["Ours"]["PEHE"])
                ate_ihdp.append(r["Ours"]["ATE_Err"])

            # ACIC
            if Path(acic_path).exists():
                r = benchmark.evaluate_acic(data_path=acic_path, seed=seed, max_samples=max_s)
                if "Ours" in r and "error" not in r["Ours"]:
                    if r["Ours"].get("PEHE") is not None:
                        pehe_acic.append(r["Ours"]["PEHE"])
                    ate_acic.append(r["Ours"]["ATE_Err"])

            # Twins
            r = benchmark.evaluate_twins(seed=seed, max_samples=max_s)
            if "Ours" in r and "error" not in r["Ours"]:
                if r["Ours"].get("PEHE") is not None:
                    pehe_twins.append(r["Ours"]["PEHE"])
                ate_twins.append(r["Ours"]["ATE_Err"])

        results[name] = {
            "ihdp": {"PEHE_mean": float(np.mean(pehe_ihdp)) if pehe_ihdp else None, "PEHE_std": float(np.std(pehe_ihdp)) if pehe_ihdp else None, "ATE_Err_mean": float(np.mean(ate_ihdp)) if ate_ihdp else None, "ATE_Err_std": float(np.std(ate_ihdp)) if ate_ihdp else None},
            "acic": {"PEHE_mean": float(np.mean(pehe_acic)) if pehe_acic else None, "PEHE_std": float(np.std(pehe_acic)) if pehe_acic else None, "ATE_Err_mean": float(np.mean(ate_acic)) if ate_acic else None, "ATE_Err_std": float(np.std(ate_acic)) if ate_acic else None},
            "twins": {"PEHE_mean": float(np.mean(pehe_twins)) if pehe_twins else None, "PEHE_std": float(np.std(pehe_twins)) if pehe_twins else None, "ATE_Err_mean": float(np.mean(ate_twins)) if ate_twins else None, "ATE_Err_std": float(np.std(ate_twins)) if ate_twins else None},
        }

    # Print side-by-side table
    print("\n" + "=" * 80)
    print("PRETRAINED vs FINETUNED COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Pretrained':<25} | {'Finetuned':<25}")
    print("-" * 80)
    for ds in ["ihdp", "acic", "twins"]:
        for metric in ["PEHE", "ATE_Err"]:
            p = results["pretrained"][ds]
            f = results["finetuned"][ds]
            pm = p.get(f"{metric}_mean")
            fm = f.get(f"{metric}_mean")
            if pm is not None or fm is not None:
                ps = p.get(f"{metric}_std") or 0
                fs = f.get(f"{metric}_std") or 0
                p_str = f"{pm:.4f} ± {ps:.4f}" if pm is not None else "N/A"
                f_str = f"{fm:.4f} ± {fs:.4f}" if fm is not None else "N/A"
                print(f"{ds} {metric:<14} | {p_str:<25} | {f_str:<25}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
