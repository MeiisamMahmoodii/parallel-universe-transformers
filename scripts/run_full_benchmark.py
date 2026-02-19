#!/usr/bin/env python3
"""Run full benchmark: IHDP / ACIC / Twins (all models) + optional Synthetic (Ours only, if checkpoint provided).

What we test:
- IHDP: Infant Health and Development Program, 80/20 train-test, PEHE and ATE error.
- ACIC: Semi-synthetic binary treatment (requires --acic-data or data/acic_sample.csv).
- Twins: Twin births dataset (observational CATE benchmark).
  Models: Ours (if checkpoint), Linear-T, Linear-S, GB-T, GB-S, DR-Linear, DR-GB,
  TabPFN (if installed), TransTEE, Dragonnet, Ridge.
- Synthetic (only if checkpoint): SCM types linear_gaussian, nonlinear_additive, etc. Ours only.

Usage:
  PYTHONPATH=code uv run python scripts/run_full_benchmark.py --output-dir results
  PYTHONPATH=code uv run python scripts/run_full_benchmark.py --datasets ihdp acic twins --output-dir results
  PYTHONPATH=code uv run python scripts/run_full_benchmark.py --checkpoint checkpoints/final_model.pt --acic-data data/acic_sample.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite (IHDP, ACIC, Twins, Synthetic)")
    parser.add_argument("--checkpoint", type=str, default="experiments/results/exp_10_ihdp_light_reg/checkpoints/finetuned_seed42.pt", help="Path to our model .pt")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for result JSONs")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or auto (default: cuda if available else cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test splits")
    parser.add_argument("--skip-synthetic", action="store_true", help="Do not run synthetic benchmark even if checkpoint exists")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ihdp"], choices=["ihdp", "acic", "twins"], help="Which real-world datasets to run")
    parser.add_argument("--acic-data", type=str, default=None, help="Path to ACIC CSV or directory (default: data/acic_sample.csv if exists)")
    parser.add_argument("--max-samples", type=int, default=500, help="Subsample Twins/ACIC for faster runs (default: 500; use 0 for full dataset)")
    parser.add_argument("--compare-pretrained", action="store_true", help="Compare pretrained vs finetuned; requires --finetuned-checkpoint")
    parser.add_argument("--finetuned-checkpoint", type=str, default=None, help="Finetuned checkpoint for comparison (with --compare-pretrained)")
    parser.add_argument("--scale-outcome", action="store_true", help="For Ours: scale outcomes at eval (use when checkpoint was trained with --scale-outcome)")
    parser.add_argument("--ensemble", type=str, default=None, help="Add ensemble: e.g. ours,gbs (average predictions of Ours and GB-S)")
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU.")
    elif device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    max_samples = args.max_samples if args.max_samples > 0 else None
    if max_samples:
        print(f"Using --max-samples {max_samples} for Twins/ACIC (faster). Use --max-samples 0 for full dataset.")

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we can import from code
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    os.chdir(repo_root)

    # If compare mode, delegate to compare script
    if args.compare_pretrained and args.finetuned_checkpoint:
        compare_script = repo_root / "scripts" / "compare_pretrained_vs_finetuned.py"
        if compare_script.exists():
            import subprocess
            acic_p = args.acic_data or str(repo_root / "data" / "acic_sample.csv")
            cmd = [
                sys.executable, str(compare_script),
                "--pretrained", args.checkpoint,
                "--finetuned", args.finetuned_checkpoint,
                "--device", device,
                "--acic-data", acic_p,
                "--max-samples", str(args.max_samples),
                "--output", str(out_dir / "pretrained_vs_finetuned_comparison.json"),
            ]
            print("Running pretrained vs finetuned comparison...")
            return subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": str(code_dir)}).returncode
        else:
            print("Error: compare_pretrained_vs_finetuned.py not found")
            return 1
    elif args.compare_pretrained:
        print("Error: --compare-pretrained requires --finetuned-checkpoint")
        return 1
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "checkpoint_exists": args.checkpoint is not None and Path(args.checkpoint).exists(),
        "ihdp": {},
        "acic": {},
        "twins": {},
        "synthetic": {},
        "what_was_tested": {
            "ihdp": "IHDP dataset, 80/20 split, PEHE and ATE error.",
            "acic": "ACIC semi-synthetic, 80/20 split, PEHE and ATE error.",
            "twins": "Twins dataset, 80/20 split, PEHE and ATE error.",
            "synthetic": "Synthetic SCMs. Ours only; requires checkpoint."
        }
    }

    from experiments.benchmarks.real_world_suite import RealWorldBenchmark
    benchmark = RealWorldBenchmark(args.checkpoint, device=device)

    # 1. IHDP (if selected)
    if "ihdp" in args.datasets:
        print("\n" + "=" * 60)
        print("1. IHDP BENCHMARK (all models)")
        print("=" * 60)
        try:
            ihdp_results = benchmark.evaluate_ihdp(seed=args.seed, scale_outcome_for_ours=args.scale_outcome)
            if args.ensemble:
                parts = [p.strip().lower() for p in args.ensemble.split(",")]
                ensemble_result = benchmark.evaluate_ihdp_ensemble(
                    seed=args.seed,
                    scale_outcome_for_ours=args.scale_outcome,
                    model_names=parts,
                )
                if ensemble_result:
                    name = "Ensemble(" + ",".join(p for p in parts) + ")"
                    ihdp_results[name] = ensemble_result
            benchmark.print_summary(ihdp_results, dataset_name="IHDP")
            summary["ihdp"] = ihdp_results
            with open(out_dir / "ihdp_results.json", "w") as f:
                json.dump(ihdp_results, f, indent=2)
        except Exception as e:
            summary["ihdp"] = {"error": str(e)}
            print(f"IHDP benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # 2. ACIC (if selected; requires data path)
    if "acic" in args.datasets:
        acic_path = args.acic_data or str(repo_root / "data" / "acic_sample.csv")
        if Path(acic_path).exists():
            print("\n" + "=" * 60)
            print("2. ACIC BENCHMARK (all models)")
            print("=" * 60)
            try:
                acic_results = benchmark.evaluate_acic(data_path=acic_path, seed=args.seed, max_samples=max_samples)
                benchmark.print_summary(acic_results, dataset_name="ACIC")
                summary["acic"] = acic_results
                with open(out_dir / "acic_results.json", "w") as f:
                    json.dump(acic_results, f, indent=2)
            except Exception as e:
                summary["acic"] = {"error": str(e)}
                print(f"ACIC benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            summary["acic"] = {"error": f"ACIC data not found: {acic_path}. Run scripts/download_datasets.py or use --acic-data path"}

    # 3. Twins (if selected)
    if "twins" in args.datasets:
        print("\n" + "=" * 60)
        print("3. TWINS BENCHMARK (all models)")
        print("=" * 60)
        try:
            twins_results = benchmark.evaluate_twins(seed=args.seed, max_samples=max_samples)
            benchmark.print_summary(twins_results, dataset_name="Twins")
            summary["twins"] = twins_results
            with open(out_dir / "twins_results.json", "w") as f:
                json.dump(twins_results, f, indent=2)
        except Exception as e:
            summary["twins"] = {"error": str(e)}
            print(f"Twins benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # 4. Synthetic (only if checkpoint and not --skip-synthetic)
    if summary["checkpoint_exists"] and not args.skip_synthetic:
        print("\n" + "=" * 60)
        print("4. SYNTHETIC BENCHMARK (Ours)")
        print("=" * 60)
        try:
            from experiments.benchmarks.synthetic_suite import SyntheticBenchmark
            syn_bench = SyntheticBenchmark(args.checkpoint, device=device)
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
            print("\n4. SYNTHETIC: skipped (no checkpoint).")
        else:
            print("\n4. SYNTHETIC: skipped (--skip-synthetic).")

    summary_path = out_dir / "full_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
