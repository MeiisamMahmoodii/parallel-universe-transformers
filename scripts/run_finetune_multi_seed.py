#!/usr/bin/env python3
"""Run fine-tuning for multiple seeds and report mean ± std for PEHE and ATE Err.

For each seed: finetune with that seed, evaluate on IHDP/ACIC/Twins with the same seed
(for consistent splits), then aggregate Ours metrics across seeds.

Usage:
  PYTHONPATH=code uv run python scripts/run_finetune_multi_seed.py \
    --resume-from checkpoints/checkpoint_step_40000.pt \
    --seeds 42 123 456 --device cuda

  # Multi-dataset finetuning (IHDP + ACIC + Twins)
  PYTHONPATH=code uv run python scripts/run_finetune_multi_seed.py \
    --resume-from checkpoints/checkpoint_step_40000.pt \
    --datasets ihdp acic twins --mix-ratio 0.5 0.25 0.25 \
    --acic-data data/acic_sample.csv --seeds 42 123 456
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Multi-seed fine-tuning and evaluation")
    parser.add_argument("--resume-from", type=str, required=True, help="Checkpoint to load")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds (default: 42, 123, 456)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for summary JSON")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for per-seed checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max finetuning steps (500-1000 for multi-dataset)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (5e-6 for multi-dataset)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay (0.1 for multi-dataset)")
    # Multi-dataset
    parser.add_argument("--datasets", type=str, nargs="+", default=["ihdp"], help="Datasets: ihdp, acic, twins")
    parser.add_argument("--mix-ratio", type=float, nargs="+", default=None, help="Mix ratio per dataset (e.g. 0.5 0.25 0.25)")
    parser.add_argument("--acic-data", type=str, default=None, help="Path to ACIC (default: data/acic_sample.csv if exists)")
    parser.add_argument("--max-samples", type=int, default=500, help="Subsample Twins/ACIC for faster eval (0=full)")
    parser.add_argument("--lambda-delta", type=float, default=None, help="Pass to finetune_ihdp (default: 2.0)")
    parser.add_argument("--synthetic-mix-ratio", type=float, default=0.0, help="Pass to finetune_ihdp (0-1, fraction synthetic)")
    parser.add_argument("--scale-outcome", action="store_true", help="Pass to finetune_ihdp for IHDP outcome scaling")
    args = parser.parse_args()

    if not Path(args.resume_from).exists():
        print(f"Checkpoint not found: {args.resume_from}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve ACIC path
    acic_path = args.acic_data
    if acic_path is None and "acic" in args.datasets:
        acic_default = repo_root / "data" / "acic_sample.csv"
        if acic_default.exists():
            acic_path = str(acic_default)
        else:
            print("Error: --acic-data required when 'acic' in --datasets. Run scripts/download_datasets.py")
            return 1

    mix_ratio = args.mix_ratio
    if len(args.datasets) > 1 and mix_ratio is None:
        mix_ratio = [1.0 / len(args.datasets)] * len(args.datasets)

    env = {**__import__("os").environ, "PYTHONPATH": str(code_dir)}
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    # Eval on same datasets we finetuned on; for acic need valid path
    eval_datasets = list(args.datasets)
    if "acic" in eval_datasets and (not acic_path or not Path(acic_path).exists()):
        eval_datasets = [d for d in eval_datasets if d != "acic"]
    if not eval_datasets:
        eval_datasets = ["ihdp"]

    per_seed = {}
    summary_by_dataset = {ds: {"PEHE": [], "ATE_Err": []} for ds in eval_datasets}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print("="*60)
        output_pt = ckpt_dir / f"finetuned_seed{seed}.pt"

        # 1. Finetune
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "finetune_ihdp.py"),
            "--resume-from", args.resume_from,
            "--output", str(output_pt),
            "--seed", str(seed),
            "--max-steps", str(args.max_steps),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--device", args.device,
        ]
        if len(args.datasets) > 1:
            cmd += ["--datasets"] + args.datasets
            if mix_ratio:
                cmd += ["--mix-ratio"] + [str(r) for r in mix_ratio]
            if acic_path:
                cmd += ["--acic-data", acic_path]
        if args.lambda_delta is not None:
            cmd += ["--lambda-delta", str(args.lambda_delta)]
        if args.synthetic_mix_ratio > 0:
            cmd += ["--synthetic-mix-ratio", str(args.synthetic_mix_ratio)]
        if args.scale_outcome:
            cmd += ["--scale-outcome"]
        ret = subprocess.run(cmd, cwd=repo_root, env=env)
        if ret.returncode != 0:
            print(f"Finetune FAILED for seed {seed}")
            continue

        # 2. Evaluate on each dataset
        from experiments.benchmarks.real_world_suite import RealWorldBenchmark

        benchmark = RealWorldBenchmark(str(output_pt), device=args.device)
        per_seed[str(seed)] = {}

        for ds in eval_datasets:
            try:
                if ds == "ihdp":
                    results = benchmark.evaluate_ihdp(seed=seed, scale_outcome_for_ours=args.scale_outcome)
                elif ds == "acic":
                    max_s = args.max_samples if args.max_samples > 0 else None
                    results = benchmark.evaluate_acic(data_path=acic_path, seed=seed, max_samples=max_s)
                else:
                    max_s = args.max_samples if args.max_samples > 0 else None
                    results = benchmark.evaluate_twins(seed=seed, max_samples=max_s)

                if "Ours" in results and "error" not in results["Ours"]:
                    pehe = results["Ours"].get("PEHE")
                    ate = results["Ours"]["ATE_Err"]
                    if pehe is not None and not (isinstance(pehe, float) and (pehe != pehe)):
                        summary_by_dataset[ds]["PEHE"].append(pehe)
                    summary_by_dataset[ds]["ATE_Err"].append(ate)
                    per_seed[str(seed)][ds] = {"PEHE": pehe, "ATE_Err": ate}
                    pehe_str = f"PEHE={pehe:.4f}, " if pehe is not None else ""
                    print(f"  {ds}: {pehe_str}ATE Err={ate:.4f}")
            except Exception as e:
                print(f"  {ds}: FAILED ({e})")

    # Aggregate
    import numpy as np
    summary = {"per_seed": per_seed}
    for ds in eval_datasets:
        pehe_list = summary_by_dataset[ds]["PEHE"]
        ate_list = summary_by_dataset[ds]["ATE_Err"]
        if pehe_list:
            summary[f"{ds}_PEHE_mean"] = float(np.mean(pehe_list))
            summary[f"{ds}_PEHE_std"] = float(np.std(pehe_list))
        if ate_list:
            summary[f"{ds}_ATE_Err_mean"] = float(np.mean(ate_list))
            summary[f"{ds}_ATE_Err_std"] = float(np.std(ate_list))

    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY (Ours)")
    print("="*60)
    for ds in eval_datasets:
        pehe_list = summary_by_dataset[ds]["PEHE"]
        ate_list = summary_by_dataset[ds]["ATE_Err"]
        if pehe_list:
            m, s = np.mean(pehe_list), np.std(pehe_list)
            print(f"{ds} PEHE:    {m:.4f} ± {s:.4f}")
        if ate_list:
            m, s = np.mean(ate_list), np.std(ate_list)
            print(f"{ds} ATE Err: {m:.4f} ± {s:.4f}")

    summary_path = out_dir / "finetune_multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
