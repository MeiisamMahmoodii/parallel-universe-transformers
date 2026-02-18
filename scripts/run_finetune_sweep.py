#!/usr/bin/env python3
"""Run hyperparameter sweep over IHDP fine-tuning (lr, max_steps, weight_decay).

Runs finetune_ihdp for each config in the grid, tracks best val PEHE, saves best checkpoint.

Usage:
  PYTHONPATH=code uv run python scripts/run_finetune_sweep.py \
    --resume-from checkpoints/checkpoint_step_40000.pt \
    --lr 1e-5 5e-5 --max-steps 3000 5000 --weight-decay 0.01 0.1 \
    --output-dir checkpoints/sweep --device cuda
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for IHDP fine-tuning")
    parser.add_argument("--resume-from", type=str, required=True, help="Checkpoint to load")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-5, 5e-5], help="Learning rates to try")
    parser.add_argument("--max-steps", type=int, nargs="+", default=[3000, 5000], help="Max steps to try")
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.01, 0.1], help="Weight decay values to try")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sweep", help="Directory for sweep checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-eval", action="store_true", help="Run full benchmark with best checkpoint after sweep")
    args = parser.parse_args()

    if not Path(args.resume_from).exists():
        print(f"Checkpoint not found: {args.resume_from}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.lr, args.max_steps, args.weight_decay))
    results = []

    for i, (lr, max_steps, wd) in enumerate(grid):
        lr_str = f"{lr:.0e}".replace("-", "m").replace(".", "p")
        tag = f"lr{lr_str}_steps{max_steps}_wd{wd}"
        output_pt = out_dir / f"finetuned_{tag}.pt"
        print(f"\n[{i+1}/{len(grid)}] lr={lr}, max_steps={max_steps}, weight_decay={wd}")
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "finetune_ihdp.py"),
            "--resume-from", args.resume_from,
            "--output", str(output_pt),
            "--lr", str(lr),
            "--max-steps", str(max_steps),
            "--weight-decay", str(wd),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        ret = subprocess.run(cmd, cwd=repo_root, env={**__import__("os").environ, "PYTHONPATH": str(repo_root / "code")})
        if ret.returncode != 0:
            print(f"  FAILED (exit {ret.returncode})")
            results.append({"lr": lr, "max_steps": max_steps, "weight_decay": wd, "best_val_pehe": None, "path": None})
            continue

        metrics_path = Path(str(output_pt) + ".metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            best_pehe = m.get("best_val_pehe")
            results.append({"lr": lr, "max_steps": max_steps, "weight_decay": wd, "best_val_pehe": best_pehe, "path": str(output_pt)})
            print(f"  best_val_pehe={best_pehe}")
        else:
            results.append({"lr": lr, "max_steps": max_steps, "weight_decay": wd, "best_val_pehe": None, "path": str(output_pt)})

    valid = [r for r in results if r["best_val_pehe"] is not None]
    best = min(valid, key=lambda r: r["best_val_pehe"]) if valid else None
    if not valid:
        print("\nNo valid runs with val PEHE. Cannot pick best.")
    elif best:
        print(f"\nBest: lr={best['lr']}, max_steps={best['max_steps']}, weight_decay={best['weight_decay']} -> val PEHE={best['best_val_pehe']:.4f}")
        print(f"Checkpoint: {best['path']}")

    sweep_summary = {"results": results, "best": best}
    summary_path = out_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"\nSweep summary: {summary_path}")
    if args.run_eval and best:
        print("\nRunning full benchmark with best checkpoint...")
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_full_benchmark.py"),
            "--checkpoint", best["path"],
            "--output-dir", str(repo_root / "results"),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, cwd=repo_root, env={**__import__("os").environ, "PYTHONPATH": str(repo_root / "code")})

    return 0


if __name__ == "__main__":
    sys.exit(main())
