"""Run evaluation for all checkpoints × all difficulties and save a matrix (CSV + JSON).

Scans checkpoints/ for *.pt or uses --checkpoint list. For each checkpoint and each
curriculum stage, runs run_checkpoint_by_difficulty logic and aggregates into
checkpoint × difficulty table (delta_correlation, delta_mae, baseline_mae, etc.).

Usage:
  python -m experiments.eval.run_all_checkpoints_matrix --checkpoint-dir checkpoints --output-dir results
  python -m experiments.eval.run_all_checkpoints_matrix --checkpoint ckpt1.pt ckpt2.pt --output-dir results
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.eval.run_checkpoint_by_difficulty import (
    load_model_from_checkpoint,
    run_eval_for_stage,
)
from episodes.config import CurriculumConfig


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints on all difficulties; output matrix CSV and JSON"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to scan for *.pt checkpoints (if --checkpoint not given)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="*",
        default=None,
        help="Explicit checkpoint path(s); overrides --checkpoint-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write matrix CSV and JSON (default: results)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional: multiple seeds. If provided, runs all seeds and outputs summary mean±std.",
    )
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_paths = [p for p in args.checkpoint if Path(p).exists()]
    else:
        cp_dir = Path(args.checkpoint_dir)
        if not cp_dir.exists():
            print(f"Checkpoint dir not found: {cp_dir}")
            return
        checkpoint_paths = sorted(cp_dir.glob("*.pt"))
        checkpoint_paths = [str(p) for p in checkpoint_paths]

    if not checkpoint_paths:
        print("No checkpoints found.")
        return

    stages = CurriculumConfig.get_default_curriculum()
    device = __import__("torch").device(args.device)
    seeds = args.seeds if args.seeds else [args.seed]

    # Collect rows: one per (checkpoint, difficulty)
    rows = []
    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem
        model = load_model_from_checkpoint(ckpt_path, str(device))
        for stage in stages:
            def _sanitize(v):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return None
                return v

            per_seed = []
            for seed in seeds:
                metrics = run_eval_for_stage(
                    model, stage, device, seed, args.num_batches
                )
                row = {
                    "checkpoint": ckpt_name,
                    "checkpoint_path": ckpt_path,
                    "difficulty": stage.name,
                    "n_features": stage.n_features,
                    "n_interventions": stage.n_interventions,
                    "seed": seed,
                    "delta_correlation": _sanitize(metrics.get("delta_correlation")),
                    "delta_mae": _sanitize(metrics.get("delta_mae")),
                    "delta_rmse": _sanitize(metrics.get("delta_rmse")),
                    "pehe": _sanitize(metrics.get("pehe")),
                    "baseline_mae": _sanitize(metrics.get("baseline_mae")),
                    "baseline_r2": _sanitize(metrics.get("baseline_r2")),
                    "cf_mae": _sanitize(metrics.get("cf_mae")),
                    "ate_mae": _sanitize(metrics.get("ate_mae")),
                }
                rows.append(row)
                per_seed.append(row)
                dc = row["delta_correlation"]
                print(
                    f"{ckpt_name} x {stage.name} seed={seed}: delta_corr={dc:.4f}"
                    if dc is not None
                    else f\"{ckpt_name} x {stage.name} seed={seed}: delta_corr=n/a\"
                )

            # Summary row (mean±std) if multiple seeds
            if len(seeds) > 1:
                summary = {
                    "checkpoint": ckpt_name,
                    "checkpoint_path": ckpt_path,
                    "difficulty": stage.name,
                    "n_features": stage.n_features,
                    "n_interventions": stage.n_interventions,
                    "seed": "summary",
                }
                for k in (
                    "delta_correlation",
                    "delta_mae",
                    "delta_rmse",
                    "pehe",
                    "baseline_mae",
                    "baseline_r2",
                    "cf_mae",
                    "ate_mae",
                ):
                    m, s = _mean_std([r.get(k) for r in per_seed])
                    summary[f"{k}_mean"] = m
                    summary[f"{k}_std"] = s
                rows.append(summary)

    os.makedirs(args.output_dir, exist_ok=True)

    # CSV: one row per (checkpoint, difficulty); columns = metrics
    csv_path = os.path.join(args.output_dir, "checkpoint_difficulty_matrix.csv")
    fieldnames = [
        "checkpoint",
        "difficulty",
        "n_features",
        "n_interventions",
        "seed",
        "delta_correlation",
        "delta_mae",
        "delta_rmse",
        "pehe",
        "baseline_mae",
        "baseline_r2",
        "cf_mae",
        "ate_mae",
        "delta_correlation_mean",
        "delta_correlation_std",
        "delta_mae_mean",
        "delta_mae_std",
        "delta_rmse_mean",
        "delta_rmse_std",
        "pehe_mean",
        "pehe_std",
        "baseline_r2_mean",
        "baseline_r2_std",
        "ate_mae_mean",
        "ate_mae_std",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                out[k] = "" if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            w.writerow(out)
    print(f"Matrix CSV: {csv_path}")

    # JSON: full rows (including checkpoint_path)
    json_path = os.path.join(args.output_dir, "checkpoint_difficulty_matrix.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Matrix JSON: {json_path}")


if __name__ == "__main__":
    main()
