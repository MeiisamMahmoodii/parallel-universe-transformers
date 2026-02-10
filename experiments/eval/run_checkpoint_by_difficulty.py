"""Evaluate one or more checkpoints on fixed eval sets per curriculum difficulty.

Fixed seeds ensure comparable results across checkpoints. Writes full metrics
(baseline, CF, delta, delta_correlation, ATE, calibration) to JSON per (checkpoint, difficulty, seed),
plus a summary JSON with mean±std across seeds.

Usage:
  python -m experiments.eval.run_checkpoint_by_difficulty --checkpoint checkpoints/checkpoint_step_5000.pt --output-dir results
  python -m experiments.eval.run_checkpoint_by_difficulty --checkpoint ckpt1.pt ckpt2.pt --difficulties all --seed 42 --num-batches 200
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.model import ParallelUniverseTransformer
from episodes.config import CurriculumConfig
from episodes.dataset import create_dataloader
from train.metrics import MetricsComputer


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Build model from checkpoint config and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = ParallelUniverseTransformer(
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1024),
        dropout=config.get("dropout", 0.1),
        cross_world_layers=config.get("cross_world_layers", [3, 5]),
        attend_to_all_worlds=config.get("attend_to_all_worlds", True),
        use_quantiles=config.get("use_quantiles", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def run_eval_for_stage(model, stage, device, seed: int, num_batches: int):
    """Run evaluation for one curriculum stage; return metrics dict."""
    dataloader = create_dataloader(
        stage,
        batch_size=8,
        num_workers=0,
        seed=seed,
        rank=0,
        world_size=1,
        pin_memory=False,
    )
    metrics_computer = MetricsComputer()

    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            support_x = batch["support_x"].to(device)
            support_y = batch["support_y"].to(device)
            query_x = batch["query_x"].to(device)
            query_y = batch["query_y"].to(device)
            feature_types = batch["feature_types"][0].to(device)
            cardinalities = batch["cardinalities"][0].to(device)
            support_mask = batch["support_mask"].to(device)
            query_mask = batch["query_mask"].to(device)
            query_lengths = batch["query_lengths"].to(device)

            B, W, max_Nq = query_y.shape
            loss_mask = (
                (torch.arange(max_Nq, device=device) < query_lengths.unsqueeze(1))
                .float()
                .unsqueeze(1)
                .expand(B, W, max_Nq)
            )

            out = model(
                support_x,
                support_y,
                query_x,
                feature_types,
                cardinalities,
                support_mask,
                query_mask,
            )
            metrics_computer.update(
                out["prediction"],
                query_y,
                out["log_var"],
                loss_mask=loss_mask,
            )
            batch_count += 1

    metrics = metrics_computer.compute_and_reset()
    # Convert to JSON-serializable (nan -> None)
    out = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = float(v)
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    # Optional: PEHE (effect MSE). Note: delta_rmse == sqrt(PEHE) when PEHE is defined on individual effects.
    if out.get("delta_rmse") is not None:
        out["pehe"] = float(out["delta_rmse"]) ** 2
    return out


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
        description="Evaluate checkpoint(s) on fixed eval sets per curriculum difficulty"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write JSON results (default: results)",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="*",
        default=["all"],
        help="Stage names (e.g. stage_0_warmup stage_1_basic) or 'all' for all stages",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42],
        help="One or more eval seeds (default: 42). Example: --seeds 42 43 44",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=200,
        help="Number of batches to evaluate per difficulty (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    stages = CurriculumConfig.get_default_curriculum()
    if args.difficulties and args.difficulties != ["all"]:
        stage_map = {s.name: s for s in stages}
        stages = [stage_map[n] for n in args.difficulties if n in stage_map]
        if len(stages) == 0:
            raise ValueError(f"Unknown difficulties: {args.difficulties}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    for ckpt_path in args.checkpoint:
        if not Path(ckpt_path).exists():
            print(f"Skip (not found): {ckpt_path}")
            continue
        ckpt_name = Path(ckpt_path).stem
        print(f"Checkpoint: {ckpt_path}")
        model = load_model_from_checkpoint(ckpt_path, str(device))

        for stage in stages:
            per_seed = []
            for seed in args.seeds:
                metrics = run_eval_for_stage(
                    model, stage, device, seed, args.num_batches
                )
                metrics["checkpoint"] = ckpt_path
                metrics["difficulty"] = stage.name
                metrics["n_features"] = stage.n_features
                metrics["n_interventions"] = stage.n_interventions
                metrics["seed"] = seed
                out_name = f"{ckpt_name}_{stage.name}_seed{seed}.json"
                out_path = os.path.join(args.output_dir, out_name)
                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                per_seed.append(metrics)
                dc = metrics.get("delta_correlation")
                dc_str = f"{dc:.4f}" if isinstance(dc, (int, float)) else "n/a"
                print(f"  {stage.name} seed={seed}: delta_corr={dc_str} -> {out_path}")

            # Summary across seeds (mean±std for key metrics)
            summary = {
                "checkpoint": ckpt_path,
                "difficulty": stage.name,
                "n_features": stage.n_features,
                "n_interventions": stage.n_interventions,
                "seeds": list(args.seeds),
            }
            key_metrics = [
                "delta_correlation",
                "delta_mae",
                "delta_rmse",
                "pehe",
                "ate_mae",
                "baseline_r2",
                "baseline_mae",
                "cf_mae",
                "calibration_ratio",
                "coverage_95",
                "sharpness",
            ]
            for k in key_metrics:
                m, s = _mean_std([m.get(k) for m in per_seed])
                summary[f"{k}_mean"] = m
                summary[f"{k}_std"] = s
            sum_name = f"{ckpt_name}_{stage.name}_summary.json"
            sum_path = os.path.join(args.output_dir, sum_name)
            with open(sum_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  {stage.name}: summary -> {sum_path}")

    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
