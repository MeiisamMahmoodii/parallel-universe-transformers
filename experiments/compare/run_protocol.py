"""Run comparison protocol: same eval data and metrics for our model and baseline(s).

Generates fixed eval data (one stage, seed), runs "ours" (checkpoint) and "baseline" (stub)
on the same batches, computes same metrics, writes results/method_ours.json and
results/method_baseline.json. Add Do-PFN later by implementing the same predict interface.

Usage:
  python -m experiments.compare.run_protocol --checkpoint checkpoints/final_model.pt --stage stage_1_basic --output-dir results
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.model import ParallelUniverseTransformer
from episodes.config import CurriculumConfig
from episodes.dataset import create_dataloader
from train.metrics import MetricsComputer
from experiments.baselines.stub import MeanBaselineStub


def load_ours(checkpoint_path: str, device: str):
    """Load our model from checkpoint."""
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


def run_method_on_dataloader(method, dataloader, device, num_batches: int):
    """Run a method (ours or baseline) on batches; method(support_x, support_y, query_x, ...) -> dict with prediction, log_var."""
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

            out = method(
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
    out = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = float(v)
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
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
        description="Run comparison protocol: ours vs baseline on same eval data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to our model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write method_ours.json and method_baseline.json",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage_1_basic",
        help="Curriculum stage name for eval (e.g. stage_0_warmup, stage_1_basic)",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only run ours, skip baseline stub",
    )
    args = parser.parse_args()

    stages = {s.name: s for s in CurriculumConfig.get_default_curriculum()}
    if args.stage not in stages:
        raise ValueError(f"Unknown stage: {args.stage}. Choose from {list(stages.keys())}")
    stage = stages[args.stage]

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    seeds = args.seeds if args.seeds else [args.seed]

    # Load ours once (weights same across seeds)
    model_ours = load_ours(args.checkpoint, str(device))

    ours_per_seed = []
    baseline_per_seed = []

    for seed in seeds:
        dataloader = create_dataloader(
            stage,
            batch_size=8,
            num_workers=0,
            seed=seed,
            rank=0,
            world_size=1,
        )
        metrics_ours = run_method_on_dataloader(model_ours, dataloader, device, args.num_batches)
        metrics_ours["method"] = "ours"
        metrics_ours["checkpoint"] = args.checkpoint
        metrics_ours["stage"] = args.stage
        metrics_ours["seed"] = seed
        out_ours = os.path.join(args.output_dir, f"method_ours_seed{seed}.json")
        with open(out_ours, "w") as f:
            json.dump(metrics_ours, f, indent=2)
        ours_per_seed.append(metrics_ours)
        print(f"Ours seed={seed}: delta_corr={metrics_ours.get('delta_correlation')} -> {out_ours}")

        if not args.skip_baseline:
            dataloader2 = create_dataloader(
                stage,
                batch_size=8,
                num_workers=0,
                seed=seed,
                rank=0,
                world_size=1,
            )
            baseline = MeanBaselineStub(device=str(device))
            metrics_baseline = run_method_on_dataloader(baseline, dataloader2, device, args.num_batches)
            metrics_baseline["method"] = "baseline_stub"
            metrics_baseline["stage"] = args.stage
            metrics_baseline["seed"] = seed
            out_baseline = os.path.join(args.output_dir, f"method_baseline_seed{seed}.json")
            with open(out_baseline, "w") as f:
                json.dump(metrics_baseline, f, indent=2)
            baseline_per_seed.append(metrics_baseline)
            print(f"Baseline seed={seed}: delta_corr={metrics_baseline.get('delta_correlation')} -> {out_baseline}")

    # Summary mean±std across seeds (key metrics)
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
    ours_summary = {
        "method": "ours",
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "seeds": list(seeds),
    }
    for k in key_metrics:
        m, s = _mean_std([m.get(k) for m in ours_per_seed])
        ours_summary[f"{k}_mean"] = m
        ours_summary[f"{k}_std"] = s
    out_ours_sum = os.path.join(args.output_dir, "method_ours_summary.json")
    with open(out_ours_sum, "w") as f:
        json.dump(ours_summary, f, indent=2)

    if baseline_per_seed:
        base_summary = {"method": "baseline_stub", "stage": args.stage, "seeds": list(seeds)}
        for k in key_metrics:
            m, s = _mean_std([m.get(k) for m in baseline_per_seed])
            base_summary[f"{k}_mean"] = m
            base_summary[f"{k}_std"] = s
        out_base_sum = os.path.join(args.output_dir, "method_baseline_summary.json")
        with open(out_base_sum, "w") as f:
            json.dump(base_summary, f, indent=2)

    print(f"Results in {args.output_dir}")


if __name__ == "__main__":
    main()
