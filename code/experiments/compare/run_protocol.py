"""Run comparison protocol: same eval data and metrics for our model and baseline(s).

Generates fixed eval data (one stage, seed), runs each method on the same batches,
computes same metrics, writes results/method_<name>_seed{N}.json per method.
Methods: ours (requires --checkpoint), mean_stub, outcome (ridge), dr, bart.

Protocol spec: Input = curriculum stage, seed, num_batches. Output per method =
baseline_rmse, baseline_mae, baseline_r2, cf_rmse, cf_mae, delta_rmse, delta_mae,
delta_correlation, ate_mae, calibration_ratio, coverage_95. Methods return dict
with 'prediction' [B,W,Nq] and 'log_var' [B,W,Nq]; MetricsComputer computes metrics.

Usage:
  python -m experiments.compare.run_protocol --checkpoint checkpoints/final_model.pt --methods ours,mean_stub,outcome --stage stage_1_basic --output-dir results
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root / "code"))

from episodes.config import CurriculumConfig
from episodes.dataset import create_dataloader
from train.metrics import MetricsComputer
from experiments.baselines.stub import MeanBaselineStub
from experiments.baselines.outcome_baseline import OutcomeBaseline
from experiments.baselines.dr_baseline import DRBaseline
from experiments.baselines.bart_baseline import BARTBaseline
from utils.checkpoint import load_model_from_checkpoint
from utils.metrics import mean_std, KEY_EVAL_METRICS


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
        "--methods",
        type=str,
        default="ours,mean_stub",
        help="Comma-separated list: ours, mean_stub, outcome. ours requires --checkpoint.",
    )
    parser.add_argument(
        "--ablate-no-cross-world",
        action="store_true",
        help="When loading ours, set cross_world_layers=[] (no cross-world attention at eval).",
    )
    args = parser.parse_args()

    stages = {s.name: s for s in CurriculumConfig.get_default_curriculum()}
    if args.stage not in stages:
        raise ValueError(f"Unknown stage: {args.stage}. Choose from {list(stages.keys())}")
    stage = stages[args.stage]

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "ours" in method_names and not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    seeds = args.seeds if args.seeds else [args.seed]

    # Build method callables (load ours once)
    methods_to_run = {}
    if "ours" in method_names:
        methods_to_run["ours"] = load_model_from_checkpoint(
            args.checkpoint, str(device), ablate_no_cross_world=args.ablate_no_cross_world
        )
    if "mean_stub" in method_names:
        methods_to_run["mean_stub"] = MeanBaselineStub(device=str(device))
    if "outcome" in method_names:
        methods_to_run["outcome"] = OutcomeBaseline(device=str(device))
    if "dr" in method_names:
        methods_to_run["dr"] = DRBaseline(device=str(device))
    if "bart" in method_names:
        methods_to_run["bart"] = BARTBaseline(device=str(device))

    results_per_method = {name: [] for name in methods_to_run}

    for seed in seeds:
        for method_name, method in methods_to_run.items():
            dataloader = create_dataloader(
                stage,
                batch_size=8,
                num_workers=0,
                seed=seed,
                rank=0,
                world_size=1,
                pin_memory=False,
            )
            metrics = run_method_on_dataloader(method, dataloader, device, args.num_batches)
            metrics["method"] = method_name
            if method_name == "ours":
                metrics["checkpoint"] = args.checkpoint
            metrics["stage"] = args.stage
            metrics["seed"] = seed
            out_path = os.path.join(args.output_dir, f"method_{method_name}_seed{seed}.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            results_per_method[method_name].append(metrics)
            print(f"{method_name} seed={seed}: delta_corr={metrics.get('delta_correlation')} -> {out_path}")

    # Summary mean±std across seeds per method
    for method_name, per_seed in results_per_method.items():
        summary = {"method": method_name, "stage": args.stage, "seeds": list(seeds)}
        if method_name == "ours":
            summary["checkpoint"] = args.checkpoint
        for k in KEY_EVAL_METRICS:
            m, s = mean_std([x.get(k) for x in per_seed])
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"] = s
        out_sum = os.path.join(args.output_dir, f"method_{method_name}_summary.json")
        with open(out_sum, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Results in {args.output_dir}")


if __name__ == "__main__":
    main()
