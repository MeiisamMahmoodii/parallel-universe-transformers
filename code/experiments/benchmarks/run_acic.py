"""Run our model (and optional baselines) on ACIC for treatment-effect metrics.

ACIC is binary treatment (W=2). Same interface as run_ihdp: support = train (X, y), query = test with two worlds.
Reports PEHE, delta_rmse, delta_correlation, ATE error.

Usage:
  python -m experiments.benchmarks.run_acic --checkpoint checkpoints/final_model.pt --data path/to/acic.csv
  python -m experiments.benchmarks.run_acic --checkpoint checkpoints/final_model.pt --data path/to/acic_dir  # directory with x.csv + zymu_*.csv
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root / "code"))

from model.model import ParallelUniverseTransformer
from train.metrics import MetricsComputer
from experiments.benchmarks.acic_data import load_acic
from experiments.baselines.outcome_baseline import OutcomeBaseline


def load_model(checkpoint_path: str, device: str):
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


def run_acic_eval(
    checkpoint_path: str,
    data_path: str,
    device: str = "cuda",
    train_frac: float = 0.8,
    seed: int = 42,
    output_path: str = None,
    run_baseline: bool = True,
):
    """Load ACIC, run model (and optional outcome baseline), return metrics dict."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential = load_acic(
        path=data_path, train_frac=train_frac, seed=seed
    )
    Ns, d = support_x.shape
    Nq = query_x_w0.shape[0]
    support_x_t = torch.from_numpy(support_x).float().unsqueeze(0).to(device)
    support_y_t = torch.from_numpy(support_y).float().unsqueeze(0).to(device)
    query_x_t = torch.stack([
        torch.from_numpy(query_x_w0).float(),
        torch.from_numpy(query_x_w1).float(),
    ], dim=0).unsqueeze(0).to(device)
    query_y_t = torch.stack([
        torch.from_numpy(query_y0).float(),
        torch.from_numpy(query_y1).float(),
    ], dim=0).unsqueeze(0).to(device)
    feature_types = torch.zeros(d, dtype=torch.long, device=device)
    cardinalities = torch.ones(d, dtype=torch.long, device=device)
    support_mask = torch.ones(1, Ns, device=device)
    query_mask = torch.ones(1, 2, Nq, device=device)
    loss_mask = torch.ones(1, 2, Nq, device=device)
    results = {}

    if Path(checkpoint_path).exists():
        model = load_model(checkpoint_path, str(device))
        with torch.no_grad():
            out = model(
                support_x_t, support_y_t, query_x_t,
                feature_types, cardinalities, support_mask, query_mask,
            )
        pred = out["prediction"].cpu().numpy()
        metrics_computer = MetricsComputer()
        metrics_computer.update(out["prediction"], query_y_t, out["log_var"], loss_mask=loss_mask)
        metrics = metrics_computer.compute()
        for k, v in metrics.items():
            v = float(v) if hasattr(v, "item") else v
            results[k] = None if isinstance(v, float) and math.isnan(v) else v
        if results.get("delta_rmse") is not None:
            results["pehe"] = float(results["delta_rmse"]) ** 2
        if has_potential:
            pred_delta = pred[0, 1, :] - pred[0, 0, :]
            true_delta = query_y1 - query_y0
            results["pehe_ite"] = float(np.mean((pred_delta - true_delta) ** 2))
            results["sqrt_pehe_ite"] = float(np.sqrt(results["pehe_ite"]))
        results["method"] = "ours"
        results["checkpoint"] = checkpoint_path
        results["has_potential_outcomes"] = has_potential
        results["N_support"] = Ns
        results["N_query"] = Nq

    if run_baseline:
        baseline = OutcomeBaseline(device=str(device))
        with torch.no_grad():
            out_b = baseline(
                support_x_t, support_y_t, query_x_t,
                feature_types, cardinalities, support_mask, query_mask,
            )
        metrics_computer_b = MetricsComputer()
        metrics_computer_b.update(out_b["prediction"], query_y_t, out_b["log_var"], loss_mask=loss_mask)
        metrics_b = metrics_computer_b.compute()
        results_baseline = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics_b.items()}
        if results_baseline.get("delta_rmse") is not None:
            results_baseline["pehe"] = float(results_baseline["delta_rmse"]) ** 2
        if has_potential:
            pred_b = out_b["prediction"].cpu().numpy()
            pred_delta_b = pred_b[0, 1, :] - pred_b[0, 0, :]
            true_delta = query_y1 - query_y0
            results_baseline["pehe_ite"] = float(np.mean((pred_delta_b - true_delta) ** 2))
            results_baseline["sqrt_pehe_ite"] = float(np.sqrt(results_baseline["pehe_ite"]))
        results["baseline_outcome"] = results_baseline

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run model on ACIC benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to ACIC CSV or directory (x.csv + zymu_*.csv)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    results = run_acic_eval(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        device=args.device,
        train_frac=args.train_frac,
        seed=args.seed,
        output_path=args.output,
        run_baseline=not args.no_baseline,
    )
    print("ACIC results:")
    print(f"  delta_correlation: {results.get('delta_correlation')}")
    print(f"  delta_rmse (sqrt(PEHE)): {results.get('delta_rmse')}")
    print(f"  pehe: {results.get('pehe')}")
    if results.get("pehe_ite") is not None:
        print(f"  pehe_ite: {results.get('pehe_ite')}")
    if args.output:
        print(f"  Written to {args.output}")


if __name__ == "__main__":
    main()
