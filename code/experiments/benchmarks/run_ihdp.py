"""Run our model (and optional baselines) on IHDP for comparable treatment-effect metrics.

IHDP is single-treatment: W=2 (do(T=0), do(T=1)). We map to our interface:
  support = train (X, y), query = test with two worlds (X with T=0, X with T=1).
Reports PEHE (and sqrt(PEHE)=delta_rmse), delta_correlation, ATE error.

Usage:
  python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/final_model.pt --data path/to/ihdp.csv
  python -m experiments.benchmarks.run_ihdp --checkpoint checkpoints/final_model.pt  # uses default URL if available
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

from train.metrics import MetricsComputer
from utils.checkpoint import load_model_from_checkpoint
from utils.metrics import mean_std, KEY_EVAL_METRICS
from experiments.benchmarks.ihdp_data import load_ihdp
from experiments.baselines.outcome_baseline import OutcomeBaseline
from experiments.baselines.stub import MeanBaselineStub
from experiments.baselines.dr_baseline import DRBaseline
from experiments.baselines.bart_baseline import BARTBaseline


def run_ihdp_eval(
    checkpoint_path: str,
    data_path: str = None,
    device: str = "cuda",
    train_frac: float = 0.8,
    seed: int = 42,
    output_path: str = None,
    run_baseline: bool = True,
):
    """Load IHDP, run model (and optional outcome baseline), return metrics dict."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, feature_names, has_potential = load_ihdp(
        path=data_path, train_frac=train_frac, seed=seed
    )
    Ns, d = support_x.shape
    Nq = query_x_w0.shape[0]

    # Batch: B=1, W=2, Nq
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

    # Our model
    if Path(checkpoint_path).exists():
        model = load_model_from_checkpoint(checkpoint_path, str(device))
        with torch.no_grad():
            out = model(
                support_x_t,
                support_y_t,
                query_x_t,
                feature_types,
                cardinalities,
                support_mask,
                query_mask,
            )
        pred = out["prediction"].cpu().numpy()
        log_var = out["log_var"].cpu().numpy()
        metrics_computer = MetricsComputer()
        metrics_computer.update(
            out["prediction"],
            query_y_t,
            out["log_var"],
            loss_mask=loss_mask,
        )
        metrics = metrics_computer.compute()
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = float(v)
            if isinstance(v, float) and math.isnan(v):
                v = None
            results[k] = v
        if results.get("delta_rmse") is not None:
            results["pehe"] = float(results["delta_rmse"]) ** 2
        if has_potential:
            pred_delta = pred[0, 1, :] - pred[0, 0, :]
            true_delta = query_y1 - query_y0
            pehe_true = np.mean((pred_delta - true_delta) ** 2)
            results["pehe_ite"] = float(pehe_true)
            results["sqrt_pehe_ite"] = float(np.sqrt(pehe_true))
        results["method"] = "ours"
        results["checkpoint"] = checkpoint_path
        results["has_potential_outcomes"] = has_potential
        results["N_support"] = Ns
        results["N_query"] = Nq

    # Outcome baseline (ridge on support)
    if run_baseline:
        baseline = OutcomeBaseline(device=str(device))
        with torch.no_grad():
            out_b = baseline(
                support_x_t,
                support_y_t,
                query_x_t,
                feature_types,
                cardinalities,
                support_mask,
                query_mask,
            )
        pred_b = out_b["prediction"].cpu().numpy()
        metrics_computer_b = MetricsComputer()
        metrics_computer_b.update(
            out_b["prediction"],
            query_y_t,
            out_b["log_var"],
            loss_mask=loss_mask,
        )
        metrics_b = metrics_computer_b.compute()
        results_baseline = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics_b.items()}
        if results_baseline.get("delta_rmse") is not None:
            results_baseline["pehe"] = float(results_baseline["delta_rmse"]) ** 2
        if has_potential:
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


def _run_one_method_ihdp(method, support_x_t, support_y_t, query_x_t, query_y_t,
                         feature_types, cardinalities, support_mask, query_mask,
                         loss_mask, has_potential, query_y0, query_y1, device):
    """Run a single method on IHDP tensors; return metrics dict (same keys as protocol)."""
    with torch.no_grad():
        out = method(
            support_x_t,
            support_y_t,
            query_x_t,
            feature_types,
            cardinalities,
            support_mask,
            query_mask,
        )
    metrics_computer = MetricsComputer()
    metrics_computer.update(
        out["prediction"],
        query_y_t,
        out["log_var"],
        loss_mask=loss_mask,
    )
    metrics = metrics_computer.compute()
    results = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = float(v)
        if isinstance(v, float) and math.isnan(v):
            v = None
        results[k] = v
    if results.get("delta_rmse") is not None:
        results["pehe"] = float(results["delta_rmse"]) ** 2
    if has_potential:
        pred = out["prediction"].cpu().numpy()
        pred_delta = pred[0, 1, :] - pred[0, 0, :]
        true_delta = query_y1 - query_y0
        results["pehe_ite"] = float(np.mean((pred_delta - true_delta) ** 2))
        results["sqrt_pehe_ite"] = float(np.sqrt(results["pehe_ite"]))
    return results


def run_ihdp_protocol(
    checkpoint_path: str,
    data_path: str = None,
    device: str = "cuda",
    train_frac: float = 0.8,
    seeds: list = None,
    output_dir: str = None,
    methods: list = None,
):
    """Run multiple methods on IHDP with same protocol as run_protocol (per-seed + summary)."""
    seeds = seeds or [42]
    methods = methods or ["ours", "mean_stub", "outcome"]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    method_callables = {}
    if "ours" in methods and checkpoint_path and Path(checkpoint_path).exists():
        method_callables["ours"] = load_model_from_checkpoint(checkpoint_path, str(device))
    if "mean_stub" in methods:
        method_callables["mean_stub"] = MeanBaselineStub(device=str(device))
    if "outcome" in methods:
        method_callables["outcome"] = OutcomeBaseline(device=str(device))
    if "dr" in methods:
        method_callables["dr"] = DRBaseline(device=str(device))
    if "bart" in methods:
        method_callables["bart"] = BARTBaseline(device=str(device))

    results_per_method = {name: [] for name in method_callables}

    for seed in seeds:
        support_x, support_y, query_x_w0, query_x_w1, query_y0, query_y1, _, has_potential = load_ihdp(
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

        for method_name, method in method_callables.items():
            metrics = _run_one_method_ihdp(
                method, support_x_t, support_y_t, query_x_t, query_y_t,
                feature_types, cardinalities, support_mask, query_mask,
                loss_mask, has_potential, query_y0, query_y1, device,
            )
            metrics["method"] = method_name
            if method_name == "ours":
                metrics["checkpoint"] = checkpoint_path
            metrics["dataset"] = "IHDP"
            metrics["seed"] = seed
            results_per_method[method_name].append(metrics)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"method_{method_name}_seed{seed}.json")
                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=2)

    if not output_dir:
        return results_per_method
    for method_name, per_seed in results_per_method.items():
        summary = {"method": method_name, "stage": "IHDP", "dataset": "IHDP", "seeds": list(seeds)}
        if method_name == "ours":
            summary["checkpoint"] = checkpoint_path
        for k in KEY_EVAL_METRICS:
            m, s = mean_std([x.get(k) for x in per_seed])
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"] = s
        with open(os.path.join(output_dir, f"method_{method_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return results_per_method


def main():
    parser = argparse.ArgumentParser(description="Run model on IHDP benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (required for ours)")
    parser.add_argument("--data", type=str, default=None, help="Path to IHDP CSV (or URL). Default: try default URL.")
    parser.add_argument("--output", type=str, default=None, help="Write single results JSON here")
    parser.add_argument("--output-dir", type=str, default=None, help="Write protocol-style method_*_seed*.json and summary here")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Multiple seeds for protocol (used with --output-dir)")
    parser.add_argument("--methods", type=str, default="ours,mean_stub,outcome", help="Comma-separated: ours, mean_stub, outcome, dr, bart")
    parser.add_argument("--no-baseline", action="store_true", help="Skip outcome baseline (legacy single-run)")
    args = parser.parse_args()

    if args.output_dir:
        methods_list = [m.strip() for m in args.methods.split(",") if m.strip()]
        seeds_list = args.seeds if args.seeds is not None else [args.seed]
        run_ihdp_protocol(
            checkpoint_path=args.checkpoint or "",
            data_path=args.data,
            device=args.device,
            train_frac=args.train_frac,
            seeds=seeds_list,
            output_dir=args.output_dir,
            methods=methods_list,
        )
        print(f"IHDP protocol results written to {args.output_dir}")
        return

    if not args.checkpoint:
        parser.error("--checkpoint required when not using --output-dir")
    results = run_ihdp_eval(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        device=args.device,
        train_frac=args.train_frac,
        seed=args.seed,
        output_path=args.output,
        run_baseline=not args.no_baseline,
    )

    print("IHDP results:")
    print(f"  delta_correlation: {results.get('delta_correlation')}")
    print(f"  delta_rmse (sqrt(PEHE)): {results.get('delta_rmse')}")
    print(f"  pehe: {results.get('pehe')}")
    if results.get("pehe_ite") is not None:
        print(f"  pehe_ite: {results.get('pehe_ite')}")
    print(f"  has_potential_outcomes: {results.get('has_potential_outcomes')}")
    if args.output:
        print(f"  Written to {args.output}")


if __name__ == "__main__":
    main()
