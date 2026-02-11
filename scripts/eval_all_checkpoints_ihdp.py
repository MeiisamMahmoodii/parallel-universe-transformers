#!/usr/bin/env python3
"""Evaluate all checkpoints on IHDP (Ours only). Write PEHE and ATE_Err per checkpoint.

Usage:
  PYTHONPATH=code uv run python scripts/eval_all_checkpoints_ihdp.py --checkpoint-dir checkpoints --output-dir results
  PYTHONPATH=code uv run python scripts/eval_all_checkpoints_ihdp.py --checkpoint checkpoints/final_model.pt checkpoints/ckpt.pt --output-dir results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root and code on path
repo_root = Path(__file__).resolve().parents[1]
code_dir = repo_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
os.chdir(repo_root)


def load_ihdp_data(data_dir: str = "data/ihdp"):
    """Load IHDP and return train/test split (same as real_world_suite). Covariates are scaled."""
    from experiments.benchmarks.real_world_suite import IHDPDataset
    from episodes.ihdp_episode_dataset import scale_covariates
    dataset = IHDPDataset(data_dir=data_dir)
    x, t, y, true_cate = dataset.load()
    if x is None:
        return None
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    x_train = x[train_idx].astype(np.float32)
    t_train = t[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx].astype(np.float32)
    true_cate_test = true_cate[test_idx]
    x_train, x_test = scale_covariates(x_train, x_test)
    support_x = np.hstack([x_train, t_train.reshape(-1, 1)]).astype(np.float32)
    support_y = y_train.astype(np.float32)
    x_test_0 = np.hstack([x_test, np.zeros((len(x_test), 1))]).astype(np.float32)
    x_test_1 = np.hstack([x_test, np.ones((len(x_test), 1))]).astype(np.float32)
    query_x = np.stack([x_test_0, x_test_1], axis=0)  # [W=2, Nq, d]
    return support_x, support_y, query_x, true_cate_test


def eval_checkpoint_ihdp(checkpoint_path: str, device: str = "cpu"):
    """Load one checkpoint, run on IHDP, return PEHE and ATE_Err for Ours."""
    from inference.api import ParallelUniverseModel
    data = load_ihdp_data()
    if data is None:
        return None, "Failed to load IHDP"
    support_x, support_y, query_x, true_cate_test = data
    try:
        model = ParallelUniverseModel.from_pretrained(checkpoint_path, device=device)
    except Exception as e:
        return None, str(e)
    B, d = 1, support_x.shape[1]
    t_support_x = torch.tensor(support_x, dtype=torch.float32, device=device).unsqueeze(0)
    t_support_y = torch.tensor(support_y, dtype=torch.float32, device=device).unsqueeze(0)
    t_query_x = torch.tensor(query_x, dtype=torch.float32, device=device).unsqueeze(0)
    ft = torch.zeros(d, dtype=torch.long, device=device)
    cards = torch.zeros(d, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model.model(
            t_support_x, t_support_y, t_query_x,
            ft.unsqueeze(0), cards.unsqueeze(0),
        )
    preds = outputs["prediction"].squeeze(0).cpu().numpy()
    pred_cate = preds[1] - preds[0]
    pehe = float(np.sqrt(np.mean((pred_cate - true_cate_test) ** 2)))
    ate_err = float(np.abs(np.mean(pred_cate) - np.mean(true_cate_test)))
    return {"PEHE": pehe, "ATE_Err": ate_err}, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on IHDP (Ours only)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to scan for *.pt")
    parser.add_argument("--checkpoint", type=str, nargs="*", default=None, help="Explicit checkpoint path(s)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output JSON/CSV")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_paths = [p for p in args.checkpoint if Path(p).exists()]
    else:
        cp_dir = Path(args.checkpoint_dir)
        if not cp_dir.exists():
            print(f"Checkpoint dir not found: {cp_dir}")
            return 1
        checkpoint_paths = sorted(cp_dir.glob("*.pt"))
        checkpoint_paths = [str(p) for p in checkpoint_paths]

    if not checkpoint_paths:
        print("No checkpoints found.")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ihdp_per_checkpoint.json", "w") as f:
            json.dump({"checkpoints": [], "best_by_pehe": None, "best_by_ate": None}, f, indent=2)
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ckpt_path in checkpoint_paths:
        name = Path(ckpt_path).name
        print(f"Evaluating {name}...")
        metrics, err = eval_checkpoint_ihdp(ckpt_path, device=args.device)
        if err:
            print(f"  Error: {err}")
            results.append({"checkpoint": name, "path": ckpt_path, "error": err})
        else:
            results.append({"checkpoint": name, "path": ckpt_path, "PEHE": metrics["PEHE"], "ATE_Err": metrics["ATE_Err"]})
            print(f"  PEHE={metrics['PEHE']:.4f}, ATE_Err={metrics['ATE_Err']:.4f}")

    valid = [r for r in results if "error" not in r]
    best_by_pehe = min(valid, key=lambda r: r["PEHE"])["checkpoint"] if valid else None
    best_by_ate = min(valid, key=lambda r: r["ATE_Err"])["checkpoint"] if valid else None

    out = {
        "checkpoints": results,
        "best_by_pehe": best_by_pehe,
        "best_by_ate": best_by_ate,
    }
    json_path = out_dir / "ihdp_per_checkpoint.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {json_path}")
    if best_by_pehe:
        print(f"Best by PEHE: {best_by_pehe}")
        print(f"Best by ATE_Err: {best_by_ate}")

    # CSV
    csv_path = out_dir / "ihdp_per_checkpoint.csv"
    with open(csv_path, "w") as f:
        f.write("checkpoint,PEHE,ATE_Err\n")
        for r in valid:
            f.write(f"{r['checkpoint']},{r['PEHE']},{r['ATE_Err']}\n")
    print(f"CSV written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
