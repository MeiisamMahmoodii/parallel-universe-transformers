#!/usr/bin/env python3
"""Fine-tune the model on IHDP (with early stopping).

Usage:
  PYTHONPATH=code uv run python scripts/finetune_ihdp.py --resume-from checkpoints/checkpoint_step_40000.pt --output checkpoints/finetuned_ihdp.pt
"""

import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
code_dir = repo_root / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
os.chdir(repo_root)


def compute_val_pehe(model, support_x, support_y, query_x, true_cate, device, scaler=None):
    """Run model on val data and return PEHE.

    PEHE = sqrt(mean((pred_cate - true_cate)^2)) where:
      - pred_cate = preds[1] - preds[0] (predicted E[Y|T=1,X] - E[Y|T=0,X])
      - true_cate = mu1 - mu0 on validation set (from IHDP potential outcomes)
    Uses same split as get_ihdp_val_data (train_frac=0.7, val_frac=0.1, seed).
    If scaler: model predicts scaled Y; unscale pred_cate before comparing to true_cate.
    """
    d = support_x.shape[1]
    t_support_x = support_x.unsqueeze(0).to(device)
    t_support_y = support_y.unsqueeze(0).to(device)
    t_query_x = query_x.unsqueeze(0).to(device)
    ft = torch.zeros(d, dtype=torch.long, device=device).unsqueeze(0)
    cards = torch.zeros(d, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(t_support_x, t_support_y, t_query_x, ft, cards)
    preds = out["prediction"].squeeze(0).cpu().numpy()
    if scaler is not None:
        pred_y0 = scaler.inverse_transform(preds[0].reshape(-1, 1)).flatten()
        pred_y1 = scaler.inverse_transform(preds[1].reshape(-1, 1)).flatten()
        pred_cate = pred_y1 - pred_y0
    else:
        pred_cate = preds[1] - preds[0]
    pehe = float(np.sqrt(np.mean((pred_cate - true_cate) ** 2)))
    return pehe


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on IHDP")
    parser.add_argument("--resume-from", type=str, required=True, help="Checkpoint to load (e.g. checkpoints/checkpoint_step_40000.pt)")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max training steps (use 500-1000 for multi-dataset)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (use 5e-6 for stronger regularization)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", type=str, default="checkpoints/finetuned_ihdp.pt", help="Output checkpoint path")
    parser.add_argument("--data-dir", type=str, default="data/ihdp", help="IHDP data directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-size", type=int, default=50, help="Number of (repeated) episodes per epoch")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (0 = no early stopping)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training (support) fraction")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate validation PEHE every N steps")
    parser.add_argument("--early-stopping-patience", type=int, default=100, help="Stop if no val PEHE improvement for N eval checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split (use same seed for finetune and eval)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW (use 0.1 for multi-dataset / stronger regularization)")
    parser.add_argument("--use-synthetic-ihdp", action="store_true", help="Use synthetic bootstrap episodes instead of real IHDP")
    parser.add_argument("--synthetic-mix-ratio", type=float, default=0.0, help="Fraction of episodes that are synthetic (0-1). 0.5 = 50%% real, 50%% bootstrap")
    parser.add_argument("--synthetic-mode", type=str, default="bootstrap", choices=["bootstrap", "linear"], help="Synthetic mode: bootstrap or linear")
    parser.add_argument("--synthetic-noise-std", type=float, default=0.5, help="Noise std for bootstrap outcome generation")
    # Multi-dataset: IHDP + ACIC + Twins
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Datasets to finetune on: ihdp, acic, twins. Default: ihdp only")
    parser.add_argument("--mix-ratio", type=float, nargs="+", default=None, help="Mix ratio per dataset (e.g. 0.5 0.25 0.25). Must match --datasets order")
    parser.add_argument("--acic-data", type=str, default=None, help="Path to ACIC CSV or directory (required when acic in --datasets)")
    parser.add_argument("--twins-max-samples", type=int, default=None, help="Subsample Twins to N rows for faster finetuning")
    parser.add_argument("--max-support", type=int, default=None, help="Cap support set per episode (reduces OOM for ACIC/Twins; default 500 for multi-dataset)")
    parser.add_argument("--max-query", type=int, default=None, help="Cap query set per episode (default 150 for multi-dataset)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation (use 4 with batch-size 1 for multi-dataset)")
    parser.add_argument("--lambda-delta", type=float, default=None, help="Override lambda_delta (default: 2.0)")
    parser.add_argument("--scale-outcome", action="store_true", help="Scale outcomes (StandardScaler on support Y) for IHDP-only")
    args = parser.parse_args()

    if not Path(args.resume_from).exists():
        print(f"Checkpoint not found: {args.resume_from}")
        return 1

    checkpoint = torch.load(args.resume_from, map_location=args.device)
    config_dict = checkpoint.get("config", {})
    device = torch.device(args.device)

    from model.model import ParallelUniverseTransformer
    from train.config import TrainingConfig
    from train.trainer import Trainer
    from episodes.ihdp_episode_dataset import create_ihdp_dataloader, get_ihdp_val_data
    from episodes.multi_dataset_episode_dataset import create_multi_dataset_dataloader, get_multi_dataset_val_data

    # Resolve datasets: default to ihdp-only if not specified
    datasets = args.datasets if args.datasets is not None else ["ihdp"]
    use_multi_dataset = len(datasets) > 1

    if use_multi_dataset and "acic" in datasets and not args.acic_data:
        acic_default = repo_root / "data" / "acic_sample.csv"
        if acic_default.exists():
            args.acic_data = str(acic_default)
        else:
            print("Error: --acic-data required when 'acic' in --datasets. Run scripts/download_datasets.py or set --acic-data")
            return 1
    if use_multi_dataset and "acic" in datasets and not Path(args.acic_data).exists():
        print(f"Error: ACIC path not found: {args.acic_data}")
        return 1

    mix_ratio = args.mix_ratio
    if use_multi_dataset and mix_ratio is None:
        mix_ratio = [1.0 / len(datasets)] * len(datasets)
    elif use_multi_dataset and len(mix_ratio) != len(datasets):
        print(f"Error: --mix-ratio must have {len(datasets)} values for {datasets}")
        return 1

    # Multi-dataset memory defaults: smaller batches, cap episode sizes
    if use_multi_dataset:
        if args.batch_size == 4:  # default
            args.batch_size = 1
        if args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = 4
        if args.max_support is None:
            args.max_support = 500
        if args.max_query is None:
            args.max_query = 150

    model = ParallelUniverseTransformer(
        d_model=config_dict.get("d_model", 256),
        n_layers=config_dict.get("n_layers", 6),
        n_heads=config_dict.get("n_heads", 8),
        d_ff=config_dict.get("d_ff", 1024),
        dropout=config_dict.get("dropout", 0.1),
        cross_world_layers=config_dict.get("cross_world_layers", [3, 5]),
        attend_to_all_worlds=config_dict.get("attend_to_all_worlds", True),
        use_gradient_checkpointing=config_dict.get("use_gradient_checkpointing", True),
        use_quantiles=config_dict.get("use_quantiles", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(output_path.parent)

    config = TrainingConfig(
        d_model=config_dict.get("d_model", 256),
        n_layers=config_dict.get("n_layers", 6),
        n_heads=config_dict.get("n_heads", 8),
        d_ff=config_dict.get("d_ff", 1024),
        dropout=config_dict.get("dropout", 0.1),
        cross_world_layers=config_dict.get("cross_world_layers", [3, 5]),
        attend_to_all_worlds=config_dict.get("attend_to_all_worlds", True),
        use_gradient_checkpointing=config_dict.get("use_gradient_checkpointing", True),
        use_quantiles=config_dict.get("use_quantiles", False),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=0,
        max_steps=args.max_steps,
        eval_every=args.max_steps + 1,
        save_every=args.max_steps + 1,
        lambda_delta=args.lambda_delta if args.lambda_delta is not None else 2.0,
        use_mixed_precision=not args.no_mixed_precision,
        num_workers=0,
        pin_memory=False,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        rank=0,
        local_rank=0,
        world_size=1,
    )

    trainer = Trainer(config, model=model)
    use_synthetic = args.use_synthetic_ihdp or args.synthetic_mix_ratio > 0

    if use_multi_dataset:
        dataloader = create_multi_dataset_dataloader(
            datasets=datasets,
            mix_ratio=mix_ratio,
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            seed=args.seed,
            num_workers=0,
            pin_memory=False,
            ihdp_data_dir=args.data_dir,
            acic_path=args.acic_data,
            twins_max_samples=args.twins_max_samples,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_support=args.max_support,
            max_query=args.max_query,
        )
        val_data = get_multi_dataset_val_data(
            datasets=datasets,
            seed=args.seed,
            ihdp_data_dir=args.data_dir,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        ) if args.val_frac > 0 else None
        print(f"Fine-tuning on {datasets} (mix={mix_ratio}) for up to {args.max_steps} steps (lr={args.lr})...")
    else:
        dataloader = create_ihdp_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            dataset_size=args.dataset_size,
            seed=args.seed,
            num_workers=0,
            pin_memory=False,
            use_synthetic=args.use_synthetic_ihdp,
            synthetic_mode=args.synthetic_mode,
            synthetic_mix_ratio=args.synthetic_mix_ratio,
            synthetic_noise_std=args.synthetic_noise_std,
            scale_outcome=args.scale_outcome,
        )
        val_data = None
        if args.val_frac > 0:
            val_data = get_ihdp_val_data(
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                seed=args.seed,
                data_dir=args.data_dir,
                scale_outcome=args.scale_outcome,
            )
            if val_data is None:
                print("Warning: val_frac>0 but get_ihdp_val_data returned None. Proceeding without early stopping.")
                val_data = None
        print(f"Fine-tuning on IHDP for up to {args.max_steps} steps (lr={args.lr})...")

    best_pehe = float("inf")
    steps_no_improve = 0
    if val_data:
        print(f"Early stopping: val_frac={args.val_frac}, eval_every={args.eval_every}, patience={args.early_stopping_patience}")

    while trainer.global_step < args.max_steps:
        trainer.train_epoch(dataloader)
        trainer.current_epoch += 1
        step = trainer.global_step

        if val_data and step > 0 and step % args.eval_every == 0:
            if len(val_data) == 5:
                support_x, support_y, query_x, true_cate, scaler = val_data
            else:
                support_x, support_y, query_x, true_cate = val_data
                scaler = None
            pehe = compute_val_pehe(model, support_x, support_y, query_x, true_cate, device, scaler=scaler)
            if pehe < best_pehe:
                best_pehe = pehe
                steps_no_improve = 0
                trainer.save_checkpoint(output_path.name)
                print(f"  step {step}: val PEHE={pehe:.4f} (best) -> saved")
            else:
                steps_no_improve += 1
                print(f"  step {step}: val PEHE={pehe:.4f} (no improvement, {steps_no_improve}/{args.early_stopping_patience})")
                if steps_no_improve >= args.early_stopping_patience:
                    print(f"Early stopping at step {step} (no improvement for {args.early_stopping_patience} eval checks)")
                    break

        if step >= args.max_steps:
            break

    # Without early stopping, save final model. With early stopping, best was already saved
    # (unless we never reached an eval point, in which case save current model).
    if not val_data or best_pehe == float("inf"):
        trainer.save_checkpoint(output_path.name)

    # Write metrics for sweep scripts (best_val_pehe, steps)
    metrics_path = Path(str(output_path) + ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"best_val_pehe": best_pehe if best_pehe != float("inf") else None, "steps": trainer.global_step}, f)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
