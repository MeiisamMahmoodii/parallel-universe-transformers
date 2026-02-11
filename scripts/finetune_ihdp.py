#!/usr/bin/env python3
"""Fine-tune the model on IHDP (with early stopping).

Usage:
  PYTHONPATH=code uv run python scripts/finetune_ihdp.py --resume-from checkpoints/checkpoint_step_40000.pt --output checkpoints/finetuned_ihdp.pt
"""

import argparse
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


def compute_val_pehe(model, support_x, support_y, query_x, true_cate, device):
    """Run model on val data and return PEHE."""
    d = support_x.shape[1]
    t_support_x = support_x.unsqueeze(0).to(device)
    t_support_y = support_y.unsqueeze(0).to(device)
    t_query_x = query_x.unsqueeze(0).to(device)
    ft = torch.zeros(d, dtype=torch.long, device=device).unsqueeze(0)
    cards = torch.zeros(d, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(t_support_x, t_support_y, t_query_x, ft, cards)
    preds = out["prediction"].squeeze(0).cpu().numpy()
    pred_cate = preds[1] - preds[0]
    pehe = float(np.sqrt(np.mean((pred_cate - true_cate) ** 2)))
    return pehe


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on IHDP")
    parser.add_argument("--resume-from", type=str, required=True, help="Checkpoint to load (e.g. checkpoints/checkpoint_step_40000.pt)")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
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
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0,
        max_steps=args.max_steps,
        eval_every=args.max_steps + 1,
        save_every=args.max_steps + 1,
        lambda_delta=2.0,
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
    dataloader = create_ihdp_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        dataset_size=args.dataset_size,
        num_workers=0,
        pin_memory=False,
    )

    val_data = None
    if args.val_frac > 0:
        val_data = get_ihdp_val_data(
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            data_dir=args.data_dir,
        )
        if val_data is None:
            print("Warning: val_frac>0 but get_ihdp_val_data returned None. Proceeding without early stopping.")
            val_data = None

    best_pehe = float("inf")
    steps_no_improve = 0

    print(f"Fine-tuning on IHDP for up to {args.max_steps} steps (lr={args.lr})...")
    if val_data:
        print(f"Early stopping: val_frac={args.val_frac}, eval_every={args.eval_every}, patience={args.early_stopping_patience}")

    while trainer.global_step < args.max_steps:
        trainer.train_epoch(dataloader)
        step = trainer.global_step

        if val_data and step > 0 and step % args.eval_every == 0:
            support_x, support_y, query_x, true_cate = val_data
            pehe = compute_val_pehe(model, support_x, support_y, query_x, true_cate, device)
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

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
