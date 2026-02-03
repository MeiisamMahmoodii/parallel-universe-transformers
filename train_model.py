"""Main training script."""

import argparse
import torch
from train.config import TrainingConfig
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Parallel Universe Transformer")
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum training steps')
    
    # Loss
    parser.add_argument('--lambda-delta', type=float, default=1.0, help='Delta loss weight')
    
    # Optimization
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Use gradient checkpointing')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=100, help='Log every N steps')
    parser.add_argument('--eval-every', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--save-every', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='parallel-universe-transformers', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lambda_delta=args.lambda_delta,
        use_mixed_precision=args.mixed_precision,
        use_gradient_checkpointing=args.gradient_checkpointing,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        num_workers=args.num_workers
    )
    
    print("Configuration:")
    print(f"  Model: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"  Training: batch_size={config.effective_batch_size}, lr={config.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  Max steps: {config.max_steps}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
