"""Main training script. Supports single-GPU and multi-GPU via DistributedDataParallel.

Single-GPU:
  python train_model.py [args...]

Multi-GPU (e.g. 2 GPUs):
  torchrun --nproc_per_node=2 train_model.py [args...]
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train.config import TrainingConfig
from train.trainer import Trainer
from model.model import ParallelUniverseTransformer


def setup_distributed():
    """Initialize process group if RANK is set (e.g. by torchrun). Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


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
    parser.add_argument('--lambda-delta', type=float, default=2.0, help='Delta loss weight (recommended 2-10 if delta plateaus; higher = more focus on effect estimation)')
    parser.add_argument('--lambda-delta-warmup-steps', type=int, default=0, help='Ramp lambda_delta from 0 to --lambda-delta over this many steps (0 = no warmup)')
    
    # Optimization
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--no-gradient-checkpointing', action='store_true', help='Disable gradient checkpointing (uses more GPU memory)')
    
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
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu); overridden by DDP to cuda:LOCAL_RANK')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Help PyTorch reuse large allocations when sequence length spikes (prevents fragmentation OOMs).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Distributed setup (no-op if not launched with torchrun)
    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device_count = torch.cuda.device_count()
        # DDP requires one GPU per process; NCCL does not allow multiple ranks on the same GPU
        if device_count < world_size:
            if rank == 0:
                print(
                    f"Error: DDP requires one GPU per process. You have world_size={world_size} "
                    f"but only {device_count} GPU(s) visible in this process.\n"
                    f"Use: torchrun --nproc_per_node={device_count} train_model.py ...",
                    file=sys.stderr,
                )
            sys.exit(1)
        device = f"cuda:{local_rank}"
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
    
    # Create config (with DDP rank/world_size so dataloader and trainer shard correctly)
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
        lambda_delta_warmup_steps=args.lambda_delta_warmup_steps,
        use_mixed_precision=args.mixed_precision,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=device,
        num_workers=args.num_workers,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    if rank == 0:
        print("Configuration:")
        print(f"  Model: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
        print(f"  Training: effective_batch_size={config.effective_batch_size}, lr={config.learning_rate}")
        print(f"  Device: {config.device} (world_size={world_size})")
        print(f"  Max steps: {config.max_steps}")
    
    # Create model and optionally wrap in DDP
    model = ParallelUniverseTransformer(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        cross_world_layers=config.cross_world_layers,
        attend_to_all_worlds=config.attend_to_all_worlds,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_quantiles=config.use_quantiles,
    )
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create trainer (with DDP-wrapped model when multi-GPU)
    trainer = Trainer(config, model=model)
    
    # Train
    if rank == 0:
        print("\nStarting training...")
    trainer.train()
    
    if rank == 0:
        print("\nTraining completed!")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
