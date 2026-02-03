"""Main training loop."""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Optional, Dict, Any
import json

from model.model import ParallelUniverseTransformer
from episodes.config import CurriculumConfig
from episodes.dataset import create_dataloader
from .config import TrainingConfig
from .losses import LossComputer
from .metrics import MetricsComputer, format_metrics


class Trainer:
    """Trainer for Parallel Universe Transformer."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[ParallelUniverseTransformer] = None
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
            model: Model to train (if None, create new model).
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.rank = getattr(config, "rank", 0)
        self.world_size = getattr(config, "world_size", 1)
        self._is_ddp = self.world_size > 1
        
        # Create model (or use provided DDP-wrapped model)
        if model is None:
            model = ParallelUniverseTransformer(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                cross_world_layers=config.cross_world_layers,
                attend_to_all_worlds=config.attend_to_all_worlds,
                use_gradient_checkpointing=config.use_gradient_checkpointing,
                use_quantiles=config.use_quantiles
            )
        
        self.model = model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Loss and metrics
        self.loss_computer = LossComputer(
            lambda_delta=config.lambda_delta,
            use_quantiles=config.use_quantiles
        )
        self.metrics_computer = MetricsComputer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # EMA loss for smoother progress bar display
        self._ema_loss = None
        # Last eval metrics (shown on bar until next eval)
        self._last_eval_metrics: Optional[dict] = None
        
        # Checkpointing (only rank 0 creates dir and saves)
        if self.rank == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
        
        # Wandb (only rank 0)
        self.use_wandb = config.use_wandb and (self.rank == 0)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project or "parallel-universe-transformers",
                    name=config.wandb_run_name,
                    config=vars(config)
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, logging disabled")
                self.use_wandb = False
    
    def train_step(self, batch: dict) -> dict:
        """Single training step.
        
        Args:
            batch: Batch of episodes.
            
        Returns:
            Dictionary of losses.
        """
        # Move batch to device
        support_x = batch['support_x'].to(self.device)
        support_y = batch['support_y'].to(self.device)
        query_x = batch['query_x'].to(self.device)
        query_y = batch['query_y'].to(self.device)
        feature_types = batch['feature_types'][0].to(self.device)  # Same for all in batch
        cardinalities = batch['cardinalities'][0].to(self.device)
        support_mask = batch['support_mask'].to(self.device)
        query_mask = batch['query_mask'].to(self.device)
        
        # Build loss mask for variable-length episodes (1 = valid, 0 = padded)
        query_lengths = batch['query_lengths'].to(self.device)  # [B]
        B, W, max_Nq = query_y.shape
        # loss_mask[b, q] = 1 if q < query_lengths[b] else 0
        loss_mask = (torch.arange(max_Nq, device=self.device) < query_lengths.unsqueeze(1)).float()  # [B, max_Nq]
        loss_mask = loss_mask.unsqueeze(1).expand(B, W, max_Nq)  # [B, W, max_Nq]
        
        # Forward pass
        if self.scaler is not None:
            with autocast():
                outputs = self.model(
                    support_x, support_y, query_x,
                    feature_types, cardinalities,
                    support_mask, query_mask
                )
                losses = self.loss_computer.compute_loss(outputs, query_y, loss_mask=loss_mask)
                loss = losses['total'] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                support_x, support_y, query_x,
                feature_types, cardinalities,
                support_mask, query_mask
            )
            losses = self.loss_computer.compute_loss(outputs, query_y, loss_mask=loss_mask)
            loss = losses['total'] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update metrics (pass loss_mask so only valid positions count)
        self.metrics_computer.update(
            outputs['prediction'],
            query_y,
            outputs['log_var'],
            loss_mask=loss_mask,
        )
        
        return {k: v.item() for k, v in losses.items()}
    
    def _progress_postfix(self) -> Dict[str, Any]:
        """Build postfix dict for progress bar: train losses + last eval metrics."""
        postfix = dict(self._ema_loss) if self._ema_loss else {}
        if self._last_eval_metrics:
            for k, v in self._last_eval_metrics.items():
                if v is not None and isinstance(v, (int, float)) and v == v:  # finite
                    postfix[k] = round(float(v), 4)
        return postfix

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch with per-epoch progress bar (0-100%), live metrics, and eval display."""
        self.model.train()
        total_losses = {}
        num_batches = 0
        num_batches_since_log = 0

        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None

        progress_bar = tqdm(
            dataloader,
            total=total_batches,
            desc=f"Epoch {self.current_epoch}",
            disable=(self.rank != 0),
            position=0,
            leave=False,
            unit="batch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            losses = self.train_step(batch)

            # Update EMA for smoother display
            if self._ema_loss is None:
                self._ema_loss = dict(losses)
            else:
                for k in losses:
                    self._ema_loss[k] = 0.95 * self._ema_loss[k] + 0.05 * losses[k]

            progress_bar.set_postfix(self._progress_postfix())

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches_since_log += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping (params: unwrap DDP for single model)
                model_for_grad = self.model.module if self._is_ddp else self.model
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_for_grad.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model_for_grad.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                # Learning rate scheduling
                if self.global_step < self.config.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every == 0:
                    n = max(num_batches_since_log, 1)
                    avg_losses = {k: v / n for k, v in total_losses.items()}
                    if self.use_wandb:
                        self.wandb.log({
                            f"train/{k}": v for k, v in avg_losses.items()
                        }, step=self.global_step)
                        self.wandb.log({
                            "train/lr": self.optimizer.param_groups[0]['lr']
                        }, step=self.global_step)
                    total_losses = {}
                    num_batches_since_log = 0

                # Evaluation: compute metrics, print results, store for bar
                if self.global_step % self.config.eval_every == 0:
                    metrics = self.metrics_computer.compute_and_reset()
                    self._last_eval_metrics = {
                        "eval_delta_mae": metrics.get("delta_mae"),
                        "eval_baseline_mae": metrics.get("baseline_mae"),
                        "eval_delta_corr": metrics.get("delta_correlation"),
                    }
                    if self.rank == 0:
                        print(f"\n--- Evaluation @ step {self.global_step} ---")
                        print(format_metrics(metrics))
                        print("---\n")
                    progress_bar.set_postfix(self._progress_postfix())
                    if self.use_wandb:
                        self.wandb.log({
                            f"train/{k}": v for k, v in metrics.items()
                        }, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                # Check if max steps reached
                if self.global_step >= self.config.max_steps:
                    break

            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def train(self):
        """Full training loop with curriculum."""
        curriculum_stages = CurriculumConfig.get_default_curriculum()

        for stage in curriculum_stages:
            if self.global_step >= self.config.max_steps:
                break

            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Starting curriculum stage: {stage.name}")
                print(f"Features: {stage.n_features}, Interventions: {stage.n_interventions}")
                print(f"Complexity: {stage.complexity}")
                print(f"{'='*60}\n")

            # Create dataloader for this stage (sharded by rank when DDP)
            dataloader = create_dataloader(
                stage,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                seed=self.current_epoch,
                rank=self.rank,
                world_size=self.world_size
            )

            # Train for this stage
            stage_start_step = self.global_step
            while self.global_step < stage_start_step + stage.min_steps:
                if self.global_step >= self.config.max_steps:
                    break
                self.train_epoch(dataloader)
                self.current_epoch += 1

        # Final checkpoint
        self.save_checkpoint("final_model.pt")
        if self.rank == 0:
            print(f"\nTraining completed! Final step: {self.global_step}")
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint (only on rank 0 when using DDP).
        
        Args:
            filename: Checkpoint filename.
        """
        if self.rank != 0:
            if self._is_ddp:
                torch.distributed.barrier()
            return
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        model_state = self.model.module.state_dict() if self._is_ddp else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': vars(self.config)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        if self._is_ddp:
            torch.distributed.barrier()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint (DDP: load into model.module).
        
        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint['model_state_dict']
        target = self.model.module if self._is_ddp else self.model
        target.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.rank == 0:
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")
