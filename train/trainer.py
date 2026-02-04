"""Main training loop."""

import os
import sys
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any
import json
from tqdm import tqdm

# Progress UI (Rich preferred; tqdm fallback)
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH = True
except Exception:  # pragma: no cover (fallback)
    _HAS_RICH = False

# Prefer new amp API (avoids FutureWarning under PyTorch 2+)
if hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
    def _autocast():
        return torch.amp.autocast("cuda")
    def _GradScaler():
        return torch.amp.GradScaler("cuda")
else:
    from torch.cuda.amp import autocast as _autocast_ctx
    from torch.cuda.amp import GradScaler as _GradScalerCls
    def _autocast():
        return _autocast_ctx()
    def _GradScaler():
        return _GradScalerCls()

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
        # Only rank 0 prints warnings so progress bar is not broken by DDP ranks
        if self.rank != 0:
            warnings.filterwarnings("ignore")
        else:
            # Suppress noisy scheduler/amp warnings so progress bar stays readable
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*", category=UserWarning)
        
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
        self.scaler = _GradScaler() if config.use_mixed_precision else None
        
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
        
        # Effective lambda_delta (warmup: ramp from 0 to config.lambda_delta over warmup steps)
        effective_lambda_delta = None
        warmup_steps = getattr(self.config, "lambda_delta_warmup_steps", 0)
        if warmup_steps > 0:
            ratio = min(1.0, self.global_step / warmup_steps)
            effective_lambda_delta = ratio * self.config.lambda_delta

        # Forward pass
        if self.scaler is not None:
            with _autocast():
                outputs = self.model(
                    support_x, support_y, query_x,
                    feature_types, cardinalities,
                    support_mask, query_mask
                )
                losses = self.loss_computer.compute_loss(
                    outputs, query_y, loss_mask=loss_mask,
                    lambda_delta_override=effective_lambda_delta,
                )
                loss = losses['total'] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                support_x, support_y, query_x,
                feature_types, cardinalities,
                support_mask, query_mask
            )
            losses = self.loss_computer.compute_loss(
                outputs, query_y, loss_mask=loss_mask,
                lambda_delta_override=effective_lambda_delta,
            )
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

    @staticmethod
    def _format_postfix(postfix: Dict[str, Any]) -> str:
        """Compact, stable-order postfix for the terminal."""
        if not postfix:
            return ""
        # Prefer a small, stable set of keys so the line doesn't jump around.
        key_order = [
            "total",
            "pred",
            "delta",
            "eval_delta_mae",
            "eval_baseline_mae",
            "eval_delta_corr",
        ]
        parts = []
        for k in key_order:
            if k in postfix:
                v = postfix[k]
                if isinstance(v, (int, float)) and v == v:
                    parts.append(f"{k}={float(v):.3f}")
        # Add any remaining keys (rare) in sorted order.
        for k in sorted(postfix.keys()):
            if k in key_order:
                continue
            v = postfix[k]
            if isinstance(v, (int, float)) and v == v:
                parts.append(f"{k}={float(v):.3f}")
        return " ".join(parts)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch.

        Uses Rich progress (preferred) to avoid overwriting previous epochs in the terminal, and prints
        a persistent per-epoch summary line for easy comparison. Falls back to tqdm if Rich is absent.
        """
        self.model.train()
        # Rolling window (for log_every)
        log_losses_sum: Dict[str, float] = {}
        log_batches = 0
        # Full-epoch averages (not reset)
        epoch_losses_sum: Dict[str, float] = {}
        epoch_batches = 0

        # Per-rank batch count: with DDP each rank sees 1/world_size of the data
        try:
            total_batches = len(dataloader)
            if self.world_size > 1:
                total_batches = max(1, total_batches // self.world_size)
        except TypeError:
            total_batches = None

        def _step_body(batch_idx: int, batch: dict, log_fn=None):
            nonlocal log_losses_sum, log_batches, epoch_losses_sum, epoch_batches
            # Training step
            losses = self.train_step(batch)

            # Update EMA for smoother display
            if self._ema_loss is None:
                self._ema_loss = dict(losses)
            else:
                for k in losses:
                    self._ema_loss[k] = 0.95 * self._ema_loss[k] + 0.05 * losses[k]

            # Accumulate losses for epoch + log window
            for k, v in losses.items():
                log_losses_sum[k] = log_losses_sum.get(k, 0.0) + float(v)
                epoch_losses_sum[k] = epoch_losses_sum.get(k, 0.0) + float(v)
            log_batches += 1
            epoch_batches += 1

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

                # Logging (windowed)
                if self.global_step % self.config.log_every == 0:
                    n = max(log_batches, 1)
                    avg_losses = {k: v / n for k, v in log_losses_sum.items()}
                    if self.use_wandb:
                        self.wandb.log({
                            f"train/{k}": v for k, v in avg_losses.items()
                        }, step=self.global_step)
                        self.wandb.log({
                            "train/lr": self.optimizer.param_groups[0]['lr']
                        }, step=self.global_step)
                    log_losses_sum = {}
                    log_batches = 0

                # Evaluation: update postfix; avoid noisy multi-line blocks
                if self.global_step % self.config.eval_every == 0:
                    metrics = self.metrics_computer.compute_and_reset()
                    self._last_eval_metrics = {
                        "eval_delta_mae": metrics.get("delta_mae"),
                        "eval_baseline_mae": metrics.get("baseline_mae"),
                        "eval_delta_corr": metrics.get("delta_correlation"),
                    }
                    if self.use_wandb:
                        self.wandb.log({
                            f"train/{k}": v for k, v in metrics.items()
                        }, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(
                        f"checkpoint_step_{self.global_step}.pt",
                        log_fn=log_fn,
                    )

        # Progress UI (rank 0 only)
        if self.rank == 0 and _HAS_RICH:
            console = Console()
            postfix = self._format_postfix(self._progress_postfix())
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]Ep {task.fields[epoch]}[/bold]"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[postfix]}"),
                console=console,
                transient=True,  # remove progress bar after epoch ends; keep printed summaries
                refresh_per_second=10,
            ) as progress:
                task_id = progress.add_task(
                    "",
                    total=total_batches,
                    epoch=str(self.current_epoch),
                    postfix=postfix,
                )

                for batch_idx, batch in enumerate(dataloader):
                    _step_body(batch_idx, batch, log_fn=console.print)
                    postfix = self._format_postfix(self._progress_postfix())
                    progress.update(task_id, advance=1, postfix=postfix, epoch=str(self.current_epoch))
                    if self.global_step >= self.config.max_steps:
                        break

            # Persistent per-epoch summary line
            if epoch_batches > 0:
                epoch_avg = {k: v / epoch_batches for k, v in epoch_losses_sum.items()}
            else:
                epoch_avg = {}
            eval_bits = []
            if self._last_eval_metrics:
                dm = self._last_eval_metrics.get("eval_delta_mae")
                bc = self._last_eval_metrics.get("eval_baseline_mae")
                dc = self._last_eval_metrics.get("eval_delta_corr")
                if isinstance(dm, (int, float)) and dm == dm:
                    eval_bits.append(f"Δ_MAE={dm:.3f}")
                if isinstance(dc, (int, float)) and dc == dc:
                    eval_bits.append(f"Δ_corr={dc:.3f}")
                if isinstance(bc, (int, float)) and bc == bc:
                    eval_bits.append(f"base_MAE={bc:.3f}")
            loss_bits = []
            for k in ("total", "pred", "delta"):
                if k in epoch_avg:
                    loss_bits.append(f"{k}={epoch_avg[k]:.3f}")
            console.print(
                f"Epoch {self.current_epoch}: "
                + (" ".join(loss_bits) if loss_bits else "(no losses)")
                + ((" | " + " ".join(eval_bits)) if eval_bits else "")
            )
            return epoch_avg

        # Non-rank0: no progress UI (keeps output clean under DDP)
        if self.rank != 0:
            for batch_idx, batch in enumerate(dataloader):
                _step_body(batch_idx, batch, log_fn=None)
                if self.global_step >= self.config.max_steps:
                    break
            return {k: v / max(epoch_batches, 1) for k, v in epoch_losses_sum.items()}

        # Rank 0 fallback: tqdm
        progress_bar = tqdm(
            dataloader,
            total=total_batches,
            desc=f"Ep {self.current_epoch}",
            disable=False,
            position=0,
            leave=False,
            unit="b",
            ncols=80,
            dynamic_ncols=False,
            file=sys.stdout,
            bar_format="{l_bar}{r_bar} {postfix}",
        )
        for batch_idx, batch in enumerate(progress_bar):
            _step_body(batch_idx, batch, log_fn=(lambda msg: progress_bar.write(msg)))
            progress_bar.set_postfix(self._progress_postfix())
            if self.global_step >= self.config.max_steps:
                break

        epoch_avg = {k: v / max(epoch_batches, 1) for k, v in epoch_losses_sum.items()}
        loss_bits = []
        for k in ("total", "pred", "delta"):
            if k in epoch_avg:
                loss_bits.append(f"{k}={epoch_avg[k]:.3f}")
        print(f"Epoch {self.current_epoch}: " + (" ".join(loss_bits) if loss_bits else "(no losses)"))
        return epoch_avg
    
    def train(self):
        """Full training loop with curriculum."""
        curriculum_stages = CurriculumConfig.get_default_curriculum()

        for stage in curriculum_stages:
            if self.global_step >= self.config.max_steps:
                break

            # Memory-safe batch size for this stage (attention ~ B*W*N^2; scale down for larger stages)
            stage_batch_size = CurriculumConfig.batch_size_for_stage(stage, self.config.batch_size)
            if self.rank == 0:
                batch_note = f", batch={stage_batch_size}" if stage_batch_size < self.config.batch_size else ""
                print(f"\n>> Stage: {stage.name} | d={stage.n_features} intv={stage.n_interventions} | {stage.complexity}{batch_note}\n")

            # Create dataloader for this stage (sharded by rank when DDP)
            dataloader = create_dataloader(
                stage,
                batch_size=stage_batch_size,
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
    
    def save_checkpoint(self, filename: str, log_fn=None):
        """Save checkpoint (only on rank 0 when using DDP).
        
        Args:
            filename: Checkpoint filename.
            log_fn: Optional callable(msg) to log the save message (e.g. progress_bar.write). If None, uses print.
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
        out = f"Checkpoint saved: {checkpoint_path}"
        (log_fn or print)(out)
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
