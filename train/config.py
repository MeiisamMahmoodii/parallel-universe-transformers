"""Training configuration."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model architecture
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    cross_world_layers: List[int] = None
    attend_to_all_worlds: bool = True
    use_gradient_checkpointing: bool = False
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000
    
    # Loss
    lambda_delta: float = 2.0
    lambda_delta_warmup_steps: int = 0  # 0 = no warmup; else ramp 0 -> lambda_delta over this many steps
    use_quantiles: bool = False
    
    # Optimization
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    # Curriculum
    use_curriculum: bool = True
    
    # Device
    device: str = "cuda"
    
    # DistributedDataParallel
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    
    def __post_init__(self):
        if self.cross_world_layers is None:
            self.cross_world_layers = [3, 5]
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size with gradient accumulation (and world_size when DDP)."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    @property
    def is_distributed(self) -> bool:
        """True when using DistributedDataParallel (world_size > 1)."""
        return self.world_size > 1
