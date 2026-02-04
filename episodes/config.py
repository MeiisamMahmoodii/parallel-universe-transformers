"""Curriculum configuration for progressive training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    n_features: int
    n_continuous: int
    n_categorical: int
    n_interventions: int
    complexity: str  # 'simple', 'moderate', 'complex'
    support_size_range: tuple = (32, 128)
    query_size_range: tuple = (16, 32)
    missingness_prob: float = 0.0
    noise_scale: float = 0.5
    min_steps: int = 10000  # Minimum steps before advancing


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""
    
    @staticmethod
    def get_default_curriculum():
        """Get default curriculum stages. PoC: d=20 max, interventions 4-8 only for better quality and bigger batches."""
        return [
            CurriculumStage(
                name="stage_0_warmup",
                n_features=5,
                n_continuous=3,
                n_categorical=2,
                n_interventions=4,
                complexity="simple",
                support_size_range=(32, 64),
                query_size_range=(16, 24),
                missingness_prob=0.0,
                noise_scale=0.3,
                min_steps=5000,
            ),
            CurriculumStage(
                name="stage_1_basic",
                n_features=10,
                n_continuous=5,
                n_categorical=5,
                n_interventions=4,
                complexity="simple",
                support_size_range=(32, 96),
                query_size_range=(16, 28),
                missingness_prob=0.0,
                noise_scale=0.5,
                min_steps=15000,
            ),
            CurriculumStage(
                name="stage_2_moderate",
                n_features=20,
                n_continuous=10,
                n_categorical=10,
                n_interventions=6,
                complexity="moderate",
                support_size_range=(64, 128),
                query_size_range=(16, 32),
                missingness_prob=0.05,
                noise_scale=0.7,
                min_steps=25000,
            ),
            CurriculumStage(
                name="stage_3_final",
                n_features=20,
                n_continuous=10,
                n_categorical=10,
                n_interventions=8,
                complexity="moderate",
                support_size_range=(64, 128),
                query_size_range=(16, 32),
                missingness_prob=0.05,
                noise_scale=0.8,
                min_steps=40000,
            ),
        ]
    
    @staticmethod
    def get_stage_by_step(step: int):
        """Get curriculum stage based on training step.
        
        Args:
            step: Current training step.
            
        Returns:
            CurriculumStage for the current step.
        """
        stages = CurriculumConfig.get_default_curriculum()
        cumulative_steps = 0
        
        for stage in stages:
            cumulative_steps += stage.min_steps
            if step < cumulative_steps:
                return stage
        
        # Return final stage if beyond curriculum
        return stages[-1]

    @staticmethod
    def batch_size_for_stage(stage: "CurriculumStage", max_batch_size: int) -> int:
        """Memory-safe batch size for a curriculum stage to avoid OOM on larger sequences.

        Attention memory scales as B * W * N^2 (batch, worlds, sequence length in tokens).
        We scale down batch size for stages with larger W or N so peak memory stays bounded.

        Args:
            stage: Current curriculum stage.
            max_batch_size: Desired batch size (e.g. config.batch_size).

        Returns:
            Batch size to use for this stage (at most max_batch_size).
        """
        stages = CurriculumConfig.get_default_curriculum()
        ref = stages[0]
        # Reference: stage_0 max tokens and worlds
        ref_n = ref.support_size_range[1] * (ref.n_features + 1) + ref.query_size_range[1] * (ref.n_features + 1)
        ref_w = ref.n_interventions + 1
        cur_n = stage.support_size_range[1] * (stage.n_features + 1) + stage.query_size_range[1] * (stage.n_features + 1)
        cur_w = stage.n_interventions + 1
        # Keep B * W * N^2 <= ref level => B_stage <= max_batch_size * (ref_w * ref_n^2) / (cur_w * cur_n^2)
        ratio = (ref_w * (ref_n ** 2)) / (cur_w * (cur_n ** 2))
        # Apply a safety margin because real batches can hit the max simultaneously in Ns/Nq and trigger spikes.
        # With SDPA we still cap at 1 for stage_1+ to avoid ~8GB attention spikes on 40GB GPUs.
        safety_margin = 0.25  # stricter so stage_1 and beyond get batch size 1
        adjusted_ratio = ratio * safety_margin
        b = max(1, int(max_batch_size * adjusted_ratio))
        return min(b, max_batch_size)
