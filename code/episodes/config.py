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
    shuffle_intervention_order: bool = True  # If True, randomize order of intervention worlds (1..K) per episode


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
    def get_long_run_curriculum():
        """Curriculum with more steps on harder stages (stage_2, stage_3) to improve delta_correlation.
        Use when you want to spend more training on moderate/final stages without changing code.
        """
        stages = CurriculumConfig.get_default_curriculum()
        # Increase min_steps for stage_2 and stage_3 (e.g. 1.5x–2x)
        out = []
        for s in stages:
            if s.name == "stage_2_moderate":
                out.append(CurriculumStage(
                    name=s.name,
                    n_features=s.n_features,
                    n_continuous=s.n_continuous,
                    n_categorical=s.n_categorical,
                    n_interventions=s.n_interventions,
                    complexity=s.complexity,
                    support_size_range=s.support_size_range,
                    query_size_range=s.query_size_range,
                    missingness_prob=s.missingness_prob,
                    noise_scale=s.noise_scale,
                    min_steps=40000,
                    shuffle_intervention_order=s.shuffle_intervention_order,
                ))
            elif s.name == "stage_3_final":
                out.append(CurriculumStage(
                    name=s.name,
                    n_features=s.n_features,
                    n_continuous=s.n_continuous,
                    n_categorical=s.n_categorical,
                    n_interventions=s.n_interventions,
                    complexity=s.complexity,
                    support_size_range=s.support_size_range,
                    query_size_range=s.query_size_range,
                    missingness_prob=s.missingness_prob,
                    noise_scale=s.noise_scale,
                    min_steps=60000,
                    shuffle_intervention_order=s.shuffle_intervention_order,
                ))
            else:
                out.append(s)
        return out

    @staticmethod
    def batch_size_for_stage(stage: "CurriculumStage", max_batch_size: int) -> int:
        """Heuristic batch size for a curriculum stage.

        Historically we used a very conservative attention-memory bound (∝ W * N^2) which
        often forced batch size 1 even when SDPA/FlashAttention was enabled.

        With the PoC curriculum capped to d<=20 and interventions<=8, we can safely use a
        milder heuristic that scales roughly with W * N (still reducing batch size when
        worlds/sequence lengths grow), while keeping larger batches for speed.

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
        # Heuristic: keep B * W * N roughly bounded relative to stage_0.
        # This is intentionally *less* conservative than N^2 (which can be overly pessimistic with SDPA).
        ratio = (ref_w * ref_n) / (cur_w * cur_n)
        safety_margin = 0.9
        b = max(1, int(max_batch_size * ratio * safety_margin))
        return min(b, max_batch_size)
