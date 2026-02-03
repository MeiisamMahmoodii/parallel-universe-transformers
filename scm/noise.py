"""Noise distribution sampler for SCMs."""

from enum import Enum
from typing import Optional
import numpy as np


class NoiseType(Enum):
    """Types of noise distributions."""
    GAUSSIAN = "gaussian"
    HETEROSKEDASTIC = "heteroskedastic"
    HEAVY_TAILED = "heavy_tailed"
    UNIFORM = "uniform"


class NoiseSampler:
    """Samples noise for structural equations."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def sample_noise(
        self,
        n_samples: int,
        noise_type: Optional[NoiseType] = None,
        scale: float = 1.0,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sample noise for a variable.
        
        Args:
            n_samples: Number of samples.
            noise_type: Type of noise (if None, use Gaussian).
            scale: Scale parameter for noise.
            X: Optional conditioning variables for heteroskedastic noise.
            
        Returns:
            Noise array of shape [n_samples].
        """
        if noise_type is None:
            noise_type = NoiseType.GAUSSIAN
        
        if noise_type == NoiseType.GAUSSIAN:
            return self.rng.randn(n_samples) * scale
        
        elif noise_type == NoiseType.HETEROSKEDASTIC:
            # Variance depends on X
            if X is None:
                # Fallback to Gaussian
                return self.rng.randn(n_samples) * scale
            
            # Compute variance as function of X
            # Use simple heuristic: variance proportional to |mean(X)|
            X_mean = np.mean(np.abs(X), axis=1) if X.ndim > 1 else np.abs(X)
            local_scale = scale * (0.5 + X_mean / (1 + X_mean))
            return self.rng.randn(n_samples) * local_scale
        
        elif noise_type == NoiseType.HEAVY_TAILED:
            # Student-t distribution with df=3
            df = 3
            return self.rng.standard_t(df, size=n_samples) * scale / np.sqrt(df / (df - 2))
        
        elif noise_type == NoiseType.UNIFORM:
            # Uniform in [-sqrt(3)*scale, sqrt(3)*scale] to match variance
            return self.rng.uniform(-np.sqrt(3) * scale, np.sqrt(3) * scale, size=n_samples)
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def sample_noise_type(self, complexity: str = "simple") -> NoiseType:
        """Sample a noise type based on complexity.
        
        Args:
            complexity: Complexity level ('simple', 'moderate', 'complex').
            
        Returns:
            Sampled NoiseType.
        """
        if complexity == "simple":
            return NoiseType.GAUSSIAN
        elif complexity == "moderate":
            return self.rng.choice([NoiseType.GAUSSIAN, NoiseType.HETEROSKEDASTIC])
        else:  # complex
            return self.rng.choice(list(NoiseType))
    
    def sample_noise_scale(self, complexity: str = "simple") -> float:
        """Sample a noise scale parameter.
        
        Args:
            complexity: Complexity level.
            
        Returns:
            Noise scale (standard deviation).
        """
        if complexity == "simple":
            return self.rng.uniform(0.1, 0.5)
        elif complexity == "moderate":
            return self.rng.uniform(0.1, 1.0)
        else:  # complex
            return self.rng.uniform(0.05, 2.0)
