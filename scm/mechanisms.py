"""Mechanism sampler for structural equations in SCMs."""

from enum import Enum
from typing import Callable, List, Optional
import numpy as np
import torch
import torch.nn as nn


class MechanismType(Enum):
    """Types of mechanisms for structural equations."""
    LINEAR = "linear"
    SHALLOW_MLP = "shallow_mlp"
    RBF = "rbf"
    SPLINE = "spline"


class LinearMechanism:
    """Linear mechanism: f(X) = w^T X."""
    
    def __init__(self, input_dim: int, rng: np.random.RandomState):
        self.weights = rng.randn(input_dim) * 0.5
        self.bias = rng.randn() * 0.1
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply mechanism.
        
        Args:
            X: Input array of shape [n_samples, input_dim].
            
        Returns:
            Output array of shape [n_samples].
        """
        return X @ self.weights + self.bias


class ShallowMLPMechanism:
    """Shallow MLP mechanism with 1-2 hidden layers."""
    
    def __init__(self, input_dim: int, rng: np.random.RandomState, n_hidden: int = 32):
        self.n_hidden = n_hidden
        
        # Layer 1
        self.w1 = rng.randn(input_dim, n_hidden) * np.sqrt(2.0 / input_dim)
        self.b1 = rng.randn(n_hidden) * 0.1
        
        # Layer 2
        self.w2 = rng.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = rng.randn(n_hidden) * 0.1
        
        # Output layer
        self.w_out = rng.randn(n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b_out = rng.randn() * 0.1
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply mechanism.
        
        Args:
            X: Input array of shape [n_samples, input_dim].
            
        Returns:
            Output array of shape [n_samples].
        """
        # Layer 1
        h1 = np.maximum(0, X @ self.w1 + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)  # ReLU
        
        # Output
        return h2 @ self.w_out + self.b_out


class RBFMechanism:
    """RBF (Radial Basis Function) mechanism."""
    
    def __init__(self, input_dim: int, rng: np.random.RandomState, n_centers: int = 10):
        self.centers = rng.randn(n_centers, input_dim)
        self.weights = rng.randn(n_centers) * 0.5
        self.lengthscales = rng.uniform(0.5, 2.0, size=n_centers)
        self.bias = rng.randn() * 0.1
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply mechanism.
        
        Args:
            X: Input array of shape [n_samples, input_dim].
            
        Returns:
            Output array of shape [n_samples].
        """
        # Compute distances to centers
        # X: [n_samples, input_dim], centers: [n_centers, input_dim]
        dists = np.sum((X[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        
        # Apply RBF kernel
        rbf_features = np.exp(-dists / (2 * self.lengthscales[None, :] ** 2))
        
        # Linear combination
        return rbf_features @ self.weights + self.bias


class SplineMechanism:
    """Piecewise linear spline mechanism."""
    
    def __init__(self, input_dim: int, rng: np.random.RandomState, n_knots: int = 5):
        # For simplicity, use a univariate spline on the first principal component
        self.weights = rng.randn(input_dim)
        self.weights /= np.linalg.norm(self.weights)
        
        # Knot locations and values
        self.knots = np.sort(rng.uniform(-3, 3, size=n_knots))
        self.knot_values = rng.randn(n_knots)
        self.bias = rng.randn() * 0.1
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply mechanism.
        
        Args:
            X: Input array of shape [n_samples, input_dim].
            
        Returns:
            Output array of shape [n_samples].
        """
        # Project to 1D
        proj = X @ self.weights
        
        # Piecewise linear interpolation
        result = np.zeros(len(proj))
        for i in range(len(proj)):
            p = proj[i]
            if p <= self.knots[0]:
                result[i] = self.knot_values[0]
            elif p >= self.knots[-1]:
                result[i] = self.knot_values[-1]
            else:
                # Find interval
                idx = np.searchsorted(self.knots, p)
                # Linear interpolation
                t = (p - self.knots[idx - 1]) / (self.knots[idx] - self.knots[idx - 1])
                result[i] = (1 - t) * self.knot_values[idx - 1] + t * self.knot_values[idx]
        
        return result + self.bias


class MechanismSampler:
    """Samples mechanisms for structural equations."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def sample_mechanism(
        self,
        input_dim: int,
        mechanism_type: Optional[MechanismType] = None,
        complexity: str = "simple"
    ) -> Callable:
        """Sample a mechanism function.
        
        Args:
            input_dim: Number of input features.
            mechanism_type: Type of mechanism (if None, sample randomly).
            complexity: Complexity level ('simple', 'moderate', 'complex').
            
        Returns:
            Callable mechanism function.
        """
        if mechanism_type is None:
            # Sample mechanism type based on complexity
            if complexity == "simple":
                types = [MechanismType.LINEAR, MechanismType.SHALLOW_MLP]
            elif complexity == "moderate":
                types = [MechanismType.LINEAR, MechanismType.SHALLOW_MLP, MechanismType.RBF]
            else:  # complex
                types = list(MechanismType)
            
            mechanism_type = self.rng.choice(types)
        
        if mechanism_type == MechanismType.LINEAR:
            return LinearMechanism(input_dim, self.rng)
        elif mechanism_type == MechanismType.SHALLOW_MLP:
            n_hidden = 16 if complexity == "simple" else 32
            return ShallowMLPMechanism(input_dim, self.rng, n_hidden=n_hidden)
        elif mechanism_type == MechanismType.RBF:
            n_centers = 5 if complexity == "simple" else 10
            return RBFMechanism(input_dim, self.rng, n_centers=n_centers)
        elif mechanism_type == MechanismType.SPLINE:
            n_knots = 3 if complexity == "simple" else 5
            return SplineMechanism(input_dim, self.rng, n_knots=n_knots)
        else:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}")
    
    def sample_outcome_mechanism(
        self,
        input_dim: int,
        complexity: str = "simple"
    ) -> Callable:
        """Sample a mechanism for the outcome variable Y.
        
        Args:
            input_dim: Number of input features.
            complexity: Complexity level.
            
        Returns:
            Callable mechanism function.
        """
        return self.sample_mechanism(input_dim, mechanism_type=None, complexity=complexity)
