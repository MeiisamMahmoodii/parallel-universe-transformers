"""Intervention operator for do-calculus."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np


class InterventionType(Enum):
    """Types of interventions."""
    SET = "set"  # Hard set: do(X_j = v)
    SHIFT = "shift"  # Shift: do(X_j = X_j + delta)
    RANDOMIZE = "randomize"  # Randomization: do(X_j ~ marginal)


@dataclass
class Intervention:
    """Specification of an intervention."""
    feature_idx: int
    intervention_type: InterventionType
    value: Optional[float] = None  # For SET and SHIFT
    
    def __post_init__(self):
        if self.intervention_type in [InterventionType.SET, InterventionType.SHIFT]:
            if self.value is None:
                raise ValueError(f"{self.intervention_type} requires a value")


class InterventionOperator:
    """Applies interventions to data."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def apply(
        self,
        X: np.ndarray,
        intervention: Intervention,
        feature_marginal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply an intervention to data.
        
        Args:
            X: Data array of shape [n_samples, n_features].
            intervention: Intervention specification.
            feature_marginal: Marginal distribution for RANDOMIZE (optional).
            
        Returns:
            Intervened data array of shape [n_samples, n_features].
        """
        X_do = X.copy()
        feature_idx = intervention.feature_idx
        
        if intervention.intervention_type == InterventionType.SET:
            # Hard set to value
            X_do[:, feature_idx] = intervention.value
        
        elif intervention.intervention_type == InterventionType.SHIFT:
            # Shift by delta
            X_do[:, feature_idx] += intervention.value
        
        elif intervention.intervention_type == InterventionType.RANDOMIZE:
            # Resample from marginal
            if feature_marginal is not None:
                # Use provided marginal
                indices = self.rng.choice(len(feature_marginal), size=len(X))
                X_do[:, feature_idx] = feature_marginal[indices]
            else:
                # Shuffle existing values
                X_do[:, feature_idx] = self.rng.permutation(X[:, feature_idx])
        
        else:
            raise ValueError(f"Unknown intervention type: {intervention.intervention_type}")
        
        return X_do
    
    def sample_intervention(
        self,
        n_features: int,
        feature_ranges: Optional[dict] = None,
        complexity: str = "simple"
    ) -> Intervention:
        """Sample a random intervention.
        
        Args:
            n_features: Number of features.
            feature_ranges: Dictionary mapping feature indices to (min, max) ranges.
            complexity: Complexity level.
            
        Returns:
            Sampled Intervention.
        """
        # Sample feature to intervene on
        feature_idx = self.rng.randint(0, n_features)
        
        # Sample intervention type
        if complexity == "simple":
            intervention_type = self.rng.choice([InterventionType.SET, InterventionType.SHIFT])
        else:
            intervention_type = self.rng.choice(list(InterventionType))
        
        # Sample value
        value = None
        if intervention_type == InterventionType.SET:
            if feature_ranges and feature_idx in feature_ranges:
                min_val, max_val = feature_ranges[feature_idx]
                value = self.rng.uniform(min_val, max_val)
            else:
                value = self.rng.uniform(-3, 3)
        
        elif intervention_type == InterventionType.SHIFT:
            if complexity == "simple":
                value = self.rng.uniform(-1, 1)
            else:
                value = self.rng.uniform(-2, 2)
        
        return Intervention(
            feature_idx=feature_idx,
            intervention_type=intervention_type,
            value=value
        )
    
    def sample_interventions(
        self,
        n_interventions: int,
        n_features: int,
        feature_ranges: Optional[dict] = None,
        complexity: str = "simple",
        allow_duplicates: bool = False
    ) -> list:
        """Sample multiple random interventions.
        
        Args:
            n_interventions: Number of interventions to sample.
            n_features: Number of features.
            feature_ranges: Dictionary mapping feature indices to (min, max) ranges.
            complexity: Complexity level.
            allow_duplicates: Whether to allow multiple interventions on same feature.
            
        Returns:
            List of Intervention objects.
        """
        interventions = []
        used_features = set()
        
        for _ in range(n_interventions):
            if not allow_duplicates and len(used_features) >= n_features:
                break
            
            # Sample intervention
            intervention = self.sample_intervention(n_features, feature_ranges, complexity)
            
            # Check for duplicates
            if not allow_duplicates:
                while intervention.feature_idx in used_features:
                    intervention = self.sample_intervention(n_features, feature_ranges, complexity)
                used_features.add(intervention.feature_idx)
            
            interventions.append(intervention)
        
        return interventions
