"""Observational data sampling from SCMs."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .schema import FeatureSchema, FeatureInfo, FeatureType
from .mechanisms import MechanismSampler
from .noise import NoiseSampler, NoiseType


@dataclass
class SCMConfig:
    """Configuration for SCM sampling."""
    n_features: int = 20
    complexity: str = "simple"  # 'simple', 'moderate', 'complex'
    seed: Optional[int] = None
    outcome_noise_scale: float = 0.5


class SCMSampler:
    """Samples observational data from a randomly generated SCM."""
    
    def __init__(self, schema: List[FeatureInfo], config: SCMConfig):
        self.schema = schema
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Initialize samplers
        self.mechanism_sampler = MechanismSampler(seed=self.rng.randint(0, 2**31))
        self.noise_sampler = NoiseSampler(seed=self.rng.randint(0, 2**31))
        
        # Sample causal order (topological sort)
        self.causal_order = self._sample_causal_order()
        
        # Sample mechanisms for each variable
        self.mechanisms = self._sample_mechanisms()
        
        # Sample noise types and scales
        self.noise_types = self._sample_noise_types()
        self.noise_scales = self._sample_noise_scales()
        
        # Sample outcome mechanism
        self.outcome_mechanism = self.mechanism_sampler.sample_outcome_mechanism(
            input_dim=len(schema),
            complexity=config.complexity
        )
        self.outcome_noise_type = self.noise_sampler.sample_noise_type(config.complexity)
        self.outcome_noise_scale = config.outcome_noise_scale
    
    def _sample_causal_order(self) -> List[int]:
        """Sample a random causal ordering of features.
        
        Returns:
            List of feature indices in causal order.
        """
        order = list(range(len(self.schema)))
        self.rng.shuffle(order)
        return order
    
    def _sample_mechanisms(self) -> Dict[int, Optional[object]]:
        """Sample mechanisms for each feature.
        
        Returns:
            Dictionary mapping feature index to mechanism (or None for root nodes).
        """
        mechanisms = {}
        
        for i, feature_idx in enumerate(self.causal_order):
            if i == 0:
                # Root node: no mechanism (sample from marginal)
                mechanisms[feature_idx] = None
            else:
                # Sample parents (subset of previous nodes in causal order)
                n_parents = self.rng.randint(1, min(i + 1, 4))  # 1-3 parents
                parent_indices = self.rng.choice(self.causal_order[:i], size=n_parents, replace=False)
                
                # Sample mechanism
                mechanism = self.mechanism_sampler.sample_mechanism(
                    input_dim=len(parent_indices),
                    complexity=self.config.complexity
                )
                
                mechanisms[feature_idx] = {
                    'parents': parent_indices,
                    'function': mechanism
                }
        
        return mechanisms
    
    def _sample_noise_types(self) -> Dict[int, NoiseType]:
        """Sample noise types for each feature."""
        return {
            i: self.noise_sampler.sample_noise_type(self.config.complexity)
            for i in range(len(self.schema))
        }
    
    def _sample_noise_scales(self) -> Dict[int, float]:
        """Sample noise scales for each feature."""
        return {
            i: self.noise_sampler.sample_noise_scale(self.config.complexity)
            for i in range(len(self.schema))
        }
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample observational data from the SCM.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Tuple of (X, Y) where:
                X: Features array of shape [n_samples, n_features]
                Y: Outcomes array of shape [n_samples]
        """
        X = np.zeros((n_samples, len(self.schema)))
        
        # Sample features in causal order
        for feature_idx in self.causal_order:
            feature = self.schema[feature_idx]
            mechanism = self.mechanisms[feature_idx]
            
            if mechanism is None:
                # Root node: sample from marginal
                if feature.feature_type == FeatureType.CONTINUOUS:
                    X[:, feature_idx] = self.rng.uniform(
                        feature.min_value,
                        feature.max_value,
                        size=n_samples
                    )
                else:
                    X[:, feature_idx] = self.rng.randint(
                        0,
                        feature.cardinality,
                        size=n_samples
                    )
            else:
                # Non-root: apply mechanism
                parent_indices = mechanism['parents']
                parent_values = X[:, parent_indices]
                
                # Apply mechanism
                values = mechanism['function'](parent_values)
                
                # Add noise
                noise = self.noise_sampler.sample_noise(
                    n_samples,
                    noise_type=self.noise_types[feature_idx],
                    scale=self.noise_scales[feature_idx],
                    X=parent_values
                )
                values += noise
                
                # Post-process based on feature type
                if feature.feature_type == FeatureType.CONTINUOUS:
                    # Clip to range
                    values = np.clip(values, feature.min_value, feature.max_value)
                else:
                    # Convert to categorical
                    # Map continuous values to categories
                    values = np.floor((values - values.min()) / (values.max() - values.min() + 1e-8) * feature.cardinality)
                    values = np.clip(values, 0, feature.cardinality - 1).astype(int)
                
                X[:, feature_idx] = values
        
        # Sample outcome Y
        Y = self.outcome_mechanism(X)
        noise = self.noise_sampler.sample_noise(
            n_samples,
            noise_type=self.outcome_noise_type,
            scale=self.outcome_noise_scale,
            X=X
        )
        Y += noise
        
        return X, Y
    
    def get_parents(self, feature_idx: int) -> Optional[List[int]]:
        """Get parent indices for a feature.
        
        Args:
            feature_idx: Index of the feature.
            
        Returns:
            List of parent indices, or None if root node.
        """
        mechanism = self.mechanisms[feature_idx]
        if mechanism is None:
            return None
        return list(mechanism['parents'])
