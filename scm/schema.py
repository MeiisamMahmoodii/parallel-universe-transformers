"""Feature schema sampler for generating diverse tabular structures."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
import numpy as np


class FeatureType(Enum):
    """Types of features in the dataset."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class FeatureInfo:
    """Information about a single feature."""
    name: str
    feature_type: FeatureType
    cardinality: Optional[int] = None  # For categorical features
    min_value: Optional[float] = None  # For continuous features
    max_value: Optional[float] = None  # For continuous features


@dataclass
class SchemaConfig:
    """Configuration for schema generation."""
    n_features: int = 20
    n_continuous: int = 10
    n_categorical: int = 10
    min_cardinality: int = 2
    max_cardinality: int = 10
    continuous_range: tuple = (-5.0, 5.0)
    seed: Optional[int] = None


class FeatureSchema:
    """Samples feature schemas for SCM generation."""
    
    def __init__(self, config: SchemaConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def sample_schema(self) -> List[FeatureInfo]:
        """Sample a random feature schema.
        
        Returns:
            List of FeatureInfo objects describing the schema.
        """
        features = []
        
        # Sample continuous features
        for i in range(self.config.n_continuous):
            features.append(FeatureInfo(
                name=f"x_cont_{i}",
                feature_type=FeatureType.CONTINUOUS,
                min_value=self.config.continuous_range[0],
                max_value=self.config.continuous_range[1],
            ))
        
        # Sample categorical features
        for i in range(self.config.n_categorical):
            cardinality = self.rng.randint(
                self.config.min_cardinality,
                self.config.max_cardinality + 1
            )
            features.append(FeatureInfo(
                name=f"x_cat_{i}",
                feature_type=FeatureType.CATEGORICAL,
                cardinality=cardinality,
            ))
        
        # Shuffle to mix continuous and categorical
        self.rng.shuffle(features)
        
        # Rename with sequential indices
        for i, feature in enumerate(features):
            feature.name = f"x_{i}"
        
        return features
    
    def get_feature_types(self, schema: List[FeatureInfo]) -> np.ndarray:
        """Get array of feature types.
        
        Args:
            schema: List of FeatureInfo objects.
            
        Returns:
            Array of shape [n_features] with 0=continuous, 1=categorical.
        """
        return np.array([
            0 if f.feature_type == FeatureType.CONTINUOUS else 1
            for f in schema
        ])
    
    def get_cardinalities(self, schema: List[FeatureInfo]) -> np.ndarray:
        """Get array of cardinalities (1 for continuous features).
        
        Args:
            schema: List of FeatureInfo objects.
            
        Returns:
            Array of shape [n_features] with cardinalities.
        """
        return np.array([
            1 if f.feature_type == FeatureType.CONTINUOUS else f.cardinality
            for f in schema
        ])
    
    def sample_feature_value(self, feature: FeatureInfo) -> float:
        """Sample a single value for a feature.
        
        Args:
            feature: FeatureInfo object.
            
        Returns:
            Sampled value (float for continuous, int for categorical).
        """
        if feature.feature_type == FeatureType.CONTINUOUS:
            return self.rng.uniform(feature.min_value, feature.max_value)
        else:
            return float(self.rng.randint(0, feature.cardinality))
