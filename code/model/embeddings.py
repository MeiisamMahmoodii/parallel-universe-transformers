"""Embedding modules for tabular tokenization."""

import torch
import torch.nn as nn
import numpy as np


class ContinuousEncoder(nn.Module):
    """Encodes continuous features using MLP or Fourier features."""
    
    def __init__(self, d_model: int, n_fourier: int = 0, hidden_dim: int = 128):
        """Initialize encoder.
        
        Args:
            d_model: Output dimension.
            n_fourier: Number of Fourier features (0 = use MLP only).
            hidden_dim: Hidden dimension for MLP.
        """
        super().__init__()
        self.d_model = d_model
        self.n_fourier = n_fourier
        
        if n_fourier > 0:
            # Fourier features
            self.register_buffer('fourier_weights', torch.randn(n_fourier) * 2 * np.pi)
            input_dim = 2 * n_fourier  # sin and cos
        else:
            input_dim = 1
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode continuous values.
        
        Args:
            x: Continuous values of shape [..., 1] or [...].
            
        Returns:
            Encoded features of shape [..., d_model].
        """
        if x.dim() >= 1 and (x.shape[-1] != 1 or x.dim() == 1):
            x = x.unsqueeze(-1)
        
        if self.n_fourier > 0:
            # Apply Fourier features; broadcast fourier_weights to (..., n_fourier)
            x_fourier = x * self.fourier_weights.view(*([1] * (x.dim() - 1)), -1)
            x_encoded = torch.cat([torch.sin(x_fourier), torch.cos(x_fourier)], dim=-1)
        else:
            x_encoded = x
        
        return self.mlp(x_encoded)


class CategoricalEncoder(nn.Module):
    """Encodes categorical features using embedding tables."""
    
    def __init__(self, d_model: int, max_cardinality: int = 100):
        """Initialize encoder.
        
        Args:
            d_model: Output dimension.
            max_cardinality: Maximum cardinality to support.
        """
        super().__init__()
        self.d_model = d_model
        
        # Shared embedding table for all categorical features
        # We'll use cardinality bucketing for efficiency
        self.embedding = nn.Embedding(max_cardinality, d_model)
    
    def forward(self, x: torch.Tensor, cardinality: torch.Tensor) -> torch.Tensor:
        """Encode categorical values.
        
        Args:
            x: Categorical values of shape [...] (integers).
            cardinality: Cardinality of each feature (for normalization).
            
        Returns:
            Encoded features of shape [..., d_model].
        """
        # Clamp to valid range
        x_clamped = torch.clamp(x.long(), 0, self.embedding.num_embeddings - 1)
        return self.embedding(x_clamped)


class FeatureIDEmbedding(nn.Module):
    """Positional/feature ID embedding."""
    
    def __init__(self, max_features: int, d_model: int):
        """Initialize embedding.
        
        Args:
            max_features: Maximum number of features.
            d_model: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_features, d_model)
    
    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        """Get feature ID embeddings.
        
        Args:
            feature_ids: Feature indices of shape [...].
            
        Returns:
            Embeddings of shape [..., d_model].
        """
        # Clamp indices so datasets with d > max_features still work (extra features reuse last embedding)
        max_idx = self.embedding.num_embeddings - 1
        feature_ids = torch.clamp(feature_ids, 0, max_idx)
        return self.embedding(feature_ids)


class WorldEmbedding(nn.Module):
    """World identifier embedding (baseline vs intervention worlds)."""
    
    def __init__(self, max_worlds: int, d_model: int):
        """Initialize embedding.
        
        Args:
            max_worlds: Maximum number of worlds.
            d_model: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_worlds, d_model)
    
    def forward(self, world_ids: torch.Tensor) -> torch.Tensor:
        """Get world embeddings.
        
        Args:
            world_ids: World indices of shape [...].
            
        Returns:
            Embeddings of shape [..., d_model].
        """
        return self.embedding(world_ids)


class RoleEmbedding(nn.Module):
    """Role embedding (support vs query)."""
    
    def __init__(self, d_model: int):
        """Initialize embedding.
        
        Args:
            d_model: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)  # 0=support, 1=query
    
    def forward(self, role_ids: torch.Tensor) -> torch.Tensor:
        """Get role embeddings.
        
        Args:
            role_ids: Role indices of shape [...] (0=support, 1=query).
            
        Returns:
            Embeddings of shape [..., d_model].
        """
        return self.embedding(role_ids)


class MissingnessEmbedding(nn.Module):
    """Missingness indicator embedding."""
    
    def __init__(self, d_model: int):
        """Initialize embedding.
        
        Args:
            d_model: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)  # 0=present, 1=missing
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Get missingness embeddings.
        
        Args:
            mask: Binary mask of shape [...] (0=present, 1=missing).
            
        Returns:
            Embeddings of shape [..., d_model].
        """
        return self.embedding(mask.long())
