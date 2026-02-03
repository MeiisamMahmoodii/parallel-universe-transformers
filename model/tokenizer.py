"""Full tabular tokenization pipeline."""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import (
    ContinuousEncoder,
    CategoricalEncoder,
    FeatureIDEmbedding,
    WorldEmbedding,
    RoleEmbedding,
    MissingnessEmbedding,
)


class TabularTokenizer(nn.Module):
    """Converts tabular data to transformer tokens."""
    
    def __init__(
        self,
        d_model: int = 256,
        max_features: int = 50,
        max_worlds: int = 20,
        max_cardinality: int = 100,
        n_fourier: int = 16,
        continuous_hidden_dim: int = 128,
    ):
        """Initialize tokenizer.
        
        Args:
            d_model: Token embedding dimension.
            max_features: Maximum number of features.
            max_worlds: Maximum number of worlds.
            max_cardinality: Maximum categorical cardinality.
            n_fourier: Number of Fourier features for continuous encoding.
            continuous_hidden_dim: Hidden dimension for continuous encoder MLP.
        """
        super().__init__()
        self.d_model = d_model
        
        # Value encoders
        self.continuous_encoder = ContinuousEncoder(d_model, n_fourier, continuous_hidden_dim)
        self.categorical_encoder = CategoricalEncoder(d_model, max_cardinality)
        
        # Metadata embeddings
        self.feature_id_embedding = FeatureIDEmbedding(max_features, d_model)
        self.world_embedding = WorldEmbedding(max_worlds, d_model)
        self.role_embedding = RoleEmbedding(d_model)
        self.missingness_embedding = MissingnessEmbedding(d_model)
        
        # Special token for Y (outcome)
        self.y_token_embedding = nn.Parameter(torch.randn(d_model))
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
    
    def encode_features(
        self,
        x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode feature values.
        
        Args:
            x: Feature values of shape [..., n_features].
            feature_types: Feature types of shape [n_features] (0=continuous, 1=categorical).
            cardinalities: Cardinalities of shape [n_features].
            mask: Missingness mask of shape [..., n_features] (optional).
            
        Returns:
            Encoded features of shape [..., n_features, d_model].
        """
        *batch_dims, n_features = x.shape
        device = x.device
        
        # Initialize output
        encoded = torch.zeros(*batch_dims, n_features, self.d_model, device=device)
        
        # Encode continuous features
        continuous_mask = (feature_types == 0)
        if continuous_mask.any():
            continuous_indices = torch.where(continuous_mask)[0]
            for idx in continuous_indices:
                encoded[..., idx, :] = self.continuous_encoder(x[..., idx])
        
        # Encode categorical features
        categorical_mask = (feature_types == 1)
        if categorical_mask.any():
            categorical_indices = torch.where(categorical_mask)[0]
            for idx in categorical_indices:
                encoded[..., idx, :] = self.categorical_encoder(
                    x[..., idx],
                    cardinalities[idx]
                )
        
        return encoded
    
    def tokenize_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Tokenize support set (with Y values).
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            support_mask: Missingness mask [B, Ns, d] (optional).
            
        Returns:
            Support tokens of shape [B, Ns, d+1, d_model].
            (d feature tokens + 1 Y token per sample)
        """
        B, Ns, d = support_x.shape
        device = support_x.device
        
        # Encode features
        feature_tokens = self.encode_features(support_x, feature_types, cardinalities, support_mask)
        
        # Add feature ID embeddings
        feature_ids = torch.arange(d, device=device).unsqueeze(0).unsqueeze(0).expand(B, Ns, -1)
        feature_tokens = feature_tokens + self.feature_id_embedding(feature_ids)
        
        # Add role embedding (support)
        role_ids = torch.zeros(B, Ns, d, dtype=torch.long, device=device)
        feature_tokens = feature_tokens + self.role_embedding(role_ids)
        
        # Add world embedding (baseline world 0)
        world_ids = torch.zeros(B, Ns, d, dtype=torch.long, device=device)
        feature_tokens = feature_tokens + self.world_embedding(world_ids)
        
        # Add missingness embedding
        if support_mask is not None:
            feature_tokens = feature_tokens + self.missingness_embedding(support_mask)
        
        # Create Y tokens
        y_tokens = self.y_token_embedding.unsqueeze(0).unsqueeze(0).expand(B, Ns, -1)
        
        # Encode Y values (treat as continuous)
        y_encoded = self.continuous_encoder(support_y.unsqueeze(-1))
        y_tokens = y_tokens + y_encoded
        
        # Add role embedding to Y tokens
        y_tokens = y_tokens + self.role_embedding(torch.zeros(B, Ns, dtype=torch.long, device=device))
        
        # Concatenate: [feature_tokens, y_token] per sample
        # Reshape: [B, Ns, d, d_model] + [B, Ns, 1, d_model] -> [B, Ns, d+1, d_model]
        y_tokens = y_tokens.unsqueeze(2)  # [B, Ns, 1, d_model]
        tokens = torch.cat([feature_tokens, y_tokens], dim=2)  # [B, Ns, d+1, d_model]
        
        # Flatten to [B, Ns*(d+1), d_model]
        tokens = tokens.reshape(B, Ns * (d + 1), self.d_model)
        
        return self.layer_norm(tokens)
    
    def tokenize_query(
        self,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        world_ids: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Tokenize query set (without Y values).
        
        Args:
            query_x: Query features [B, W, Nq, d].
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            world_ids: World IDs [W] (0=baseline, 1..K=interventions).
            query_mask: Missingness mask [B, W, Nq, d] (optional).
            
        Returns:
            Query tokens of shape [B, W, Nq, d+1, d_model].
            (d feature tokens + 1 placeholder Y token per sample)
        """
        B, W, Nq, d = query_x.shape
        device = query_x.device
        
        # Reshape to [B*W, Nq, d] for batch processing
        query_x_flat = query_x.reshape(B * W, Nq, d)
        if query_mask is not None:
            query_mask_flat = query_mask.reshape(B * W, Nq, d)
        else:
            query_mask_flat = None
        
        # Encode features
        feature_tokens = self.encode_features(query_x_flat, feature_types, cardinalities, query_mask_flat)
        
        # Add feature ID embeddings
        feature_ids = torch.arange(d, device=device).unsqueeze(0).unsqueeze(0).expand(B * W, Nq, -1)
        feature_tokens = feature_tokens + self.feature_id_embedding(feature_ids)
        
        # Add role embedding (query)
        role_ids = torch.ones(B * W, Nq, d, dtype=torch.long, device=device)
        feature_tokens = feature_tokens + self.role_embedding(role_ids)
        
        # Add world embeddings (different for each world)
        world_ids_expanded = world_ids.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, W, Nq, d)
        world_ids_flat = world_ids_expanded.reshape(B * W, Nq, d)
        feature_tokens = feature_tokens + self.world_embedding(world_ids_flat)
        
        # Add missingness embedding
        if query_mask_flat is not None:
            feature_tokens = feature_tokens + self.missingness_embedding(query_mask_flat)
        
        # Create placeholder Y tokens (masked)
        y_tokens = self.y_token_embedding.unsqueeze(0).unsqueeze(0).expand(B * W, Nq, -1)
        y_tokens = y_tokens + self.role_embedding(torch.ones(B * W, Nq, dtype=torch.long, device=device))
        
        # Concatenate: [feature_tokens, y_token] per sample
        y_tokens = y_tokens.unsqueeze(2)  # [B*W, Nq, 1, d_model]
        tokens = torch.cat([feature_tokens, y_tokens], dim=2)  # [B*W, Nq, d+1, d_model]
        
        # Flatten to [B*W, Nq*(d+1), d_model]
        tokens = tokens.reshape(B * W, Nq * (d + 1), self.d_model)
        
        # Reshape back to [B, W, Nq*(d+1), d_model]
        tokens = tokens.reshape(B, W, Nq * (d + 1), self.d_model)
        
        return self.layer_norm(tokens)
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """Full tokenization of support and query sets.
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            query_x: Query features [B, W, Nq, d].
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            support_mask: Support missingness [B, Ns, d] (optional).
            query_mask: Query missingness [B, W, Nq, d] (optional).
            
        Returns:
            Tuple of (support_tokens, query_tokens):
                support_tokens: [B, Ns*(d+1), d_model]
                query_tokens: [B, W, Nq*(d+1), d_model]
        """
        B, W, Nq, d = query_x.shape
        device = query_x.device
        
        # Tokenize support
        support_tokens = self.tokenize_support(
            support_x, support_y, feature_types, cardinalities, support_mask
        )
        
        # Tokenize query (with world IDs)
        world_ids = torch.arange(W, device=device)
        query_tokens = self.tokenize_query(
            query_x, feature_types, cardinalities, world_ids, query_mask
        )
        
        return support_tokens, query_tokens
