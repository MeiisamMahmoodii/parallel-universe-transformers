"""Inter-world cross-attention module."""

import torch
import torch.nn as nn
from .attention import CrossAttention


class CrossWorldAttention(nn.Module):
    """Cross-attention between parallel worlds."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attend_to_all: bool = True
    ):
        """Initialize cross-world attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            attend_to_all: If True, attend to all worlds. If False, only attend to baseline.
        """
        super().__init__()
        self.d_model = d_model
        self.attend_to_all = attend_to_all
        
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        W: int,
        Ns: int,
        Nq: int
    ) -> torch.Tensor:
        """Apply inter-world cross-attention.
        
        Args:
            hidden_states: Hidden states of shape [B*W, Ns+Nq*(d+1), d_model].
                           Contains support set + query sets for all worlds.
            W: Number of worlds.
            Ns: Support set size (in tokens).
            Nq: Query set size (in tokens) per world.
            
        Returns:
            Updated hidden states of shape [B*W, Ns+Nq, d_model].
        """
        BW, N, d_model = hidden_states.shape
        B = BW // W
        
        # Reshape to separate worlds: [B, W, N, d_model]
        hidden_states_worlds = hidden_states.view(B, W, N, d_model)
        
        # Extract query portions (skip support set)
        query_hidden = hidden_states_worlds[:, :, Ns:, :]  # [B, W, Nq, d_model]
        
        # Process each world
        updated_query = []
        
        for w in range(W):
            # Current world's query hidden states
            query_w = query_hidden[:, w, :, :]  # [B, Nq, d_model]
            
            # Build memory from other worlds
            if self.attend_to_all:
                # Attend to all worlds' queries
                memory = query_hidden.reshape(B, W * Nq, d_model)  # [B, W*Nq, d_model]
            else:
                # Attend only to baseline world (world 0)
                memory = query_hidden[:, 0, :, :]  # [B, Nq, d_model]
            
            # Apply cross-attention
            query_w_updated = self.cross_attn(query_w, memory)
            
            # Residual connection and layer norm
            query_w_updated = self.layer_norm(query_w + self.dropout(query_w_updated))
            
            updated_query.append(query_w_updated)
        
        # Stack back: [B, W, Nq, d_model]
        updated_query = torch.stack(updated_query, dim=1)
        
        # Combine with support set (unchanged)
        support_hidden = hidden_states_worlds[:, :, :Ns, :]  # [B, W, Ns, d_model]
        
        # Concatenate: [B, W, Ns+Nq, d_model]
        combined = torch.cat([support_hidden, updated_query], dim=2)
        
        # Flatten back to [B*W, Ns+Nq, d_model]
        return combined.view(BW, N, d_model)


class PooledCrossWorldAttention(nn.Module):
    """Cross-attention using pooled representations of other worlds."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        pool_method: str = "mean"
    ):
        """Initialize pooled cross-world attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            pool_method: Pooling method ('mean', 'max', or 'attention').
        """
        super().__init__()
        self.d_model = d_model
        self.pool_method = pool_method
        
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if pool_method == "attention":
            self.pool_attn = nn.Linear(d_model, 1)
    
    def pool_world(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pool hidden states within a world.
        
        Args:
            hidden: Hidden states of shape [B, N, d_model].
            
        Returns:
            Pooled representation of shape [B, 1, d_model].
        """
        if self.pool_method == "mean":
            return hidden.mean(dim=1, keepdim=True)
        elif self.pool_method == "max":
            return hidden.max(dim=1, keepdim=True)[0]
        elif self.pool_method == "attention":
            # Attention-based pooling
            scores = self.pool_attn(hidden)  # [B, N, 1]
            weights = torch.softmax(scores, dim=1)
            return (hidden * weights).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        W: int,
        Ns: int,
        Nq: int
    ) -> torch.Tensor:
        """Apply pooled inter-world cross-attention.
        
        Args:
            hidden_states: Hidden states of shape [B*W, Ns+Nq, d_model].
            W: Number of worlds.
            Ns: Support set size (in tokens).
            Nq: Query set size (in tokens) per world.
            
        Returns:
            Updated hidden states of shape [B*W, Ns+Nq, d_model].
        """
        BW, N, d_model = hidden_states.shape
        B = BW // W
        
        # Reshape to separate worlds: [B, W, N, d_model]
        hidden_states_worlds = hidden_states.view(B, W, N, d_model)
        
        # Extract query portions
        query_hidden = hidden_states_worlds[:, :, Ns:, :]  # [B, W, Nq, d_model]
        
        # Pool each world's representation
        pooled_worlds = []
        for w in range(W):
            pooled = self.pool_world(query_hidden[:, w, :, :])  # [B, 1, d_model]
            pooled_worlds.append(pooled)
        
        pooled_memory = torch.cat(pooled_worlds, dim=1)  # [B, W, d_model]
        
        # Process each world
        updated_query = []
        
        for w in range(W):
            query_w = query_hidden[:, w, :, :]  # [B, Nq, d_model]
            
            # Cross-attend to pooled representations
            query_w_updated = self.cross_attn(query_w, pooled_memory)
            
            # Residual connection and layer norm
            query_w_updated = self.layer_norm(query_w + self.dropout(query_w_updated))
            
            updated_query.append(query_w_updated)
        
        # Stack back
        updated_query = torch.stack(updated_query, dim=1)
        
        # Combine with support set
        support_hidden = hidden_states_worlds[:, :, :Ns, :]
        combined = torch.cat([support_hidden, updated_query], dim=2)
        
        return combined.view(BW, N, d_model)
