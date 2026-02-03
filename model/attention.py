"""Attention mechanism primitives."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """Initialize attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape [B, N, d_model].
            mask: Optional attention mask of shape [B, N, N].
            return_attention: Whether to return attention weights.
            
        Returns:
            Output tensor of shape [B, N, d_model].
        """
        B, N, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [B, N, d_model]
        K = self.k_proj(x)  # [B, N, d_model]
        V = self.v_proj(x)  # [B, N, d_model]
        
        # Reshape to [B, n_heads, N, d_head]
        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, N, N]
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, n_heads, N, d_head]
        
        # Reshape back to [B, N, d_model]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        # Output projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out


class CrossAttention(nn.Module):
    """Multi-head cross-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """Initialize cross-attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply multi-head cross-attention.
        
        Args:
            query: Query tensor of shape [B, Nq, d_model].
            memory: Memory tensor of shape [B, Nm, d_model].
            mask: Optional attention mask of shape [B, Nq, Nm].
            
        Returns:
            Output tensor of shape [B, Nq, d_model].
        """
        B, Nq, _ = query.shape
        _, Nm, _ = memory.shape
        
        # Project
        Q = self.q_proj(query)  # [B, Nq, d_model]
        K = self.k_proj(memory)  # [B, Nm, d_model]
        V = self.v_proj(memory)  # [B, Nm, d_model]
        
        # Reshape to [B, n_heads, N, d_head]
        Q = Q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Nm, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Nm, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, Nq, Nm]
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, n_heads, Nq, d_head]
        
        # Reshape back to [B, Nq, d_model]
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        
        # Output projection
        out = self.out_proj(out)
        
        return out
