"""Transformer encoder backbone."""

import torch
import torch.nn as nn
from typing import List, Optional
from .attention import MultiHeadAttention
from .cross_world import CrossWorldAttention


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward dimension.
            dropout: Dropout probability.
            activation: Activation function ('relu' or 'gelu').
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, N, d_model].
            mask: Optional attention mask.
            
        Returns:
            Output tensor of shape [B, N, d_model].
        """
        # Self-attention with residual
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with optional cross-world attention layers."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        cross_world_layers: Optional[List[int]] = None,
        attend_to_all_worlds: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        """Initialize transformer encoder.
        
        Args:
            d_model: Model dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            d_ff: Feedforward dimension.
            dropout: Dropout probability.
            cross_world_layers: Indices of layers to insert cross-world attention (e.g., [3, 5]).
            attend_to_all_worlds: If True, attend to all worlds in cross-attention.
            use_gradient_checkpointing: Whether to use gradient checkpointing.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.cross_world_layers = cross_world_layers or []
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Cross-world attention modules
        self.cross_world_attns = nn.ModuleDict({
            str(layer_idx): CrossWorldAttention(d_model, n_heads, dropout, attend_to_all_worlds)
            for layer_idx in self.cross_world_layers
        })
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        W: int,
        Ns: int,
        Nq: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape [B*W, Ns+Nq, d_model].
               Contains support tokens + query tokens for all worlds.
            W: Number of worlds.
            Ns: Number of support tokens.
            Nq: Number of query tokens per world.
            mask: Optional attention mask.
            
        Returns:
            Output tensor of shape [B*W, Ns+Nq, d_model].
        """
        hidden = x
        
        for layer_idx, block in enumerate(self.blocks):
            # Self-attention block
            if self.use_gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(block, hidden, mask)
            else:
                hidden = block(hidden, mask)
            
            # Cross-world attention (if applicable)
            if layer_idx in self.cross_world_layers:
                cross_world_attn = self.cross_world_attns[str(layer_idx)]
                if self.use_gradient_checkpointing and self.training:
                    hidden = torch.utils.checkpoint.checkpoint(
                        cross_world_attn, hidden, W, Ns, Nq
                    )
                else:
                    hidden = cross_world_attn(hidden, W, Ns, Nq)
        
        return self.final_norm(hidden)
