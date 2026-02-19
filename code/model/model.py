"""Full Parallel Universe Transformer model."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .tokenizer import TabularTokenizer
from .backbone import TransformerEncoder
from .heads import CombinedHead


class ParallelUniverseTransformer(nn.Module):
    """Parallel Universe Transformer for causal effect estimation."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_features: int = 50,
        max_worlds: int = 20,
        max_cardinality: int = 100,
        n_fourier: int = 16,
        cross_world_layers: Optional[List[int]] = None,
        attend_to_all_worlds: bool = True,
        use_gradient_checkpointing: bool = False,
        use_quantiles: bool = False,
        n_quantiles: int = 5
    ):
        """Initialize model.
        
        Args:
            d_model: Model dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            d_ff: Feedforward dimension.
            dropout: Dropout probability.
            max_features: Maximum number of features.
            max_worlds: Maximum number of worlds.
            max_cardinality: Maximum categorical cardinality.
            n_fourier: Number of Fourier features for continuous encoding.
            cross_world_layers: Layers to insert cross-world attention (e.g., [3, 5]).
            attend_to_all_worlds: Whether to attend to all worlds in cross-attention.
            use_gradient_checkpointing: Whether to use gradient checkpointing.
            use_quantiles: Whether to use quantile prediction.
            n_quantiles: Number of quantiles (if use_quantiles=True).
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_features = max_features
        
        # Tokenizer
        self.tokenizer = TabularTokenizer(
            d_model=d_model,
            max_features=max_features,
            max_worlds=max_worlds,
            max_cardinality=max_cardinality,
            n_fourier=n_fourier
        )
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            cross_world_layers=cross_world_layers or [3, 5],
            attend_to_all_worlds=attend_to_all_worlds,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Prediction heads
        self.heads = CombinedHead(
            d_model=d_model,
            hidden_dim=128,
            dropout=dropout,
            use_quantiles=use_quantiles,
            n_quantiles=n_quantiles
        )
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            query_x: Query features [B, W, Nq, d].
            feature_types: Feature types [d] (0=continuous, 1=categorical).
            cardinalities: Cardinalities [d].
            support_mask: Support missingness [B, Ns, d] (optional).
            query_mask: Query missingness [B, W, Nq, d] (optional).
            
        Returns:
            Dictionary with:
                - 'prediction': [B, W, Nq] - predicted outcomes per world
                - 'log_var': [B, W, Nq] - log variance per world
                - 'deltas': [B, W-1, Nq] - effect estimates (world_i - baseline)
                - 'quantiles': [B, W, Nq, n_quantiles] (if use_quantiles=True)
        """
        B, Ns, d = support_x.shape
        _, W, Nq, _ = query_x.shape
        
        # Tokenize
        support_tokens, query_tokens = self.tokenizer(
            support_x, support_y, query_x,
            feature_types, cardinalities,
            support_mask, query_mask
        )
        
        # support_tokens: [B, Ns*(d+1), d_model]
        # query_tokens: [B, W, Nq*(d+1), d_model]
        
        # Concatenate support and query tokens
        # We need to expand support for each world
        support_tokens_expanded = support_tokens.unsqueeze(1).expand(B, W, -1, -1)
        
        # Concatenate: [B, W, Ns*(d+1) + Nq*(d+1), d_model]
        all_tokens = torch.cat([support_tokens_expanded, query_tokens], dim=2)
        
        # Flatten worlds into batch: [B*W, Ns*(d+1) + Nq*(d+1), d_model]
        all_tokens_flat = all_tokens.reshape(B * W, -1, self.d_model)
        
        # Pass through encoder
        Ns_tokens = Ns * (d + 1)
        Nq_tokens = Nq * (d + 1)
        
        hidden = self.encoder(all_tokens_flat, W, Ns_tokens, Nq_tokens)
        
        # Extract query hidden states (skip support)
        query_hidden = hidden[:, Ns_tokens:, :]  # [B*W, Nq*(d+1), d_model]
        
        # Extract Y token representations (last token per sample)
        # Y tokens are at positions: d, 2*(d+1)-1, 3*(d+1)-1, ...
        y_token_indices = torch.arange(d, Nq_tokens, d + 1, device=hidden.device)
        y_hidden = query_hidden[:, y_token_indices, :]  # [B*W, Nq, d_model]
        
        # Apply prediction heads
        outputs = self.heads(y_hidden)
        
        # Reshape outputs
        prediction = outputs['prediction'].squeeze(-1).view(B, W, Nq)
        log_var = outputs['log_var'].squeeze(-1).view(B, W, Nq)
        
        # Compute deltas (intervention - baseline)
        baseline_pred = prediction[:, 0:1, :]  # [B, 1, Nq]
        deltas = prediction[:, 1:, :] - baseline_pred  # [B, W-1, Nq]
        
        result = {
            'prediction': prediction,
            'log_var': log_var,
            'deltas': deltas
        }
        
        if 'quantiles' in outputs:
            quantiles = outputs['quantiles'].view(B, W, Nq, -1)
            result['quantiles'] = quantiles
        
        return result
    
    def predict_interventions(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x_baseline: torch.Tensor,
        interventions: List,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Predict outcomes for multiple interventions (inference mode).
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            query_x_baseline: Query features [B, Nq, d] (baseline).
            interventions: List of Intervention objects.
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            support_mask: Support missingness [B, Ns, d] (optional).
            chunk_size: Number of interventions to process at once.
            
        Returns:
            Dictionary with predictions for all interventions.
        """
        from scm.intervene import InterventionOperator
        
        B, Nq, d = query_x_baseline.shape
        device = query_x_baseline.device
        
        # Initialize intervention operator
        intv_op = InterventionOperator()
        
        # Process interventions in chunks. Each chunk forward returns [Baseline, Intv_1, ..., Intv_k].
        # We must take baseline only from the first chunk and only counterfactuals from the rest,
        # then concatenate, so the result is [Baseline, Intv_1, ..., Intv_K] with no duplicate baselines.
        all_cf_predictions = []
        all_cf_log_vars = []
        all_deltas = []
        baseline_pred = None
        baseline_log_var = None

        for i in range(0, len(interventions), chunk_size):
            chunk_interventions = interventions[i:i + chunk_size]
            W_chunk = len(chunk_interventions) + 1  # +1 for baseline

            # Build query_x for this chunk
            query_x_chunk = torch.zeros(B, W_chunk, Nq, d, device=device)
            query_x_chunk[:, 0, :, :] = query_x_baseline  # Baseline

            for j, intervention in enumerate(chunk_interventions):
                # Apply intervention
                query_x_intv = intv_op.apply(
                    query_x_baseline.cpu().numpy(),
                    intervention
                )
                query_x_chunk[:, j + 1, :, :] = torch.from_numpy(query_x_intv).to(device)

            # Forward pass
            outputs = self.forward(
                support_x, support_y, query_x_chunk,
                feature_types, cardinalities,
                support_mask, None
            )

            # First chunk: keep baseline; all chunks: keep only counterfactuals (strip index 0)
            cf_chunk = outputs['prediction'][:, 1:, :]  # [B, W_chunk-1, Nq]
            if baseline_pred is None:
                baseline_pred = outputs['prediction'][:, 0:1, :]   # [B, 1, Nq]
                baseline_log_var = outputs['log_var'][:, 0:1, :]
            all_cf_predictions.append(cf_chunk)
            all_cf_log_vars.append(outputs['log_var'][:, 1:, :])
            all_deltas.append(outputs['deltas'])

        # Single baseline + all counterfactuals in order
        counterfactuals = torch.cat(all_cf_predictions, dim=1)  # [B, K, Nq]
        predictions = torch.cat([baseline_pred, counterfactuals], dim=1)  # [B, 1+K, Nq]
        log_vars = torch.cat([baseline_log_var, torch.cat(all_cf_log_vars, dim=1)], dim=1)
        deltas = torch.cat(all_deltas, dim=1)

        K = len(interventions)
        ret_cf = predictions[:, 1:, :]
        # Ensure we return exactly K counterfactuals (fixes chunking edge case when K > chunk_size)
        if ret_cf.shape[1] != K:
            ret_cf = ret_cf[:, :K, :]
        ret_deltas = deltas[:, :K, :] if deltas.shape[1] != K else deltas

        return {
            'baseline': predictions[:, 0, :],
            'counterfactuals': ret_cf,
            'deltas': ret_deltas,
            'log_var': log_vars,
            'uncertainty': torch.exp(0.5 * log_vars)  # Standard deviation
        }
