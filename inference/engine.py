"""Low-level inference engine."""

import torch
import numpy as np
from typing import List, Dict, Optional

from model.model import ParallelUniverseTransformer
from scm.intervene import Intervention, InterventionOperator


class InferenceEngine:
    """Low-level inference engine for batch predictions."""
    
    def __init__(
        self,
        model: ParallelUniverseTransformer,
        device: str = "cuda"
    ):
        """Initialize engine.
        
        Args:
            model: Trained model.
            device: Device to run on.
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.intervention_operator = InterventionOperator()
    
    @torch.no_grad()
    def predict_batch(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        interventions: Optional[List[Intervention]] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict for a batch with optional interventions.
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            query_x: Query features [B, Nq, d] (baseline).
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            interventions: Optional list of interventions.
            
        Returns:
            Dictionary of predictions.
        """
        B, Nq, d = query_x.shape
        
        if interventions is None:
            # No interventions, just predict baseline
            query_x_worlds = query_x.unsqueeze(1)  # [B, 1, Nq, d]
        else:
            # Apply interventions
            W = len(interventions) + 1
            query_x_worlds = torch.zeros(B, W, Nq, d, device=self.device)
            query_x_worlds[:, 0, :, :] = query_x  # Baseline
            
            for i, intervention in enumerate(interventions):
                query_x_intv = self.intervention_operator.apply(
                    query_x.cpu().numpy(),
                    intervention
                )
                query_x_worlds[:, i + 1, :, :] = torch.from_numpy(query_x_intv).to(self.device)
        
        # Forward pass
        outputs = self.model(
            support_x, support_y, query_x_worlds,
            feature_types, cardinalities
        )
        
        return outputs
    
    @torch.no_grad()
    def predict_chunked(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        feature_types: torch.Tensor,
        cardinalities: torch.Tensor,
        interventions: List[Intervention],
        chunk_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Predict with chunked interventions for efficiency.
        
        Args:
            support_x: Support features [B, Ns, d].
            support_y: Support outcomes [B, Ns].
            query_x: Query features [B, Nq, d] (baseline).
            feature_types: Feature types [d].
            cardinalities: Cardinalities [d].
            interventions: List of interventions.
            chunk_size: Number of interventions per chunk.
            
        Returns:
            Dictionary of predictions.
        """
        all_predictions = []
        all_log_vars = []
        all_deltas = []
        
        # Process baseline once
        baseline_outputs = self.predict_batch(
            support_x, support_y, query_x,
            feature_types, cardinalities,
            interventions=None
        )
        baseline_pred = baseline_outputs['prediction'][:, 0, :]  # [B, Nq]
        baseline_log_var = baseline_outputs['log_var'][:, 0, :]
        
        all_predictions.append(baseline_pred.unsqueeze(1))
        all_log_vars.append(baseline_log_var.unsqueeze(1))
        
        # Process interventions in chunks
        for i in range(0, len(interventions), chunk_size):
            chunk = interventions[i:i + chunk_size]
            
            outputs = self.predict_batch(
                support_x, support_y, query_x,
                feature_types, cardinalities,
                interventions=chunk
            )
            
            # Extract intervention predictions (skip baseline)
            chunk_pred = outputs['prediction'][:, 1:, :]  # [B, chunk_size, Nq]
            chunk_log_var = outputs['log_var'][:, 1:, :]
            chunk_deltas = outputs['deltas']
            
            all_predictions.append(chunk_pred)
            all_log_vars.append(chunk_log_var)
            all_deltas.append(chunk_deltas)
        
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=1)  # [B, 1+K, Nq]
        log_vars = torch.cat(all_log_vars, dim=1)
        deltas = torch.cat(all_deltas, dim=1)  # [B, K, Nq]
        
        return {
            'prediction': predictions,
            'log_var': log_vars,
            'deltas': deltas
        }
    
    def compute_ate(
        self,
        predictions: torch.Tensor,
        baseline_idx: int = 0
    ) -> torch.Tensor:
        """Compute Average Treatment Effect.
        
        Args:
            predictions: Predictions of shape [B, W, Nq].
            baseline_idx: Index of baseline world.
            
        Returns:
            ATE for each intervention [W-1].
        """
        baseline = predictions[:, baseline_idx:baseline_idx+1, :]
        interventions = torch.cat([
            predictions[:, :baseline_idx, :],
            predictions[:, baseline_idx+1:, :]
        ], dim=1)
        
        deltas = interventions - baseline
        ate = deltas.mean(dim=(0, 2))  # Average over batch and samples
        
        return ate
    
    def compute_cate(
        self,
        predictions: torch.Tensor,
        baseline_idx: int = 0
    ) -> torch.Tensor:
        """Compute Conditional Average Treatment Effect.
        
        Args:
            predictions: Predictions of shape [B, W, Nq].
            baseline_idx: Index of baseline world.
            
        Returns:
            CATE for each sample [B, W-1, Nq].
        """
        baseline = predictions[:, baseline_idx:baseline_idx+1, :]
        interventions = torch.cat([
            predictions[:, :baseline_idx, :],
            predictions[:, baseline_idx+1:, :]
        ], dim=1)
        
        cate = interventions - baseline
        
        return cate
