"""Counterfactual outcome computation."""

from typing import List, Tuple, Optional
import numpy as np

from .schema import FeatureInfo, FeatureType
from .sample import SCMSampler
from .intervene import Intervention, InterventionOperator


class CounterfactualGenerator:
    """Generates counterfactual outcomes under interventions."""
    
    def __init__(self, scm_sampler: SCMSampler):
        self.scm_sampler = scm_sampler
        self.schema = scm_sampler.schema
        self.intervention_operator = InterventionOperator(seed=scm_sampler.rng.randint(0, 2**31))
    
    def generate_counterfactual(
        self,
        X: np.ndarray,
        intervention: Intervention
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate counterfactual data and outcomes under an intervention.
        
        This implements the three-step counterfactual inference:
        1. Abduction: Infer noise terms from observed data
        2. Action: Apply intervention
        3. Prediction: Forward simulate with intervened values
        
        Args:
            X: Observational data of shape [n_samples, n_features].
            intervention: Intervention to apply.
            
        Returns:
            Tuple of (X_cf, Y_cf) where:
                X_cf: Counterfactual features [n_samples, n_features]
                Y_cf: Counterfactual outcomes [n_samples]
        """
        n_samples = len(X)
        feature_idx = intervention.feature_idx
        
        # Step 1: Abduction (simplified - we'll use the observed values as basis)
        # In a full implementation, we would invert the mechanisms to get noise terms
        # For now, we use a simpler approach: keep non-descendants, recompute descendants
        
        # Step 2: Action - apply intervention
        X_cf = self.intervention_operator.apply(X, intervention)
        
        # Step 3: Prediction - recompute descendants
        # Get causal order
        causal_order = self.scm_sampler.causal_order
        
        # Find position of intervened feature in causal order
        intervened_position = causal_order.index(feature_idx)
        
        # Recompute all features that come after the intervened feature
        for i in range(intervened_position + 1, len(causal_order)):
            current_feature_idx = causal_order[i]
            mechanism = self.scm_sampler.mechanisms[current_feature_idx]
            
            if mechanism is not None:
                # Get parent values
                parent_indices = mechanism['parents']
                parent_values = X_cf[:, parent_indices]
                
                # Apply mechanism
                values = mechanism['function'](parent_values)
                
                # Add noise (same as in forward sampling)
                noise = self.scm_sampler.noise_sampler.sample_noise(
                    n_samples,
                    noise_type=self.scm_sampler.noise_types[current_feature_idx],
                    scale=self.scm_sampler.noise_scales[current_feature_idx],
                    X=parent_values
                )
                values += noise
                
                # Post-process based on feature type
                feature = self.schema[current_feature_idx]
                if feature.feature_type == FeatureType.CONTINUOUS:
                    values = np.clip(values, feature.min_value, feature.max_value)
                else:
                    values = np.floor((values - values.min()) / (values.max() - values.min() + 1e-8) * feature.cardinality)
                    values = np.clip(values, 0, feature.cardinality - 1).astype(int)
                
                X_cf[:, current_feature_idx] = values
        
        # Compute counterfactual outcome
        Y_cf = self.scm_sampler.outcome_mechanism(X_cf)
        noise = self.scm_sampler.noise_sampler.sample_noise(
            n_samples,
            noise_type=self.scm_sampler.outcome_noise_type,
            scale=self.scm_sampler.outcome_noise_scale,
            X=X_cf
        )
        Y_cf += noise
        
        return X_cf, Y_cf
    
    def generate_counterfactuals_batch(
        self,
        X: np.ndarray,
        interventions: List[Intervention]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate counterfactuals for multiple interventions.
        
        Args:
            X: Observational data of shape [n_samples, n_features].
            interventions: List of interventions.
            
        Returns:
            Tuple of (X_cf_batch, Y_cf_batch) where:
                X_cf_batch: [n_interventions, n_samples, n_features]
                Y_cf_batch: [n_interventions, n_samples]
        """
        X_cf_batch = []
        Y_cf_batch = []
        
        for intervention in interventions:
            X_cf, Y_cf = self.generate_counterfactual(X, intervention)
            X_cf_batch.append(X_cf)
            Y_cf_batch.append(Y_cf)
        
        return np.array(X_cf_batch), np.array(Y_cf_batch)
    
    def compute_ate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        intervention: Intervention
    ) -> float:
        """Compute Average Treatment Effect (ATE).
        
        Args:
            X: Observational data of shape [n_samples, n_features].
            Y: Observational outcomes of shape [n_samples].
            intervention: Intervention to evaluate.
            
        Returns:
            ATE: mean(Y_cf - Y)
        """
        _, Y_cf = self.generate_counterfactual(X, intervention)
        return np.mean(Y_cf - Y)
    
    def compute_cate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        intervention: Intervention
    ) -> np.ndarray:
        """Compute Conditional Average Treatment Effect (CATE).
        
        Args:
            X: Observational data of shape [n_samples, n_features].
            Y: Observational outcomes of shape [n_samples].
            intervention: Intervention to evaluate.
            
        Returns:
            CATE: Individual treatment effects [n_samples]
        """
        _, Y_cf = self.generate_counterfactual(X, intervention)
        return Y_cf - Y
