"""Public API for inference."""

from dataclasses import dataclass
from typing import List, Optional, Literal, Union
import pandas as pd
import torch
import numpy as np

from model.model import ParallelUniverseTransformer
from scm.intervene import Intervention as SCMIntervention, InterventionType
from scm.schema import FeatureType


def predict(
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    interventions_list: List["Intervention"],
    checkpoint_path: str,
    device: str = "cuda",
    feature_schema: Optional[dict] = None,
    chunk_size: int = 8,
) -> "InterventionResults":
    """Single-call prediction: load checkpoint, infer schema, run model, return results.

    Infers feature types and cardinalities from the dataframes, builds tensors,
    runs the model (with chunking if many interventions), and returns baseline,
    interventional predictions, deltas, and uncertainty.

    Args:
        support_df: Observational (x, y) support set. Must include outcome column for conditioning;
            if no outcome column is present, a placeholder is used (model may still run).
        query_df: Query rows (features only or with outcome for validation).
        interventions_list: List of Intervention specs (feature, type, value).
        checkpoint_path: Path to model checkpoint (.pt).
        device: Device to run on ("cuda" or "cpu").
        feature_schema: Optional dict mapping column name to {type, cardinality}; if None, inferred from data.
        chunk_size: Number of interventions to process at once (for memory).

    Returns:
        InterventionResults with baseline, counterfactuals, deltas, uncertainty, feature_names, intervention_specs.
    """
    model = ParallelUniverseModel.from_pretrained(checkpoint_path, device=device)
    return model.predict_interventions(
        support_df,
        query_df,
        interventions_list,
        feature_schema=feature_schema,
        chunk_size=chunk_size,
    )


@dataclass
class Intervention:
    """User-facing intervention specification."""
    feature: str
    type: Literal['set', 'shift', 'randomize']
    value: Optional[float] = None


@dataclass
class InterventionResults:
    """Results from intervention predictions."""
    baseline: np.ndarray  # [n_samples]
    counterfactuals: np.ndarray  # [n_interventions, n_samples]
    deltas: np.ndarray  # [n_interventions, n_samples]
    uncertainty: np.ndarray  # [n_interventions+1, n_samples] (baseline + interventions)
    feature_names: List[str]
    intervention_specs: List[Intervention]


class ParallelUniverseModel:
    """High-level API for the Parallel Universe Transformer."""
    
    def __init__(
        self,
        model: ParallelUniverseTransformer,
        device: str = "cuda"
    ):
        """Initialize API.
        
        Args:
            model: Trained model.
            device: Device to run on.
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cuda"):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.
            device: Device to run on.
            
        Returns:
            ParallelUniverseModel instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        
        # Create model
        model = ParallelUniverseTransformer(
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 1024),
            dropout=config.get('dropout', 0.1),
            cross_world_layers=config.get('cross_world_layers', [3, 5]),
            attend_to_all_worlds=config.get('attend_to_all_worlds', True),
            use_quantiles=config.get('use_quantiles', False)
        )
        
        # Load weights (strict=False for backward compatibility with checkpoints missing delta_head)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return cls(model, device)
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        feature_schema: Optional[dict] = None
    ) -> tuple:
        """Prepare data for model input.
        
        Args:
            data: DataFrame with features.
            feature_schema: Optional schema with feature types and cardinalities.
            
        Returns:
            Tuple of (x, feature_types, cardinalities, feature_names).
        """
        feature_names = list(data.columns)
        x = data.values.astype(np.float32)
        
        # Infer feature types if not provided
        if feature_schema is None:
            feature_types = []
            cardinalities = []
            
            for col in data.columns:
                # Simple heuristic: if dtype is object or int with few unique values, treat as categorical
                if data[col].dtype == 'object' or (
                    data[col].dtype in ['int64', 'int32'] and data[col].nunique() < 20
                ):
                    feature_types.append(1)  # Categorical
                    cardinalities.append(int(data[col].nunique()))
                else:
                    feature_types.append(0)  # Continuous
                    cardinalities.append(1)
        else:
            feature_types = [schema_info['type'] for schema_info in feature_schema.values()]
            cardinalities = [schema_info.get('cardinality', 1) for schema_info in feature_schema.values()]
        
        feature_types = torch.tensor(feature_types, dtype=torch.long)
        cardinalities = torch.tensor(cardinalities, dtype=torch.long)
        
        return x, feature_types, cardinalities, feature_names
    
    def _interventions_to_scm_format(
        self,
        interventions: List[Intervention],
        feature_names: List[str]
    ) -> List[SCMIntervention]:
        """Convert user interventions to SCM format.
        
        Args:
            interventions: List of user interventions.
            feature_names: List of feature names.
            
        Returns:
            List of SCM interventions.
        """
        scm_interventions = []
        
        for intv in interventions:
            # Find feature index
            if intv.feature not in feature_names:
                raise ValueError(f"Feature '{intv.feature}' not found in data")
            
            feature_idx = feature_names.index(intv.feature)
            
            # Convert type
            if intv.type == 'set':
                intv_type = InterventionType.SET
            elif intv.type == 'shift':
                intv_type = InterventionType.SHIFT
            elif intv.type == 'randomize':
                intv_type = InterventionType.RANDOMIZE
            else:
                raise ValueError(f"Unknown intervention type: {intv.type}")
            
            scm_interventions.append(SCMIntervention(
                feature_idx=feature_idx,
                intervention_type=intv_type,
                value=intv.value
            ))
        
        return scm_interventions
    
    @torch.no_grad()
    def predict_interventions(
        self,
        data: pd.DataFrame,
        query: pd.DataFrame,
        interventions: List[Intervention],
        feature_schema: Optional[dict] = None,
        chunk_size: int = 8
    ) -> InterventionResults:
        """Predict outcomes under interventions.
        
        Args:
            data: Support set (observational data for conditioning).
            query: Query set (rows to predict for).
            interventions: List of interventions to evaluate.
            feature_schema: Optional feature schema.
            chunk_size: Number of interventions to process at once.
            
        Returns:
            InterventionResults object.
        """
        # Prepare data
        support_x, feature_types, cardinalities, feature_names = self._prepare_data(
            data, feature_schema
        )
        query_x, _, _, _ = self._prepare_data(query, feature_schema)
        
        # Convert to tensors
        support_x = torch.from_numpy(support_x).unsqueeze(0).to(self.device)  # [1, Ns, d]
        support_y = torch.zeros(1, support_x.shape[1], device=self.device)  # Placeholder
        query_x_baseline = torch.from_numpy(query_x).unsqueeze(0).to(self.device)  # [1, Nq, d]
        feature_types = feature_types.to(self.device)
        cardinalities = cardinalities.to(self.device)
        
        # Convert interventions
        scm_interventions = self._interventions_to_scm_format(interventions, feature_names)
        
        # Predict
        results = self.model.predict_interventions(
            support_x, support_y, query_x_baseline,
            scm_interventions,
            feature_types, cardinalities,
            chunk_size=chunk_size
        )
        
        # Convert to numpy
        baseline = results['baseline'].squeeze(0).cpu().numpy()
        counterfactuals = results['counterfactuals'].squeeze(0).cpu().numpy()
        deltas = results['deltas'].squeeze(0).cpu().numpy()
        uncertainty = results['uncertainty'].squeeze(0).cpu().numpy()
        
        return InterventionResults(
            baseline=baseline,
            counterfactuals=counterfactuals,
            deltas=deltas,
            uncertainty=uncertainty,
            feature_names=feature_names,
            intervention_specs=interventions
        )
    
    @torch.no_grad()
    def predict_all_features(
        self,
        data: pd.DataFrame,
        query: pd.DataFrame,
        intervention_type: Literal['set', 'shift'] = 'set',
        intervention_values: Optional[dict] = None,
        feature_schema: Optional[dict] = None
    ) -> InterventionResults:
        """Predict outcomes for interventions on all features.
        
        Args:
            data: Support set.
            query: Query set.
            intervention_type: Type of intervention.
            intervention_values: Dictionary mapping feature names to intervention values.
            feature_schema: Optional feature schema.
            
        Returns:
            InterventionResults object.
        """
        feature_names = list(data.columns)
        
        # Create interventions for all features
        interventions = []
        for feature in feature_names:
            if intervention_values and feature in intervention_values:
                value = intervention_values[feature]
            else:
                # Default values
                if intervention_type == 'set':
                    value = data[feature].mean()
                elif intervention_type == 'shift':
                    value = data[feature].std()
                else:
                    value = None
            
            interventions.append(Intervention(
                feature=feature,
                type=intervention_type,
                value=value
            ))
        
        return self.predict_interventions(
            data, query, interventions, feature_schema
        )
