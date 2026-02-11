"""Advanced usage example with custom SCM."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from inference.api import ParallelUniverseModel, Intervention
from scm.schema import FeatureSchema, SchemaConfig
from scm.sample import SCMSampler, SCMConfig
from scm.intervene import InterventionOperator, InterventionType
from scm.counterfactual import CounterfactualGenerator


def main():
    print("Parallel Universe Transformer - Advanced Usage Example")
    print("=" * 60)
    
    # Generate data from a known SCM
    print("\n1. Generating data from custom SCM...")
    
    schema_config = SchemaConfig(
        n_features=10,
        n_continuous=5,
        n_categorical=5,
        seed=42
    )
    schema_sampler = FeatureSchema(schema_config)
    schema = schema_sampler.sample_schema()
    
    scm_config = SCMConfig(
        n_features=10,
        complexity='moderate',
        seed=43
    )
    scm_sampler = SCMSampler(schema, scm_config)
    
    # Generate observational data
    X_obs, Y_obs = scm_sampler.sample(500)
    
    print(f"   Generated {len(X_obs)} observational samples")
    print(f"   Features: {[f.name for f in schema]}")
    
    # Create interventions
    print("\n2. Creating interventions...")
    intv_op = InterventionOperator(seed=44)
    
    interventions_scm = intv_op.sample_interventions(
        n_interventions=5,
        n_features=10,
        complexity='moderate'
    )
    
    print(f"   Created {len(interventions_scm)} interventions")
    
    # Generate ground truth counterfactuals
    print("\n3. Generating ground truth counterfactuals...")
    cf_generator = CounterfactualGenerator(scm_sampler)
    
    X_query = X_obs[:10]
    Y_query = Y_obs[:10]
    
    X_cf_batch, Y_cf_batch = cf_generator.generate_counterfactuals_batch(
        X_query, interventions_scm
    )
    
    print(f"   Generated counterfactuals for {len(X_query)} query samples")
    print(f"   Shape: {X_cf_batch.shape}")
    
    # Compute true treatment effects
    true_deltas = Y_cf_batch - Y_query[None, :]
    true_ate = true_deltas.mean(axis=1)
    
    print("\n4. True Average Treatment Effects (ATE):")
    for i, ate in enumerate(true_ate):
        print(f"   Intervention {i}: ATE = {ate:.4f}")
    
    # Now use the model to predict
    print("\n5. Predicting with model...")
    
    try:
        model = ParallelUniverseModel.from_pretrained('checkpoints/final_model.pt')
        
        # Prepare data
        feature_names = [f.name for f in schema]
        support_df = pd.DataFrame(X_obs[:400], columns=feature_names)
        query_df = pd.DataFrame(X_query, columns=feature_names)
        
        # Convert interventions to API format
        api_interventions = []
        for intv in interventions_scm:
            if intv.intervention_type == InterventionType.SET:
                intv_type = 'set'
            elif intv.intervention_type == InterventionType.SHIFT:
                intv_type = 'shift'
            else:
                intv_type = 'randomize'
            
            api_interventions.append(Intervention(
                feature=feature_names[intv.feature_idx],
                type=intv_type,
                value=intv.value
            ))
        
        # Predict
        results = model.predict_interventions(
            data=support_df,
            query=query_df,
            interventions=api_interventions
        )
        
        # Compare predictions to ground truth
        print("\n6. Comparing predictions to ground truth:")
        
        pred_deltas = results.deltas
        pred_ate = pred_deltas.mean(axis=1)
        
        print("\n   ATE Comparison:")
        print("   " + "-" * 50)
        print(f"   {'Intervention':<15} {'True ATE':<12} {'Pred ATE':<12} {'Error':<10}")
        print("   " + "-" * 50)
        
        for i in range(len(api_interventions)):
            error = abs(true_ate[i] - pred_ate[i])
            print(f"   {i:<15} {true_ate[i]:<12.4f} {pred_ate[i]:<12.4f} {error:<10.4f}")
        
        mae = np.mean(np.abs(true_ate - pred_ate))
        print("   " + "-" * 50)
        print(f"   Mean Absolute Error: {mae:.4f}")
        
        # Individual treatment effects
        print("\n7. Individual Treatment Effect (ITE) accuracy:")
        ite_errors = np.abs(true_deltas - pred_deltas)
        print(f"   Mean ITE error: {ite_errors.mean():.4f}")
        print(f"   Std ITE error: {ite_errors.std():.4f}")
        
    except FileNotFoundError:
        print("   Error: Model checkpoint not found!")
        print("   Please train a model first.")
    
    print("\n" + "=" * 60)
    print("Advanced example completed!")


if __name__ == '__main__':
    main()
