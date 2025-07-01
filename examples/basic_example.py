"""
Basic example of using the Changes-in-Changes model.

This example demonstrates how to:
1. Generate synthetic data
2. Fit the CiC model
3. Interpret results
4. Compare with traditional DiD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycic import ChangesInChanges

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_control=500, n_treatment=500, treatment_effect=2.0, heterogenous_effect=False, random_state=42):
    """Generate synthetic data for example."""
    np.random.seed(random_state)
    control_before = np.random.normal(10, 2, n_control)
    control_after = control_before + np.random.normal(1, 0.5, n_control)
    treatment_before = np.random.normal(12, 2, n_treatment)
    
    if heterogenous_effect:
        effect_multiplier = (treatment_before - treatment_before.min()) / (treatment_before.max() - treatment_before.min())
        individual_effects = treatment_effect * (0.5 + effect_multiplier)
        treatment_after = treatment_before + np.random.normal(1, 0.5, n_treatment) + individual_effects
    else:
        treatment_after = treatment_before + np.random.normal(1, 0.5, n_treatment) + treatment_effect
    
    return pd.DataFrame({
        'outcome': np.concatenate([control_before, control_after, treatment_before, treatment_after]),
        'group': np.concatenate([['control']*n_control, ['control']*n_control, ['treatment']*n_treatment, ['treatment']*n_treatment]),
        'period': np.concatenate([['before']*n_control, ['after']*n_control, ['before']*n_treatment, ['after']*n_treatment])
    })

def compute_did_estimate(data, outcome='outcome', group='group', period='period', 
                        control_group='control', treatment_group='treatment', 
                        before_period='before', after_period='after'):
    """Compute traditional DiD estimate."""
    control_before_mean = data[(data[group] == control_group) & (data[period] == before_period)][outcome].mean()
    control_after_mean = data[(data[group] == control_group) & (data[period] == after_period)][outcome].mean()
    treatment_before_mean = data[(data[group] == treatment_group) & (data[period] == before_period)][outcome].mean()
    treatment_after_mean = data[(data[group] == treatment_group) & (data[period] == after_period)][outcome].mean()
    return (treatment_after_mean - treatment_before_mean) - (control_after_mean - control_before_mean)

def main():
    print("=" * 60)
    print("Changes-in-Changes Model: Basic Example")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(
        n_control=500,
        n_treatment=500,
        treatment_effect=2.0,
        heterogenous_effect=False,
        random_state=42
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Control group size: {len(data[data['group'] == 'control'])}")
    print(f"Treatment group size: {len(data[data['group'] == 'treatment'])}")
    
    # Display summary statistics
    print("\n2. Data summary:")
    summary = data.groupby(['group', 'period'])['outcome'].agg(['count', 'mean', 'std']).round(3)
    print(summary)
    
    # Fit the Changes-in-Changes model
    print("\n3. Fitting Changes-in-Changes model...")
    cic = ChangesInChanges(n_quantiles=100, random_state=42)
    results = cic.fit(
        data, 
        outcome='outcome', 
        group='group', 
        period='period',
        control_group='control', 
        treatment_group='treatment',
        before_period='before', 
        after_period='after'
    )
    
    # Display results
    print("\n4. Model results:")
    print(results.summary())
    
    # Compare with traditional DiD
    print("\n5. Comparison with traditional DiD:")
    did_estimate = compute_did_estimate(
        data, 'outcome', 'group', 'period',
        'control', 'treatment', 'before', 'after'
    )
    print(f"Traditional DiD estimate: {did_estimate:.4f}")
    print(f"CiC mean treatment effect: {results.mean_treatment_effect:.4f}")
    print(f"Difference: {abs(did_estimate - results.mean_treatment_effect):.4f}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    
    # Plot comprehensive results
    results.plot(figsize=(12, 8))
    plt.savefig('basic_example_results.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive results plot to 'basic_example_results.png'")
    
    # Plot treatment effects with confidence intervals
    print("\n7. Computing bootstrap confidence intervals...")
    lower_bound, upper_bound = cic.bootstrap_ci(n_bootstrap=500, confidence_level=0.95)
    
    results.plot_treatment_effects(
        confidence_intervals=(lower_bound, upper_bound),
        figsize=(10, 6)
    )
    plt.savefig('basic_example_treatment_effects.png', dpi=300, bbox_inches='tight')
    print("Saved treatment effects plot to 'basic_example_treatment_effects.png'")
    
    # Export results to CSV
    print("\n8. Exporting results...")
    results_df = results.to_dataframe()
    results_df.to_csv('basic_example_results.csv', index=False)
    print("Saved results to 'basic_example_results.csv'")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 