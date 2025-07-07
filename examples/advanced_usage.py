"""
Advanced Changes-in-Changes (CiC) estimation example.

This example demonstrates heterogeneous treatment effects, bootstrap confidence intervals,
visualization, and exporting results to spreadsheet format.
"""

import sys
import os
# Add parent directory to path to import pycic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycic import ChangesInChanges

# Set random seed for reproducibility
np.random.seed(1988)

def generate_heterogeneous_data(n_control=500, n_treatment=500, base_effect=2.0, random_state=42):
    """Generate synthetic data with heterogeneous treatment effects."""
    np.random.seed(random_state)
    
    # Control group
    control_before = np.random.normal(10, 2, n_control)
    control_after = control_before + np.random.normal(1, 0.5, n_control)
    
    # Treatment group with heterogeneous effects
    treatment_before = np.random.normal(12, 2, n_treatment)
    
    # Create heterogeneous effects based on pre-treatment outcomes
    effect_multiplier = (treatment_before - treatment_before.min()) / (treatment_before.max() - treatment_before.min())
    individual_effects = base_effect * (0.5 + effect_multiplier)  # Effects vary from 0.5*base to 1.5*base
    
    treatment_after = treatment_before + np.random.normal(1, 0.5, n_treatment) + individual_effects
    
    return pd.DataFrame({
        'outcome': np.concatenate([control_before, control_after, treatment_before, treatment_after]),
        'group': np.concatenate([[0]*n_control, [0]*n_control, [1]*n_treatment, [1]*n_treatment]),
        'period': np.concatenate([[0]*n_control, [1]*n_control, [0]*n_treatment, [1]*n_treatment])
    })

def main():
    print("Advanced Changes-in-Changes Estimation")
    print("=" * 50)
    print("Heterogeneous Treatment Effects with Confidence Intervals")
    print("=" * 50)
    
    # Generate data with heterogeneous effects
    data = generate_heterogeneous_data(n_control=500, n_treatment=500, base_effect=2.0)
    
    # Fit CiC model
    cic = ChangesInChanges(n_quantiles=100)  # More quantiles for smoother curves
    results = cic.fit(data, outcome='outcome', group='group', period='period')
    
    # Display treatment effects at key quantiles
    print("\nTreatment Effects at Key Quantiles:")
    print("-" * 35)
    
    # Get treatment effects at specific quantiles
    quantiles = results.quantiles
    effects = results.treatment_effects
    
    # Show effects at 10th, 25th, 50th, 75th, and 90th percentiles
    key_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for q in key_quantiles:
        # Find closest quantile in our results
        idx = np.argmin(np.abs(quantiles - q))
        effect = effects[idx]
        print(f"  {q:.0%} quantile: {effect:.3f}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Mean treatment effect: {results.mean_treatment_effect:.3f}")
    print(f"  Median treatment effect: {results.median_treatment_effect:.3f}")
    print(f"  Min treatment effect: {results.min_treatment_effect:.3f}")
    print(f"  Max treatment effect: {results.max_treatment_effect:.3f}")
    
    # Bootstrap confidence intervals
    print(f"\nBootstrap Confidence Intervals (95%):")
    print("-" * 40)
    
    # Compute bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    lower_bound, upper_bound = cic.bootstrap_ci(n_bootstrap=500)  # Reduced for speed
    
    # Display confidence intervals at key quantiles
    for q in key_quantiles:
        idx = np.argmin(np.abs(quantiles - q))
        effect = effects[idx]
        lower_ci = lower_bound[idx]
        upper_ci = upper_bound[idx]
        print(f"  {q:.0%} quantile: {effect:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]")
    
    # Create plot with confidence intervals
    print(f"\nCreating visualization with confidence intervals...")
    results.plot(confidence_intervals=(lower_bound, upper_bound), 
                figsize=(12, 8), 
                save_path='examples/results.png')
    print("Saved plot to 'results.png'")
    
    # Export results to CSV files
    print(f"\nExporting results to CSV files...")
    
    # Get results with confidence intervals
    results_df = results.to_dataframe(confidence_intervals=(lower_bound, upper_bound))
    
    # Add quantile labels for better readability
    results_df['quantile_label'] = [f'q{q:.1%}' for q in results_df['quantile']]
    
    # Reorder columns for better presentation
    results_df = results_df[['quantile', 'quantile_label', 'treatment_effect', 
                            'ci_lower', 'ci_upper', 'counterfactual']]
    
    # Save main results to CSV
    results_df.to_csv('examples/results.csv', index=False)
    print("Saved main results to 'results.csv'")
    
    
if __name__ == "__main__":
    main() 