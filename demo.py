#!/usr/bin/env python3
"""
Simple demo of the Changes-in-Changes model.

This script demonstrates the basic usage of the pycic package
with synthetic data and shows the key features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycic import ChangesInChanges

def generate_synthetic_data(n_control=300, n_treatment=300, treatment_effect=2.5, heterogenous_effect=True, random_state=42):
    """Generate synthetic data for demo."""
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
    print("🎯 Changes-in-Changes Model Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    data = generate_synthetic_data(
        n_control=300,
        n_treatment=300,
        treatment_effect=2.5,
        heterogenous_effect=True,  # Include heterogeneous effects
        random_state=42
    )
    
    print(f"Dataset shape: {data.shape}")
    print(f"Sample sizes:")
    print(data.groupby(['group', 'period']).size().unstack())
    
    # Fit the Changes-in-Changes model
    print("\n🔧 Fitting Changes-in-Changes model...")
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
    print("\n📈 Model Results:")
    print(results.summary())
    
    # Compare with traditional DiD
    print("\n🔄 Comparison with Traditional DiD:")
    did_estimate = compute_did_estimate(
        data, 'outcome', 'group', 'period',
        'control', 'treatment', 'before', 'after'
    )
    print(f"Traditional DiD estimate: {did_estimate:.4f}")
    print(f"CiC mean treatment effect: {results.mean_treatment_effect:.4f}")
    print(f"CiC median treatment effect: {results.median_treatment_effect:.4f}")
    
    # Analyze heterogeneity
    treatment_effects = results.treatment_effects
    effect_std = np.std(treatment_effects)
    effect_cv = effect_std / abs(results.mean_treatment_effect)
    
    print(f"\n📊 Heterogeneity Analysis:")
    print(f"Standard deviation of effects: {effect_std:.4f}")
    print(f"Coefficient of variation: {effect_cv:.3f}")
    print(f"Effect range: [{results.min_treatment_effect:.4f}, {results.max_treatment_effect:.4f}]")
    
    if effect_cv > 0.1:
        print("✅ Treatment effects are heterogeneous - CiC provides valuable insights!")
    else:
        print("ℹ️  Treatment effects are relatively homogeneous")
    
    # Create visualization (simplified - only main plot)
    print("\n🎨 Creating visualization...")
    
    plt.figure(figsize=(8, 5))
    plt.plot(results.quantiles, treatment_effects, 'b-', linewidth=2, label='CiC Treatment Effect')
    plt.axhline(y=did_estimate, color='red', linestyle='--', alpha=0.7, 
                label=f'Traditional DiD: {did_estimate:.3f}')
    plt.axhline(y=results.mean_treatment_effect, color='orange', linestyle=':', 
                alpha=0.7, label=f'CiC Mean: {results.mean_treatment_effect:.3f}')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Quantile')
    plt.ylabel('Treatment Effect')
    plt.title('Treatment Effects Across Quantiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cic_demo_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved results to 'cic_demo_results.png'")
    
    # Export results
    print("\n💾 Exporting results...")
    results_df = results.to_dataframe()
    results_df.to_csv('cic_demo_results.csv', index=False)
    print("✅ Saved results to 'cic_demo_results.csv'")
    
    # Key insights
    print("\n🔍 Key Insights:")
    print("-" * 30)
    print(f"• Traditional DiD assumes constant effect: {did_estimate:.4f}")
    print(f"• CiC reveals varying effects: {results.min_treatment_effect:.4f} to {results.max_treatment_effect:.4f}")
    print(f"• Effects vary by {effect_cv:.1%} across the distribution")
    print(f"• Largest effects at {results.quantiles[np.argmax(treatment_effects)]:.1%} quantile")
    print(f"• Smallest effects at {results.quantiles[np.argmin(treatment_effects)]:.1%} quantile")
    
    print("\n🎉 Demo completed successfully!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 