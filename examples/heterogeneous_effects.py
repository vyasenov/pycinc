"""
Example demonstrating heterogeneous treatment effects.

This example shows how the Changes-in-Changes model can capture
heterogeneous treatment effects that vary across the outcome distribution,
unlike traditional DiD which assumes constant effects.
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
    print("Changes-in-Changes Model: Heterogeneous Effects Example")
    print("=" * 60)
    
    # Generate data with heterogeneous treatment effects
    print("\n1. Generating data with heterogeneous treatment effects...")
    data = generate_synthetic_data(
        n_control=500,
        n_treatment=500,
        treatment_effect=2.0,
        heterogenous_effect=True,  # This creates heterogeneous effects
        random_state=42
    )
    
    print(f"Data shape: {data.shape}")
    
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
    print(f"CiC median treatment effect: {results.median_treatment_effect:.4f}")
    print(f"CiC treatment effect range: [{results.min_treatment_effect:.4f}, {results.max_treatment_effect:.4f}]")
    
    # Analyze heterogeneity
    print("\n6. Heterogeneity analysis:")
    treatment_effects = results.treatment_effects
    quantiles = results.quantiles
    
    # Find where effects are largest and smallest
    max_idx = np.argmax(treatment_effects)
    min_idx = np.argmin(treatment_effects)
    
    print(f"Largest treatment effect: {treatment_effects[max_idx]:.4f} at quantile {quantiles[max_idx]:.3f}")
    print(f"Smallest treatment effect: {treatment_effects[min_idx]:.4f} at quantile {quantiles[min_idx]:.3f}")
    print(f"Range of effects: {treatment_effects[max_idx] - treatment_effects[min_idx]:.4f}")
    
    # Check if effects are significantly heterogeneous
    effect_std = np.std(treatment_effects)
    effect_mean = np.mean(treatment_effects)
    cv = effect_std / abs(effect_mean) if effect_mean != 0 else 0
    print(f"Coefficient of variation: {cv:.3f}")
    
    if cv > 0.1:
        print("Treatment effects appear to be heterogeneous (CV > 0.1)")
    else:
        print("Treatment effects appear to be relatively homogeneous")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    # Plot comprehensive results
    results.plot(figsize=(12, 8))
    plt.savefig('heterogeneous_effects_results.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive results plot to 'heterogeneous_effects_results.png'")
    
    # Create detailed treatment effects plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Treatment effects across quantiles
    ax1.plot(quantiles, treatment_effects, 'b-', linewidth=2, label='Treatment Effect')
    ax1.axhline(y=did_estimate, color='red', linestyle='--', alpha=0.7, 
                label=f'Traditional DiD: {did_estimate:.3f}')
    ax1.axhline(y=results.mean_treatment_effect, color='orange', linestyle=':', 
                alpha=0.7, label=f'CiC Mean: {results.mean_treatment_effect:.3f}')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Quantile')
    ax1.set_ylabel('Treatment Effect')
    ax1.set_title('Heterogeneous Treatment Effects')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of treatment effects
    ax2.hist(treatment_effects, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(did_estimate, color='red', linestyle='--', linewidth=2, 
                label=f'Traditional DiD: {did_estimate:.3f}')
    ax2.axvline(results.mean_treatment_effect, color='orange', linestyle='--', linewidth=2,
                label=f'CiC Mean: {results.mean_treatment_effect:.3f}')
    ax2.axvline(results.median_treatment_effect, color='green', linestyle='--', linewidth=2,
                label=f'CiC Median: {results.median_treatment_effect:.3f}')
    ax2.set_xlabel('Treatment Effect')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Treatment Effects')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heterogeneous_effects_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved heterogeneity analysis plot to 'heterogeneous_effects_analysis.png'")
    
    # Export detailed results
    print("\n8. Exporting detailed results...")
    results_df = results.to_dataframe()
    results_df['quantile_label'] = [f'q{q:.1%}' for q in results_df['quantile']]
    results_df = results_df[['quantile', 'quantile_label', 'treatment_effect', 'counterfactual']]
    results_df.to_csv('heterogeneous_effects_results.csv', index=False)
    print("Saved detailed results to 'heterogeneous_effects_results.csv'")
    
    # Summary of key findings
    print("\n9. Key findings:")
    print("-" * 40)
    print(f"• Traditional DiD assumes constant treatment effect: {did_estimate:.4f}")
    print(f"• CiC reveals heterogeneous effects ranging from {results.min_treatment_effect:.4f} to {results.max_treatment_effect:.4f}")
    print(f"• Effects vary by {cv:.1%} (coefficient of variation)")
    print(f"• Largest effects occur at the {quantiles[max_idx]:.1%} quantile")
    print(f"• Smallest effects occur at the {quantiles[min_idx]:.1%} quantile")
    
    if cv > 0.1:
        print("• Conclusion: Treatment effects are heterogeneous - CiC provides valuable insights")
    else:
        print("• Conclusion: Treatment effects are relatively homogeneous - DiD may be adequate")
    
    print("\n" + "=" * 60)
    print("Heterogeneous effects example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 