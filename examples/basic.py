"""
Basic Changes-in-Changes (CiC) estimation example.

This example demonstrates the core functionality of the CiC model
with minimal setup and output.
"""

# Add parent directory to path to import pycic
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pycic import ChangesInChanges

# Set random seed for reproducibility
np.random.seed(1988)

#################
# Generate data #
#################

n_control=500
n_treatment=500
treatment_effect=2.0

# Control group
control_before = np.random.normal(10, 2, n_control)
control_after = control_before + np.random.normal(1, 0.5, n_control)

# Treatment group
treatment_before = np.random.normal(12, 2, n_treatment)
treatment_after = treatment_before + np.random.normal(1, 0.5, n_treatment) + treatment_effect

data = pd.DataFrame({
    'outcome': np.concatenate([control_before, control_after, treatment_before, treatment_after]),
    'group': np.concatenate([[0]*n_control, [0]*n_control, [1]*n_treatment, [1]*n_treatment]),
    'period': np.concatenate([[0]*n_control, [1]*n_control, [0]*n_treatment, [1]*n_treatment])
})

#################
# Fit CiC model #
#################

cic = ChangesInChanges(n_quantiles=50)
results = cic.fit(data, outcome='outcome', group='group', period='period')

#################
# Print results #
#################

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
