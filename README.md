# pycic

A Python package for estimating the Changes-in-Changes (CiC) model introduced by Athey and Imbens (2006) in "Identification and Inference in Nonlinear Difference-in-Differences Models" published in Econometrica.

## Overview

The Changes-in-Changes model is a generalization of the standard difference-in-differences approach that allows for nonlinear treatment effects and heterogeneous treatment effects across different quantiles of the outcome distribution. Unlike traditional DiD, CiC doesn't require the parallel trends assumption and can handle cases where the treatment effect varies across the distribution of outcomes.

## Key Features

- **2x2 Design**: Focuses on the simplest case with treatment and control groups, before and after periods
- **Quantile Treatment Effects**: Estimates treatment effects at different quantiles of the outcome distribution
- **Nonlinear Effects**: Handles cases where treatment effects are not additive
- **Robust Estimation**: Provides confidence intervals and standard errors
- **Visualization**: Built-in plotting functions for results interpretation

## Installation

```bash
pip install pycic
```

## Quick Start

```python
import numpy as np
import pandas as pd
from pycic import ChangesInChanges

# Generate sample data
np.random.seed(42)
n = 1000

# Control group: before and after
y_control_before = np.random.normal(10, 2, n)
y_control_after = y_control_before + np.random.normal(1, 0.5, n)

# Treatment group: before and after
y_treat_before = np.random.normal(12, 2, n)
y_treat_after = y_treat_before + np.random.normal(1, 0.5, n) + 3  # Treatment effect

# Create data frame
data = pd.DataFrame({
    'outcome': np.concatenate([y_control_before, y_control_after, 
                              y_treat_before, y_treat_after]),
    'group': np.concatenate([['control']*n, ['control']*n, 
                            ['treatment']*n, ['treatment']*n]),
    'period': np.concatenate([['before']*n, ['after']*n, 
                             ['before']*n, ['after']*n])
})

# Initialize and fit the model
cic = ChangesInChanges()
results = cic.fit(data, outcome='outcome', group='group', period='period',
                  control_group='control', treatment_group='treatment',
                  before_period='before', after_period='after')

# Get results
print(results.summary())

# Plot results
results.plot()
```

## Theoretical Background

The Changes-in-Changes model relaxes the parallel trends assumption of traditional DiD by allowing for:

1. **Nonlinear treatment effects**: The treatment effect can vary across the distribution
2. **Heterogeneous effects**: Different quantiles may have different treatment effects
3. **Distributional changes**: The model can capture changes in the entire distribution, not just the mean

The key insight is that under certain assumptions, we can identify the counterfactual distribution of outcomes for the treatment group in the absence of treatment by using the change in the control group's distribution as a guide.

## Assumptions

1. **Monotonicity**: The outcome variable is strictly increasing in unobservables
2. **Stable distribution**: The distribution of unobservables is stable over time within groups
3. **Common support**: The support of the outcome variable is the same across groups and periods

## API Reference

### ChangesInChanges

The main class for estimating the Changes-in-Changes model.

#### Methods

- `fit(data, outcome, group, period, ...)`: Fit the CiC model
- `predict_quantiles(quantiles)`: Predict treatment effects at specific quantiles
- `bootstrap_ci(n_bootstrap, confidence_level)`: Compute bootstrap confidence intervals

#### Attributes

- `results`: Fitted results object containing estimates and diagnostics
- `data`: Processed data used for estimation

## Examples

See the `examples/` directory for detailed examples including:

- Basic 2x2 estimation
- Bootstrap confidence intervals
- Visualization of results
- Comparison with traditional DiD

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Athey, S., & Imbens, G. W. (2006). Identification and inference in nonlinear difference-in-differences models. Econometrica, 74(2), 431-497.
