# pycic

A Python package for estimating Changes-in-Changes (CiC) models, as introduced by Athey and Imbens (2006). This approach generalizes standard difference-in-differences (DiD) methods and allows for nonlinear and heterogeneous effects across the outcome distribution.

## Installation

Install the package using pip:

```bash
pip install pycic
````

## Features

* 2x2 DiD Design: Implements the simplest two-group, two-period setup
* Quantile Treatment Effects: Estimates how treatment effects vary across the distribution of outcomes
* Nonlinear Effects: Captures effects beyond simple average changes
* Confidence Intervals: Bootstrap-based confidence intervals and standard errors
* Visualization: Built-in plotting of estimated effects and distributions
* Integration with Pandas
  
## Quick Start

```python
import numpy as np
import pandas as pd
from pycic import ChangesInChanges

# Generate synthetic data
np.random.seed(1988)
n = 1000

# Control group
y_control_before = np.random.normal(10, 2, n)
y_control_after = y_control_before + np.random.normal(1, 0.5, n)

# Treatment group
y_treat_before = np.random.normal(12, 2, n)
y_treat_after = y_treat_before + np.random.normal(1, 0.5, n) + 3  # Treatment effect

# Combine into DataFrame
data = pd.DataFrame({
    'outcome': np.concatenate([y_control_before, y_control_after, 
                               y_treat_before, y_treat_after]),
    'group':   np.concatenate([[0]*n, [0]*n, 
                               [1]*n, [1]*n]),
    'period':  np.concatenate([[0]*n, [1]*n, 
                               [0]*n, [1]*n])
})

# Fit CiC model
cic = ChangesInChanges()
results = cic.fit(data, outcome='outcome', group='group', period='period')

# View results
print(results.summary())

# Plot treatment effects across quantiles
results.plot()
```

## Examples

See the `examples/heterogeneous_effects.py` file for a complete walkthrough demonstrating:

* Heterogeneous treatment effects across the outcome distribution
* Comparison with traditional difference-in-differences
* Visualization of quantile treatment effects
* Bootstrap confidence intervals

## Background

### From Average Effects to Distributional Insights

Traditional difference-in-differences (DiD) methods focus on average treatment effects and rely on the parallel trends assumption—that treated and control groups would have followed similar trends in the absence of treatment. The Changes-in-Changes (CiC) estimator, proposed by Athey and Imbens (2006), relaxes this by allowing for:

* Heterogeneous effects across the distribution of outcomes
* Nonlinear changes rather than additive shifts
* Distributional treatment effects, not just mean effects

This makes CiC especially useful in applications where policy effects may differ between low and high quantiles (e.g. tax credits, training programs, subsidies).

---

### Notation

Let's establish the following mathematical notation:

* $Y_{gt}$ denote the outcome for group $g \in \{0,1\}$ at time $t \in \{0,1\}$, where 1 stands for what you would expect.
* $F_{gt}$ denote the cumulative distribution function (CDF) of $Y_{gt}$
* $Q_{gt}(u) = F_{gt}^{-1}(u)$ be the corresponding quantile function
* $U \sim \text{Uniform}(0,1)$ represent unobservable rank

---

### Main Result

Athey and Imbens showed that the counterfactual (unobserved) outcome for a treated individual at time 1 (had they not been treated) can be computed as:

$$
\tilde{Y}_{11}^{\text{c.f.}} = Q_{01}\left( F_{00}(Y_{10}) \right)
$$

Here's what this equation does:

* $Y_{10}$ is the treated unit's outcome *before* treatment (group 1, time 0). Find where the treated individual ranked in the control group before treatment
* $F_{00}(Y_{10})$ finds the rank of that pre-treatment outcome in the control group's pre-treatment distribution. See where someone at that same rank ended up in the control group after treatment
* $Q_{01}(\cdot)$ maps that rank to the control group's *post-treatment* distribution (group 0, time 1). Use that as the counterfactual outcome

The treatment effect for each treated unit is then:

$$
\tau_i = Y_{11,i}^{\text{obs}} - \tilde{Y}_{11,i}^{\text{c.f.}}
$$

Aggregating these effects across quantiles yields the quantile treatment effects (QTEs), allowing you to estimate how treatment affects different parts of the outcome distribution. 

---

### Assumptions

The identification strategy relies on the following four assumptions:

1. Model: The outcome in absence of intervention can be expressed as $Y_0=h(U,T)$ for some function $h(\cdot)$. The assumptions below place restrictions either on $h(\cdot)$ or on the joint distribution of $(Y,G,T)$.
2. Monotonicity: The outcome variable $Y$ is strictly increasing in unobserved heterogeneity $U$. This ensures $h(\cdot)$ is invertible in $U$.
3. Time-invariant unobservables: The distribution of unobserved factors $U$ is stable over time within each group, $U\perp T \mid G$. This is indeed the main assumption, requiring difference between units in both groups be stable over time.
4. Common support: The supports of outcome distributions overlap across groups and time periods.

To summarize, the key identification assumption is rank invariance of untreated potential outcomes over time: for each unit, the rank (quantile) of their untreated outcome remains constant across periods. Additionally, it assumes the distribution of untreated potential outcomes evolves over time in the same way for treated and control groups.

---

### Confidence Intervals

Because the CiC estimator involves nonparametric transformations of empirical distributions, analytical standard errors are not readily available. Instead, the package provides support for bootstrap confidence intervals, which work as follows:

1. Re-sample observations with replacement within each group × time cell.
2. Recompute the CiC estimator on each bootstrap sample.
3. Use the empirical distribution of bootstrap estimates to form percentile or bias-corrected intervals.

This ensures robust inference for both point estimates and quantile treatment effects.

## References

* Athey, S., & Imbens, G. W. (2006). *Identification and inference in nonlinear difference-in-differences models*. *Econometrica*, 74(2), 431–497.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

To cite this package in publications, use the following BibTeX entry:

```bibtex
@misc{yasenov2025pycic,
  author       = {Vasco Yasenov},
  title        = {pycic: Python Implementation of the Changes-in-Changes Estimator},
  year         = {2025},
  howpublished = {\url{https://github.com/vyasenov/pycic}},
  note         = {Version 0.1.0}
}
```
