"""
Core implementation of the Changes-in-Changes model.

This module contains the main ChangesInChanges class that implements
the methodology from Athey and Imbens (2006).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Dict, Any
import warnings

from .results import CiCResults


class ChangesInChanges:
    """
    Changes-in-Changes model implementation.
    
    This class implements the Changes-in-Changes (CiC) model introduced by
    Athey and Imbens (2006) for estimating treatment effects in a 2x2 design
    (treatment/control groups, before/after periods).
    
    The model relaxes the parallel trends assumption of traditional DiD by
    allowing for nonlinear treatment effects and heterogeneous effects across
    the outcome distribution.
    
    Parameters
    ----------
    n_quantiles : int, default=50
        Number of quantiles to use for estimation. Higher values provide
        more granular estimates but may be computationally intensive.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    results : CiCResults or None
        Fitted results object containing estimates and diagnostics.
        None before fitting.
    
    data : pd.DataFrame or None
        Processed data used for estimation. None before fitting.
    
    References
    ----------
    Athey, S., & Imbens, G. W. (2006). Identification and inference in 
    nonlinear difference-in-differences models. Econometrica, 74(2), 431-497.
    """
    
    def __init__(self, n_quantiles: int = 50, random_state: Optional[int] = None):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.results = None
        self.data = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            data: pd.DataFrame,
            outcome: str,
            group: str,
            period: str) -> CiCResults:
        """
        Fit the Changes-in-Changes model.
        
        Assumes: group=0 for control, group=1 for treatment
                 period=0 for before, period=1 for after
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome, group, and period variables.
        
        outcome : str
            Name of the outcome variable column.
        
        group : str
            Name of the group variable column (0=control, 1=treatment).
        
        period : str
            Name of the period variable column (0=before, 1=after).
        
        Returns
        -------
        CiCResults
            Results object containing estimates and diagnostics.
        
        Raises
        ------
        ValueError
            If data validation fails or required columns are missing.
        """
        # Validate input data
        self._validate_data(data, outcome, group, period)
        
        # Store processed data
        self.data = data.copy()
        
        # Extract the four groups using numeric values
        y_cb = data[(data[group] == 0) & (data[period] == 0)][outcome].values
        y_ca = data[(data[group] == 0) & (data[period] == 1)][outcome].values
        y_tb = data[(data[group] == 1) & (data[period] == 0)][outcome].values
        y_ta = data[(data[group] == 1) & (data[period] == 1)][outcome].values
        
        # Compute quantiles
        quantiles = np.linspace(0, 1, self.n_quantiles)
        
        # Estimate counterfactual distribution
        counterfactual = self._estimate_counterfactual(y_cb, y_ca, y_tb, quantiles)
        
        # Compute treatment effects
        treatment_effects = self._compute_treatment_effects(y_ta, counterfactual, quantiles)
        
        # Store results
        self.results = CiCResults(
            quantiles=quantiles,
            treatment_effects=treatment_effects,
            counterfactual=counterfactual,
            observed_treatment=y_ta,
            observed_control_before=y_cb,
            observed_control_after=y_ca,
            observed_treatment_before=y_tb,
            observed_treatment_after=y_ta,
            data_info={
                'outcome': outcome,
                'group': group,
                'period': period
            }
        )
        
        return self.results
    
    def _estimate_counterfactual(self, 
                                y_cb: np.ndarray,
                                y_ca: np.ndarray, 
                                y_tb: np.ndarray,
                                quantiles: np.ndarray) -> np.ndarray:
        """
        Estimate the counterfactual distribution for the treatment group.
        
        This implements the key insight of CiC: using the change in the
        control group's distribution to predict what the treatment group's
        distribution would have been in the absence of treatment.
        
        Parameters
        ----------
        y_cb : np.ndarray
            Control group outcomes before treatment.
        
        y_ca : np.ndarray
            Control group outcomes after treatment.
        
        y_tb : np.ndarray
            Treatment group outcomes before treatment.
        
        quantiles : np.ndarray
            Quantiles at which to estimate the counterfactual.
        
        Returns
        -------
        np.ndarray
            Counterfactual outcomes at each quantile.
        """
        # Compute quantiles of treatment group before
        q_tb = np.percentile(y_tb, quantiles * 100)
        
        # For each quantile of treatment group before, find corresponding quantile in control group after
        counterfactual = []
        
        for q_val in q_tb:
            # Find rank in control group before
            rank = stats.percentileofscore(y_cb, q_val, kind='weak') / 100
            
            # Ensure rank is within valid bounds
            rank = np.clip(rank, 0.001, 0.999)  # Avoid extreme quantiles
            
            # Map to control group after
            cf_val = np.percentile(y_ca, rank * 100)
            counterfactual.append(cf_val)
        
        return np.array(counterfactual)
        
    
    def _compute_treatment_effects(self,
                                 y_ta: np.ndarray,
                                 counterfactual: np.ndarray,
                                 quantiles: np.ndarray) -> np.ndarray:
        """
        Compute treatment effects at each quantile.
        
        Parameters
        ----------
        y_ta : np.ndarray
            Observed treatment group outcomes after treatment.
        
        counterfactual : np.ndarray
            Counterfactual outcomes for the treatment group.
        
        quantiles : np.ndarray
            Quantiles at which to compute treatment effects.
        
        Returns
        -------
        np.ndarray
            Treatment effects at each quantile.
        """
        # Compute quantiles of observed treatment outcomes
        q_ta = np.percentile(y_ta, quantiles * 100)
        
        # Compute quantiles of counterfactual outcomes
        q_counterfactual = np.percentile(counterfactual, quantiles * 100)
        
        # Treatment effect is the difference between observed and counterfactual quantiles
        treatment_effects = q_ta - q_counterfactual
        
        return treatment_effects
    
    def bootstrap_ci(self, 
                n_bootstrap: int = 1000,
                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for treatment effects.
        
        Parameters
        ----------
        n_bootstrap : int, default=1000
            Number of bootstrap samples.
        
        confidence_level : float, default=0.95
            Confidence level for the intervals.
        
        Returns
        -------
        tuple
            (lower_bound, upper_bound) arrays for each quantile.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before computing confidence intervals.")
        
        if self.data is None:
            raise ValueError("Data not available for bootstrap.")
        
        # Extract data for bootstrap
        outcome = self.results.data_info['outcome']
        group = self.results.data_info['group']
        period = self.results.data_info['period']
        
        # Bootstrap samples
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement from each group separately
            bootstrap_data = self._stratified_bootstrap_sample(self.data, group, period)
            
            # Fit model on bootstrap sample
            bootstrap_cic = ChangesInChanges(n_quantiles=self.n_quantiles)
            bootstrap_results = bootstrap_cic.fit(bootstrap_data, outcome, group, period)
            
            bootstrap_effects.append(bootstrap_results.treatment_effects)
        
        bootstrap_effects = np.array(bootstrap_effects)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_effects, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_effects, upper_percentile, axis=0)
        
        return lower_bound, upper_bound

    def _stratified_bootstrap_sample(self, data, group, period):
        """
        Create bootstrap sample maintaining group structure.
        """
        bootstrap_parts = []
        
        # Sample from each of the four groups separately
        for g in [0, 1]:
            for p in [0, 1]:
                group_data = data[(data[group] == g) & (data[period] == p)]
                n = len(group_data)
                if n > 0:
                    indices = np.random.choice(n, size=n, replace=True)
                    bootstrap_parts.append(group_data.iloc[indices])
        
        return pd.concat(bootstrap_parts, ignore_index=True)

    def _validate_data(self, data, outcome, group, period):
        required_columns = [outcome, group, period]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        missing_data = data[required_columns].isnull().sum()
        if missing_data.sum() > 0:
            raise ValueError(f"Missing values found in columns: {missing_data[missing_data > 0].to_dict()}")
        if not pd.api.types.is_numeric_dtype(data[outcome]):
            raise ValueError(f"Outcome variable '{outcome}' must be numeric")
        unique_groups = data[group].unique()
        unique_periods = data[period].unique()
        if len(unique_groups) != 2 or not set(unique_groups) == {0, 1}:
            raise ValueError("Group column must contain only 0 and 1")
        if len(unique_periods) != 2 or not set(unique_periods) == {0, 1}:
            raise ValueError("Period column must contain only 0 and 1")

    @staticmethod
    def _compute_quantiles(individual_effects, quantiles):
        return np.percentile(individual_effects, quantiles * 100)

    @staticmethod
    def _bootstrap_sample(data, group, period):
        """
        Bootstrap sample while maintaining the same group sizes as the original data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Original data.
        group : str
            Name of the group column.
        period : str
            Name of the period column.
        
        Returns
        -------
        pd.DataFrame
            Bootstrap sample with same group sizes as original data.
        """
        # Get original group sizes
        original_sizes = data.groupby([group, period]).size()
        
        # Sample within each group-period combination
        bootstrap_samples = []
        
        for (g, p), size in original_sizes.items():
            # Get data for this group-period combination
            group_data = data[(data[group] == g) & (data[period] == p)]
            
            # Sample with replacement within this group
            indices = np.random.choice(len(group_data), size=size, replace=True)
            bootstrap_group = group_data.iloc[indices].reset_index(drop=True)
            
            bootstrap_samples.append(bootstrap_group)
        
        # Combine all samples
        bootstrap_data = pd.concat(bootstrap_samples, ignore_index=True)
        
        return bootstrap_data 