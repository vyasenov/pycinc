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
    n_quantiles : int, default=100
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
    
    def __init__(self, n_quantiles: int = 100, random_state: Optional[int] = None):
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
            period: str,
            control_group: str = 'control',
            treatment_group: str = 'treatment',
            before_period: str = 'before',
            after_period: str = 'after') -> CiCResults:
        """
        Fit the Changes-in-Changes model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome, group, and period variables.
        
        outcome : str
            Name of the outcome variable column.
        
        group : str
            Name of the group variable column (treatment/control).
        
        period : str
            Name of the period variable column (before/after).
        
        control_group : str, default='control'
            Value indicating the control group.
        
        treatment_group : str, default='treatment'
            Value indicating the treatment group.
        
        before_period : str, default='before'
            Value indicating the before period.
        
        after_period : str, default='after'
            Value indicating the after period.
        
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
        self._validate_data(data, outcome, group, period, 
                     control_group, treatment_group, 
                     before_period, after_period)
        
        # Store processed data
        self.data = data.copy()
        
        # Extract the four groups
        y_cb = data[(data[group] == control_group) & 
                   (data[period] == before_period)][outcome].values
        y_ca = data[(data[group] == control_group) & 
                   (data[period] == after_period)][outcome].values
        y_tb = data[(data[group] == treatment_group) & 
                   (data[period] == before_period)][outcome].values
        y_ta = data[(data[group] == treatment_group) & 
                   (data[period] == after_period)][outcome].values
        
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
                'period': period,
                'control_group': control_group,
                'treatment_group': treatment_group,
                'before_period': before_period,
                'after_period': after_period
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
            Counterfactual outcomes for the treatment group.
        """
        # Compute quantiles for each group
        q_cb = self._compute_quantiles(y_cb, quantiles)
        q_ca = self._compute_quantiles(y_ca, quantiles)
        q_tb = self._compute_quantiles(y_tb, quantiles)
        
        # Estimate the transformation function from control group
        # This maps from before to after in the control group
        transformation = q_ca - q_cb
        
        # Apply this transformation to the treatment group's before distribution
        counterfactual = q_tb + transformation
        
        return counterfactual
    
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
        q_ta = self._compute_quantiles(y_ta, quantiles)
        
        # Treatment effect is the difference between observed and counterfactual
        treatment_effects = q_ta - counterfactual
        
        return treatment_effects
    
    def predict_quantiles(self, quantiles: np.ndarray) -> np.ndarray:
        """
        Predict treatment effects at specific quantiles.
        
        Parameters
        ----------
        quantiles : np.ndarray
            Quantiles at which to predict treatment effects.
            Must be between 0 and 1.
        
        Returns
        -------
        np.ndarray
            Predicted treatment effects at the specified quantiles.
        
        Raises
        ------
        ValueError
            If model hasn't been fitted yet.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            raise ValueError("Quantiles must be between 0 and 1.")
        
        # Interpolate from the fitted quantiles
        treatment_effects = np.interp(quantiles, 
                                    self.results.quantiles, 
                                    self.results.treatment_effects)
        
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
        
        Raises
        ------
        ValueError
            If model hasn't been fitted yet.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before computing confidence intervals.")
        
        if self.data is None:
            raise ValueError("Data not available for bootstrap.")
        
        # Extract data for bootstrap
        outcome = self.results.data_info['outcome']
        group = self.results.data_info['group']
        period = self.results.data_info['period']
        control_group = self.results.data_info['control_group']
        treatment_group = self.results.data_info['treatment_group']
        before_period = self.results.data_info['before_period']
        after_period = self.results.data_info['after_period']
        
        # Bootstrap samples
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_data = self._bootstrap_sample(self.data)
            
            # Fit model on bootstrap sample
            bootstrap_cic = ChangesInChanges(n_quantiles=self.n_quantiles)
            bootstrap_results = bootstrap_cic.fit(
                bootstrap_data, outcome, group, period,
                control_group, treatment_group, before_period, after_period
            )
            
            bootstrap_effects.append(bootstrap_results.treatment_effects)
        
        bootstrap_effects = np.array(bootstrap_effects)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_effects, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_effects, upper_percentile, axis=0)
        
        return lower_bound, upper_bound 

    def _validate_data(self, data, outcome, group, period, control_group, treatment_group, before_period, after_period):
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
        if control_group not in unique_groups:
            raise ValueError(f"Control group '{control_group}' not found in group column")
        if treatment_group not in unique_groups:
            raise ValueError(f"Treatment group '{treatment_group}' not found in group column")
        if before_period not in unique_periods:
            raise ValueError(f"Before period '{before_period}' not found in period column")
        if after_period not in unique_periods:
            raise ValueError(f"After period '{after_period}' not found in period column")

    @staticmethod
    def _compute_quantiles(data, quantiles):
        if len(data) == 0:
            raise ValueError("Cannot compute quantiles of empty array")
        return np.percentile(data, quantiles * 100)

    @staticmethod
    def _bootstrap_sample(data):
        n = len(data)
        indices = np.random.choice(n, size=n, replace=True)
        return data.iloc[indices].reset_index(drop=True) 