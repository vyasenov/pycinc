"""
Results class for Changes-in-Changes model.

This module contains the CiCResults class that stores and displays
the results from fitting the Changes-in-Changes model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from scipy import stats


class CiCResults:
    """
    Results object for Changes-in-Changes model.
    
    This class stores the results from fitting a Changes-in-Changes model
    and provides methods for summarizing and visualizing the results.
    
    Parameters
    ----------
    quantiles : np.ndarray
        Quantiles used in the estimation.
    
    treatment_effects : np.ndarray
        Estimated treatment effects at each quantile.
    
    counterfactual : np.ndarray
        Counterfactual outcomes for the treatment group.
    
    observed_treatment : np.ndarray
        Observed treatment group outcomes after treatment.
    
    observed_control_before : np.ndarray
        Observed control group outcomes before treatment.
    
    observed_control_after : np.ndarray
        Observed control group outcomes after treatment.
    
    observed_treatment_before : np.ndarray
        Observed treatment group outcomes before treatment.
    
    observed_treatment_after : np.ndarray
        Observed treatment group outcomes after treatment.
    
    data_info : dict
        Information about the data structure and variable names.
    """
    
    def __init__(self,
                 quantiles: np.ndarray,
                 treatment_effects: np.ndarray,
                 counterfactual: np.ndarray,
                 observed_treatment: np.ndarray,
                 observed_control_before: np.ndarray,
                 observed_control_after: np.ndarray,
                 observed_treatment_before: np.ndarray,
                 observed_treatment_after: np.ndarray,
                 data_info: Dict[str, Any]):
        
        self.quantiles = quantiles
        self.treatment_effects = treatment_effects
        self.counterfactual = counterfactual
        self.observed_treatment = observed_treatment
        self.observed_control_before = observed_control_before
        self.observed_control_after = observed_control_after
        self.observed_treatment_before = observed_treatment_before
        self.observed_treatment_after = observed_treatment_after
        self.data_info = data_info
        
        # Compute summary statistics
        self._compute_summary_stats()
    
    def _compute_summary_stats(self):
        """Compute summary statistics for the results."""
        # Mean treatment effect
        self.mean_treatment_effect = np.mean(self.treatment_effects)
        
        # Median treatment effect
        self.median_treatment_effect = np.median(self.treatment_effects)
        
        # Treatment effects at key quantiles
        key_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.key_quantile_effects = {}
        for q in key_quantiles:
            idx = np.argmin(np.abs(self.quantiles - q))
            self.key_quantile_effects[f'q{q:.0%}'] = self.treatment_effects[idx]
        
        # Standard deviation of treatment effects
        self.std_treatment_effect = np.std(self.treatment_effects)
        
        # Range of treatment effects
        self.min_treatment_effect = np.min(self.treatment_effects)
        self.max_treatment_effect = np.max(self.treatment_effects)
    
    def summary(self) -> str:
        """
        Generate a summary of the results.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("Changes-in-Changes Model Results")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Data information
        summary_lines.append("Data Information:")
        summary_lines.append(f"  Outcome variable: {self.data_info['outcome']}")
        summary_lines.append(f"  Group variable: {self.data_info['group']}")
        summary_lines.append(f"  Period variable: {self.data_info['period']}")
        summary_lines.append(f"  Control group: {self.data_info['control_group']}")
        summary_lines.append(f"  Treatment group: {self.data_info['treatment_group']}")
        summary_lines.append(f"  Before period: {self.data_info['before_period']}")
        summary_lines.append(f"  After period: {self.data_info['after_period']}")
        summary_lines.append("")
        
        # Sample sizes
        summary_lines.append("Sample Sizes:")
        summary_lines.append(f"  Control before: {len(self.observed_control_before)}")
        summary_lines.append(f"  Control after: {len(self.observed_control_after)}")
        summary_lines.append(f"  Treatment before: {len(self.observed_treatment_before)}")
        summary_lines.append(f"  Treatment after: {len(self.observed_treatment_after)}")
        summary_lines.append("")
        
        # Treatment effect summary
        summary_lines.append("Treatment Effect Summary:")
        summary_lines.append(f"  Mean treatment effect: {self.mean_treatment_effect:.4f}")
        summary_lines.append(f"  Median treatment effect: {self.median_treatment_effect:.4f}")
        summary_lines.append(f"  Standard deviation: {self.std_treatment_effect:.4f}")
        summary_lines.append(f"  Minimum effect: {self.min_treatment_effect:.4f}")
        summary_lines.append(f"  Maximum effect: {self.max_treatment_effect:.4f}")
        summary_lines.append("")
        
        # Treatment effects at key quantiles
        summary_lines.append("Treatment Effects at Key Quantiles:")
        for quantile_name, effect in self.key_quantile_effects.items():
            summary_lines.append(f"  {quantile_name}: {effect:.4f}")
        summary_lines.append("")
        
        # Distributional analysis
        summary_lines.append("Distributional Analysis:")
        positive_effects = np.sum(self.treatment_effects > 0)
        negative_effects = np.sum(self.treatment_effects < 0)
        zero_effects = np.sum(self.treatment_effects == 0)
        
        summary_lines.append(f"  Positive effects: {positive_effects}/{len(self.treatment_effects)} ({positive_effects/len(self.treatment_effects):.1%})")
        summary_lines.append(f"  Negative effects: {negative_effects}/{len(self.treatment_effects)} ({negative_effects/len(self.treatment_effects):.1%})")
        summary_lines.append(f"  Zero effects: {zero_effects}/{len(self.treatment_effects)} ({zero_effects/len(self.treatment_effects):.1%})")
        summary_lines.append("")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
    
    def plot(self, 
             figsize: tuple = (8, 5),
             style: str = 'seaborn-v0_8',
             save_path: str = None) -> None:
        """
        Plot treatment effects across quantiles (minimal figure).
        """
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.quantiles, self.treatment_effects, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Treatment Effects Across Quantiles')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_treatment_effects(self, 
                             confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             style: str = 'seaborn-v0_8',
                             save_path: Optional[str] = None) -> None:
        """
        Plot treatment effects with optional confidence intervals.
        
        Parameters
        ----------
        confidence_intervals : tuple, optional
            (lower_bound, upper_bound) arrays for confidence intervals.
        
        figsize : tuple, default=(10, 6)
            Figure size (width, height).
        
        style : str, default='seaborn-v0_8'
            Matplotlib style to use.
        
        save_path : str, optional
            Path to save the plot. If None, plot is displayed.
        """
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot treatment effects
        ax.plot(self.quantiles, self.treatment_effects, 'b-', linewidth=2, 
               label='Treatment Effect')
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower_bound, upper_bound = confidence_intervals
            ax.fill_between(self.quantiles, lower_bound, upper_bound, 
                          alpha=0.3, color='blue', label='95% CI')
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add mean and median lines
        ax.axhline(y=self.mean_treatment_effect, color='red', linestyle=':', 
                  alpha=0.7, label=f'Mean: {self.mean_treatment_effect:.3f}')
        ax.axhline(y=self.median_treatment_effect, color='orange', linestyle=':', 
                  alpha=0.7, label=f'Median: {self.median_treatment_effect:.3f}')
        
        # Add key quantile markers
        for q_name, effect in self.key_quantile_effects.items():
            q_val = float(q_name[1:-1]) / 100
            ax.scatter(q_val, effect, color='red', s=50, zorder=5)
            ax.annotate(q_name, (q_val, effect), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Changes-in-Changes Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing quantiles and treatment effects.
        """
        df = pd.DataFrame({
            'quantile': self.quantiles,
            'treatment_effect': self.treatment_effects,
            'counterfactual': self.counterfactual
        })
        
        return df
    
    def __repr__(self) -> str:
        """String representation of the results."""
        return f"CiCResults(mean_effect={self.mean_treatment_effect:.4f}, " \
               f"median_effect={self.median_treatment_effect:.4f}, " \
               f"n_quantiles={len(self.quantiles)})" 