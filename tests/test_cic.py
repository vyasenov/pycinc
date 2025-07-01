"""
Tests for the Changes-in-Changes model.

This module contains comprehensive tests for the CiC implementation,
including unit tests, integration tests, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from pycic import ChangesInChanges


class TestChangesInChanges(unittest.TestCase):
    """Test cases for the ChangesInChanges class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = self._generate_synthetic_data(
            n_control=100,
            n_treatment=100,
            treatment_effect=2.0,
            heterogenous_effect=False,
            random_state=42
        )
        
        self.cic = ChangesInChanges(n_quantiles=50, random_state=42)
    
    def _generate_synthetic_data(self, n_control=100, n_treatment=100, treatment_effect=2.0, heterogenous_effect=False, random_state=42):
        """Generate synthetic data for testing."""
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
    
    def _compute_did_estimate(self, data, outcome='outcome', group='group', period='period', 
                            control_group='control', treatment_group='treatment', 
                            before_period='before', after_period='after'):
        """Compute traditional DiD estimate."""
        control_before_mean = data[(data[group] == control_group) & (data[period] == before_period)][outcome].mean()
        control_after_mean = data[(data[group] == control_group) & (data[period] == after_period)][outcome].mean()
        treatment_before_mean = data[(data[group] == treatment_group) & (data[period] == before_period)][outcome].mean()
        treatment_after_mean = data[(data[group] == treatment_group) & (data[period] == after_period)][outcome].mean()
        return (treatment_after_mean - treatment_before_mean) - (control_after_mean - control_before_mean)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.cic.n_quantiles, 50)
        self.assertEqual(self.cic.random_state, 42)
        self.assertIsNone(self.cic.results)
        self.assertIsNone(self.cic.data)
    
    def test_fit_basic(self):
        """Test basic model fitting."""
        results = self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Check that results are not None
        self.assertIsNotNone(results)
        self.assertIsNotNone(self.cic.results)
        self.assertIsNotNone(self.cic.data)
        
        # Check results structure
        self.assertEqual(len(results.quantiles), 50)
        self.assertEqual(len(results.treatment_effects), 50)
        self.assertEqual(len(results.counterfactual), 50)
        
        # Check that quantiles are in [0, 1]
        self.assertTrue(np.all(results.quantiles >= 0))
        self.assertTrue(np.all(results.quantiles <= 1))
        
        # Check that treatment effects are finite
        self.assertTrue(np.all(np.isfinite(results.treatment_effects)))
    
    def test_fit_with_different_quantiles(self):
        """Test fitting with different numbers of quantiles."""
        for n_quantiles in [10, 25, 100]:
            cic = ChangesInChanges(n_quantiles=n_quantiles, random_state=42)
            results = cic.fit(
                self.data,
                outcome='outcome',
                group='group',
                period='period',
                control_group='control',
                treatment_group='treatment',
                before_period='before',
                after_period='after'
            )
            
            self.assertEqual(len(results.quantiles), n_quantiles)
            self.assertEqual(len(results.treatment_effects), n_quantiles)
    
    def test_predict_quantiles(self):
        """Test prediction at specific quantiles."""
        # Fit the model first
        self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Test prediction at specific quantiles
        test_quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        predictions = self.cic.predict_quantiles(test_quantiles)
        
        self.assertEqual(len(predictions), len(test_quantiles))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
        # Test that predictions are reasonable
        # (should be close to the fitted values at those quantiles)
        fitted_effects = self.cic.results.treatment_effects
        fitted_quantiles = self.cic.results.quantiles
        
        for i, q in enumerate(test_quantiles):
            # Find closest fitted quantile
            idx = np.argmin(np.abs(fitted_quantiles - q))
            expected = fitted_effects[idx]
            actual = predictions[i]
            
            # Should be close (allowing for interpolation differences)
            self.assertAlmostEqual(actual, expected, places=3)
    
    def test_predict_quantiles_without_fit(self):
        """Test that prediction fails without fitting."""
        with self.assertRaises(ValueError):
            self.cic.predict_quantiles(np.array([0.5]))
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        # Fit the model first
        self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Test bootstrap with small number of samples
        lower, upper = self.cic.bootstrap_ci(n_bootstrap=50, confidence_level=0.95)
        
        self.assertEqual(len(lower), len(self.cic.results.quantiles))
        self.assertEqual(len(upper), len(self.cic.results.quantiles))
        
        # Check that lower bound is below upper bound
        self.assertTrue(np.all(lower <= upper))
        
        # Check that treatment effects are within bounds
        treatment_effects = self.cic.results.treatment_effects
        self.assertTrue(np.all(treatment_effects >= lower))
        self.assertTrue(np.all(treatment_effects <= upper))
    
    def test_bootstrap_ci_without_fit(self):
        """Test that bootstrap fails without fitting."""
        with self.assertRaises(ValueError):
            self.cic.bootstrap_ci()
    
    def test_heterogeneous_effects(self):
        """Test with heterogeneous treatment effects."""
        # Generate data with heterogeneous effects
        data_het = self._generate_synthetic_data(
            n_control=100,
            n_treatment=100,
            treatment_effect=2.0,
            heterogenous_effect=True,
            random_state=42
        )
        
        cic_het = ChangesInChanges(n_quantiles=50, random_state=42)
        results_het = cic_het.fit(
            data_het,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Check that effects vary (heterogeneous case)
        treatment_effects = results_het.treatment_effects
        effect_std = np.std(treatment_effects)
        
        # In heterogeneous case, effects should vary more
        self.assertGreater(effect_std, 0.1)
    
    def test_results_summary(self):
        """Test results summary generation."""
        results = self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        summary = results.summary()
        
        # Check that summary is a string
        self.assertIsInstance(summary, str)
        
        # Check that it contains expected information
        self.assertIn("Changes-in-Changes Model Results", summary)
        self.assertIn("Treatment Effect Summary", summary)
        self.assertIn("Data Information", summary)
    
    def test_results_plotting(self):
        """Test results plotting methods."""
        results = self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Test that plotting methods don't raise errors
        try:
            results.plot()
        except Exception as e:
            self.fail(f"plot() raised {e} unexpectedly!")
        
        try:
            results.plot_treatment_effects()
        except Exception as e:
            self.fail(f"plot_treatment_effects() raised {e} unexpectedly!")
    
    def test_results_to_dataframe(self):
        """Test conversion to DataFrame."""
        results = self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        df = results.to_dataframe()
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('quantile', df.columns)
        self.assertIn('treatment_effect', df.columns)
        self.assertIn('counterfactual', df.columns)
        
        # Check DataFrame size
        self.assertEqual(len(df), len(results.quantiles))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small sample sizes
        small_data = self._generate_synthetic_data(
            n_control=5,
            n_treatment=5,
            treatment_effect=1.0,
            heterogenous_effect=False,
            random_state=42
        )
        
        # Should work but with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self.cic.fit(
                small_data,
                outcome='outcome',
                group='group',
                period='period',
                control_group='control',
                treatment_group='treatment',
                before_period='before',
                after_period='after'
            )
            
            # Should have warnings about small sample sizes
            self.assertGreater(len(w), 0)
    
    def test_invalid_quantiles(self):
        """Test prediction with invalid quantiles."""
        # Fit the model first
        self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Test with quantiles outside [0, 1]
        with self.assertRaises(ValueError):
            self.cic.predict_quantiles(np.array([-0.1, 0.5, 1.1]))
    
    def test_comparison_with_did(self):
        """Test comparison with traditional DiD."""
        # Fit CiC model
        results = self.cic.fit(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Compute traditional DiD
        did_estimate = self._compute_did_estimate(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        # Check that both estimates are finite
        self.assertTrue(np.isfinite(did_estimate))
        self.assertTrue(np.isfinite(results.mean_treatment_effect))
        
        # Check that they are reasonably close (for homogeneous effects)
        difference = abs(did_estimate - results.mean_treatment_effect)
        self.assertLess(difference, 1.0)  # Should be close for homogeneous effects


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = self._generate_synthetic_data(
            n_control=50,
            n_treatment=50,
            treatment_effect=2.0,
            heterogenous_effect=False,
            random_state=42
        )
    
    def _generate_synthetic_data(self, n_control=50, n_treatment=50, treatment_effect=2.0, heterogenous_effect=False, random_state=42):
        """Generate synthetic data for testing."""
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
    
    def _compute_did_estimate(self, data, outcome='outcome', group='group', period='period', 
                            control_group='control', treatment_group='treatment', 
                            before_period='before', after_period='after'):
        """Compute traditional DiD estimate."""
        control_before_mean = data[(data[group] == control_group) & (data[period] == before_period)][outcome].mean()
        control_after_mean = data[(data[group] == control_group) & (data[period] == after_period)][outcome].mean()
        treatment_before_mean = data[(data[group] == treatment_group) & (data[period] == before_period)][outcome].mean()
        treatment_after_mean = data[(data[group] == treatment_group) & (data[period] == after_period)][outcome].mean()
        return (treatment_after_mean - treatment_before_mean) - (control_after_mean - control_before_mean)
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        # Should not raise any exceptions
        self.cic._validate_data(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        with self.assertRaises(ValueError):
            self.cic._validate_data(
                self.data,
                outcome='nonexistent',
                group='group',
                period='period',
                control_group='control',
                treatment_group='treatment',
                before_period='before',
                after_period='after'
            )
    
    def test_validate_data_missing_values(self):
        """Test data validation with missing values."""
        data_with_nulls = self.data.copy()
        data_with_nulls.loc[0, 'outcome'] = np.nan
        
        with self.assertRaises(ValueError):
            self.cic._validate_data(
                data_with_nulls,
                outcome='outcome',
                group='group',
                period='period',
                control_group='control',
                treatment_group='treatment',
                before_period='before',
                after_period='after'
            )
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        # Test basic generation
        data = self._generate_synthetic_data(
            n_control=100,
            n_treatment=100,
            treatment_effect=2.0,
            heterogenous_effect=False,
            random_state=42
        )
        
        self.assertEqual(len(data), 400)  # 100 * 4 groups
        self.assertIn('outcome', data.columns)
        self.assertIn('group', data.columns)
        self.assertIn('period', data.columns)
        
        # Check group sizes
        self.assertEqual(len(data[data['group'] == 'control']), 200)
        self.assertEqual(len(data[data['group'] == 'treatment']), 200)
        
        # Check period sizes
        self.assertEqual(len(data[data['period'] == 'before']), 200)
        self.assertEqual(len(data[data['period'] == 'after']), 200)
        
        # Test heterogeneous effects
        data_het = self._generate_synthetic_data(
            n_control=100,
            n_treatment=100,
            treatment_effect=2.0,
            heterogenous_effect=True,
            random_state=42
        )
        
        self.assertEqual(len(data_het), 400)
    
    def test_compute_did_estimate(self):
        """Test traditional DiD computation."""
        did_estimate = self._compute_did_estimate(
            self.data,
            outcome='outcome',
            group='group',
            period='period',
            control_group='control',
            treatment_group='treatment',
            before_period='before',
            after_period='after'
        )
        
        self.assertTrue(np.isfinite(did_estimate))
        self.assertIsInstance(did_estimate, float)


if __name__ == '__main__':
    unittest.main() 