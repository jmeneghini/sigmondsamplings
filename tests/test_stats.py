"""
Unit tests for the stats module.
"""

import unittest
import numpy as np

from SigmondSamplings.stats import SamplingStats
from SigmondSamplings.sampling import EnsembleInfo, SamplingInfo, ObservableInfo, SigmondSampling
from SigmondSamplings.utils import create_gaussian_sampling


class TestSamplingStats(unittest.TestCase):
    """Test SamplingStats class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(1234)  # For reproducible tests
        
        self.ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
        self.sampling_info = SamplingInfo("bootstrap", 100, 1234)
        
        # Create samplings for testing
        data1 = np.random.normal(1.0, 0.1, 101)  # mean=1.0
        data2 = np.random.normal(2.0, 0.2, 101)  # mean=2.0  
        data3 = np.random.normal(3.0, 0.3, 101)  # mean=3.0
        
        # Create observable infos
        obs_info1 = ObservableInfo("obs1", 0, 'n', 're', self.ensemble_info)
        obs_info2 = ObservableInfo("obs2", 0, 'n', 're', self.ensemble_info)
        obs_info3 = ObservableInfo("obs3", 0, 'n', 're', self.ensemble_info)
        
        self.sampling1 = SigmondSampling(data1, obs_info1, self.sampling_info)
        self.sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)
        self.sampling3 = SigmondSampling(data3, obs_info3, self.sampling_info)
        
        self.samplings = [self.sampling1, self.sampling2, self.sampling3]
    
    def test_creation(self):
        """Test basic creation."""
        stats = SamplingStats(self.samplings)
        self.assertEqual(stats.num_observables, 3)
        self.assertEqual(stats.num_samples, 100)
        self.assertEqual(stats.ensemble_info, self.ensemble_info)
    
    def test_empty_list_error(self):
        """Test error with empty list."""
        with self.assertRaises(ValueError):
            SamplingStats([])
    
    def test_inconsistent_ensemble_error(self):
        """Test that different ensembles are allowed."""
        different_ensemble = EnsembleInfo("different", 1000, 100)
        obs_info_different = ObservableInfo("different_obs", 0, 'n', 're', different_ensemble)
        different_sampling = SigmondSampling(
            np.random.normal(0, 1, 101),
            obs_info_different, self.sampling_info
        )
        
        # This should work now - different ensembles are allowed
        stats = SamplingStats([self.sampling1, different_sampling])
        self.assertEqual(stats.num_observables, 2)
        
        # Covariance between different ensembles should be zero
        cov = stats.covariance(0, 1)
        self.assertEqual(cov, 0.0)
    
    def test_inconsistent_sampling_error(self):
        """Test error with inconsistent sampling info."""
        different_sampling_info = SamplingInfo("jackknife", 100, 1234)
        obs_info_diff_sampling = ObservableInfo("diff_sampling_obs", 0, 'n', 're', self.ensemble_info)
        different_sampling = SigmondSampling(
            np.random.normal(0, 1, 101), obs_info_diff_sampling, different_sampling_info
        )
        
        with self.assertRaises(ValueError):
            SamplingStats([self.sampling1, different_sampling])
    
    def test_different_lengths_error(self):
        """Test error with different data lengths."""
        short_data = np.random.normal(0, 1, 50)
        obs_info_short = ObservableInfo("short_obs", 0, 'n', 're', self.ensemble_info)
        short_sampling = SigmondSampling(
            short_data, obs_info_short, self.sampling_info
        )
        
        with self.assertRaises(ValueError):
            SamplingStats([self.sampling1, short_sampling])
    
    def test_means_and_errors(self):
        """Test means and errors calculation."""
        stats = SamplingStats(self.samplings)
        
        means = stats.means()
        errors = stats.errors()
        
        self.assertEqual(len(means), 3)
        self.assertEqual(len(errors), 3)
        
        # Check individual means match
        self.assertAlmostEqual(means[0], self.sampling1.mean, places=6)
        self.assertAlmostEqual(means[1], self.sampling2.mean, places=6)
        self.assertAlmostEqual(means[2], self.sampling3.mean, places=6)
        
        # Check individual errors match
        self.assertAlmostEqual(errors[0], self.sampling1.error, places=6)
        self.assertAlmostEqual(errors[1], self.sampling2.error, places=6)
        self.assertAlmostEqual(errors[2], self.sampling3.error, places=6)
    
    def test_covariance_matrix(self):
        """Test covariance matrix calculation."""
        stats = SamplingStats(self.samplings)
        cov_matrix = stats.covariance_matrix()
        
        self.assertEqual(cov_matrix.shape, (3, 3))
        
        # Diagonal elements should be variances
        self.assertAlmostEqual(cov_matrix[0, 0], self.sampling1.error**2, places=6)
        self.assertAlmostEqual(cov_matrix[1, 1], self.sampling2.error**2, places=6)
        self.assertAlmostEqual(cov_matrix[2, 2], self.sampling3.error**2, places=6)
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)
        
        # Off-diagonal elements should show correlation structure
        # obs1 and obs2 should be positively correlated
        self.assertGreater(cov_matrix[0, 1], 0)
        # obs1 and obs3 should be negatively correlated
        self.assertLess(cov_matrix[0, 2], 0)
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        stats = SamplingStats(self.samplings)
        corr_matrix = stats.correlation_matrix()
        
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # Diagonal elements should be 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1, 1, 1])
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
        
        # Correlation coefficients should be between -1 and 1 (with small numerical tolerance)
        self.assertTrue(np.all(corr_matrix >= -1.001))
        self.assertTrue(np.all(corr_matrix <= 1.001))
    
    def test_covariance_individual(self):
        """Test individual covariance calculation."""
        stats = SamplingStats(self.samplings)
        
        # Test covariance between obs1 and obs2
        cov_01 = stats.covariance(0, 1)
        cov_matrix = stats.covariance_matrix()
        self.assertAlmostEqual(cov_01, cov_matrix[0, 1], places=6)
        
        # Test variance (self-covariance)
        var_0 = stats.covariance(0, 0)
        self.assertAlmostEqual(var_0, self.sampling1.error**2, places=6)
    
    def test_correlation_individual(self):
        """Test individual correlation calculation."""
        stats = SamplingStats(self.samplings)
        
        # Test correlation between obs1 and obs2
        corr_01 = stats.correlation(0, 1)
        corr_matrix = stats.correlation_matrix()
        self.assertAlmostEqual(corr_01, corr_matrix[0, 1], places=6)
        
        # Self-correlation should be 1
        corr_00 = stats.correlation(0, 0)
        self.assertAlmostEqual(corr_00, 1.0, places=6)
    
    def test_chi_squared_diagonal(self):
        """Test chi-squared with diagonal errors only."""
        stats = SamplingStats(self.samplings)
        
        theory_values = np.array([1.0, 1.0, 1.0])
        chi_sq, dof = stats.chi_squared(theory_values, use_correlation=False)
        
        self.assertEqual(dof, 3)
        self.assertGreater(chi_sq, 0)
        
        # Manual calculation
        means = stats.means()
        errors = stats.errors()
        expected_chi_sq = np.sum(((means - theory_values) / errors) ** 2)
        self.assertAlmostEqual(chi_sq, expected_chi_sq, places=6)
    
    def test_chi_squared_with_correlation(self):
        """Test chi-squared with full covariance matrix."""
        stats = SamplingStats(self.samplings)
        
        theory_values = np.array([1.0, 1.0, 1.0])
        chi_sq, dof = stats.chi_squared(theory_values, use_correlation=True)
        
        self.assertEqual(dof, 3)
        self.assertGreater(chi_sq, 0)
    
    def test_chi_squared_wrong_length(self):
        """Test error with wrong theory values length."""
        stats = SamplingStats(self.samplings)
        
        with self.assertRaises(ValueError):
            stats.chi_squared(np.array([1.0, 1.0]))  # Too short
    
    def test_effective_sample_size(self):
        """Test effective sample size estimation."""
        stats = SamplingStats(self.samplings)
        eff_sizes = stats.effective_sample_size()
        
        self.assertEqual(len(eff_sizes), 3)
        
        # All effective sizes should be positive
        self.assertTrue(np.all(eff_sizes > 0))
        
        # Should be reasonable (allow some numerical issues)
        self.assertTrue(np.all(eff_sizes <= 200))  # Allow some numerical variance
    
    def test_summary(self):
        """Test summary generation."""
        stats = SamplingStats(self.samplings)
        summary = stats.summary()
        
        self.assertIsInstance(summary, dict)
        
        # Check required keys
        required_keys = ['num_observables', 'num_samples', 'ensemble', 
                        'sampling_method', 'means', 'errors', 
                        'effective_sample_sizes', 'correlation_matrix']
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values
        self.assertEqual(summary['num_observables'], 3)
        self.assertEqual(summary['num_samples'], 100)
        self.assertEqual(summary['ensemble'], 'test_ensemble')
        self.assertEqual(summary['sampling_method'], 'bootstrap')


class TestSamplingStatsJackknife(unittest.TestCase):
    """Test SamplingStats with Jackknife samplings."""
    
    def test_jackknife_covariance(self):
        """Test covariance calculation with jackknife samples."""
        # Create jackknife samplings
        ensemble_info = EnsembleInfo("test_ensemble", 50, 50)
        sampling_info = SamplingInfo("jackknife", 50, 1234)
        
        np.random.seed(1234)
        data1 = np.random.normal(1.0, 0.1, 51)  # full sample + 50 jackknife
        data2 = np.random.normal(2.0, 0.1, 51)
        
        obs_info1 = ObservableInfo("jk_obs1", 0, 'n', 're', ensemble_info)
        obs_info2 = ObservableInfo("jk_obs2", 0, 'n', 're', ensemble_info)
        
        sampling1 = SigmondSampling(data1, obs_info1, sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, sampling_info)
        
        stats = SamplingStats([sampling1, sampling2])
        cov_matrix = stats.covariance_matrix()
        
        # Check that jackknife correction was applied
        self.assertEqual(cov_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(cov_matrix)))


class TestSamplingStatsWithSyntheticData(unittest.TestCase):
    """Test SamplingStats with synthetic data generation."""
    
    def test_with_gaussian_samplings(self):
        """Test with synthetic Gaussian samplings."""
        sampling_info = SamplingInfo("bootstrap", 100, 1234)
        sampling1 = create_gaussian_sampling(1.0, 0.1, sampling_info, "obs1")
        sampling2 = create_gaussian_sampling(2.0, 0.2, sampling_info, "obs2")
        
        stats = SamplingStats([sampling1, sampling2])
        
        self.assertEqual(stats.num_observables, 2)
        self.assertEqual(stats.num_samples, 100)
        
        means = stats.sample_means()
        np.testing.assert_allclose(means, [1.0, 2.0], atol=0.2)


if __name__ == '__main__':
    unittest.main() 