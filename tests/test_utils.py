"""
Unit tests for the utils module.
"""

import unittest
import numpy as np
from SigmondSamplings.utils import (
    create_gaussian_sampling,
    create_uniform_sampling,
    create_complex_gaussian_sampling,
    bootstrap_resample,
    jackknife_resample,
    combine_real_imaginary,
    split_complex_sampling,
    effective_sample_size,
    block_average,
)
from SigmondSamplings.sampling import (
    SigmondSampling,
    ObservableInfo,
    EnsembleInfo,
    SamplingInfo,
)


class TestSyntheticSamplings(unittest.TestCase):
    """Test synthetic sampling creation functions."""

    def test_create_gaussian_sampling(self):
        """Test Gaussian sampling creation."""
        sampling_info = SamplingInfo("bootstrap", 100, 1234)
        sampling = create_gaussian_sampling(
            mean=1.0, std=0.1, sampling_info=sampling_info
        )

        self.assertIsInstance(sampling, SigmondSampling)
        self.assertEqual(len(sampling.data), 101)  # full sample + 100 resamples
        self.assertEqual(sampling.full_sample_value, 1.0)
        self.assertEqual(sampling.sampling_info.method, "bootstrap")
        self.assertFalse(sampling.is_complex)

        # Check that resampled values are roughly Gaussian
        self.assertAlmostEqual(np.mean(sampling.resampled_values), 1.0, delta=0.1)
        self.assertAlmostEqual(np.std(sampling.resampled_values), 0.1, delta=0.05)

    def test_create_gaussian_sampling_reproducible(self):
        """Test that Gaussian sampling is reproducible with seed."""
        sampling_info1 = SamplingInfo("bootstrap", 50, 1234)
        sampling_info2 = SamplingInfo("bootstrap", 50, 1234)
        sampling1 = create_gaussian_sampling(1.0, 0.1, sampling_info1)
        sampling2 = create_gaussian_sampling(1.0, 0.1, sampling_info2)

        np.testing.assert_array_equal(sampling1.data, sampling2.data)

    def test_create_uniform_sampling(self):
        """Test uniform sampling creation."""
        sampling = create_uniform_sampling(
            low=0.5, high=1.5, num_samples=100, seed=1234
        )

        self.assertIsInstance(sampling, SigmondSampling)
        self.assertEqual(len(sampling.data), 101)
        self.assertEqual(sampling.full_sample_value, 1.0)  # (0.5 + 1.5) / 2

        # Check that all values are within bounds
        self.assertTrue(np.all(sampling.resampled_values >= 0.5))
        self.assertTrue(np.all(sampling.resampled_values <= 1.5))

    def test_create_complex_gaussian_sampling(self):
        """Test complex Gaussian sampling creation."""
        sampling_info = SamplingInfo("bootstrap", 100, 1234)
        sampling = create_complex_gaussian_sampling(
            mean_real=1.0,
            std_real=0.1,
            mean_imag=0.5,
            std_imag=0.05,
            sampling_info=sampling_info,
        )

        self.assertIsInstance(sampling, SigmondSampling)
        self.assertTrue(sampling.is_complex)
        self.assertEqual(len(sampling.data), 101)
        self.assertEqual(sampling.full_sample_value, 1.0 + 0.5j)

        # Check real and imaginary parts separately
        real_parts = np.real(sampling.resampled_values)
        imag_parts = np.imag(sampling.resampled_values)

        self.assertAlmostEqual(np.mean(real_parts), 1.0, delta=0.1)
        self.assertAlmostEqual(np.mean(imag_parts), 0.5, delta=0.1)


class TestResamplingFunctions(unittest.TestCase):
    """Test resampling functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(1234)
        self.data = np.random.normal(5.0, 1.0, 50)

    def test_bootstrap_resample(self):
        """Test bootstrap resampling."""
        samples = bootstrap_resample(self.data, num_samples=100, seed=1234)

        self.assertEqual(len(samples), 100)
        # Bootstrap samples should be roughly centered on original mean
        self.assertAlmostEqual(np.mean(samples), np.mean(self.data), delta=0.2)

        # Should have some variability
        self.assertGreater(np.std(samples), 0.05)

    def test_bootstrap_reproducible(self):
        """Test bootstrap reproducibility."""
        samples1 = bootstrap_resample(self.data, 50, seed=1234)
        samples2 = bootstrap_resample(self.data, 50, seed=1234)

        np.testing.assert_array_equal(samples1, samples2)

    def test_jackknife_resample(self):
        """Test jackknife resampling."""
        samples = jackknife_resample(self.data)

        self.assertEqual(len(samples), len(self.data))
        # Jackknife samples should be close to original mean
        self.assertAlmostEqual(np.mean(samples), np.mean(self.data), delta=0.1)

        # Each sample should be leave-one-out mean
        for i, sample in enumerate(samples):
            expected = (np.sum(self.data) - self.data[i]) / (len(self.data) - 1)
            self.assertAlmostEqual(sample, expected, places=10)


class TestComplexOperations(unittest.TestCase):
    """Test complex number operations."""

    def setUp(self):
        """Set up test data."""
        self.real_data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        self.imag_data = np.array([0.5, 0.55, 0.45, 0.52, 0.48])

        self.ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
        self.sampling_info = SamplingInfo("bootstrap", 4, 1234)

        # Create observable infos
        real_obs_info = ObservableInfo("real_obs", 0, "n", "re", self.ensemble_info)
        imag_obs_info = ObservableInfo("imag_obs", 0, "n", "im", self.ensemble_info)

        self.real_sampling = SigmondSampling(
            self.real_data, real_obs_info, self.sampling_info
        )
        self.imag_sampling = SigmondSampling(
            self.imag_data, imag_obs_info, self.sampling_info
        )

    def test_combine_real_imaginary(self):
        """Test combining real and imaginary parts."""
        complex_sampling = combine_real_imaginary(
            self.real_sampling, self.imag_sampling
        )

        self.assertTrue(complex_sampling.is_complex)
        expected_data = self.real_data + 1j * self.imag_data
        np.testing.assert_array_equal(complex_sampling.data, expected_data)

    def test_combine_incompatible_error(self):
        """Test error when combining incompatible samplings."""
        different_ensemble = EnsembleInfo("different", 1000, 100)
        different_obs_info = ObservableInfo(
            "different_obs", 0, "n", "im", different_ensemble
        )
        different_sampling = SigmondSampling(
            self.imag_data, different_obs_info, self.sampling_info
        )

        with self.assertRaises(ValueError):
            combine_real_imaginary(self.real_sampling, different_sampling)

    def test_split_complex_sampling(self):
        """Test splitting complex sampling."""
        complex_data = self.real_data + 1j * self.imag_data
        complex_obs_info = ObservableInfo(
            "complex_obs", 0, "n", "cx", self.ensemble_info
        )
        complex_sampling = SigmondSampling(
            complex_data, complex_obs_info, self.sampling_info, is_complex=True
        )

        real_part, imag_part = split_complex_sampling(complex_sampling)

        self.assertFalse(real_part.is_complex)
        self.assertFalse(imag_part.is_complex)
        np.testing.assert_array_equal(real_part.data, self.real_data)
        np.testing.assert_array_equal(imag_part.data, self.imag_data)

    def test_split_real_sampling_error(self):
        """Test error when trying to split real sampling."""
        with self.assertRaises(ValueError):
            split_complex_sampling(self.real_sampling)


class TestStatisticalFunctions(unittest.TestCase):
    """Test statistical analysis functions."""

    def setUp(self):
        """Set up test data."""
        # Create data with known autocorrelation
        np.random.seed(1234)
        self.uncorrelated_data = np.random.normal(0, 1, 1000)

        # Create correlated data
        self.correlated_data = np.zeros(1000)
        self.correlated_data[0] = np.random.normal(0, 1)
        for i in range(1, 1000):
            self.correlated_data[i] = 0.5 * self.correlated_data[
                i - 1
            ] + np.random.normal(0, 1)

    def test_effective_sample_size_uncorrelated(self):
        """Test effective sample size for uncorrelated data."""
        eff_size = effective_sample_size(self.uncorrelated_data)

        # For uncorrelated data, effective size should be close to actual size
        self.assertGreater(eff_size, 500)  # Should be substantial fraction of 1000

    def test_effective_sample_size_correlated(self):
        """Test effective sample size for correlated data."""
        eff_size = effective_sample_size(self.correlated_data)

        # For correlated data, effective size should be much smaller
        self.assertLess(eff_size, 1000)
        self.assertGreater(eff_size, 1)  # But still positive

    def test_block_average(self):
        """Test block averaging."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        block_means, block_error = block_average(data, block_size=2)

        expected_means = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
        np.testing.assert_array_equal(block_means, expected_means)

        # Error should be standard error of block means
        expected_error = np.std(expected_means, ddof=1) / np.sqrt(5)
        self.assertAlmostEqual(block_error, expected_error, places=10)

    def test_block_average_large_block_error(self):
        """Test error when block size is too large."""
        data = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            block_average(data, block_size=5)


class TestCustomObservableNames(unittest.TestCase):
    """Test custom observable naming."""

    def test_custom_names_gaussian(self):
        """Test custom names in Gaussian sampling."""
        sampling_info = SamplingInfo("jackknife", 10, 1234)
        sampling = create_gaussian_sampling(
            mean=1.0,
            std=0.1,
            sampling_info=sampling_info,
            observable_name="my_observable",
            ensemble_name="my_ensemble",
        )

        self.assertEqual(sampling.ensemble_info.ensemble_name, "my_ensemble")
        self.assertEqual(sampling.sampling_info.method, "jackknife")
        self.assertEqual(len(sampling.data), 11)

    def test_custom_names_complex(self):
        """Test custom names in complex sampling."""
        sampling_info = SamplingInfo("bootstrap", 10, 1234)
        sampling = create_complex_gaussian_sampling(
            mean_real=1.0,
            std_real=0.1,
            mean_imag=0.5,
            std_imag=0.05,
            sampling_info=sampling_info,
            observable_name="complex_obs",
            ensemble_name="test_ensemble",
        )

        self.assertTrue(sampling.is_complex)
        self.assertEqual(sampling.ensemble_info.ensemble_name, "test_ensemble")


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_create_uniform_sampling(self):
        """Test creating uniform sampling."""
        sampling = create_uniform_sampling(0.0, 1.0, 100)

        self.assertEqual(len(sampling.data), 101)  # full sample + 100 resamples
        self.assertFalse(sampling.is_complex)
        self.assertTrue(0.4 <= sampling.mean <= 0.6)  # Should be around 0.5

    def test_create_complex_gaussian_sampling(self):
        """Test creating complex gaussian sampling."""
        sampling_info = SamplingInfo("bootstrap", 100, 1234)
        sampling = create_complex_gaussian_sampling(1.0, 0.1, 2.0, 0.2, sampling_info)

        self.assertEqual(len(sampling.data), 101)
        self.assertTrue(sampling.is_complex)

        # Check means are approximately correct
        real_mean = np.mean(np.real(sampling.resampled_values))
        imag_mean = np.mean(np.imag(sampling.resampled_values))
        self.assertAlmostEqual(real_mean, 1.0, delta=0.2)
        self.assertAlmostEqual(imag_mean, 2.0, delta=0.2)

    def test_combine_real_imaginary_basic(self):
        """Test combining real and imaginary samplings."""
        np.random.seed(1234)

        # Create test samplings
        ensemble_info = EnsembleInfo("test", 1000, 10)
        sampling_info = SamplingInfo("bootstrap", 10, 1234)

        real_data = np.random.normal(1.0, 0.1, 11)
        imag_data = np.random.normal(2.0, 0.1, 11)

        real_obs_info = ObservableInfo("real_obs", 0, "n", "re", ensemble_info)
        imag_obs_info = ObservableInfo("imag_obs", 0, "n", "im", ensemble_info)

        real_sampling = SigmondSampling(real_data, real_obs_info, sampling_info)
        imag_sampling = SigmondSampling(imag_data, imag_obs_info, sampling_info)

        # Test combination
        result = combine_real_imaginary(real_sampling, imag_sampling)

        self.assertTrue(result.is_complex)
        np.testing.assert_array_equal(np.real(result.data), real_data)
        np.testing.assert_array_equal(np.imag(result.data), imag_data)

    def test_split_complex_basic(self):
        """Test splitting complex sampling."""
        np.random.seed(5678)

        # Create complex data
        ensemble_info = EnsembleInfo("test", 1000, 10)
        sampling_info = SamplingInfo("bootstrap", 10, 5678)

        real_part = np.random.normal(1.0, 0.1, 11)
        imag_part = np.random.normal(2.0, 0.1, 11)
        complex_data = real_part + 1j * imag_part

        complex_obs_info = ObservableInfo("complex_test", 0, "n", "cx", ensemble_info)
        complex_sampling = SigmondSampling(
            complex_data, complex_obs_info, sampling_info, is_complex=True
        )

        # Test splitting
        real_result, imag_result = split_complex_sampling(complex_sampling)

        self.assertFalse(real_result.is_complex)
        self.assertFalse(imag_result.is_complex)
        np.testing.assert_array_equal(real_result.data, real_part)
        np.testing.assert_array_equal(imag_result.data, imag_part)

    def test_split_non_complex_error(self):
        """Test error when splitting non-complex sampling."""
        sampling = create_uniform_sampling(0.0, 1.0, 10)

        with self.assertRaises(ValueError):
            split_complex_sampling(sampling)


class TestCreateAndCombineComplexSampling(unittest.TestCase):
    """Test complex sampling creation and manipulation."""

    def test_create_and_combine(self):
        """Test creating and combining samplings."""
        # Create uniform sampling
        sampling = create_uniform_sampling(0.0, 1.0, 10)
        self.assertEqual(len(sampling.data), 11)

        # Create complex gaussian
        sampling_info = SamplingInfo("bootstrap", 10, 1234)
        complex_sampling = create_complex_gaussian_sampling(
            1.0, 0.1, 2.0, 0.1, sampling_info
        )
        self.assertTrue(complex_sampling.is_complex)


if __name__ == "__main__":
    unittest.main()
