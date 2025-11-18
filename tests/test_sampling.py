"""
Unit tests for the sampling module.
"""

import unittest
import numpy as np
from sigmondsamplings.sampling import (
    EnsembleInfo,
    SamplingInfo,
    ObservableInfo,
    SigmondSampling,
    UNCERTAINTIES_AVAILABLE,
)

try:
    from uncertainties import ufloat
except ImportError:
    pass


class TestEnsembleInfo(unittest.TestCase):
    """Test EnsembleInfo class."""

    def test_creation(self):
        """Test basic creation of EnsembleInfo."""
        ensemble = EnsembleInfo("test_ensemble", 1000, 100)
        self.assertEqual(ensemble.ensemble_name, "test_ensemble")
        self.assertEqual(ensemble.num_measurements, 1000)
        self.assertEqual(ensemble.num_bins, 100)
        self.assertEqual(ensemble.tweak_info, {})

    def test_creation_with_tweak(self):
        """Test creation with tweak info."""
        tweak = {"Rebin": "25"}
        ensemble = EnsembleInfo("test_ensemble", 1000, 100, tweak)
        self.assertEqual(ensemble.tweak_info, tweak)

    def test_equality(self):
        """Test equality comparison."""
        e1 = EnsembleInfo("test", 1000, 100)
        e2 = EnsembleInfo("test", 1000, 100)
        e3 = EnsembleInfo("test2", 1000, 100)

        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, e3)
        self.assertNotEqual(e1, "not_an_ensemble")

    def test_repr(self):
        """Test string representation."""
        ensemble = EnsembleInfo("test", 1000, 100)
        expected = "EnsembleInfo('test', 1000, 100)"
        self.assertEqual(repr(ensemble), expected)


class TestSamplingInfo(unittest.TestCase):
    """Test SamplingInfo class."""

    def test_creation_bootstrap(self):
        """Test bootstrap sampling info creation."""
        sampling = SamplingInfo("bootstrap", 1000, 1234, 0)
        self.assertEqual(sampling.method, "bootstrap")
        self.assertEqual(sampling.num_resamplings, 1000)
        self.assertEqual(sampling.seed, 1234)
        self.assertEqual(sampling.boot_skip, 0)

    def test_creation_jackknife(self):
        """Test jackknife sampling info creation."""
        sampling = SamplingInfo("jackknife", 500)
        self.assertEqual(sampling.method, "jackknife")
        self.assertEqual(sampling.num_resamplings, 500)
        self.assertEqual(sampling.seed, 0)

    def test_case_insensitive(self):
        """Test method name is case insensitive."""
        sampling = SamplingInfo("BOOTSTRAP", 1000)
        self.assertEqual(sampling.method, "bootstrap")

    def test_equality(self):
        """Test equality comparison."""
        s1 = SamplingInfo("bootstrap", 1000, 1234, 0)
        s2 = SamplingInfo("bootstrap", 1000, 1234, 0)
        s3 = SamplingInfo("jackknife", 1000, 1234, 0)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_repr(self):
        """Test string representation."""
        sampling = SamplingInfo("bootstrap", 1000, 1234)
        expected = "SamplingInfo('bootstrap', 1000, seed=1234)"
        self.assertEqual(repr(sampling), expected)


class TestObservableInfo(unittest.TestCase):
    """Test ObservableInfo class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = EnsembleInfo("test_ensemble", 1000, 100)

    def test_creation(self):
        """Test basic creation."""
        obs = ObservableInfo("kaon", 0, "n", "re", self.ensemble)
        self.assertEqual(obs.name, "kaon")
        self.assertEqual(obs.index, 0)
        self.assertEqual(obs.op_type, "n")
        self.assertEqual(obs.re_im, "re")
        self.assertEqual(obs.ensemble_info, self.ensemble)

    def test_equality(self):
        """Test equality comparison."""
        o1 = ObservableInfo("kaon", 0, "n", "re", self.ensemble)
        o2 = ObservableInfo("kaon", 0, "n", "re", self.ensemble)
        o3 = ObservableInfo("pion", 0, "n", "re", self.ensemble)

        self.assertEqual(o1, o2)
        self.assertNotEqual(o1, o3)
        self.assertNotEqual(o1, "not_an_observable")

    def test_from_string(self):
        """Test creation from string."""
        obs = ObservableInfo.from_string("kaon 0 n re", self.ensemble)
        self.assertEqual(obs.name, "kaon")
        self.assertEqual(obs.index, 0)
        self.assertEqual(obs.op_type, "n")
        self.assertEqual(obs.re_im, "re")
        self.assertEqual(obs.ensemble_info, self.ensemble)

    def test_from_string_invalid(self):
        """Test invalid string raises error."""
        with self.assertRaises(ValueError):
            ObservableInfo.from_string("kaon", self.ensemble)

        with self.assertRaises(ValueError):
            ObservableInfo.from_string("kaon invalid_index n re", self.ensemble)

    def test_repr(self):
        """Test string representation."""
        obs = ObservableInfo("kaon", 0, "n", "re", self.ensemble)
        repr_str = repr(obs)
        self.assertIn("kaon", repr_str)
        self.assertIn("test_ensemble", repr_str)

        str_repr = str(obs)
        self.assertIn("kaon", str_repr)
        self.assertIn("test_ensemble", str_repr)


class TestSigmondSampling(unittest.TestCase):
    """Test SigmondSampling class."""

    def setUp(self):
        """Set up test data."""
        self.data = np.array(
            [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]
        )  # full sample + 5 resamples
        self.ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
        self.sampling_info = SamplingInfo("bootstrap", 5, 1234)
        self.observable_info = ObservableInfo(
            "test_obs", 0, "n", "re", self.ensemble_info
        )

    def test_creation(self):
        """Test basic creation."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        self.assertEqual(len(sampling.data), 6)
        self.assertFalse(sampling.is_complex)
        self.assertEqual(sampling.ensemble_info, self.ensemble_info)

    def test_creation_complex(self):
        """Test creation with complex data."""
        complex_data = self.data + 1j * self.data * 0.1
        sampling = SigmondSampling(
            complex_data, self.observable_info, self.sampling_info, is_complex=True
        )
        self.assertTrue(sampling.is_complex)

    def test_creation_from_list(self):
        """Test creation from list."""
        data_list = [1.0, 1.1, 0.9, 1.05]
        sampling = SigmondSampling(data_list, self.observable_info, self.sampling_info)
        self.assertEqual(len(sampling.data), 4)

    def test_invalid_data_shape(self):
        """Test invalid data shapes raise errors."""
        with self.assertRaises(ValueError):
            SigmondSampling(
                np.array([[1, 2], [3, 4]]), self.observable_info, self.sampling_info
            )

    def test_insufficient_data(self):
        """Test insufficient data raises error."""
        with self.assertRaises(ValueError):
            SigmondSampling(np.array([1.0]), self.observable_info, self.sampling_info)

    def test_properties(self):
        """Test properties."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)

        self.assertEqual(sampling.full_sample_value, 1.0)
        np.testing.assert_array_equal(
            sampling.resampled_values, np.array([1.1, 0.9, 1.05, 0.95, 1.02])
        )
        self.assertAlmostEqual(sampling.mean, 1.004, places=3)

    def test_bootstrap_error(self):
        """Test bootstrap error calculation."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        # Error should be std of resampled values
        expected_error = np.std(sampling.resampled_values, ddof=1)
        self.assertAlmostEqual(sampling.error, expected_error, places=6)

    def test_jackknife_error(self):
        """Test jackknife error calculation."""
        jackknife_info = SamplingInfo("jackknife", 5, 1234)
        obs_info_jk = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling = SigmondSampling(self.data, obs_info_jk, jackknife_info)
        # Error should be std * sqrt(n-1) for jackknife
        expected_error = np.std(sampling.resampled_values, ddof=1) * np.sqrt(4)
        self.assertAlmostEqual(sampling.error, expected_error, places=6)

    def test_addition_with_sampling(self):
        """Test addition with another sampling."""
        data2 = np.array([2.0, 2.1, 1.9, 2.05, 1.95, 2.02])
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)

        result = sampling1 + sampling2
        expected_data = self.data + data2
        np.testing.assert_array_equal(result.data, expected_data)

    def test_addition_with_scalar(self):
        """Test addition with scalar."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        result = sampling + 5.0
        expected_data = self.data + 5.0
        np.testing.assert_array_equal(result.data, expected_data)

    def test_subtraction_with_sampling(self):
        """Test subtraction with another sampling."""
        data2 = np.array([2.0, 2.1, 1.9, 2.05, 1.95, 2.02])
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)

        result = sampling1 - sampling2
        expected_data = self.data - data2
        np.testing.assert_array_equal(result.data, expected_data)

    def test_multiplication_with_sampling(self):
        """Test multiplication with another sampling."""
        data2 = np.array([2.0, 2.1, 1.9, 2.05, 1.95, 2.02])
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)

        result = sampling1 * sampling2
        expected_data = self.data * data2
        np.testing.assert_array_equal(result.data, expected_data)

    def test_multiplication_with_scalar(self):
        """Test multiplication with scalar."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        result = sampling * 2.0
        expected_data = self.data * 2.0
        np.testing.assert_array_equal(result.data, expected_data)

    def test_division_with_sampling(self):
        """Test division with another sampling."""
        data2 = np.array([2.0, 2.1, 1.9, 2.05, 1.95, 2.02])
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)

        result = sampling1 / sampling2
        expected_data = self.data / data2
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_incompatible_observables_error(self):
        """Test error when operating on incompatible samplings."""
        # Create observables with different info to test incompatibility
        obs_info2 = ObservableInfo("different_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(self.data, obs_info2, self.sampling_info)

        with self.assertRaises(ValueError):
            sampling1 + sampling2

    def test_incompatible_sampling_methods_error(self):
        """Test error when operating on incompatible sampling methods."""
        sampling_info2 = SamplingInfo("jackknife", 5, 1234)
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(self.data, obs_info2, sampling_info2)

        with self.assertRaises(ValueError):
            sampling1 + sampling2

    def test_different_lengths_error(self):
        """Test error when operating on different length data."""
        data2 = np.array([1.0, 1.1, 0.9])  # Different length
        obs_info2 = ObservableInfo("test_obs", 0, "n", "re", self.ensemble_info)
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(data2, obs_info2, self.sampling_info)

        with self.assertRaises(ValueError):
            sampling1 + sampling2

    def test_right_operations(self):
        """Test right-hand operations (e.g., 5 + sampling)."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)

        result_add = 5.0 + sampling
        expected_add = 5.0 + self.data
        np.testing.assert_array_equal(result_add.data, expected_add)

        result_mul = 2.0 * sampling
        expected_mul = 2.0 * self.data
        np.testing.assert_array_equal(result_mul.data, expected_mul)

    def test_complex_arithmetic(self):
        """Test arithmetic with complex numbers."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)

        result = sampling + 1j
        self.assertTrue(result.is_complex)
        expected_data = self.data + 1j
        np.testing.assert_array_equal(result.data, expected_data)

    def test_str_repr(self):
        """Test string representations."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)

        str_repr = str(sampling)
        self.assertIn("±", str_repr)

        repr_str = repr(sampling)
        self.assertIn("SigmondSampling", repr_str)

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_to_ufloat(self):
        """Test conversion to ufloat object."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        ufloat_obj = sampling.to_ufloat()

        # Check that value and error match
        self.assertAlmostEqual(
            ufloat_obj.nominal_value, sampling.full_sample_value, places=6
        )
        self.assertAlmostEqual(ufloat_obj.std_dev, sampling.error, places=6)

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_to_ufloat_complex_error(self):
        """Test that complex samplings raise error when converting to ufloat."""
        complex_data = self.data + 1j * self.data * 0.1
        sampling = SigmondSampling(
            complex_data, self.observable_info, self.sampling_info, is_complex=True
        )

        with self.assertRaises(ValueError):
            sampling.to_ufloat()

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_pdg_format_shorthand(self):
        """Test PDG formatting with shorthand notation."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        formatted = sampling.pdg_format(".1uS")

        # Should contain parentheses for shorthand notation
        self.assertIn("(", formatted)
        self.assertIn(")", formatted)

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_pdg_format_pretty(self):
        """Test PDG formatting with pretty-print notation."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        formatted = sampling.pdg_format(".1uP")

        # Should contain ± symbol for pretty-print
        self.assertIn("±", formatted)

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_pdg_format_scientific(self):
        """Test PDG formatting with scientific notation."""
        # Use larger values to trigger scientific notation
        large_data = self.data * 1e6
        sampling = SigmondSampling(large_data, self.observable_info, self.sampling_info)
        formatted = sampling.pdg_format(".1ue")

        # Should contain 'e' for scientific notation
        self.assertIn("e", formatted.lower())

    def test_uncertainties_unavailable_error(self):
        """Test proper error when uncertainties package not available."""
        # Temporarily monkey-patch to simulate missing package
        original_available = SigmondSampling.__module__.__dict__.get(
            "UNCERTAINTIES_AVAILABLE"
        )

        # Patch the module-level variable
        import sigmondsamplings.sampling as sampling_module

        original_available = sampling_module.UNCERTAINTIES_AVAILABLE
        sampling_module.UNCERTAINTIES_AVAILABLE = False

        try:
            sampling = SigmondSampling(
                self.data, self.observable_info, self.sampling_info
            )

            with self.assertRaises(ImportError):
                sampling.to_ufloat()

            with self.assertRaises(ImportError):
                sampling.pdg_format()
        finally:
            # Restore original value
            sampling_module.UNCERTAINTIES_AVAILABLE = original_available


class TestUncertaintiesIntegration(unittest.TestCase):
    """Test uncertainties package integration."""

    def setUp(self):
        """Set up test data with known values for easier testing."""
        # Use simple data: full sample = 1.0, resamples with std = 0.1
        self.data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        self.ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
        self.sampling_info = SamplingInfo("bootstrap", 4, 1234)
        self.observable_info = ObservableInfo(
            "test_obs", 0, "n", "re", self.ensemble_info
        )

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_ufloat_arithmetic_compatibility(self):
        """Test that ufloat objects from SigmondSampling work with uncertainties arithmetic."""
        sampling1 = SigmondSampling(self.data, self.observable_info, self.sampling_info)
        sampling2 = SigmondSampling(
            self.data * 2, self.observable_info, self.sampling_info
        )

        ufloat1 = sampling1.to_ufloat()
        ufloat2 = sampling2.to_ufloat()

        # Test arithmetic operations
        result_add = ufloat1 + ufloat2
        result_mul = ufloat1 * ufloat2

        # Check that results are ufloat objects (type name varies by uncertainties version)
        self.assertIn(type(result_add).__name__, ["Variable", "AffineScalarFunc"])
        self.assertIn(type(result_mul).__name__, ["Variable", "AffineScalarFunc"])

    @unittest.skipUnless(UNCERTAINTIES_AVAILABLE, "uncertainties package not available")
    def test_format_comparison_with_estimate_str(self):
        """Compare PDG formatting with existing estimate_str method."""
        sampling = SigmondSampling(self.data, self.observable_info, self.sampling_info)

        # Get both formats
        estimate_str = sampling.estimate_str(sig_figs=1)
        pdg_format = sampling.pdg_format(".1uS")

        # Both should contain the main value
        value_str = f"{sampling.full_sample_value:.1f}"
        self.assertIn(value_str[0], estimate_str)  # At least first digit should match
        self.assertIn(value_str[0], pdg_format)


if __name__ == "__main__":
    unittest.main()
