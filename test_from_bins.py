#!/usr/bin/env python3
"""
Test script for the new from_bins() functionality.
"""

import sys
import os

# Add the sigmondsamplings directory to the path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sigmondsamplings.sampling import (
    SigmondSampling,
    ObservableInfo,
    EnsembleInfo,
    SamplingInfo,
)


def test_basic_from_bins():
    """Test basic from_bins functionality."""
    print("=" * 60)
    print("Test 1: Basic from_bins with bootstrap (no rebinning)")
    print("=" * 60)

    # Create synthetic time-series data
    np.random.seed(42)
    raw_bins = np.random.normal(5.0, 1.0, 1000)

    # Create metadata (no rebinning specified)
    ensemble_info = EnsembleInfo("test_ensemble", 1000)
    obs_info = ObservableInfo("energy", 0, "n", "re", ensemble_info)
    samp_info = SamplingInfo("bootstrap", 500, seed=1234)

    # Create sampling from bins
    mean_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="mean",
    )

    print(f"Mean: {mean_samp.full_sample_value:.6f} ± {mean_samp.error:.6f}")
    print(f"Ensemble info: {mean_samp.ensemble_info}")
    print(f"Number of bins: {mean_samp.ensemble_info.num_bins}")
    print(f"Rebin factor: {mean_samp.ensemble_info.tweak_info.get('rebin', 1)}")
    print()


def test_rebinning_with_rebin_size():
    """Test rebinning functionality using rebin_size."""
    print("=" * 60)
    print("Test 2: From bins with rebin_size")
    print("=" * 60)

    # Create synthetic time-series data
    np.random.seed(42)
    raw_bins = np.random.normal(5.0, 1.0, 1000)

    # Create metadata with rebin_size
    ensemble_info = EnsembleInfo("test_ensemble", 1000, rebin_size=2)
    obs_info = ObservableInfo("energy", 0, "n", "re", ensemble_info)
    samp_info = SamplingInfo("bootstrap", 500, seed=1234)

    # Create sampling with rebinning
    mean_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="mean",
    )

    print(f"Mean: {mean_samp.full_sample_value:.6f} ± {mean_samp.error:.6f}")
    print(f"Original bins: 1000")
    print(f"Number of bins after rebinning: {mean_samp.ensemble_info.num_bins}")
    print(f"Rebin factor: {mean_samp.ensemble_info.tweak_info['rebin']}")
    print()


def test_num_bins_parameter():
    """Test using num_bins parameter."""
    print("=" * 60)
    print("Test 3: Using target num_bins")
    print("=" * 60)

    # Create synthetic time-series data
    np.random.seed(42)
    raw_bins = np.random.normal(5.0, 1.0, 1000)

    # Create metadata targeting 250 bins
    ensemble_info = EnsembleInfo("test_ensemble", 1000, num_bins=250)
    obs_info = ObservableInfo("energy", 0, "n", "re", ensemble_info)
    samp_info = SamplingInfo("bootstrap", 500, seed=1234)

    # Create sampling
    mean_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="mean",
    )

    print(f"Mean: {mean_samp.full_sample_value:.6f} ± {mean_samp.error:.6f}")
    print(f"Target bins: 250")
    print(f"Actual bins: {mean_samp.ensemble_info.num_bins}")
    print(f"Calculated rebin factor: {mean_samp.ensemble_info.tweak_info['rebin']}")
    print()


def test_multiple_statistics():
    """Test creating multiple statistics from the same bins."""
    print("=" * 60)
    print("Test 4: Multiple statistics (mean and variance)")
    print("=" * 60)

    # Create synthetic time-series data
    np.random.seed(42)
    raw_bins = np.random.normal(5.0, 1.0, 1000)

    # Create metadata with rebinning
    ensemble_info = EnsembleInfo("test_ensemble", 1000, rebin_size=2)
    obs_info = ObservableInfo("energy", 0, "n", "re", ensemble_info)
    samp_info = SamplingInfo("bootstrap", 500, seed=1234)

    # Create mean sampling
    mean_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="mean",
    )

    # Create variance sampling (uses same seed, so same resamples)
    var_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="variance",
    )

    print(f"Mean: {mean_samp.full_sample_value:.6f} ± {mean_samp.error:.6f}")
    print(f"Variance: {var_samp.full_sample_value:.6f} ± {var_samp.error:.6f}")
    print()
    print("Note: Both use the same seed, so bootstrap resamples are identical.")
    print()


def test_jackknife():
    """Test jackknife resampling."""
    print("=" * 60)
    print("Test 5: Jackknife resampling")
    print("=" * 60)

    # Create synthetic time-series data (fewer bins for jackknife)
    np.random.seed(42)
    raw_bins = np.random.normal(5.0, 1.0, 100)

    # Create metadata
    ensemble_info = EnsembleInfo("test_ensemble", 100)
    obs_info = ObservableInfo("energy", 0, "n", "re", ensemble_info)
    samp_info = SamplingInfo("jackknife", 100)  # num_resamplings = num_bins

    # Create sampling with jackknife
    mean_samp = SigmondSampling.from_bins(
        bins_data=raw_bins,
        observable_info=obs_info,
        sampling_info=samp_info,
        statistic="mean",
    )

    print(f"Mean: {mean_samp.full_sample_value:.6f} ± {mean_samp.error:.6f}")
    print(f"Number of jackknife samples: {len(mean_samp.resampled_values)}")
    print(f"Ensemble info: {mean_samp.ensemble_info}")
    print()


if __name__ == "__main__":
    test_basic_from_bins()
    test_rebinning_with_rebin_size()
    test_num_bins_parameter()
    test_multiple_statistics()
    test_jackknife()

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
