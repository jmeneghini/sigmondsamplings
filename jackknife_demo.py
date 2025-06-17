#!/usr/bin/env python3
"""
Demonstration of jackknife functionality with real Sigmond HDF5 files.

This script shows:
1. Loading jackknife HDF5 files
2. Comparing bootstrap vs jackknife error calculations
3. Statistical analysis with jackknife data
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from SigmondSamplings import (
    SigmondLoader, SamplingStats, 
    create_gaussian_sampling
)
from SigmondSamplings.utils import create_complex_gaussian_sampling


def main():
    print("=== Jackknife vs Bootstrap Demonstration ===\n")
    
    loader = SigmondLoader()
    
    # 1. Load bootstrap HDF5 file
    print("1. Loading Bootstrap HDF5 file...")
    bootstrap_file = 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5[/samplings/]'
    
    try:
        bootstrap_ensemble, bootstrap_sampling, bootstrap_obs = loader.get_file_info(bootstrap_file)
        print(f"   Ensemble: {bootstrap_ensemble.ensemble_name}")
        print(f"   Method: {bootstrap_sampling.method}")
        print(f"   Resamplings: {bootstrap_sampling.num_resamplings}")
        print(f"   Seed: {bootstrap_sampling.seed}")
        print(f"   Observables: {len(bootstrap_obs)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Load jackknife HDF5 file
    print("\n2. Loading Jackknife HDF5 file...")
    jackknife_file = 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_J-samplings.hdf5[/samplings/]'
    
    try:
        jackknife_ensemble, jackknife_sampling, jackknife_obs = loader.get_file_info(jackknife_file)
        print(f"   Ensemble: {jackknife_ensemble.ensemble_name}")
        print(f"   Method: {jackknife_sampling.method}")
        print(f"   Resamplings: {jackknife_sampling.num_resamplings}")
        print(f"   Observables: {len(jackknife_obs)}")
        
        # Note: For jackknife, num_resamplings should equal num_bins
        print(f"   Bins: {jackknife_ensemble.num_bins}")
        print(f"   Resamplings == Bins: {jackknife_sampling.num_resamplings == jackknife_ensemble.num_bins}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Compare synthetic bootstrap vs jackknife
    print("\n3. Synthetic Bootstrap vs Jackknife Comparison...")
    
    # Create synthetic data with same parameters
    np.random.seed(1234)
    true_mean = 1.0
    true_std = 0.1
    n_samples = 1000
    
    # Bootstrap sampling
    bootstrap_synthetic = create_gaussian_sampling(
        mean=true_mean, std=true_std, num_samples=n_samples,
        observable_name="test_obs", seed=1234
    )
    
    # Create jackknife sampling manually
    from SigmondSamplings.sampling import SigmondSampling, SamplingInfo, EnsembleInfo, ObservableInfo
    from SigmondSamplings.utils import jackknife_resample
    
    # Generate base data
    np.random.seed(1234)
    base_data = np.random.normal(true_mean, true_std, n_samples)
    
    # Create jackknife resamples
    jackknife_samples = jackknife_resample(base_data)
    
    # Create info objects
    ensemble_info = EnsembleInfo("test_ensemble", n_samples, n_samples)
    jackknife_info = SamplingInfo("jackknife", len(jackknife_samples))
    obs_info = ObservableInfo("test_obs", 0, "test", "real")
    
    # Combine full sample + jackknife samples
    full_data = np.concatenate([[base_data.mean()], jackknife_samples])
    
    jackknife_synthetic = SigmondSampling(
        full_data, obs_info, ensemble_info, jackknife_info
    )
    
    print(f"   Bootstrap: {bootstrap_synthetic}")
    print(f"   Jackknife: {jackknife_synthetic}")
    print(f"   Bootstrap error: {bootstrap_synthetic.error:.6f}")
    print(f"   Jackknife error: {jackknife_synthetic.error:.6f}")
    print(f"   Error ratio (J/B): {jackknife_synthetic.error / bootstrap_synthetic.error:.3f}")
    
    # 4. Statistical analysis comparison
    print("\n4. Statistical Analysis Comparison...")
    
    try:
        # Create multiple synthetic samplings for stats
        bootstrap_samplings = []
        jackknife_samplings = []
        
        for i in range(3):
            # Bootstrap
            bs = create_gaussian_sampling(
                mean=1.0 + i * 0.2, std=0.1, num_samples=500,
                observable_name=f"obs_{i}", seed=1234 + i
            )
            bootstrap_samplings.append(bs)
            
            # Jackknife equivalent
            np.random.seed(1234 + i)
            base = np.random.normal(1.0 + i * 0.2, 0.1, 500)
            jk_samples = jackknife_resample(base)
            jk_data = np.concatenate([[base.mean()], jk_samples])
            
            jk = SigmondSampling(
                jk_data, 
                ObservableInfo(f"obs_{i}", i, "test", "real"),
                EnsembleInfo(f"ensemble_{i}", 500, 500),
                SamplingInfo("jackknife", len(jk_samples))
            )
            jackknife_samplings.append(jk)
        
        # Statistical analysis
        bootstrap_stats = SamplingStats(bootstrap_samplings)
        jackknife_stats = SamplingStats(jackknife_samplings)
        
        print(f"   Bootstrap means: {bootstrap_stats.means()}")
        print(f"   Jackknife means: {jackknife_stats.means()}")
        print(f"   Bootstrap errors: {bootstrap_stats.errors()}")
        print(f"   Jackknife errors: {jackknife_stats.errors()}")
        
        # Correlation matrices
        bs_corr = bootstrap_stats.correlation_matrix()
        jk_corr = jackknife_stats.correlation_matrix()
        
        print(f"   Bootstrap correlation matrix diagonal: {np.diag(bs_corr)}")
        print(f"   Jackknife correlation matrix diagonal: {np.diag(jk_corr)}")
        
    except Exception as e:
        print(f"   Stats comparison error: {e}")
    
    # 5. Complex jackknife demonstration
    print("\n5. Complex Jackknife Demonstration...")
    
    try:
        # Create complex jackknife sampling
        np.random.seed(5678)
        n_complex = 200
        real_data = np.random.normal(1.0, 0.1, n_complex)
        imag_data = np.random.normal(0.5, 0.05, n_complex)
        complex_data = real_data + 1j * imag_data
        
        # Jackknife resample complex data
        complex_jk_samples = jackknife_resample(complex_data)
        complex_jk_full = np.concatenate([[complex_data.mean()], complex_jk_samples])
        
        complex_jk = SigmondSampling(
            complex_jk_full,
            ObservableInfo("complex_obs", 0, "test", "complex"),
            EnsembleInfo("complex_ensemble", n_complex, n_complex),
            SamplingInfo("jackknife", len(complex_jk_samples)),
            is_complex=True
        )
        
        print(f"   Complex jackknife: {complex_jk}")
        print(f"   Real part: {np.real(complex_jk.mean):.6f} ± {np.real(complex_jk.error):.6f}")
        print(f"   Imaginary part: {np.imag(complex_jk.mean):.6f} ± {np.imag(complex_jk.error):.6f}")
        
    except Exception as e:
        print(f"   Complex jackknife error: {e}")
    
    print("\n=== Jackknife demonstration completed ===")


if __name__ == '__main__':
    main() 