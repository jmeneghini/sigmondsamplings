#!/usr/bin/env python3
"""
Example usage of the SigmondSamplings package.

This script demonstrates how to:
1. Load Sigmond samplings files (both fstream and HDF5)
2. Perform arithmetic operations on samplings
3. Calculate statistics and correlations
4. Create synthetic data for testing
5. Generate XML configurations for KB fits
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from SigmondSamplings import (
    SigmondLoader, SigmondSampling, SamplingStats, SamplingInfo, EnsembleInfo,
    create_gaussian_sampling, KBfitXMLHelper, BoxQuantizationInfo, 
    ParticleInfo, KElementInfo, FitParameterInfo, ObservableInfo,
    create_uniform_sampling, create_complex_gaussian_sampling, combine_real_imaginary, split_complex_sampling, DEFAULT_ENSEMBLE
)


def main():
    """Run all examples."""
    print("Sigmond Samplings Package - Example Usage")
    print("=" * 50)
    
    # Create shared objects for examples
    ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
    sampling_info = SamplingInfo("bootstrap", 5, 1234)
    
    # Example 1: Creating basic sampling
    print("=== Creating Basic Sampling ===")
    data = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02])  # full sample + resamples
    observable_info = ObservableInfo("test_obs", 0, "n", "re", ensemble_info)
    
    # Create sampling
    sampling = SigmondSampling(data, observable_info, sampling_info)
    
    print(f"Sampling: {sampling}")
    print(f"Full sample value: {sampling.full_sample_value}")
    print(f"Mean: {sampling.mean:.4f}")
    print(f"Error: {sampling.error:.4f}")
    print()
    
    # Example 2: Working with utility functions
    print("=== Using Utility Functions ===")
    
    # Create uniform sampling
    uniform_sampling = create_uniform_sampling(0.0, 2.0, 100, seed=1234)
    print(f"Uniform sampling: {uniform_sampling}")
    
    # Create complex Gaussian sampling
    complex_sampling_info = SamplingInfo("bootstrap", 100, 5678)
    complex_sampling = create_complex_gaussian_sampling(
        1.0, 0.1, 0.5, 0.05, complex_sampling_info
    )
    print(f"Complex sampling: {complex_sampling}")
    print()
    
    # Example 3: Arithmetic operations
    print("=== Arithmetic Operations ===")
    data1 = np.array([1.0, 1.1, 0.9, 1.05])
    data2 = np.array([2.0, 2.1, 1.9, 2.05])
    
    ensemble_info_arith = EnsembleInfo("test", 1000, 100)
    sampling_info_arith = SamplingInfo("bootstrap", 3, 1234)
    
    # Different observables should be compatible if sampling methods match
    obs_info1 = ObservableInfo("obs1", 0, "n", "re", ensemble_info_arith)
    obs_info2 = ObservableInfo("obs2", 1, "n", "re", ensemble_info_arith)
    
    sampling1 = SigmondSampling(data1, obs_info1, sampling_info_arith)
    sampling2 = SigmondSampling(data2, obs_info2, sampling_info_arith)
    
    # Operations
    sum_result = sampling1 + sampling2
    diff_result = sampling1 - sampling2
    prod_result = sampling1 * sampling2
    
    print(f"Sampling 1: {sampling1}")
    print(f"Sampling 2: {sampling2}")
    print(f"Sum: {sum_result}")
    print(f"Difference: {diff_result}")
    print(f"Product: {prod_result}")
    print()
    
    # Example 4: Complex number operations
    print("=== Complex Operations ===")
    real_data = np.array([1.0, 1.1, 0.9])
    imag_data = np.array([0.5, 0.55, 0.45])
    
    complex_ensemble_info = EnsembleInfo("complex_test", 1000, 100)
    real_obs_info = ObservableInfo("complex_obs", 0, "n", "re", complex_ensemble_info)
    imag_obs_info = ObservableInfo("complex_obs", 0, "n", "im", complex_ensemble_info)
    
    real_sampling = SigmondSampling(real_data, real_obs_info, 
                                   SamplingInfo("bootstrap", 2, 1234))
    imag_sampling = SigmondSampling(imag_data, imag_obs_info,
                                   SamplingInfo("bootstrap", 2, 1234))
    
    # Combine into complex
    combined = combine_real_imaginary(real_sampling, imag_sampling)
    print(f"Combined complex: {combined}")
    
    # Split back
    real_part, imag_part = split_complex_sampling(combined)
    print(f"Real part: {real_part}")
    print(f"Imaginary part: {imag_part}")
    print()
    
    # Example 5: Statistical analysis
    print("=== Statistical Analysis ===")
    samplings = [
        create_uniform_sampling(i, i+1, 50, seed=1234) for i in range(3)
    ]
    
    stats = SamplingStats(samplings)
    print(f"Number of observables: {stats.num_observables}")
    print(f"Sample means: {stats.sample_means()}")
    print(f"Sample errors: {stats.sample_errors()}")
    print()
    print("Correlation matrix:")
    print(stats.correlation_matrix())
    print()
    
    # Example 6: KBfitXMLHelper demonstrations
    print("=== KBfitXMLHelper Demonstrations ===")
    kb_xml_example(ensemble_info, sampling_info)
    
    # Example 7: New DEFAULT_ENSEMBLE and mixed operations
    print("=== DEFAULT_ENSEMBLE and Mixed Operations ===")
    demonstrate_mixed_operations()
    
    print("\n=== Example completed ===")


def kb_xml_example(ensemble_info, sampling_info):
    """Demonstrate KB XML generation capabilities."""
    print("Creating KB XML configurations for fits...")
    
    # =====================================
    # KB Fit XML Generation Examples
    # =====================================
    print("\n" + "="*60)
    print("KB Fit XML Generation Examples")
    print("="*60)
    
    # Create KB XML Helper
    kb_helper = KBfitXMLHelper()
    
    # Create observables with ensemble info
    obs1 = ObservableInfo("isosinglet_S=0_A1g_PSQ=0_elab_1_ref", 0, "n", "re", ensemble_info)
    obs2 = ObservableInfo("isosinglet_S=0_A1g_PSQ=1_elab_1_ref", 0, "n", "re", ensemble_info)
    obs3 = ObservableInfo("isosinglet_S=0_A1g_PSQ=2_elab_1_ref", 0, "n", "re", ensemble_info)
    observables = [obs1, obs2, obs3]
    
    # Create box quantizations
    box_quants = [
        BoxQuantizationInfo("0 0 0", 0, "A1g", "0"),
        BoxQuantizationInfo("1 0 0", 1, "A1g", "1"),
        BoxQuantizationInfo("1 1 0", 2, "A1g", "2")
    ]
    
    # Create particle info and other parameters
    decay_channels = [ParticleInfo("pion", 0, True)]
    particle_masses = {"pion": 0.14}
    sampling_files = ["samplings_file1.xml", "samplings_file2.xml"]
    
    # Example 1: Determinant Residual Fit XML (simplified)
    try:
        detres_xml = kb_helper.create_detres_xml(
            project_name="detres_fit",
            observables=observables,
            sampling_info=sampling_info,
            reference_particle="pion",
            particle_masses=particle_masses,
            box_quantizations=box_quants,
            sampling_files=sampling_files,
            default_energy_format="reference_ratio"
        )
        print("✓ Determinant residual XML generated successfully")
    except Exception as e:
        print(f"✗ Error generating determinant residual XML: {e}")
    
    # Example 2: Single Channel Fit XML (simplified)
    try:
        single_xml = kb_helper.create_single_channel_xml(
            project_name="single_channel_fit",
            observables=observables,
            sampling_info=sampling_info,
            reference_particle="pion",
            particle_masses=particle_masses,
            decay_channels=decay_channels,
            box_quantizations=box_quants,
            sampling_files=sampling_files,
            output_stub="single_channel_output",
            default_energy_format="time_spacing_product"
        )
        print("✓ Single channel XML generated successfully")
    except Exception as e:
        print(f"✗ Error generating single channel XML: {e}")
    
    # Example 3: Print Task XML (simplified)
    try:
        print_xml = kb_helper.create_print_xml(
            project_name="print_task",
            observables=[obs1],  # Single observable for print
            sampling_info=sampling_info,
            reference_particle="pion",
            particle_masses=particle_masses,
            energy_range=(0.1, 0.5, 0.01),  # min, max, step
            decay_channels=decay_channels,
            k_elements=[],  # Empty for simple example
            polynomial_powers={},  # Empty for simple example
            starting_values=[],  # Empty for simple example
            sampling_files=sampling_files,
            output_stub="print_output",
            default_energy_format="reference_ratio"
        )
        print("✓ Print task XML generated successfully")
    except Exception as e:
        print(f"✗ Error generating print task XML: {e}")
    
    print("\nXML generation examples completed successfully!")


def demonstrate_mixed_operations():
    """Demonstrate DEFAULT_ENSEMBLE usage and mixed observable operations."""
    print(f"Default ensemble: {DEFAULT_ENSEMBLE}")
    
    # Create samplings with different observables using DEFAULT_ENSEMBLE
    sampling_info = SamplingInfo("bootstrap", 100, 1234)
    
    # Using utility functions (now use DEFAULT_ENSEMBLE by default)
    mass_sampling = create_gaussian_sampling(1.5, 0.1, sampling_info, "mass")
    energy_sampling = create_gaussian_sampling(2.0, 0.15, sampling_info, "energy")
    
    print(f"Mass sampling observable: {mass_sampling.observable_info.name}")
    print(f"Energy sampling observable: {energy_sampling.observable_info.name}")
    
    # Arithmetic with different observables - should create 'mixed' result
    ratio = energy_sampling / mass_sampling
    print(f"Ratio observable: {ratio.observable_info.name}")
    print(f"Ratio ensemble: {ratio.observable_info.ensemble_info.ensemble_name}")
    
    # Arithmetic with same observables - should preserve observable
    mass_squared = mass_sampling * mass_sampling  
    print(f"Mass squared observable: {mass_squared.observable_info.name}")
    
    # Scalar operations - should preserve observable
    mass_plus_one = mass_sampling + 1.0
    print(f"Mass+1 observable: {mass_plus_one.observable_info.name}")
    
    print(f"Mass: {mass_sampling}")
    print(f"Energy: {energy_sampling}")
    print(f"Energy/Mass ratio: {ratio}")
    print(f"Mass squared: {mass_squared}")
    print(f"Mass + 1: {mass_plus_one}")


if __name__ == '__main__':
    main() 