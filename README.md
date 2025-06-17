# SigmondSamplings

A Python package for handling Sigmond samplings files, supporting both fstream and HDF5 formats with comprehensive statistical analysis capabilities and KB fit XML generation.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-116%20passed-brightgreen.svg)](tests/)

## Overview

SigmondSamplings is designed for lattice QCD researchers working with Monte Carlo sampling data. It provides a high-level Python interface to load, manipulate, and analyze Sigmond samplings files, with support for both bootstrap and jackknife resampling methods, plus powerful XML generation capabilities for KB fits.

### Key Features

- **File Format Support**: Load both fstream (`.smp`) and HDF5 (`.hdf5`) Sigmond files
- **Automatic Path Detection**: Intelligent HDF5 path discovery and error handling
- **Statistical Operations**: Bootstrap and jackknife error analysis with proper error propagation
- **Multi-Ensemble Support**: Handle samplings from different ensembles with proper covariance treatment
- **Arithmetic Operations**: Full support for sampling arithmetic with flexible compatibility checking
- **Complex Numbers**: Native support for complex-valued observables
- **Correlation Analysis**: Covariance and correlation matrix calculations with cross-ensemble handling
- **KB Fit XML Generation**: Comprehensive XML generation for KB fits (determinant residual, single channel, print tasks)
- **Synthetic Data**: Generate synthetic samplings for testing and validation
- **Comprehensive Testing**: 116 unit tests covering all functionality

## Installation

### From Source

```bash
git clone https://github.com/johndoe/SigmondSamplings.git
cd SigmondSamplings
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/johndoe/SigmondSamplings.git
cd SigmondSamplings
pip install -e ".[dev]"
```

### Requirements

- Python 3.8+
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- `sigmond_query` command-line tool (for loading Sigmond files)

## Quick Start

### Loading Sigmond Files

```python
from SigmondSamplings import SigmondLoader

# Initialize loader
loader = SigmondLoader()

# Load fstream file
ensemble_info, sampling_info, observables = loader.get_file_info('data.smp')

# Load HDF5 file (with automatic path detection)
is_valid, file_type, paths = loader.check_file_validity('data.hdf5')
if paths:
    file_with_path = f'data.hdf5[{paths[0]}]'
    ensemble_info, sampling_info, observables = loader.get_file_info(file_with_path)

# Load specific observable
sampling = loader.load_observable('data.smp', 'pion_mass')
print(f"Pion mass: {sampling.mean:.6f} ± {sampling.error:.6f}")
```

### Working with Samplings

```python
from SigmondSamplings import SigmondSampling, EnsembleInfo, SamplingInfo
import numpy as np

# Create sampling from data
data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])  # full sample + resamples
ensemble_info = EnsembleInfo("test_ensemble", 1000, 100)
sampling_info = SamplingInfo("bootstrap", 4, 1234)

sampling = SigmondSampling(data, ensemble_info, sampling_info)

# Access properties
print(f"Full sample value: {sampling.full_sample_value}")
print(f"Mean: {sampling.mean:.6f}")
print(f"Error: {sampling.error:.6f}")
print(f"Resampled values: {sampling.resampled_values}")
```

### Multi-Ensemble Statistical Analysis

```python
from SigmondSamplings import SamplingStats, create_uniform_sampling

# Create samplings from different ensembles (same sampling_info)
sampling_info = SamplingInfo("bootstrap", 100, 1234)
ensemble1 = EnsembleInfo("ensemble_a", 1000, 100)
ensemble2 = EnsembleInfo("ensemble_b", 2000, 100)

# These samplings are compatible despite different ensembles
sampling1 = create_uniform_sampling(0.0, 1.0, 100, seed=1234)
sampling2 = create_uniform_sampling(1.0, 2.0, 100, seed=1234)

# Multi-observable analysis
stats = SamplingStats([sampling1, sampling2])

# Covariance between different ensembles is zero
cov_matrix = stats.covariance_matrix()
print(f"Cross-ensemble covariance: {cov_matrix[0,1]}")  # Should be 0.0

# Standard statistical analysis
means = stats.sample_means()
errors = stats.sample_errors()
```

### KB Fit XML Generation

```python
from SigmondSamplings import (
    KBfitXMLHelper, BoxQuantizationInfo, ParticleInfo, 
    KElementInfo, FitParameterInfo
)

# Initialize XML helper
xml_helper = KBfitXMLHelper()

# Create ensemble and sampling info
ensemble_info = EnsembleInfo("cls21_s64_t128_D200", 10000, 500)
sampling_info = SamplingInfo("bootstrap", 1000, 1234)

# Define observables
observables = [
    "isotriplet_nucleon_T1u_mom0_p0_0_0_A1p_mom0_0",
    "isotriplet_nucleon_T1u_mom1_p1_0_0_A1p_mom0_0"
]

# Create reference particle info
reference_particle = ParticleInfo("nucleon", 2)  # spin-1/2

# Generate determinant residual fit XML
detres_xml = KBfitXMLHelper.create_detres_xml(
    observables,
    ensemble_info,
    t_min=3,
    t_max=15,
    energy_fit_info={"fit_form": "polynomial", "order": 2},
    amplitude_fit_info={"fit_form": "polynomial", "order": 1}
)

# Generate single channel fit XML
single_channel_xml = KBfitXMLHelper.create_single_channel_xml(
    observables,
    ensemble_info,
    reference_particle,
    t_min=3,
    t_max=15,
    num_exponentials=3
)

# Generate print task XML
print_xml = KBfitXMLHelper.create_print_xml(
    observables,
    ensemble_info,
    [reference_particle],
    print_correlations=True,
    print_effective_masses=True
)

# Save to file
with open("kb_fit_config.xml", "w") as f:
    f.write(detres_xml)
```

## API Reference

### Core Classes

#### `SigmondSampling`

Represents a single observable with its sampling data.

**Constructor:**
```python
SigmondSampling(data, ensemble_info, sampling_info, is_complex=False)
```

**Properties:**
- `full_sample_value`: Original (unsampled) value
- `resampled_values`: Array of resampled values
- `mean`: Sample mean
- `std`: Sample standard deviation  
- `error`: Statistical error (bootstrap or jackknife corrected)
- `ensemble_info`: Ensemble metadata
- `sampling_info`: Sampling method information
- `is_complex`: Whether the sampling contains complex data

**Arithmetic Operations:**
Samplings are compatible for arithmetic if they have the **same `sampling_info`** (different ensembles allowed):
- Addition: `sampling1 + sampling2` or `sampling + scalar`
- Subtraction: `sampling1 - sampling2` or `sampling - scalar`
- Multiplication: `sampling1 * sampling2` or `sampling * scalar`
- Division: `sampling1 / sampling2` or `sampling / scalar`

#### `SamplingStats`

Multi-observable statistical analysis with support for mixed ensembles.

**Constructor:**
```python
SamplingStats(samplings)  # List of SigmondSampling objects
```

**Methods:**
- `sample_means()` → `np.ndarray`: Mean values
- `sample_errors()` → `np.ndarray`: Statistical errors
- `covariance_matrix()` → `np.ndarray`: Covariance matrix (zero between different ensembles)
- `correlation_matrix()` → `np.ndarray`: Correlation matrix
- `covariance(i, j)` → `float`: Covariance between observables i and j
- `chi_squared(theory_values, use_correlation=True)` → `(chi_sq, dof)`
- `effective_sample_size()` → `np.ndarray`: Effective sample sizes
- `unique_ensembles` → `List[EnsembleInfo]`: List of unique ensembles in the dataset

#### `KBfitXMLHelper`

XML generation for KB fits with comprehensive configuration options.

**Key Methods:**
- `create_detres_xml()`: Generate determinant residual fit XML
- `create_single_channel_xml()`: Generate single channel fit XML  
- `create_print_xml()`: Generate print task XML
- `extract_momentum_info_from_observable()`: Parse momentum info from observable names
- `group_observables_by_momentum()`: Group observables by momentum quantum numbers
- `create_box_quantization_from_momentum()`: Create box quantization from momentum info

#### `SigmondLoader`

Main interface for loading Sigmond files.

**Methods:**
- `check_file_validity(filepath)` → `(is_valid, file_type, hdf5_paths)`
- `get_file_info(filepath)` → `(ensemble_info, sampling_info, observable_infos)`
- `load_observable(filepath, observable_name)` → `SigmondSampling`
- `load_all_observables(filepath)` → `List[SigmondSampling]`
- `find_observables(filepath, pattern)` → `List[ObservableInfo]`

### Information Classes

#### `EnsembleInfo`
- `ensemble_name`: Ensemble identifier
- `num_measurements`: Number of measurements
- `num_bins`: Number of bins
- `tweak_info`: Dictionary of ensemble tweaks

#### `SamplingInfo`
- `method`: "bootstrap" or "jackknife"
- `num_resamplings`: Number of resamples
- `seed`: Random seed (bootstrap only)
- `boot_skip`: Bootstrap skip parameter (bootstrap only)

#### `ObservableInfo`
- `name`: Observable name
- `index`: Observable index
- `op_type`: Operator type
- `re_im`: Real/imaginary/complex type ("re", "im", "cx")

#### `BoxQuantizationInfo`
- `momentum_ray`: Momentum ray type ("ar", "oa", "pd", "cd")
- `momentum_int_squared`: Momentum squared integer
- `lg_irrep`: Little group irreducible representation
- `lmax_values`: Maximum angular momentum values

#### `ParticleInfo`
- `name`: Particle name
- `spin_times_two`: Particle spin × 2 (integer)
- `identical`: Whether particles are identical

#### `KElementInfo`
- `j_times_two`: Total angular momentum × 2
- `k_index1`: First K-matrix index
- `k_index2`: Second K-matrix index

### Utility Functions

#### Synthetic Data Generation
```python
from SigmondSamplings.utils import (
    create_gaussian_sampling,
    create_uniform_sampling,
    create_complex_gaussian_sampling
)

# Create sampling info first
sampling_info = SamplingInfo("bootstrap", 1000, 1234)

# Real-valued Gaussian sampling
sampling = create_gaussian_sampling(
    mean=1.0, std=0.1, 
    sampling_info=sampling_info,
    observable_name="test_obs"
)

# Complex-valued Gaussian sampling
complex_sampling = create_complex_gaussian_sampling(
    mean_real=1.0, std_real=0.1,
    mean_imag=0.5, std_imag=0.05,
    sampling_info=sampling_info
)

# Uniform sampling (simpler interface)
uniform_sampling = create_uniform_sampling(
    low=0.0, high=1.0, num_samples=1000, seed=1234
)
```

#### Complex Number Utilities
```python
from SigmondSamplings.utils import combine_real_imaginary, split_complex_sampling

# Combine real and imaginary samplings
complex_sampling = combine_real_imaginary(real_sampling, imag_sampling)

# Split complex sampling
real_part, imag_part = split_complex_sampling(complex_sampling)
```

## Compatibility Rules

### Sampling Compatibility

Two samplings are compatible for arithmetic operations if they have:
1. **Same `sampling_info`** (method, number of resamplings, seed)
2. **Same data length**
3. **Different `ensemble_info` is allowed**

```python
# These are compatible despite different ensembles
ensemble1 = EnsembleInfo("ensemble_a", 1000, 100)
ensemble2 = EnsembleInfo("ensemble_b", 2000, 100)
sampling_info = SamplingInfo("bootstrap", 100, 1234)

sampling1 = SigmondSampling(data1, ensemble1, sampling_info)
sampling2 = SigmondSampling(data2, ensemble2, sampling_info)

result = sampling1 + sampling2  # This works!
```

### Covariance Rules

- **Same ensemble**: Normal covariance calculation
- **Different ensembles**: Covariance = 0.0
- **Diagonal elements**: Always return variance (error²)

```python
stats = SamplingStats([sampling1, sampling2])
cov_matrix = stats.covariance_matrix()
# cov_matrix[0,1] = 0.0 (different ensembles)
# cov_matrix[0,0] = variance of sampling 0
```

## KB Fit XML Generation

### Observable Momentum Parsing

The package can automatically extract momentum information from observable names:

```python
obs_name = "isosinglet_S=0_A1g_1_PSQ=0_elab_2_ref"
momentum_info = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
if momentum_info:
    psq, irrep = momentum_info
    print(f"PSQ={psq}, irrep={irrep}")
```

### XML Generation Examples

```python
# Determinant residual fit
detres_xml = KBfitXMLHelper.create_detres_xml(
    observables=observable_list,
    ensemble_info=ensemble_info,
    t_min=3, t_max=15,
    energy_fit_info={"fit_form": "polynomial", "order": 2},
    amplitude_fit_info={"fit_form": "string", "expression": "A0*exp(-E0*t)"}
)

# Single channel fit
single_xml = KBfitXMLHelper.create_single_channel_xml(
    observables=observable_list,
    ensemble_info=ensemble_info,
    reference_particle=ParticleInfo("nucleon", 2),
    t_min=3, t_max=15,
    num_exponentials=3
)

# Print task with correlations
print_xml = KBfitXMLHelper.create_print_xml(
    observables=observable_list,
    ensemble_info=ensemble_info,
    reference_particles=[ParticleInfo("nucleon", 2)],
    print_correlations=True,
    print_effective_masses=True,
    print_energy_differences=True
)
```

## File Format Support

### Fstream Files (`.smp`)

Traditional Sigmond binary format. Files are loaded directly:

```python
loader = SigmondLoader()
ensemble_info, sampling_info, observables = loader.get_file_info('data.smp')
```

### HDF5 Files (`.hdf5`)

Modern HDF5 format with hierarchical structure. Requires path specification:

```python
# Check available paths
is_valid, file_type, paths = loader.check_file_validity('data.hdf5')
print(f"Available paths: {paths}")

# Load with specific path
file_with_path = f'data.hdf5[{paths[0]}]'
ensemble_info, sampling_info, observables = loader.get_file_info(file_with_path)
```

## Statistical Methods

### Bootstrap vs Jackknife

The package automatically handles the differences between bootstrap and jackknife methods:

- **Bootstrap**: Error = std(resampled_values)
- **Jackknife**: Error = sqrt(N-1) × std(resampled_values)

### Multi-Ensemble Analysis

```python
# Mix samplings from different ensembles
ensemble_a_samplings = [create_uniform_sampling(i, i+1, 100, seed=1234) for i in range(3)]
ensemble_b_samplings = [create_uniform_sampling(i+10, i+11, 100, seed=1234) for i in range(2)]

# Combine for analysis
all_samplings = ensemble_a_samplings + ensemble_b_samplings
stats = SamplingStats(all_samplings)

# Cross-ensemble covariances are automatically zero
cov_matrix = stats.covariance_matrix()
print(f"Number of unique ensembles: {len(stats.unique_ensembles)}")
```

## Examples

### Complete Analysis Workflow

```python
from SigmondSamplings import SigmondLoader, SamplingStats, create_gaussian_sampling, SamplingInfo
import numpy as np

# Load data
loader = SigmondLoader()
pion = loader.load_observable('hadrons.smp', 'pion_mass')
kaon = loader.load_observable('hadrons.smp', 'kaon_mass')

# Calculate derived quantities
mass_ratio = kaon / pion
mass_difference = kaon - pion

# Add synthetic data from different ensemble
sampling_info = SamplingInfo("bootstrap", 100, 5678)
synthetic_obs = create_gaussian_sampling(0.3, 0.02, sampling_info, "eta_mass")

# Multi-ensemble statistical analysis
stats = SamplingStats([pion, kaon, mass_ratio, synthetic_obs])
print(f"Analyzing {stats.num_observables} observables from {len(stats.unique_ensembles)} ensembles")
print(stats.summary())

# Covariance matrix shows zero cross-ensemble correlations
cov_matrix = stats.covariance_matrix()
print("Covariance matrix shape:", cov_matrix.shape)
```

### KB Fit XML Workflow

```python
from SigmondSamplings import KBfitXMLHelper, EnsembleInfo, ParticleInfo

# Setup
ensemble_info = EnsembleInfo("cls21_s64_t128_D200", 10000, 500)
observables = [
    "isotriplet_nucleon_T1u_mom0_p0_0_0_A1p_mom0_0",
    "isotriplet_nucleon_T1u_mom1_p1_0_0_A1p_mom0_0",
    "isotriplet_nucleon_T1u_mom2_p2_0_0_A1p_mom0_0"
]

# Group by momentum
grouped = KBfitXMLHelper.group_observables_by_momentum(observables)
print(f"Momentum groups: {list(grouped.keys())}")

# Generate XML configurations
helper = KBfitXMLHelper()

# For determinant residual fits
detres_xml = helper.create_detres_xml(
    observables, ensemble_info,
    t_min=3, t_max=15,
    energy_fit_info={"fit_form": "polynomial", "order": 2}
)

# Save to files
with open("detres_fit.xml", "w") as f:
    f.write(detres_xml)

print("KB fit XML files generated successfully!")
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test module
pytest tests/test_sampling.py -v

# KB XML helper tests
pytest tests/test_kb_xml_helper.py -v

# With coverage
pytest tests/ --cov=SigmondSamplings --cov-report=html
```

**Test Coverage:**
- `test_sampling.py`: 40 tests - Core sampling functionality
- `test_loader.py`: 16 tests - File loading and parsing
- `test_stats.py`: 14 tests - Statistical analysis
- `test_utils.py`: 20 tests - Utility functions
- `test_kb_xml_helper.py`: 26 tests - KB XML generation

## Performance Notes

- **Memory Usage**: Samplings store full resampling arrays in memory
- **Multi-Ensemble**: Different ensembles are handled efficiently with zero covariance
- **File Loading**: HDF5 files require `sigmond_query` subprocess calls
- **Arithmetic**: Operations create new sampling objects (not in-place)
- **XML Generation**: Large XML files may require significant memory

## Troubleshooting

### Common Issues

1. **`sigmond_query` not found**
   ```bash
   # Ensure sigmond_query is in PATH
   which sigmond_query
   ```

2. **Incompatible samplings**
   ```python
   # Check sampling_info compatibility (ensembles can differ)
   print(f"Sampling info 1: {sampling1.sampling_info}")
   print(f"Sampling info 2: {sampling2.sampling_info}")
   ```

3. **Zero covariances**
   ```python
   # This is expected for different ensembles
   stats = SamplingStats([sampling1, sampling2])
   if sampling1.ensemble_info != sampling2.ensemble_info:
       print("Zero covariance expected - different ensembles")
   ```

4. **XML generation issues**
   ```python
   # Ensure observables have proper momentum information
   obs_name = "isotriplet_nucleon_T1u_mom0_p0_0_0_A1p_mom0_0"
   momentum_info = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
   if not momentum_info:
       print(f"Cannot parse momentum from: {obs_name}")
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run the test suite (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/johndoe/SigmondSamplings.git
cd SigmondSamplings
pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SigmondSamplings in your research, please cite:

```bibtex
@software{sigmondsampling2024,
  title={SigmondSamplings: A Python Package for Lattice QCD Sampling Analysis},
  author={John Doe},
  year={2024},
  url={https://github.com/johndoe/SigmondSamplings}
}
```

## Acknowledgments

- The Sigmond collaboration for the original file formats
- The lattice QCD community for feedback and testing
- Contributors and maintainers 