# sigmondsamplings

A Python package for handling Sigmond samplings files with comprehensive statistical analysis capabilities for lattice QCD research.

## Overview

SigmondSamplings provides a high-level Python interface to load, manipulate, and analyze Sigmond samplings files. It supports both bootstrap and jackknife resampling methods with robust error propagation and multi-ensemble analysis capabilities.

## Key Features

- **File Format Support**: Load both fstream and HDF5 (`.hdf5`) Sigmond files
- **Statistical Operations**: Bootstrap and jackknife error analysis with proper error propagation
- **Multi-Ensemble Support**: Handle samplings from different ensembles with automatic covariance treatment
- **Arithmetic Operations**: Fast and full support for sampling arithmetic via numpy, with observable compatibility checking
- **Complex Numbers**: Native support for complex-valued observables
- **Correlation Analysis**: Covariance and correlation matrix calculations
- **Synthetic Data**: Generate synthetic samplings for testing and validation

## Installation

### From Source

```bash
git clone <repository-url>
cd sigmondsamplings
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.8+
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- `sigmond_query` command-line tool (see [Sigmond](https://github.com/jmeneghini/sigmond/tree/pip))

## Quick Start

```python
from SigmondSamplings import SigmondLoader, SamplingStats
import numpy as np

# Load Sigmond files
loader = SigmondLoader()
file_name = 'data.smp' # fstream format
file_name = 'data.hdf5[\path\to\samplings]' # HDF5 format (tag must be specified in filename)
ensemble_info, sampling_info, observables = loader.get_file_info('data.smp')

# Load specific observable
pion_mass = loader.load_observable('data.smp', 'pion_mass')
print(f"Pion mass: {pion_mass.mean:.6f} ± {pion_mass.error:.6f}")

# Multi-observable analysis
kaon_mass = loader.load_observable('data.smp', 'kaon_mass')
stats = SamplingStats([pion_mass, kaon_mass])

# Calculate correlations
corr_matrix = stats.correlation_matrix()
print("Correlation matrix:", corr_matrix)
```

## Core Classes

### SigmondSampling

Main class for individual observables with statistical methods:

```python
# Access properties
print(f"Full sample: {sampling.full_sample_value}")
print(f"Mean: {sampling.mean:.6f}")
print(f"Error: {sampling.error:.6f}")

# Arithmetic operations
mass_ratio = kaon_mass / pion_mass
mass_difference = kaon_mass - pion_mass
```

### SamplingStats

Multi-observable statistical analysis:

```python
stats = SamplingStats([pion_mass, kaon_mass, mass_ratio])

# Statistical analysis
means = stats.sample_means()
errors = stats.sample_errors() 
cov_matrix = stats.covariance_matrix()

# Chi-squared analysis
chi_sq, dof = stats.chi_squared(theory_values)
```

### SigmondLoader

Main interface for loading Sigmond files:

```python
loader = SigmondLoader()

# Check file validity
is_valid, file_type, paths = loader.check_file_validity('data.hdf5')

# Load all observables
all_samplings = loader.load_all_observables('data.smp')

# Find observables by pattern
pion_observables = loader.find_observables('data.smp', '*pion*')
```

## File Format Support

### Fstream Files (`.smp`)
Traditional Sigmond binary format - loaded directly.

### HDF5 Files (`.hdf5`)
Modern hierarchical format - requires path specification:

```python
# Auto-detect available paths
is_valid, file_type, paths = loader.check_file_validity('data.hdf5')
print(f"Available paths: {paths}")

# Load with specific path
file_with_path = f'data.hdf5[{paths[0]}]'
sampling = loader.load_observable(file_with_path, 'observable_name')
```

## Sampling Compatibility

Samplings are compatible for arithmetic operations if they have:
1. **Same `sampling_info`** (method, number of resamplings, seed)
2. **Same data length**
3. **Different `ensemble_info` is allowed** (enables multi-ensemble analysis)

```python
# Compatible despite different ensembles
ensemble1 = EnsembleInfo("ensemble_a", 1000, 100)
ensemble2 = EnsembleInfo("ensemble_b", 2000, 100)
sampling_info = SamplingInfo("bootstrap", 100, 1234)

sampling1 = SigmondSampling(data1, ensemble1, sampling_info)
sampling2 = SigmondSampling(data2, ensemble2, sampling_info)

result = sampling1 + sampling2  # This works!
```

## Multi-Ensemble Analysis

SigmondSamplings automatically handles covariance between different ensembles:
- **Same ensemble**: Normal covariance calculation
- **Different ensembles**: Covariance = 0.0 (automatic)
- **Diagonal elements**: Always return variance (error²)

## Utility Functions

Generate synthetic data for testing:

```python
from SigmondSamplings import (
    create_gaussian_sampling, create_uniform_sampling, 
    create_complex_gaussian_sampling
)

# Create synthetic sampling
sampling_info = SamplingInfo("bootstrap", 1000, 1234)
synthetic = create_gaussian_sampling(
    mean=1.0, std=0.1, 
    sampling_info=sampling_info,
    observable_name="test_observable"
)
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_sampling.py -v

# Run with coverage
python -m pytest tests/ --cov=SigmondSamplings --cov-report=html
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
