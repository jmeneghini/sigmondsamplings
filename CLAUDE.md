# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SigmondSamplings is a comprehensive Python package for handling Sigmond samplings files in lattice QCD research. It supports both fstream (.smp) and HDF5 (.hdf5) formats, provides statistical analysis capabilities (bootstrap/jackknife), and generates XML configurations for KB fits.

## Development Commands

### Installation
```bash
# Install in editable mode for development
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests (if pytest is available)
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_sampling.py -v
python -m pytest tests/test_loader.py -v
python -m pytest tests/test_stats.py -v
python -m pytest tests/test_kb_xml_helper.py -v

# Run with coverage (if pytest-cov is available)
python -m pytest tests/ --cov=SigmondSamplings --cov-report=html

# Run individual test modules directly
python -m unittest tests.test_sampling -v
python -m unittest tests.test_loader -v
```

### Building
```bash
# Build the package
python -m build

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Architecture

### Core Components

- **`SigmondSamplings/sampling.py`**: Core data structures
  - `SigmondSampling`: Main class for sampling data with statistical methods
  - `EnsembleInfo`: Monte Carlo ensemble metadata
  - `SamplingInfo`: Bootstrap/jackknife sampling parameters
  - `ObservableInfo`: Observable metadata

- **`SigmondSamplings/loader.py`**: File I/O operations
  - `SigmondLoader`: Main interface for loading Sigmond files
  - Supports both fstream (.smp) and HDF5 (.hdf5) formats
  - Uses external `sigmond_query` command-line tool

- **`SigmondSamplings/stats.py`**: Multi-observable statistical analysis
  - `SamplingStats`: Covariance analysis, chi-squared fits
  - Handles mixed-ensemble datasets with proper covariance treatment

- **`SigmondSamplings/kb_xml_helper.py`**: KB fit XML generation
  - `KBfitXMLHelper`: Generate XML for determinant residual, spectrum, and print tasks
  - Comprehensive configuration classes for KB fitting parameters

- **`SigmondSamplings/utils.py`**: Utility functions
  - Synthetic data generation (Gaussian, uniform distributions)
  - Complex number handling
  - Bootstrap/jackknife resampling utilities

### Key Design Patterns

1. **Multi-Ensemble Support**: Different ensembles can be analyzed together with automatic zero covariance between them
2. **Sampling Compatibility**: Samplings are compatible for arithmetic if they have the same `sampling_info` (different ensembles allowed)
3. **Error Propagation**: Automatic bootstrap/jackknife error correction
4. **XML Configuration**: Comprehensive KB fit XML generation with validation

### Dependencies

- **Core**: numpy ≥1.20.0, scipy ≥1.7.0
- **External**: `sigmond_query` command-line tool (for file loading)
- **Dev**: pytest ≥6.0, pytest-cov ≥2.0
- **Python**: ≥3.8

### Data Flow

1. **Loading**: `SigmondLoader` → uses `sigmond_query` → parses XML output → creates `SigmondSampling` objects
2. **Analysis**: `SamplingStats` → handles multiple samplings → computes covariances/correlations
3. **KB Fitting**: `KBfitXMLHelper` → parses observables → generates XML configurations

## Cross-Repository Refactoring Opportunities

### Shared Components with KBAnalysis

1. **Data Visualization**: Both projects handle lattice QCD data plotting
   - SigmondSamplings: Statistical plots, error bars, correlation visualization
   - KBAnalysis: Energy level plots, interactive matplotlib windows
   - **Potential**: Shared plotting utilities with consistent styling

2. **CSV/Data Processing**: 
   - SigmondSamplings: Processes Sigmond output data
   - KBAnalysis: Processes KBfit CSV output
   - **Potential**: Common data processing utilities, consistent file I/O patterns

3. **Statistical Analysis**:
   - SigmondSamplings: Bootstrap/jackknife analysis, covariance matrices
   - KBAnalysis: Could benefit from statistical error analysis
   - **Potential**: Shared statistical utilities package

4. **Configuration Management**:
   - SigmondSamplings: Complex XML generation with many parameters
   - KBAnalysis: Simple configuration but could be expanded
   - **Potential**: Shared configuration validation and management

5. **Testing Infrastructure**:
   - Both projects use pytest with fixture-based testing
   - **Potential**: Shared test utilities for lattice QCD data generation

### Integration Opportunities

1. **Analysis Pipeline**: SigmondSamplings → generates KB fit configurations → KBAnalysis analyzes results
2. **Data Exchange**: Common data structures for energy levels, observables
3. **Plotting Integration**: Combine SigmondSamplings statistical plots with KBAnalysis interactive features

### Suggested Refactoring Structure
```
lattice_qcd_utils/
├── plotting/          # Shared plotting utilities
├── statistics/        # Bootstrap/jackknife, error propagation  
├── data_io/          # Common file I/O patterns
├── config/           # Configuration management
└── testing/          # Shared test utilities
```

## Testing Strategy

- **Unit Tests**: 116 tests across 5 modules
- **Test Coverage**: Comprehensive coverage of core functionality
- **Test Structure**: Uses unittest framework with fixtures for mock data
- **Integration Tests**: Tests interaction with `sigmond_query` tool

## External Dependencies

- **sigmond_query**: Required command-line tool for Sigmond file parsing
- **File Formats**: Supports both legacy fstream and modern HDF5 Sigmond formats
- **XML Processing**: Uses Python's built-in xml.etree.ElementTree for KB fit XML generation