"""
SigmondSamplings: A Python package for handling Sigmond samplings files.

This package provides tools to load, manipulate, and analyze Sigmond samplings files
in both fstream and HDF5 formats using the sigmond_query tool.
"""

from .loader import SigmondLoader
from .sampling import (
    EnsembleInfo, SamplingInfo, ObservableInfo, SigmondSampling, DEFAULT_ENSEMBLE
)
from .stats import SamplingStats
from .utils import *

__version__ = "0.1.0"
__all__ = [
    # Core SigmondSamplings functionality
    'SigmondLoader',
    'SigmondSampling', 
    'SamplingInfo', 
    'EnsembleInfo',
    'ObservableInfo',
    'DEFAULT_ENSEMBLE',
    'SamplingStats',
    
    # Utility functions
    'create_gaussian_sampling',
    'create_uniform_sampling', 
    'create_complex_gaussian_sampling',
    'combine_real_imaginary',
    'split_complex_sampling',
    'bootstrap_resample',
    'jackknife_resample',
    'effective_sample_size',
    'block_average',
]