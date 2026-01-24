"""
sigmondsamplings: A Python package for handling Sigmond samplings files.

This package provides tools to load, manipulate, and analyze Sigmond samplings files
in both fstream and HDF5 formats using the sigmond_query tool.
"""

from .loader import SigmondLoader
from .obervable_collection import ObservableCollection
from .ensemble_collection import (
    SingleEnsembleCollection,
    MultiEnsembleCollection,
    group_by_ensemble_and_sampling,
)
from .energy_level_collection import (
    EnergyLevelMixin,
    SingleEnsembleEnergyCollection,
    MultiEnsembleEnergyCollection,
)
from .writer import SigmondWriter
from .plotter import SamplingPlotter
from .model_func import SigmondModelFunc
from .info import (
    KnownEnsembles,
    EnsembleInfo,
    SamplingInfo,
    ObservableInfo,
    SectorInfo,
    DEFAULT_ENSEMBLE,
)
from .sampling import SigmondSampling
from .energy_levels import (
    Particle,
    EnergyObsInfo,
    SHEnergyObsInfo,
    create_energy_obs_info,
)
from .stats import SamplingStats
from .utils import *
from .pycalq_loader import (
    PyCALQLoader,
    PyCALQSamplingResultType,
    PyCALQEstimateResultType,
    PyCALQRotateInfo,
    PyCALQPivotType,
    PyCALQPaths,
)
from .project_utils import (
    OSInfo,
    LinuxDistro,
    get_os_info,
    string_of_list_to_list,
    get_gamma_from_elab_and_ecm,
    get_g_ref_from_Gamma_ref,
    get_Gamma_ref_from_g_ref,
    find_files_with_pattern,
    extract_numeric_values_from_filename,
    get_momentum_squared_from_momentum,
    get_momentum_from_momentum_squared,
    group_observables_by_momentum,
)

__version__ = "0.1.0"
__all__ = [
    # Core SigmondSamplings functionality
    "SigmondLoader",
    "ObservableCollection",
    "SingleEnsembleCollection",
    "MultiEnsembleCollection",
    "group_by_ensemble_and_sampling",
    # Energy-level collections
    "EnergyLevelMixin",
    "SingleEnsembleEnergyCollection",
    "MultiEnsembleEnergyCollection",
    "SigmondWriter",
    "SigmondSampling",
    "KnownEnsembles",
    "SamplingInfo",
    "EnsembleInfo",
    "ObservableInfo",
    "SectorInfo",
    "DEFAULT_ENSEMBLE",
    "SamplingStats",
    # Energy level functionality
    "Particle",
    "EnergyObsInfo",
    "SHEnergyObsInfo",
    "create_energy_obs_info",
    # PyCALQ functionality
    "PyCALQLoader",
    "PyCALQSamplingResultType",
    "PyCALQEstimateResultType",
    "PyCALQRotateInfo",
    "PyCALQPivotType",
    "PyCALQPaths",
    # Project utilities
    "OSInfo",
    "LinuxDistro",
    "get_os_info",
    "string_of_list_to_list",
    "get_gamma_from_elab_and_ecm",
    "get_g_ref_from_Gamma_ref",
    "get_Gamma_ref_from_g_ref",
    "find_files_with_pattern",
    "extract_numeric_values_from_filename",
    "get_momentum_squared_from_momentum",
    "get_momentum_from_momentum_squared",
    "group_observables_by_momentum",
    # Utility functions
    "create_gaussian_sampling",
    "create_uniform_sampling",
    "create_complex_gaussian_sampling",
    "combine_real_imaginary",
    "split_complex_sampling",
    "bootstrap_resample",
    "jackknife_resample",
    "compute_autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "rebin_data",
]
