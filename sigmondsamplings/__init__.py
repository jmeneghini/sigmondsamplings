"""
sigmondsamplings: A Python package for handling Sigmond samplings files.

This package provides tools to load, manipulate, and analyze Sigmond samplings files
in both fstream and HDF5 formats using the sigmond_query tool.
"""

from .bins import SigmondBins
from .energy_level_collection import (
    EnergyLevelMixin,
    MultiEnsembleEnergyCollection,
    SingleEnsembleEnergyCollection,
)
from .energy_levels import (
    EnergyObsInfo,
    Particle,
    SHEnergyObsInfo,
    create_energy_obs_info,
)
from .ensemble_collection import (
    MultiEnsembleCollection,
    SingleEnsembleCollection,
    group_by_ensemble_and_sampling,
)
from .fit import (
    SamplingFit,
    SamplingFitResult,
    default_num_workers,
    make_process_pool,
    set_thread_counts,
)
from .info import (
    DEFAULT_ENSEMBLE,
    EnsembleInfo,
    KnownEnsembles,
    ObservableInfo,
    SamplingInfo,
    SectorInfo,
)
from .loader import SigmondLoader
from .model_func import SigmondModelFunc as SigmondModelFunc
from .obervable_collection import ObservableCollection
from .plotter import SamplingPlotter as SamplingPlotter
from .project_utils import (
    LinuxDistro,
    OSInfo,
    extract_numeric_values_from_filename,
    find_files_with_pattern,
    get_g_ref_from_Gamma_ref,
    get_gamma_from_elab_and_ecm,
    get_Gamma_ref_from_g_ref,
    get_momentum_from_momentum_squared,
    get_momentum_squared_from_momentum,
    get_os_info,
    group_observables_by_momentum,
    string_of_list_to_list,
)
from .pycalq_loader import (
    PyCALQEstimateResultType,
    PyCALQLoader,
    PyCALQPaths,
    PyCALQPivotType,
    PyCALQRotateInfo,
    PyCALQSamplingResultType,
)
from .sampling import SigmondSampling
from .sampling_array import (
    ArrayElementObsInfo,
    ArrayObsInfo,
    AxisMeta,
    SigmondSamplingArray,
)
from .spectrum_plotter import (
    HMarker,
    SectorSpectrumPlotter,
    SpectrumPlotter,
    SpectrumStyle,
)
from .stats import SamplingStats
from .utils import (
    bootstrap_resample,
    combine_real_imaginary,
    compute_autocorrelation,
    create_complex_gaussian_sampling,
    create_gaussian_sampling,
    create_uniform_sampling,
    effective_sample_size,
    integrated_autocorrelation_time,
    jackknife_resample,
    rebin_data,
    split_complex_sampling,
)
from .writer import SigmondWriter

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
    "SigmondSamplingArray",
    "ArrayObsInfo",
    "ArrayElementObsInfo",
    "AxisMeta",
    "SigmondBins",
    "KnownEnsembles",
    "SamplingInfo",
    "EnsembleInfo",
    "ObservableInfo",
    "SectorInfo",
    "DEFAULT_ENSEMBLE",
    "SamplingStats",
    "SamplingFit",
    "SamplingFitResult",
    "set_thread_counts",
    "default_num_workers",
    "make_process_pool",
    # Spectrum plotting
    "SpectrumPlotter",
    "SectorSpectrumPlotter",
    "SpectrumStyle",
    "HMarker",
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
