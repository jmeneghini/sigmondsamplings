"""
Abstract base class for project-specific data acquisition and parsing.

This module provides a framework for creating project-specific data parsers
that can use either PyCALQLoader or the base SigmondLoader depending on the
project's data format and structure.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from .sampling import SigmondSampling, EnsembleInfo, SamplingInfo
from .loader import SigmondLoader
from .pycalq_loader import PyCALQLoader


class AbstractProjectDataParser(ABC):
    """
    Abstract base class for project-specific data acquisition and parsing.

    This class provides a framework for implementing project-specific data parsers
    that can handle different data formats and structures while providing a
    consistent interface for data acquisition.
    """

    def __init__(self, project_base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the project data parser.

        Parameters
        ----------
        project_base_dir : Optional[Union[str, Path]]
            Base directory for the project files. If None, uses current working directory.
        """
        self.project_base_dir = (
            Path(project_base_dir) if project_base_dir else Path.cwd()
        )
        self.all_observables = None
        self._loader = None

    @abstractmethod
    def _initialize_loader(self) -> None:
        """
        Initialize the appropriate loader (PyCALQLoader or SigmondLoader).

        This method should be implemented by subclasses to set up self._loader
        based on the project's specific requirements.
        """
        pass

    @abstractmethod
    def load_energy_levels(self, **kwargs) -> Dict[str, SigmondSampling]:
        """
        Load energy level data for a specific dataset.

        Parameters
        ----------
        **kwargs
            Additional parameters specific to the project implementation.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        pass

    @abstractmethod
    def get_ensemble_and_sampling_info(
        self, **kwargs
    ) -> Tuple[EnsembleInfo, SamplingInfo]:
        """
        Get ensemble and sampling information for a dataset.

        Parameters
        ----------
        **kwargs
            Additional parameters specific to the project implementation.

        Returns
        -------
        Tuple[EnsembleInfo, SamplingInfo]
            Ensemble and sampling information.
        """
        pass

    # TODO: SHOULD BE IN SIGMONDLOADER
    def load_observables_by_pattern(
        self, name_patterns: Union[str, List[str]], **kwargs
    ) -> Dict[str, SigmondSampling]:
        """
        Load observables matching specific name patterns.

        Parameters
        ----------
        name_patterns : Union[str, List[str]]
            Regex patterns to match observable names.
        **kwargs
            Additional parameters specific to the project implementation.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        # Default implementation - subclasses can override for optimization
        if self.all_observables is None:
            self.all_observables = self.load_all_observables(**kwargs)

        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        filtered_observables = {}
        for obs_name, sampling in self.all_observables.items():
            for pattern in name_patterns:
                if re.search(pattern, obs_name):
                    filtered_observables[obs_name] = sampling
                    break

        return filtered_observables

    def load_all_observables(self, **kwargs) -> Dict[str, SigmondSampling]:
        """
        Load all observables for a dataset.

        Default implementation that subclasses can override.

        Parameters
        ----------
        **kwargs
            Additional parameters specific to the project implementation.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        raise NotImplementedError(
            "Subclasses must implement load_all_observables or override this method"
        )


class PyCALQProjectDataParser(AbstractProjectDataParser):
    """
    Project data parser that uses PyCALQLoader for data acquisition.

    This implementation is suitable for projects that use PyCALQ data format
    and file organization conventions.
    """

    def __init__(
        self,
        project_base_dir: Optional[Union[str, Path]] = None,
        hdf5_path: str = "/samplings",
    ):
        """
        Initialize the PyCALQ project data parser.

        Parameters
        ----------
        project_base_dir : Optional[Union[str, Path]]
            Base directory for the project. If None, uses current working directory.
        hdf5_path : str, optional
            Path inside HDF5 files for sigmond_query. Defaults to "/samplings".
        """
        self.hdf5_path = hdf5_path
        super().__init__(project_base_dir)

    def _initialize_loader(self) -> None:
        """Initialize PyCALQLoader with HDF5 path."""
        self._loader = PyCALQLoader(self.project_base_dir, self.hdf5_path)

    def get_available_datasets(self) -> List[str]:
        """
        Get a list of available datasets/tags in the project.

        Returns
        -------
        List[str]
            List of available dataset identifiers (tags).
        """
        return self._loader.get_all_available_tags()

    def load_energy_levels(self, **kwargs) -> Dict[str, SigmondSampling]:
        """
        Load energy level data for a specific dataset.

        Parameters
        ----------
        **kwargs
            Additional parameters specific to PyCALQ implementation:
            - tag: str (required)
            - resampling_method: Optional[str]
            - pivot_type: Optional[PyCALQPivotType]
            - rotate_info: Optional[PyCALQRotateInfo]
            - num_bins: Optional[int]

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        return self._loader.load_energy_levels(**kwargs)

    def get_ensemble_and_sampling_info(
        self, **kwargs
    ) -> Tuple[EnsembleInfo, SamplingInfo]:
        """
        Get ensemble and sampling info from PyCALQ files.

        Parameters
        ----------
        **kwargs
            Additional parameters to filter the dataset.

        Returns
        -------
        Tuple[EnsembleInfo, SamplingInfo]
            Ensemble and sampling information.
        """
        # Find a file for this dataset to extract info
        files = self._loader.find_files_by_criteria(**kwargs)
        if not files:
            raise ValueError(f"No files found for dataset: {kwargs}")

        # Use the first file to get ensemble info
        ensemble_info, sampling_info, _ = self._loader.get_file_info(files[0])
        return ensemble_info, sampling_info

    @staticmethod
    def filter_levels_by_criteria(
        results: Dict[int, Dict[int, Tuple[List, Dict[str, SigmondSampling]]]],
        is_relevant_level_from_energy_and_NI: Optional[callable] = None,
        is_relevant_psq_from_levels: Optional[callable] = None,
    ) -> Dict[int, Dict[int, Tuple[List, Dict[str, SigmondSampling]]]]:
        """
        Filter results based on user-defined criteria.

        Parameters
        ----------
        results : Dict[int, Dict[int, Tuple[List, Dict[str, SigmondSampling]]]]
            Results from load_energy_levels_with_NI.
        is_relevant_level_from_energy_and_NI : Optional[callable]
            Function that takes (psq, level_index) and level_data and returns bool.
        is_relevant_psq_from_levels : Optional[callable]
            Function that takes psq_data and returns bool.

        Returns
        -------
        Dict[int, Dict[int, Tuple[List, Dict[str, SigmondSampling]]]]
            Filtered results.
        """
        # Make a copy to avoid modifying the original
        filtered_results = {}
        for psq, psq_data in results.items():
            filtered_results[psq] = {}
            for level_index, level_data in psq_data.items():
                filtered_results[psq][level_index] = level_data

        # Apply level-based filtering
        if is_relevant_level_from_energy_and_NI is not None:
            for psq in list(filtered_results.keys()):
                for level_index in list(filtered_results[psq].keys()):
                    level_data = filtered_results[psq][level_index]
                    psq_level_index = (psq, level_index)
                    if not is_relevant_level_from_energy_and_NI(
                        psq_level_index, level_data
                    ):
                        del filtered_results[psq][level_index]

        # Apply PSQ-based filtering
        if is_relevant_psq_from_levels is not None:
            for psq in list(filtered_results.keys()):
                psq_data = filtered_results[psq]
                if not is_relevant_psq_from_levels(psq_data):
                    del filtered_results[psq]

        # Remove empty levels and PSQ entries
        for psq in list(filtered_results.keys()):
            for level_index in list(filtered_results[psq].keys()):
                _, energy_dict = filtered_results[psq][level_index]
                if not energy_dict:
                    del filtered_results[psq][level_index]
            if not filtered_results[psq]:
                del filtered_results[psq]

        return filtered_results
