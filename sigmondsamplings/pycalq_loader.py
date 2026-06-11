"""
Loader module for Sigmond samplings files of PyCALQ format.
"""

import re
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from .io.loader import SigmondLoader
from .project_utils import string_of_list_to_list
from .sampling import SigmondSampling


class PyCALQEstimateResultType(Enum):
    """
    Enum for PyCALQ estimate types that appear in the 3fit_spectrum output data.

    This enum defines the types of estimates that can be extracted from 3fit_spectrum tasks.
    """

    INTERACTING = "interacting"
    SINGLE_HADRON = "single_hadrons"


class PyCALQSamplingResultType(Enum):
    """
    Enum for PyCALQ result types that appear in the 3fit_spectrum output data.

    This enum defines the types of results that can be extracted from 3fit_spectrum tasks.
    """

    ENERGY_LEVELS = "levels_sigmond"
    OP_OVERLAPS = "operator_overlaps"
    INTERACTING_FIT_PARAMETERS = "fitparams"
    SINGLE_HADRON_FIT_PARAMETERS = "sh_fitparams"


@dataclass
class PyCALQRotateInfo:
    """
    Data class for correlator matrix rotation information.
    """

    tN: int  # normalization time
    t0: int  # initial time
    tD: int  # diagonalization time

    def __init__(self, tN: int, t0: int, tD: int):
        self.tN = tN
        self.t0 = t0
        self.tD = tD

    def __str__(self):
        return f"tN={self.tN}, t0={self.t0}, tD={self.tD}"

    def get_file_tag(self) -> str:
        """
        Get the file tag for this rotation info.

        Returns
        -------
        str
            The file tag in the format "tN-t0-tD".
        """
        return f"{self.tN}tN-{self.t0}t0-{self.tD}tD"


class PyCALQPivotType(Enum):
    """
    Enum for pivot types used in PyCALQ.

    This enum defines the types of pivot algorithms that can be used in PyCALQ
    """

    SINGLE_PIVOT = "SP"
    ROLLING_PIVOT = "RP"


class PyCALQPaths(Enum):
    """
    Enum for PyCALQ paths to data directories.
    The following folders only refer to the 3fit_spectrum output data.
    """

    ESTIMATES = "3fit_spectrum/data/estimates"
    SAMPLINGS = "3fit_spectrum/data/samples"


class PyCALQLoader(SigmondLoader):
    """
    Loader for PyCALQ format Sigmond samplings files.

    This loader extends the base SigmondLoader to handle PyCALQ-specific XML structures
    and provides methods to extract observables, ensembles, and sampling information.
    """

    def __init__(
        self,
        pycalq_project_base_dir: str | Path | None = None,
        hdf5_path: str = "/samplings",
    ):
        """
        Initialize the PyCALQLoader.

        Parameters
        ----------
        pycalq_project_base_dir : Optional[Union[str, Path]]
            Base directory for PyCALQ project files. If None, defaults to current working directory.
        hdf5_path : str, optional
            Path inside HDF5 files for sigmond_query. Defaults to "/samplings".
        """
        pycalq_project_base_dir = (
            Path(pycalq_project_base_dir) if pycalq_project_base_dir else Path.cwd()
        )
        self.pycalq_project_base_dir = pycalq_project_base_dir
        self.hdf5_path = hdf5_path
        super().__init__()

    def _format_hdf5_file_path(self, file_path: str) -> str:
        """
        Format HDF5 file path for sigmond_query by appending the HDF5 internal path.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file

        Returns
        -------
        str
            Formatted path in the format 'file.hdf5[/path]' for sigmond_query
        """
        if file_path.endswith(".hdf5"):
            return f"{file_path}[{self.hdf5_path}]"
        return file_path

    def load_all_observables(self, filename: str) -> dict[str, "SigmondSampling"]:
        """
        Load all observables from the file, with HDF5 path handling.

        For HDF5 files, automatically appends the HDF5 internal path in the format
        required by sigmond_query: 'file.hdf5[/path]'

        Parameters
        ----------
        filename : str
            Path to the samplings file

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable string to SigmondSampling objects
        """
        formatted_filename = self._format_hdf5_file_path(filename)
        return super().load_all_observables(formatted_filename)

    def get_file_info(self, filename: str):
        """
        Get file information with HDF5 path handling.

        Parameters
        ----------
        filename : str
            Path to the samplings file

        Returns
        -------
        Tuple
            Ensemble info, sampling info, and list of observable info objects
        """
        formatted_filename = self._format_hdf5_file_path(filename)
        return super().get_file_info(formatted_filename)

    def get_estimate_files(
        self,
        result_type: PyCALQEstimateResultType | list[PyCALQEstimateResultType] | None = None,
    ) -> list[str]:
        """
        Get the list of estimate files from a PyCALQ project file.

        Parameters
        ----------
        result_type : Optional[Union[PyCALQEstimateResultType, List[PyCALQEstimateResultType]]]
            The type of estimate results to retrieve. If None, retrieves all types.

        Returns
        -------
        List[str]
            List of estimate file paths.
        """
        if result_type is None:
            result_type = [
                PyCALQEstimateResultType.INTERACTING,
                PyCALQEstimateResultType.SINGLE_HADRON,
            ]
        if isinstance(result_type, PyCALQEstimateResultType):
            result_type = [result_type]

        estimate_files = []
        for rtype in result_type:
            files = self.pycalq_project_base_dir.glob(
                f"3fit_spectrum/data/estimates/*-{rtype.value}_estimates.csv"
            )
            estimate_files.extend([str(f) for f in files])

        return estimate_files

    def get_estimate_file_path(
        self,
        result_type: PyCALQEstimateResultType | list[PyCALQEstimateResultType],
        tag: str | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
        pivot_type: PyCALQPivotType | None = None,
        resampling_method: str | None = None,
    ) -> str | list[str]:
        """
        Get the actual file path(s) for the specified PyCALQ estimate result type(s) using glob.

        Parameters
        ----------
        result_type : PyCALQEstimateResultType | List[PyCALQEstimateResultType]
            The type of estimate results to retrieve. Can be a single type or a list of types.
        tag : Optional[str]
            The tag name for the file. If None, uses wildcard.
        rotate_info : Optional[PyCALQRotateInfo]
            The rotation info for the file. If None, uses wildcard.
        num_bins : Optional[int]
            The number of bins. If None, uses wildcard.
        pivot_type : Optional[PyCALQPivotType]
            The pivot type. If None, uses wildcard.
        resampling_method : Optional[str]
            The resampling method ("B" or "J"). If None, uses wildcard.

        Returns
        -------
        str | List[str]
            The actual file path(s) for the specified estimate result type(s).
            Returns first matching file for single result type, or list of files for multiple types.
        """
        if isinstance(result_type, PyCALQEstimateResultType):
            result_types = [result_type]
            single_result = True
        else:
            result_types = result_type
            single_result = False

        # Build glob pattern for each result type
        def build_glob_pattern(rtype: PyCALQEstimateResultType) -> str:
            components = ["fit_spectrum_fit_spectrum_-"]

            if tag is not None:
                components.append(tag)
            else:
                components.append("*")

            if num_bins is not None:
                components.append(f"-Nbin{num_bins}")
            else:
                components.append("-Nbin*")

            if pivot_type is not None:
                components.append(f"-{pivot_type.value}")
            else:
                components.append("-*")

            if rotate_info is not None:
                components.append(f"-{rotate_info.get_file_tag()}")
            else:
                components.append("-*tN-*t0-*tD")

            if resampling_method is not None:
                components.append(f"_{resampling_method}-samplings")
            else:
                components.append("_*-samplings")

            components.append(f"-{rtype.value}_estimates")

            filename = "".join(components) + ".csv"

            return str(
                Path(self.pycalq_project_base_dir)
                / "3fit_spectrum"
                / "data"
                / "estimates"
                / filename
            )

        # Use glob to find actual matching files
        all_matching_files = []
        for rtype in result_types:
            pattern = build_glob_pattern(rtype)
            matching_files = glob(pattern)
            all_matching_files.extend(matching_files)

        if single_result:
            return all_matching_files[0] if all_matching_files else None
        else:
            return all_matching_files

    def get_sampling_files(
        self,
        result_type: PyCALQSamplingResultType | list[PyCALQSamplingResultType] | None = None,
    ) -> list[str]:
        """
        Get the list of sampling files from a PyCALQ project file.

        Parameters
        ----------
        result_type : Optional[Union[PyCALQSamplingResultType, List[PyCALQSamplingResultType]]]
            The type of sampling results to retrieve. If None, retrieves all types.

        Returns
        -------
        List[str]
            List of sampling file paths.
        """

        if result_type is None:
            result_type = [
                PyCALQSamplingResultType.ENERGY_LEVELS,
                PyCALQSamplingResultType.OP_OVERLAPS,
                PyCALQSamplingResultType.INTERACTING_FIT_PARAMETERS,
                PyCALQSamplingResultType.SINGLE_HADRON_FIT_PARAMETERS,
            ]

        if isinstance(result_type, PyCALQSamplingResultType):
            result_type = [result_type]

        sampling_files = []
        for rtype in result_type:
            files = self.pycalq_project_base_dir.glob(
                f"3fit_spectrum/data/samples/fit_spectrum_{rtype.value}*.hdf5"
            )
            sampling_files.extend([str(f) for f in files])

        return sampling_files

    def get_sampling_file_path(
        self,
        result_type: PyCALQSamplingResultType | list[PyCALQSamplingResultType],
        tag: str | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
        pivot_type: PyCALQPivotType | None = None,
        resampling_method: str | None = None,
    ) -> str | list[str]:
        """
        Get the file path(s) for the specified PyCALQ sampling result type(s) by globbing for existing files.

        Parameters
        ----------
        result_type : PyCALQSamplingResultType | List[PyCALQSamplingResultType]
            The type of sampling results to retrieve. Can be a single type or a list of types.
        tag : Optional[str]
            The tag name for the file. If None, matches any tag.
        rotate_info : Optional[PyCALQRotateInfo]
            The rotation info for the file. If None, matches any rotation.
        num_bins : Optional[int]
            The number of bins. If None, matches any number of bins.
        pivot_type : Optional[PyCALQPivotType]
            The pivot type. If None, matches any pivot type.
        resampling_method : Optional[str]
            The resampling method ("B" or "J"). If None, matches any resampling method.

        Returns
        -------
        str | List[str]
            The actual file path(s) that match the criteria. Returns empty list if no files found.
        """
        if isinstance(result_type, PyCALQSamplingResultType):
            result_type = [result_type]
            single_result = True
        else:
            single_result = False

        # Build glob pattern for each result type
        def build_glob_pattern(rtype: PyCALQSamplingResultType) -> str:
            components = ["fit_spectrum_", rtype.value]

            if tag is not None:
                components.append(f"-{tag}")
            else:
                components.append("-*")

            if num_bins is not None:
                components.append(f"-Nbin{num_bins}")
            else:
                components.append("-Nbin*")

            if pivot_type is not None:
                components.append(f"-{pivot_type.value}")
            else:
                components.append("-*")

            if rotate_info is not None:
                components.append(f"-{rotate_info.get_file_tag()}")
            else:
                components.append("-*tN-*t0-*tD")

            if resampling_method is not None:
                components.append(f"_{resampling_method}-samplings")
            else:
                components.append("_*-samplings")

            filename = "".join(components) + ".hdf5"

            return str(
                Path(self.pycalq_project_base_dir) / "3fit_spectrum" / "data" / "samples" / filename
            )

        # Use glob to find actual matching files
        all_matching_files = []
        for rtype in result_type:
            pattern = build_glob_pattern(rtype)
            matching_files = glob(pattern)
            all_matching_files.extend(matching_files)

        if single_result:
            return all_matching_files[0] if all_matching_files else ""
        else:
            return all_matching_files

    def get_all_available_tags(self) -> list[str]:
        """
        Get all available tags for PyCALQ sampling results.

        Returns
        -------
        List[str]
            A list of all available tags.
        """
        sampling_dir = Path(PyCALQPaths.SAMPLINGS.value)
        full_sampling_dir_path = self.pycalq_project_base_dir / sampling_dir

        if not full_sampling_dir_path.exists():
            raise FileNotFoundError(
                f"PyCALQ sampling directory not found: {full_sampling_dir_path}"
            )

        tags = []
        for file in full_sampling_dir_path.glob("*.hdf5"):
            tag = self.get_file_name_tag(file.name)
            if tag and tag not in tags:
                tags.append(tag)

        return sorted(tags)

    def get_file_name_info(self, file_path: str) -> dict[str, Any]:
        """
        Get metadata information contained in the name of a PyCALQ sampling or estimate file.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the data type ("data_type": PyCALQSamplingResultType | PyCALQEstimateResultType),
            tag name ("tag_name": str), rotate info ("rotate_info": PyCALQRotateInfo | None),
            number of bins ("num_bins": int), pivot type ("pivot_type": PyCALQPivotType | None),
            and resampling method ("resampling_method": "J" or "B").
        """

        data_type = self.get_file_data_type(file_path)
        tag_name = self.get_file_name_tag(file_path)
        rotate_info = self.get_file_name_rotate_info(file_path)
        num_bins = self.get_file_name_num_bins(file_path)
        pivot_type = self.get_file_name_pivot_type(file_path)
        resampling_method = self.get_file_name_resampling_method(file_path)

        return {
            "data_type": data_type,
            "tag_name": tag_name,
            "rotate_info": rotate_info,
            "num_bins": num_bins,
            "pivot_type": pivot_type,
            "resampling_method": resampling_method,
        }

    def get_file_data_type(
        self, file_path: str
    ) -> PyCALQSamplingResultType | PyCALQEstimateResultType:
        """
        Get the data type of a PyCALQ sampling/estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        PyCALQSamplingResultType | PyCALQEstimateResultType
            The type of the PyCALQ file based on its naming convention.
        """
        file_name = Path(file_path).name
        for rtype in PyCALQSamplingResultType:
            if rtype.value in file_name:
                return rtype
        for rtype in PyCALQEstimateResultType:
            if rtype.value in file_name:
                return rtype
        raise ValueError(f"Unknown PyCALQ file type in file: {file_path}")

    def get_file_name_rotate_info(self, file_path: str) -> PyCALQRotateInfo | None:
        """
        Get the rotation info from a PyCALQ sampling or estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        Optional[PyCALQRotateInfo]
            The rotation info if present, otherwise None.
        """
        file_name = Path(file_path).name
        rotate_match = re.search(r"(\d+)tN-(\d+)t0-(\d+)tD", file_name)
        if rotate_match:
            tN, t0, tD = map(int, rotate_match.groups())
            return PyCALQRotateInfo(tN, t0, tD)
        return None

    def get_file_name_pivot_type(self, file_path: str) -> PyCALQPivotType | None:
        """
        Get the pivot type from a PyCALQ sampling or estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        Optional[PyCALQPivotType]
            The pivot type if present, otherwise None.
        """
        file_name = Path(file_path).name
        if "SP" in file_name:
            return PyCALQPivotType.SINGLE_PIVOT
        elif "RP" in file_name:
            return PyCALQPivotType.ROLLING_PIVOT
        return None

    def get_file_name_resampling_method(self, file_path: str) -> str:
        """
        Get the resampling method from a PyCALQ sampling or estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        str
            The resampling method, either "J" for jackknife or "B" for bootstrap.
        """
        file_name = Path(file_path).name
        return "J" if "J" in file_name else "B"

    def get_file_name_tag(self, file_path: str) -> str:
        """
        Get the tag from a PyCALQ sampling or estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        str
            The tag extracted from the filename.
        """

        file_name = Path(file_path).name
        # If the file type is an estimate, then the tag follows 'fit_spectrum_fit_spectrum_-'
        # and ends at '-Nbin'.
        if isinstance(self.get_file_data_type(file_name), PyCALQEstimateResultType):
            tag_match = re.search(r"fit_spectrum_fit_spectrum_-(.*?)-Nbin", file_name)

        # If the file type is a sampling, then the tag follows 'fit_spectrum_{data_type.value}-'
        # and ends at '-Nbin'.
        else:
            tag_match = re.search(
                rf"fit_spectrum_{self.get_file_data_type(file_name).value}-(.*?)-Nbin",
                file_name,
            )

        return tag_match.group(1) if tag_match else ""

    def get_file_name_num_bins(self, file_path: str) -> int:
        """
        Get the number of bins from a PyCALQ sampling or estimate file based on its filename.

        Parameters
        ----------
        file_path : str
            The path to the PyCALQ file or the filename.

        Returns
        -------
        int
            The number of bins extracted from the filename.
        """
        file_name = Path(file_path).name
        num_bins_match = re.search(r"Nbin(\d+)", file_name)
        return int(num_bins_match.group(1)) if num_bins_match else 0

    def load_observables_by_tag(
        self,
        tag: str,
        result_types: list[PyCALQSamplingResultType] | None = None,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> dict[str, dict[str, "SigmondSampling"]]:
        """
        Load observables from PyCALQ files by tag, with optional filtering.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        result_types : Optional[List[PyCALQSamplingResultType]]
            Types of results to load. If None, loads all available types.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Dict[str, Dict[str, SigmondSampling]]
            Nested dictionary: {result_type: {observable_name: SigmondSampling}}
        """
        results = {}

        for rtype in result_types or []:
            this_type_files = self.get_sampling_file_path(
                tag=tag,
                result_type=rtype,
                resampling_method=resampling_method,
                pivot_type=pivot_type,
                rotate_info=rotate_info,
                num_bins=num_bins,
            )
            if not this_type_files:
                continue
            results[rtype.value] = {}
            if isinstance(this_type_files, str):
                this_type_files = [this_type_files]
            for file_path in this_type_files:
                try:
                    # Load all observables from this file
                    file_observables = self.load_all_observables(str(file_path))
                    results[rtype.value].update(file_observables)
                except Exception as e:
                    print(f"Warning: Failed to load file {file_path}: {e}")
                    continue

        return results

    def load_interacting_estimates_to_dataframe(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> Any:
        """
        Load estimates into a DataFrame.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Any
            A DataFrame containing the estimates data.
        """

        estimate_files = self.get_estimate_file_path(
            result_type=PyCALQEstimateResultType.INTERACTING,
            tag=tag,
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        if isinstance(estimate_files, str):
            estimate_files = [estimate_files]

        if not estimate_files:
            return None

        # Load the first matching CSV file
        try:
            return pd.read_csv(estimate_files[0])
        except Exception as e:
            print(f"Error loading estimates file {estimate_files[0]}: {e}")
            return None

    def load_NIs_to_dict(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> dict[int, list[list[str]]]:
        """
        Load NIs (non-interacting) particle pairs into a dictionary.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Dict[int, List[List[str]]]
            A dictionary where keys are momenta and values are lists of non-interacting
            levels (NIs) for each rotate level. Each NI is represented as a list of strings.
        """
        int_estimates_df = self.load_interacting_estimates_to_dataframe(
            tag=tag,
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        if int_estimates_df is None or int_estimates_df.empty:
            return {}

        df_reduce = int_estimates_df[["momentum", "rotate level", "non-interacting level"]].copy()
        df_reduce.rename(columns={"non-interacting level": "NI"}, inplace=True)
        # create a dictionary with momentum as key and list of NIs as value
        NI_dict = {}
        for _, row in df_reduce.iterrows():
            momentum = int(row["momentum"])
            NI = row["NI"]
            if momentum not in NI_dict:
                NI_dict[momentum] = []
            NI_dict[momentum].append(string_of_list_to_list(NI))
        return NI_dict

    def load_single_hadron_estimates_to_dataframe(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> Any:
        """
        Load single hadron estimates into a DataFrame.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Any
            A DataFrame containing the single hadron estimates data.
        """
        estimate_file_pattern = self.get_estimate_file_path(
            result_type=PyCALQEstimateResultType.SINGLE_HADRON,
            tag=tag,
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        # Find matching files using glob pattern
        matching_files = glob(estimate_file_pattern)

        if not matching_files:
            return None

        # Load the first matching CSV file
        try:
            return pd.read_csv(matching_files[0])
        except Exception as e:
            print(f"Error loading single hadron estimates file {matching_files[0]}: {e}")
            return None

    def load_energy_levels(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> dict[str, "SigmondSampling"]:
        """
        Load energy level observables for a specific tag.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        results = self.load_observables_by_tag(
            tag=tag,
            result_types=[PyCALQSamplingResultType.ENERGY_LEVELS],
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        return results.get(PyCALQSamplingResultType.ENERGY_LEVELS.value, {})

    def load_energy_levels_and_build_mom_dict(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
        NIs: bool | None = True,
        ref: bool | None = True,
    ) -> dict[
        int,
        dict[
            int,
            tuple[list[str], dict[str, SigmondSampling]] | dict[str, SigmondSampling],
        ],
    ]:
        """
        Load energy levels and build a dictionary mapping momenta to lists of SigmondSampling objects.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.
        ref : Optional[str]
            Reference momentum to filter the results. If None, includes all momenta.

        Returns
        -------

        """
        energy_levels = self.load_energy_levels(
            tag=tag,
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        if NIs:
            NIs_dict = self.load_NIs_to_dict(
                tag=tag,
                resampling_method=resampling_method,
                pivot_type=pivot_type,
                rotate_info=rotate_info,
                num_bins=num_bins,
            )

        mom_dict = {}
        for obs_name, sampling in energy_levels.items():
            parsed_info = self.parse_observable_name(obs_name, ref)
            if parsed_info is None:
                continue
            psq, energy_type, level_idx = parsed_info

            if NIs:
                NI = NIs_dict[psq][level_idx]

            if psq not in mom_dict:
                mom_dict[psq] = {}
            if level_idx not in mom_dict[psq]:
                if NIs:
                    mom_dict[psq][level_idx] = (NI, {})
                else:
                    mom_dict[psq][level_idx] = {}

            if NIs:
                mom_dict[psq][level_idx][1][energy_type] = sampling
            else:
                mom_dict[psq][level_idx][energy_type] = sampling

        return mom_dict

    def load_operator_overlaps(
        self,
        tag: str,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> dict[str, "SigmondSampling"]:
        """
        Load operator overlap observables for a specific tag.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        results = self.load_observables_by_tag(
            tag=tag,
            result_types=[PyCALQSamplingResultType.OP_OVERLAPS],
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        return results.get(PyCALQSamplingResultType.OP_OVERLAPS.value, {})

    def load_fit_parameters(
        self,
        tag: str,
        parameter_type: PyCALQSamplingResultType,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> dict[str, "SigmondSampling"]:
        """
        Load fit parameter observables for a specific tag.

        Parameters
        ----------
        tag : str
            The tag to search for in filenames.
        parameter_type : PyCALQSamplingResultType
            Type of fit parameters (INTERACTING_FIT_PARAMETERS or SINGLE_HADRON_FIT_PARAMETERS).
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J"). If None, loads both.
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type. If None, loads all pivot types.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters. If None, loads all.
        num_bins : Optional[int]
            Filter by number of bins. If None, loads all.

        Returns
        -------
        Dict[str, SigmondSampling]
            Dictionary mapping observable names to SigmondSampling objects.
        """
        if parameter_type not in [
            PyCALQSamplingResultType.INTERACTING_FIT_PARAMETERS,
            PyCALQSamplingResultType.SINGLE_HADRON_FIT_PARAMETERS,
        ]:
            raise ValueError(
                "parameter_type must be INTERACTING_FIT_PARAMETERS or SINGLE_HADRON_FIT_PARAMETERS"
            )

        results = self.load_observables_by_tag(
            tag=tag,
            result_types=[parameter_type],
            resampling_method=resampling_method,
            pivot_type=pivot_type,
            rotate_info=rotate_info,
            num_bins=num_bins,
        )

        return results.get(parameter_type.value, {})

    def get_available_files_info(self, tag: str | None = None) -> list[dict[str, Any]]:
        """
        Get information about all available PyCALQ sampling files.

        Parameters
        ----------
        tag : Optional[str]
            Filter by specific tag. If None, returns info for all files.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing file information including:
            - file_path: Path to the file
            - data_type: PyCALQSamplingResultType
            - tag_name: Tag extracted from filename
            - rotate_info: PyCALQRotateInfo if present
            - num_bins: Number of bins
            - pivot_type: PyCALQPivotType if present
            - resampling_method: "B" or "J"
        """
        sampling_dir = self.pycalq_project_base_dir / PyCALQPaths.SAMPLINGS.value

        if not sampling_dir.exists():
            return []

        pattern = "*.hdf5"
        if tag is not None:
            pattern = f"*{tag}*.hdf5"

        files_info = []
        for file_path in sampling_dir.glob(pattern):
            try:
                file_info = self.get_file_name_info(str(file_path))
                file_info["file_path"] = str(file_path)
                files_info.append(file_info)
            except Exception as e:
                print(f"Warning: Could not parse file {file_path}: {e}")
                continue

        return files_info

    def find_files_by_criteria(
        self,
        tag: str | None = None,
        result_type: PyCALQSamplingResultType | PyCALQEstimateResultType | None = None,
        resampling_method: str | None = None,
        pivot_type: PyCALQPivotType | None = None,
        rotate_info: PyCALQRotateInfo | None = None,
        num_bins: int | None = None,
    ) -> list[str]:
        """
        Find PyCALQ files matching specific criteria.

        Parameters
        ----------
        tag : Optional[str]
            Filter by tag name.
        result_type : Optional[PyCALQSamplingResultType]
            Filter by result type.
        resampling_method : Optional[str]
            Filter by resampling method ("B" or "J").
        pivot_type : Optional[PyCALQPivotType]
            Filter by pivot type.
        rotate_info : Optional[PyCALQRotateInfo]
            Filter by specific rotation parameters.
        num_bins : Optional[int]
            Filter by number of bins.

        Returns
        -------
        List[str]
            List of file paths matching the criteria.
        """
        all_files = self.get_available_files_info()
        matching_files = []

        for file_info in all_files:
            match = True

            if tag is not None and file_info["tag_name"] != tag:
                match = False
            if result_type is not None and file_info["data_type"] != result_type:
                match = False
            if (
                resampling_method is not None
                and file_info["resampling_method"] != resampling_method
            ):
                match = False
            if pivot_type is not None and file_info["pivot_type"] != pivot_type:
                match = False
            if num_bins is not None and file_info["num_bins"] != num_bins:
                match = False
            if rotate_info is not None:
                file_rotate = file_info["rotate_info"]
                if (
                    file_rotate is None
                    or file_rotate.tN != rotate_info.tN
                    or file_rotate.t0 != rotate_info.t0
                    or file_rotate.tD != rotate_info.tD
                ):
                    match = False

            if match:
                matching_files.append(file_info["file_path"])

        return matching_files

    @staticmethod
    def parse_observable_name(obs_name: str, ref: bool = False) -> tuple[int, str, int] | None:
        """
        Parse a PyCALQ observable name to extract PSQ, energy type, and level index.

        This is a universal parser for PyCALQ observable naming conventions.

        Parameters
        ----------
        obs_name : str
            Observable name like "isosinglet_S=0_A1g_1_PSQ=0_elab_1_ref 0"
        ref : bool, optional
            Whether to expect "_ref" suffix in the observable name.

        Returns
        -------
        Optional[Tuple[int, str, int]]
            Tuple of (psq, energy_type, level_index) or None if parsing fails.

        Examples
        --------
        >>> parse_observable_name("isosinglet_S=0_A1g_1_PSQ=0_elab_1_ref 0", ref=True)
        (0, 'elab', 1)

        >>> parse_observable_name("channel_P=(1,0,0)_ecm_2 1", ref=False)
        (1, 'ecm', 2)
        """
        # Extract the actual observable name (before the space and index)
        obs_parts = obs_name.split()
        if len(obs_parts) != 2:
            return None
        actual_name = obs_parts[0]

        # Build the regex pattern based on whether ref is expected
        ref_suffix = "_ref" if ref else ""

        # Pattern to match PSQ format: PSQ=N
        psq_pattern = r"PSQ=(\d+)"
        psq_match = re.search(psq_pattern, actual_name)

        if psq_match:
            psq = int(psq_match.group(1))
            # Pattern to match energy type and level: _{energy_type}_{level}{_ref}
            energy_pattern = rf"_([a-zA-Z]+)_(\d+){re.escape(ref_suffix)}$"
            energy_match = re.search(energy_pattern, actual_name)
            if energy_match:
                energy_type = energy_match.group(1)
                level_index = int(energy_match.group(2))
                return psq, energy_type, level_index
        else:
            # Try P=(x,y,z) format
            p_pattern = r"P=\(([^)]+)\)"
            p_match = re.search(p_pattern, actual_name)
            if p_match:
                p_coords = p_match.group(1).split(",")
                psq = sum(int(x) ** 2 for x in p_coords)
                # Pattern to match energy type and level
                energy_pattern = rf"_([a-zA-Z]+)_(\d+){re.escape(ref_suffix)}$"
                energy_match = re.search(energy_pattern, actual_name)
                if energy_match:
                    energy_type = energy_match.group(1)
                    level_index = int(energy_match.group(2))
                    return psq, energy_type, level_index

        return None
