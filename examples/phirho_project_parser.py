"""
Example implementation of PyCALQProjectDataParser for the phi-rho project.

This demonstrates how to create a project-specific data parser that uses
the PyCALQ infrastructure for simplified data acquisition.
"""

from pathlib import Path

import pandas as pd

import sigmondsamplings as ss


class PhiRhoProjectDataParser(ss.PyCALQProjectDataParser):
    """
    Project-specific data parser for phi-rho analysis.

    This parser extends PyCALQProjectDataParser to provide phi-rho specific
    functionality while leveraging the PyCALQ infrastructure.
    """

    def __init__(self, L: int | str, project_base_dir: str | Path | None = None):
        """
        Initialize the phi-rho project parser.

        Parameters
        ----------
        L : Union[int, str]
            Lattice spatial extent (e.g., 24, 32, 48).
        project_base_dir : Optional[Union[str, Path]]
            Base directory for the project. If None, uses OS-specific default.
        """
        if project_base_dir is None:
            project_base_dir = self._get_default_project_dir(L)

        super().__init__(project_base_dir)
        self.L = L

    def _get_default_project_dir(self, L: int | str) -> Path:
        """Get default project directory based on OS and lattice size."""
        os_info = ss.get_os_info()

        if os_info.name == "macOS":
            base = "/pi-mnt"
        elif os_info.name == "Linux":
            if os_info.distro and "Ubuntu" in os_info.distro:
                base = "/pi-mnt"
            else:
                base = "/home/jmeneghini"
        else:
            raise NotImplementedError(f"Unsupported OS: {os_info.name}")

        return (
            Path(base)
            / f"latticeQCD/spectrum_analysis/channels/phirho/levels/john_results/s{L}/3fit_spectrum"
        )

    def get_available_datasets(self) -> list[str]:
        """
        Get available tags for this lattice size.

        Returns
        -------
        List[str]
            List of available dataset tags.
        """
        # Use the parent implementation
        return super().get_available_datasets()

    def load_energy_levels_with_NI(
        self,
        dataset_id: str,
        resampling_method: str,
        energy_types: list[str] | None = None,
        ref: bool = False,
    ) -> tuple[
        ss.EnsembleInfo,
        ss.SamplingInfo,
        dict[int, dict[int, tuple[list, dict[str, ss.SigmondSampling]]]],
    ]:
        """
        Load energy levels with non-interacting (NI) information.

        This method replicates the functionality of the original get_relevant_ensemble_data
        but uses the new PyCALQ infrastructure.

        Parameters
        ----------
        dataset_id : str
            Tag identifier for the dataset.
        resampling_method : str
            Resampling method ("B" or "J").
        energy_types : Optional[List[str]]
            Energy types to load (e.g., ["elab", "dElab", "ecm"]).
        ref : bool
            Whether to expect "_ref" suffix in observable names.

        Returns
        -------
        Tuple containing:
            - EnsembleInfo: Ensemble information
            - SamplingInfo: Sampling information
            - Dict[int, Dict[int, Tuple[List, Dict[str, SigmondSampling]]]]: Structured data
        """
        if energy_types is None:
            energy_types = ["elab", "dElab", "ecm"]

        # Load all observables for the dataset
        all_observables = self.load_all_observables(
            dataset_id=dataset_id, resampling_method=resampling_method
        )

        # Get ensemble and sampling info
        ensemble_info, sampling_info = self.get_ensemble_info(dataset_id)

        # Load NI dictionary
        NI_dict = self._get_NI_dict(dataset_id, resampling_method)

        # Parse observables and organize by PSQ and level
        out_results = {}

        for obs_name, sampling in all_observables.items():
            # Parse observable name
            parsed_info = ss.parse_observable_name(obs_name, ref)
            if parsed_info is None:
                continue

            psq, energy_type, level_index = parsed_info

            # Check if this energy type is requested
            if energy_type not in energy_types:
                continue

            # Get NI for this PSQ and level
            if psq not in NI_dict or level_index >= len(NI_dict[psq]):
                continue

            NI = NI_dict[psq][level_index]

            # Add to results
            if psq not in out_results:
                out_results[psq] = {}
            if level_index not in out_results[psq]:
                out_results[psq][level_index] = (NI, {})

            # Add the energy type and sampling
            out_results[psq][level_index][1][energy_type] = sampling

        return ensemble_info, sampling_info, out_results

    def _get_NI_dict(self, dataset_id: str, resampling_method: str) -> dict[int, list[list[str]]]:
        """
        Load the NI (non-interacting) dictionary from CSV file.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier (tag).
        resampling_method : str
            Resampling method ("B" or "J").

        Returns
        -------
        Dict[int, List[List[str]]]
            Dictionary mapping PSQ to list of NI level lists.
        """
        csv_path = self._get_levels_csv_path(dataset_id, resampling_method)
        df = pd.read_csv(csv_path)
        df_reduce = df[["momentum", "rotate level", "non-interacting level"]].copy()
        df_reduce.rename(columns={"non-interacting level": "NI"}, inplace=True)

        # Create dictionary with momentum as key and list of NIs as value
        NI_dict = {}
        for _, row in df_reduce.iterrows():
            momentum = int(row["momentum"])
            NI = row["NI"]
            if momentum not in NI_dict:
                NI_dict[momentum] = []
            NI_dict[momentum].append(ss.string_of_list_to_list(NI))

        return NI_dict

    def _get_levels_csv_path(self, dataset_id: str, resampling_method: str) -> str:
        """
        Get path to the CSV file containing level information.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier (tag).
        resampling_method : str
            Resampling method ("B" or "J").

        Returns
        -------
        str
            Path to the CSV file.
        """
        estimates_dir = self.project_base_dir / "data" / "estimates"

        # The dataset_id should match the tag in the filename
        pattern = f"fit_spectrum_fit_spectrum_-{dataset_id}-Nbin*-SP-*tN-*t0-*tD_{resampling_method}-samplings-interacting_estimates.csv"

        matches = list(estimates_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No CSV file found matching pattern: {pattern}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple CSV files found: {matches}")

        return str(matches[0])

    def filter_levels_by_criteria(
        self,
        results: dict[int, dict[int, tuple[list, dict[str, ss.SigmondSampling]]]],
        is_relevant_level_from_energy_and_NI: callable | None = None,
        is_relevant_psq_from_levels: callable | None = None,
    ) -> dict[int, dict[int, tuple[list, dict[str, ss.SigmondSampling]]]]:
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
                    if not is_relevant_level_from_energy_and_NI(psq_level_index, level_data):
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


# Example usage:
if __name__ == "__main__":
    # Create parser for L=32 lattice
    parser = PhiRhoProjectDataParser(L=32)

    # Get available datasets
    datasets = parser.get_available_datasets()
    print(f"Available datasets: {datasets}")

    if datasets:
        # Load energy levels for first dataset
        dataset_id = datasets[0]
        print(f"Loading data for dataset: {dataset_id}")

        ensemble_info, sampling_info, results = parser.load_energy_levels_with_NI(
            dataset_id=dataset_id, resampling_method="B", energy_types=["elab", "ecm"]
        )

        print(f"Ensemble: {ensemble_info.name}")
        print(f"Sampling: {sampling_info.method} with {sampling_info.num_resamplings} resamplings")
        print(f"Found data for PSQs: {list(results.keys())}")
