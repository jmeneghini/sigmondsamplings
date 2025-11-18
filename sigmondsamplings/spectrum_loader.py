"""
SpectrumLoader for organizing energy level spectra from Sigmond samplings.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
from .loader import SigmondLoader
from .sampling import SigmondSampling
from .energy_levels import EnergyObsInfo, SHEnergyObsInfo, create_energy_obs_info

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class SpectrumLoader(SigmondLoader):
    """
    Enhanced SigmondLoader that organizes energy level observables into spectra.

    Organizes spectra by (irrep, psq, energy_type, reference) tuples where:
    - irrep: Irreducible representation (A1g, T1u, etc.)
    - psq: Momentum squared (0, 1, 4, 9, etc.)
    - energy_type: Energy type (elab, ecm, delab, decm)
    - reference: Whether energy is divided by reference mass (True/False)

    Each spectrum is a list of SigmondSampling objects ordered by energy value.
    """

    def __init__(self, filename: str = None, auto_organize: bool = True, **kwargs):
        """
        Initialize SpectrumLoader.

        Args:
            filename: Path to samplings file
            auto_organize: Automatically organize spectra after loading
            **kwargs: Arguments passed to SigmondLoader
        """
        super().__init__(filename, **kwargs)

        # Spectrum organization: {(irrep, psq, energy_type, reference): [SigmondSampling]}
        self._spectra = {}
        self._organized = False

        if auto_organize and self._all_samplings:
            self.organize_spectra()

    def organize_spectra(self, force_reorg: bool = False):
        """
        Organize loaded samplings into energy level spectra.

        Args:
            force_reorg: Force reorganization even if already organized
        """
        if self._organized and not force_reorg:
            return

        if not self._all_samplings:
            raise ValueError("No samplings loaded. Load a file first.")

        self._spectra = {}

        # Convert observables to energy levels and group into spectra
        spectrum_groups = defaultdict(list)

        # First pass: convert to energy levels and collect them
        energy_samplings = {}
        for key, sampling in list(self._all_samplings.items()):
            energy_sampling = sampling.as_energy_level()
            obs = energy_sampling.observable_info

            # Only organize actual energy level observables
            if not isinstance(obs, (EnergyObsInfo, SHEnergyObsInfo)):
                continue

            # Use standard key format
            standard_key = f"{obs.name} {obs.index}"
            energy_samplings[standard_key] = energy_sampling

            # Create spectrum key
            reference = obs.ref_particle is not None
            spectrum_key = (obs.irrep, obs.psq, obs.energy_type, reference)
            spectrum_groups[spectrum_key].append(energy_sampling)

        # Second pass: update _all_samplings with energy levels using standard keys
        self._all_samplings.clear()
        self._all_samplings.update(energy_samplings)

        # Sort each spectrum by energy value and store
        for spectrum_key, levels in spectrum_groups.items():
            # Sort by full sample value (energy)
            sorted_levels = sorted(levels, key=lambda x: x.full_sample_value)
            self._spectra[spectrum_key] = sorted_levels

        self._organized = True

    # Discovery Methods
    def list_irreps(self) -> List[str]:
        """List all available irreps in the spectra."""
        self._ensure_organized()
        irreps = set()
        for irrep, psq, energy_type, reference in self._spectra.keys():
            irreps.add(irrep)
        return sorted(list(irreps))

    def list_psq_values(self) -> List[int]:
        """List all available PSQ values in the spectra."""
        self._ensure_organized()
        psq_values = set()
        for irrep, psq, energy_type, reference in self._spectra.keys():
            psq_values.add(psq)
        return sorted(list(psq_values))

    def list_energy_types(self) -> List[str]:
        """List all available energy types in the spectra."""
        self._ensure_organized()
        energy_types = set()
        for irrep, psq, energy_type, reference in self._spectra.keys():
            energy_types.add(energy_type)
        return sorted(list(energy_types))

    def list_particles(self) -> List[str]:
        """List all particles found in single hadron spectra."""
        self._ensure_organized()
        particles = set()
        for sampling in self._all_samplings.values():
            obs = sampling.observable_info
            if isinstance(obs, SHEnergyObsInfo) and obs.particle:
                particles.add(obs.particle)
        return sorted(list(particles))

    def list_spectra(self) -> List[Tuple[str, int, str, bool]]:
        """List all available spectrum tuples (irrep, psq, energy_type, reference)."""
        self._ensure_organized()
        return sorted(list(self._spectra.keys()))

    # Access Methods
    def get_spectrum(
        self, irrep: str, psq: int, energy_type: str, reference: bool = False
    ) -> List[SigmondSampling]:
        """
        Get spectrum for specific quantum numbers.

        Args:
            irrep: Irreducible representation
            psq: Momentum squared
            energy_type: Energy type (elab, ecm, delab, decm)
            reference: Whether to get reference-divided spectrum

        Returns:
            List of SigmondSampling objects ordered by energy value
        """
        self._ensure_organized()
        spectrum_key = (irrep, psq, energy_type, reference)
        return self._spectra.get(spectrum_key, [])

    def get_spectra(
        self,
        irreps: List[str] = None,
        psq_values: List[int] = None,
        energy_types: List[str] = None,
        reference: bool = False,
    ) -> Dict[Tuple[str, int, str, bool], List[SigmondSampling]]:
        """
        Get multiple spectra matching filters.

        Args:
            irreps: List of irreps to include (None = all)
            psq_values: List of PSQ values to include (None = all)
            energy_types: List of energy types to include (None = all)
            reference: Whether to get reference-divided spectra

        Returns:
            Dictionary mapping spectrum tuples to lists of samplings
        """
        self._ensure_organized()

        result = {}
        for spectrum_key, levels in self._spectra.items():
            irrep, psq, energy_type, ref = spectrum_key

            # Apply filters
            if irreps is not None and irrep not in irreps:
                continue
            if psq_values is not None and psq not in psq_values:
                continue
            if energy_types is not None and energy_type not in energy_types:
                continue
            if ref != reference:
                continue

            result[spectrum_key] = levels

        return result

    def get_single_hadron_spectrum(
        self,
        particle: str,
        irrep: str = None,
        psq: int = None,
        energy_type: str = "elab",
        reference: bool = False,
    ) -> List[SigmondSampling]:
        """
        Get single hadron spectrum for specific particle.

        Args:
            particle: Particle name (pi, K, etc.)
            irrep: Specific irrep (None = any)
            psq: Specific PSQ (None = any)
            energy_type: Energy type
            reference: Whether to get reference-divided spectrum

        Returns:
            List of SigmondSampling objects for the particle
        """
        self._ensure_organized()

        result = []
        for sampling in self._all_samplings.values():
            obs = sampling.observable_info

            # Must be single hadron with matching particle
            if not isinstance(obs, SHEnergyObsInfo) or obs.particle != particle:
                continue

            # Check reference mode
            if (obs.ref_particle is not None) != reference:
                continue

            # Apply filters
            if irrep is not None and obs.irrep != irrep:
                continue
            if psq is not None and obs.psq != psq:
                continue
            if obs.energy_type != energy_type:
                continue

            result.append(sampling)

        # Sort by energy value
        return sorted(result, key=lambda x: x.full_sample_value)

    def to_dataframe(
        self,
        reference: bool = None,
        format: str = "long",
        include_metadata: bool = True,
    ) -> "pd.DataFrame":
        """
        Export spectra to pandas DataFrame.

        Args:
            reference: None=both, True=ref only, False=non-ref only
            format: 'long' (one row per level) or 'summary' (one row per spectrum)
            include_metadata: Include ensemble and sampling information

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for DataFrame export. Install with: pip install pandas"
            )

        self._ensure_organized()

        if format == "long":
            return self._create_long_dataframe(reference, include_metadata)
        elif format == "summary":
            return self._create_summary_dataframe(reference, include_metadata)
        else:
            raise ValueError(f"Invalid format '{format}'. Must be 'long' or 'summary'")

    def _create_long_dataframe(
        self, reference: bool, include_metadata: bool
    ) -> "pd.DataFrame":
        """Create long format DataFrame (one row per energy level)."""
        rows = []

        for spectrum_key, levels in self._spectra.items():
            irrep, psq, energy_type, ref = spectrum_key

            # Apply reference filter
            if reference is not None and ref != reference:
                continue

            for level_idx, sampling in enumerate(levels):
                obs = sampling.observable_info

                row = {
                    "irrep": irrep,
                    "psq": psq,
                    "energy_type": energy_type,
                    "reference": ref,
                    "level": level_idx,
                    "value": sampling.full_sample_value,
                    "mean": sampling.mean,
                    "error": sampling.error,
                    "observable_name": obs.name,
                    "observable_index": obs.index,
                }

                # Add particle info for single hadrons
                if isinstance(obs, SHEnergyObsInfo):
                    row["particle"] = obs.particle
                    row["hadron_type"] = "single"
                else:
                    row["particle"] = None
                    row["hadron_type"] = "multi"

                # Add particles list for multi-hadron
                if hasattr(obs, "particles") and obs.particles:
                    row["particles"] = obs.particles
                else:
                    row["particles"] = []

                # Add level index if available
                if hasattr(obs, "level_index") and obs.level_index is not None:
                    row["level_index"] = obs.level_index
                else:
                    row["level_index"] = None

                # Add reference particle info
                if hasattr(obs, "ref_particle"):
                    row["ref_particle"] = obs.ref_particle
                else:
                    row["ref_particle"] = None

                # Add metadata if requested
                if include_metadata:
                    row["ensemble"] = sampling.ensemble_info.ensemble_name
                    row["sampling_method"] = sampling.sampling_info.method
                    row["num_resamplings"] = sampling.sampling_info.num_resamplings

                rows.append(row)

        return pd.DataFrame(rows)

    def _create_summary_dataframe(
        self, reference: bool, include_metadata: bool
    ) -> "pd.DataFrame":
        """Create summary format DataFrame (one row per spectrum)."""
        rows = []

        for spectrum_key, levels in self._spectra.items():
            irrep, psq, energy_type, ref = spectrum_key

            # Apply reference filter
            if reference is not None and ref != reference:
                continue

            if not levels:
                continue

            # Collect particles in this spectrum
            particles = set()
            hadron_types = set()
            for sampling in levels:
                obs = sampling.observable_info
                if isinstance(obs, SHEnergyObsInfo) and obs.particle:
                    particles.add(obs.particle)
                    hadron_types.add("single")
                else:
                    hadron_types.add("multi")

            row = {
                "irrep": irrep,
                "psq": psq,
                "energy_type": energy_type,
                "reference": ref,
                "num_levels": len(levels),
                "particles": sorted(list(particles)),
                "hadron_types": sorted(list(hadron_types)),
                "energy_min": min(s.full_sample_value for s in levels),
                "energy_max": max(s.full_sample_value for s in levels),
            }

            # Add metadata if requested
            if include_metadata and levels:
                first_sampling = levels[0]
                row["ensemble"] = first_sampling.ensemble_info.ensemble_name
                row["sampling_method"] = first_sampling.sampling_info.method

            rows.append(row)

        return pd.DataFrame(rows)

    def _ensure_organized(self):
        """Ensure spectra are organized."""
        if not self._organized:
            self.organize_spectra()

    def __repr__(self):
        if not self._organized:
            return f"SpectrumLoader(file='{self._filename}', organized=False)"

        num_spectra = len(self._spectra)
        num_levels = sum(len(levels) for levels in self._spectra.values())
        return f"SpectrumLoader(file='{self._filename}', {num_spectra} spectra, {num_levels} levels)"
