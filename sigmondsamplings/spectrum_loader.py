"""
SpectrumLoader for organizing energy level spectra from Sigmond samplings.
"""

import logging
from typing import List, Optional
from functools import cached_property
from .loader import SigmondLoader
from .sampling import SigmondSampling
from .energy_levels import EnergyObsInfo, SHEnergyObsInfo, create_energy_obs_info

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .spectra_collection import SpectraCollection


class SpectrumLoader(SigmondLoader):
    """
    Enhanced SigmondLoader that organizes energy level observables into spectra.

    Provides queryable SpectraCollection objects for interacting and single-hadron spectra:
    - loader.interacting_spectra - All multi-hadron energy levels
    - loader.single_hadron_spectra - All single-hadron energy levels

    Use SpectraCollection filtering methods to query:
        loader.interacting_spectra.filter(irrep='A1g', psq=0)
        loader.single_hadron_spectra.filter(particle='pi', psq=0)

    Or use convenience methods to get specific spectra:
        loader.get_interacting_spectrum(irrep='A1g', psq=0, energy_type='elab')
        loader.get_single_hadron_spectrum(particle='pi', psq=0, energy_type='elab')
    """

    def __init__(self, filename: str = None, **kwargs):
        """
        Initialize SpectrumLoader.

        Args:
            filename: Path to samplings file
            **kwargs: Arguments passed to SigmondLoader
        """
        super().__init__(filename, **kwargs)
        self._organized = False

        # Always organize on construction if samplings are loaded
        if self._all_samplings:
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

        # Clear cached properties
        for attr in ['interacting_spectra', 'single_hadron_spectra', 'irreps', 'psq_values', 'energy_types', 'particles']:
            if attr in self.__dict__:
                del self.__dict__[attr]

        # Convert all observables to energy levels
        energy_samplings = []
        for sampling in self._all_samplings:
            energy_sampling = sampling.as_energy_level()
            obs = energy_sampling.observable_info

            # Only keep actual energy level observables
            if not isinstance(obs, (EnergyObsInfo, SHEnergyObsInfo)):
                logging.warning(
                    f"ObservableInfo is not recognized as an energy level: {obs}. Skipping."
                )
                continue

            # Skip single hadrons missing particle name
            if isinstance(obs, SHEnergyObsInfo) and obs.particle is None:
                logging.warning(
                    f"SHEnergyObsInfo for single hadron missing particle name: {obs}. Skipping."
                )
                continue

            energy_samplings.append(energy_sampling)

        # Replace _all_samplings with energy level samplings
        temp_all_samplings = SpectraCollection(energy_samplings)
        if len(temp_all_samplings) != len(self._all_samplings):
            logging.warning(
                f"After organizing, {len(temp_all_samplings)} energy level samplings remain "
                f"out of {len(self._all_samplings)} total samplings."
            )
        self._all_samplings = SpectraCollection(energy_samplings)
        self._organized = True

    # Filtered Collections
    @cached_property
    def interacting_spectra(self) -> SpectraCollection:
        """All interacting (multi-hadron) energy level spectra.

        Returns SpectraCollection that can be further filtered:
            loader.interacting_spectra.filter(irrep='A1g', psq=0)
        """
        return self._all_samplings.find(
            lambda obs: isinstance(obs, EnergyObsInfo) and not isinstance(obs, SHEnergyObsInfo)
        )

    @cached_property
    def single_hadron_spectra(self) -> SpectraCollection:
        """All single hadron energy level spectra.

        Returns SpectraCollection that can be further filtered:
            loader.single_hadron_spectra.filter(particle='pi', psq=0)
        """
        return self._all_samplings.find(
            lambda obs: isinstance(obs, SHEnergyObsInfo)
        )

    # Discovery Properties
    @cached_property
    def irreps(self) -> List[str]:
        """All available irreps in interacting spectra."""
        return sorted(set(s.observable_info.irrep for s in self.interacting_spectra))

    @cached_property
    def psq_values(self) -> List[int]:
        """All available PSQ values in all spectra."""
        return sorted(set(s.observable_info.psq for s in self._all_samplings))

    @cached_property
    def energy_types(self) -> List[str]:
        """All available energy types in all spectra."""
        return sorted(set(s.observable_info.energy_type for s in self._all_samplings))

    @cached_property
    def particles(self) -> List[str]:
        """All particles found in single hadron spectra."""
        return sorted(set(
            s.observable_info.particle
            for s in self.single_hadron_spectra
            if s.observable_info.particle
        ))

    # Access Methods
    def get_interacting_spectrum(
        self,
        irrep: str,
        psq: int,
        energy_type: str,
        reference: bool = False,
    ) -> List[SigmondSampling]:
        """
        Get a single interacting (multi-hadron) spectrum for specific quantum numbers.

        Args:
            irrep: Irreducible representation (required)
            psq: Momentum squared (required)
            energy_type: Energy type (elab, ecm, delab, decm) (required)
            reference: Whether to get reference-divided spectrum

        Returns:
            List of SigmondSampling objects ordered by energy value
        """
        # Filter for matching interacting levels
        def predicate(obs_info):
            if not isinstance(obs_info, EnergyObsInfo) or isinstance(obs_info, SHEnergyObsInfo):
                return False
            ref = obs_info.ref_particle is not None
            return (obs_info.irrep == irrep and obs_info.psq == psq and
                    obs_info.energy_type == energy_type and ref == reference)

        filtered = self._all_samplings.find(predicate)
        # Sort by energy value
        levels = list(filtered)
        return sorted(levels, key=lambda x: x.full_sample_value)

    def get_single_hadron_spectrum(
        self,
        particle: str,
        psq: int,
        energy_type: str = "elab",
        reference: bool = False,
    ) -> Optional[SigmondSampling]:
        """
        Get a single single-hadron spectrum for specific quantum numbers.

        Args:
            particle: Particle name (required)
            psq: Momentum squared (required)
            energy_type: Energy type (default: 'elab')
            reference: Whether to get reference-divided spectrum

        Returns:
            Single SigmondSampling object for the particle, or None if not found
        """
        # Filter for matching single hadron
        def predicate(obs_info):
            if not isinstance(obs_info, SHEnergyObsInfo):
                return False
            ref = obs_info.ref_particle is not None
            return (obs_info.particle == particle and obs_info.psq == psq and
                    obs_info.energy_type == energy_type and ref == reference)

        filtered = self._all_samplings.find(predicate)
        # Return first match (should be only one)
        for sampling in filtered:
            return sampling
        return None


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

        for sampling in self._all_samplings:
            obs = sampling.observable_info
            # Apply reference filter
            ref = obs.ref_particle is not None
            if reference is not None and ref != reference:
                continue

            row = {
                "psq": obs.psq,
                "energy_type": obs.energy_type,
                "reference": ref,
                "value": sampling.full_sample_value,
                "mean": sampling.mean,
                "error": sampling.error,
                "observable_name": obs.name,
                "observable_index": obs.index,
            }

            # Add irrep for interacting (non-single-hadron)
            if isinstance(obs, EnergyObsInfo) and not isinstance(obs, SHEnergyObsInfo):
                row["irrep"] = obs.irrep
                row["hadron_type"] = "multi"
                row["particle"] = None
            else:
                row["irrep"] = None

            # Add particle info for single hadrons
            if isinstance(obs, SHEnergyObsInfo):
                row["particle"] = obs.particle
                row["hadron_type"] = "single"

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
        from collections import defaultdict

        # Group interacting levels by spectrum key
        spectra = defaultdict(list)

        for sampling in self._all_samplings:
            obs = sampling.observable_info
            # Only group interacting (non-single-hadron) levels
            if not isinstance(obs, EnergyObsInfo) or isinstance(obs, SHEnergyObsInfo):
                continue

            ref = obs.ref_particle is not None

            # Apply reference filter
            if reference is not None and ref != reference:
                continue

            spectrum_key = (obs.irrep, obs.psq, obs.energy_type, ref)
            spectra[spectrum_key].append(sampling)

        rows = []
        for spectrum_key, levels in spectra.items():
            irrep, psq, energy_type, ref = spectrum_key

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

    def __repr__(self):
        if not self._organized:
            return f"SpectrumLoader(file='{self._filename}', organized=False)"

        # Count unique spectra and total levels using collection directly
        multi_hadron = [
            s for s in self._all_samplings
            if isinstance(s.observable_info, EnergyObsInfo)
            and not isinstance(s.observable_info, SHEnergyObsInfo)
        ]

        spectra_keys = set(
            (obs.irrep, obs.psq, obs.energy_type, obs.ref_particle is not None)
            for s in multi_hadron
            if (obs := s.observable_info)
        )

        num_spectra = len(spectra_keys)
        num_levels = len(multi_hadron)
        return f"SpectrumLoader(file='{self._filename}', {num_spectra} spectra, {num_levels} levels)"
