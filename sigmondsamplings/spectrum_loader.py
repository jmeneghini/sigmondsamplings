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

from .obervable_collection import ObservableCollection


class SpectrumLoader(SigmondLoader):
    """
    Enhanced SigmondLoader that organizes energy level observables into spectra.

    Provides queryable ObservableCollection objects for interacting and single-hadron spectra:
    - loader.interacting_spectra - All multi-hadron energy levels
    - loader.single_hadron_spectra - All single-hadron energy levels

    Use ObservableCollection filtering methods to query:
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
        temp_all_samplings = ObservableCollection(energy_samplings)
        if len(temp_all_samplings) != len(self._all_samplings):
            logging.warning(
                f"After organizing, {len(temp_all_samplings)} energy level samplings remain "
                f"out of {len(self._all_samplings)} total samplings."
            )
        self._all_samplings = ObservableCollection(energy_samplings)
        self._organized = True

    # Filtered Collections
    @cached_property
    def interacting_spectra(self) -> ObservableCollection:
        """All interacting (multi-hadron) energy level spectra.

        Returns ObservableCollection that can be further filtered:
            loader.interacting_spectra.filter(irrep='A1g', psq=0)
        """
        return self._all_samplings.filter(
            predicate = lambda obs: isinstance(obs, EnergyObsInfo) and not isinstance(obs, SHEnergyObsInfo)
        )

    @cached_property
    def single_hadron_spectra(self) -> ObservableCollection:
        """All single hadron energy level spectra.

        Returns ObservableCollection that can be further filtered:
            loader.single_hadron_spectra.filter(particle='pi', psq=0)
        """
        return self._all_samplings.filter(
            predicate = lambda obs: isinstance(obs, SHEnergyObsInfo)
        )

    # Discovery Properties
    @cached_property
    def irreps(self) -> List[str]:
        """All available irreps in interacting spectra."""
        return sorted(set(self.interacting_spectra.obs.irrep))

    @cached_property
    def psq_values(self) -> List[int]:
        """All available PSQ values in all spectra."""
        return sorted(set(self._all_samplings.obs_info.psq))

    @cached_property
    def energy_types(self) -> List[str]:
        """All available energy types in all spectra."""
        return sorted(set(self._all_samplings.obs_info.energy_type))

    @cached_property
    def particles(self) -> List[str]:
        """All particles found in single hadron spectra."""
        return sorted(set(
            s.observable_info.particle
            for s in self.single_hadron_spectra
            if s.observable_info.particle
        ))

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
