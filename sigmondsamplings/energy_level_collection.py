"""
Energy-Level Collection Classes: Specialized collections for energy-level observables.

This module provides collection classes specifically designed for energy-level data:
- EnergyLevelMixin: Provides energy-level-specific helpers (irreps, psqs, etc.)
- SingleEnsembleEnergyCollection: Energy levels from one ensemble/sampling configuration
- MultiEnsembleEnergyCollection: Energy levels from multiple ensembles
"""

import logging
from collections.abc import Iterable
from typing import (
    TypeVar,
    Union,
)

from .energy_levels import EnergyObsInfo, Particle, SHEnergyObsInfo
from .ensemble_collection import MultiEnsembleCollection, SingleEnsembleCollection
from .sampling import EnsembleInfo, SigmondSampling

__all__ = [
    "EnergyLevelMixin",
    "SingleEnsembleEnergyCollection",
    "MultiEnsembleEnergyCollection",
]

T = TypeVar("T", bound="SingleEnsembleEnergyCollection")
M = TypeVar("M", bound="MultiEnsembleEnergyCollection")


class EnergyLevelMixin:
    """
    Mixin providing energy-level specific helpers and validation.

    This mixin adds energy-level discovery properties and convenience methods
    to collection classes. It assumes the collection has the standard
    ObservableCollection interface (obs accessor, filter, group_by, etc.).
    """

    # TODO: really need to consider if we want mutability here.
    _data: list[SigmondSampling]

    # -------------------------------------------------------------------------
    # Discovery Properties
    # -------------------------------------------------------------------------

    @property
    def irreps(self) -> list[str]:
        """
        All unique irreps in the collection, sorted.

        Returns:
            List[str]: Sorted list of irreducible representations
        """
        return self.unique("irrep")

    @property
    def psqs(self) -> Iterable[int]:
        """
        All unique PSQ (momentum squared) values in the collection, sorted.

        Returns:
            Iterable[int]: Sorted list/arr of momentum squared values
        """
        return self.unique("psq")

    @property
    def level_indexes(self) -> Iterable[int]:
        """
        All unique energy level indexes in the collection, sorted.

        Returns:
            Iterable[int]: Sorted list/arr of energy level indexes
        """
        return self.unique("level_index")

    @property
    def energy_types(self) -> list[str]:
        """
        All unique energy types in the collection, sorted.

        Energy types include: 'elab', 'ecm', 'delab', 'decm'

        Returns:
            List[str]: Sorted list of energy types
        """
        return self.unique("energy_type")

    @property
    def ref_particles(self) -> list[str]:
        """
        All unique reference particle names in the collection, sorted.

        Returns:
            List[str]: Sorted list of reference particle names
        """
        return self.unique("ref_particle")

    @property
    def particles(self) -> list[str]:
        """
        All unique particle names in the single hadron collection, sorted.

        Returns:
            List[str]: Sorted list of particle names
        """
        return self.single_hadron_spectra.unique("particle")

    @property
    def sectors(self) -> list[tuple[int, str]]:
        """
        All unique momentum-irrep sectors in the collection.

        Returns:
            List[Tuple[int, str]]: Sorted list of ``(psq, irrep)`` pairs.
        """
        return self.unique("sector")

    @property
    def psq_irrep_pairs(self) -> list[tuple[int, str]]:
        """
        Get all unique (PSQ, irrep) combinations in the collection.

        Returns:
            List[Tuple[int, str]]: Sorted list of (psq, irrep) pairs
        """
        return self.sectors

    def group_by_energy_type(self) -> dict[str, "EnergyLevelMixin"]:
        """
        Group collection by energy type.

        Convenience wrapper around group_by() for energy type-based grouping.

        Returns:
            Dict[str, Collection]: Dictionary mapping energy type to collection
        """
        return self.group_by(key="energy_type")

    def group_by_irrep(self) -> dict[str, "EnergyLevelMixin"]:
        """
        Group collection by irrep.

        Convenience wrapper around group_by() for irrep-based grouping.

        Returns:
            Dict[str, Collection]: Dictionary mapping irrep to collection
        """
        return self.group_by(key="irrep")

    def group_by_psq(self) -> dict[int, "EnergyLevelMixin"]:
        """
        Group collection by PSQ (momentum squared).

        Convenience wrapper around group_by() for PSQ-based grouping.

        Returns:
            Dict[int, Collection]: Dictionary mapping psq to collection
        """
        return self.group_by(key="psq")

    def group_by_level_index(self) -> dict[int, "EnergyLevelMixin"]:
        """
        Group collection by energy level index.

        Convenience wrapper around group_by() for level index-based grouping.

        Returns:
            Dict[int, Collection]: Dictionary mapping level index to collection
        """
        return self.group_by(key="level_index")

    def group_by_sector(self) -> dict[tuple[int, str], "EnergyLevelMixin"]:
        """
        Group collection by (PSQ, irrep) sector.

        A sector is defined by the combination of momentum squared and irrep.

        Returns:
            Dict[Tuple[int, str], Collection]: Dictionary mapping (psq, irrep) to collection
        """
        return self.group_by(key="sector")

    # -------------------------------------------------------------------------
    # Organize Spectra - Filtering by Type
    # -------------------------------------------------------------------------

    @property
    def interacting_spectra(self):
        """
        All interacting (multi-hadron) energy level spectra.

        Filters for EnergyObsInfo that are not single hadron types.

        Returns:
            Collection of the same type with only interacting energy levels

        Example:
            >>> # Get all interacting levels in A1g irrep at rest
            >>> interacting = collection.interacting_spectra.filter(irrep='A1g', psq=0)
        """
        return self.filter(
            predicate=lambda obs_info: (
                isinstance(obs_info, EnergyObsInfo) and not isinstance(obs_info, SHEnergyObsInfo)
            )
        )

    @property
    def single_hadron_spectra(self):
        """
        All single hadron energy level spectra.

        Filters for SHEnergyObsInfo types only.

        Returns:
            Collection of the same type with only single hadron energy levels

        Example:
            >>> # Get all pion energy levels
            >>> pions = collection.single_hadron_spectra.filter(particle='pi')
        """
        return self.filter(predicate=lambda obs_info: isinstance(obs_info, SHEnergyObsInfo))

    # -------------------------------------------------------------------------
    # Factory Method - Convert Observables to Energy Levels
    # -------------------------------------------------------------------------

    @classmethod
    def from_collection(
        cls,
        observables: Iterable[SigmondSampling],
        skip_missing_particles: bool = True,
        return_type: str = "numpy",
    ):
        """
        Create an energy-level collection from generic observables.

        Converts each observable to an energy level using `as_energy_level()`,
        filtering out incompatible observables.

        Args:
            observables: Iterable of SigmondSampling objects (may not be energy-level types)
            skip_missing_particles: If True, skip single-hadron observables missing particle names
            return_type: Return type for attribute access

        Returns:
            Energy-level collection of the same type as the class

        Raises:
            ValueError: If no valid energy-level observables remain after conversion

        Example:
            >>> # Convert generic observables to energy levels
            >>> energy_coll = SingleEnsembleEnergyCollection.from_observables(
            ...     generic_observables,
            ...     skip_missing_particles=True
            ... )
        """
        energy_samplings = []

        for sampling in observables:
            try:
                if isinstance(sampling.observable_info, (EnergyObsInfo, SHEnergyObsInfo)):
                    # Already an energy level, use as is (with canonical name)
                    sampling.observable_info.update_name()
                    energy_samplings.append(sampling)
                    continue
                energy_sampling = sampling.as_energy_level()
                obs = energy_sampling.observable_info
                # Skip single hadrons missing particle name if requested
                if skip_missing_particles:
                    if isinstance(obs, SHEnergyObsInfo) and obs.particle is None:
                        logging.warning(
                            f"SHEnergyObsInfo for single hadron missing particle name: {obs}. Skipping."
                        )
                        continue

                energy_samplings.append(energy_sampling)
            except (AttributeError, ValueError) as e:
                # Skip observables that can't be converted to energy levels
                logging.warning(
                    f"Could not convert observable {sampling.observable_info} to energy level: {e}. Skipping."
                )
                continue

        if not energy_samplings:
            raise ValueError("No valid energy-level observables found after conversion.")

        return cls(energy_samplings, return_type=return_type)

    def _validate_energy_levels(self) -> None:
        """
        Validate that all observables in the collection are energy-level types.
        """
        obs_types = set([type(obs.observable_info) for obs in self])
        invalid_types = obs_types - {EnergyObsInfo, SHEnergyObsInfo}
        if invalid_types:
            raise ValueError(
                f"Collection contains non-energy-level observable types: {invalid_types}"
            )

    # -------------------------------------------------------------------------
    # Reference and Shift Particle Setters (Mutable)
    # -------------------------------------------------------------------------

    def set_ref(self, particle_name: str) -> None:
        """
        Set reference particle for all observables with is_ref=True (mutable).

        This method mutates the observable_info in place for all energy levels
        that have is_ref=True.

        Args:
            particle_name: Name of the reference particle (e.g., 'L', 'pi')

        Example:
            >>> collection.set_ref('L')
            >>> # All ref observables now have ref_particle='L'
        """
        for sampling in self._data:
            obs_info = sampling.observable_info
            if hasattr(obs_info, "is_ref") and obs_info.is_ref:
                obs_info.ref_particle = particle_name

    def create_ref(self, particle_samp: SigmondSampling) -> None:
        """
        Create reference observables for all observables without is_ref = True (mutable).

        This method mutates the collection in place by creating new reference observables
        for all viable energy levels, using the provided particle observable.

        Args:
            particle_obs: SigmondSampling representing the particle to use as reference.
            If of type SHEnergyObsInfo, its particle name will be used as the reference particle name,
            otherwise the reference particle name will be set to 'ref'.
        Example:
            >>> # Create reference observables using a pion sampling
            >>> pion_samp = SigmondSampling(..., observable_info=SHEnergyObsInfo(particle='pi', ...))
            >>> collection.create_ref(pion_samp)
            >>> # All ref observables now have ref_particle='pi' and new reference samplings created
        """
        new_obs = []
        particle_samp = particle_samp.copy()
        for sampling in self._data:
            obs_info = sampling.observable_info
            if hasattr(obs_info, "is_ref") and not obs_info.is_ref:
                new_ref = sampling.create_ref_sampling(particle_samp)
                new_obs.append(new_ref)
        self._data.extend(new_obs)

    def set_shift_particles(
        self, irrep_psq_levels_map: dict[tuple[str, int, int], list[Particle]]
    ) -> None:
        """
        Set non-interacting particle names for shift-type observables (mutable).

        This method mutates the observable_info in place for shift-type energy levels.

        Args:
            irrep_psq_levels_map: Dict mapping (irrep, psq, level_idx) to list of Particle objects

        Example:
            >>> mapping = {
            ...     ('A1g', 0, 0): [Particle('pi', psq=0), Particle('pi', psq=1)],
            ...     ('A1g', 0, 1): [Particle('rho', psq=0), Particle('pi', psq=0)],
            ... }
            >>> collection.set_shift_particles(mapping)
        """
        for sampling in self._data:
            obs_info = sampling.observable_info
            if isinstance(obs_info, EnergyObsInfo) and obs_info.needs_ni_pair:
                key = (obs_info.irrep, obs_info.psq, obs_info.level_index)
                if key in irrep_psq_levels_map:
                    obs_info.particles = tuple(irrep_psq_levels_map[key])

    def _parse_pycalq_yml(self, yml_path: str) -> dict[tuple[str, int, int], list[Particle]]:
        """
        Parse a PyCalQ YAML file and extract shift particle assignments.

        Args:
            yml_path: Path to the PyCalQ YAML configuration file

        Returns:
            Dict mapping (irrep, psq, level_idx) to list of Particle objects
        """
        import re

        import yaml

        with open(yml_path) as f:
            config = yaml.safe_load(f)

        # Build regex patterns from collection's irreps and psq values
        irrep_pattern = (
            "(" + "|".join(re.escape(irrep) for irrep in self.irreps if irrep is not None) + ")"
        )

        psq_pattern = (
            "(" + "|".join(re.escape(str(psq)) for psq in self.psqs if psq is not None) + ")"
        )

        sector_pattern = rf"{irrep_pattern}\s+PSQ={psq_pattern}"

        # Navigate down through YAML until we find sector data
        current = config
        while isinstance(current, dict):
            keys = list(current.keys())

            # Check if this is the sector level using collection's irreps/psq
            if any(re.search(sector_pattern, k) for k in keys):
                break

            # Go down one level if there's only one key
            if len(keys) != 1:
                raise ValueError(f"Multiple branches at YAML level with keys: {keys}")
            current = current[keys[0]]

        # Build the irrep_psq_levels_map
        irrep_psq_levels_map = {}

        for sector_key, levels in current.items():
            # Parse sector string using collection's pattern
            irrep_match = re.search(sector_pattern, sector_key)
            if not irrep_match:
                continue

            irrep = irrep_match.group(1)
            psq = int(irrep_match.group(2))

            # Parse levels
            for level_idx, particle_pairs in enumerate(levels):
                shift_particles = []
                for particle_str in particle_pairs:
                    shift_particles.append(Particle.from_string(particle_str))
                irrep_psq_levels_map[(irrep, psq, level_idx)] = shift_particles

        return irrep_psq_levels_map

    def create_pycalq_yml_shift_particles(self, yml_path: str) -> None:
        """
        Create a PyCalQ YAML file with shift particle assignments based on the collection's
        current shift-type observables.

        Output format matches the PyCalQ non_interacting_levels YAML structure
        expected by set_shift_particles_from_pycalq_yml / _parse_pycalq_yml.

        Args:
            yml_path: Path to write the PyCalQ YAML configuration file
        """
        import yaml

        sectors = {}
        for sampling in self._data:
            obs_info = sampling.observable_info
            if (
                isinstance(obs_info, EnergyObsInfo)
                and obs_info.is_shift_type
                and obs_info.particles
            ):
                sector_key = f"{obs_info.irrep} PSQ={obs_info.psq}"
                if sector_key not in sectors:
                    sectors[sector_key] = {}
                sectors[sector_key][obs_info.level_index] = [
                    str(p) for p in obs_info.particles
                ]

        # Convert to sorted lists indexed by level (matching parser's enumerate expectation)
        non_interacting_levels = {}
        for sector_key, levels in sectors.items():
            max_idx = max(levels.keys())
            non_interacting_levels[sector_key] = [levels.get(i, []) for i in range(max_idx + 1)]

        # Nest under wrapper keys to match PyCalQ YAML structure
        output = {"fit_spectrum": {"non_interacting_levels": non_interacting_levels}}

        with open(yml_path, "w") as f:
            yaml.dump(output, f, default_flow_style=False)

    # -------------------------------------------------------------------------
    # Spec Filtering and Persistence
    # -------------------------------------------------------------------------

    def filter_by_spec(
        self,
        spec: Iterable[tuple],
    ):
        """
        Filter collection to only include observables matching the given spec.

        Each entry in ``spec`` is a tuple of the form:
        - ``(psq, irrep, n_levels)``       – number of levels [0, ..., N-1]
        - ``(psq, irrep, [level_index, ...])`` – multiple levels (flattened)

        Args:
            spec: Iterable of (psq, irrep, n_levels_or_list) tuples

        Returns:
            Collection of the same type containing only matching observables

        Example:
            >>> result = coll.filter_by_spec([(0, 'A1g', 0), (1, 'E', [0, 1])])
        """
        allowed: set[tuple[int, str, int]] = set()
        for entry in spec:
            psq, irrep, level = entry
            if isinstance(level, list):
                for lvl in level:
                    allowed.add((psq, irrep, lvl))
            else:
                allowed.add((psq, irrep, level))

        return self.filter(
            predicate=lambda obs_info: (
                (
                    obs_info.psq,
                    obs_info.irrep,
                    obs_info.level_index,
                )
                in allowed
            )
        )

    @property
    def spec(self) -> list[tuple[int, str, list[int]]]:
        """
        List the (psq, irrep, [level_indices]) spec of the current collection.

        Returns:
            List[Tuple[int, str, List[int]]]: List of sectors with their embedded level indices.

        Example:
            >>> spec = coll.spec
            >>> print(spec)
            [(0, 'A1g', [0, 1]), (1, 'E', [0])]
        """
        return [
            (psq, irrep, sorted(set(obs.observable_info.level_index for obs in sub_coll)))
            for (psq, irrep), sub_coll in sorted(
                item for item in self.group_by_sector().items() if None not in item[0]
            )
        ]

    def save_spec(self, toml_path: str) -> None:
        """
        Save the current collection's (psq, irrep, level_index) spec to a TOML file.

        The TOML groups level indices by (psq, irrep) sector under a ``spectrum``
        array of tables:

        .. code-block:: toml

            [[spectrum]]
            psq = 0
            irrep = "A1g"
            levels = [0, 1, 2]

            [[spectrum]]
            psq = 1
            irrep = "E"
            levels = [0]

        Args:
            toml_path: Path to write the spec TOML file

        Example:
            >>> coll.save_spec('my_spec.toml')
        """
        import tomlkit

        sectors: dict[tuple[int, str], set[int]] = {}
        for sampling in self._data:
            obs_info = sampling.observable_info
            key = (obs_info.psq, obs_info.irrep)
            sectors.setdefault(key, set()).add(obs_info.level_index)

        spec_entries = tomlkit.aot()
        for (psq, irrep), levels in sorted(
            item for item in sectors.items() if None not in item[0]
        ):
            entry = tomlkit.table()
            entry.add("psq", psq)
            entry.add("irrep", irrep)
            entry.add("levels", sorted(levels))
            spec_entries.append(entry)

        document = tomlkit.document()
        document.add("spectrum", spec_entries)

        with open(toml_path, "w") as f:
            tomlkit.dump(document, f)

    def filter_from_toml(self, toml_path: str):
        """
        Load a spec from a TOML file and filter this collection to match it.

        Reads a TOML file produced by :meth:`save_spec` and delegates to
        :meth:`filter_by_spec`.

        Args:
            toml_path: Path to a spec TOML file

        Returns:
            Collection of the same type filtered to the spec

        Example:
            >>> filtered = coll.filter_from_toml('my_spec.toml')
        """
        import tomlkit

        with open(toml_path) as f:
            config = tomlkit.load(f)

        spec = [(entry["psq"], entry["irrep"], entry["levels"]) for entry in config["spectrum"]]
        return self.filter_by_spec(spec)

    def set_shift_particles_from_pycalq_yml(self, yml_path: str) -> None:
        """
        Set non-interacting particle names from a PyCalQ YAML configuration file (mutable).

        This method parses a PyCalQ YAML file and extracts shift particle assignments,
        then applies them to the collection using set_shift_particles().

        Args:
            yml_path: Path to the PyCalQ YAML configuration file

        Example:
            >>> collection.set_shift_particles_from_pycalq_yml('config.yml')
        """
        self.set_shift_particles(self._parse_pycalq_yml(yml_path))


class SingleEnsembleEnergyCollection(SingleEnsembleCollection, EnergyLevelMixin):
    """
    Energy-level observable collection from a single ensemble/sampling configuration.

    Extends SingleEnsembleCollection with energy-level specific helpers.
    All observables must be EnergyObsInfo or SHEnergyObsInfo types.

    Provides:
    - Energy-level validation
    - Discovery properties (irreps, psq_values, energy_types, particles)
    - Convenience grouping methods (group_by_irrep, group_by_psq, group_by_sector)
    - All SingleEnsembleCollection features (filter, sort, etc.)

    Example:
    --------
    >>> # Create from energy level samplings
    >>> coll = SingleEnsembleEnergyCollection(energy_samplings)
    >>>
    >>> # Discover available irreps and PSQs
    >>> print(f"Irreps: {coll.irreps}")
    >>> print(f"PSQs: {coll.psq_values}")
    >>>
    >>> # Get (psq, irrep) pairs
    >>> sectors = coll.psq_irrep_pairs()
    >>>
    >>> # Group by irrep
    >>> by_irrep = coll.group_by_irrep()
    >>> a1g_levels = by_irrep["A1g"]
    """

    def __init__(
        self,
        data: Iterable[SigmondSampling],
        return_type: str = "numpy",
    ):
        """
        Initialize SingleEnsembleEnergyCollection with optional auto-conversion.

        Args:
            data: Iterable of SigmondSampling objects
            return_type: Return type for attribute access - 'dict', 'list', or 'numpy'

        Raises:
            ValueError: If observables have different ensemble_info/sampling_info,
                       or if any observable is not an energy-level type and auto_convert=False
        """
        # Call parent to validate single ensemble/sampling
        super().__init__(data, return_type)

        # Validate energy-level types (should always pass after conversion)
        self._validate_energy_levels()

    def __repr__(self) -> str:
        ensemble_id = (
            self._ensemble_info.id
            if hasattr(self._ensemble_info, "id")
            else str(self._ensemble_info)
        )
        sampling_str = str(self._sampling_info)
        return (
            f"SingleEnsembleEnergyCollection("
            f"n_obs={len(self._data)}, "
            f"ensemble='{ensemble_id}', "
            f"sampling='{sampling_str}'"
            f")"
        )


class MultiEnsembleEnergyCollection(MultiEnsembleCollection, EnergyLevelMixin):
    """
    Energy-level observable collection from multiple ensembles.

    Extends MultiEnsembleCollection with energy-level specific helpers.
    All observables must be EnergyObsInfo or SHEnergyObsInfo types.

    Provides:
    - Energy-level validation
    - Discovery properties (irreps, psq_values, energy_types, particles)
    - Convenience grouping methods (group_by_irrep, group_by_psq, group_by_sector)
    - All MultiEnsembleCollection features (filter, by_ensemble, etc.)

    Type Consistency:
    - `by_ensemble` returns Dict[EnsembleInfo, SingleEnsembleEnergyCollection]
    - `__getitem__` with EnsembleInfo returns SingleEnsembleEnergyCollection
    - Filter/sort operations return MultiEnsembleEnergyCollection

    Example:
    --------
    >>> # Create from mixed ensemble data
    >>> multi = MultiEnsembleEnergyCollection(all_energy_samplings)
    >>>
    >>> # Discover across all ensembles
    >>> print(f"All irreps: {multi.irreps}")
    >>> print(f"All PSQs: {multi.psq_values}")
    >>>
    >>> # Filter and maintain type
    >>> a1g_data = multi.filter(irrep="A1g", psq=0)
    >>> assert isinstance(a1g_data, MultiEnsembleEnergyCollection)
    >>>
    >>> # Access by ensemble (returns energy-level type)
    >>> for ens_info, energy_coll in multi.by_ensemble.items():
    ...     assert isinstance(energy_coll, SingleEnsembleEnergyCollection)
    ...     print(f"{ens_info}: {energy_coll.irreps}")
    """

    def __init__(
        self,
        data: Iterable[SigmondSampling] | dict[EnsembleInfo, SingleEnsembleEnergyCollection],
        return_type: str = "numpy",
    ):
        """
        Initialize MultiEnsembleEnergyCollection with optional auto-conversion.

        Parameters:
        -----------
        data : Union[Iterable[SigmondSampling], Dict, MultiEnsembleEnergyCollection]
            Input data - can be:
            - Iterable of SigmondSampling objects
            - Dict mapping EnsembleInfo to SingleEnsembleEnergyCollection
            - Another MultiEnsembleEnergyCollection (copy)
        return_type : str
            Return type for attribute accessors ('list', 'dict', or 'numpy')

        Raises:
            ValueError: If any observable is not an energy-level type and auto_convert=False
        """
        # Call parent to validate single ensemble/sampling
        super().__init__(data, return_type)

        # Validate energy-level types (should always pass after conversion)
        self._validate_energy_levels()

    # -------------------------------------------------------------------------
    # Override Ensemble Properties to Return Energy-Level Types
    # -------------------------------------------------------------------------

    @property
    def by_ensemble(self) -> dict[EnsembleInfo, SingleEnsembleEnergyCollection]:
        """
        Group data by ensemble, returning Dict[EnsembleInfo, SingleEnsembleEnergyCollection].

        Overrides parent to return SingleEnsembleEnergyCollection instead of
        SingleEnsembleCollection for type consistency.

        Returns:
            Dict[EnsembleInfo, SingleEnsembleEnergyCollection]: Energy-level collections by ensemble
        """
        # Use inherited group_by to group by ensemble_info directly
        groups = self.group_by(values=self.obs.ensemble_info)

        # Convert each group to SingleEnsembleEnergyCollection (energy-level specific type)
        return {
            ens: SingleEnsembleEnergyCollection(group, return_type=self._return_type)
            for ens, group in groups.items()
        }

    # -------------------------------------------------------------------------
    # Override Dict-like Access to Return Energy-Level Types
    # -------------------------------------------------------------------------

    def __getitem__(
        self, key: EnsembleInfo | int | slice
    ) -> Union[SingleEnsembleEnergyCollection, "MultiEnsembleEnergyCollection", SigmondSampling]:
        """
        Access by EnsembleInfo, index, or slice.

        Overrides parent to return energy-level collection types.

        Parameters:
        -----------
        key : Union[EnsembleInfo, int, slice]
            - EnsembleInfo: Get SingleEnsembleEnergyCollection for that ensemble
            - int: Get sampling by index
            - slice: Get subset as MultiEnsembleEnergyCollection

        Returns:
        --------
        Union[SingleEnsembleEnergyCollection, MultiEnsembleEnergyCollection, SigmondSampling]
            Appropriate type based on key
        """
        if isinstance(key, EnsembleInfo):
            return self.by_ensemble[key]
        else:
            # Use parent's indexing logic
            result = super(MultiEnsembleCollection, self).__getitem__(key)
            if isinstance(key, slice):
                # Return MultiEnsembleEnergyCollection for slices
                return self._fast_load(result._data, self._return_type)
            # Return individual SigmondSampling for int index
            return result

    def __repr__(self):
        n_ensembles = len(self.ensembles)
        total = len(self)
        return f"MultiEnsembleEnergyCollection(ensembles={n_ensembles}, total_obs={total})"
