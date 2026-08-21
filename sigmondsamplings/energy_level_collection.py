"""Specialized collection types for energy-level observables.

Classes
-------
EnergyLevelMixin
    Energy-specific discovery, grouping, filtering, mutation, and persistence helpers.
SingleEnsembleEnergyCollection
    Energy levels from one ensemble and sampling configuration.
MultiEnsembleEnergyCollection
    Energy levels spanning multiple ensembles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

from .energy_levels import EnergyObsInfo, Particle, SHEnergyObsInfo
from .ensemble_collection import MultiEnsembleCollection, SingleEnsembleCollection
from .sampling import EnsembleInfo, SigmondSampling

if TYPE_CHECKING:
    from .observable_collection import ObservableCollection
    from .spectrum_spec import SpectrumResolved

    class _EnergyLevelCollectionBase(ObservableCollection):
        """Collection API required by :class:`EnergyLevelMixin`."""

else:

    class _EnergyLevelCollectionBase:
        """Keep the typing-only collection base out of the runtime MRO."""

__all__ = [
    "EnergyLevelMixin",
    "SingleEnsembleEnergyCollection",
    "MultiEnsembleEnergyCollection",
]

class EnergyLevelMixin(_EnergyLevelCollectionBase):
    """Provide energy-level-specific collection helpers and validation.

    Notes
    -----
    The host class must provide the standard :class:`ObservableCollection`
    interface, including ``obs``, ``filter``, ``group_by``, and ``unique``.
    """

    # TODO: really need to consider if we want mutability here.
    _data: list[SigmondSampling]

    # -------------------------------------------------------------------------
    # Construction and Validation
    # -------------------------------------------------------------------------

    @classmethod
    def from_collection(
        cls,
        observables: Iterable[SigmondSampling],
        skip_missing_particles: bool = True,
        return_type: str = "numpy",
    ):
        """Create an energy-level collection from generic observables.

        Convert each observable using :meth:`SigmondSampling.as_energy_level`,
        filtering out incompatible observables.

        Parameters
        ----------
        observables : Iterable[SigmondSampling]
            Samplings that may or may not already carry energy-level metadata.
        skip_missing_particles : bool, default=True
            Skip single-hadron observables whose particle name is unavailable.
        return_type : {"numpy", "list", "dict"}, default="numpy"
            Representation used by collection attribute accessors.

        Returns
        -------
        EnergyLevelMixin
            An energy-level collection of the concrete class on which this
            method was called.

        Raises
        ------
        ValueError
            If no valid energy-level observables remain after conversion.

        Examples
        --------
        >>> energy_coll = SingleEnsembleEnergyCollection.from_collection(
        ...     generic_observables,
        ...     skip_missing_particles=True,
        ... )
        """
        energy_samplings = []

        for sampling in observables:
            try:
                if isinstance(sampling.observable_info, EnergyObsInfo):
                    # Preserve the input while canonicalizing an independent metadata copy.
                    energy_obs_info = sampling.observable_info.copy()
                    energy_obs_info.update_name()
                    energy_sampling = sampling.with_observable_info(energy_obs_info)
                else:
                    energy_sampling = sampling.as_energy_level()
                obs = energy_sampling.observable_info
                # Skip single hadrons missing particle name if requested
                if (
                    skip_missing_particles
                    and isinstance(obs, SHEnergyObsInfo)
                    and obs.particle is None
                ):
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
        """Validate that every observable carries energy-level metadata.

        Raises
        ------
        ValueError
            If any observable is not represented by :class:`EnergyObsInfo` or
            one of its subclasses.
        """
        invalid_types = {
            type(sampling.observable_info)
            for sampling in self
            if not isinstance(sampling.observable_info, EnergyObsInfo)
        }
        if invalid_types:
            raise ValueError(
                f"Collection contains non-energy-level observable types: {invalid_types}"
            )

    # -------------------------------------------------------------------------
    # Discovery Properties
    # -------------------------------------------------------------------------

    @property
    def irreps(self) -> list[str]:
        """Return the unique irreducible representations in sorted order.

        Returns
        -------
        list[str]
            Sorted irreducible-representation names.
        """
        return cast(list[str], self.unique("irrep"))

    @property
    def psqs(self) -> Iterable[int]:
        """Return the unique momentum-squared values in sorted order.

        Returns
        -------
        Iterable[int]
            Sorted momentum-squared values.
        """
        return cast(Iterable[int], self.unique("psq"))

    @property
    def level_indexes(self) -> Iterable[int]:
        """Return the unique energy-level indices in sorted order.

        Returns
        -------
        Iterable[int]
            Sorted energy-level indices.
        """
        return cast(Iterable[int], self.unique("level_index"))

    @property
    def energy_types(self) -> list[str]:
        """Return the unique energy types in sorted order.

        Returns
        -------
        list[str]
            Sorted energy types, such as ``"elab"``, ``"ecm"``, ``"delab"``,
            and ``"decm"``.
        """

        return cast(list[str], self.unique("energy_type"))

    @property
    def ref_particles(self) -> list[str]:
        """Return the unique reference-particle names in sorted order.

        Returns
        -------
        list[str]
            Sorted reference-particle names.
        """
        return cast(list[str], self.unique("ref_particle"))

    @property
    def particles(self) -> list[str]:
        """Return unique single-hadron particle names in sorted order.

        Returns
        -------
        list[str]
            Sorted particle names from the single-hadron subset.
        """
        return cast(list[str], self.single_hadron_spectra.unique("particle"))

    @property
    def sectors(self) -> list[tuple[int, str]]:
        """Return the unique momentum-irrep sectors.

        Returns
        -------
        list[tuple[int, str]]
            Sorted ``(psq, irrep)`` pairs.
        """
        return cast(list[tuple[int, str]], self.unique("sector"))

    @property
    def psq_irrep_pairs(self) -> list[tuple[int, str]]:
        """Return the unique momentum-irrep pairs.

        Returns
        -------
        list[tuple[int, str]]
            Alias for :attr:`sectors`.
        """
        return self.sectors

    # -------------------------------------------------------------------------
    # Spectral Views
    # -------------------------------------------------------------------------

    @property
    def interacting_spectra(self):
        """Return the interacting, multi-hadron energy levels.

        Returns
        -------
        EnergyLevelMixin
            A collection of the same concrete type containing only
            :class:`EnergyObsInfo` entries that are not single-hadron entries.

        Examples
        --------
        >>> interacting = collection.interacting_spectra.filter(irrep="A1g", psq=0)
        """
        return self.filter(
            predicate=lambda obs_info: (
                isinstance(obs_info, EnergyObsInfo) and not isinstance(obs_info, SHEnergyObsInfo)
            )
        )

    @property
    def single_hadron_spectra(self):
        """Return the single-hadron energy levels.

        Returns
        -------
        EnergyLevelMixin
            A collection of the same concrete type containing only
            :class:`SHEnergyObsInfo` entries.

        Examples
        --------
        >>> pions = collection.single_hadron_spectra.filter(particle="pi")
        """
        return self.filter(predicate=lambda obs_info: isinstance(obs_info, SHEnergyObsInfo))

    # -------------------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------------------

    def group_by_energy_type(self) -> dict[str, EnergyLevelMixin]:
        """Group the collection by energy type.

        Returns
        -------
        dict[str, EnergyLevelMixin]
            Energy type mapped to a collection of the same concrete type.
        """
        return cast(dict[str, EnergyLevelMixin], self.group_by(key="energy_type"))

    def group_by_irrep(self) -> dict[str, EnergyLevelMixin]:
        """Group the collection by irrep.

        Returns
        -------
        dict[str, EnergyLevelMixin]
            Irrep name mapped to a collection of the same concrete type.
        """
        return cast(dict[str, EnergyLevelMixin], self.group_by(key="irrep"))

    def group_by_psq(self) -> dict[int, EnergyLevelMixin]:
        """Group the collection by momentum squared.

        Returns
        -------
        dict[int, EnergyLevelMixin]
            Momentum squared mapped to a collection of the same concrete type.
        """
        return cast(dict[int, EnergyLevelMixin], self.group_by(key="psq"))

    def group_by_level_index(self) -> dict[int, EnergyLevelMixin]:
        """Group the collection by energy-level index.

        Returns
        -------
        dict[int, EnergyLevelMixin]
            Level index mapped to a collection of the same concrete type.
        """
        return cast(dict[int, EnergyLevelMixin], self.group_by(key="level_index"))

    def group_by_sector(self) -> dict[tuple[int, str], EnergyLevelMixin]:
        """Group the collection by ``(psq, irrep)`` sector.

        Returns
        -------
        dict[tuple[int, str], EnergyLevelMixin]
            Sector mapped to a collection of the same concrete type.
        """
        return cast(dict[tuple[int, str], EnergyLevelMixin], self.group_by(key="sector"))

    # -------------------------------------------------------------------------
    # Spectrum Selection and Persistence
    # -------------------------------------------------------------------------

    def filter_by_spec(
        self,
        spec: Iterable[tuple[int, str, int | Iterable[int]]],
    ):
        """Filter the collection using a spectrum specification.

        Each entry in ``spec`` is a tuple of the form:

        - ``(psq, irrep, level_index)``         – a single level index
        - ``(psq, irrep, [level_index, ...])``  – multiple level indices

        Entries are resolved through
        :class:`~sigmondsamplings.spectrum_spec.SectorSpec`. The same ``(psq, irrep)``
        sector may appear in more than one entry; the selected indices are unioned.

        Parameters
        ----------
        spec : Iterable[tuple[int, str, int | Iterable[int]]]
            ``(psq, irrep, levels)`` entries, where ``levels`` is either one
            index or an iterable of indices.

        Returns
        -------
        EnergyLevelMixin
            A collection of the same concrete type containing matching levels.

        Examples
        --------
        >>> result = coll.filter_by_spec([(0, "A1g", 0), (1, "E", [0, 1])])
        """
        from .spectrum_spec import SectorSpec

        allowed: set[tuple[int, str, int]] = set()
        for psq, irrep, levels in spec:
            level_list = [levels] if isinstance(levels, int) else list(levels)
            sector = SectorSpec(psq=psq, irrep=irrep, levels=level_list).resolve()
            allowed.update(sector.keys())

        return self._filter_by_level_keys(allowed)

    def _filter_by_level_keys(self, allowed: set[tuple[int, str, int]]):
        """Filter against resolved ``(psq, irrep, level_index)`` keys.

        Parameters
        ----------
        allowed : set[tuple[int, str, int]]
            Exact level keys to retain.

        Returns
        -------
        EnergyLevelMixin
            A collection of the same concrete type containing allowed levels.
        """
        return self.filter(
            predicate=lambda obs_info: (
                isinstance(obs_info, EnergyObsInfo)
                and (obs_info.psq, obs_info.irrep, obs_info.level_index) in allowed
            )
        )

    def spectrum_spec(self) -> SpectrumResolved:
        """Build the resolved spectrum specification for this collection.

        Level indices are gathered per ``(psq, irrep)`` sector into a
        :class:`~sigmondsamplings.spectrum_spec.SpectrumResolved`. Observables whose
        psq, irrep, or level index is unset are skipped.

        Returns
        -------
        SpectrumResolved
            Canonical spectrum selection for the collection.
        """
        from .spectrum_spec import SectorResolved, SpectrumResolved

        sectors: dict[tuple[int, str], set[int]] = {}
        for sampling in self._data:
            obs_info = sampling.observable_info
            if not isinstance(obs_info, EnergyObsInfo):
                continue
            if (
                obs_info.psq is None
                or obs_info.irrep is None
                or obs_info.level_index is None
            ):
                continue
            key = (obs_info.psq, obs_info.irrep)
            sectors.setdefault(key, set()).add(obs_info.level_index)

        return SpectrumResolved(
            spectrum=[
                SectorResolved(psq=psq, irrep=irrep, levels=sorted(levels))
                for (psq, irrep), levels in sorted(sectors.items())
            ]
        )

    def save_spec(self, toml_path: str) -> None:
        """Save the current spectrum specification to TOML.

        Delegates serialization to
        :class:`~sigmondsamplings.spectrum_spec.SpectrumSpec`, writing the level
        indices grouped by (psq, irrep) sector under a ``spectrum`` array of tables.

        Parameters
        ----------
        toml_path : str
            Destination path for the TOML file.
        """
        self.spectrum_spec().to_spec().to_toml(toml_path)

    def filter_from_toml(self, toml_path: str):
        """Load a spectrum specification from TOML and filter the collection.

        Parameters
        ----------
        toml_path : str
            Path to a spectrum-specification TOML file.

        Returns
        -------
        EnergyLevelMixin
            A collection of the same concrete type containing matching levels.
        """
        from .spectrum_spec import SpectrumSpec

        resolved = SpectrumSpec.from_file(toml_path).resolve()
        return self._filter_by_level_keys(resolved.allowed_keys())

    # -------------------------------------------------------------------------
    # Reference and Shift-Particle Mutations
    # -------------------------------------------------------------------------

    def set_ref(self, particle_name: str) -> None:
        """Set the reference particle for all reference-mode observables.

        Parameters
        ----------
        particle_name : str
            Reference-particle name, such as ``"L"`` or ``"pi"``.

        Notes
        -----
        This method mutates matching observable metadata in place, regenerates
        canonical names, and invalidates cached shared attributes.

        Examples
        --------
        >>> collection.set_ref("L")
        """
        changed = False
        for sampling in self._data:
            obs_info = sampling.observable_info
            if (
                isinstance(obs_info, EnergyObsInfo)
                and obs_info.is_ref
                and obs_info.ref_particle != particle_name
            ):
                obs_info.ref_particle = particle_name
                obs_info.update_name(strict=False)
                changed = True
        if changed:
            self.clear_shared_attr_cache()

    def create_ref(self, particle_samp: SigmondSampling) -> list[SigmondSampling]:
        """Create missing reference observables using a particle sampling.

        Parameters
        ----------
        particle_samp : SigmondSampling
            Particle sampling used as the denominator. If it carries
            :class:`SHEnergyObsInfo`, its particle name becomes the reference
            particle; otherwise ``"ref"`` is used.

        Returns
        -------
        list[SigmondSampling]
            Reference samplings created and appended to this collection. The
            list is empty if every reference observable already exists.

        Notes
        -----
        The operation is idempotent. Existing reference observables are identified
        using exact :class:`EnergyObsInfo` equality, including ``ref_particle``.

        Examples
        --------
        >>> created = collection.create_ref(pion_samp)
        """
        existing = {samp.observable_info for samp in self._data}
        new_obs = []
        particle_samp = particle_samp.copy()
        for sampling in self._data:
            obs_info = sampling.observable_info
            if isinstance(obs_info, EnergyObsInfo) and not obs_info.is_ref:
                # Predict the metadata first: skipping here avoids the division that
                # create_ref_sampling would otherwise perform and then discard.
                ref_info = sampling.ref_observable_info(particle_samp)
                if ref_info in existing:
                    continue
                existing.add(ref_info)
                new_obs.append(sampling.create_ref_sampling(particle_samp))
        if new_obs:
            self._data.extend(new_obs)
            self.clear_shared_attr_cache()
        return new_obs

    def set_shift_particles(
        self, irrep_psq_levels_map: dict[tuple[str, int, int], list[Particle]]
    ) -> None:
        """Set non-interacting particles for shift-compatible observables.

        Parameters
        ----------
        irrep_psq_levels_map : dict[tuple[str, int, int], list[Particle]]
            Map ``(irrep, psq, level_index)`` keys to particle lists.

        Notes
        -----
        This method mutates matching observable metadata in place, regenerates
        canonical names, and invalidates cached shared attributes.

        Examples
        --------
        >>> assignments = {
        ...     ("A1g", 0, 0): [Particle("pi", psq=0), Particle("pi", psq=1)],
        ...     ("A1g", 0, 1): [Particle("rho", psq=0), Particle("pi", psq=0)],
        ... }
        >>> collection.set_shift_particles(assignments)
        """
        changed = False
        for sampling in self._data:
            obs_info = sampling.observable_info
            if isinstance(obs_info, EnergyObsInfo) and obs_info.needs_ni_pair:
                if (
                    obs_info.irrep is None
                    or obs_info.psq is None
                    or obs_info.level_index is None
                ):
                    continue
                key = (obs_info.irrep, obs_info.psq, obs_info.level_index)
                if key in irrep_psq_levels_map:
                    particles = tuple(irrep_psq_levels_map[key])
                    if obs_info.particles != particles:
                        obs_info.particles = particles
                        obs_info.update_name(strict=False)
                        changed = True
        if changed:
            self.clear_shared_attr_cache()

    # -------------------------------------------------------------------------
    # PyCalQ Import and Export
    # -------------------------------------------------------------------------

    def set_shift_particles_from_pycalq_yml(self, yml_path: str) -> None:
        """Load PyCalQ particle assignments into this collection.

        Parameters
        ----------
        yml_path : str
            Path to a PyCalQ YAML configuration.
        """
        self.set_shift_particles(self._parse_pycalq_yml(yml_path))

    def create_pycalq_yml_shift_particles(self, yml_path: str) -> None:
        """Write the collection's particle assignments as PyCalQ YAML.

        Parameters
        ----------
        yml_path : str
            Destination path for the YAML file.

        Notes
        -----
        The output uses PyCalQ's ``non_interacting_levels`` structure and can be
        loaded by :meth:`set_shift_particles_from_pycalq_yml`.
        """
        from .io.pycalq import ShiftParticleMap, write_shift_particles

        assignments: ShiftParticleMap = {}
        for sampling in self._data:
            obs_info = sampling.observable_info
            if (
                isinstance(obs_info, EnergyObsInfo)
                and obs_info.needs_ni_pair
                and obs_info.particles
                and obs_info.irrep is not None
                and obs_info.psq is not None
                and obs_info.level_index is not None
            ):
                assignments[(obs_info.irrep, obs_info.psq, obs_info.level_index)] = list(
                    obs_info.particles
                )

        write_shift_particles(yml_path, assignments)

    def _parse_pycalq_yml(self, yml_path: str) -> dict[tuple[str, int, int], list[Particle]]:
        """Parse shift-particle assignments from a PyCalQ YAML file.

        Parameters
        ----------
        yml_path : str
            Path to a PyCalQ YAML configuration.

        Returns
        -------
        dict[tuple[str, int, int], list[Particle]]
            ``(irrep, psq, level_index)`` keys mapped to particle lists. Sectors
            absent from this collection are omitted.
        """
        from .io.pycalq import read_shift_particles

        allowed_sectors = {sector for sector in self.sectors if sector is not None}
        return read_shift_particles(yml_path, allowed_sectors=allowed_sectors)


class SingleEnsembleEnergyCollection(SingleEnsembleCollection, EnergyLevelMixin):
    """Energy-level observables from one ensemble and sampling configuration.

    Parameters
    ----------
    data : Iterable[SigmondSampling]
        Energy-level samplings from one ensemble and sampling configuration.
    return_type : {"numpy", "list", "dict"}, default="numpy"
        Representation used by collection attribute accessors.

    Notes
    -----
    Every sampling must carry :class:`EnergyObsInfo` or a subclass such as
    :class:`SHEnergyObsInfo`. Filtering and grouping preserve the concrete
    collection type.

    Examples
    --------
    >>> coll = SingleEnsembleEnergyCollection(energy_samplings)
    >>> by_irrep = coll.group_by_irrep()
    >>> a1g_levels = by_irrep["A1g"]
    """

    def __init__(
        self,
        data: Iterable[SigmondSampling],
        return_type: str = "numpy",
    ):
        """Initialize a single-ensemble energy collection.

        Parameters
        ----------
        data : Iterable[SigmondSampling]
            Energy-level samplings from one ensemble and sampling configuration.
        return_type : {"numpy", "list", "dict"}, default="numpy"
            Representation used by collection attribute accessors.

        Raises
        ------
        ValueError
            If samplings use different ensemble or sampling metadata, or if any
            sampling does not carry energy-level metadata.
        """
        # Call parent to validate single ensemble/sampling
        super().__init__(data, return_type)

        # Validate energy-level types (should always pass after conversion)
        self._validate_energy_levels()

class MultiEnsembleEnergyCollection(MultiEnsembleCollection, EnergyLevelMixin):
    """Energy-level observables spanning multiple ensembles.

    Parameters
    ----------
    data : Iterable[SigmondSampling] or Mapping[EnsembleInfo, SingleEnsembleEnergyCollection]
        Energy-level samplings or pre-grouped single-ensemble collections.
    return_type : {"numpy", "list", "dict"}, default="numpy"
        Representation used by collection attribute accessors.

    Notes
    -----
    Every sampling must carry :class:`EnergyObsInfo` or a subclass. The
    :attr:`by_ensemble` view returns :class:`SingleEnsembleEnergyCollection`
    instances, while slicing and filtering preserve this multi-ensemble type.

    Examples
    --------
    >>> multi = MultiEnsembleEnergyCollection(all_energy_samplings)
    >>> a1g_data = multi.filter(irrep="A1g", psq=0)
    >>> assert isinstance(a1g_data, MultiEnsembleEnergyCollection)
    """

    single_ensemble_collection_type = SingleEnsembleEnergyCollection

    def __init__(
        self,
        data: Iterable[SigmondSampling]
        | Mapping[EnsembleInfo, SingleEnsembleEnergyCollection],
        return_type: str = "numpy",
    ):
        """Initialize a multi-ensemble energy collection.

        Parameters
        ----------
        data : Iterable[SigmondSampling] or Mapping[EnsembleInfo, SingleEnsembleEnergyCollection]
            Energy-level samplings or pre-grouped single-ensemble collections.
        return_type : {"numpy", "list", "dict"}, default="numpy"
            Representation used by collection attribute accessors.

        Raises
        ------
        ValueError
            If samplings use incompatible sampling metadata or if any sampling
            does not carry energy-level metadata.
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
        """Return energy levels grouped by ensemble.

        Returns
        -------
        dict[EnsembleInfo, SingleEnsembleEnergyCollection]
            Ensemble metadata mapped to its energy-level collection.
        """
        return cast(dict[EnsembleInfo, SingleEnsembleEnergyCollection], super().by_ensemble)

    # -------------------------------------------------------------------------
    # Override Dict-like Access to Return Energy-Level Types
    # -------------------------------------------------------------------------

    @overload
    def __getitem__(self, key: EnsembleInfo) -> SingleEnsembleEnergyCollection: ...

    @overload
    def __getitem__(self, key: int) -> SigmondSampling: ...

    @overload
    def __getitem__(self, key: slice) -> MultiEnsembleEnergyCollection: ...

    def __getitem__(
        self, key: EnsembleInfo | int | slice
    ) -> SingleEnsembleEnergyCollection | MultiEnsembleEnergyCollection | SigmondSampling:
        """Return an ensemble group, sampling, or sliced collection.

        Parameters
        ----------
        key : EnsembleInfo or int or slice
            Ensemble metadata, sampling index, or collection slice.

        Returns
        -------
        SingleEnsembleEnergyCollection or MultiEnsembleEnergyCollection or SigmondSampling
            The ensemble group for an :class:`EnsembleInfo`, one sampling for an
            integer, or a multi-ensemble collection for a slice.
        """
        return cast(
            SingleEnsembleEnergyCollection | MultiEnsembleEnergyCollection | SigmondSampling,
            super().__getitem__(key),
        )
