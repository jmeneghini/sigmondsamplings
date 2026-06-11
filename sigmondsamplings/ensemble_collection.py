"""
Ensemble Collection Classes: Containers for managing observables with ensemble awareness.

This module provides collection classes that maintain ensemble separation:
- SingleEnsembleCollection: All observables from one ensemble/sampling configuration
- MultiEnsembleCollection: Observables from multiple ensembles, stored separately
"""

from collections.abc import Iterable
from typing import (
    TypeVar,
    Union,
)

from .obervable_collection import ObservableCollection
from .sampling import EnsembleInfo, SamplingInfo, SigmondSampling

__all__ = [
    "SingleEnsembleCollection",
    "MultiEnsembleCollection",
]

T = TypeVar("T", bound="SingleEnsembleCollection")
M = TypeVar("M", bound="MultiEnsembleCollection")


class SingleEnsembleCollection(ObservableCollection):
    """
    Observable collection with single ensemble and sampling configuration.

    All observables must have identical ensemble_info and sampling_info.
    This is a common pattern for single-ensemble analyses where all data
    comes from the same Monte Carlo ensemble with the same resampling strategy.

    Provides direct access to the common ensemble_info and sampling_info as properties.
    """

    __slots__ = ("_data", "_return_type", "_ensemble_info", "_sampling_info")

    dataframe_excluded_attrs = (
        *ObservableCollection.dataframe_excluded_attrs,
        "ensemble_info",
        "sampling_info",
    )

    def __init__(self, data: Iterable[SigmondSampling], return_type: str = "numpy"):
        """
        Initialize SingleEnsembleCollection with validation.

        Args:
            data: Iterable of SigmondSampling objects
            return_type: Return type for attribute access - 'dict', 'list', or 'numpy'

        Raises:
            ValueError: If observables have different ensemble_info or sampling_info
        """
        # Initialize parent class
        super().__init__(data, return_type)

        self._validate_ens_and_samp_info()

    @classmethod
    def _fast_load(
        cls: type[T],
        data: list[SigmondSampling],
        return_type: str,
    ) -> T:
        """
        Internal constructor for efficient creation from trusted data.

        Used by filter/sort methods. Assumes energy data is already validated.
        Rechecks ensemble/sampling consistency.
        """
        instance = cls.__new__(cls)
        instance._data = data
        instance._return_type = return_type
        instance._shared_attr_cache = {}

        instance._validate_ens_and_samp_info()
        return instance

    def _validate_ens_and_samp_info(self):
        """Validate that all observables share the same ensemble and sampling info."""
        # Validate and extract shared ensemble/sampling using inherited shared_attr
        try:
            self._ensemble_info = self.shared_attr(
                values=self.obs.ensemble_info, strict=True, cache=False
            )
        except ValueError:
            ensemble_ids = {
                e.id if hasattr(e, "id") else str(e) for e in set(self.obs.ensemble_info)
            }
            raise ValueError(
                f"All observables must have the same ensemble_info. "
                f"Found multiple ensembles: {ensemble_ids}"
            )

        try:
            self._sampling_info = self.shared_attr(
                values=self.val.sampling_info, strict=True, cache=False
            )
        except ValueError:
            sampling_strs = {str(s) for s in set(self.val.sampling_info)}
            raise ValueError(
                f"All observables must have the same sampling_info. "
                f"Found multiple sampling configurations: {sampling_strs}"
            )

    @property
    def ensemble_info(self):
        """Single ensemble info shared by all observables"""
        return self._ensemble_info

    @property
    def sampling_info(self):
        """Single sampling info shared by all observables"""
        return self._sampling_info

    def __repr__(self) -> str:
        ensemble_id = (
            self._ensemble_info.id
            if hasattr(self._ensemble_info, "id")
            else str(self._ensemble_info)
        )
        sampling_str = str(self._sampling_info)
        return (
            f"SingleEnsembleCollection("
            f"n_obs={len(self._data)}, "
            f"ensemble='{ensemble_id}', "
            f"sampling='{sampling_str}'"
            f")"
        )


def group_by_ensemble_and_sampling(
    collection: ObservableCollection,
) -> dict[tuple, SingleEnsembleCollection]:
    """
    Group an ObservableCollection into SingleEnsembleCollections by ensemble and sampling.

    Creates a dictionary mapping (ensemble_info, sampling_info) tuples to their
    corresponding SingleEnsembleCollection objects. This is useful for organizing
    multi-ensemble data into separate single-ensemble collections.

    Parameters:
    -----------
    collection : ObservableCollection
        Collection of observables potentially from multiple ensembles/samplings

    Returns:
    --------
    Dict[Tuple[EnsembleInfo, SamplingInfo], SingleEnsembleCollection]
        Dictionary mapping (ensemble_info, sampling_info) tuples to their
        corresponding SingleEnsembleCollection

    Examples:
    ---------
    >>> # Group mixed ensemble data
    >>> grouped = group_by_ensemble_and_sampling(all_observables)
    >>> for (ens, samp), single_coll in grouped.items():
    ...     print(f"Processing {ens.id} with {len(single_coll)} observables")

    >>> # Extract specific ensemble/sampling combination
    >>> key = (my_ensemble_info, my_sampling_info)
    >>> specific_data = grouped.get(key)
    """
    if not collection:
        return {}

    # Use inherited group_by with tuple key for (ensemble_info, sampling_info)
    groups = collection.group_by(
        values=list(zip(collection.obs.ensemble_info, collection.val.sampling_info))
    )

    # Convert each group to SingleEnsembleCollection
    return {
        key: SingleEnsembleCollection(group, return_type=collection.return_type)
        for key, group in groups.items()
    }


class MultiEnsembleCollection(ObservableCollection):
    """
    A container for observables from multiple ensembles.

    Inherits from ObservableCollection to reuse filter/find/sort methods,
    while providing ensemble-aware grouping via the `by_ensemble` property.

    This class is designed to simplify workflows where you need to:
    - Load data from multiple ensembles
    - Have all data share the same SamplingInfo (e.g. for KBfit)
    - Apply consistent filters across all ensembles
    - Select subsets based on attributes or data values
    - Access data grouped by ensemble when needed

    Key Features:
    -------------
    - Inherits all ObservableCollection methods (filter, filter_data, find, etc.)
    - `by_ensemble` property provides Dict[EnsembleInfo, SingleEnsembleCollection] view
    - `ensembles` property lists all unique ensembles

    Example:
    --------
    >>> # Create from mixed data
    >>> multi = MultiEnsembleCollection(all_samplings)
    >>>
    >>> # Use inherited filter methods
    >>> filtered = multi.filter(irrep="A1g", psq=0)
    >>> filtered = multi.filter_data(gt=0.5, lt=2.0)
    >>>
    >>> # Access by ensemble
    >>> for ens_info, collection in multi.by_ensemble.items():
    ...     print(f"{ens_info}: {len(collection)} observables")
    >>>
    >>> # List values for membership filtering (inherited from ObservableCollection)
    >>> selected = multi.filter(irrep=["A1g", "T1u"], psq=[0, 1, 2])
    """

    # No extra slots - uses parent's _data and _return_type

    def __init__(
        self,
        data: Iterable[SigmondSampling] | dict[EnsembleInfo, SingleEnsembleCollection],
        return_type: str = "numpy",
    ):
        """
        Initialize MultiEnsembleCollection.

        Parameters:
        -----------
        data : Union[Iterable[SigmondSampling], Dict[EnsembleInfo, SingleEnsembleCollection], MultiEnsembleCollection]
            Input data - can be:
            - Iterable of SigmondSampling objects
            - Dict mapping EnsembleInfo to SingleEnsembleCollection (flattened)
            - Another MultiEnsembleCollection (copy)
        return_type : str
            Return type for attribute accessors ('list', 'dict', or 'numpy')
        """
        if isinstance(data, dict):
            # Flatten dict of collections into single list
            flat_data = []
            for collection in data.values():
                flat_data.extend(collection)
            super().__init__(flat_data, return_type)
        else:
            # Standard initialization from iterable
            super().__init__(data, return_type)
        # Validate that all observables share the same sampling_info
        self.shared_attr(values=self.val.sampling_info, strict=True)

    @classmethod
    def _fast_load(
        cls: type[M],
        data: list[SigmondSampling],
        return_type: str,
    ) -> M:
        """Internal constructor bypassing validation for trusted data."""
        instance = cls.__new__(cls)
        instance._data = data
        instance._return_type = return_type
        instance._shared_attr_cache = {}
        return instance

    # -------------------------------------------------------------------------
    # Ensemble-Aware Properties
    # -------------------------------------------------------------------------

    @property
    def sampling_info(self) -> SamplingInfo:
        """Sampling info shared by all observables in the collection."""
        return self.shared_attr(values=self.val.sampling_info, strict=True)

    @property
    def ensembles(self) -> list[EnsembleInfo]:
        """List of unique ensemble infos in the collection."""
        return list(set(self.obs.ensemble_info))

    @property
    def by_ensemble(self) -> dict[EnsembleInfo, SingleEnsembleCollection]:
        """
        Group data by ensemble, returning Dict[EnsembleInfo, SingleEnsembleCollection].

        This is computed on-demand from the flat internal storage.
        """
        groups = self.group_by(values=self.obs.ensemble_info)

        # Convert each group to SingleEnsembleCollection
        return {
            ens: SingleEnsembleCollection(group, return_type=self._return_type)
            for ens, group in groups.items()
        }

    def reduced(self):
        """Collapse to the lone ``SingleEnsembleCollection`` when one ensemble is present.

        A collection spanning exactly one ensemble is unwrapped to its
        single-ensemble view -- so scalar ``ensemble_info``/``sampling_info`` and
        single-ensemble methods become available -- via :attr:`by_ensemble`, which
        subclasses override to yield their specific single type (e.g.
        ``SingleEnsembleEnergyCollection``). A genuinely multi-ensemble (or empty)
        collection is returned unchanged.
        """
        ensembles = self.ensembles
        if len(ensembles) == 1:
            return self.by_ensemble[ensembles[0]]
        return self

    # -------------------------------------------------------------------------
    # Dict-like Access by Ensemble
    # -------------------------------------------------------------------------

    def __getitem__(
        self, key: EnsembleInfo | int | slice
    ) -> Union[SingleEnsembleCollection, "MultiEnsembleCollection", SigmondSampling]:
        """
        Access by EnsembleInfo, index, or slice.

        Parameters:
        -----------
        key : Union[EnsembleInfo, int, slice]
            - EnsembleInfo: Get SingleEnsembleCollection for that ensemble
            - int: Get sampling by index (inherited behavior)
            - slice: Get subset (inherited behavior)
        """
        if isinstance(key, EnsembleInfo):
            return self.by_ensemble[key]
        else:
            # Delegate to parent for int/slice
            result = super().__getitem__(key)
            if isinstance(key, slice):
                return self._fast_load(list(result._data), self._return_type)
            return result

    def items(self):
        """Iterate over (EnsembleInfo, SingleEnsembleCollection) pairs."""
        return self.by_ensemble.items()

    def keys(self):
        """Iterate over EnsembleInfo keys."""
        return self.by_ensemble.keys()

    def values(self):
        """Iterate over SingleEnsembleCollection values."""
        return self.by_ensemble.values()

    def __repr__(self):
        n_ensembles = len(self.ensembles)
        total = len(self)
        return f"MultiEnsembleCollection(ensembles={n_ensembles}, total_obs={total})"
