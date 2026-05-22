"""
ObservableCollection: A fast, queryable collection of observables.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import (
    Any,
    TypeVar,
)

import numpy as np

from .sampling import SigmondSampling

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# TypeVar for fluent interface
T = TypeVar("T", bound="ObservableCollection")


class PandasExportMixin:
    """
    Mixin for exporting collection data to Pandas DataFrames.
    Separates I/O logic from the main collection.
    """

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame dynamically.

        Returns:
            DataFrame with columns for ObsInfo attributes and data
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas required for to_dataframe(). Install with: pip install pandas"
            )

        # Access _data from the main class
        data_source = getattr(self, "_data", [])

        if not data_source:
            return pd.DataFrame()

        samp_val_attrs = ["full_sample_value", "mean", "error"]
        samp_val_methods = []

        rows = []
        for s in data_source:
            # Base row with data value
            row = {"name": str(s.observable_info.name), "data_str": s.pdg_format()}
            for key in samp_val_attrs:
                if hasattr(s, key):
                    row[key] = getattr(s, key)
            CI_tuple = s.confidence_interval()
            row["CI_upper"] = CI_tuple[0]
            row["CI_lower"] = CI_tuple[1]
            for method in samp_val_methods:
                if hasattr(s, method):
                    try:
                        row[method] = getattr(s, method)()
                    except Exception:
                        row[method] = None

            # Dynamically add all available ObsInfo attributes
            # This replaces the hardcoded list of attributes
            obs_info = s.observable_info
            if hasattr(obs_info, "__dict__"):
                attrs = vars(obs_info)
            elif hasattr(obs_info, "_asdict"):  # NamedTuple
                attrs = obs_info._asdict()
            else:
                # Fallback for other objects
                import dataclasses

                if dataclasses.is_dataclass(obs_info):
                    attrs = dataclasses.asdict(obs_info)
                else:
                    attrs = {}

            row.update(attrs)
            rows.append(row)

        return pd.DataFrame(rows)


class AttributeAccessor:
    """
    Provides safe, namespaced access to both properties (data) and methods (actions).

    Enhanced to support batch attribute updates via replace().

    Examples:
        # Reading (unchanged)
        names = collection.obs.name  # Returns ['n1', 'n2', ...]

        # Writing (new - returns new collection)
        new_coll = collection.obs.replace(name='uniform_name')
        new_coll = collection.obs.replace(name=['n1', 'n2', ...])
        new_coll = collection.obs.replace(name=lambda obs: f'new_{obs.name}')
    """

    def __init__(self, collection, target_extractor: Callable[[Any], Any]):
        self._collection = collection
        self._extractor = target_extractor

    @staticmethod
    def _is_numeric_value(value: Any) -> bool:
        """Fast numeric check for numpy-return optimization."""
        return isinstance(value, (int, float, np.number, bool))

    def replace(self, **kwargs) -> "ObservableCollection":
        """
        Replace one or more attributes and return new collection.

        Args:
            **kwargs: Attribute-value pairs to replace. Values can be:
                - Scalar: Applied to all items
                - List/tuple: One per item (must match collection length)
                - Callable: Called with target object to compute value

        Returns:
            New ObservableCollection with replaced attribute values

        Raises:
            ValueError: If list length doesn't match collection length

        Examples:
            new_coll = collection.obs.replace(name='uniform_name')
            new_coll = collection.obs.replace(name=['n1', 'n2', ...])
            new_coll = collection.obs.replace(name=lambda obs: f'new_{obs.name}')
            new_coll = collection.obs.replace(name='test', index=99)
        """
        if not kwargs:
            raise ValueError("Must provide at least one attribute to replace")

        if not self._collection._data:
            # Empty collection - return new empty collection
            return self._collection._fast_load([], self._collection._return_type)

        # Validate all list lengths upfront
        for attr_name, value in kwargs.items():
            if isinstance(value, (list, tuple)) and len(value) != len(self._collection):
                raise ValueError(
                    f"List length for '{attr_name}' ({len(value)}) must match collection length ({len(self._collection)})"
                )

        # Create new samplings with updated attributes
        new_samplings = []
        for idx, sampling in enumerate(self._collection._data):
            target = self._extractor(sampling)

            # Optimize for observable_info (most common case)
            if target is sampling.observable_info:
                new_obs_info = sampling.observable_info.copy()

                # Update all requested attributes
                for attr_name, value in kwargs.items():
                    # Resolve value for this item
                    if callable(value):
                        new_value = value(new_obs_info)
                    elif isinstance(value, (list, tuple)):
                        new_value = value[idx]
                    else:
                        new_value = value

                    # Set the attribute
                    setattr(new_obs_info, attr_name, new_value)

                # Create new sampling with updated observable_info
                new_sampling = SigmondSampling(
                    data=sampling.data,  # Share array (immutable usage)
                    observable_info=new_obs_info,
                    sampling_info=sampling.sampling_info,  # Share
                    is_complex=sampling.is_complex,
                )
            else:
                new_sampling = sampling.copy()
                target = self._extractor(new_sampling)

                # Update all requested attributes
                for attr_name, value in kwargs.items():
                    # Resolve value for this item
                    if callable(value):
                        new_value = value(target)
                    elif isinstance(value, (list, tuple)):
                        new_value = value[idx]
                    else:
                        new_value = value

                    # Set the attribute
                    setattr(target, attr_name, new_value)

            new_samplings.append(new_sampling)

        # Return new collection using fast path
        return self._collection._fast_load(new_samplings, self._collection._return_type)

    def __getattr__(self, name):
        # 1. Safety check on empty collection
        if not self._collection._data:
            if self._collection.return_type == "dict":
                return {}
            if self._collection.return_type == "numpy":
                return np.array([])
            return []

        # 2. Peek at the first item to check the type of the attribute
        first_item = self._collection._data[0]
        first_target = self._extractor(first_item)

        if not hasattr(first_target, name):
            raise AttributeError(f"'{type(first_target).__name__}' has no attribute '{name}'")

        sample_attr = getattr(first_target, name)

        # 3. CASE A: It is a Method/Callable -> Return a wrapper function
        if callable(sample_attr):

            def method_proxy(*args, **kwargs):
                results = []
                for item in self._collection._data:
                    target = self._extractor(item)
                    method = getattr(target, name)
                    results.append(method(*args, **kwargs))

                # Format return based on preference
                if self._collection.return_type == "dict":
                    return {
                        item.observable_info: res
                        for item, res in zip(self._collection._data, results)
                    }
                return results

            return method_proxy

        # 4. CASE B: It is a Property/Data -> Return the values immediately
        values = [getattr(self._extractor(item), name) for item in self._collection._data]

        if self._collection.return_type == "dict":
            return {item.observable_info: val for item, val in zip(self._collection._data, values)}
        if self._collection.return_type == "numpy":
            if values and self._is_numeric_value(values[0]):
                arr = np.asarray(values)
                if arr.dtype != object:
                    return arr
        return values


class ObservableCollection(PandasExportMixin):
    """
    A queryable collection of observables with fast filtering and iteration.

    Provides convenient filtering, iteration, and conversion methods for observables.
    Uses __slots__ for memory efficiency and fast attribute access.
    """

    __slots__ = ("_data", "_return_type", "_shared_attr_cache")

    def __init__(self, data: Iterable[SigmondSampling], return_type: str = "numpy"):
        """
        Initialize ObservableCollection with deduplication.

        Args:
            data: List of SigmondSampling objects
            return_type: Return type for attribute access - 'dict', 'list', or 'numpy'
        """
        # Deduplicate while preserving order
        self._data = list(dict.fromkeys(data))

        # Validate and set return type
        if return_type not in ("dict", "list", "numpy"):
            raise ValueError("return_type must be 'dict', 'list', or 'numpy'")
        self._return_type = return_type
        self._shared_attr_cache = {}

    @classmethod
    def _fast_load(cls: type[T], data: list[SigmondSampling], return_type: str) -> T:
        """
        Internal constructor to bypass validation/deduplication for trusted data.
        Used by filter/sort methods to return new instances efficiently.
        """
        instance = cls.__new__(cls)
        instance._data = data
        instance._return_type = return_type
        instance._shared_attr_cache = {}
        return instance

    def copy(self: T) -> T:
        """Return a copy with each contained SigmondSampling copied independently."""
        return self._fast_load([s.copy() for s in self._data], self._return_type)

    @property
    def return_type(self) -> str:
        """Get the return type for attribute access ('dict' or 'list')."""
        return self._return_type

    @return_type.setter
    def return_type(self, value: str):
        if value not in ("dict", "list", "numpy"):
            raise ValueError("return_type must be 'dict', 'list', or 'numpy'")
        self._return_type = value

    @property
    def obs(self) -> AttributeAccessor:
        """
        Namespace for ObservableInfo attributes.
        Usage: collection.obs.irrep -> ['A1g', ...]
        """
        return AttributeAccessor(self, lambda s: s.observable_info)

    @property
    def samp(self) -> AttributeAccessor:
        """
        Namespace for SamplingInfo attributes.
        Usage: collection.samp.method -> ['bootstrap', ...]
        """
        return AttributeAccessor(self, lambda s: s.sampling_info)

    @property
    def val(self) -> AttributeAccessor:
        """
        Namespace for Data/Value attributes.
        Usage: collection.val.mean -> [0.5, 0.6, ...]
        """
        return AttributeAccessor(self, lambda s: s)

    def filter(self: T, predicate=None, **kwargs) -> T:
        """
        Filter observables by attributes or direct object comparison.

        When both predicate and kwargs are provided, they are combined with AND logic
        (all conditions must be satisfied).

        Attribute values can be:
            - Single value: exact match (e.g., irrep="A1g")
            - List/tuple/set: membership test (e.g., irrep=["A1g", "T1u"])

        Examples:
            collection.filter(irrep="A1g", psq=0)  # Exact match
            collection.filter(irrep=["A1g", "T1u"], psq=[0, 1, 2])  # Membership
            collection.filter(ensemble_info=[ens1, ens2])  # Multiple ensembles
            collection.filter(predicate=lambda obs: obs.level_index < 5, irrep="A1g")  # Combined
        """
        # Return shallow copy if no filters provided
        if predicate is None and not kwargs:
            return self._fast_load(self._data[:], self._return_type)

        if not self._data:
            return self._fast_load([], self._return_type)

        # Prepare attribute-based criteria
        direct_checks, obs_filters, samp_filters = self._prepare_criteria(kwargs)

        # Build combined filter function
        if predicate is not None:
            # Add predicate as an additional check
            def predicate_check(samp):
                return predicate(samp.observable_info)
        else:
            predicate_check = None

        # Optimized loop combining all filters with AND logic
        filtered = [
            samp
            for samp in self._data
            if (predicate_check is None or predicate_check(samp))
            and (not direct_checks or all(chk(samp) for chk in direct_checks))
            and (
                not obs_filters
                or all(getattr(samp.observable_info, k, None) == v for k, v in obs_filters)
            )
            and (
                not samp_filters
                or all(getattr(samp.sampling_info, k, None) == v for k, v in samp_filters)
            )
        ]

        return self._fast_load(filtered, self._return_type)

    def _normalize_values(
        self, values: Sequence[Any] | Mapping[Any, Any] | np.ndarray
    ) -> list[Any]:
        """Normalize values input to a list aligned with _data ordering."""
        if isinstance(values, Mapping):
            values_list = list(values.values())
        else:
            values_list = list(values)
        if len(values_list) != len(self._data):
            raise ValueError("Values length must match number of observables")
        return values_list

    def _attr_values(self, attr: str, force_list: bool = True) -> list[Any]:
        """
        Collect attribute values from samplings or their metadata.

        Searches across namespaces in order: val, obs, samp.
        Returns a plain list (not formatted by return_type) for internal use.
        """
        if not self._data:
            return []

        # Try each accessor in order: val (SigmondSampling), obs (ObservableInfo), samp (SamplingInfo)
        accessors = [self.val, self.obs, self.samp]

        for accessor in accessors:
            try:
                result = getattr(accessor, attr)
                # AttributeAccessor returns formatted results based on return_type
                # We need to normalize back to a list for internal use
                if isinstance(result, dict):
                    # Dict return type - extract values in data order
                    return list(result.values())
                if isinstance(result, np.ndarray):
                    # NumPy return type - convert to list when requested
                    return result.tolist() if force_list else result
                if isinstance(result, list):
                    # Plain list (non-numeric AttributeAccessor return)
                    return result
            except AttributeError:
                # Attribute not found in this namespace, try next
                continue

        # Fallback: try observable_info with default None
        return [getattr(s.observable_info, attr, None) for s in self._data]

    def shared_attr(
        self,
        key: str | Callable[[SigmondSampling], Any] | Sequence[Any] | None = None,
        *,
        values: Sequence[Any] | Mapping[Any, Any] | np.ndarray | None = None,
        default: Any = None,
        strict: bool = False,
        cache: bool = True,
    ) -> Any:
        """
        Return a shared attribute value if all entries match.
        After mutating data that could affect shared attributes,
        call this with strict=True and cache=False to validate and refresh cache.

        Args:
            key: Attribute name, callable mapping sampling -> value, or a values sequence.
                Use a callable or values for autocomplete-friendly access.
            values: Explicit values sequence (e.g., collection.obs.ensemble_info).
            default: Value to return if not shared (when strict=False).
            strict: If True, raise ValueError when values differ.
            cache: Cache results by key when possible.

        Examples:
            # Autocomplete-friendly
            collection.shared_attr(values=collection.obs.ensemble_info, strict=True)

            # Callable
            collection.shared_attr(lambda s: s.sampling_info, strict=True)

            # Attribute name
            collection.shared_attr("ensemble_info", default=None)
        """
        if values is None and key is None:
            raise ValueError("Provide key or values")

        cache_key = None
        if values is None:
            if cache:
                cache_key = key
                if cache_key in self._shared_attr_cache:
                    return self._shared_attr_cache[cache_key]
            if isinstance(key, str):
                values_list = self._attr_values(key)
            elif callable(key):
                values_list = [key(s) for s in self._data]
            else:
                values_list = self._normalize_values(key)
        else:
            values_list = self._normalize_values(values)

        if not values_list:
            return default

        first_val = values_list[0]
        all_same = True
        for val in values_list[1:]:
            if isinstance(first_val, np.ndarray) or isinstance(val, np.ndarray):
                if not np.array_equal(first_val, val):
                    all_same = False
                    break
            elif val != first_val:
                all_same = False
                break

        if not all_same:
            if strict:
                raise ValueError("Values are not shared across the collection")
            if cache_key is not None:
                self._shared_attr_cache[cache_key] = default
            return default

        if cache_key is not None:
            self._shared_attr_cache[cache_key] = first_val
        return first_val

    def clear_shared_attr_cache(self) -> None:
        """Clear shared attribute cache."""
        self._shared_attr_cache.clear()

    def group_by(
        self,
        key: str | Callable[[SigmondSampling], Hashable] | Sequence[Any] | None = None,
        *,
        values: Sequence[Any] | Mapping[Any, Any] | np.ndarray | None = None,
    ) -> dict[Hashable, T]:
        """
        Group samplings by a key or provided values.

        Args:
            key: Attribute name, callable mapping sampling -> group key, or values sequence.
                Use a callable or values for autocomplete-friendly access.
            values: Explicit values sequence (e.g., collection.obs.ensemble_info).

        Returns:
            Dict mapping group key -> collection of the same type.

        Examples:
            # Group by an attribute string
            collection.group_by("ensemble_info")

            # Group by a tuple key
            collection.group_by(lambda s: (s.observable_info.ensemble_info, s.sampling_info))

            # Autocomplete-friendly
            collection.group_by(values=collection.obs.irrep)
        """
        if values is None and key is None:
            raise ValueError("Provide key or values")

        if values is None:
            if isinstance(key, str):
                values_list = self._attr_values(key)
            elif callable(key):
                values_list = [key(s) for s in self._data]
            else:
                values_list = self._normalize_values(key)
        else:
            values_list = self._normalize_values(values)

        groups: dict[Hashable, list[SigmondSampling]] = {}
        for sampling, group_key in zip(self._data, values_list):
            groups.setdefault(group_key, []).append(sampling)

        return {
            group_key: self._fast_load(group, self._return_type)
            for group_key, group in groups.items()
        }

    def unique(
        self,
        key: str | Callable[[SigmondSampling], Hashable] | Sequence[Any] | None = None,
        *,
        values: Sequence[Any] | Mapping[Any, Any] | np.ndarray | None = None,
    ) -> list[Any] | np.ndarray | None:
        """
        Get unique values for a key or provided values.

        Works identically to group_by for extracting values, but returns only the
        unique values in the collection's return_type format.

        Args:
            key: Attribute name, callable mapping sampling -> value, or values sequence.
                Use a callable or values for autocomplete-friendly access.
            values: Explicit values sequence (e.g., collection.obs.ensemble_info).

        Returns:
            Unique values formatted according to return_type:
            - "list": List of unique values (order preserved)
            - "numpy": NumPy array of unique values

        Examples:
            # Get unique ensemble_info values
            collection.unique("ensemble_info")

            # Get unique irrep values using autocomplete
            collection.unique(values=collection.obs.irrep)

            # Get unique composite keys
            collection.unique(lambda s: (s.observable_info.irrep, s.observable_info.psq))
        """
        if values is None and key is None:
            raise ValueError("Provide key or values")

        if values is None:
            if isinstance(key, str):
                values_list = self._attr_values(key)
            elif callable(key):
                values_list = [key(s) for s in self._data]
            else:
                values_list = self._normalize_values(key)
        else:
            values_list = self._normalize_values(values)

        # Get unique values while preserving order
        seen = set()
        unique_values = []
        for value in values_list:
            # Handle unhashable types
            try:
                if value not in seen:
                    seen.add(value)
                    unique_values.append(value)
            except TypeError:
                # Value is unhashable, check manually
                if value not in unique_values:
                    unique_values.append(value)

        if len(unique_values) == 0:
            return None

        # Format return based on return_type
        # Sort with None-safe key (None values sort first)
        try:
            if self._return_type == "numpy":
                if unique_values and AttributeAccessor._is_numeric_value(unique_values[0]):
                    # For numpy with numeric values, use np.sort (handles None as NaN)
                    arr = np.array(unique_values)
                    return np.sort(arr)
            # Use None-safe sorting: (value is None, value) puts None first
            return sorted(unique_values, key=lambda x: (x is None, x))
        except TypeError:
            # If still can't sort (e.g., mixed incompatible types), return unsorted
            return unique_values

    def find(self, predicate: Callable = None, **kwargs) -> SigmondSampling | None:
        """
        Find first sampling matching predicate or attribute criteria.

        When both predicate and kwargs are provided, they are combined with AND logic
        (all conditions must be satisfied).

        Examples:
            collection.find(irrep="A1g", psq=0)  # Find first matching attributes
            collection.find(predicate=lambda obs: obs.level_index < 5)  # Find by predicate
            collection.find(predicate=lambda obs: obs.level_index < 5, irrep="A1g")  # Combined
        """
        if not self._data:
            return None

        # Return first element if no filters
        if predicate is None and not kwargs:
            return self._data[0]

        # Prepare attribute-based criteria
        direct_checks, obs_filters, samp_filters = self._prepare_criteria(kwargs)

        # Build combined filter function
        if predicate is not None:

            def predicate_check(samp):
                return predicate(samp.observable_info)
        else:
            predicate_check = None

        # Optimized search loop combining all filters with AND logic
        for samp in self._data:
            if (
                (predicate_check is None or predicate_check(samp))
                and (not direct_checks or all(chk(samp) for chk in direct_checks))
                and (
                    not obs_filters
                    or all(getattr(samp.observable_info, k, None) == v for k, v in obs_filters)
                )
                and (
                    not samp_filters
                    or all(getattr(samp.sampling_info, k, None) == v for k, v in samp_filters)
                )
            ):
                return samp

        return None

    def sort(
        self: T,
        key: str | Callable | Sequence[str] | None = None,
        *,
        values: Sequence[Any] | Mapping[Any, Any] | np.ndarray | None = None,
        reverse: bool = False,
        nulls_last: bool = False,
    ) -> T:
        """
        Sort collection by key or provided values.
        Returns a NEW instance (Immutable).

        Args:
            key: Attribute name(s), callable, or values sequence.
                 Can be a string, list of strings, or callable.
            values: Explicit values sequence (e.g., collection.obs.psq).
            reverse: If True, sort in descending order.
            nulls_last: If True, None values sort last; if False (default), None values sort first.

        Returns:
            New sorted collection of the same type.

        Examples:
            # Sort by single attribute
            collection.sort(key="psq")

            # Sort by multiple attributes (psq first, then irrep)
            collection.sort(key=["psq", "irrep"])

            # Sort with None values last
            collection.sort(key="level_index", nulls_last=True)

            # Custom key function
            collection.sort(key=lambda s: s.observable_info.psq)

            # Autocomplete-friendly using values
            collection.sort(values=collection.obs.psq)
        """
        if values is None and key is None:
            raise ValueError("Provide key or values")

        if not self._data:
            return self._fast_load([], self._return_type)

        # Determine the sort key function
        if values is not None:
            # Pair each sampling with its corresponding value
            values_list = self._normalize_values(values)
            paired = list(zip(self._data, values_list))

            # Sort with None-safe key
            def safe_sort_key(pair):
                val = pair[1]
                return (val is None, val) if nulls_last else (val is not None, val)

            sorted_pairs = sorted(paired, key=safe_sort_key, reverse=reverse)
            new_data = [s for s, _ in sorted_pairs]
        elif callable(key):
            # Custom function - wrap with None handling
            def safe_sort_key(s):
                val = key(s)
                return (val is None, val) if nulls_last else (val is not None, val)

            new_data = sorted(self._data, key=safe_sort_key, reverse=reverse)
        else:
            # String or list of strings - use existing composite key builder
            sort_key = self._build_composite_sort_key(key, nulls_last=nulls_last)
            new_data = sorted(self._data, key=sort_key, reverse=reverse)

        return self._fast_load(new_data, self._return_type)

    def _prepare_criteria(self, kwargs: dict) -> tuple:
        """
        Internal helper to parse kwargs into direct object checks and attribute filters.
        Routes attributes to ObservableInfo or SamplingInfo based on first item inspection.

        Supports list values for membership testing:
            - irrep="A1g" -> exact match
            - irrep=["A1g", "T1u"] -> membership test (any of these values)

        Returns:
            (direct_checks, obs_filters, samp_filters)
        """
        direct_checks = []
        obs_filters = []
        samp_filters = []

        # 1. Direct Object Matching (pop from kwargs so they aren't treated as attributes)
        if "sampling_info" in kwargs:
            target = kwargs.pop("sampling_info")
            direct_checks.append(lambda s, t=target: s.sampling_info == t)

        if "observable_info" in kwargs:
            target = kwargs.pop("observable_info")
            direct_checks.append(lambda s, t=target: s.observable_info == t)

        # If we have no data to inspect, we can't route attributes.
        # Return what we have; the caller handles the empty data case.
        if not self._data or not kwargs:
            return direct_checks, obs_filters, samp_filters

        # 2. Attribute Routing (Inspection Strategy)
        first_item = self._data[0]

        for k, v in kwargs.items():
            # Check if value is a list/sequence (but not string) -> membership test
            is_sequence = isinstance(v, (list, tuple, set, frozenset))

            # Determine which object has this attribute
            if hasattr(first_item.observable_info, k):
                if is_sequence:
                    val_set = set(v)
                    direct_checks.append(
                        lambda s, attr=k, vals=val_set: (
                            getattr(s.observable_info, attr, None) in vals
                        )
                    )
                else:
                    obs_filters.append((k, v))
            elif hasattr(first_item.sampling_info, k):
                if is_sequence:
                    val_set = set(v)
                    direct_checks.append(
                        lambda s, attr=k, vals=val_set: getattr(s.sampling_info, attr, None) in vals
                    )
                else:
                    samp_filters.append((k, v))
            else:
                # Fallback: Assume it's an ObsInfo attribute
                if is_sequence:
                    val_set = set(v)
                    direct_checks.append(
                        lambda s, attr=k, vals=val_set: (
                            getattr(s.observable_info, attr, None) in vals
                        )
                    )
                else:
                    obs_filters.append((k, v))

        return direct_checks, obs_filters, samp_filters

    def _build_composite_sort_key(self, by, nulls_last: bool = False) -> Callable:
        """
        Internal helper to build a sort key function.
        Inspects the first item to find where attributes live.

        Args:
            by: Attribute name(s) to sort by
            nulls_last: If True, None values sort last; if False, None values sort first
        """
        if isinstance(by, str):
            by = [by]

        first_item = self._data[0]
        extractors = []

        for attr in by:
            # Priority 1: Check SigmondSampling itself
            if hasattr(first_item, attr):
                extractors.append(lambda s, a=attr: getattr(s, a))

            # Priority 2: Check ObservableInfo
            elif hasattr(first_item.observable_info, attr):
                extractors.append(lambda s, a=attr: getattr(s.observable_info, a))

            # Priority 3: Check SamplingInfo
            elif hasattr(first_item.sampling_info, attr):
                extractors.append(lambda s, a=attr: getattr(s.sampling_info, a))

            else:
                raise AttributeError(
                    f"Attribute '{attr}' not found in SigmondSampling, ObservableInfo, or SamplingInfo"
                )

        # Return optimized key function that handles None values
        # Using tuple (is_none, value) ensures None sorts first (False < True)
        # or last (True > False) depending on nulls_last
        def safe_key(sampling):
            return tuple(
                (val is not None, val) if not nulls_last else (val is None, val)
                for val in (ex(sampling) for ex in extractors)
            )

        return safe_key

    def filter_data(self: T, predicate=None, min_val=None, max_val=None, **kwargs) -> T:
        """
        Filter observables based on their data values (full_sample_value).

        When both predicate and other arguments are provided, they are combined with AND logic
        (all conditions must be satisfied).

        Args:
            predicate: Callable that takes a SigmondSampling and returns bool.
            min_val: Minimum value (inclusive). Can be float/int or another SigmondSampling.
            max_val: Maximum value (inclusive). Can be float/int or another SigmondSampling.
            **kwargs: Comparison operators:
                - gt: Greater than
                - lt: Less than
                - ge: Greater than or equal
                - le: Less than or equal
                - eq: Equal to
                - ne: Not equal to

        Returns:
            New ObservableCollection with filtered results

        Examples:
            # Filter by range
            collection.filter_data(min_val=0.5, max_val=1.5)

            # Filter using predicate function
            collection.filter_data(predicate=lambda s: s.full_sample_value > 0.5)

            # Filter using comparison operators
            collection.filter_data(gt=0.5)
            collection.filter_data(lt=1.5, gt=0.5)

            # Filter against another sampling's value
            collection.filter_data(gt=reference_sampling)

            # Combine predicate with other filters
            collection.filter_data(predicate=lambda s: s.mean > 0, min_val=0.5)
        """

        def get_value(val):
            return val.full_sample_value if isinstance(val, SigmondSampling) else val

        # Build list of check functions
        value_checks = []  # Checks on full_sample_value
        sampling_checks = []  # Checks on entire SigmondSampling

        # Add predicate check if provided
        if predicate is not None:
            sampling_checks.append(predicate)

        # Add min_val check
        if min_val is not None:
            v_min = get_value(min_val)
            value_checks.append(lambda v: v >= v_min)

        # Add max_val check
        if max_val is not None:
            v_max = get_value(max_val)
            value_checks.append(lambda v: v <= v_max)

        # Add comparison operator checks
        ops = {
            "gt": lambda t: lambda v: v > t,
            "lt": lambda t: lambda v: v < t,
            "ge": lambda t: lambda v: v >= t,
            "le": lambda t: lambda v: v <= t,
            "eq": lambda t: lambda v: v == t,
            "ne": lambda t: lambda v: v != t,
        }

        for op, target in kwargs.items():
            if op in ops:
                target_val = get_value(target)
                value_checks.append(ops[op](target_val))

        # Apply all checks with AND logic
        filtered = [
            samp
            for samp in self._data
            if all(chk(samp) for chk in sampling_checks)
            and all(chk(samp.full_sample_value) for chk in value_checks)
        ]

        return self._fast_load(filtered, self._return_type)

    def find_data(self, mode, target=None, key=None) -> SigmondSampling | None:
        """
        Find a single sampling based on data value criteria.

        Args:
            mode: Search mode (required) - 'min', 'max', 'closest'
                - 'min': Find sampling with minimum value (default: full_sample_value)
                - 'max': Find sampling with maximum value (default: full_sample_value)
                - 'closest': Find sampling closest to target value
            target: Target value for 'closest' mode. Can be float/int or SigmondSampling.
            key: Optional callable or attribute name to extract comparison value.
                 Default: 'full_sample_value'
                 Can be: 'mean', 'std', 'error', or custom function

        Returns:
            Single SigmondSampling object, or None if not found

        Examples:
            # Find min/max by full_sample_value
            collection.find_data(mode='min')
            collection.find_data(mode='max')

            # Find min/max by other attributes
            collection.find_data(mode='min', key='error')
            collection.find_data(mode='max', key=lambda s: s.mean / s.error)

            # Find closest to value
            collection.find_data(mode='closest', target=1.0)
            collection.find_data(mode='closest', target=reference_sampling)
        """
        if not self._data:
            return None

        def get_value(val):
            return val.full_sample_value if isinstance(val, SigmondSampling) else val

        if key is None:

            def key_func(s):
                return s.full_sample_value
        elif callable(key):
            key_func = key
        else:

            def key_func(s):
                return getattr(s, key)

        if mode == "min":
            return min(self._data, key=key_func)
        elif mode == "max":
            return max(self._data, key=key_func)
        elif mode == "closest":
            if target is None:
                raise ValueError("target must be provided for mode='closest'")
            target_val = get_value(target)
            return min(self._data, key=lambda s: abs(key_func(s) - target_val))
        else:
            raise ValueError(f"Unknown mode '{mode}'. Options: 'min', 'max', 'closest'")

    def sort_data(self: T, key="full_sample_value", reverse=False) -> T:
        """
        Sort collection by data values.

        Args:
            key: Data attribute to sort by (default: 'full_sample_value').
                Options: 'full_sample_value', 'mean', 'std', 'error'
                Can also be a callable that takes SigmondSampling and returns sortable value.
            reverse: If True, sort in descending order (default: False)

        Returns:
            Copy of ObservableCollection sorted by specified data

        Examples:
            # Sort by full sample value (ascending)
            collection.sort_data()

            # Sort by mean (descending)
            collection.sort_data(key='mean', reverse=True)

            # Sort by error
            collection.sort_data(key='error')

            # Custom sort key
            collection.sort_data(key=lambda s: s.mean / s.error)

            # Method chaining
            collection.filter_data(gt=0.5).sort_data()
        """
        if callable(key):
            sort_key = key
        else:
            valid_keys = ["full_sample_value", "mean", "std", "error"]
            if key not in valid_keys:
                raise ValueError(f"key must be one of {valid_keys} or a callable function")

            def sort_key(s):
                return getattr(s, key)

        new_data = sorted(self._data, key=sort_key, reverse=reverse)
        return self._fast_load(new_data, self._return_type)

    def to_list(self) -> list[SigmondSampling]:
        """Return list of SigmondSampling objects."""
        return self._data[:]

    def to_dict(self) -> dict:
        """Return dictionary mapping ObsInfo to SigmondSampling."""
        return {samp.observable_info: samp for samp in self._data}

    def to_numpy(self) -> "np.ndarray":
        """Return NumPy array of data values.
        Shape: (N, M)
            where N is number of observables and M is the sample size
            (num_resamplings + 1 for SigmondSampling, num_bins for SigmondBins)."""
        if not self._data:
            return np.array([])
        try:
            self.shared_attr(lambda s: len(s.data), strict=True)
        except ValueError:
            raise ValueError(
                "All samplings must have the same sample length to convert to NumPy array."
            )
        return np.array([np.asarray(samp.data) for samp in self._data])

    # TODO: would like a better updating mechanism. Want to be able to 'update' or 'append'.
    def to_hdf5(
        self, filename: str, create_backups: bool = True, root_path: str = "samplings"
    ) -> None:
        """
        Export collection data to HDF5 file in Sigmond format.

        If the collection holds ``SigmondSampling`` objects, the result is a Sigmond
        samplings file. If it holds ``SigmondBins`` objects, a Sigmond bins file is
        written instead. Mixed collections are rejected.

        Args:
            filename: Path to output HDF5 file
            create_backups: If True, create backups of existing files (default: True)
            root_path: Root path in HDF5 file to store data (default: "samplings")
        """
        from .bins import SigmondBins
        from .ensemble_collection import SingleEnsembleCollection
        from .writer import SigmondWriter

        # Validate single-ensemble / shared sampling
        try:
            SingleEnsembleCollection(self._data)
        except ValueError:
            raise ValueError(
                "All observables must belong to the same ensemble and have compatible "
                "sampling metadata to export to HDF5."
            )

        all_bins = all(isinstance(s, SigmondBins) for s in self._data)
        all_samplings = all(isinstance(s, SigmondSampling) for s in self._data)
        if not (all_bins or all_samplings):
            raise ValueError(
                "Cannot export a mixed collection of SigmondBins and SigmondSampling "
                "to a single HDF5 file."
            )

        writer = SigmondWriter(create_backups=create_backups)
        if all_bins:
            writer.write_bins_hdf5(filename, self._data, root_path=root_path, overwrite=True)
        else:
            writer.write_hdf5(filename, self._data, root_path=root_path, overwrite=True)

    def __iter__(self):
        """Iterate over SigmondSampling objects."""
        return iter(self._data)

    def __len__(self):
        """Return number of observables in collection."""
        return len(self._data)

    def __getitem__(self: T, key) -> SigmondSampling | T:
        """
        Get sampling by integer index or slice.
        """
        if isinstance(key, slice):
            # Return new collection for slices via fast path
            return self._fast_load(self._data[key], self._return_type)

        if isinstance(key, (list, tuple, np.ndarray)):
            mask = None
            indices = None

            if isinstance(key, np.ndarray):
                if key.dtype == bool:
                    mask = key
                elif np.issubdtype(key.dtype, np.integer):
                    indices = key.tolist()
                else:
                    raise TypeError("Index array must be bool or int dtype")
            else:
                if key:
                    is_bool_mask = all(isinstance(k, (bool, np.bool_)) for k in key)
                    if is_bool_mask:
                        mask = key
                    else:
                        indices = key
                else:
                    indices = key

            if mask is not None:
                if len(mask) != len(self._data):
                    raise IndexError("Boolean mask length must match collection length")
                data = [s for s, keep in zip(self._data, mask) if keep]
            else:
                data = [self._data[i] for i in indices]
            return self._fast_load(data, self._return_type)

        return self._data[key]

    def __contains__(self, item):
        """
        Check if a SigmondSampling or ObsInfo exists in collection.
        """
        from .sampling import ObservableInfo

        if isinstance(item, SigmondSampling):
            return item in self._data
        elif isinstance(item, ObservableInfo):
            return any(samp.observable_info == item for samp in self._data)
        else:
            return False

    def __repr__(self):
        return f"ObservableCollection({len(self)} observables)"

    def __bool__(self):
        """Return True if collection is non-empty."""
        return bool(self._data)

    def __add__(self: T, other: "ObservableCollection") -> T:
        if not isinstance(other, ObservableCollection):
            raise TypeError(
                f"unsupported operand type(s) for +: "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )

        # "other" wins: include all of other first, then add only self items not present in other
        other_data = other._data
        other_keys = {s.observable_info for s in other_data}

        out_data = list(other_data)  # copy; don't mutate other._data
        out_data.extend(s for s in self._data if s.observable_info not in other_keys)

        return self.__class__(out_data, return_type=self._return_type)

    def __sub__(self: T, other: "ObservableCollection") -> T:
        if not isinstance(other, ObservableCollection):
            raise TypeError(
                f"unsupported operand type(s) for -: "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )

        other_keys = {s.observable_info for s in other._data}
        out_data = [s for s in self._data if s.observable_info not in other_keys]

        return self.__class__(out_data, return_type=self._return_type)
