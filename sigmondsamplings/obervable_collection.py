
"""
ObservableCollection: A fast, queryable collection of observables.
"""

from typing import Dict, List, Callable, Optional, TypeVar, Type, Iterable, Union, Any
from .sampling import SigmondSampling
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# TypeVar for fluent interface
T = TypeVar('T', bound='ObservableCollection')


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
            raise ImportError("pandas required for to_dataframe(). Install with: pip install pandas")
        
        # Access _data from the main class
        data_source = getattr(self, '_data', [])
        
        if not data_source:
            return pd.DataFrame()

        rows = []
        for s in data_source:
            # Base row with data value
            row = {
                "name": str(s.observable_info.name),
                "data": s.pdg_format()
            }
            
            # Dynamically add all available ObsInfo attributes
            # This replaces the hardcoded list of attributes
            obs_info = s.observable_info
            if hasattr(obs_info, '__dict__'):
                attrs = vars(obs_info)
            elif hasattr(obs_info, '_asdict'):  # NamedTuple
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
    """
    def __init__(self, collection, target_extractor: Callable[[Any], Any]):
        self._collection = collection
        self._extractor = target_extractor

    def __getattr__(self, name):
        # 1. Safety check on empty collection
        if not self._collection._data:
            return [] if self._collection.return_type == 'list' else {}

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
                if self._collection.return_type == 'dict':
                    return {
                        item.observable_info: res 
                        for item, res in zip(self._collection._data, results)
                    }
                return results
            
            return method_proxy

        # 4. CASE B: It is a Property/Data -> Return the values immediately
        values = [getattr(self._extractor(item), name) for item in self._collection._data]

        if self._collection.return_type == 'dict':
            return {
                item.observable_info: val 
                for item, val in zip(self._collection._data, values)
            }
        return values


class ObservableCollection(PandasExportMixin):
    """
    A queryable collection of observables with fast filtering and iteration.

    Provides convenient filtering, iteration, and conversion methods for observables.
    Uses __slots__ for memory efficiency and fast attribute access.
    """

    __slots__ = ('_data', '_return_type')

    def __init__(self, data: Iterable[SigmondSampling], return_type: str = 'list'):
        """
        Initialize ObservableCollection with deduplication.

        Args:
            data: List of SigmondSampling objects
            return_type: Return type for attribute access - 'dict' or 'list' (default: 'dict')
        """
        # Deduplicate while preserving order (Python 3.7+ dict ordering)
        self._data = list(dict.fromkeys(data))

        # Validate and set return type
        if return_type not in ('dict', 'list'):
            raise ValueError("return_type must be 'dict' or 'list'")
        self._return_type = return_type

    @classmethod
    def _fast_load(cls: Type[T], data: List[SigmondSampling], return_type: str) -> T:
        """
        Internal constructor to bypass validation/deduplication for trusted data.
        Used by filter/sort methods to return new instances efficiently.
        """
        instance = cls.__new__(cls)
        instance._data = data
        instance._return_type = return_type
        return instance

    @property
    def return_type(self) -> str:
        """Get the return type for attribute access ('dict' or 'list')."""
        return self._return_type

    @return_type.setter
    def return_type(self, value: str):
        if value not in ('dict', 'list'):
            raise ValueError("return_type must be 'dict' or 'list'")
        self._return_type = value

    @property
    def obs(self) -> AttributeAccessor:
        """
        Namespace for ObservableInfo attributes.
        Usage: collection.obs.irrep -> ['A1g', ...]
        """
        return AttributeAccessor(self, lambda s: s.observable_info)

    @property
    def val(self) -> AttributeAccessor:
        """
        Namespace for Data/Value attributes.
        Usage: collection.val.mean -> [0.5, 0.6, ...]
        """
        return AttributeAccessor(self, lambda s: s)
    
    """
    Supports filtering/finding/sorting by:
    1. ObservableInfo attributes (e.g., irrep='A1g')
    2. SamplingInfo attributes (e.g., method='jackknife')
    3. Direct object comparison (e.g., sampling_info=my_sampling_obj)
    """

    def filter(self: T, predicate=None, **kwargs) -> T:
        """
        Filter observables by attributes or direct object comparison.
        """
        if predicate is not None:
            # Use custom predicate function (applied to ObsInfo only)
            filtered = [samp for samp in self._data if predicate(samp.observable_info)]
        elif not kwargs:
            # Return shallow copy if no filters provided
            return self._fast_load(self._data[:], self._return_type)
        else:
            if not self._data:
                 return self._fast_load([], self._return_type)

            # Use shared logic to prepare filter lists
            direct_checks, obs_filters, samp_filters = self._prepare_criteria(kwargs)

            # Optimized loop
            filtered = [
                samp for samp in self._data
                if (not direct_checks or all(chk(samp) for chk in direct_checks))
                and (not obs_filters or all(getattr(samp.observable_info, k, None) == v for k, v in obs_filters))
                and (not samp_filters or all(getattr(samp.sampling_info, k, None) == v for k, v in samp_filters))
            ]
            
        return self._fast_load(filtered, self._return_type)

    def find(self, predicate: Callable = None, **kwargs) -> Optional[SigmondSampling]:
        """
        Find first sampling matching predicate or attribute criteria.
        """
        if not self._data:
            return None

        if predicate is not None:
            # Use custom predicate function
            for samp in self._data:
                if predicate(samp.observable_info):
                    return samp
            return None
        
        if not kwargs:
            return self._data[0]

        # Use shared logic to prepare filter lists
        direct_checks, obs_filters, samp_filters = self._prepare_criteria(kwargs)

        # Optimized search loop
        for samp in self._data:
            if (not direct_checks or all(chk(samp) for chk in direct_checks)) \
            and (not obs_filters or all(getattr(samp.observable_info, k, None) == v for k, v in obs_filters)) \
            and (not samp_filters or all(getattr(samp.sampling_info, k, None) == v for k, v in samp_filters)):
                return samp
                
        return None

    def sort(self: T, by=None, key=None, reverse=False) -> T:
        """
        Sort collection by attribute(s) or custom key function.
        Returns a NEW instance (Immutable).
        """
        if (by is None) == (key is None):
            raise ValueError("Must specify exactly one of 'by' or 'key'")

        if not self._data:
            return self._fast_load([], self._return_type)

        if key is not None:
            sort_key = key
        else:
            # Use shared logic to build the optimized key function
            sort_key = self._build_composite_sort_key(by)

        # Create sorted COPY of data
        new_data = sorted(self._data, key=sort_key, reverse=reverse)
        return self._fast_load(new_data, self._return_type)
    
    def _prepare_criteria(self, kwargs: Dict) -> tuple:
        """
        Internal helper to parse kwargs into direct object checks and attribute filters.
        Routes attributes to ObservableInfo or SamplingInfo based on first item inspection.
        
        Returns:
            (direct_checks, obs_filters, samp_filters)
        """
        direct_checks = []
        obs_filters = []
        samp_filters = []

        # 1. Direct Object Matching (pop from kwargs so they aren't treated as attributes)
        if 'sampling_info' in kwargs:
            target = kwargs.pop('sampling_info')
            direct_checks.append(lambda s: s.sampling_info == target)
        
        if 'observable_info' in kwargs:
            target = kwargs.pop('observable_info')
            direct_checks.append(lambda s: s.observable_info == target)

        # If we have no data to inspect, we can't route attributes.
        # Return what we have; the caller handles the empty data case.
        if not self._data or not kwargs:
            return direct_checks, obs_filters, samp_filters

        # 2. Attribute Routing (Inspection Strategy)
        first_item = self._data[0]

        for k, v in kwargs.items():
            # Priority 1: Check ObservableInfo attributes
            if hasattr(first_item.observable_info, k):
                obs_filters.append((k, v))
            # Priority 2: Check SamplingInfo attributes
            elif hasattr(first_item.sampling_info, k):
                samp_filters.append((k, v))
            else:
                # Fallback: Assume it's an ObsInfo attribute
                obs_filters.append((k, v))
        
        return direct_checks, obs_filters, samp_filters

    def _build_composite_sort_key(self, by) -> Callable:
        """
        Internal helper to build a sort key function.
        Inspects the first item to find where attributes live.
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
        
        # Return optimized key function
        return lambda sampling: tuple(ex(sampling) for ex in extractors)

    def filter_data(self: T, predicate=None, min_val=None, max_val=None, **kwargs) -> T:
        """
        Filter observables based on their data values (full_sample_value).

        Args:
            predicate: Callable that takes a SigmondSampling and returns bool.
                      If provided, other arguments are ignored.
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
        """
        def get_value(val):
            return val.full_sample_value if isinstance(val, SigmondSampling) else val

        if predicate is not None:
            filtered = [samp for samp in self._data if predicate(samp)]
            return self._fast_load(filtered, self._return_type)

        # Performance Optimization: 
        # Build list of check functions OUTSIDE the loop
        checks = []
        if min_val is not None: 
            v_min = get_value(min_val)
            checks.append(lambda v: v >= v_min)
        if max_val is not None: 
            v_max = get_value(max_val)
            checks.append(lambda v: v <= v_max)

        ops = {
            'gt': lambda t: (lambda v: v > t),
            'lt': lambda t: (lambda v: v < t),
            'ge': lambda t: (lambda v: v >= t),
            'le': lambda t: (lambda v: v <= t),
            'eq': lambda t: (lambda v: v == t),
            'ne': lambda t: (lambda v: v != t),
        }
        
        for op, target in kwargs.items():
            if op in ops:
                target_val = get_value(target)
                checks.append(ops[op](target_val))

        filtered = [
            samp for samp in self._data
            if all(chk(samp.full_sample_value) for chk in checks)
        ]

        return self._fast_load(filtered, self._return_type)

    def find_data(self, mode, target=None, key=None) -> Optional[SigmondSampling]:
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
            key_func = lambda s: s.full_sample_value
        elif callable(key):
            key_func = key
        else:
            key_func = lambda s: getattr(s, key)

        if mode == 'min':
            return min(self._data, key=key_func)
        elif mode == 'max':
            return max(self._data, key=key_func)
        elif mode == 'closest':
            if target is None:
                raise ValueError("target must be provided for mode='closest'")
            target_val = get_value(target)
            return min(self._data, key=lambda s: abs(key_func(s) - target_val))
        else:
            raise ValueError(f"Unknown mode '{mode}'. Options: 'min', 'max', 'closest'")

    def sort_data(self: T, key='full_sample_value', reverse=False) -> T:
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
            valid_keys = ['full_sample_value', 'mean', 'std', 'error']
            if key not in valid_keys:
                raise ValueError(f"key must be one of {valid_keys} or a callable function")
            sort_key = lambda s: getattr(s, key)

        new_data = sorted(self._data, key=sort_key, reverse=reverse)
        return self._fast_load(new_data, self._return_type)

    def to_dict(self) -> Dict:
        """Return dictionary mapping ObsInfo to SigmondSampling."""
        return {samp.observable_info: samp for samp in self._data}
    
    def to_numpy(self) -> "np.ndarray":
        """Return NumPy array of data values.
        Shape: (N, M)
            where N is number of observables and M is sample size."""
        return np.array(self.val.data)
    
    def to_hdf5(self, filename: str, create_backups: bool = True, root_path: str = "/samplings") -> None:
        """
        Export collection data to HDF5 file in Sigmond format.

        Args:
            filename: Path to output HDF5 file
            create_backups: If True, create backups of existing files (default: True)
            root_path: Root path in HDF5 file to store data (default: "/")
                This is ONLY used if all ensemble/sampling info are the same.
                Otherwise, data is organized by ensemble/sampling info pairs.
        """
        from .writer import SigmondWriter
        writer = SigmondWriter(create_backups=create_backups)
        # Seperate observables by ensemble and sampling info pairs
        output_dict = {}
        unique_ensembles = set(self.obs.ensemble_info)
        for ensemble in unique_ensembles:
            these_samplings = self.filter(ensemble_info=ensemble)
            unique_sampling_info = set(these_samplings.val.sampling_info)
            for sampling_info in unique_sampling_info:
                final_samplings = these_samplings.filter(sampling_info=sampling_info)
                output_dict[(ensemble, sampling_info)] = list(final_samplings)
        
        for (ensemble, sampling_info), samplings in output_dict.items():
            if len(output_dict) > 1:
                root_path = f"/{ensemble.slug}_{sampling_info.slug}"
            writer.write_hdf5(
                filename,
                samplings,
                root_path=root_path,
                overwrite=True
            )
            # turn off backups after first write
            writer = SigmondWriter(create_backups=False)
        
        
    def __iter__(self):
        """Iterate over SigmondSampling objects."""
        return iter(self._data)

    def __len__(self):
        """Return number of observables in collection."""
        return len(self._data)

    def __getitem__(self: T, key) -> Union[SigmondSampling, T]:
        """
        Get sampling by integer index or slice.
        Performance Fix: Removed unnecessary list() cast.
        """
        if isinstance(key, slice):
            # Return new collection for slices via fast path
            return self._fast_load(self._data[key], self._return_type)
            
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
        """
        Add two ObservableCollection objects together, combining their data.
        """
        if not isinstance(other, ObservableCollection):
            raise TypeError(
                f"unsupported operand type(s) for +: 'ObservableCollection' and '{type(other).__name__}'"
            )

        # Standard init used here to ensure proper deduplication logic
        combined_data = list(self._data) + list(other._data)
        return self.__class__(combined_data, return_type=self._return_type)

    def apply(self: T, func_or_method, *args, **kwargs):
        """
        Apply a method or function to all samplings in the collection.

        Args:
            func_or_method: Either a callable function or a method name (string)
            *args: Positional arguments to pass to the method/function
            **kwargs: Keyword arguments to pass to the method/function

        Returns:
            - ObservableCollection if all results are SigmondSampling objects
            - Dict mapping obs_info to results otherwise

        Examples:
            # Apply a method by name
            means = collection.apply('mean')  # Returns dict {obs_info: mean_value}

            # Apply with arguments
            scaled = collection.apply('__mul__', 2.0)  # Returns ObservableCollection

            # Apply a lambda function
            energies = collection.apply(lambda s: s.full_sample_value)

            # Chain operations
            filtered = collection.filter(irrep='A1g').apply('rescale', 2.0)
        """
        results = {}
        for sampling in self._data:
            if callable(func_or_method):
                result = func_or_method(sampling, *args, **kwargs)
            else:
                method = getattr(sampling, func_or_method)
                result = method(*args, **kwargs)
            results[sampling.observable_info] = result

        # Check if all results are SigmondSampling objects
        if results and all(isinstance(r, SigmondSampling) for r in results.values()):
            return self._fast_load(list(results.values()), self._return_type)
        else:
            if self._return_type == 'dict':
                return results
            else:
                return list(results.values())