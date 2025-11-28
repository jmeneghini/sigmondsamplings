"""
SpectraCollection: A fast, queryable collection of spectra.
"""

from typing import Dict, List, Callable
from .sampling import SigmondSampling

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class SpectraCollection:
    """
    A queryable collection of spectra with fast filtering and iteration.

    Provides convenient filtering, iteration, and conversion methods for spectra.
    Uses __slots__ for memory efficiency and fast attribute access.
    """

    __slots__ = ('_data', '_return_type')

    def __init__(self, data: List[SigmondSampling], return_type: str = 'list'):
        """
        Initialize SpectraCollection with deduplication.

        Args:
            data: List of SigmondSampling objects
            return_type: Return type for attribute access - 'dict' or 'list' (default: 'dict')
                - 'dict': Returns {obs_info: value} mapping
                - 'list': Returns [value1, value2, ...] list

        Note:
            Deduplicates based on SigmondSampling.__hash__ and __eq__
            while preserving insertion order.
        """
        # Deduplicate while preserving order (Python 3.7+ dict ordering)
        self._data = list(dict.fromkeys(data))

        # Validate and set return type
        if return_type not in ('dict', 'list'):
            raise ValueError("return_type must be 'dict' or 'list'")
        self._return_type = return_type

    @property
    def return_type(self) -> str:
        """Get the return type for attribute access ('dict' or 'list')."""
        return self._return_type

    @return_type.setter
    def return_type(self, value: str):
        """
        Set the return type for attribute access.

        Args:
            value: 'dict' or 'list'
        """
        if value not in ('dict', 'list'):
            raise ValueError("return_type must be 'dict' or 'list'")
        self._return_type = value

    def filter(self, **kwargs) -> "SpectraCollection":
        """
        Filter spectra by ObsInfo attributes (fast attribute-based filtering).

        Args:
            **kwargs: Attribute filters (e.g., irrep='A1g', psq=0, particle='pi')

        Returns:
            New SpectraCollection with filtered results

        Example:
            collection.filter(irrep='A1g', psq=0)
            collection.filter(particle='pi', reference=True)
        """
        # Fast path for single filter
        if len(kwargs) == 1:
            key, val = next(iter(kwargs.items()))
            filtered = [
                samp for samp in self._data
                if getattr(samp.observable_info, key, None) == val
            ]
        else:
            # Multiple filters - still optimized with generator
            filtered = [
                samp for samp in self._data
                if all(getattr(samp.observable_info, k, None) == v for k, v in kwargs.items())
            ]
        return SpectraCollection(filtered, return_type=self._return_type)

    def find(self, predicate: Callable) -> "SpectraCollection":
        """
        Filter using a custom predicate function (flexible but slower than filter).

        Args:
            predicate: Function that takes ObsInfo and returns bool

        Returns:
            New SpectraCollection with filtered results

        Example:
            collection.find(lambda obs: obs.psq < 5)
            collection.find(lambda obs: obs.irrep.startswith('A'))
        """
        filtered = [samp for samp in self._data if predicate(samp.observable_info)]
        return SpectraCollection(filtered, return_type=self._return_type)

    def sort(self, by=None, key=None, reverse=False):
        """
        Sort collection in-place by attribute(s) or custom key function.

        Args:
            by: Attribute name (str) or list of attribute names to sort by.
                Attributes can be on SigmondSampling or its observable_info.
            key: Custom key function (takes SigmondSampling, returns sortable value).
                Cannot be used together with 'by'.
            reverse: If True, sort in descending order (default: False)

        Returns:
            Self for method chaining

        Examples:
            # Sort by mean energy value
            collection.sort(by='mean')

            # Multi-level sort: by psq, then irrep, then mean
            collection.sort(by=['psq', 'irrep', 'mean'])

            # Custom key function
            collection.sort(key=lambda s: s.mean / s.error)

            # Descending order
            collection.sort(by='mean', reverse=True)

            # Method chaining
            collection.filter(irrep='A1g').sort(by='mean')
        """
        if by is not None and key is not None:
            raise ValueError("Cannot specify both 'by' and 'key' arguments")

        if by is None and key is None:
            raise ValueError("Must specify either 'by' or 'key' argument")

        if key is not None:
            # Use custom key function
            self._data.sort(key=key, reverse=reverse)
        else:
            # Sort by attribute(s)
            if isinstance(by, str):
                by = [by]

            # Create composite key function
            def composite_key(sampling):
                values = []
                for attr in by:
                    # Try to get from sampling first, then from observable_info
                    if hasattr(sampling, attr):
                        values.append(getattr(sampling, attr))
                    elif hasattr(sampling.observable_info, attr):
                        values.append(getattr(sampling.observable_info, attr))
                    else:
                        raise AttributeError(
                            f"Neither SigmondSampling nor its observable_info has attribute '{attr}'"
                        )
                return tuple(values)

            self._data.sort(key=composite_key, reverse=reverse)

        return self

    def to_dict(self) -> Dict:
        """Return dictionary mapping ObsInfo to SigmondSampling."""
        return {samp.observable_info: samp for samp in self._data}

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame with ObsInfo attributes as columns.

        Returns:
            DataFrame with columns for ObsInfo attributes and data

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for to_dataframe(). Install with: pip install pandas")

        # Pre-allocate list for better performance
        rows = []
        for sampling in self._data:
            obs_info = sampling.observable_info
            # Build row with common ObsInfo attributes
            row = {"name": str(obs_info.name), "data": sampling.pdg_format()}

            # Add all available attributes efficiently
            for attr in ("irrep", "psq", "energy_type", "particle", "ref_particle", "index"):
                if hasattr(obs_info, attr):
                    row[attr] = getattr(obs_info, attr)

            # Add reference flag
            row["reference"] = getattr(obs_info, "ref_particle", None) is not None

            rows.append(row)

        return pd.DataFrame(rows)

    def __iter__(self):
        """Iterate over SigmondSampling objects."""
        return iter(self._data)

    def __len__(self):
        """Return number of spectra in collection."""
        return len(self._data)

    def __getitem__(self, key):
        """
        Get sampling by integer index or slice.

        Args:
            key: Integer index or slice

        Returns:
            SigmondSampling for integer, or new SpectraCollection for slice

        Examples:
            collection[0]         # Get first sampling
            collection[0:5]       # Get new collection with first 5 items
        """
        # Integer indexing
        if isinstance(key, int):
            if key < 0:
                key = len(self._data) + key
            if key < 0 or key >= len(self._data):
                raise IndexError(f"Index {key} out of range for collection of size {len(self._data)}")
            return list(self._data)[key]

        # Slice indexing
        elif isinstance(key, slice):
            samplings = list(self._data)[key]
            return SpectraCollection(samplings, return_type=self._return_type)

        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key).__name__}")

    def __contains__(self, item):
        """
        Check if a SigmondSampling or ObsInfo exists in collection.

        Args:
            item: SigmondSampling or ObservableInfo to check

        Returns:
            True if item exists in collection
        """
        from .sampling import ObservableInfo

        if isinstance(item, SigmondSampling):
            return item in self._data
        elif isinstance(item, ObservableInfo):
            return any(samp.observable_info == item for samp in self._data)
        else:
            return False

    def __repr__(self):
        return f"SpectraCollection({len(self)} spectra)"

    def __bool__(self):
        """Return True if collection is non-empty."""
        return bool(self._data)

    def apply(self, func_or_method, *args, **kwargs):
        """
        Apply a method or function to all samplings in the collection.

        Args:
            func_or_method: Either a callable function or a method name (string)
            *args: Positional arguments to pass to the method/function
            **kwargs: Keyword arguments to pass to the method/function

        Returns:
            - SpectraCollection if all results are SigmondSampling objects
            - Dict mapping obs_info to results otherwise

        Examples:
            # Apply a method by name
            means = collection.apply('mean')  # Returns dict {obs_info: mean_value}

            # Apply with arguments
            scaled = collection.apply('__mul__', 2.0)  # Returns SpectraCollection

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
                # Assume it's a method name (string)
                method = getattr(sampling, func_or_method)
                result = method(*args, **kwargs)
            results[sampling.observable_info] = result

        # Check if all results are SigmondSampling objects
        if results and all(isinstance(r, SigmondSampling) for r in results.values()):
            return SpectraCollection(list(results.values()), return_type=self._return_type)
        else:
            # Return dict or list based on return_type setting
            if self._return_type == 'dict':
                return results
            else:
                return list(results.values())

    def __getattr__(self, name):
        """
        Delegate method/property access to all samplings.

        This allows calling SigmondSampling methods/properties directly on the collection:
            collection.mean  # Returns dict {obs_info: mean}
            collection.rescale(2.0)  # Returns SpectraCollection

        Args:
            name: Attribute/method name to access on each sampling

        Returns:
            Dict for properties, or method wrapper for methods
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Check if it's valid on SigmondSampling
        if not hasattr(SigmondSampling, name):
            raise AttributeError(
                f"'{type(self).__name__}' and 'SigmondSampling' have no attribute '{name}'"
            )

        attr = getattr(SigmondSampling, name)

        # Handle properties - return dict or list based on return_type setting
        if isinstance(attr, property):
            if self._return_type == 'dict':
                return {samp.observable_info: getattr(samp, name) for samp in self._data}
            else:
                return [getattr(samp, name) for samp in self._data]

        # Handle methods - return wrapper function
        def method_wrapper(*args, **kwargs):
            return self.apply(name, *args, **kwargs)

        return method_wrapper
