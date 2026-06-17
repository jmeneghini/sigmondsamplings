"""
Lazy (deferred-read) variants of :class:`SigmondSampling` / :class:`SigmondBins`.

These are produced by :class:`~sigmondsamplings.loader.SigmondLoader` when
``lazy=True``. They subclass the concrete container types so that every
``isinstance(obj, SigmondSampling)`` / ``isinstance(obj, SigmondBins)`` check,
arithmetic ufunc, equality, and dispatch site throughout the codebase keeps
working unchanged. The *only* deferred quantity is the sample array (``data``);
all metadata (``observable_info``, ``sampling_info``, ``ensemble_info``,
``is_complex``, and for bins ``num_bins``) is available without touching disk.

Method taxonomy
---------------
Lazy-safe (no read): ``observable_info``, ``sampling_info``, ``ensemble_info``,
``is_complex``, ``num_bins`` (bins), ``__hash__``, ``__eq__``, ``__len__``,
``__repr__``, and therefore every ``ObservableCollection`` query
(``filter``/``find``/``sort``/``group_by``/``unique``/``obs.*``/``samp.*``).

Materializing (reads the dataset on first touch): ``data``, ``mean``,
``error``, ``full_sample_value``, ``pdg_format``, arithmetic, ``to_numpy``,
``to_hdf5``, ``filter_data``/``sort_data``/``find_data``.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from .bins import SigmondBins
from .info import ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

__all__ = [
    "HDF5ObservableRecord",
    "LazySigmondSampling",
    "LazySigmondBins",
]


@dataclass(frozen=True)
class _FileRef:
    """Shared (deduplicated) reference to an HDF5 file + ``Values`` group.

    Records point at one of these instead of each storing their own filename so
    a file with thousands of observables doesn't carry thousands of copies of
    the same string. ``filename`` is stored absolute so deferred reads survive a
    later change of the process working directory.
    """

    filename: str
    values_group: str


@dataclass(frozen=True)
class HDF5ObservableRecord:
    """Everything needed to (a) answer metadata queries and (b) later read the array.

    A single record represents one *logical* observable, which may be backed by
    one HDF5 dataset (real- or imag-only) or two (a fused complex observable).
    Shapes are captured at index time (cheap header reads, not data) so lazy
    bins can report ``num_bins`` without materializing.
    """

    file: _FileRef
    observable_info: ObservableInfo
    sampling_info: SamplingInfo | None
    file_kind: str  # "samplings" or "bins"
    real_name: str | None
    imag_name: str | None
    real_shape: tuple[int, ...] | None
    imag_shape: tuple[int, ...] | None

    @property
    def is_complex(self) -> bool:
        """True iff the observable is backed by both a Re and an Im dataset."""
        return self.real_name is not None and self.imag_name is not None

    @property
    def sample_length(self) -> int:
        """Length of the sample axis, known from shapes without reading data."""
        shape = self.real_shape if self.real_shape is not None else self.imag_shape
        if shape is None:
            raise ValueError("Record has neither a real nor an imaginary dataset")
        return int(shape[0])


def _read_record(record: HDF5ObservableRecord) -> np.ndarray:
    """Materialization phase: read and fuse exactly the dataset(s) for one observable."""
    with h5py.File(record.file.filename, "r") as f:
        group = f[record.file.values_group]
        re_part = group[record.real_name][:] if record.real_name is not None else None
        im_part = group[record.imag_name][:] if record.imag_name is not None else None
    if re_part is not None and im_part is not None:
        return re_part + 1j * im_part
    return re_part if re_part is not None else im_part


class LazyDataMixin:
    """Defers ``data`` until first touch; everything else inherited from the base.

    Subclasses must set ``self._record`` and ``self._data_cache = None`` in
    ``__init__`` and provide ``_eager()`` returning a concrete (eager) copy.
    """

    # Minimum sample length expected after read (overridden per concrete type).
    _min_data_len = 1

    @property
    def data(self) -> np.ndarray:
        if self._data_cache is None:
            self._data_cache = self._coerce_and_validate(_read_record(self._record))
        return self._data_cache

    @data.setter
    def data(self, value: np.ndarray) -> None:
        # Materializing assignments (e.g. from in-place transforms) still work.
        self._data_cache = value

    def _coerce_and_validate(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"{type(self).__name__} data must be 1-dimensional")
        if len(arr) < self._min_data_len:
            raise ValueError(
                f"{type(self).__name__} data must have at least "
                f"{self._min_data_len} element(s)"
            )
        return arr.astype(complex) if self.is_complex else arr.astype(float)

    @property
    def is_materialized(self) -> bool:
        """True once the dataset has been read into memory."""
        return self._data_cache is not None

    def materialize(self):
        """Force the deferred read in place and return ``self``."""
        _ = self.data
        return self

    def copy(self):
        """Materialize, then return a plain *eager* copy (never a lazy clone)."""
        return self.materialize()._eager().copy()


class LazySigmondSampling(LazyDataMixin, SigmondSampling):
    """A :class:`SigmondSampling` whose sample array is read on first access."""

    _min_data_len = 2

    def __init__(self, record: HDF5ObservableRecord):
        # Deliberately bypass SigmondSampling.__init__: its len/ndim checks would
        # force a read. Validation is deferred to the first .data access.
        self._record = record
        self._data_cache = None
        self.observable_info = record.observable_info
        self.sampling_info = record.sampling_info
        self.is_complex = record.is_complex

    def _eager(self) -> SigmondSampling:
        return SigmondSampling(
            self.data, self.observable_info, self.sampling_info, is_complex=self.is_complex
        )

    def with_observable_info(self, observable_info: ObservableInfo) -> "LazySigmondSampling":
        """Return a lazy clone with updated observable metadata only."""
        new = self.__class__.__new__(self.__class__)
        new._record = self._record
        new._data_cache = self._data_cache
        new.observable_info = observable_info
        new.sampling_info = self.sampling_info
        new.is_complex = self.is_complex
        return new

    def __repr__(self) -> str:
        if self.is_materialized:
            return super().__repr__()
        return f"LazySigmondSampling(name='{self.observable_info}', state='lazy')"


class LazySigmondBins(LazyDataMixin, SigmondBins):
    """A :class:`SigmondBins` whose raw bins are read on first access.

    ``num_bins`` is overridden to read from cached shape metadata; this keeps
    ``__hash__``/``__eq__``/``__len__`` (which all route through ``num_bins``)
    lazy-safe, so collection construction and dedup do not trigger reads.
    """

    _min_data_len = 1

    def __init__(self, record: HDF5ObservableRecord):
        self._record = record
        self._data_cache = None
        self.observable_info = record.observable_info
        self.is_complex = record.is_complex
        # Deferred file I/O is orthogonal to the dask out-of-core path.
        self.use_dask = False
        self.xp = np

    @property
    def num_bins(self) -> int:
        return int(self._record.sample_length)

    def _eager(self) -> SigmondBins:
        return SigmondBins(self.data, self.observable_info, is_complex=self.is_complex)

    def with_observable_info(self, observable_info: ObservableInfo) -> "LazySigmondBins":
        """Return a lazy clone with updated observable metadata only."""
        new = self.__class__.__new__(self.__class__)
        new._record = self._record
        new._data_cache = self._data_cache
        new.observable_info = observable_info
        new.is_complex = self.is_complex
        new.use_dask = self.use_dask
        new.xp = self.xp
        return new

    def __repr__(self) -> str:
        if self.is_materialized:
            return super().__repr__()
        return (
            f"LazySigmondBins(name='{self.observable_info}', "
            f"n_bins={self.num_bins}, state='lazy')"
        )
