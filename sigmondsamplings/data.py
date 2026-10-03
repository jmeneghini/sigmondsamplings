"""
Shared base for Monte Carlo data containers of any shape.

:class:`SigmondData` holds a sample buffer of shape ``S + (N,)`` -- the sample
axis is always last -- together with its metadata, and implements everything
that does not depend on how the sample axis is read: arithmetic via the NumPy
ufunc protocol, comparisons, formatting and the simple transformations.
Everything that *does* depend on it (central value, error, which part of the
axis holds resamples) is delegated to the analyzer chosen by ``sampling_info``
(see :mod:`sigmondsamplings.analysis`).

``S = ()`` is a single observable: ``data`` is 1-D, ``observable_info`` is its
``ObservableInfo`` and the statistics are Python scalars. ``S != ()`` is an
N-D, label-indexed array of observables: ``observable_info`` describes the array
as a whole, ``infos`` holds the per-cell ``ObservableInfo`` objects, and the
statistics are ``xarray.DataArray`` objects over the spatial dims. Labels,
selection and N-D construction live in :mod:`sigmondsamplings.sampling_array`,
which (with xarray) is only imported once an N-D object is used.

Concrete types:
    - :class:`~sigmondsamplings.sampling.SigmondSampling`: jackknife/bootstrap
      samplings, ``data[..., :] = [full, resample_1, ..., resample_n]``.
    - :class:`~sigmondsamplings.bins.SigmondBins`: raw bins, ``sampling_info is None``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .analysis import _Analyzer, analyzer_for
from .info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo

try:
    from uncertainties import ufloat

    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    UNCERTAINTIES_AVAILABLE = False


_NO_COORDS: Mapping[str, np.ndarray] = MappingProxyType({})


def _nd():
    """The N-D support module, imported on first use (it pulls in xarray)."""
    from . import sampling_array

    return sampling_array


def _kind_cls(sampling_info: SamplingInfo | None) -> type[SigmondData]:
    """The eager concrete class for ``sampling_info`` (``None`` means bins)."""
    if sampling_info is None:
        from .bins import SigmondBins

        return SigmondBins
    from .sampling import SigmondSampling

    return SigmondSampling


def _merged_observable_info(operands: list[SigmondData]) -> ObservableInfo:
    """Observable info for the result of an operation on 0-d ``operands``."""
    first_info = operands[0].observable_info
    # TODO: this shouldn't preserve observable_info. However, this may have significant
    # downstream consequences, so we should consider how to handle this in a future PR.
    # For now, we preserve observable_info if all operands have the same one, and
    # otherwise create a new one with a mixed name.
    if all(s.observable_info == first_info for s in operands):
        return first_info
    if all(s.observable_info.ensemble_info == first_info.ensemble_info for s in operands):
        return ObservableInfo("mixed_operation", 0, "n", "re", first_info.ensemble_info)
    return ObservableInfo("mixed_operation", 0, "n", "re", INDEP_ENSEMBLE)


class SigmondData:
    """Abstract base: a sample buffer of shape ``S + (N,)`` plus metadata.

    Subclasses set ``data``, ``observable_info``, ``sampling_info`` and
    ``is_complex`` for a single observable, and implement :meth:`_kind_key`.
    N-D instances additionally carry ``_dims``, ``_coords`` and ``_infos``; the
    class-level defaults below describe a single observable, so 0-d objects
    (including the lazy subclasses) never have to set them.
    """

    data: np.ndarray
    observable_info: ObservableInfo
    sampling_info: SamplingInfo | None
    is_complex: bool

    _dims: tuple[str, ...] = ()
    _coords: Mapping[str, np.ndarray] = _NO_COORDS
    _infos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _from_parts(
        cls,
        data: np.ndarray,
        observable_info: ObservableInfo,
        sampling_info: SamplingInfo | None,
        is_complex: bool,
        dims: tuple[str, ...] = (),
        coords: Mapping[str, Any] | None = None,
        infos: np.ndarray | None = None,
    ) -> SigmondData:
        """Trusted constructor for any shape; bypasses the 0-d ``__init__`` checks."""
        data = np.asarray(data)
        dims = tuple(dims)
        if data.ndim != len(dims) + 1:
            raise ValueError(f"data has rank {data.ndim}, expected {len(dims) + 1} for dims {dims}")
        new = cls.__new__(cls)
        new.data = data.astype(complex if is_complex else float, copy=False)
        new.observable_info = observable_info
        new.sampling_info = sampling_info
        new.is_complex = is_complex
        if dims:
            if infos is None or infos.shape != data.shape[:-1]:
                raise ValueError(f"infos must have shape {data.shape[:-1]}")
            new._dims = dims
            new._coords = MappingProxyType(
                {d: np.asarray(v) for d, v in (coords or {}).items() if d in dims}
            )
            new._infos = infos
        return new

    @classmethod
    def from_data(cls, data, obs_info, sampling_info, **kwargs) -> SigmondData:
        """Wrap a raw buffer of shape ``S + (N,)``; see :func:`sampling_array.from_data`."""
        return _nd().from_data(cls, data, obs_info, sampling_info, **kwargs)

    @classmethod
    def from_samplings(cls, nested, **kwargs) -> SigmondData:
        """Build from nested SigmondSamplings/Bins; see :func:`sampling_array.from_samplings`."""
        return _nd().from_samplings(cls, nested, **kwargs)

    @classmethod
    def from_collection(cls, source, shape_by, **kwargs) -> SigmondData:
        """Pivot a collection onto axes; see :func:`sampling_array.from_collection`."""
        return _nd().from_collection(cls, source, shape_by, **kwargs)

    @staticmethod
    def stack(arrays, axis: int = 0, dim_name: str = "stack", coords=None) -> SigmondData:
        """Like ``np.stack`` but with an explicit name (and optional coords) for the new dim."""
        return _nd().stack(arrays, axis=axis, dim_name=dim_name, coords=coords)

    def _new(self, data, observable_info: ObservableInfo, is_complex: bool) -> SigmondData:
        """An eager container of the same kind and labels around new data."""
        return _kind_cls(self.sampling_info)._from_parts(
            data,
            observable_info,
            self.sampling_info,
            is_complex,
            self._dims,
            self._coords,
            self._infos,
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _kind_key(self) -> tuple:
        """Metadata key for the kind (must not read data)."""
        raise NotImplementedError

    def _identity_key(self) -> tuple:
        """Metadata-only key used for ``__eq__``/``__hash__`` (must not read data)."""
        if not self._dims:
            return self._kind_key()
        labels = (
            self._dims,
            self.shape,
            tuple(self._infos.flat),
            tuple((d, tuple(v.tolist())) for d, v in self._coords.items()),
        )
        return self._kind_key() + labels

    def _as_numpy(self) -> np.ndarray:
        """The sample buffer as a concrete numpy array."""
        return np.asarray(self.data)

    @property
    def _sample_length(self) -> int:
        return int(self.data.shape[-1])

    @property
    def _analyzer(self) -> _Analyzer:
        return analyzer_for(self.sampling_info)

    # ------------------------------------------------------------------
    # Shape and labels
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dims (the sample axis is not counted)."""
        return len(self._dims)

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape ``S``; ``()`` for a single observable."""
        return tuple(self.data.shape[:-1]) if self._dims else ()

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        """Index coordinates per spatial dim (dims without labels are absent)."""
        return self._coords

    @property
    def infos(self) -> np.ndarray:
        """Per-cell ObservableInfo objects, an object array of shape ``S``."""
        if self._dims:
            return self._infos
        out = np.empty((), dtype=object)
        out[()] = self.observable_info
        return out

    @property
    def num_samples(self) -> int:
        """Length of the sample axis (resamplings + 1, or the number of bins)."""
        return self._sample_length

    def _require_scalar(self, what: str) -> SigmondData:
        if self._dims:
            raise TypeError(f"{what} needs a single observable; index or select a cell first")
        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def ensemble_info(self) -> EnsembleInfo:
        """Get ensemble info from the observable."""
        return self.observable_info.ensemble_info

    @ensemble_info.setter
    def ensemble_info(self, value: EnsembleInfo):
        """Set ensemble info for the observable."""
        self.observable_info.ensemble_info = value

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _stat(self, func):
        """Reduce the sample axis with ``func``: a scalar if 0-d, else a DataArray over ``S``."""
        out = func(self._as_numpy())
        if self._dims:
            return _nd().wrap_stat(self, out)
        return np.asarray(out).item()

    @property
    def full_sample_value(self):
        """The estimate from the full ensemble (no resampling)."""
        return self._stat(self._analyzer.full)

    @property
    def mean(self):
        """Mean over the resamples (over the bins, for raw bins)."""
        return self._stat(self._analyzer.mean)

    @property
    def error(self):
        """Statistical error estimate, following sigmond's jack/boot_analyze."""
        return self._stat(self._analyzer.error)

    def confidence_interval(self, confidence_level: float = 0.68) -> tuple:
        """Percentile confidence interval (bootstrap resamples, or raw bins).

        Args:
            confidence_level: Confidence level (0.68 = 1σ, 0.95 = 2σ, etc.)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Raises:
            ValueError: For jackknife samplings.
        """
        lower, upper = self._analyzer.confidence_interval(self._as_numpy(), confidence_level)
        if self._dims:
            nd = _nd()
            return nd.wrap_stat(self, lower), nd.wrap_stat(self, upper)
        return (np.asarray(lower).item(), np.asarray(upper).item())

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _ufloat(self, value) -> ufloat:
        if not UNCERTAINTIES_AVAILABLE:
            raise ImportError(
                "uncertainties package is required for ufloat conversion. "
                "Install with: pip install uncertainties"
            )
        self._require_scalar("ufloat conversion")
        if self.is_complex:
            raise ValueError(
                f"Complex {self._analyzer.noun} cannot be converted to ufloat. "
                "Use .to_real() first."
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using UFloat objects with std_dev==0 may give unexpected results.",
                category=UserWarning,
            )
            return ufloat(value, self.error)

    def to_ufloat(self) -> ufloat:
        """Return an uncertainties.ufloat built from (full_sample_value, error)."""
        return self._ufloat(self._require_scalar("to_ufloat").full_sample_value)

    def pdg_format(self, format_spec: str = ".2uS") -> str:
        """Format value and error using PDG conventions via the uncertainties package.

        Args:
            format_spec: Format specification for uncertainties package
                       Common options:
                       - '.1uS': Shorthand notation like 1.23(4)
                       - '.1uP': Pretty-print with ± symbol
                       - '.1uL': LaTeX notation
                       - '.1ue': Scientific notation

        Complex data is formatted as ``re ± im i``, each part with its own error.
        """
        if self.is_complex:
            return self._complex_pdg_format(format_spec)
        return f"{self.to_ufloat():{format_spec}}"

    def _complex_pdg_format(self, format_spec: str, **kwargs) -> str:
        """``re ± im i``: ``pdg_format`` of each part, with its own error."""
        self._require_scalar("pdg_format")
        real = self.to_real().pdg_format(format_spec, **kwargs)
        imag = self._new(self.data.imag, self.observable_info, is_complex=False)
        imag = imag.pdg_format(format_spec, **kwargs)
        sign = "-" if imag.startswith("-") else "+"
        return f"{real} {sign} {imag.removeprefix('-')}i"

    @property
    def latex_str(self):
        return f"{self.observable_info.latex_str} = {self.pdg_format()}"

    def __str__(self):
        if self._dims:
            return repr(self)
        return f"{self.full_sample_value:.6f} ± {self.error:.6f}"

    def _array_repr(self) -> str:
        return (
            f"{type(self).__name__}(name='{self.observable_info.name}', "
            f"shape={self.shape}, dims={self.dims}, "
            f"N={self.num_samples}, ensemble='{self.ensemble_info}')"
        )

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def component(self, component: Literal["real", "imag"]) -> SigmondData:
        """Project a component, keeping array and cell metadata consistent."""
        if component not in ("real", "imag"):
            raise ValueError("component must be 'real' or 'imag'")
        if component == "imag" and not self.is_complex:
            raise ValueError("real data has no imaginary part")
        info = self.observable_info.copy()
        info.re_im = "re" if component == "real" else "im"
        data = self.data.real if component == "real" else self.data.imag
        out = self._new(data, info, is_complex=False)
        if self.ndim:
            infos = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                infos[idx] = self.infos[idx].copy()
                infos[idx].re_im = info.re_im
            out = out._with_replaced_infos(infos)
        return out

    def to_real(self) -> SigmondData:
        """Return the real component with real component metadata."""
        return self.component("real")

    def to_complex(self):
        """Return a new instance with the data promoted to complex dtype."""
        if self.is_complex:
            return self
        return self._new(self.data.astype(complex), self.observable_info, is_complex=True)

    def with_observable_info(self, observable_info: ObservableInfo):
        """
        Return a metadata-only view with replaced observable info.

        The sample array (and, for lazy subclasses, the deferred read) is shared
        with the original object.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.observable_info = observable_info
        return new

    def copy(self):
        """Return a copy with an independent data array and observable info(s).

        sampling_info is shared (treated as immutable). data is copied so
        in-place mutations don't bleed across; each ObservableInfo goes through
        its own copy() to preserve subclass type.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.data = self.data.copy()
        new.observable_info = self.observable_info.copy()
        if self._dims:
            new._infos = _copy_infos(self._infos)
        return new

    def bounded(self, lower: float, upper: float):
        """
        Return a new instance with the resamples clipped into [lower, upper].

        For samplings the full-sample value [0] is unchanged; for raw bins every
        bin is clipped.
        """
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        arr = self._as_numpy().copy()
        sl = self._analyzer.resample_slice
        arr[..., sl] = np.clip(arr[..., sl], lower, upper)
        return self._new(arr, self.observable_info, self.is_complex)

    def unwrap(self, discont=np.pi, axis=-1):
        """
        Unwrap phase angles by changing jumps greater than discont to their 2*pi complement.

        Args:
            discont: Maximum discontinuity between values (default: pi)
            axis: Axis along which unwrap will operate (default: -1)
        """
        unwrapped = np.unwrap(self._as_numpy(), discont=discont, axis=axis)
        return self._new(unwrapped, self.observable_info, self.is_complex)

    # ------------------------------------------------------------------
    # N-D: selection, conversion, attribute accessors, editing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """Positional indexing over the spatial dims (never the sample axis)."""
        self._require_indexable()
        return _nd().getitem(self, key)

    def sel(self, **kwargs):
        """Label-based selection; a fully-indexed result is a single observable."""
        self._require_indexable()
        return _nd().sel(self, **kwargs)

    def isel(self, **kwargs):
        """Integer-based selection; a fully-indexed result is a single observable."""
        self._require_indexable()
        return _nd().isel(self, **kwargs)

    def transpose(self, *dims: str) -> SigmondData:
        """Reorder the spatial dims (the sample axis stays last)."""
        self._require_indexable()
        return _nd().transpose(self, *dims)

    def rename_dims(self, mapping: Mapping[str, str] | None = None, **names: str) -> SigmondData:
        """Rename spatial dims (their coords follow); ``obj.rename_dims(operator='sink')``."""
        self._require_indexable()
        return _nd().rename_dims(self, {**(mapping or {}), **names})

    def diagonal(self, dim1: str, dim2: str, new_dim: str = "diagonal") -> SigmondData:
        """The diagonal over two equal-length dims, as one new dim (infos are kept)."""
        self._require_indexable()
        return _nd().diagonal(self, dim1, dim2, new_dim)

    def apply_matrix(
        self,
        matrix,
        dim: str,
        *,
        new_dim: str | None = None,
        coords=None,
        conj: bool = False,
    ) -> SigmondData:
        """Contract ``dim`` with ``matrix``: ``out[.., b, ..] = sum_a M[a, b] obj[.., a, ..]``.

        Applied identically to every sample. ``conj=True`` uses ``conj(M)``.
        See :func:`sampling_array.apply_matrix`.
        """
        self._require_indexable()
        return _nd().apply_matrix(self, matrix, dim, new_dim=new_dim, coords=coords, conj=conj)

    def _require_indexable(self):
        if not self._dims:
            raise TypeError(
                f"a single {type(self).__name__} cannot be indexed; use .data for the samples"
            )

    def as_xarray(self):
        """The data as an ``xarray.DataArray`` with the per-cell infos as an ``info`` coord."""
        return _nd().as_xarray(self)

    def to_collection(self):
        """Flatten to a SingleEnsembleCollection of single observables, row-major."""
        return _nd().to_collection(self)

    @property
    def obs(self):
        """Per-cell ObservableInfo attributes as a DataArray over ``S``."""
        return _nd()._ArrayAttributeAccessor(self, lambda s: s.observable_info)

    @property
    def samp(self):
        return _nd()._ArrayAttributeAccessor(self, lambda s: s.sampling_info)

    @property
    def val(self):
        return _nd()._ArrayAttributeAccessor(self, lambda s: s)

    def editor(self):
        """A mutable editor seeded from a copy of this object; ``.freeze()`` gives the result."""
        return _nd()._Editor(self)

    def _with_replaced_infos(self, new_infos: np.ndarray) -> SigmondData:
        """A new N-D object sharing ``data`` but with new per-cell infos."""
        return self._from_parts(
            self.data,
            self.observable_info,
            self.sampling_info,
            self.is_complex,
            self._dims,
            self._coords,
            new_infos,
        )

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def _check_compatible(self, others) -> bool:
        """
        Check that ``others`` can be combined elementwise with self.

        Different observables are allowed; compatibility is about the sample
        axis (sampling method and length for samplings; length and ensemble
        for bins).

        Raises:
            ValueError: listing the reasons for each incompatible instance.
        """
        msgs = []
        for other in others:
            reasons = self._analyzer.incompatibilities(self, other)
            if reasons:
                msgs.append(f" - {other!r}: {', '.join(reasons)}")
        if msgs:
            raise ValueError(f"Incompatible {self._analyzer.noun} found:\n" + "\n".join(msgs))
        return True

    # ------------------------------------------------------------------
    # Comparisons / hashing
    # ------------------------------------------------------------------

    def __lt__(self, other):
        return self.full_sample_value < (
            other.full_sample_value if isinstance(other, SigmondData) else other
        )

    def __le__(self, other):
        return self.full_sample_value <= (
            other.full_sample_value if isinstance(other, SigmondData) else other
        )

    def __gt__(self, other):
        return self.full_sample_value > (
            other.full_sample_value if isinstance(other, SigmondData) else other
        )

    def __ge__(self, other):
        return self.full_sample_value >= (
            other.full_sample_value if isinstance(other, SigmondData) else other
        )

    def __eq__(self, other):
        """Metadata equality (observable(s), sampling, dtype and labels), never data."""
        if not isinstance(other, SigmondData):
            return False
        return (self.sampling_info is None) == (other.sampling_info is None) and (
            self._identity_key() == other._identity_key()
        )

    def __hash__(self):
        return hash(self._identity_key())

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Length of axis 0 for N-D objects; the sample length for a single observable."""
        return self.shape[0] if self._dims else self._sample_length

    def __iter__(self):
        """Iterate along spatial axis 0, yielding sub-arrays or single observables."""
        self._require_indexable()
        for i in range(self.shape[0]):
            yield self[i]

    # ------------------------------------------------------------------
    # NumPy protocols
    # ------------------------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """NumPy array protocol — enables ``np.asarray(obj)`` (shape ``S + (N,)``)."""
        arr = self._as_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
        """Apply ``ufunc`` elementwise, broadcasting spatial dims by name."""
        if method != "__call__" or kwargs:
            return NotImplemented

        is_data = [isinstance(arg, SigmondData) for arg in inputs]
        operands = [arg for arg, flag in zip(inputs, is_data) if flag]

        if any((op.sampling_info is None) != (self.sampling_info is None) for op in operands):
            raise TypeError(
                "Cannot mix SigmondBins and SigmondSampling in a single operation. "
                "Call SigmondBins.resample(...) first to convert."
            )
        self._check_compatible([op for op in operands if op is not self])
        self._analyzer.check_ufunc(ufunc, is_data)

        if any(op._dims for op in operands):
            return _nd().array_ufunc(self, ufunc, inputs, operands)

        new_inputs = [arg._as_numpy() if flag else arg for arg, flag in zip(inputs, is_data)]
        result_data = ufunc(*new_inputs)
        if result_data is NotImplemented:
            return NotImplemented

        is_complex = (
            any(op.is_complex for op in operands)
            or any(np.iscomplexobj(arg) for arg in new_inputs)
            or np.iscomplexobj(result_data)
        )
        return self._new(result_data, _merged_observable_info(operands), is_complex)

    def __array_function__(self, func, types, args, kwargs):
        """``np.concatenate`` / ``np.stack`` of N-D objects keep labels; else plain numpy."""
        if _has_nd_operand(args):
            handler = _nd().HANDLED_NP_FUNCS.get(func)
            if handler is not None:
                return handler(*args, **kwargs)
        return func._implementation(*args, **kwargs)

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __neg__(self):
        return np.negative(self)

    def __pos__(self):
        return np.positive(self)

    def __abs__(self):
        return np.absolute(self)


def _copy_infos(infos: np.ndarray) -> np.ndarray:
    """Copy each ObservableInfo in an object array (the array copy alone is shallow)."""
    out = np.empty(infos.shape, dtype=object)
    for idx in np.ndindex(*infos.shape):
        out[idx] = infos[idx].copy()
    return out


def _has_nd_operand(args) -> bool:
    for arg in args:
        if isinstance(arg, SigmondData):
            if arg._dims:
                return True
        elif isinstance(arg, (list, tuple)):
            if any(isinstance(a, SigmondData) and a._dims for a in arg):
                return True
    return False
