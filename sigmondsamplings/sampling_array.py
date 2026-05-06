"""
SigmondSamplingArray: an N-D, label-indexed container of SigmondSamplings.

Backed by an xarray.DataArray of shape ``S + (R,)`` where ``S`` is the logical
shape of the array and ``R = num_resamplings + 1`` is the resampling axis
(always last). Per-cell ObservableInfo objects are stored eagerly in a parallel
``np.ndarray`` of dtype object with shape ``S``.

The public class is composition-based: it is *not* a subclass of ``xarray.DataArray``
or ``numpy.ndarray``. It implements ``__array__`` / ``__array_ufunc__`` so it
participates in NumPy's protocols, and exposes ``sel`` / ``isel`` / ``as_xarray``
for label-based access.

Status: SKELETON. Constructors, selection, conversion, basic ufunc handling,
and the editing context are wired up. The mixed-name ufunc policy and the
``shape_by`` pivot in ``from_collection`` follow the existing
``SigmondSampling.__array_ufunc__`` and ``ObservableCollection.filter/sort``
conventions.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .info import DEFAULT_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

RESAMPLING_DIM = "resampling"


# ---------------------------------------------------------------------------
# Axis metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AxisMeta:
    """Lightweight per-axis metadata. Coordinates also live on the xarray buffer."""

    name: str | None = None
    coords: tuple | None = None  # tuple of hashables aligned to axis length
    units: str | None = None


# ---------------------------------------------------------------------------
# ObservableInfo subclasses
# ---------------------------------------------------------------------------


def _default_element_name(parent: ArrayObsInfo, idx: tuple[int, ...]) -> str:
    return f"{parent.name}[{','.join(str(i) for i in idx)}]"


def _coerce_array_obs_info(
    obs_info: ArrayObsInfo | str,
    shape: tuple[int, ...],
    dim_names: Sequence[str] | None,
    ensemble_info: EnsembleInfo,
    name_fn: Callable | None = None,
) -> ArrayObsInfo:
    """Centralized ArrayObsInfo construction. Accepts either a string name or an existing info."""
    if isinstance(obs_info, ArrayObsInfo):
        if name_fn is not None:
            obs_info.name_fn = name_fn
        return obs_info
    return ArrayObsInfo(
        name=obs_info,
        shape=shape,
        dim_names=dim_names,
        name_fn=name_fn,
        ensemble_info=ensemble_info,
    )


def _default_element_infos(parent: ArrayObsInfo, shape: tuple[int, ...]) -> np.ndarray:
    """Build a shape-S object array of fresh ArrayElementObsInfo cells."""
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        out[idx] = ArrayElementObsInfo(parent=parent, array_index=idx)
    return out


def _promote_object_array(arr: np.ndarray) -> np.ndarray:
    """Try to promote a shape-S object ndarray to a numeric dtype; return unchanged on failure."""
    try:
        promoted = np.asarray(arr.tolist())
    except (ValueError, TypeError):
        return arr
    if promoted.dtype != object and promoted.shape == arr.shape:
        return promoted
    return arr


class ArrayObsInfo(ObservableInfo):
    """ObservableInfo describing a SigmondSamplingArray as a whole.

    Holds the array shape, dim names, and the rule used to derive per-cell
    names. The cell-naming rule is consulted at element-info construction time
    (eager), so renaming the parent does not propagate to existing children.
    """

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dim_names: Sequence[str] | None = None,
        name_fn: Callable[[ArrayObsInfo, tuple[int, ...]], str] | None = None,
        index: int = 0,
        op_type: str = "n",
        re_im: str = "re",
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        latex_str: str | None = None,
    ):
        super().__init__(name, index, op_type, re_im, ensemble_info, latex_str)
        self.shape = tuple(int(d) for d in shape)
        self.dim_names = (
            tuple(dim_names) if dim_names is not None else tuple(f"dim_{i}" for i in range(len(self.shape)))
        )
        if len(self.dim_names) != len(self.shape):
            raise ValueError(
                f"dim_names length ({len(self.dim_names)}) must match shape rank ({len(self.shape)})"
            )
        self.name_fn = name_fn or _default_element_name

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __eq__(self, other):
        if not isinstance(other, ArrayObsInfo):
            return False
        return (
            super().__eq__(other)
            and self.shape == other.shape
            and self.dim_names == other.dim_names
        )

    def __hash__(self):
        return hash((super().__hash__(), self.shape, self.dim_names))

    def __repr__(self):
        return (
            f"ArrayObsInfo(name='{self.name}', shape={self.shape}, "
            f"dims={self.dim_names}, ensemble='{self.ensemble_info}')"
        )


class ArrayElementObsInfo(ObservableInfo):
    """ObservableInfo for a single cell of a SigmondSamplingArray.

    Eagerly constructed: ``name`` is materialized at build time using the
    parent's ``name_fn``. Carries a back-pointer to the parent ArrayObsInfo
    and its position in the array.
    """

    def __init__(
        self,
        parent: ArrayObsInfo,
        array_index: tuple[int, ...],
        op_type: str | None = None,
        re_im: str | None = None,
        ensemble_info: EnsembleInfo | None = None,
        name: str | None = None,
        latex_str: str | None = None,
    ):
        cell_name = name if name is not None else parent.name_fn(parent, array_index)
        super().__init__(
            name=cell_name,
            index=0,
            op_type=op_type if op_type is not None else parent.op_type,
            re_im=re_im if re_im is not None else parent.re_im,
            ensemble_info=ensemble_info if ensemble_info is not None else parent.ensemble_info,
            latex_str=latex_str,
        )
        self.parent = parent
        self.array_index = tuple(array_index)

    def __repr__(self):
        return (
            f"ArrayElementObsInfo(name='{self.name}', index={self.array_index}, "
            f"parent='{self.parent.name}')"
        )


# ---------------------------------------------------------------------------
# Attribute accessor (parallel to ObservableCollection's, returns xr.DataArray)
# ---------------------------------------------------------------------------


class _ArrayAttributeAccessor:
    """Returns shape-S xarray.DataArray of attribute values across the array.

    Mirrors ``ObservableCollection.AttributeAccessor`` semantics but preserves
    array structure: numeric attributes return a numeric DataArray, mixed
    attributes return an object-dtype DataArray.
    """

    def __init__(
        self,
        array: SigmondSamplingArray,
        target_extractor: Callable[[SigmondSampling], Any],
    ):
        self._array = array
        self._extractor = target_extractor

    def _build_view_sampling(self, idx: tuple[int, ...]) -> SigmondSampling:
        """Build a SigmondSampling that *views* the underlying buffer at idx."""
        data_view = np.asarray(self._array._data.values)[idx]  # 1-D view (R,)
        return SigmondSampling(
            data=data_view,
            observable_info=self._array._element_infos[idx],
            sampling_info=self._array._sampling_info,
            is_complex=self._array._is_complex,
        )

    def _wrap_array(self, raw: np.ndarray) -> xr.DataArray:
        dims = self._array._data.dims[:-1]
        coords = {d: self._array._data.coords[d] for d in dims if d in self._array._data.coords}
        return xr.DataArray(raw, dims=dims, coords=coords)

    def _collect(self, name: str, call_args=None) -> xr.DataArray:
        """Collect attribute ``name`` across all cells; if call_args is given, invoke as a method."""
        shape = self._array.shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            target = self._extractor(self._build_view_sampling(idx))
            attr = getattr(target, name)
            out[idx] = attr(*call_args[0], **call_args[1]) if call_args is not None else attr
        return self._wrap_array(_promote_object_array(out))

    def __getattr__(self, name: str):
        shape = self._array.shape
        if not shape or any(d == 0 for d in shape):
            return self._wrap_array(np.array([]).reshape(shape))

        # Probe one cell to discover the attribute kind
        probe = getattr(self._extractor(self._build_view_sampling((0,) * len(shape))), name, None)
        if probe is None and not hasattr(
            self._extractor(self._build_view_sampling((0,) * len(shape))), name
        ):
            raise AttributeError(f"attribute '{name}' not found on accessor target")

        if callable(probe):
            return lambda *args, **kwargs: self._collect(name, call_args=(args, kwargs))
        return self._collect(name)

    def replace(self, **kwargs) -> SigmondSamplingArray:
        """Return a new SigmondSamplingArray with updated element-info attributes.

        Values may be scalars (broadcast), arrays of shape S, or callables
        invoked with the per-cell ObservableInfo. Only meaningful when the
        accessor targets ``observable_info``.
        """
        if not kwargs:
            raise ValueError("Must provide at least one attribute to replace")

        shape = self._array.shape
        new_infos = np.empty(shape, dtype=object)

        for idx in np.ndindex(*shape):
            current = self._array._element_infos[idx]
            target = self._extractor(self._build_view_sampling(idx))

            if target is current:
                new_target = current.copy()
            else:
                # Editing samp/val attributes via replace() is not supported on
                # an immutable array — those would require rewriting the buffer.
                raise NotImplementedError(
                    "replace() only supports observable_info attributes; "
                    "use editing() context for buffer-level changes."
                )

            for attr_name, value in kwargs.items():
                if callable(value):
                    new_value = value(new_target)
                elif isinstance(value, np.ndarray) and value.shape == shape:
                    new_value = value[idx]
                elif isinstance(value, (list, tuple)) and len(value) == shape[0] and len(shape) == 1:
                    new_value = value[idx[0]]
                else:
                    new_value = value
                setattr(new_target, attr_name, new_value)

            new_infos[idx] = new_target

        return self._array._with_replaced_infos(new_infos)


# ---------------------------------------------------------------------------
# Batch-edit context
# ---------------------------------------------------------------------------


class _Editor:
    """Mutable wrapper for batched edits. Yields a new SigmondSamplingArray on freeze()."""

    def __init__(self, array: SigmondSamplingArray):
        self._data = array._data.copy()
        self._element_infos = array._element_infos.copy()
        self._obs_info = array._obs_info  # ArrayObsInfo treated as immutable; rebind to mutate
        self._sampling_info = array._sampling_info
        self._is_complex = array._is_complex
        self._dim_meta = array._dim_meta

    def set_data(self, idx, value: np.ndarray) -> None:
        self._data.values[idx] = value

    def set_dim_meta(self, axis: int, **kwargs) -> None:
        new_meta = list(self._dim_meta)
        existing = new_meta[axis]
        new_meta[axis] = AxisMeta(
            name=kwargs.get("name", existing.name),
            coords=kwargs.get("coords", existing.coords),
            units=kwargs.get("units", existing.units),
        )
        self._dim_meta = tuple(new_meta)

    def replace_obs(self, **kwargs) -> None:
        """In-place attribute update on each element ObservableInfo."""
        shape = self._data.shape[:-1]
        for idx in np.ndindex(*shape):
            info = self._element_infos[idx]
            for k, v in kwargs.items():
                value = v(info) if callable(v) else v
                setattr(info, k, value)

    def freeze(self) -> SigmondSamplingArray:
        return SigmondSamplingArray(
            data=self._data,
            element_infos=self._element_infos,
            obs_info=self._obs_info,
            sampling_info=self._sampling_info,
            is_complex=self._is_complex,
            dim_meta=self._dim_meta,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SigmondSamplingArray:
    """Immutable N-D container of SigmondSamplings backed by xarray."""

    __slots__ = (
        "_data",
        "_element_infos",
        "_obs_info",
        "_sampling_info",
        "_is_complex",
        "_dim_meta",
    )

    def __init__(
        self,
        data: xr.DataArray,
        element_infos: np.ndarray,
        obs_info: ArrayObsInfo,
        sampling_info: SamplingInfo,
        is_complex: bool = False,
        dim_meta: tuple[AxisMeta, ...] | None = None,
    ):
        """Trusted constructor. Public users should prefer the from_* classmethods."""
        if not isinstance(data, xr.DataArray):
            raise TypeError("data must be an xarray.DataArray")
        if data.dims[-1] != RESAMPLING_DIM:
            raise ValueError(f"Last dim of data must be '{RESAMPLING_DIM}', got {data.dims[-1]!r}")
        if element_infos.shape != data.shape[:-1]:
            raise ValueError(
                f"element_infos shape {element_infos.shape} must match data spatial shape {data.shape[:-1]}"
            )

        self._data = data
        self._element_infos = element_infos
        self._obs_info = obs_info
        self._sampling_info = sampling_info
        self._is_complex = is_complex
        self._dim_meta = (
            dim_meta
            if dim_meta is not None
            else tuple(AxisMeta(name=d) for d in data.dims[:-1])
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        data: np.ndarray | xr.DataArray,
        obs_info: ArrayObsInfo | str,
        sampling_info: SamplingInfo,
        *,
        name_fn: Callable | None = None,
        dim_names: Sequence[str] | None = None,
        coords: dict[str, Sequence] | None = None,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        is_complex: bool = False,
    ) -> SigmondSamplingArray:
        """Wrap a raw buffer of shape ``S + (R,)`` in a SigmondSamplingArray.

        If ``data`` is an ndarray, it is wrapped in an xarray.DataArray with the
        given ``dim_names`` (resampling axis appended automatically).
        """
        if isinstance(data, xr.DataArray):
            if data.dims[-1] != RESAMPLING_DIM:
                data = data.rename({data.dims[-1]: RESAMPLING_DIM})
            xdata = data
        else:
            arr = np.asarray(data)
            if arr.ndim < 1:
                raise ValueError("data must have at least one dimension (resampling)")
            spatial_rank = arr.ndim - 1
            if dim_names is None:
                dim_names = tuple(f"dim_{i}" for i in range(spatial_rank))
            if len(dim_names) != spatial_rank:
                raise ValueError(
                    f"dim_names length ({len(dim_names)}) must match spatial rank ({spatial_rank})"
                )
            full_dims = tuple(dim_names) + (RESAMPLING_DIM,)
            xdata = xr.DataArray(
                arr.astype(complex if is_complex else float),
                dims=full_dims,
                coords=coords,
            )

        spatial_shape = xdata.shape[:-1]
        obs_info = _coerce_array_obs_info(
            obs_info, spatial_shape, tuple(xdata.dims[:-1]), ensemble_info, name_fn
        )
        return cls(
            data=xdata,
            element_infos=_default_element_infos(obs_info, spatial_shape),
            obs_info=obs_info,
            sampling_info=sampling_info,
            is_complex=is_complex,
        )

    @classmethod
    def from_samplings(
        cls,
        nested: Iterable,
        *,
        obs_info: ArrayObsInfo | str = "array",
        dim_names: Sequence[str] | None = None,
        coords: dict[str, Sequence] | None = None,
    ) -> SigmondSamplingArray:
        """Build from a nested iterable of SigmondSamplings.

        Walks the nesting to determine shape, then validates that all leaves
        share the same ``sampling_info`` and resampling length.
        """
        flat: list[SigmondSampling] = []
        shape = _walk_nested(nested, flat)

        if not flat:
            raise ValueError("from_samplings requires at least one SigmondSampling")

        first = flat[0]
        for s in flat[1:]:
            if s.sampling_info != first.sampling_info:
                raise ValueError("All samplings must share sampling_info")
            if len(s.data) != len(first.data):
                raise ValueError("All samplings must have the same data length")

        is_complex = any(s.is_complex for s in flat)
        dtype = complex if is_complex else float
        buf = np.empty(shape + (len(first.data),), dtype=dtype)
        elem_infos = np.empty(shape, dtype=object)

        for idx, samp in zip(np.ndindex(*shape), flat):
            buf[idx] = samp.data
            elem_infos[idx] = samp.observable_info  # preserve original infos

        obs_info = _coerce_array_obs_info(
            obs_info, shape, dim_names, first.observable_info.ensemble_info
        )
        dim_names = obs_info.dim_names
        full_dims = tuple(dim_names) + (RESAMPLING_DIM,)
        xdata = xr.DataArray(buf, dims=full_dims, coords=coords)

        return cls(
            data=xdata,
            element_infos=elem_infos,
            obs_info=obs_info,
            sampling_info=first.sampling_info,
            is_complex=is_complex,
        )

    @classmethod
    def from_collection(
        cls,
        source,
        shape_by: Sequence[str],
        *,
        filter: dict | None = None,
        sort: Sequence[str] | None = None,
        sentinel: SigmondSampling | float | complex | None = None,
        obs_info: ArrayObsInfo | str = "array",
    ) -> SigmondSamplingArray:
        """Build from a 1-D iterable / ObservableCollection by pivoting attributes onto axes.

        Each attribute in ``shape_by`` becomes one axis; the axis coords are
        ``unique(values)`` for that attribute (ordered as in the collection).

        If the cartesian product is not fully covered, missing cells are filled
        with ``sentinel``. ``sentinel`` may be a SigmondSampling (whose data is
        used directly), a scalar (broadcast across the resampling axis), or
        ``None`` (zero-fill with a warning).
        """
        from .obervable_collection import ObservableCollection  # local import

        if isinstance(source, ObservableCollection):
            collection = source
        else:
            collection = ObservableCollection(list(source))

        if filter:
            collection = collection.filter(**filter)
        if sort:
            collection = collection.sort(key=list(sort))

        if not collection:
            raise ValueError("from_collection requires a non-empty collection")

        # Build coords for each axis from unique attribute values (insertion order).
        coords_per_axis: list[list] = []
        for attr in shape_by:
            seen = []
            seen_set = set()
            for samp in collection:
                v = getattr(samp.observable_info, attr)
                try:
                    if v not in seen_set:
                        seen_set.add(v)
                        seen.append(v)
                except TypeError:
                    if v not in seen:
                        seen.append(v)
            coords_per_axis.append(seen)

        shape = tuple(len(c) for c in coords_per_axis)
        index_lookup = [{v: i for i, v in enumerate(c)} for c in coords_per_axis]

        first = collection[0]
        sampling_info = first.sampling_info
        ensemble_info = first.observable_info.ensemble_info
        is_complex = any(s.is_complex for s in collection)
        R = len(first.data)
        dtype = complex if is_complex else float

        # Resolve sentinel data (warning deferred until we know if any cells are unfilled)
        if sentinel is None:
            sentinel_data = np.zeros(R, dtype=dtype)
            sentinel_was_default = True
        elif isinstance(sentinel, SigmondSampling):
            if len(sentinel.data) != R:
                raise ValueError("sentinel SigmondSampling has incompatible resampling length")
            sentinel_data = sentinel.data.astype(dtype)
            sentinel_was_default = False
        else:
            sentinel_data = np.full(R, sentinel, dtype=dtype)
            sentinel_was_default = False

        # Allocate buffer pre-filled with sentinel; mark which cells were set
        buf = np.broadcast_to(sentinel_data, shape + (R,)).astype(dtype, copy=True)
        filled = np.zeros(shape, dtype=bool)
        elem_infos = np.empty(shape, dtype=object)

        for samp in collection:
            idx = tuple(
                index_lookup[a][getattr(samp.observable_info, attr)]
                for a, attr in enumerate(shape_by)
            )
            if filled[idx]:
                raise ValueError(
                    f"Multiple samplings map to cell {idx} for shape_by={list(shape_by)}; "
                    f"add another attribute to disambiguate."
                )
            buf[idx] = samp.data
            elem_infos[idx] = samp.observable_info
            filled[idx] = True

        if sentinel_was_default and not filled.all():
            n_missing = int((~filled).sum())
            warnings.warn(
                f"from_collection: {n_missing} cell(s) not specified by inputs; "
                f"zero-filled. Pass `sentinel=` to silence.",
                stacklevel=2,
            )

        # Fill placeholder ArrayElementObsInfo for unfilled cells
        if isinstance(obs_info, str):
            obs_info = ArrayObsInfo(
                name=obs_info,
                shape=shape,
                dim_names=tuple(shape_by),
                ensemble_info=ensemble_info,
            )
        for idx in np.ndindex(*shape):
            if not filled[idx]:
                elem_infos[idx] = ArrayElementObsInfo(parent=obs_info, array_index=idx)

        full_dims = tuple(shape_by) + (RESAMPLING_DIM,)
        xdata = xr.DataArray(
            buf,
            dims=full_dims,
            coords={attr: coords_per_axis[a] for a, attr in enumerate(shape_by)},
        )

        return cls(
            data=xdata,
            element_infos=elem_infos,
            obs_info=obs_info,
            sampling_info=sampling_info,
            is_complex=is_complex,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape[:-1])

    @property
    def ndim(self) -> int:
        return self._data.ndim - 1

    @property
    def num_resamplings(self) -> int:
        return self._data.shape[-1] - 1

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self._data.dims[:-1])

    @property
    def coords(self):
        return self._data.coords

    @property
    def observable_info(self) -> ArrayObsInfo:
        return self._obs_info

    @property
    def sampling_info(self) -> SamplingInfo:
        return self._sampling_info

    @property
    def ensemble_info(self) -> EnsembleInfo:
        return self._obs_info.ensemble_info

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @property
    def dim_meta(self) -> tuple[AxisMeta, ...]:
        return self._dim_meta

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def obs(self) -> _ArrayAttributeAccessor:
        return _ArrayAttributeAccessor(self, lambda s: s.observable_info)

    @property
    def samp(self) -> _ArrayAttributeAccessor:
        return _ArrayAttributeAccessor(self, lambda s: s.sampling_info)

    @property
    def val(self) -> _ArrayAttributeAccessor:
        return _ArrayAttributeAccessor(self, lambda s: s)

    # ------------------------------------------------------------------
    # Per-cell statistics — return DataArray of shape S
    # ------------------------------------------------------------------

    @property
    def full_sample_value(self) -> xr.DataArray:
        return self._data.isel({RESAMPLING_DIM: 0})

    @property
    def resampled_values(self) -> xr.DataArray:
        return self._data.isel({RESAMPLING_DIM: slice(1, None)})

    @property
    def mean(self) -> xr.DataArray:
        return self.resampled_values.mean(dim=RESAMPLING_DIM)

    @property
    def error(self) -> xr.DataArray:
        std = self.resampled_values.std(dim=RESAMPLING_DIM, ddof=1)
        if self._sampling_info.method == "jackknife":
            n = self._data.shape[-1] - 1
            return std * np.sqrt(n - 1)
        return std

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def sel(self, **kwargs) -> SigmondSamplingArray | SigmondSampling:
        """Label-based selection, delegating to xarray.

        If selection collapses all spatial dims, returns a SigmondSampling.
        Otherwise returns a new SigmondSamplingArray.
        """
        new_data = self._data.sel(**kwargs)
        # Translate to positional indices for element_infos
        indexers = {
            d: self._data.indexes[d].get_loc(kwargs[d]) if d in kwargs else slice(None)
            for d in self._data.dims[:-1]
        }
        # Build a tuple of indexers in spatial-dim order
        idx = tuple(indexers[d] for d in self._data.dims[:-1])
        new_infos = self._element_infos[idx]
        return self._from_selected(new_data, new_infos)

    def isel(self, **kwargs) -> SigmondSamplingArray | SigmondSampling:
        """Integer-based selection, delegating to xarray."""
        new_data = self._data.isel(**kwargs)
        idx = tuple(
            kwargs[d] if d in kwargs else slice(None) for d in self._data.dims[:-1]
        )
        new_infos = self._element_infos[idx]
        return self._from_selected(new_data, new_infos)

    def _from_selected(
        self, new_data: xr.DataArray, new_infos
    ) -> SigmondSamplingArray | SigmondSampling:
        # Selection collapsed everything → return a SigmondSampling
        if new_data.ndim == 1:
            info = new_infos if not isinstance(new_infos, np.ndarray) else new_infos.item()
            return SigmondSampling(
                data=np.asarray(new_data.values),
                observable_info=info,
                sampling_info=self._sampling_info,
                is_complex=self._is_complex,
            )

        # Rebuild ArrayObsInfo for the reduced shape
        spatial_dims = tuple(new_data.dims[:-1])
        spatial_shape = tuple(new_data.shape[:-1])
        new_obs = ArrayObsInfo(
            name=self._obs_info.name,
            shape=spatial_shape,
            dim_names=spatial_dims,
            name_fn=self._obs_info.name_fn,
            ensemble_info=self._obs_info.ensemble_info,
        )

        if not isinstance(new_infos, np.ndarray):
            new_infos = np.array(new_infos, dtype=object)
        if new_infos.shape != spatial_shape:
            new_infos = new_infos.reshape(spatial_shape)

        return SigmondSamplingArray(
            data=new_data,
            element_infos=new_infos,
            obs_info=new_obs,
            sampling_info=self._sampling_info,
            is_complex=self._is_complex,
        )

    def __getitem__(self, key) -> SigmondSamplingArray | SigmondSampling:
        """Positional indexing. Tuples are interpreted positionally over spatial dims."""
        if not isinstance(key, tuple):
            key = (key,)
        # Forbid touching the resampling axis directly via __getitem__
        full_key = key + (slice(None),)
        new_data = self._data.values[full_key]
        new_infos = self._element_infos[key]

        # Determine remaining dims
        remaining_dims = []
        for d, k in zip(self._data.dims[:-1], key):
            if isinstance(k, slice) or isinstance(k, (list, np.ndarray)):
                remaining_dims.append(d)

        if new_data.ndim == 1:
            info = new_infos if not isinstance(new_infos, np.ndarray) else new_infos.item()
            return SigmondSampling(
                data=new_data,
                observable_info=info,
                sampling_info=self._sampling_info,
                is_complex=self._is_complex,
            )

        new_xdata = xr.DataArray(
            new_data, dims=tuple(remaining_dims) + (RESAMPLING_DIM,)
        )
        return self._from_selected(new_xdata, np.asarray(new_infos, dtype=object))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_collection(self):
        """Flatten to a SingleEnsembleCollection in row-major order.

        The returned SigmondSamplings are *views* into ``_data`` (the buffer
        is treated as immutable, so this is safe).
        """
        from .ensemble_collection import SingleEnsembleCollection

        samplings = []
        for idx in np.ndindex(*self.shape):
            data_view = np.asarray(self._data.values)[idx]
            samplings.append(
                SigmondSampling(
                    data=data_view,
                    observable_info=self._element_infos[idx],
                    sampling_info=self._sampling_info,
                    is_complex=self._is_complex,
                )
            )
        return SingleEnsembleCollection(samplings)

    def as_xarray(self) -> xr.DataArray:
        """Return the underlying xarray.DataArray (no copy)."""
        return self._data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = np.asarray(self._data.values)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    # ------------------------------------------------------------------
    # NumPy ufunc protocol
    # ------------------------------------------------------------------

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            return NotImplemented

        arrays = {arg for arg in inputs if isinstance(arg, SigmondSamplingArray)}
        # Compatibility check on sampling_info and resampling length
        ref = self
        for a in arrays - {self}:
            if a._sampling_info != ref._sampling_info:
                raise ValueError("Incompatible sampling_info between SigmondSamplingArrays")
            if a._data.shape[-1] != ref._data.shape[-1]:
                raise ValueError("Incompatible resampling length between SigmondSamplingArrays")

        # Unwrap to xarray DataArrays so xarray handles broadcasting/alignment.
        # SigmondSampling inputs are wrapped into a 1-D xr.DataArray on the
        # resampling axis so they broadcast cleanly.
        new_inputs = []
        for arg in inputs:
            if isinstance(arg, SigmondSamplingArray):
                new_inputs.append(arg._data)
            elif isinstance(arg, SigmondSampling):
                new_inputs.append(xr.DataArray(arg.data, dims=(RESAMPLING_DIM,)))
            else:
                new_inputs.append(arg)

        result = ufunc(*new_inputs)
        if result is NotImplemented:
            return NotImplemented

        is_complex = (
            any(a._is_complex for a in arrays)
            or np.iscomplexobj(np.asarray(result))
        )

        if not isinstance(result, xr.DataArray):
            result = xr.DataArray(result, dims=ref._data.dims, coords=ref._data.coords)

        # Mixed-name policy mirrors SigmondSampling.__array_ufunc__: keep the
        # shared obs_info if all inputs match; else synthesize a mixed one.
        result_shape = tuple(result.shape[:-1])
        first = next(iter(arrays))._obs_info
        if all(a._obs_info == first for a in arrays):
            new_obs = first
            new_infos = (
                ref._element_infos
                if ref._element_infos.shape == result_shape
                else _default_element_infos(new_obs, result_shape)
            )
        else:
            same_ens = all(a._obs_info.ensemble_info == first.ensemble_info for a in arrays)
            new_obs = ArrayObsInfo(
                name="mixed_operation",
                shape=result_shape,
                dim_names=tuple(result.dims[:-1]),
                ensemble_info=first.ensemble_info if same_ens else DEFAULT_ENSEMBLE,
            )
            new_infos = _default_element_infos(new_obs, result_shape)

        return SigmondSamplingArray(
            data=result,
            element_infos=new_infos,
            obs_info=new_obs,
            sampling_info=ref._sampling_info,
            is_complex=is_complex,
            dim_meta=ref._dim_meta if new_infos.shape == ref._element_infos.shape else None,
        )

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def editor(self) -> _Editor:
        """Return a mutable _Editor seeded from this array; call ``.freeze()`` for the new array.

        Use this when applying several edits in succession to avoid copying on each step::

            ed = arr.editor()
            ed.replace_obs(name="renamed")
            ed.set_dim_meta(0, name="psq")
            new_arr = ed.freeze()
        """
        return _Editor(self)

    def copy(self) -> SigmondSamplingArray:
        new_infos = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            new_infos[idx] = self._element_infos[idx].copy()
        return SigmondSamplingArray(
            data=self._data.copy(),
            element_infos=new_infos,
            obs_info=self._obs_info,
            sampling_info=self._sampling_info,
            is_complex=self._is_complex,
            dim_meta=self._dim_meta,
        )

    def _with_replaced_infos(self, new_infos: np.ndarray) -> SigmondSamplingArray:
        """Internal: return a new array sharing _data but with new element infos."""
        return SigmondSamplingArray(
            data=self._data,
            element_infos=new_infos,
            obs_info=self._obs_info,
            sampling_info=self._sampling_info,
            is_complex=self._is_complex,
            dim_meta=self._dim_meta,
        )

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.shape[0] if self.ndim > 0 else 0

    def __iter__(self):
        """Iterate along axis 0, yielding sub-arrays or SigmondSamplings."""
        for i in range(self.shape[0]):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"SigmondSamplingArray(name='{self._obs_info.name}', "
            f"shape={self.shape}, dims={self.dims}, "
            f"R={self.num_resamplings + 1}, ensemble='{self.ensemble_info}')"
        )

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

    def __neg__(self):
        return np.negative(self)

    # ------------------------------------------------------------------
    # NumPy high-level function protocol
    # ------------------------------------------------------------------

    def __array_function__(self, func, types, args, kwargs):
        if func not in _HANDLED_NP_FUNCS:
            return NotImplemented
        if not all(issubclass(t, SigmondSamplingArray) for t in types):
            return NotImplemented
        return _HANDLED_NP_FUNCS[func](*args, **kwargs)

    @staticmethod
    def stack(
        arrays: Sequence[SigmondSamplingArray],
        axis: int = 0,
        dim_name: str = "stack",
        coords: Sequence | None = None,
    ) -> SigmondSamplingArray:
        """Like ``np.stack`` but with an explicit name (and optional coords) for the new dim."""
        return _stack_impl(arrays, axis=axis, dim_name=dim_name, coords=coords)


# ---------------------------------------------------------------------------
# np.* function registry
# ---------------------------------------------------------------------------


_HANDLED_NP_FUNCS: dict = {}


def _implements(np_func):
    def decorator(func):
        _HANDLED_NP_FUNCS[np_func] = func
        return func

    return decorator


def _check_concat_compatible(arrays: Sequence[SigmondSamplingArray]) -> SigmondSamplingArray:
    if not arrays:
        raise ValueError("Need at least one array to concatenate/stack")
    ref = arrays[0]
    for a in arrays[1:]:
        if a._sampling_info != ref._sampling_info:
            raise ValueError("Incompatible sampling_info")
        if a._data.shape[-1] != ref._data.shape[-1]:
            raise ValueError("Incompatible resampling length")
    return ref


def _combined_obs_info(
    arrays: Sequence[SigmondSamplingArray],
    new_shape: tuple[int, ...],
    new_dims: tuple[str, ...],
) -> tuple[ArrayObsInfo, bool]:
    """Apply mixed-name policy across a list of arrays. Returns (obs_info, was_shared)."""
    ref = arrays[0]._obs_info
    if all(a._obs_info == ref for a in arrays):
        return ref, True
    same_ens = all(a._obs_info.ensemble_info == ref.ensemble_info for a in arrays)
    return (
        ArrayObsInfo(
            name="mixed_operation",
            shape=new_shape,
            dim_names=new_dims,
            ensemble_info=ref.ensemble_info if same_ens else DEFAULT_ENSEMBLE,
        ),
        False,
    )


@_implements(np.concatenate)
def _concatenate(arrays, axis=0, **_):
    ref = _check_concat_compatible(arrays)
    dim = ref.dims[axis] if isinstance(axis, int) else axis
    pos_axis = ref.dims.index(dim)

    new_data = xr.concat([a._data for a in arrays], dim=dim, coords="minimal", compat="override")
    new_infos = np.concatenate([a._element_infos for a in arrays], axis=pos_axis)
    new_shape = new_infos.shape

    new_obs, shared = _combined_obs_info(arrays, new_shape, tuple(new_data.dims[:-1]))
    if not shared:
        new_infos = _default_element_infos(new_obs, new_shape)

    return SigmondSamplingArray(
        data=new_data,
        element_infos=new_infos,
        obs_info=new_obs,
        sampling_info=ref._sampling_info,
        is_complex=any(a._is_complex for a in arrays),
    )


def _stack_impl(
    arrays, axis: int, dim_name: str, coords: Sequence | None = None
) -> SigmondSamplingArray:
    ref = _check_concat_compatible(arrays)
    if any(a.shape != ref.shape for a in arrays):
        raise ValueError("All arrays must have identical shape to stack")
    if dim_name in ref.dims:
        raise ValueError(f"dim_name {dim_name!r} collides with existing dim")
    if coords is not None and len(coords) != len(arrays):
        raise ValueError("coords length must match number of arrays")

    expanded = [a._data.expand_dims({dim_name: 1}, axis=axis) for a in arrays]
    new_data = xr.concat(expanded, dim=dim_name, coords="minimal", compat="override")
    if coords is not None:
        new_data = new_data.assign_coords({dim_name: list(coords)})
    new_infos = np.stack([a._element_infos for a in arrays], axis=axis)
    new_shape = new_infos.shape

    new_obs, shared = _combined_obs_info(arrays, new_shape, tuple(new_data.dims[:-1]))
    if not shared:
        new_infos = _default_element_infos(new_obs, new_shape)

    return SigmondSamplingArray(
        data=new_data,
        element_infos=new_infos,
        obs_info=new_obs,
        sampling_info=ref._sampling_info,
        is_complex=any(a._is_complex for a in arrays),
    )


def _unique_dim_name(arrays: Sequence[SigmondSamplingArray], base: str = "stack") -> str:
    used = {d for a in arrays for d in a.dims}
    if base not in used:
        return base
    i = 0
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


@_implements(np.stack)
def _stack(arrays, axis=0, **_):
    return _stack_impl(arrays, axis=axis, dim_name=_unique_dim_name(arrays))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_nested(obj, flat_out: list) -> tuple[int, ...]:
    """Recursively walk a nested iterable of SigmondSamplings; return the shape."""
    if isinstance(obj, SigmondSampling):
        flat_out.append(obj)
        return ()
    seq = list(obj)
    if not seq:
        raise ValueError("Empty nested level in from_samplings")
    sub_shapes = [_walk_nested(x, flat_out) for x in seq]
    first = sub_shapes[0]
    if any(s != first for s in sub_shapes[1:]):
        raise ValueError(f"Ragged nesting in from_samplings: shapes {sub_shapes}")
    return (len(seq),) + first
