"""
N-D, label-indexed support for Sigmond data.

An N-D array of observables is an ordinary :class:`~sigmondsamplings.sampling.SigmondSampling`
or :class:`~sigmondsamplings.bins.SigmondBins` whose ``data`` has shape
``S + (N,)`` with ``S != ()`` (see :mod:`sigmondsamplings.data`); a single
observable is the ``S = ()`` case of the same classes. Statistics, arithmetic,
compatibility checks, transforms and resampling are written once, shape-generically,
on :class:`~sigmondsamplings.data.SigmondData`. This module adds what only N-D
objects need, and is imported lazily so single observables never import xarray:

- construction (:func:`from_data`, :func:`from_samplings`, :func:`from_collection`,
  :func:`stack`), reached through the classmethods of the same names;
- label-based selection, delegated to xarray with a cell-index coordinate
  carried along so the per-cell infos stay aligned with the data;
- arithmetic between N-D operands, broadcast by dim name with exact label
  alignment (mismatched coords raise instead of silently dropping cells);
- dim operations: :func:`transpose`, :func:`rename_dims`, :func:`diagonal`
  and the labeled linear map :func:`apply_matrix` (NumPy's ``matmul`` and
  ``einsum`` cannot be used directly: the sample axis is last);
- attribute accessors and the batch editor.

``SigmondDataArray``, ``SigmondSamplingArray`` and ``SigmondBinsArray`` are
aliases of ``SigmondData``, ``SigmondSampling`` and ``SigmondBins``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from .bins import SigmondBins
from .data import SigmondData, _copy_infos, _kind_cls
from .info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

SAMPLE_DIM = "sample"
RESAMPLING_DIM = SAMPLE_DIM  # backwards-compatible alias
INFO_COORD = "info"
_CELL_COORD = "_cell"

SigmondDataArray = SigmondData
SigmondSamplingArray = SigmondSampling
SigmondBinsArray = SigmondBins


# ---------------------------------------------------------------------------
# ObservableInfo subclasses
# ---------------------------------------------------------------------------


def _default_element_name(parent: ArrayObsInfo, idx: tuple[int, ...]) -> str:
    return f"{parent.name}[{','.join(str(i) for i in idx)}]"


class ArrayObsInfo(ObservableInfo):
    """ObservableInfo describing an N-D array as a whole.

    Shape and dims belong to the array itself, not to this info, so selections
    and broadcasts can reuse it unchanged. ``name_fn`` derives the names of
    placeholder cells; it is consulted at element-info construction time
    (eager), so renaming the parent does not propagate to existing children.
    """

    def __init__(
        self,
        name: str,
        name_fn: Callable[[ArrayObsInfo, tuple[int, ...]], str] | None = None,
        index: int = 0,
        op_type: str = "n",
        re_im: str = "re",
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        latex_str: str | None = None,
    ):
        super().__init__(name, index, op_type, re_im, ensemble_info, latex_str)
        self.name_fn = name_fn or _default_element_name

    def __eq__(self, other):
        return isinstance(other, ArrayObsInfo) and super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return f"ArrayObsInfo(name='{self.name}', ensemble='{self.ensemble_info}')"


class ArrayElementObsInfo(ObservableInfo):
    """Placeholder ObservableInfo for one cell of an N-D array.

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


def _coerce_array_obs_info(
    obs_info: ArrayObsInfo | str,
    ensemble_info: EnsembleInfo,
    name_fn: Callable | None = None,
) -> ArrayObsInfo:
    """Accept either a string name or an existing ArrayObsInfo."""
    if isinstance(obs_info, ArrayObsInfo):
        if name_fn is not None:
            obs_info.name_fn = name_fn
        return obs_info
    return ArrayObsInfo(name=obs_info, name_fn=name_fn, ensemble_info=ensemble_info)


def _default_element_infos(parent: ArrayObsInfo, shape: tuple[int, ...]) -> np.ndarray:
    """A shape-``S`` object array of fresh ArrayElementObsInfo cells."""
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


def _mixed_obs_info(obs_infos: Sequence[ObservableInfo]) -> ArrayObsInfo:
    first = obs_infos[0]
    same_ens = all(o.ensemble_info == first.ensemble_info for o in obs_infos)
    return ArrayObsInfo(
        name="mixed_operation",
        ensemble_info=first.ensemble_info if same_ens else INDEP_ENSEMBLE,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _resolve_cls(cls: type, sampling_info: SamplingInfo | None) -> type[SigmondData]:
    """Eager class for ``sampling_info``; ``cls`` must be ``SigmondData`` or that kind."""
    target = _kind_cls(sampling_info)
    if cls is SigmondData or issubclass(cls, target):
        return target
    raise TypeError(
        f"{cls.__name__} cannot hold {'bins' if sampling_info is None else 'samplings'}; "
        f"use {target.__name__} or SigmondData"
    )


def _build(
    cls: type[SigmondData],
    buf: np.ndarray,
    obs_info: ObservableInfo,
    sampling_info: SamplingInfo | None,
    is_complex: bool,
    dims: tuple[str, ...],
    coords: dict | None,
    infos: np.ndarray | None,
) -> SigmondData:
    """Build an N-D object, or a 0-d one (from the single info) when ``dims`` is empty."""
    if not dims:
        return cls._from_parts(buf, infos[()], sampling_info, is_complex)
    return cls._from_parts(buf, obs_info, sampling_info, is_complex, dims, coords, infos)


def from_data(
    cls,
    data: np.ndarray | xr.DataArray,
    obs_info: ArrayObsInfo | str,
    sampling_info: SamplingInfo | None,
    *,
    name_fn: Callable | None = None,
    dim_names: Sequence[str] | None = None,
    coords: dict[str, Sequence] | None = None,
    ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
    is_complex: bool = False,
    infos: np.ndarray | None = None,
) -> SigmondData:
    """Wrap a raw buffer of shape ``S + (N,)``.

    An ``xarray.DataArray`` keeps its dims and index coords; a ``sample`` dim is
    moved last, otherwise the last dim is taken as the sample axis. An ndarray
    gets ``dim_names`` (default ``dim_0, ...``) and ``coords``.
    ``sampling_info=None`` means the buffer holds raw bins.

    ``infos`` optionally gives each cell its own ``ObservableInfo`` (an object
    array of shape ``S``), e.g. the exact primary key of a derived observable.
    The objects are used as given, never mutated; without ``infos`` the cells
    get placeholder infos named by ``name_fn``.
    """
    target = _resolve_cls(cls, sampling_info)
    if isinstance(data, xr.DataArray):
        if SAMPLE_DIM in data.dims:
            data = data.transpose(..., SAMPLE_DIM)
        else:
            data = data.rename({data.dims[-1]: SAMPLE_DIM})
        dims = tuple(data.dims[:-1])
        coords = {d: data.indexes[d].values for d in dims if d in data.indexes}
        arr = np.asarray(data.values)
    else:
        arr = np.asarray(data)
        if arr.ndim < 1:
            raise ValueError("data must have at least one dimension (sample)")
        rank = arr.ndim - 1
        dims = tuple(dim_names) if dim_names is not None else tuple(f"dim_{i}" for i in range(rank))
        if len(dims) != rank:
            raise ValueError(f"dim_names length ({len(dims)}) must match spatial rank ({rank})")

    if infos is not None:
        infos = _checked_infos(infos, arr.shape[:-1])
    if not dims:
        if infos is not None:
            return target._from_parts(arr, infos[()], sampling_info, is_complex)
        info = obs_info if isinstance(obs_info, ObservableInfo) else None
        info = info or ObservableInfo(obs_info, ensemble_info=ensemble_info)
        return target._from_parts(arr, info, sampling_info, is_complex)

    array_info = _coerce_array_obs_info(obs_info, ensemble_info, name_fn)
    return target._from_parts(
        arr,
        array_info,
        sampling_info,
        is_complex,
        dims,
        coords,
        infos if infos is not None else _default_element_infos(array_info, arr.shape[:-1]),
    )


def _checked_infos(infos, shape: tuple[int, ...]) -> np.ndarray:
    """A fresh object array of the given per-cell infos, validated against ``shape``."""
    src = np.asarray(infos, dtype=object)
    if src.shape != shape:
        raise ValueError(f"infos must have shape {shape}, got {src.shape}")
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        if not isinstance(src[idx], ObservableInfo):
            raise TypeError(f"infos{list(idx)} is not an ObservableInfo: {src[idx]!r}")
        out[idx] = src[idx]
    return out


def _attr_getter(attr: str) -> Callable[[ObservableInfo], Any]:
    return lambda info: getattr(info, attr)


def _walk_nested(obj, flat_out: list) -> tuple[int, ...]:
    """Recursively walk a nested iterable of single observables; return the shape."""
    if isinstance(obj, SigmondData) and not obj.ndim:
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


def from_samplings(
    cls,
    nested: Iterable,
    *,
    obs_info: ArrayObsInfo | str = "array",
    dim_names: Sequence[str] | None = None,
    coords: dict[str, Sequence] | None = None,
) -> SigmondData:
    """Build from a nested iterable of single SigmondSamplings or SigmondBins.

    Walks the nesting to determine shape, then validates that all leaves share
    the same ``sampling_info`` and sample length. The cells keep their own
    ObservableInfo objects.
    """
    flat: list[SigmondData] = []
    shape = _walk_nested(nested, flat)
    if not flat:
        raise ValueError("from_samplings requires at least one SigmondSampling or SigmondBins")

    first = flat[0]
    for s in flat[1:]:
        if s.sampling_info != first.sampling_info:
            raise ValueError("All samplings must share sampling_info")
        if s.num_samples != first.num_samples:
            raise ValueError("All samplings must have the same data length")
    target = _resolve_cls(cls, first.sampling_info)

    is_complex = any(s.is_complex for s in flat)
    buf = np.empty(shape + (first.num_samples,), dtype=complex if is_complex else float)
    infos = np.empty(shape, dtype=object)
    for idx, samp in zip(np.ndindex(*shape), flat):
        buf[idx] = np.asarray(samp)
        infos[idx] = samp.observable_info

    dims = (
        tuple(dim_names) if dim_names is not None else tuple(f"dim_{i}" for i in range(len(shape)))
    )
    if len(dims) != len(shape):
        raise ValueError(f"dim_names length ({len(dims)}) must match nesting depth ({len(shape)})")
    array_info = _coerce_array_obs_info(obs_info, first.observable_info.ensemble_info)
    return target._from_parts(buf, array_info, first.sampling_info, is_complex, dims, coords, infos)


def from_collection(
    cls,
    source,
    shape_by: Sequence[str] | Mapping[str, str | Callable[[ObservableInfo], Any]],
    *,
    filter: dict | None = None,
    sort: Sequence[str] | None = None,
    sentinel: SigmondData | float | complex | None = None,
    missing: str = "fill",
    obs_info: ArrayObsInfo | str = "array",
) -> SigmondData:
    """Build from a 1-D iterable / ObservableCollection by pivoting attributes onto axes.

    Each entry of ``shape_by`` becomes one axis; the axis coords are the unique
    values (ordered as in the collection). A sequence names ``ObservableInfo``
    attributes. A mapping ``{dim: attr_or_function}`` names the dims itself and
    may give a key function, called with each item's ``ObservableInfo``.

    With ``missing="fill"`` (the default), cells not covered by the inputs are
    filled with ``sentinel``: a SigmondSampling/SigmondBins (whose data is used
    directly), a scalar (broadcast across the sample axis), or ``None``
    (zero-fill with a warning). ``missing="raise"`` rejects any uncovered cell
    instead. All items must share ``sampling_info``.
    """
    if missing not in ("fill", "raise"):
        raise ValueError(f"missing must be 'fill' or 'raise', got {missing!r}")
    if missing == "raise" and sentinel is not None:
        raise ValueError("sentinel is only used with missing='fill'")
    if isinstance(shape_by, Mapping):
        axes = [(dim, key if callable(key) else _attr_getter(key)) for dim, key in shape_by.items()]
    else:
        axes = [(attr, _attr_getter(attr)) for attr in shape_by]
    shape_by = [dim for dim, _ in axes]
    from .observable_collection import ObservableCollection

    collection = (
        source if isinstance(source, ObservableCollection) else ObservableCollection(list(source))
    )
    if filter:
        collection = collection.filter(**filter)
    if sort:
        collection = collection.sort(key=list(sort))
    if not collection:
        raise ValueError("from_collection requires a non-empty collection")

    # Coords for each axis from unique attribute values (insertion order).
    keys = [[get(samp.observable_info) for _, get in axes] for samp in collection]
    coords_per_axis: list[list] = []
    for a in range(len(axes)):
        seen = []
        seen_set = set()
        for row in keys:
            v = row[a]
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
    if any(s.sampling_info != sampling_info for s in collection):
        raise ValueError("All items must share sampling_info")
    target = _resolve_cls(cls, sampling_info)
    is_complex = any(s.is_complex for s in collection)
    n_samples = first.num_samples
    dtype = complex if is_complex else float

    if sentinel is None:
        sentinel_data = np.zeros(n_samples, dtype=dtype)
    elif isinstance(sentinel, SigmondData):
        if sentinel.num_samples != n_samples:
            raise ValueError("sentinel has incompatible sample length")
        sentinel_data = np.asarray(sentinel).astype(dtype)
    else:
        sentinel_data = np.full(n_samples, sentinel, dtype=dtype)

    buf = np.broadcast_to(sentinel_data, shape + (n_samples,)).astype(dtype, copy=True)
    filled = np.zeros(shape, dtype=bool)
    infos = np.empty(shape, dtype=object)

    for samp, row in zip(collection, keys):
        idx = tuple(index_lookup[a][v] for a, v in enumerate(row))
        if filled[idx]:
            raise ValueError(
                f"Multiple samplings map to cell {idx} for shape_by={list(shape_by)}; "
                f"add another attribute to disambiguate."
            )
        buf[idx] = np.asarray(samp)
        infos[idx] = samp.observable_info
        filled[idx] = True

    if missing == "raise" and not filled.all():
        holes = [
            {dim: coords_per_axis[a][i] for a, (dim, i) in enumerate(zip(shape_by, idx))}
            for idx in zip(*np.nonzero(~filled))
        ]
        shown = "\n".join(f"  {h}" for h in holes[:10])
        more = f"\n  ... and {len(holes) - 10} more" if len(holes) > 10 else ""
        raise ValueError(f"from_collection: {len(holes)} cell(s) missing:\n{shown}{more}")
    if sentinel is None and not filled.all():
        warnings.warn(
            f"from_collection: {int((~filled).sum())} cell(s) not specified by inputs; "
            f"zero-filled. Pass `sentinel=` to silence.",
            stacklevel=3,
        )

    array_info = _coerce_array_obs_info(obs_info, first.observable_info.ensemble_info)
    for idx in zip(*np.nonzero(~filled)):
        infos[idx] = ArrayElementObsInfo(parent=array_info, array_index=idx)

    return target._from_parts(
        buf,
        array_info,
        sampling_info,
        is_complex,
        tuple(shape_by),
        {attr: coords_per_axis[a] for a, attr in enumerate(shape_by)},
        infos,
    )


# ---------------------------------------------------------------------------
# xarray bridge
# ---------------------------------------------------------------------------


def _to_xr(obj: SigmondData, with_cells: bool = True) -> xr.DataArray:
    """View ``obj`` as a DataArray (no data copy).

    With ``with_cells``, a flat cell-index coordinate rides along so a selection
    result can look its infos up in ``obj.infos``. (Object coords would work too,
    but xarray copies their elements, which breaks info identity.)
    """
    coords: dict[str, Any] = dict(obj.coords)
    if with_cells:
        coords[_CELL_COORD] = (obj.dims, np.arange(int(np.prod(obj.shape))).reshape(obj.shape))
    return xr.DataArray(obj.data, dims=obj.dims + (SAMPLE_DIM,), coords=coords)


def _from_xr(template: SigmondData, x: xr.DataArray) -> SigmondData:
    """Rebuild a selection of ``template`` from a DataArray carrying the cell-index coord."""
    x = x.transpose(..., SAMPLE_DIM)
    dims = tuple(x.dims[:-1])
    cells = np.asarray(x.coords[_CELL_COORD].values).reshape(x.shape[:-1])
    infos = template.infos.reshape(-1)[cells]
    if not dims:
        infos = np.asarray(infos, dtype=object).reshape(())
    coords = {d: x.indexes[d].values for d in dims if d in x.indexes}
    for d in dims:
        if d not in coords:
            # Vectorized indexing (e.g. a diagonal) leaves the source dims' labels
            # as non-index coords along the new dim; keep them if they agree.
            along = [
                np.asarray(c.values)
                for name, c in x.coords.items()
                if name != _CELL_COORD and c.dims == (d,)
            ]
            if along and all(np.array_equal(along[0], v) for v in along[1:]):
                coords[d] = along[0]
    return _build(
        _kind_cls(template.sampling_info),
        np.asarray(x.values),
        template.observable_info,
        template.sampling_info,
        template.is_complex,
        dims,
        coords,
        infos,
    )


def wrap_stat(obj: SigmondData, values: np.ndarray) -> xr.DataArray:
    """A per-cell statistic as a DataArray over ``obj``'s spatial dims and coords."""
    return xr.DataArray(np.asarray(values), dims=obj.dims, coords=dict(obj.coords))


def as_xarray(obj: SigmondData) -> xr.DataArray:
    """The data with the per-cell infos as an ``info`` coord (elements are copies)."""
    return _to_xr(obj, with_cells=False).assign_coords({INFO_COORD: (obj.dims, obj.infos)})


def sel(obj: SigmondData, **kwargs) -> SigmondData:
    return _from_xr(obj, _to_xr(obj).sel(**kwargs))


def isel(obj: SigmondData, **kwargs) -> SigmondData:
    return _from_xr(obj, _to_xr(obj).isel(**kwargs))


def getitem(obj: SigmondData, key) -> SigmondData:
    """Positional indexing over the spatial dims, as orthogonal (per-dim) indexing."""
    if not isinstance(key, tuple):
        key = (key,)
    if key.count(Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if Ellipsis in key:
        i = key.index(Ellipsis)
        key = key[:i] + (slice(None),) * (obj.ndim - len(key) + 1) + key[i + 1 :]
    if len(key) > obj.ndim:
        raise IndexError(
            f"too many indices: {len(key)} given for {obj.ndim} spatial dims "
            "(the sample axis cannot be indexed; use .data)"
        )
    return isel(obj, **dict(zip(obj.dims, key)))


# ---------------------------------------------------------------------------
# Dim operations: transpose, rename, diagonal, linear maps
# ---------------------------------------------------------------------------


def _axis(obj: SigmondData, dim: str) -> int:
    try:
        return obj.dims.index(dim)
    except ValueError:
        raise KeyError(f"no dim {dim!r}; dims are {obj.dims}") from None


def transpose(obj: SigmondData, *dims: str) -> SigmondData:
    """Reorder the spatial dims; ``dims`` must be a permutation of ``obj.dims``."""
    if sorted(dims) != sorted(obj.dims):
        raise ValueError(f"transpose needs a permutation of {obj.dims}, got {dims}")
    perm = [_axis(obj, d) for d in dims]
    return _kind_cls(obj.sampling_info)._from_parts(
        np.transpose(obj.data, perm + [obj.ndim]),
        obj.observable_info,
        obj.sampling_info,
        obj.is_complex,
        tuple(dims),
        dict(obj.coords),
        np.transpose(obj.infos, perm),
    )


def rename_dims(obj: SigmondData, mapping: Mapping[str, str]) -> SigmondData:
    """Rename spatial dims; their coords move with them. Data and infos are shared."""
    for old in mapping:
        _axis(obj, old)
    dims = tuple(mapping.get(d, d) for d in obj.dims)
    if len(set(dims)) != len(dims) or SAMPLE_DIM in dims:
        raise ValueError(f"renaming {obj.dims} with {dict(mapping)} gives invalid dims {dims}")
    coords = {mapping.get(d, d): v for d, v in obj.coords.items()}
    return _kind_cls(obj.sampling_info)._from_parts(
        obj.data, obj.observable_info, obj.sampling_info, obj.is_complex, dims, coords, obj.infos
    )


def diagonal(obj: SigmondData, dim1: str, dim2: str, new_dim: str = "diagonal") -> SigmondData:
    """Cells with equal positions along ``dim1`` and ``dim2``, as one new dim.

    The dims must have equal length and, where both are labeled, equal
    coords; the new dim takes those coords. It sits where ``dim1`` was. The
    per-cell infos of the selected cells are kept.
    """
    a1, a2 = _axis(obj, dim1), _axis(obj, dim2)
    if a1 == a2:
        raise ValueError("diagonal needs two different dims")
    n = obj.shape[a1]
    if obj.shape[a2] != n:
        raise ValueError(f"dims {dim1!r} and {dim2!r} have different lengths")
    c1, c2 = obj.coords.get(dim1), obj.coords.get(dim2)
    if c1 is not None and c2 is not None and not np.array_equal(c1, c2):
        raise ValueError(f"dims {dim1!r} and {dim2!r} have different coords")
    labels = c1 if c1 is not None else c2
    rest = [d for d in obj.dims if d not in (dim1, dim2)]
    if new_dim in rest or new_dim == SAMPLE_DIM:
        raise ValueError(f"new dim {new_dim!r} collides with an existing dim")
    where = sum(1 for d in obj.dims[:a1] if d != dim2)
    idx = np.arange(n)

    def take(arr):
        moved = np.moveaxis(arr, [a1, a2], [0, 1])[idx, idx]
        return np.moveaxis(moved, 0, where)

    dims = tuple(rest[:where]) + (new_dim,) + tuple(rest[where:])
    coords = {d: v for d, v in obj.coords.items() if d in rest}
    if labels is not None:
        coords[new_dim] = labels
    return _kind_cls(obj.sampling_info)._from_parts(
        take(obj.data),
        obj.observable_info,
        obj.sampling_info,
        obj.is_complex,
        dims,
        coords,
        take(obj.infos),
    )


def apply_matrix(
    obj: SigmondData,
    matrix,
    dim: str,
    *,
    new_dim: str | None = None,
    coords: Sequence | None = None,
    conj: bool = False,
) -> SigmondData:
    """Apply a linear map over one labeled dim, identically to every sample.

    ``out[..., b, ...] = sum_a M[a, b] * obj[..., a, ...]`` with ``M = matrix``
    (or ``conj(matrix)`` if ``conj``), of shape ``(len(dim), n_out)``. The
    result keeps ``dim``'s position under ``new_dim`` (default: the same name)
    with ``coords`` as its labels. The map is linear, so it is exact for bins
    and for every resample of a sampling alike. Cells get placeholder infos.
    """
    axis = _axis(obj, dim)
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != obj.shape[axis]:
        raise ValueError(
            f"matrix must have shape ({obj.shape[axis]}, n_out) to act on {dim!r}, got {m.shape}"
        )
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix has non-finite entries")
    if conj:
        m = np.conj(m)
    new_dim = dim if new_dim is None else new_dim
    if new_dim != dim and (new_dim in obj.dims or new_dim == SAMPLE_DIM):
        raise ValueError(f"new dim {new_dim!r} collides with an existing dim")
    if coords is not None and len(coords) != m.shape[1]:
        raise ValueError(f"coords has {len(coords)} labels for {m.shape[1]} outputs")

    data = np.moveaxis(np.tensordot(obj.data, m, axes=([axis], [0])), -1, axis)
    is_complex = obj.is_complex or np.iscomplexobj(m)
    if not is_complex:
        data = data.real
    dims = obj.dims[:axis] + (new_dim,) + obj.dims[axis + 1 :]
    new_coords = {d: v for d, v in obj.coords.items() if d != dim}
    if coords is not None:
        new_coords[new_dim] = list(coords)
    array_info = obj.observable_info
    if not isinstance(array_info, ArrayObsInfo):
        array_info = ArrayObsInfo(array_info.name, ensemble_info=array_info.ensemble_info)
    return _kind_cls(obj.sampling_info)._from_parts(
        data,
        array_info,
        obj.sampling_info,
        is_complex,
        dims,
        new_coords,
        _default_element_infos(array_info, data.shape[:-1]),
    )


def to_collection(obj: SigmondData):
    """Flatten to a SingleEnsembleCollection of single observables, row-major.

    The cells are views into ``obj.data`` (treated as immutable).
    """
    from .ensemble_collection import SingleEnsembleCollection

    cls = _kind_cls(obj.sampling_info)
    infos = obj.infos
    return SingleEnsembleCollection(
        [
            cls._from_parts(obj.data[idx], infos[idx], obj.sampling_info, obj.is_complex)
            for idx in np.ndindex(*obj.shape)
        ]
    )


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def array_ufunc(self: SigmondData, ufunc: np.ufunc, inputs, operands) -> SigmondData:
    """Apply ``ufunc`` with at least one N-D operand, broadcasting spatial dims by name.

    Labels must align exactly. Cell infos survive only when every N-D operand
    has the same array info, dims and cells; otherwise placeholder cells are
    built (under a ``mixed_operation`` info when the array infos differ).
    """
    nd_ops = [op for op in operands if op.ndim]
    xin = [_to_xr(arg, with_cells=False) if isinstance(arg, SigmondData) else arg for arg in inputs]
    with xr.set_options(arithmetic_join="exact"):
        result = ufunc(*xin)
    if result is NotImplemented:
        return NotImplemented
    result = result.transpose(..., SAMPLE_DIM)

    dims = tuple(result.dims[:-1])
    shape = tuple(result.shape[:-1])
    coords = {d: result.indexes[d].values for d in dims if d in result.indexes}

    first = nd_ops[0]
    if all(op.observable_info == first.observable_info for op in nd_ops):
        obs_info = first.observable_info
        same_cells = all(
            op.dims == dims
            and (op._infos is first._infos or np.array_equal(op._infos, first._infos))
            for op in nd_ops
        )
        infos = first._infos if same_cells else _default_element_infos(obs_info, shape)
    else:
        obs_info = _mixed_obs_info([op.observable_info for op in nd_ops])
        infos = _default_element_infos(obs_info, shape)

    is_complex = any(op.is_complex for op in operands) or np.iscomplexobj(result.values)
    return _kind_cls(self.sampling_info)._from_parts(
        result.values, obs_info, self.sampling_info, is_complex, dims, coords, infos
    )


# ---------------------------------------------------------------------------
# np.concatenate / np.stack
# ---------------------------------------------------------------------------


HANDLED_NP_FUNCS: dict = {}


def _implements(np_func):
    def decorator(func):
        HANDLED_NP_FUNCS[np_func] = func
        return func

    return decorator


def _check_concat_compatible(arrays: Sequence[SigmondData]) -> SigmondData:
    if not arrays:
        raise ValueError("Need at least one array to concatenate/stack")
    ref = arrays[0]
    for a in arrays[1:]:
        if a.sampling_info != ref.sampling_info:
            raise ValueError("Incompatible sampling_info")
        if a.num_samples != ref.num_samples:
            raise ValueError("Incompatible sample length")
    return ref


def _joined(arrays: Sequence[SigmondData], x: xr.DataArray, infos: np.ndarray) -> SigmondData:
    """Finish a concatenate/stack: apply the shared-or-mixed array-info policy."""
    ref = arrays[0]
    array_infos = [a.observable_info for a in arrays]
    if all(a.ndim for a in arrays) and all(o == array_infos[0] for o in array_infos):
        obs_info = array_infos[0]
    elif not any(a.ndim for a in arrays):
        # stacking single observables keeps their infos under a fresh array info
        obs_info = ArrayObsInfo("array", ensemble_info=_mixed_obs_info(array_infos).ensemble_info)
    else:
        obs_info = _mixed_obs_info(array_infos)
        infos = _default_element_infos(obs_info, infos.shape)
    dims = tuple(x.dims[:-1])
    return _kind_cls(ref.sampling_info)._from_parts(
        x.values,
        obs_info,
        ref.sampling_info,
        any(a.is_complex for a in arrays),
        dims,
        {d: x.indexes[d].values for d in dims if d in x.indexes},
        infos,
    )


@_implements(np.concatenate)
def _concatenate(arrays, axis=0, **_):
    ref = _check_concat_compatible(arrays)
    dim = ref.dims[axis] if isinstance(axis, int) else axis
    x = xr.concat(
        [_to_xr(a, with_cells=False) for a in arrays],
        dim=dim,
        join="exact",
        coords="minimal",
        compat="override",
    )
    infos = np.concatenate([a.infos for a in arrays], axis=ref.dims.index(dim))
    return _joined(arrays, x, infos)


def stack(arrays, axis: int = 0, dim_name: str = "stack", coords: Sequence | None = None):
    """Stack equal-shape objects (single observables included) along a new named dim."""
    ref = _check_concat_compatible(arrays)
    if any(a.shape != ref.shape or a.dims != ref.dims for a in arrays):
        raise ValueError("All arrays must have identical shape and dims to stack")
    if dim_name in ref.dims:
        raise ValueError(f"dim_name {dim_name!r} collides with existing dim")
    if coords is not None and len(coords) != len(arrays):
        raise ValueError("coords length must match number of arrays")

    expanded = [_to_xr(a, with_cells=False).expand_dims({dim_name: 1}, axis=axis) for a in arrays]
    x = xr.concat(expanded, dim=dim_name, join="exact", coords="minimal", compat="override")
    if coords is not None:
        x = x.assign_coords({dim_name: list(coords)})
    infos = np.stack([a.infos for a in arrays], axis=axis)
    return _joined(arrays, x, infos)


def _unique_dim_name(arrays: Sequence[SigmondData], base: str = "stack") -> str:
    used = {d for a in arrays for d in a.dims}
    if base not in used:
        return base
    i = 0
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


@_implements(np.stack)
def _stack(arrays, axis=0, **_):
    return stack(arrays, axis=axis, dim_name=_unique_dim_name(arrays))


# ---------------------------------------------------------------------------
# Attribute accessor (parallel to ObservableCollection's, returns xr.DataArray)
# ---------------------------------------------------------------------------

_VECTORIZED_STATS = frozenset({"full_sample_value", "mean", "error"})


class _ArrayAttributeAccessor:
    """Returns a shape-S xarray.DataArray of attribute values across the array.

    Mirrors ``ObservableCollection.AttributeAccessor`` semantics but preserves
    array structure: numeric attributes return a numeric DataArray, mixed
    attributes return an object-dtype DataArray.
    """

    def __init__(self, array: SigmondData, target_extractor: Callable[[SigmondData], Any]):
        self._array = array
        self._extractor = target_extractor

    def _cell(self, idx: tuple[int, ...]) -> SigmondData:
        """A single observable that *views* the underlying buffer at ``idx``."""
        a = self._array
        return _kind_cls(a.sampling_info)._from_parts(
            a.data[idx], a.infos[idx], a.sampling_info, a.is_complex
        )

    def _wrap_array(self, raw: np.ndarray) -> xr.DataArray:
        return xr.DataArray(raw, dims=self._array.dims, coords=dict(self._array.coords))

    def _collect(self, name: str, call_args=None) -> xr.DataArray:
        """Collect attribute ``name`` across all cells; if call_args is given, invoke as a method."""
        shape = self._array.shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            attr = getattr(self._extractor(self._cell(idx)), name)
            out[idx] = attr(*call_args[0], **call_args[1]) if call_args is not None else attr
        return self._wrap_array(_promote_object_array(out))

    def __getattr__(self, name: str):
        shape = self._array.shape
        if any(d == 0 for d in shape):
            return self._wrap_array(np.array([]).reshape(shape))

        probe_target = self._extractor(self._cell((0,) * len(shape)))
        if (
            name in _VECTORIZED_STATS
            and probe_target is not None
            and isinstance(probe_target, SigmondData)
        ):
            return self._wrap_array(np.asarray(getattr(self._array, name)))
        if not hasattr(probe_target, name):
            raise AttributeError(f"attribute '{name}' not found on accessor target")
        if callable(getattr(probe_target, name)):
            return lambda *args, **kwargs: self._collect(name, call_args=(args, kwargs))
        return self._collect(name)

    def replace(self, **kwargs) -> SigmondData:
        """Return a new object with updated per-cell ObservableInfo attributes.

        Values may be scalars (broadcast), arrays of shape S, or callables
        invoked with the per-cell ObservableInfo. Only meaningful when the
        accessor targets ``observable_info``.
        """
        if not kwargs:
            raise ValueError("Must provide at least one attribute to replace")

        a = self._array
        shape = a.shape
        infos = a.infos
        new_infos = np.empty(shape, dtype=object)

        for idx in np.ndindex(*shape):
            current = infos[idx]
            if self._extractor(self._cell(idx)) is not current:
                # Editing samp/val attributes would require rewriting the buffer.
                raise NotImplementedError(
                    "replace() only supports observable_info attributes; "
                    "use editor() for buffer-level changes."
                )
            new_target = current.copy()
            for attr_name, value in kwargs.items():
                if callable(value):
                    new_value = value(new_target)
                elif isinstance(value, np.ndarray) and value.shape == shape:
                    new_value = value[idx]
                elif (
                    isinstance(value, (list, tuple)) and len(shape) == 1 and len(value) == shape[0]
                ):
                    new_value = value[idx[0]]
                else:
                    new_value = value
                setattr(new_target, attr_name, new_value)
            new_infos[idx] = new_target

        if not a.ndim:
            return a.with_observable_info(new_infos[()])
        return a._with_replaced_infos(new_infos)


# ---------------------------------------------------------------------------
# Batch-edit context
# ---------------------------------------------------------------------------


class _Editor:
    """Mutable working copy for batched edits. Yields a new object on freeze().

    The data buffer and every per-cell ObservableInfo are copied up front, so
    edits never reach the object the editor was created from.
    """

    def __init__(self, array: SigmondData):
        self._template = array
        self._data = np.array(array.data, copy=True)
        self._infos = _copy_infos(array.infos)
        self._obs_info = array.observable_info
        self._coords = dict(array.coords)

    def set_data(self, idx, value: np.ndarray) -> None:
        self._data[idx] = value

    def set_coords(self, dim: str, values: Sequence) -> None:
        if dim not in self._template.dims:
            raise KeyError(f"unknown dim {dim!r}")
        self._coords[dim] = np.asarray(values)

    def replace_obs(self, **kwargs) -> None:
        """Attribute update on each (copied) per-cell ObservableInfo."""
        for idx in np.ndindex(*self._infos.shape):
            info = self._infos[idx]
            for k, v in kwargs.items():
                setattr(info, k, v(info) if callable(v) else v)

    def freeze(self) -> SigmondData:
        t = self._template
        return _build(
            _kind_cls(t.sampling_info),
            self._data,
            self._obs_info,
            t.sampling_info,
            t.is_complex,
            t.dims,
            self._coords,
            self._infos,
        )
