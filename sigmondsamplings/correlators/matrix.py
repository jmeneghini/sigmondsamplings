"""
Correlation matrices: assembly from an index, and the :class:`CorrelatorMatrix` wrapper.

Assembly contract
-----------------
**Key form.** The stored key flags used for a request are its *key form*:
``"plain"``, ``"HermMat"``, ``"SubVEV"`` or ``"HermMat SubVEV"``. One form is
used for the whole request, never a mix. A Hermitian request may read
``HermMat`` or unflagged keys; a non-Hermitian one only unflagged keys. A
connected (``subtract_vev``) request reads stored ``SubVEV`` keys, or derives
every cell from raw keys and VEVs. When exactly one allowed form is stored for
the requested operators it is used; when several are, the request raises and
names them, and ``key_form=`` chooses. (A Sigmond file normally holds one
form: Sigmond averages ``HermMat`` pairs on the raw data before binning.)

**Times.** Explicit ``times`` must all be complete. ``times=None`` takes every
time at which all cells are stored, and warns about observed times left out.

**Hermitian completion** (``hermitian=True``): at each time,
``C[i,j] = (C_ij + conj(C_ji)) / 2`` sample by sample when both directions are
stored, the stored one (or the conjugate of the flipped one) when only one
is, and a missing-cell error otherwise. Diagonals keep their recorded complex
value; see :meth:`CorrelatorMatrix.check_hermitian`. A non-Hermitian request
reads each ordered direction independently.

**VEV subtraction.** ``C_conn[i,j] = C[i,j] - v[i] * conj(v[j])`` at every
sample, including the full estimate. Bins are first stacked with the VEV
bins and resampled jointly (one draw plan for all), then subtracted, which
keeps the nonlinear full-sample estimator and the correlations between cells.

**Basis change.** ``O_new[a] = sum_i conj(T[i,a]) O_old[i]`` gives
``C_new = T^H C T`` and ``v_new = T^H v``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..bins import _block_resample
from ..data import SigmondData, _kind_cls
from ..info import EnsembleInfo, SamplingInfo
from ..lazy import materialize, stack_data
from ..sampling_array import ArrayObsInfo
from .keys import CorrelatorAtTimeInfo, CorrelatorInfo, CorrelatorMatrixInfo, KeyForm, VEVInfo
from .observables import CorrelatorObsInfo, VEVObsInfo
from .operators import GenIrrepOperator, Operator, _int

__all__ = [
    "CorrelatorMatrix",
    "HermiticityCheck",
    "Transformation",
    "assemble",
    "assemble_correlator",
    "available_bases",
    "coverage",
]

#: Up to this many operators, :func:`available_bases` lists every maximal available
#: sub-basis; above it, it returns one found greedily.
EXACT_BASIS_LIMIT = 16

TIME, SINK, SOURCE, OPERATOR = "time", "sink", "source", "operator"


@dataclass(frozen=True)
class Transformation:
    """Coefficients of a basis change: ``O_new[a] = sum_i conj(T[i,a]) O_source[i]``.

    GI labels do not encode these coefficients; they live only here.
    """

    source_operators: tuple[Operator, ...]
    coefficients: np.ndarray


# ---------------------------------------------------------------------------
# Key forms
# ---------------------------------------------------------------------------


def _form_name(form: tuple[bool, bool]) -> str:
    herm, subvev = form
    return " ".join(f for f, on in (("HermMat", herm), ("SubVEV", subvev)) if on) or "plain"


def _parse_form(text: str) -> tuple[bool, bool]:
    tokens = set(str(text).split()) - {"plain"}
    if not tokens <= {"HermMat", "SubVEV"}:
        raise ValueError(
            f"key_form must be 'plain', 'HermMat', 'SubVEV' or 'HermMat SubVEV', got {text!r}"
        )
    return "HermMat" in tokens, "SubVEV" in tokens


def _allowed_forms(hermitian: bool, subtract_vev: bool) -> list[tuple[bool, bool]]:
    herm = (True, False) if hermitian else (False,)
    subvev = (True, False) if subtract_vev else (False,)
    return [(h, s) for s in subvev for h in herm]


# ---------------------------------------------------------------------------
# Cell infos
# ---------------------------------------------------------------------------


def _cell_infos(
    correlators: Sequence[CorrelatorInfo],
    times: Sequence[int],
    hermitian: bool,
    subtract_vev: bool,
    ensemble_info: EnsembleInfo,
) -> np.ndarray:
    """``(T, len(correlators))`` correlator infos; their names are generated only when needed."""
    infos = np.empty((len(times), len(correlators)), dtype=object)
    for t, time in enumerate(times):
        for c, corr in enumerate(correlators):
            key = CorrelatorAtTimeInfo._unchecked(corr, time, hermitian, subtract_vev)
            infos[t, c] = CorrelatorObsInfo(key=key, ensemble_info=ensemble_info)
    return infos


def _matrix_correlators(ops: Sequence[Operator]) -> list[CorrelatorInfo]:
    return [CorrelatorInfo(a, b) for a in ops for b in ops]


def _matrix_infos(ops, times, hermitian, subtract_vev, ens) -> np.ndarray:
    n = len(ops)
    infos = _cell_infos(_matrix_correlators(ops), times, hermitian, subtract_vev, ens)
    return infos.reshape(len(times), n, n)


def _vev_infos(operators: Sequence[Operator], ensemble_info: EnsembleInfo) -> np.ndarray:
    infos = np.empty(len(operators), dtype=object)
    for i, op in enumerate(operators):
        infos[i] = VEVObsInfo(key=VEVInfo(op), ensemble_info=ensemble_info)
    return infos


# ---------------------------------------------------------------------------
# Assembly over a list of cells
# ---------------------------------------------------------------------------


@dataclass
class _Assembled:
    buf: np.ndarray  # (T, n_cells, S)
    stored: np.ndarray  # (T, n_cells) bool: the (i, j) direction itself was stored
    vevs: np.ndarray | None  # (n_ops, S)
    sampling_info: SamplingInfo | None
    times: list[int]
    omitted: list[int]
    form: tuple[bool, bool]


class _Request:
    """The cells of one request and their stored series, per key form."""

    def __init__(self, index, ops, cells, hermitian: bool):
        self.index, self.ops, self.cells, self.hermitian = index, ops, cells, hermitian
        self._series = {}

    def series(self, i, j, form) -> dict[int, SigmondData]:
        key = (i, j, form)
        if key not in self._series:
            self._series[key] = self.index._lookup(self.ops[i], self.ops[j], *form)
        return self._series[key]

    def flipped(self, i, j, form) -> dict[int, SigmondData]:
        if self.hermitian and i != j:
            return self.series(j, i, form)
        return {}

    def cell_times(self, i, j, form) -> set[int]:
        direct, flipped = self.series(i, j, form), self.flipped(i, j, form)
        return {
            t
            for t in direct.keys() | flipped.keys()
            if all(_usable_record(series[t]) for series in (direct, flipped) if t in series)
        }

    def present(self, form, *, either_direction: bool = False) -> int:
        """How many cells have any stored time in ``form``.

        ``either_direction`` also counts cells stored only flipped, even for a
        non-Hermitian request (used to suggest other forms).
        """
        return sum(
            1
            for i, j in self.cells
            if self.series(i, j, form)
            or self.flipped(i, j, form)
            or (either_direction and self.series(j, i, form))
        )

    def coverage(self, form, times) -> str:
        lines = []
        for t in times:
            holes = [(i, j) for i, j in self.cells if t not in self.cell_times(i, j, form)]
            if holes:
                shown = ", ".join(
                    f"({self.ops[i].coordinate}, {self.ops[j].coordinate})" for i, j in holes[:3]
                )
                more = f" and {len(holes) - 3} more" if len(holes) > 3 else ""
                lines.append(f"  time {t}: missing {shown}{more}")
        return "\n".join(lines[:8]) + ("\n  ..." if len(lines) > 8 else "")


def _choose_form(req: _Request, hermitian, subtract_vev, key_form) -> tuple[bool, bool]:
    allowed = _allowed_forms(hermitian, subtract_vev)
    if key_form is not None:
        form = _parse_form(key_form)
        if form not in allowed:
            names = ", ".join(repr(_form_name(f)) for f in allowed)
            raise ValueError(f"key_form={key_form!r} is not allowed here; allowed: {names}")
        return form
    counts = {form: req.present(form) for form in allowed}
    present = [form for form, n in counts.items() if n]
    if len(present) == 1:
        return present[0]
    if not present:
        others = [
            f
            for f in _allowed_forms(True, True)
            if f not in allowed and req.present(f, either_direction=True)
        ]
        hint = ""
        if others:
            stored = ", ".join(repr(_form_name(f)) for f in others)
            hint = f"; stored forms are {stored} (set hermitian=/subtract_vev= to match)"
        raise ValueError(
            "no stored correlators for the requested operators in the allowed key forms "
            f"{', '.join(repr(_form_name(f)) for f in allowed)}{hint}"
        )
    detail = ", ".join(f"{_form_name(f)!r} ({counts[f]} cells)" for f in present)
    raise ValueError(
        f"several key forms are stored for these operators: {detail}; pass key_form= "
        "to choose one (forms are never mixed)"
    )


class IncompleteCorrelatorError(ValueError):
    """A request has missing cells at its requested times."""


def _choose_times(req: _Request, form, times) -> tuple[list[int], list[int]]:
    per_cell = [req.cell_times(i, j, form) for i, j in req.cells]
    complete = set.intersection(*per_cell)
    observed = set.union(*per_cell)
    label = _form_name(form)
    if times is not None:
        missing = [t for t in times if t not in complete]
        if missing:
            raise IncompleteCorrelatorError(
                f"not every requested cell is stored at times {missing} "
                f"(key form {label!r}):\n{req.coverage(form, missing)}"
            )
        return list(times), []
    if not complete:
        raise IncompleteCorrelatorError(
            f"no time has every requested cell (key form {label!r}):\n"
            f"{req.coverage(form, sorted(observed))}"
        )
    omitted = sorted(observed - complete)
    if omitted:
        warnings.warn(
            f"times {omitted} are stored for some cells only and were left out "
            f"(key form {label!r})",
            stacklevel=5,
        )
    return sorted(complete), omitted


def coverage(
    index,
    ops: Sequence[Operator],
    hermitian: bool,
    subtract_vev: bool,
    key_form: KeyForm | None,
) -> xr.DataArray:
    """Which cells of the ``ops`` matrix :func:`assemble` could read, per time.

    A boolean ``(time, sink, source)`` array over every time at which any cell
    is stored. The key form is chosen by the same rules as a matrix request, a
    Hermitian request counts a cell stored in either direction, and a connected
    request derived from raw data also needs both operators' VEVs. ``attrs``
    carry the ``operators`` (so a plot can label the axes without the index),
    the ``key_form`` and the flags.
    """
    ops = tuple(ops)
    n = len(ops)
    cells = [(i, j) for i in range(n) for j in range(n)]
    req = _Request(index, ops, cells, hermitian)
    form = _choose_form(req, hermitian, subtract_vev, key_form)
    per_cell = {(i, j): req.cell_times(i, j, form) for i, j in cells}
    times = sorted(
        set().union(*(set(req.series(i, j, form)) | set(req.flipped(i, j, form)) for i, j in cells))
    )
    has_vevs = np.ones(n, dtype=bool)
    if subtract_vev and not form[1]:
        has_vevs = np.array([_usable_record(index._vev(op)) for op in ops])
    available = np.zeros((len(times), n, n), dtype=bool)
    for t, time in enumerate(times):
        for i, j in cells:
            available[t, i, j] = (
                time in per_cell[i, j]
                and has_vevs[i]
                and has_vevs[j]
                and ops[i].backwards == ops[j].backwards
            )
    coords = [op.coordinate for op in ops]
    return xr.DataArray(
        available,
        dims=(TIME, SINK, SOURCE),
        coords={TIME: times, SINK: coords, SOURCE: coords},
        attrs={
            "operators": ops,
            "key_form": _form_name(form),
            "hermitian": hermitian,
            "subtract_vev": subtract_vev,
        },
    )


def available_bases(
    cov: xr.DataArray, times: Iterable[int] | None = None
) -> list[tuple[Operator, ...]]:
    """Maximal available sub-bases of a :func:`coverage` array, largest first.

    A sub-basis is available when every one of its cells is available at every
    one of ``times`` (default: every time in ``cov``, so ``matrix()`` of it
    keeps them all). Up to :data:`EXACT_BASIS_LIMIT` operators every maximal
    one is listed (ties in size ordered by basis position); above it, the one
    found by repeatedly dropping the operator with the most holes. Operators
    keep their basis order within each sub-basis.
    """
    ops = tuple(cov.attrs["operators"])
    if times is None:
        chosen = cov.values
    else:
        times = _unique_times(times)
        stored = set(int(t) for t in cov.coords[TIME].values)
        if any(t not in stored for t in times):
            return []
        chosen = cov.sel({TIME: times}).values
    ok = chosen.all(axis=0) if chosen.shape[0] else np.zeros(chosen.shape[1:], dtype=bool)
    # Operators whose own diagonal is available, and which pairs are both available.
    nodes = [i for i in range(len(ops)) if ok[i, i]]
    linked = ok & ok.T
    if len(nodes) <= EXACT_BASIS_LIMIT:
        cliques = _maximal_cliques(nodes, linked)
    else:
        cliques = [_greedy_clique(nodes, linked)]
    cliques = [sorted(c) for c in cliques if c]
    cliques.sort(key=lambda c: (-len(c), c))
    return [tuple(ops[i] for i in c) for c in cliques]


def _maximal_cliques(nodes: list[int], linked: np.ndarray) -> list[set[int]]:
    """Bron-Kerbosch with pivoting: every maximal set of mutually linked nodes."""
    out: list[set[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            out.append(r)
            return
        pivot = max(p | x, key=lambda u: sum(linked[u, v] for v in p if v != u))
        # P minus the pivot's neighbours; the pivot is not its own neighbour.
        for v in [v for v in p if v == pivot or not linked[pivot, v]]:
            neighbours = {u for u in nodes if u != v and linked[v, u]}
            expand(r | {v}, p & neighbours, x & neighbours)
            p = p - {v}
            x = x | {v}

    expand(set(), set(nodes), set())
    return out


def _greedy_clique(nodes: list[int], linked: np.ndarray) -> set[int]:
    """Drop the node with the most missing links until the rest are all linked."""
    keep = list(nodes)
    while keep:
        holes = {i: sum(not linked[i, j] for j in keep if j != i) for i in keep}
        worst = max(keep, key=lambda i: (holes[i], i))
        if not holes[worst]:
            break
        keep.remove(worst)
    return set(keep)


def _check_sampling_plan(index, sampling_info, needs_resample: bool) -> None:
    if index.is_bins:
        if needs_resample and sampling_info is None:
            raise ValueError(
                "connected correlators from bins need an explicit sampling_info: the raw "
                "correlators and VEVs are resampled jointly before subtracting"
            )
    elif sampling_info is not None and sampling_info != index.sampling_info:
        raise ValueError(
            f"the data are samplings with {index.sampling_info!r}; cannot use {sampling_info!r}"
        )


def _unique_times(times: Iterable[int]) -> list[int]:
    out = []
    for t in times:
        try:
            t = _int(t, "time")
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"times must be nonnegative integers, got {t!r}") from None
        if t < 0:
            raise ValueError(f"times must be nonnegative integers, got {t!r}")
        if int(t) in out:
            raise ValueError(f"time {t} requested twice")
        out.append(int(t))
    if not out:
        raise ValueError("no times requested")
    return out


def _usable_record(obs) -> bool:
    return obs is not None and (obs.is_complex or obs.observable_info.re_im != "im")


def _check_not_lone_imaginary(obs_list) -> None:
    for obs in obs_list:
        if not _usable_record(obs):
            raise ValueError(f"only the imaginary part of {obs.observable_info.name} is stored")


def _assemble(
    index,
    ops: Sequence[Operator],
    cells: Sequence[tuple[int, int]],
    hermitian: bool,
    subtract_vev: bool,
    times,
    key_form,
    sampling_info,
    include_vevs: bool,
) -> _Assembled:
    """Select the key form and times, read the data in one block, complete,
    resample and subtract."""
    _check_sampling_plan(index, sampling_info, needs_resample=False)
    req = _Request(index, ops, cells, hermitian)
    form = _choose_form(req, hermitian, subtract_vev, key_form)
    derived = subtract_vev and not form[1]
    _check_sampling_plan(index, sampling_info, needs_resample=derived)
    if derived or include_vevs:
        missing = [op.coordinate for op in ops if index._vev(op) is None]
        if missing:
            why = "connected correlators derived from raw data" if derived else "include_vevs"
            raise ValueError(f"{why} need VEVs; missing for: {', '.join(missing)}")
    chosen, omitted = _choose_times(req, form, None if times is None else _unique_times(times))

    # Row numbers into one block of stored observables: direct (i, j) and flipped (j, i).
    rows: dict[int, int] = {}
    obs_list: list = []

    def row(obs) -> int:
        k = rows.get(id(obs))
        if k is None:
            k = rows[id(obs)] = len(obs_list)
            obs_list.append(obs)
        return k

    n_t, n_c = len(chosen), len(cells)
    direct = np.full((n_t, n_c), -1, dtype=np.intp)
    flip = np.full((n_t, n_c), -1, dtype=np.intp)
    for c, (i, j) in enumerate(cells):
        d_series, f_series = req.series(i, j, form), req.flipped(i, j, form)
        for t, time in enumerate(chosen):
            obs = d_series.get(time)
            if obs is not None:
                direct[t, c] = row(obs)
            obs = f_series.get(time)
            if obs is not None:
                flip[t, c] = row(obs)

    _check_not_lone_imaginary(obs_list)
    materialize(obs_list)
    data = [obs.data for obs in obs_list]
    length = index.sample_length
    if any(d.shape != (length,) for d in data):
        raise ValueError(f"stored correlators must all have {length} samples")
    dtype = complex if any(obs.is_complex for obs in obs_list) else float
    buf = np.empty((n_t, n_c, length), dtype=dtype)
    flat = buf.reshape(-1, length)
    # Gather each cell's own direction (else the flipped one, conjugated below)
    # straight into the output; cells with both directions are then averaged.
    has_d, has_f = direct >= 0, flip >= 0
    np.stack([data[r] for r in np.where(has_d, direct, flip).ravel()], out=flat)
    if dtype is complex:
        conj_rows = np.flatnonzero(~has_d.ravel())
        flat[conj_rows] = np.conj(flat[conj_rows])
    both = np.flatnonzero((has_d & has_f).ravel())
    if both.size:
        other = np.empty((both.size, length), dtype=dtype)
        np.stack([data[r] for r in flip.ravel()[both]], out=other)
        np.conjugate(other, out=other)
        other += flat[both]
        other *= 0.5
        flat[both] = other

    v = None
    if derived or include_vevs:
        vev_obs = [index._vev(op) for op in ops]
        _check_not_lone_imaginary(vev_obs)
        v = stack_data(vev_obs)

    out_sampling = index.sampling_info
    if index.is_bins and sampling_info is not None:
        # One joint resample of every correlator and VEV bin trace.
        if v is None:
            (buf,) = _joint_resample([buf], index.ensemble_info, sampling_info)
        else:
            buf, v = _joint_resample([buf, v], index.ensemble_info, sampling_info)
        out_sampling = sampling_info
    if derived:
        ci = np.array([i for i, _ in cells])
        cj = np.array([j for _, j in cells])
        buf = buf - v[ci][None] * np.conj(v[cj])[None]

    return _Assembled(buf, has_d, v, out_sampling, chosen, omitted, form)


def _joint_resample(parts, ensemble_info, sampling_info) -> list[np.ndarray]:
    """Resample bin buffers ``(..., n_bins)`` with one shared plan, so derived
    products (e.g. VEV subtraction) see the same resamples in every part."""
    flat = [part.reshape(-1, part.shape[-1]) for part in parts]
    dtype = np.result_type(*flat)
    stacked = np.concatenate([f.astype(dtype, copy=False) for f in flat], axis=0)
    sampled = _block_resample(stacked, ensemble_info, sampling_info, ["mean"])[0]
    out, start = [], 0
    for part, f in zip(parts, flat):
        out.append(sampled[start : start + len(f)].reshape(part.shape[:-1] + sampled.shape[-1:]))
        start += len(f)
    return out


def assemble(
    index,
    info: CorrelatorMatrixInfo,
    times: Iterable[int] | None,
    key_form: KeyForm | None,
    sampling_info: SamplingInfo | None,
    include_vevs: bool,
) -> CorrelatorMatrix:
    """Build a :class:`CorrelatorMatrix` from ``index`` (see the module contract)."""
    ops = info.operators
    n = len(ops)
    cells = [(i, j) for i in range(n) for j in range(i if info.hermitian else 0, n)]
    a = _assemble(
        index,
        ops,
        cells,
        info.hermitian,
        info.subtract_vev,
        times,
        key_form,
        sampling_info,
        include_vevs,
    )
    ens = index.ensemble_info
    coords = list(info.coordinates)
    n_t = len(a.times)
    if info.hermitian:
        buf = np.empty((n_t, n, n, a.buf.shape[-1]), dtype=a.buf.dtype)
        stored_mask = np.empty((n_t, n, n), dtype=bool)
        for c, (i, j) in enumerate(cells):
            buf[:, i, j] = a.buf[:, c]
            stored_mask[:, i, j] = a.stored[:, c]
            if i != j:
                buf[:, j, i] = a.buf[:, c].conj()
                series = index._lookup(ops[j], ops[i], *a.form)
                stored_mask[:, j, i] = [t in series for t in a.times]
    else:
        buf = a.buf.reshape(n_t, n, n, a.buf.shape[-1])
        stored_mask = a.stored.reshape(n_t, n, n)
    values = _kind_cls(a.sampling_info)._from_parts(
        buf,
        ArrayObsInfo("correlator_matrix", ensemble_info=ens),
        a.sampling_info,
        bool(np.iscomplexobj(a.buf)),
        (TIME, SINK, SOURCE),
        {TIME: a.times, SINK: coords, SOURCE: coords},
        _matrix_infos(ops, a.times, info.hermitian, info.subtract_vev, ens),
    )
    vevs = None if a.vevs is None else _vev_array(ops, a.vevs, a.sampling_info, ens)
    stored = xr.DataArray(
        stored_mask,
        dims=(TIME, SINK, SOURCE),
        coords={TIME: a.times, SINK: coords, SOURCE: coords},
    )
    return CorrelatorMatrix(
        info,
        values,
        vevs,
        stored=stored,
        key_form=_form_name(a.form),
        omitted_times=tuple(a.omitted),
    )


def _vev_array(ops, buf, sampling_info, ens) -> SigmondData:
    return _kind_cls(sampling_info)._from_parts(
        buf,
        ArrayObsInfo("vevs", ensemble_info=ens),
        sampling_info,
        bool(np.iscomplexobj(buf)),
        (OPERATOR,),
        {OPERATOR: [op.coordinate for op in ops]},
        _vev_infos(ops, ens),
    )


def assemble_correlator(
    index,
    corr: CorrelatorInfo,
    times: Iterable[int] | None,
    hermitian: bool,
    subtract_vev: bool,
    key_form: KeyForm | None,
    sampling_info: SamplingInfo | None,
) -> SigmondData:
    """One correlator over time, by the same rules as a matrix."""
    same = corr.sink == corr.source
    ops = (corr.sink,) if same else (corr.sink, corr.source)
    cell = (0, 0) if same else (0, 1)
    a = _assemble(
        index, ops, [cell], hermitian, subtract_vev, times, key_form, sampling_info, False
    )
    ens = index.ensemble_info
    return _kind_cls(a.sampling_info)._from_parts(
        a.buf[:, 0],
        ArrayObsInfo("correlator", ensemble_info=ens),
        a.sampling_info,
        bool(np.iscomplexobj(a.buf)),
        (TIME,),
        {TIME: a.times},
        _cell_infos([corr], a.times, hermitian, subtract_vev, ens)[:, 0],
    )


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HermiticityCheck:
    """Deviations from Hermiticity in units of the statistical error.

    ``residual[t, i, j]`` is ``max(|Re D|, |Im D|) / error`` of the full-sample
    estimate for ``D = C_ij - conj(C_ji)`` (zero on the diagonal);
    ``diagonal_imag[t, i]`` is ``|Im C_ii| / error``.
    """

    residual: xr.DataArray
    diagonal_imag: xr.DataArray
    tol: float

    @property
    def ok(self) -> bool:
        return bool(
            np.isfinite(self.residual).all()
            and np.isfinite(self.diagonal_imag).all()
            and (self.residual <= self.tol).all()
            and (self.diagonal_imag <= self.tol).all()
        )

    @property
    def max_residual(self) -> float:
        return float(
            np.maximum(self.residual.max(skipna=False), self.diagonal_imag.max(skipna=False))
        )


def _ratio(value, error):
    valid = np.isfinite(value) & np.isfinite(error) & (error >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(error > 0, value / error, np.where(value > 0, np.inf, 0.0))
    return np.where(valid, ratio, np.nan)


class CorrelatorMatrix:
    """A labeled, sample-aligned correlation matrix ``C_ij(t)``.

    ``values`` is an ordinary ``SigmondData`` with dims ``(time, sink, source)``
    whose ``sink``/``source`` coords are the basis coordinates; it provides SS
    statistics, arithmetic, selection and fitting input. The wrapper adds its
    identity ``info`` (a :class:`CorrelatorMatrixInfo`: the basis and assembly
    rules), attached ``vevs`` (dims ``(operator,)``) and basis transforms, plus
    how the matrix was read:

    - ``key_form``: the stored key flags used (``"HermMat"``, ``"plain"``, ...),
      ``None`` for a transformed or hand-built matrix;
    - ``stored``: a boolean ``(time, sink, source)`` array, true where that
      ordered direction itself was stored. For a Hermitian matrix a cell is the
      average of both directions where ``stored`` and its transpose are true,
      and the conjugate of the flipped direction where only the transpose is;
    - ``omitted_times``: observed but incomplete times left out of ``times=None``.

    ``numeric_type`` is the optional pivot convention. Complex pivots emit
    explicit zero Im datasets when writing real buffers, so CN Sigmond can
    consume them. Generic basis changes clear this pivot convention.

    Identity lives in the coords: arithmetic on ``values`` may replace its cell
    infos with placeholders, and :meth:`to_collection` rebuilds every primary
    key from the coords and :attr:`info` instead of trusting them.
    """

    def __init__(
        self,
        info: CorrelatorMatrixInfo,
        values: SigmondData,
        vevs: SigmondData | None = None,
        *,
        stored: xr.DataArray | None = None,
        key_form: KeyForm | None = None,
        omitted_times: tuple[int, ...] = (),
        transformation: Transformation | None = None,
        numeric_type: str | None = None,
    ):
        coords = list(info.coordinates)
        if values.dims != (TIME, SINK, SOURCE):
            raise ValueError(f"values must have dims (time, sink, source), got {values.dims}")
        for dim in (SINK, SOURCE):
            if list(values.coords.get(dim, [])) != coords:
                raise ValueError(f"values' {dim} coords must equal the basis coordinates")
        if TIME not in values.coords:
            raise ValueError("values need time coords")
        if vevs is not None:
            if vevs.dims != (OPERATOR,) or list(vevs.coords.get(OPERATOR, [])) != coords:
                raise ValueError("vevs must have dim (operator,) with the basis coordinates")
            from .index import _check_same_plan

            _check_same_plan(values, vevs)
        if stored is not None and (
            stored.dims != (TIME, SINK, SOURCE)
            or stored.shape != values.shape
            or any(
                dim not in stored.coords
                or not np.array_equal(stored.coords[dim], values.coords[dim])
                for dim in (TIME, SINK, SOURCE)
            )
        ):
            raise ValueError("stored must be aligned with values")
        self.info = info
        self.values = values
        self.vevs = vevs
        self.stored = stored
        self.key_form = key_form
        self.omitted_times = tuple(omitted_times)
        if numeric_type not in (None, "complex", "real"):
            raise ValueError("numeric_type must be complex, real or None")
        self.transformation = transformation
        self.numeric_type = numeric_type
        self._by_coordinate = {op.coordinate: op for op in info.operators}

    # -- metadata -------------------------------------------------------------

    @property
    def operators(self) -> tuple[Operator, ...]:
        return self.info.operators

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(int(t) for t in self.values.coords[TIME])

    @property
    def sampling_info(self) -> SamplingInfo | None:
        return self.values.sampling_info

    @property
    def is_bins(self) -> bool:
        return self.values.sampling_info is None

    def __len__(self) -> int:
        return len(self.info)

    def __repr__(self) -> str:
        kind = "bins" if self.is_bins else repr(self.sampling_info)
        flags = [
            f
            for f, on in (
                ("hermitian", self.info.hermitian),
                ("VEV-subtracted", self.info.subtract_vev),
            )
            if on
        ]
        form = f", key form {self.key_form!r}" if self.key_form else ""
        return (
            f"CorrelatorMatrix({len(self)}x{len(self)}, times={list(self.times)}, "
            f"{', '.join(flags) or 'raw'}{form}, {kind})"
        )

    def _replace(self, **changes) -> CorrelatorMatrix:
        fields = dict(
            info=self.info,
            values=self.values,
            vevs=self.vevs,
            stored=self.stored,
            key_form=self.key_form,
            omitted_times=self.omitted_times,
            transformation=self.transformation,
            numeric_type=self.numeric_type,
        )
        fields.update(changes)
        info, values, vevs = fields.pop("info"), fields.pop("values"), fields.pop("vevs")
        return CorrelatorMatrix(info, values, vevs, **fields)

    # -- slicing --------------------------------------------------------------

    def _coord(self, op) -> str:
        if isinstance(op, str):
            if op in self._by_coordinate:
                return op
            from .operators import parse_operator

            try:
                op = parse_operator(op)
            except ValueError:
                raise KeyError(f"{op!r} is not a basis coordinate or operator") from None
        if op not in self.info.operators:
            raise KeyError(f"{op.coordinate} is not in the basis")
        return op.coordinate

    def _labels(self, value):
        if isinstance(value, (list, tuple)):
            return [self._coord(v) for v in value]
        return self._coord(value)

    def sel(self, **indexers) -> SigmondData:
        """``values.sel`` that also accepts operators (objects or strings) for ``sink``/``source``."""
        for dim in (SINK, SOURCE):
            if dim in indexers:
                indexers[dim] = self._labels(indexers[dim])
        return self.values.sel(**indexers)

    def at(self, time: int) -> SigmondData:
        """The ``(sink, source)`` matrix at one time."""
        return self.values.sel(time=time)

    def correlator(self, sink, source) -> SigmondData:
        """``C_ij`` over time, as a 1-D ``SigmondData``."""
        return self.values.sel(sink=self._coord(sink), source=self._coord(source))

    def diagonal(self, component: str = "complex") -> SigmondData:
        """The diagonal correlators, dims ``(time, operator)``.

        ``component="real"`` explicitly projects to the real part; the
        default keeps the complex values.
        """
        diag = self.values.diagonal(SINK, SOURCE, OPERATOR)
        if component == "complex":
            return diag
        return diag.component(component)

    def select(self, operators: Sequence) -> CorrelatorMatrix:
        """The sub-matrix of ``operators`` (in the given order); ``info`` and VEVs follow."""
        coords = [self._coord(op) for op in operators]
        pos = [self.info.coordinates.index(c) for c in coords]
        return self._replace(
            info=self.info.with_operators([self._by_coordinate[c] for c in coords]),
            values=self.values.isel(sink=pos, source=pos),
            vevs=None if self.vevs is None else self.vevs.isel(operator=pos),
            stored=None if self.stored is None else self.stored.isel(sink=pos, source=pos),
            transformation=None
            if self.transformation is None
            else Transformation(
                self.transformation.source_operators, self.transformation.coefficients[:, pos]
            ),
        )

    def sel_times(self, times: Sequence[int]) -> CorrelatorMatrix:
        """The same matrix restricted to ``times``."""
        times = list(times)
        return self._replace(
            values=self.values.sel(time=times),
            stored=None if self.stored is None else self.stored.sel(time=times),
        )

    # -- basis change ---------------------------------------------------------

    def transform(
        self,
        coefficients,
        operators: Sequence[Operator | str] | None = None,
        *,
        name: str = "ROT",
    ) -> CorrelatorMatrix:
        """Change basis: ``O_new[a] = sum_i conj(T[i,a]) O_old[i]``, ``C_new = T^H C T``.

        ``coefficients`` is ``T`` with shape ``(N_original, N_new)`` (columns
        are the new operators); ``operators`` are the new GI labels (objects or
        strings), which must share the exact basis channel. Original labels
        may be retained by exact unit-selection columns. Attached VEVs become
        ``T^H v``. The coefficients are kept in :attr:`transformation`,
        composed across repeated transforms.
        """
        return self._transform(coefficients, operators, name=name)

    def _transform(self, coefficients, operators=None, *, name="ROT", canonical_hermitian=False):
        """Shared basis/metadata transformation, with private pivot canonicalization."""
        from .operators import parse_operator

        T = np.asarray(coefficients)
        if operators is None:
            if T.ndim != 2:
                raise ValueError("coefficients must be a matrix")
            operators = [
                GenIrrepOperator.from_channel(self.info.channel, name=name, index=k)
                for k in range(T.shape[1])
            ]
        ops = tuple(parse_operator(op) for op in operators)
        n = len(self.info)
        if T.ndim != 2 or T.shape != (n, len(ops)):
            raise ValueError(f"coefficients must have shape ({n}, {len(ops)}), got {T.shape}")
        if not np.all(np.isfinite(T)):
            raise ValueError("coefficients must be finite")
        channel = self.info.channel
        for k, op in enumerate(ops):
            if channel.momentum is not None and isinstance(op, GenIrrepOperator) and not op.has_definite_momentum:
                raise ValueError(f"{op.coordinate}: a PSQ-only label is not a definite-momentum operator")
            if isinstance(op, GenIrrepOperator):
                continue
            if op not in self.operators:
                raise TypeError(f"new operators must be GI labels or selected original labels, got {op!r}")
            column = np.zeros(n)
            column[self.operators.index(op)] = 1
            if not np.array_equal(T[:, k], column):
                raise ValueError(f"{op.coordinate}: original labels require exact unit selection")
        new_info = self.info.with_operators(ops)
        if new_info.channel != channel:
            raise ValueError(f"new operators are not in the basis channel {channel}")
        coords = list(new_info.coordinates)

        from ..sampling_array import _contract

        ens = self.values.ensemble_info

        def wrap(data, original, dims, labels, infos):
            return _kind_cls(original.sampling_info)._from_parts(
                data,
                original.observable_info,
                original.sampling_info,
                np.iscomplexobj(data),
                dims,
                labels,
                infos,
            )

        from ._kernels import rotate_matrix

        data = rotate_matrix(self.values.data, T, canonical_hermitian=canonical_hermitian)
        c = wrap(
            data,
            self.values,
            (TIME, SINK, SOURCE),
            {TIME: list(self.times), SINK: coords, SOURCE: coords},
            _matrix_infos(ops, self.times, new_info.hermitian, new_info.subtract_vev, ens),
        )
        # Exact unit selections preserve existing bindings; mixtures establish none.
        selected = []
        for column in T.T:
            positions = np.flatnonzero(column)
            selected.append(int(positions[0]) if len(positions) == 1 and column[positions[0]] == 1 else None)
        for t, i, j in np.ndindex(c.infos.shape):
            a, b = selected[i], selected[j]
            sink = None if a is None else getattr(self.values.infos[t, a, a], "sink_rotation", None)
            source = None if b is None else getattr(self.values.infos[t, b, b], "source_rotation", None)
            c.infos[t, i, j] = c.infos[t, i, j].with_rotation(sink, source)
        vevs = None
        if self.vevs is not None:
            vevs = wrap(
                _contract(self.vevs.data, T.conj(), 0),
                self.vevs,
                (OPERATOR,),
                {OPERATOR: coords},
                _vev_infos(ops, ens),
            )

        if self.transformation is None:
            transformation = Transformation(self.operators, T.copy())
        else:
            prev = self.transformation
            transformation = Transformation(prev.source_operators, prev.coefficients @ T)
        return CorrelatorMatrix(
            new_info,
            c,
            vevs,
            omitted_times=self.omitted_times,
            transformation=transformation,
        )

    def improve(self, basis: Sequence[Operator | str], definitions) -> CorrelatorMatrix:
        """Build ``T`` from labeled columns and return ``T† C T``.

        Definition coefficients are T[i,a], so operator coefficients are their
        conjugates. Plain basis entries are exact source selections.
        """
        from .operators import parse_operator
        ops = tuple(parse_operator(op) for op in basis)
        parsed = {}
        for key, terms in definitions.items():
            op = parse_operator(key)
            if op in parsed:
                raise ValueError(f"duplicate improved definition: {op.coordinate}")
            if op not in ops:
                raise ValueError(f"definition is not in target basis: {op.coordinate}")
            parsed[op] = terms
        T = np.zeros((len(self), len(ops)), dtype=complex)
        for col, op in enumerate(ops):
            if op not in parsed:
                if op not in self.operators:
                    raise ValueError(f"missing improved definition: {op.coordinate}")
                T[self.operators.index(op), col] = 1
                continue
            seen = set()
            for source, coefficient in parsed[op].items():
                source = parse_operator(source)
                if source in seen:
                    raise ValueError(f"duplicate source term: {source.coordinate}")
                seen.add(source)
                if source not in self.operators:
                    raise ValueError(f"missing source operator: {source.coordinate}")
                if not np.isscalar(coefficient) or not np.isfinite(coefficient):
                    raise ValueError("improved coefficients must be finite scalars")
                T[self.operators.index(source), col] = coefficient
            if not np.any(T[:, col]):
                raise ValueError(f"empty/all-zero definition: {op.coordinate}")
        if not np.any(T.imag):
            T = T.real
        # Labels, uniqueness and channel are validated by the transform.
        return self._transform(T, ops, canonical_hermitian=self.info.hermitian)

    # -- diagnostics ----------------------------------------------------------

    def normalized(self, norm_time: int) -> SigmondData:
        """``C_ij(t) / sqrt(f_i f_j)``: Sigmond's ``TemporalCorrelatorMatrixRescaled``.

        ``f_i`` is the full-sample estimate of ``Re C_ii(norm_time)`` for a
        Hermitian matrix and of ``|C_ii(norm_time)|`` otherwise. It is one
        constant per cell, so every sample (real and imaginary parts alike) is
        divided by the same number: the rescale is linear and adds no bias.
        ``|C~_ij(norm_time)|`` near 1 means operators ``i`` and ``j`` are nearly the same.

        Returns a plain ``SigmondData`` with dims ``(time, sink, source)``, not
        a ``CorrelatorMatrix``: it is a display quantity, not a new basis.

        Raises:
            ValueError: ``norm_time`` is not a time of the matrix, or some
                ``f_i`` is not positive (named).
        """
        data = np.asarray(self.values.data)
        scale = self._normalization_scale(norm_time)
        coords = list(self.info.coordinates)
        return SigmondData.from_data(
            data * scale[None, :, :, None],
            ArrayObsInfo(f"normalized_t0_{norm_time}", ensemble_info=self.values.ensemble_info),
            self.sampling_info,
            dim_names=(TIME, SINK, SOURCE),
            coords={TIME: list(self.times), SINK: coords, SOURCE: coords},
            ensemble_info=self.values.ensemble_info,
            is_complex=self.values.is_complex,
        )

    def _normalization_scale(self, norm_time: int) -> np.ndarray:
        if norm_time not in self.times:
            raise ValueError(
                f"norm_time {norm_time} is not a time of the matrix {list(self.times)}"
            )
        n = len(self.info)
        data = np.asarray(self.values.data)
        at = data[self.times.index(norm_time)]
        full = np.asarray(self.values._analyzer.full(at[np.arange(n), np.arange(n)]))
        factors = full.real if self.info.hermitian else np.abs(full)
        bad = [op.coordinate for op, f in zip(self.operators, factors) if not f > 0]
        if bad:
            what = "Re C_ii" if self.info.hermitian else "|C_ii|"
            raise ValueError(
                f"cannot normalize at t0={norm_time}: {what}(t0) is not positive for {bad}"
            )
        return 1.0 / np.sqrt(np.outer(factors, factors))

    def significance(self, time: int) -> xr.DataArray:
        """``|C_ij| / error`` at ``time``, from the full-sample estimate.

        A real ``(sink, source)`` array; small values flag noise-dominated
        cells. For complex data the error combines both parts. A zero error
        gives ``inf`` (or 0 for a zero value).
        """
        if time not in self.times:
            raise ValueError(f"time {time} is not a time of the matrix {list(self.times)}")
        at = np.asarray(self.values.data)[self.times.index(time)]
        value = np.abs(self.values._analyzer.full(at))
        error = self.values._analyzer.error(at)
        ratio = _ratio(value, error)
        coords = list(self.info.coordinates)
        return xr.DataArray(ratio, dims=(SINK, SOURCE), coords={SINK: coords, SOURCE: coords})

    def check_hermitian(self, tol: float = 3.0) -> HermiticityCheck:
        """Compare stored directions and imaginary diagonals in error units.

        Hermitian assembly has already averaged off-diagonal disagreement;
        use a non-Hermitian request to diagnose the original directions.
        Invalid diagnostics remain NaN and never pass ``ok``.
        """
        data = np.asarray(self.values.data)
        err = self.values._analyzer.error
        d = data - np.conj(np.swapaxes(data, 1, 2))
        full = self.values._analyzer.full

        def sigma(x):
            e = err(x)
            val = np.abs(full(x))
            return _ratio(val, e)

        residual = np.maximum(sigma(d.real), sigma(d.imag))
        n = len(self.info)
        residual[:, np.arange(n), np.arange(n)] = 0.0  # the diagonal is diagonal_imag
        diag = np.diagonal(data, axis1=1, axis2=2)  # (time, sample, operator)
        diagonal = np.moveaxis(diag, -1, 1)
        diag_imag = np.where(np.isfinite(diagonal).all(axis=-1), sigma(diagonal.imag), np.nan)
        coords = list(self.info.coordinates)
        return HermiticityCheck(
            xr.DataArray(
                residual,
                dims=(TIME, SINK, SOURCE),
                coords={TIME: list(self.times), SINK: coords, SOURCE: coords},
            ),
            xr.DataArray(
                diag_imag, dims=(TIME, OPERATOR), coords={TIME: list(self.times), OPERATOR: coords}
            ),
            tol,
        )

    # -- output ---------------------------------------------------------------

    def to_collection(self, include_vevs: bool = True, *,
                      diagonal_only_before_time: int | None = None,
                      remove_off_diagonal: bool = False,
                      upper_triangle_only: bool = False):
        """Single observables whose primary keys are rebuilt from the coords and ``info``.

        ``diagonal_only_before_time`` and ``remove_off_diagonal`` filter exported
        keys without changing the numerical matrix. ``upper_triangle_only``
        exports just the upper triangle in the current basis order.

        Correlator keys carry the matrix's flags (``HermMat`` for a Hermitian
        matrix, ``SubVEV`` for a connected one); VEVs are included if attached.
        """
        from ..ensemble_collection import SingleEnsembleCollection

        ens = self.values.ensemble_info
        ops = [self._by_coordinate[c] for c in self.values.coords[SINK]]
        infos = _matrix_infos(ops, self.times, self.info.hermitian, self.info.subtract_vev, ens)
        values = self.values._with_replaced_infos(infos)
        entries = values.to_collection()
        keep = np.ones(self.values.shape, dtype=bool)
        offdiag = ~np.eye(len(ops), dtype=bool)
        if upper_triangle_only:
            keep[:, *np.tril_indices(len(ops), -1)] = False
        if remove_off_diagonal:
            keep[:, offdiag] = False
        elif diagonal_only_before_time is not None:
            keep[np.asarray(self.times) < diagonal_only_before_time] &= ~offdiag
        items = [value for value, selected in zip(entries, keep.ravel()) if selected]
        if include_vevs and self.vevs is not None:
            vev_ops = [self._by_coordinate[c] for c in self.vevs.coords[OPERATOR]]
            items += list(self.vevs._with_replaced_infos(_vev_infos(vev_ops, ens)).to_collection())
        return SingleEnsembleCollection(items)

    def write(self, filename, group: str = "samplings", **kwargs):
        """Write :meth:`to_collection` with :class:`~sigmondsamplings.io.writer.SigmondWriter`."""
        from ..io.writer import SigmondWriter

        items = list(self.to_collection(
            include_vevs=kwargs.pop("include_vevs", True),
            diagonal_only_before_time=kwargs.pop("diagonal_only_before_time", None),
            remove_off_diagonal=kwargs.pop("remove_off_diagonal", False),
            upper_triangle_only=kwargs.pop("upper_triangle_only", False),
        ))
        writer = SigmondWriter(create_backups=kwargs.pop("create_backups", False),
                               include_imaginary=kwargs.pop("include_imaginary", self.numeric_type == "complex"))
        if self.is_bins:
            writer.write_bins_hdf5(str(filename), items, group=group, **kwargs)
        else:
            writer.write_hdf5(str(filename), items, group=group, **kwargs)
