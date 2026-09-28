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
from .keys import CorrelatorKey, CorrelatorMatrixSpec, CorrelatorPair, VEVKey
from .observables import CorrelatorObsInfo, VEVObsInfo
from .operators import GeneralOperator, Operator

__all__ = [
    "CorrelatorMatrix",
    "HermiticityCheck",
    "Transformation",
    "assemble",
    "assemble_pair",
]

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
    pairs: Sequence[CorrelatorPair],
    times: Sequence[int],
    hermitian: bool,
    subtract_vev: bool,
    ensemble_info: EnsembleInfo,
) -> np.ndarray:
    """``(T, len(pairs))`` correlator infos; their names are generated only when needed."""
    infos = np.empty((len(times), len(pairs)), dtype=object)
    for t, time in enumerate(times):
        for c, pair in enumerate(pairs):
            key = CorrelatorKey._unchecked(pair, time, hermitian, subtract_vev)
            infos[t, c] = CorrelatorObsInfo(key=key, ensemble_info=ensemble_info)
    return infos


def _matrix_pairs(ops: Sequence[Operator]) -> list[CorrelatorPair]:
    return [CorrelatorPair(a, b) for a in ops for b in ops]


def _matrix_infos(ops, times, hermitian, subtract_vev, ens) -> np.ndarray:
    n = len(ops)
    infos = _cell_infos(_matrix_pairs(ops), times, hermitian, subtract_vev, ens)
    return infos.reshape(len(times), n, n)


def _vev_infos(operators: Sequence[Operator], ensemble_info: EnsembleInfo) -> np.ndarray:
    infos = np.empty(len(operators), dtype=object)
    for i, op in enumerate(operators):
        infos[i] = VEVObsInfo(key=VEVKey(op), ensemble_info=ensemble_info)
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

    def series(self, i, j, form) -> dict[int, SigmondData]:
        return self.index._lookup(self.ops[i], self.ops[j], *form)

    def flipped(self, i, j, form) -> dict[int, SigmondData]:
        if self.hermitian and i != j:
            return self.series(j, i, form)
        return {}

    def cell_times(self, i, j, form) -> set[int]:
        return set(self.series(i, j, form)) | set(self.flipped(i, j, form))

    def present(self, form, *, either_direction: bool = False) -> int:
        """How many cells have any stored time in ``form``.

        ``either_direction`` also counts cells stored only flipped, even for a
        non-Hermitian request (used to suggest other forms).
        """
        return sum(
            1
            for i, j in self.cells
            if self.cell_times(i, j, form) or (either_direction and self.series(j, i, form))
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


def _choose_times(req: _Request, form, times) -> tuple[list[int], list[int]]:
    per_cell = [req.cell_times(i, j, form) for i, j in req.cells]
    complete = set.intersection(*per_cell)
    observed = set.union(*per_cell)
    label = _form_name(form)
    if times is not None:
        missing = [t for t in times if t not in complete]
        if missing:
            raise ValueError(
                f"not every requested cell is stored at times {missing} "
                f"(key form {label!r}):\n{req.coverage(form, missing)}"
            )
        return list(times), []
    if not complete:
        raise ValueError(
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
        if isinstance(t, bool) or int(t) != t or t < 0:
            raise ValueError(f"times must be nonnegative integers, got {t!r}")
        if int(t) in out:
            raise ValueError(f"time {t} requested twice")
        out.append(int(t))
    if not out:
        raise ValueError("no times requested")
    return out


def _check_not_lone_imaginary(obs_list) -> None:
    for obs in obs_list:
        if not obs.is_complex and obs.observable_info.re_im == "im":
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
    chosen, omitted = _choose_times(
        req, form, None if times is None else _unique_times(times)
    )

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
        parts = [buf.reshape(-1, buf.shape[-1])] + ([v] if v is not None else [])
        dtype = np.result_type(*parts)
        stacked = np.concatenate([p.astype(dtype, copy=False) for p in parts], axis=0)
        resampled = _block_resample(stacked, index.ensemble_info, sampling_info, ["mean"])[0]
        n_rows = n_t * n_c
        buf = resampled[:n_rows].reshape(n_t, n_c, resampled.shape[-1])
        if v is not None:
            v = resampled[n_rows:]
        out_sampling = sampling_info
    if derived:
        ci = np.array([i for i, _ in cells])
        cj = np.array([j for _, j in cells])
        buf = buf - v[ci][None] * np.conj(v[cj])[None]

    return _Assembled(buf, has_d, v, out_sampling, chosen, omitted, form)


def assemble(
    index,
    spec: CorrelatorMatrixSpec,
    times: Iterable[int] | None,
    key_form: str | None,
    sampling_info: SamplingInfo | None,
    include_vevs: bool,
) -> CorrelatorMatrix:
    """Build a :class:`CorrelatorMatrix` from ``index`` (see the module contract)."""
    ops = spec.operators
    n = len(ops)
    cells = [(i, j) for i in range(n) for j in range(n)]
    a = _assemble(
        index,
        ops,
        cells,
        spec.hermitian,
        spec.subtract_vev,
        times,
        key_form,
        sampling_info,
        include_vevs,
    )
    ens = index.ensemble_info
    coords = list(spec.coordinates)
    n_t = len(a.times)
    values = _kind_cls(a.sampling_info)._from_parts(
        a.buf.reshape(n_t, n, n, a.buf.shape[-1]),
        ArrayObsInfo("correlator_matrix", ensemble_info=ens),
        a.sampling_info,
        bool(np.iscomplexobj(a.buf)),
        (TIME, SINK, SOURCE),
        {TIME: a.times, SINK: coords, SOURCE: coords},
        _matrix_infos(ops, a.times, spec.hermitian, spec.subtract_vev, ens),
    )
    vevs = None if a.vevs is None else _vev_array(ops, a.vevs, a.sampling_info, ens)
    stored = xr.DataArray(
        a.stored.reshape(n_t, n, n),
        dims=(TIME, SINK, SOURCE),
        coords={TIME: a.times, SINK: coords, SOURCE: coords},
    )
    return CorrelatorMatrix(
        spec,
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


def assemble_pair(
    index,
    pair: CorrelatorPair,
    times: Iterable[int] | None,
    hermitian: bool,
    subtract_vev: bool,
    key_form: str | None,
    sampling_info: SamplingInfo | None,
) -> SigmondData:
    """One correlator over time, by the same rules as a one-pair matrix."""
    same = pair.sink == pair.source
    ops = (pair.sink,) if same else (pair.sink, pair.source)
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
        _cell_infos([pair], a.times, hermitian, subtract_vev, ens)[:, 0],
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
        return bool((self.residual <= self.tol).all() and (self.diagonal_imag <= self.tol).all())

    @property
    def max_residual(self) -> float:
        return float(max(self.residual.max(), self.diagonal_imag.max()))


class CorrelatorMatrix:
    """A labeled, sample-aligned correlation matrix ``C_ij(t)``.

    ``values`` is an ordinary ``SigmondData`` with dims ``(time, sink, source)``
    whose ``sink``/``source`` coords are the basis coordinates; it provides SS
    statistics, arithmetic, selection and fitting input. The wrapper adds the
    basis rules (``spec``), attached ``vevs`` (dims ``(operator,)``) and basis
    transforms, plus how the matrix was read:

    - ``key_form``: the stored key flags used (``"HermMat"``, ``"plain"``, ...),
      ``None`` for a transformed or hand-built matrix;
    - ``stored``: a boolean ``(time, sink, source)`` array, true where that
      ordered direction itself was stored. For a Hermitian matrix a cell is the
      average of both directions where ``stored`` and its transpose are true,
      and the conjugate of the flipped direction where only the transpose is;
    - ``omitted_times``: observed but incomplete times left out of ``times=None``.

    Identity lives in the coords: arithmetic on ``values`` may replace its cell
    infos with placeholders, and :meth:`to_collection` rebuilds every primary
    key from the coords and ``spec`` instead of trusting them.
    """

    def __init__(
        self,
        spec: CorrelatorMatrixSpec,
        values: SigmondData,
        vevs: SigmondData | None = None,
        *,
        stored: xr.DataArray | None = None,
        key_form: str | None = None,
        omitted_times: tuple[int, ...] = (),
        transformation: Transformation | None = None,
    ):
        coords = list(spec.coordinates)
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
            if vevs.sampling_info != values.sampling_info:
                raise ValueError("vevs and values must share one sampling plan")
        if stored is not None and (
            stored.dims != (TIME, SINK, SOURCE) or stored.shape != values.shape
        ):
            raise ValueError("stored must be aligned with values")
        self.spec = spec
        self.values = values
        self.vevs = vevs
        self.stored = stored
        self.key_form = key_form
        self.omitted_times = tuple(omitted_times)
        self.transformation = transformation
        self._by_coordinate = {op.coordinate: op for op in spec.operators}

    # -- metadata -------------------------------------------------------------

    @property
    def operators(self) -> tuple[Operator, ...]:
        return self.spec.operators

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
        return len(self.spec)

    def __repr__(self) -> str:
        kind = "bins" if self.is_bins else repr(self.sampling_info)
        flags = [
            f
            for f, on in (
                ("hermitian", self.spec.hermitian),
                ("VEV-subtracted", self.spec.subtract_vev),
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
            spec=self.spec,
            values=self.values,
            vevs=self.vevs,
            stored=self.stored,
            key_form=self.key_form,
            omitted_times=self.omitted_times,
            transformation=self.transformation,
        )
        fields.update(changes)
        spec, values, vevs = fields.pop("spec"), fields.pop("values"), fields.pop("vevs")
        return CorrelatorMatrix(spec, values, vevs, **fields)

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
        if op not in self.spec.operators:
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
        if component == "real":
            real = diag.to_real()
            infos = np.empty(real.shape, dtype=object)
            for idx in np.ndindex(*real.shape):
                info = real.infos[idx].copy()
                info.re_im = "re"
                infos[idx] = info
            return real._with_replaced_infos(infos)
        raise ValueError(f"component must be 'complex' or 'real', got {component!r}")

    def select(self, operators: Sequence) -> CorrelatorMatrix:
        """The sub-matrix of ``operators`` (in the given order); spec and VEVs follow."""
        coords = [self._coord(op) for op in operators]
        pos = [self.spec.coordinates.index(c) for c in coords]
        return self._replace(
            spec=self.spec.with_operators([self._by_coordinate[c] for c in coords]),
            values=self.values.isel(sink=pos, source=pos),
            vevs=None if self.vevs is None else self.vevs.isel(operator=pos),
            stored=None if self.stored is None else self.stored.isel(sink=pos, source=pos),
        )

    def sel_times(self, times: Sequence[int]) -> CorrelatorMatrix:
        """The same matrix restricted to ``times``."""
        times = list(times)
        return self._replace(
            values=self.values.sel(time=times),
            stored=None if self.stored is None else self.stored.sel(time=times),
        )

    # -- basis change ---------------------------------------------------------

    def transform(self, coefficients, operators: Sequence[GeneralOperator]) -> CorrelatorMatrix:
        """Change basis: ``O_new[a] = sum_i conj(T[i,a]) O_old[i]``, ``C_new = T^H C T``.

        ``coefficients`` is ``T`` with shape ``(N_original, N_new)`` (columns
        are the new operators); ``operators`` are the new GI labels (objects or
        strings), which must share the basis channel with the same definite
        momentum. Attached VEVs become ``T^H v``. The coefficients are kept in
        :attr:`transformation`, composed across repeated transforms.
        """
        from .operators import parse_operator

        T = np.asarray(coefficients)
        ops = tuple(parse_operator(op, "general") for op in operators)
        n = len(self.spec)
        if T.ndim != 2 or T.shape != (n, len(ops)):
            raise ValueError(f"coefficients must have shape ({n}, {len(ops)}), got {T.shape}")
        if not np.all(np.isfinite(T)):
            raise ValueError("coefficients must be finite")
        channel = self.spec.channel
        if channel.momentum is None:
            raise ValueError("basis changes need a definite-momentum basis")
        for op in ops:
            if not isinstance(op, GeneralOperator):
                raise TypeError(f"new operators must be GI labels, got {op!r}")
            if not op.has_definite_momentum:
                raise ValueError(
                    f"{op.coordinate}: a PSQ-only label is not a definite-momentum operator"
                )
            if op.channel != channel:
                raise ValueError(f"{op.coordinate} is not in the basis channel {channel}")
        new_spec = self.spec.with_operators(ops)
        coords = list(new_spec.coordinates)

        c = self.values.apply_matrix(T, SINK, conj=True, coords=coords)
        c = c.apply_matrix(T, SOURCE, coords=coords)
        ens = self.values.ensemble_info
        c = c._with_replaced_infos(
            _matrix_infos(ops, self.times, new_spec.hermitian, new_spec.subtract_vev, ens)
        )
        vevs = None
        if self.vevs is not None:
            v = self.vevs.apply_matrix(T, OPERATOR, conj=True, coords=coords)
            vevs = v._with_replaced_infos(_vev_infos(ops, ens))

        if self.transformation is None:
            transformation = Transformation(self.operators, T.copy())
        else:
            prev = self.transformation
            transformation = Transformation(prev.source_operators, prev.coefficients @ T)
        return CorrelatorMatrix(
            new_spec,
            c,
            vevs,
            omitted_times=self.omitted_times,
            transformation=transformation,
        )

    # -- checks ---------------------------------------------------------------

    def check_hermitian(self, tol: float = 3.0) -> HermiticityCheck:
        """Compare ``C_ij`` with ``conj(C_ji)`` and the diagonal's imaginary part."""
        data = np.asarray(self.values.data)
        err = self.values._analyzer.error
        d = data - np.conj(np.swapaxes(data, 1, 2))
        full = self.values._analyzer.full

        def sigma(x):
            e = err(x)
            val = np.abs(full(x))
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(e > 0, val / e, np.where(val > 0, np.inf, 0.0))

        residual = np.maximum(sigma(d.real), sigma(d.imag))
        n = len(self.spec)
        residual[:, np.arange(n), np.arange(n)] = 0.0  # the diagonal is diagonal_imag
        diag = np.diagonal(data, axis1=1, axis2=2)  # (time, sample, operator)
        diag_imag = sigma(np.moveaxis(diag, -1, 1).imag)
        coords = list(self.spec.coordinates)
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

    def to_collection(self, include_vevs: bool = True):
        """Single observables whose primary keys are rebuilt from the coords and spec.

        Correlator keys carry the spec's flags (``HermMat`` for a Hermitian
        spec, ``SubVEV`` for a connected one); VEVs are included if attached.
        """
        from ..ensemble_collection import SingleEnsembleCollection

        ens = self.values.ensemble_info
        ops = [self._by_coordinate[c] for c in self.values.coords[SINK]]
        infos = _matrix_infos(ops, self.times, self.spec.hermitian, self.spec.subtract_vev, ens)
        items = list(self.values._with_replaced_infos(infos).to_collection())
        if include_vevs and self.vevs is not None:
            vev_ops = [self._by_coordinate[c] for c in self.vevs.coords[OPERATOR]]
            items += list(self.vevs._with_replaced_infos(_vev_infos(vev_ops, ens)).to_collection())
        return SingleEnsembleCollection(items)

    def write(self, filename, group: str = "samplings", **kwargs):
        """Write :meth:`to_collection` with :class:`~sigmondsamplings.io.writer.SigmondWriter`."""
        from ..io.writer import SigmondWriter

        items = list(self.to_collection())
        writer = SigmondWriter(create_backups=kwargs.pop("create_backups", False))
        if self.is_bins:
            writer.write_bins_hdf5(str(filename), items, group=group, **kwargs)
        else:
            writer.write_hdf5(str(filename), items, group=group, **kwargs)
