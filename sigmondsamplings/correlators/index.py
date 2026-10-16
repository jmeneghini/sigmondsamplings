"""
A metadata-only index from typed correlator/VEV keys to loaded observables.

:class:`CorrelatorIndex` reads the key of every correlator (``<CorrT>``) and
VEV observable of a collection and maps it to the existing (possibly lazy)
observable. Building and listing an index never reads samples; requesting a
correlator or matrix reads only the selected cells and times.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from ..info import EnsembleInfo, SamplingInfo
from .keys import CorrelatorAtTimeInfo, CorrelatorInfo, CorrelatorMatrixInfo, KeyForm, VEVInfo
from .observables import key_of
from .operators import BasicLapHOperator, Channel, GenIrrepOperator, Operator, parse_operator

if TYPE_CHECKING:
    import xarray as xr

    from ..data import SigmondData
    from .matrix import CorrelatorMatrix

__all__ = ["CorrelatorIndex"]


class CorrelatorIndex:
    """Correlators and VEVs of one ensemble and one sampling plan, by typed key.

    Build with :meth:`from_collection` (or ``SigmondLoader(...).correlators``).
    The index is a snapshot of keys and copied observable metadata, with
    shared eager sample buffers or deferred lazy records. Later metadata edits
    to the collection do not change it; rebuild it, or call the loader's
    ``refresh_correlators()``. The snapshot metadata and sample buffers should be treated as immutable.
    """

    def __init__(
        self,
        correlators: dict[CorrelatorAtTimeInfo, SigmondData],
        vevs: dict[Operator, SigmondData],
        *,
        ensemble_info: EnsembleInfo | None,
        sampling_info: SamplingInfo | None,
        is_bins: bool,
        sample_length: int | None,
        unparsed: Sequence[tuple[str, str]] = (),
    ):
        self._by_key = dict(correlators)
        self._vevs = dict(vevs)
        self.ensemble_info = ensemble_info
        self.sampling_info = sampling_info
        self.is_bins = is_bins
        self.sample_length = sample_length
        self.unparsed = tuple(unparsed)
        """``(name, reason)`` for key-like names that did not parse (they are not indexed)."""

        # (correlator, hermitian, subtract_vev) -> {time: observable}
        self._series: dict[tuple, dict[int, SigmondData]] = {}
        ops: dict[Operator, None] = {}
        for key, obs in self._by_key.items():
            form = (key.correlator, key.hermitian, key.subtract_vev)
            self._series.setdefault(form, {})[key.time] = obs
            ops.setdefault(key.sink)
            ops.setdefault(key.source)
        for op in self._vevs:
            ops.setdefault(op)
        self._operators = tuple(ops)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_collection(
        cls, collection: Iterable[SigmondData], strict: bool = False
    ) -> CorrelatorIndex:
        """Index the correlator and VEV observables of ``collection``.

        Unrelated observables are skipped with ``strict=False`` and rejected
        with ``strict=True``. Key-like names that do not parse are listed in
        :attr:`unparsed` with a warning (``strict=True`` raises instead). Two
        observables with the same semantic key always raise.

        All indexed observables must share one ensemble, sampling plan, kind
        (bins or samplings) and sample length; make separate indices for
        separate ensembles. Equal ``SamplingInfo`` alone does not prove that
        two independently generated bootstrap files used the same draws.
        """
        correlators: dict[CorrelatorAtTimeInfo, SigmondData] = {}
        vevs: dict[Operator, SigmondData] = {}
        names: dict[object, str] = {}
        unparsed: list[tuple[str, str]] = []
        unrelated: list[str] = []
        first = None
        for obs in collection:
            obs = obs.with_observable_info(obs.observable_info.copy())
            info = obs.observable_info
            try:
                key = key_of(info)
            except ValueError as e:
                unparsed.append((info.name, str(e)))
                continue
            if key is None:
                error = getattr(info, "key_error", None)
                if error is not None:
                    unparsed.append((info.name, str(error)))
                else:
                    unrelated.append(str(info.name))
                continue
            if getattr(obs, "ndim", 0):
                raise ValueError(f"an index needs single observables, got an N-D array for {key}")
            label = key if isinstance(key, CorrelatorAtTimeInfo) else key.operator
            if label in names:
                raise ValueError(
                    f"duplicate semantic key {key}:\n  {names[label]!r}\n  {info.name!r}"
                )
            names[label] = info.name
            if first is None:
                first = obs
            else:
                _check_same_plan(first, obs)
            if isinstance(key, CorrelatorAtTimeInfo):
                correlators[key] = obs
            else:
                vevs[key.operator] = obs

        if strict and (unparsed or unrelated):
            detail = [n for n, _ in unparsed[:5]] + unrelated[:5]
            raise ValueError(
                f"strict index: {len(unparsed)} unparsed and {len(unrelated)} unrelated "
                f"observable(s), e.g. {detail}"
            )
        if unparsed:
            name, reason = unparsed[0]
            warnings.warn(
                f"{len(unparsed)} correlator/VEV key(s) did not parse and are not indexed "
                f"(see CorrelatorIndex.unparsed); first: {name!r}: {reason}",
                stacklevel=2,
            )

        return cls(
            correlators,
            vevs,
            ensemble_info=None if first is None else first.observable_info.ensemble_info,
            sampling_info=None if first is None else first.sampling_info,
            is_bins=first is not None and first.sampling_info is None,
            sample_length=None if first is None else first._sample_length,
            unparsed=unparsed,
        )

    # ------------------------------------------------------------------
    # Metadata queries (never read samples)
    # ------------------------------------------------------------------

    @property
    def operators(self) -> tuple[Operator, ...]:
        """Every operator seen as a sink, source or VEV, in first-seen order."""
        return self._operators

    @property
    def channels(self) -> dict[Channel, tuple[Operator, ...]]:
        """The operators grouped by channel (candidate matrix bases), in first-seen order."""
        out: dict[Channel, list[Operator]] = {}
        for op in self._operators:
            if not op.is_vacuum:
                out.setdefault(op.channel, []).append(op)
        return {ch: tuple(ops) for ch, ops in out.items()}

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(sorted({key.time for key in self._by_key}))

    @property
    def correlator_infos(self) -> tuple[CorrelatorInfo, ...]:
        return tuple(dict.fromkeys(key.correlator for key in self._by_key))

    @property
    def vev_operators(self) -> tuple[Operator, ...]:
        return tuple(self._vevs)

    def keys(self) -> tuple[CorrelatorAtTimeInfo | VEVInfo, ...]:
        return (*self._by_key, *(VEVInfo(op) for op in self._vevs))

    def __len__(self) -> int:
        """Stored observables (one per correlator at each time, plus VEVs), matching :meth:`keys`."""
        return len(self._by_key) + len(self._vevs)

    def __contains__(self, key) -> bool:
        if isinstance(key, VEVInfo):
            return key.operator in self._vevs
        return key in self._by_key

    def get(self, key: CorrelatorAtTimeInfo | VEVInfo) -> SigmondData:
        """The stored observable view for ``key``, with snapshot metadata."""
        if isinstance(key, VEVInfo):
            try:
                return self._vevs[key.operator]
            except KeyError:
                raise KeyError(f"no VEV for {key.operator.coordinate}") from None
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"no correlator {key}") from None

    def series(self) -> dict[tuple[CorrelatorInfo, bool, bool], tuple[int, ...]]:
        """Every stored series, ``(correlator, hermitian, subtract_vev) -> times``.

        In first-seen order.
        """
        return {form: tuple(sorted(by_time)) for form, by_time in self._series.items()}

    def series_times(
        self, correlator: CorrelatorInfo, hermitian: bool = False, subtract_vev: bool = False
    ) -> tuple[int, ...]:
        """Times stored for one correlator and key form."""
        return tuple(sorted(self._series.get((correlator, hermitian, subtract_vev), {})))

    def _lookup(self, sink, source, hermitian, subtract_vev) -> dict[int, SigmondData]:
        return self._series.get((CorrelatorInfo(sink, source), hermitian, subtract_vev), {})

    def _vev(self, op: Operator):
        return self._vevs.get(op)

    def __repr__(self) -> str:
        kind = "bins" if self.is_bins else repr(self.sampling_info)
        extra = f", {len(self.unparsed)} unparsed" if self.unparsed else ""
        return (
            f"CorrelatorIndex({len(self.correlator_infos)} correlators, {len(self._vevs)} VEVs, "
            f"{len(self._operators)} operators, {len(self.channels)} channels, "
            f"times={_span(self.times)}, {kind}{extra})"
        )

    # ------------------------------------------------------------------
    # Data requests
    # ------------------------------------------------------------------

    def matrix(
        self,
        basis: Channel | CorrelatorMatrixInfo | Sequence[Operator | str],
        times: Iterable[int] | None = None,
        *,
        hermitian: bool | None = None,
        subtract_vev: bool | None = None,
        key_form: KeyForm | None = None,
        sampling_info: SamplingInfo | None = None,
        include_vevs: bool = False,
    ) -> CorrelatorMatrix:
        """Assemble the complete matrix of ``basis`` over ``times``.

        ``basis`` is a :class:`Channel`, :class:`CorrelatorMatrixInfo` or a sequence of operators
        (``Operator`` objects or short strings), kept in the given order.
        ``hermitian`` (default ``True``) and ``subtract_vev`` (default
        ``False``) override the rules of a ``CorrelatorMatrixInfo``. See
        :mod:`sigmondsamplings.correlators.matrix` for the assembly contract;
        missing cells and times raise, nothing is zero-filled.
        """
        from .matrix import assemble

        info = _matrix_info(self._basis(basis), hermitian, subtract_vev)
        return assemble(self, info, times, key_form, sampling_info, include_vevs)

    def coverage(
        self,
        basis: Channel | CorrelatorMatrixInfo | Sequence[Operator | str],
        *,
        hermitian: bool | None = None,
        subtract_vev: bool | None = None,
        key_form: KeyForm | None = None,
    ) -> xr.DataArray:
        """Which cells of the ``basis`` matrix are available, per time. Reads no samples.

        ``basis`` is a :class:`Channel` (its operators, in first-seen order), a
        :class:`CorrelatorMatrixInfo` or a sequence of operators. Returns a
        boolean ``(time, sink, source)`` ``xarray.DataArray`` over every time
        any cell is stored, true where :meth:`matrix` with the same flags would
        find the cell: the key form is chosen by the same rules (and the same
        ambiguity error), and a Hermitian request counts either direction.
        ``attrs["operators"]`` holds the basis, ``attrs["key_form"]`` the form.
        """
        from .matrix import coverage

        resolved = self._basis(basis)
        if not isinstance(resolved, CorrelatorMatrixInfo):
            ops = tuple(parse_operator(op) for op in resolved)
            if len({op.backwards for op in ops}) > 1:
                if len({op.channel for op in ops}) > 1:
                    raise ValueError("basis operators span several channels")
                for direction in (False, True):
                    _matrix_info(
                        [op for op in ops if op.backwards == direction], hermitian, subtract_vev
                    )
                return coverage(
                    self,
                    ops,
                    True if hermitian is None else hermitian,
                    False if subtract_vev is None else subtract_vev,
                    key_form,
                )
        info = _matrix_info(resolved, hermitian, subtract_vev)
        return coverage(self, info.operators, info.hermitian, info.subtract_vev, key_form)

    def available_bases(
        self,
        basis: Channel | CorrelatorMatrixInfo | Sequence[Operator | str],
        times: Iterable[int] | None = None,
        *,
        hermitian: bool | None = None,
        subtract_vev: bool | None = None,
        key_form: KeyForm | None = None,
    ) -> list[tuple[Operator, ...]]:
        """The maximal sub-bases of ``basis`` whose matrix is fully stored, largest first.

        Available means the metadata provide usable records at every one of
        ``times``; bins still need a resampling plan for connected requests,
        and runtime read failures are not detected. ``times`` defaults to every
        time any cell of ``basis`` is stored. Exact
        for up to 16 operators, greedy above; see
        :func:`~sigmondsamplings.correlators.matrix.available_bases`. Reads no
        samples. An empty list means no operator is available on its own.
        """
        from .matrix import available_bases

        cov = self.coverage(
            basis, hermitian=hermitian, subtract_vev=subtract_vev, key_form=key_form
        )
        return available_bases(cov, times)

    def _basis(self, basis):
        """A channel's operators; anything else unchanged."""
        if isinstance(basis, Channel):
            try:
                return self.channels[basis]
            except KeyError:
                raise KeyError(f"channel {basis} is not in the index") from None
        return basis

    def correlator(
        self,
        sink: Operator | str | CorrelatorInfo,
        source: Operator | str | None = None,
        times: Iterable[int] | None = None,
        *,
        hermitian: bool | None = None,
        subtract_vev: bool = False,
        key_form: KeyForm | None = None,
        sampling_info: SamplingInfo | None = None,
    ) -> SigmondData:
        """One correlator ``C(sink, source)`` as a 1-D ``SigmondData`` over ``time``.

        Pass the two operators (objects or short strings) or one
        :class:`CorrelatorInfo`; omitting ``source`` requests the diagonal.
        ``hermitian=None`` follows the unique stored form, raising on ambiguity.
        The flags choose the stored key form or derive
        it by the same rules as a matrix (``hermitian=True`` may use the flipped
        correlator; ``subtract_vev=True`` may subtract VEVs from raw data).
        """
        from .matrix import assemble_correlator
        
        if isinstance(sink, CorrelatorInfo):
            if source is not None:
                raise TypeError("pass a CorrelatorInfo or two operators, not both")
            corr = sink
        else:
            corr = CorrelatorInfo(
                parse_operator(sink), parse_operator(sink if source is None else source)
            )
        if hermitian is None:
            from .matrix import _choose_form, _Request

            ops = (corr.sink,) if corr.sink == corr.source else (corr.sink, corr.source)
            cell = (0, len(ops) - 1)
            req = _Request(self, ops, [cell], True)
            hermitian = _choose_form(req, True, subtract_vev, key_form)[0]
        return assemble_correlator(
            self, corr, times, hermitian, subtract_vev, key_form, sampling_info
        )


def _matrix_info(basis, hermitian, subtract_vev) -> CorrelatorMatrixInfo:
    if isinstance(basis, CorrelatorMatrixInfo):
        info = basis
        if hermitian is not None or subtract_vev is not None:
            info = CorrelatorMatrixInfo(
                info.operators,
                info.hermitian if hermitian is None else hermitian,
                info.subtract_vev if subtract_vev is None else subtract_vev,
            )
        return info
    if isinstance(basis, (str, BasicLapHOperator, GenIrrepOperator)):
        raise TypeError("basis must be a sequence of operators or a CorrelatorMatrixInfo")
    return CorrelatorMatrixInfo(
        tuple(basis),
        True if hermitian is None else hermitian,
        False if subtract_vev is None else subtract_vev,
    )


def _check_same_plan(first, obs) -> None:
    reasons = []
    if obs.observable_info.ensemble_info != first.observable_info.ensemble_info:
        reasons.append("ensemble")
    if (obs.sampling_info is None) != (first.sampling_info is None):
        reasons.append("bins vs samplings")
    elif obs.sampling_info != first.sampling_info:
        reasons.append("sampling plan")
    elif obs._sample_length != first._sample_length:
        reasons.append("sample length")
    if reasons:
        raise ValueError(
            f"{obs.observable_info.name!r} differs from {first.observable_info.name!r} in "
            f"{', '.join(reasons)}; build one index per ensemble and sampling plan"
        )


def _span(times: Sequence[int]) -> str:
    if not times:
        return "[]"
    if list(times) == list(range(times[0], times[-1] + 1)):
        return f"{times[0]}..{times[-1]}"
    return str(list(times))
