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
from .keys import CorrelatorKey, CorrelatorMatrixSpec, CorrelatorPair, VEVKey
from .observables import key_of
from .operators import BasicOperator, Channel, GeneralOperator, Operator, parse_operator

if TYPE_CHECKING:
    from ..data import SigmondData
    from .matrix import CorrelatorMatrix

__all__ = ["CorrelatorIndex"]


def _sample_length(obs) -> int:
    """Sample-axis length without reading a lazy record's data."""
    record = getattr(obs, "_record", None)
    if record is not None:
        return record.sample_length
    return int(obs.data.shape[-1])


class CorrelatorIndex:
    """Correlators and VEVs of one ensemble and one sampling plan, by typed key.

    Build with :meth:`from_collection` (or ``SigmondLoader(...).correlators``).
    The index holds the original observables as values (never copies), keyed
    by immutable parsed labels.
    """

    def __init__(
        self,
        correlators: dict[CorrelatorKey, SigmondData],
        vevs: dict[Operator, SigmondData],
        *,
        ensemble_info: EnsembleInfo | None,
        sampling_info: SamplingInfo | None,
        is_bins: bool,
        sample_length: int | None,
        unparsed: Sequence[tuple[str, str]] = (),
    ):
        self._correlators = dict(correlators)
        self._vevs = dict(vevs)
        self.ensemble_info = ensemble_info
        self.sampling_info = sampling_info
        self.is_bins = is_bins
        self.sample_length = sample_length
        self.unparsed = tuple(unparsed)
        """``(name, reason)`` for key-like names that did not parse (they are not indexed)."""

        # (pair, hermitian, subtract_vev) -> {time: observable}
        self._series: dict[tuple, dict[int, SigmondData]] = {}
        ops: dict[Operator, None] = {}
        for key, obs in self._correlators.items():
            self._series.setdefault((key.pair, key.hermitian, key.subtract_vev), {})[key.time] = obs
            ops.setdefault(key.sink)
            ops.setdefault(key.source)
        for op in self._vevs:
            ops.setdefault(op)
        self._operators = tuple(ops)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_collection(cls, collection: Iterable[SigmondData], strict: bool = False):
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
        correlators: dict[CorrelatorKey, SigmondData] = {}
        vevs: dict[Operator, SigmondData] = {}
        names: dict[object, str] = {}
        unparsed: list[tuple[str, str]] = []
        unrelated: list[str] = []
        first = None
        for obs in collection:
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
            label = key if isinstance(key, CorrelatorKey) else key.operator
            if label in names:
                raise ValueError(
                    f"duplicate semantic key {key}:\n  {names[label]!r}\n  {info.name!r}"
                )
            names[label] = info.name
            if first is None:
                first = obs
            else:
                _check_same_plan(first, obs)
            if isinstance(key, CorrelatorKey):
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
            sample_length=None if first is None else _sample_length(first),
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
        return tuple(sorted({key.time for key in self._correlators}))

    @property
    def pairs(self) -> tuple[CorrelatorPair, ...]:
        return tuple(dict.fromkeys(key.pair for key in self._correlators))

    @property
    def vev_operators(self) -> tuple[Operator, ...]:
        return tuple(self._vevs)

    def keys(self) -> tuple[CorrelatorKey | VEVKey, ...]:
        return (*self._correlators, *(VEVKey(op) for op in self._vevs))

    def __len__(self) -> int:
        return len(self._correlators) + len(self._vevs)

    def __contains__(self, key) -> bool:
        if isinstance(key, VEVKey):
            return key.operator in self._vevs
        return key in self._correlators

    def get(self, key: CorrelatorKey | VEVKey) -> SigmondData:
        """The stored observable for ``key`` (the original object)."""
        if isinstance(key, VEVKey):
            try:
                return self._vevs[key.operator]
            except KeyError:
                raise KeyError(f"no VEV for {key.operator.coordinate}") from None
        try:
            return self._correlators[key]
        except KeyError:
            raise KeyError(f"no correlator {key}") from None

    def series(self) -> dict[tuple[CorrelatorPair, bool, bool], tuple[int, ...]]:
        """Every stored series, ``(pair, hermitian, subtract_vev) -> times``, in first-seen order."""
        return {form: tuple(sorted(by_time)) for form, by_time in self._series.items()}

    def series_times(
        self, pair: CorrelatorPair, hermitian: bool = False, subtract_vev: bool = False
    ) -> tuple[int, ...]:
        """Times stored for one pair and key form."""
        return tuple(sorted(self._series.get((pair, hermitian, subtract_vev), {})))

    def _lookup(self, sink, source, hermitian, subtract_vev) -> dict[int, SigmondData]:
        return self._series.get((CorrelatorPair(sink, source), hermitian, subtract_vev), {})

    def _vev(self, op: Operator):
        return self._vevs.get(op)

    def __repr__(self) -> str:
        kind = "bins" if self.is_bins else repr(self.sampling_info)
        extra = f", {len(self.unparsed)} unparsed" if self.unparsed else ""
        return (
            f"CorrelatorIndex({len(self._correlators)} correlators, {len(self._vevs)} VEVs, "
            f"{len(self._operators)} operators, {len(self.channels)} channels, "
            f"times={_span(self.times)}, {kind}{extra})"
        )

    # ------------------------------------------------------------------
    # Data requests
    # ------------------------------------------------------------------

    def matrix(
        self,
        basis: CorrelatorMatrixSpec | Sequence[Operator | str],
        times: Iterable[int] | None = None,
        *,
        hermitian: bool | None = None,
        subtract_vev: bool | None = None,
        key_form: str | None = None,
        sampling_info: SamplingInfo | None = None,
        include_vevs: bool = False,
    ) -> CorrelatorMatrix:
        """Assemble the complete matrix of ``basis`` over ``times``.

        ``basis`` is a :class:`CorrelatorMatrixSpec` or a sequence of operators
        (``Operator`` objects or short strings), kept in the given order.
        ``hermitian`` (default ``True``) and ``subtract_vev`` (default
        ``False``) override a spec's rules. See
        :mod:`sigmondsamplings.correlators.matrix` for the assembly contract;
        missing cells and times raise, nothing is zero-filled.
        """
        from .matrix import assemble

        spec = _spec(basis, hermitian, subtract_vev)
        return assemble(self, spec, times, key_form, sampling_info, include_vevs)

    def correlator(
        self,
        sink: Operator | str | CorrelatorPair,
        source: Operator | str | None = None,
        times: Iterable[int] | None = None,
        *,
        hermitian: bool = False,
        subtract_vev: bool = False,
        key_form: str | None = None,
        sampling_info: SamplingInfo | None = None,
    ) -> SigmondData:
        """One correlator ``C(sink, source)`` as a 1-D ``SigmondData`` over ``time``.

        Pass the two operators (objects or short strings) or one
        :class:`CorrelatorPair`. The flags choose the stored key form or derive
        it by the same rules as a one-pair matrix (``hermitian=True`` may use
        the flipped pair; ``subtract_vev=True`` may subtract VEVs from raw data).
        """
        from .matrix import assemble_pair

        if isinstance(sink, CorrelatorPair):
            if source is not None:
                raise TypeError("pass a CorrelatorPair or two operators, not both")
            pair = sink
        else:
            if source is None:
                raise TypeError("correlator() needs a sink and a source operator")
            pair = CorrelatorPair(parse_operator(sink), parse_operator(source))
        return assemble_pair(
            self, pair, times, hermitian, subtract_vev, key_form, sampling_info
        )


def _spec(basis, hermitian, subtract_vev) -> CorrelatorMatrixSpec:
    if isinstance(basis, CorrelatorMatrixSpec):
        spec = basis
        if hermitian is not None or subtract_vev is not None:
            spec = CorrelatorMatrixSpec(
                spec.operators,
                spec.hermitian if hermitian is None else hermitian,
                spec.subtract_vev if subtract_vev is None else subtract_vev,
            )
        return spec
    if isinstance(basis, (str, BasicOperator, GeneralOperator)):
        raise TypeError("basis must be a sequence of operators or a CorrelatorMatrixSpec")
    return CorrelatorMatrixSpec(
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
    elif _sample_length(obs) != _sample_length(first):
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
