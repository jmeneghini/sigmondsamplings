"""Effective energies of temporal correlators, as Sigmond computes them.

:func:`effective_energy` maps any real ``SigmondData`` with a ``time`` dim to
its effective energy, sample by sample, with the full sample included. Other
dims broadcast, so one call covers a single correlator, the diagonal of a
matrix ``(time, operator)`` or a fit's theory curve.

The four kinds are Sigmond's ``EffectiveEnergyCalculator`` types; see
``docs/correlator_views_design.md`` §3.1 for their models and conventions.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from ..data import SigmondData
from ..metadata import AtObsInfo, EffectiveEnergyInfo, source_elements
from ..sampling_array import ArrayObsInfo
from . import _kernels
from .operators import _int

__all__ = [
    "KINDS",
    "EffectiveEnergyInfo",
    "EffectiveEnergyWarning",
    "effective_energy",
    "effective_time_offset",
]

Kind = Literal["forward", "symmetric", "forward_const", "symmetric_const"]
FailurePolicy = Literal["drop", "nan", "raise"]

TIME = "time"

#: Kind names and Sigmond's type codes.
KINDS: dict[str, int] = {
    "forward": _kernels.FORWARD,
    "symmetric": _kernels.SYMMETRIC,
    "forward_const": _kernels.FORWARD_CONST,
    "symmetric_const": _kernels.SYMMETRIC_CONST,
}

_ON_FAIL = ("drop", "nan", "raise")


class EffectiveEnergyWarning(UserWarning):
    """Times were dropped because the effective energy failed for some sample."""


def _check_kind(kind: str) -> int:
    try:
        return KINDS[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {sorted(KINDS)}, got {kind!r}") from None


def effective_time_offset(kind: str, step: int) -> float:
    """Where Sigmond plots an effective energy keyed at source time ``t``, minus ``t``.

    ``step/2`` for the plain kinds, which use ``C(t)`` and ``C(t+step)``; ``0``
    for the added-constant kinds, which are centred on ``t``.
    """
    return step / 2 if _check_kind(kind) < _kernels.FORWARD_CONST else 0.0


def _integer_times(values) -> np.ndarray:
    times = np.asarray(values)
    try:
        as_int = np.array([_int(t, "time coordinate") for t in times], dtype=int)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("time coords must be integers") from None
    if len(set(as_int.tolist())) != as_int.size:
        raise ValueError("time coords must be unique")
    return as_int


def effective_energy(
    corr: SigmondData,
    *,
    kind: Kind = "forward",
    step: int = 1,
    time_extent: int | None = None,
    subtract_const: float = 0.0,
    on_fail: FailurePolicy = "drop",
) -> SigmondData:
    """Effective energy of every correlator in ``corr``, per sample.

    Args:
        corr: Real samplings with a ``time`` dim whose coords are integers.
            Other dims broadcast; each of their cells is a separate correlator.
        kind: ``"forward"`` (``A e^{-mt}``), ``"symmetric"`` (a cosh),
            ``"forward_const"`` or ``"symmetric_const"`` (the same plus a
            constant). Sigmond's types 0 to 3.
        step: Solve from ``C(t)`` and ``C(t+step)`` (and ``C(t-step)`` for the
            added-constant kinds).
        time_extent: Lattice extent ``T``; defaults to
            ``corr.ensemble_info.temporal_extent``. Required only for symmetric
            kinds; forward kinds use only step and available neighbours.
        subtract_const: Removed from every correlator value first, like
            Sigmond's ``subtract_const``.
        on_fail: What to do at a time where some sample fails.
            ``"drop"`` (Sigmond's rule) removes the time and warns naming it,
            ``"nan"`` keeps it with NaN in the failed samples (what plots use),
            ``"raise"`` raises and names the times.

    Returns:
        A ``SigmondData`` with the same dims. Its ``time`` coord is the source
        time ``t`` of each value (see :func:`effective_time_offset` for where
        Sigmond draws it), and it holds only times whose inputs are all
        present. Each cell carries an :class:`AtObsInfo` effective-energy element.
        The central value is the effective energy of the full-sample
        correlator, not a mean over resamples.

    Raises:
        TypeError: ``corr`` is not ``SigmondData`` or holds raw bins.
        ValueError: Complex data, no ``time`` dim, invalid arguments, or no
            time left with an effective energy.
    """
    code = _check_kind(kind)
    if on_fail not in _ON_FAIL:
        raise ValueError(f"on_fail must be one of {_ON_FAIL}, got {on_fail!r}")
    if not isinstance(corr, SigmondData):
        raise TypeError(f"effective_energy expects SigmondData, got {type(corr).__name__}")
    if TIME not in corr.dims:
        raise ValueError(f"effective_energy needs a {TIME!r} dim, got dims {corr.dims}")
    if corr.sampling_info is None:
        raise TypeError(
            "effective_energy needs samplings, not raw bins: the map is nonlinear, "
            "so resample the bins first"
        )
    if corr.is_complex:
        raise ValueError(
            "effective_energy needs real data; project explicitly first, e.g. .to_real() "
            "or CorrelatorMatrix.diagonal(component='real')"
        )

    T = corr.ensemble_info.temporal_extent if time_extent is None else time_extent
    step = _int(step, "step")
    if step < 1:
        raise ValueError("step must be positive")
    if T is not None:
        T = _int(T, "time_extent")
        if T < 1:
            raise ValueError("time_extent must be positive")
    if code in (_kernels.SYMMETRIC, _kernels.SYMMETRIC_CONST):
        if T is None:
            raise ValueError("the ensemble has no temporal extent; pass time_extent= explicitly")
        if T < 8 or step > T // 4:
            raise ValueError(
                "symmetric energies need time_extent >= 8 and step <= time_extent // 4"
            )

    times = _integer_times(corr.coords[TIME]) if TIME in corr.coords else None
    if times is None:
        raise ValueError(f"the {TIME!r} dim needs integer coords")

    positions = {int(t): i for i, t in enumerate(times)}
    candidates = np.array(
        [
            i
            for i, t in enumerate(times)
            if int(t) + step in positions
            and (code < _kernels.FORWARD_CONST or int(t) - step in positions)
        ],
        dtype=int,
    )
    candidates = candidates[np.argsort(times[candidates])]
    if not candidates.size:
        raise ValueError(
            f"no time has a {kind} effective energy (step {step}); the correlator has times {list(times)}"
        )
    axis = corr.dims.index(TIME)
    elements = source_elements(corr)
    if not np.isfinite(subtract_const):
        raise ValueError("subtract_const must be finite")
    raw = np.moveaxis(np.asarray(corr._as_numpy(), dtype=float), axis, -2) - subtract_const
    energy, failed = _kernels.effective_energy(raw, times, kind=code, step=step, time_extent=T or 0)
    failing = failed[..., candidates, :].any(axis=-1)
    failing = failing.reshape(-1, candidates.size).any(axis=0)
    bad_times = [int(t) for t in times[candidates[failing]]]
    if bad_times and on_fail == "raise":
        raise ValueError(
            f"the {kind} effective energy (step {step}) fails for some sample at times {bad_times}"
        )
    if on_fail == "drop":
        if bad_times:
            warnings.warn(
                f"dropped times {bad_times}: the {kind} effective energy (step {step}) "
                "fails there for some sample",
                EffectiveEnergyWarning,
                stacklevel=2,
            )
        keep = candidates[~failing]
    else:
        keep = candidates
    if keep.size == 0:
        raise ValueError(
            f"no time has a {kind} effective energy (step {step}); "
            f"the correlator has times {list(times)}"
        )

    out = np.moveaxis(energy[..., keep, :], -2, axis)
    coords = {dim: corr.coords[dim] for dim in corr.dims if dim in corr.coords}
    coords[TIME] = times[keep]
    info = ArrayObsInfo("effective_energy", ensemble_info=corr.ensemble_info)
    infos = np.empty(out.shape[:-1], dtype=object)
    along = np.moveaxis(infos, axis, -1)
    for index in np.ndindex(elements.shape):
        element = EffectiveEnergyInfo(elements[index], kind, step, T, subtract_const)
        for position, time in enumerate(times[keep]):
            along[index + (position,)] = AtObsInfo(element, "time", time, ensemble_info=corr.ensemble_info)
    return SigmondData.from_data(
        out, info, corr.sampling_info, dim_names=corr.dims, coords=coords,
        ensemble_info=corr.ensemble_info, infos=infos,
    )
