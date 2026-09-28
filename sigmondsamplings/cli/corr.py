"""Correlator views for ``ss query corr``.

Record builders over a :class:`~sigmondsamplings.correlators.CorrelatorIndex`.
Listings (series, operators, channels, VEVs) read only metadata; the correlator
and matrix views read just the selected cells and times.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sigmondsamplings.io.loader import SigmondLoader, is_hdf5_file

__all__ = [
    "channel_records",
    "correlator_records",
    "load_index",
    "matrix_records",
    "operator_records",
    "parse_operator_numbers",
    "parse_sampling",
    "resolve_channel",
    "resolve_hermitian",
    "resolve_operator",
    "series_records",
    "vev_records",
]


def load_index(filename: str | Path, *, group: str | None = None):
    """The correlator index of a Sigmond file (lazy for HDF5)."""
    filename = str(filename)
    loader = SigmondLoader(filename, group=group, lazy=is_hdf5_file(filename))
    return loader.correlators


# ----------------------------------------------------------------------
# Metadata listings
# ----------------------------------------------------------------------


def series_records(index) -> list[dict[str, Any]]:
    """One record per stored correlator series (pair and key form)."""
    label = _op_labels(index)
    return [
        {
            "sink": label[pair.sink],
            "source": label[pair.source],
            "hermitian": hermitian,
            "subtract_vev": subtract_vev,
            "times": _span(times),
            "n_times": len(times),
        }
        for (pair, hermitian, subtract_vev), times in index.series().items()
    ]


def operator_records(index) -> list[dict[str, Any]]:
    """One record per operator, numbered for use with ``--sink``/``--source``."""
    vevs = set(index.vev_operators)
    return [
        {
            "#": i,
            "operator": str(op) or "vacuum",
            "kind": op.coordinate[:2],
            "channel": "" if op.is_vacuum else str(op.channel),
            "vev": op in vevs,
        }
        for i, op in enumerate(index.operators)
    ]


def channel_records(index) -> list[dict[str, Any]]:
    """One record per channel (candidate matrix basis), numbered for ``--matrix``."""
    return [
        {"#": i, "channel": str(channel), "n_operators": len(ops)}
        for i, (channel, ops) in enumerate(index.channels.items())
    ]


def vev_records(index) -> list[dict[str, Any]]:
    """One record per stored VEV, with its value."""
    from sigmondsamplings.correlators import VEVKey

    label = _op_labels(index)
    return [
        {"operator": label[op], **_value_fields(index.get(VEVKey(op)))}
        for op in index.vev_operators
    ]


# ----------------------------------------------------------------------
# Data views
# ----------------------------------------------------------------------


def correlator_records(
    index,
    sink: str,
    source: str,
    *,
    times: Sequence[int] | None = None,
    hermitian: bool | None = None,
    subtract_vev: bool = False,
    sampling_info=None,
) -> list[dict[str, Any]]:
    """``C(sink, source)`` over time, one record per time.

    ``hermitian=None`` follows the file (see :func:`resolve_hermitian`).
    """
    sink_op, source_op = resolve_operator(index, sink), resolve_operator(index, source)
    if hermitian is None:
        hermitian = resolve_hermitian(index, sink_op, source_op)
    corr = index.correlator(
        sink_op,
        source_op,
        times,
        hermitian=hermitian,
        subtract_vev=subtract_vev,
        sampling_info=sampling_info,
    )
    return [
        {"time": int(t), **_value_fields(corr.isel(time=k))}
        for k, t in enumerate(corr.coords["time"])
    ]


def matrix_records(
    index,
    channel: str,
    time: int | None = None,
    *,
    hermitian: bool = True,
    subtract_vev: bool = False,
    sampling_info=None,
    operators: Sequence[int] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """The correlation matrix of one channel at one time (default: the first stored).

    ``operators`` (numbers from ``--operators``, in the order given) restricts
    the basis to part of the channel. Returns ``(time, records)`` with one
    record per sink operator and one column per source operator (numbered as
    in the row's ``#``).
    """
    resolved = resolve_channel(index, channel)
    ops = index.channels[resolved]
    if operators is not None:
        ops = _channel_subset(index, resolved, ops, operators)
    if time is None:
        if not index.times:
            raise ValueError("no correlators in the file")
        time = index.times[0]
    try:
        matrix = index.matrix(
            ops,
            [time],
            hermitian=hermitian,
            subtract_vev=subtract_vev,
            sampling_info=sampling_info,
        )
    except ValueError as exc:
        if "not every requested cell" not in str(exc):
            raise
        raise ValueError(f"{exc}\n{_subset_hint(index, ops, time, hermitian)}") from exc
    values = matrix.at(time)
    records = []
    for i, op in enumerate(matrix.operators):
        row: dict[str, Any] = {"#": i, "sink": str(op)}
        for j in range(len(matrix.operators)):
            row[str(j)] = values.isel(sink=i, source=j).pdg_format()
        records.append(row)
    return time, records


# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------


def resolve_operator(index, text: str):
    """An operator by its ``--operators`` number or its short string."""
    from sigmondsamplings.correlators import parse_operator

    ops = index.operators
    if text.strip().isdigit():
        i = int(text)
        if i >= len(ops):
            raise ValueError(f"operator #{i} out of range (file has {len(ops)} operators)")
        return ops[i]
    op = parse_operator(text)
    if op not in ops:
        raise ValueError(f"operator {text!r} is not in the file; see --operators")
    return op


def parse_operator_numbers(text: str) -> list[int]:
    """``--ops`` text such as ``0-6,8`` as operator numbers, in order."""
    numbers: list[int] = []
    for part in text.split(","):
        lo, sep, hi = part.strip().partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"--ops takes operator numbers and ranges like '0-6,8', got {text!r}")
        numbers += range(int(lo), int(hi) + 1) if sep else [int(lo)]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"--ops repeats an operator: {text!r}")
    return numbers


def resolve_hermitian(index, sink, source) -> bool:
    """Whether a ``(sink, source)`` request should be Hermitian, following the file.

    Non-Hermitian when that direction is stored without ``HermMat``; Hermitian
    when only ``HermMat`` keys (either direction) hold the pair.
    """
    from sigmondsamplings.correlators import CorrelatorPair

    direct, flipped = CorrelatorPair(sink, source), CorrelatorPair(source, sink)
    forms = {(pair, herm) for pair, herm, _ in index.series()}
    if (direct, False) in forms:
        return False
    return (direct, True) in forms or (flipped, True) in forms


def parse_sampling(index, text: str):
    """``jackknife`` or ``bootstrap:N[:SEED]`` as a :class:`SamplingInfo` for resampling bins.

    Jackknife uses one resample per stored bin.
    """
    from sigmondsamplings.info import SamplingInfo

    if not index.is_bins:
        raise ValueError("--sampling only applies to bins files; samplings keep their own plan")
    method, *params = text.strip().lower().split(":")
    if method in ("jack", "jackknife") and not params:
        return SamplingInfo("jackknife", index.sample_length)
    if method in ("boot", "bootstrap") and 1 <= len(params) <= 2:
        try:
            nums = [int(p) for p in params]
        except ValueError:
            nums = []
        if nums and nums[0] > 0:
            return SamplingInfo("bootstrap", *nums)
    raise ValueError(f"--sampling must be 'jackknife' or 'bootstrap:N[:SEED]', got {text!r}")


def resolve_channel(index, text: str):
    """A channel by its ``--channels`` number or a unique substring of its label."""
    channels = list(index.channels)
    if not channels:
        raise ValueError("no channels in the file")
    if text.strip().isdigit():
        i = int(text)
        if i >= len(channels):
            raise ValueError(f"channel #{i} out of range (file has {len(channels)} channels)")
        return channels[i]
    matches = [ch for ch in channels if text in str(ch)]
    if len(matches) != 1:
        found = "no" if not matches else f"{len(matches)}"
        raise ValueError(f"{found} channels match {text!r}; see --channels")
    return matches[0]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _channel_subset(index, channel, channel_ops, numbers: Sequence[int]) -> list:
    ops = index.operators
    subset = []
    for i in numbers:
        if i >= len(ops):
            raise ValueError(f"operator #{i} out of range (file has {len(ops)} operators)")
        if ops[i] not in channel_ops:
            raise ValueError(f"operator #{i} is not in channel {channel}")
        subset.append(ops[i])
    return subset


def _subset_hint(index, ops, time: int, hermitian: bool) -> str:
    """Suggest ``--ops`` for a complete sub-basis: greedily drop the operator with most holes.

    Any stored key form counts, so the suggestion is a hint, not a guarantee.
    """
    from sigmondsamplings.correlators import CorrelatorPair

    stored = {pair for (pair, _, _), times in index.series().items() if time in times}

    def has(a, b) -> bool:
        return CorrelatorPair(a, b) in stored or (hermitian and CorrelatorPair(b, a) in stored)

    keep = list(ops)
    while keep:
        holes = {op: sum(not has(op, o) or not has(o, op) for o in keep) for op in keep}
        worst = max(keep, key=holes.get)
        if not holes[worst]:
            break
        keep.remove(worst)
    if len(keep) < 2:
        return "No complete sub-basis of two or more operators is stored; pick one with --ops."
    numbers = [index.operators.index(op) for op in keep]
    return f"Pick a complete sub-basis with --ops, e.g. --ops {_ranges(numbers)}"


def _ranges(numbers: Sequence[int]) -> str:
    parts, start = [], None
    for k, n in enumerate(numbers):
        if start is None:
            start = n
        if k + 1 == len(numbers) or numbers[k + 1] != n + 1:
            parts.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ",".join(parts)


def _value_fields(obs) -> dict[str, Any]:
    return {"value": obs.pdg_format(), "full_sample_value": obs.full_sample_value, "error": obs.error}


def _op_labels(index) -> dict:
    """``"#: operator"`` labels, numbered as in ``--operators``."""
    return {op: f"{i}: {str(op) or 'vacuum'}" for i, op in enumerate(index.operators)}


def _span(times: Sequence[int]) -> str:
    if not times:
        return ""
    if list(times) == list(range(times[0], times[-1] + 1)):
        return f"{times[0]}..{times[-1]}"
    return ",".join(str(t) for t in times)
