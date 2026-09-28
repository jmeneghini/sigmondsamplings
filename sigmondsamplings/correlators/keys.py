"""
Typed keys for correlators, VEVs and correlation-matrix bases.

- :class:`CorrelatorPair` -- (sink, source) operators, Sigmond's ``CorrelatorInfo``.
- :class:`CorrelatorKey` -- a pair at one time with the ``HermMat`` / ``SubVEV``
  key flags, Sigmond's ``CorrelatorAtTimeInfo``.
- :class:`VEVKey` -- the vacuum expectation value of one operator.
- :class:`CorrelatorMatrixSpec` -- an ordered, validated operator basis.

Short forms print the **sink first** (``<CorrT>BL{sink} GI{source} time=5
HermMat SubVEV</CorrT>``). Flags are written in Sigmond's required order
(``time=<n> HermMat SubVEV``); the reader accepts either order. The flags are
key conventions, not statements that a stored dataset was checked numerically.

Only the short forms are handled: they are what Sigmond writes as HDF5 keys.
This module is pure Python so labels stay cheap to import.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

from .operators import (
    MAX_TIME,
    OPERATOR_TAGS,
    BasicOperator,
    Channel,
    GeneralOperator,
    Operator,
    SigmondParseError,
    _as_element,
    _parse_cached,
    _tokens,
    operator_from_xml,
)

__all__ = [
    "CorrelatorPair",
    "CorrelatorKey",
    "VEVKey",
    "CorrelatorMatrixSpec",
    "parse_key",
]

_OPERATOR_TYPES = (BasicOperator, GeneralOperator)


def _check_operator(op, role: str) -> None:
    if not isinstance(op, _OPERATOR_TYPES):
        raise TypeError(f"{role} must be a BasicOperator or GeneralOperator, got {op!r}")
    if op.is_vacuum:
        raise ValueError(f"the vacuum cannot be a correlator {role}")


# ---------------------------------------------------------------------------
# Brace-aware scanning of short correlator text
# ---------------------------------------------------------------------------

_KINDS = ("BL", "GI")


def _scan_operator(text: str, pos: int) -> tuple[str, str, int, int]:
    """Read ``BL{...}``/``GI{...}`` starting at or after ``pos``.

    Returns ``(kind, body, body_start, next_pos)``. Braces are matched, so the
    body may contain spaces (it never contains braces in valid labels).
    """
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    kind = text[pos : pos + 2]
    if kind not in _KINDS or text[pos + 2 : pos + 3] != "{":
        raise SigmondParseError(text, pos, "expected an operator 'BL{...}' or 'GI{...}'")
    depth, k = 0, pos + 2
    while k < n:
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return kind, text[pos + 3 : k], pos + 3, k + 1
        k += 1
    raise SigmondParseError(text, pos, f"unterminated '{kind}{{'")


def _operator(kind: str, body: str, start: int, text: str) -> Operator:
    try:
        return _parse_cached(kind, body)
    except SigmondParseError as e:
        # Re-anchor the position in the full correlator text.
        raise SigmondParseError(text, start + e.position, e.reason) from None


def _parse_pair_text(text: str) -> tuple[CorrelatorPair, int]:
    kind, body, start, pos = _scan_operator(text, 0)
    sink = _operator(kind, body, start, text)
    kind, body, start, pos = _scan_operator(text, pos)
    source = _operator(kind, body, start, text)
    try:
        return CorrelatorPair(sink, source), pos
    except (TypeError, ValueError) as e:
        raise SigmondParseError(text, 0, str(e)) from None


def _parse_key_text(text: str) -> CorrelatorKey:
    pair, pos = _parse_pair_text(text)
    time = None
    herm = subvev = False
    for tok, tpos in _tokens(text[pos:], offset=pos):
        if tok.startswith("time=") and time is None:
            value = tok[5:]
            if not value.isdigit():
                raise SigmondParseError(text, tpos, f"invalid time {tok!r}")
            time = int(value)
        elif tok == "HermMat" and not herm:
            herm = True
        elif tok == "SubVEV" and not subvev:
            subvev = True
        else:
            raise SigmondParseError(text, tpos, f"unexpected token {tok!r}")
    if time is None:
        raise SigmondParseError(text, len(text), "missing 'time=<n>'")
    return CorrelatorKey(pair, time, herm, subvev)


def _short_text(tag: str, text: str) -> str:
    """The body of ``<tag>...</tag>`` (unescaping XML), or ``text`` if it is not wrapped."""
    stripped = text.strip()
    if not stripped.startswith("<"):
        return stripped
    el = _as_element(stripped)
    if el.tag != tag:
        raise ValueError(f"expected <{tag}>, got <{el.tag}>")
    return el.text or ""


# ---------------------------------------------------------------------------
# Pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelatorPair:
    """A (sink, source) operator pair: the label of ``C_ij(t)`` for all ``t``."""

    sink: Operator
    source: Operator

    def __post_init__(self):
        _check_operator(self.sink, "sink")
        _check_operator(self.source, "source")

    def flipped(self) -> CorrelatorPair:
        """The time-flipped pair ``(source, sink)``, i.e. ``C_ji``."""
        return CorrelatorPair(self.source, self.sink)

    def at(self, time: int, hermitian: bool = False, subtract_vev: bool = False) -> CorrelatorKey:
        return CorrelatorKey(self, time, hermitian, subtract_vev)

    @property
    def backwards(self) -> bool:
        return self.sink.backwards

    def __str__(self) -> str:
        return f"{self.sink.coordinate} {self.source.coordinate}"

    @classmethod
    def from_sigmond(cls, text: str) -> CorrelatorPair:
        """Parse ``<Corr>BL{sink} GI{source}</Corr>`` or the bare text."""
        body = _short_text("Corr", text)
        pair, pos = _parse_pair_text(body)
        if body[pos:].strip():
            raise SigmondParseError(body, pos, "unexpected text after the source operator")
        return pair

    def to_sigmond(self) -> str:
        el = ET.Element("Corr")
        el.text = f"{_coordinate(self.sink)} {_coordinate(self.source)}"
        return ET.tostring(el, encoding="unicode")


def _coordinate(op: Operator) -> str:
    """``BL{...}``/``GI{...}`` after checking Sigmond's encoding limits."""
    op._check_sigmond()
    return op.coordinate


# ---------------------------------------------------------------------------
# Correlator at a time
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelatorKey:
    """``C_ij(t)``: a pair, a nonnegative time, and the stored-key flags."""

    pair: CorrelatorPair
    time: int
    hermitian: bool = False
    subtract_vev: bool = False

    def __post_init__(self):
        if not isinstance(self.pair, CorrelatorPair):
            raise TypeError("pair must be a CorrelatorPair")
        time = self.time
        if isinstance(time, bool) or int(time) != time or time < 0:
            raise ValueError(f"time must be a nonnegative integer, got {time!r}")
        object.__setattr__(self, "time", int(time))
        object.__setattr__(self, "hermitian", bool(self.hermitian))
        object.__setattr__(self, "subtract_vev", bool(self.subtract_vev))

    @classmethod
    def _unchecked(cls, pair, time, hermitian, subtract_vev) -> CorrelatorKey:
        """Build from already-validated parts (used for every cell of a matrix)."""
        key = object.__new__(cls)
        object.__setattr__(key, "pair", pair)
        object.__setattr__(key, "time", time)
        object.__setattr__(key, "hermitian", hermitian)
        object.__setattr__(key, "subtract_vev", subtract_vev)
        return key

    @property
    def sink(self) -> Operator:
        return self.pair.sink

    @property
    def source(self) -> Operator:
        return self.pair.source

    def flipped(self) -> CorrelatorKey:
        return CorrelatorKey(self.pair.flipped(), self.time, self.hermitian, self.subtract_vev)

    def with_time(self, time: int) -> CorrelatorKey:
        return CorrelatorKey(self.pair, time, self.hermitian, self.subtract_vev)

    def __str__(self) -> str:
        text = f"{self.pair} time={self.time}"
        if self.hermitian:
            text += " HermMat"
        if self.subtract_vev:
            text += " SubVEV"
        return text

    @classmethod
    def from_sigmond(cls, text: str) -> CorrelatorKey:
        """Parse ``<CorrT>...</CorrT>`` or the bare short text."""
        return _parse_key_text(_short_text("CorrT", text))

    def to_sigmond(self) -> str:
        """The key as XML text, e.g. ``<CorrT>BL{..} GI{..} time=5 HermMat</CorrT>``."""
        if self.time > MAX_TIME:
            raise ValueError(f"time {self.time} exceeds Sigmond's limit")
        el = ET.Element("CorrT")
        flags = (" HermMat" if self.hermitian else "") + (" SubVEV" if self.subtract_vev else "")
        el.text = f"{_coordinate(self.sink)} {_coordinate(self.source)} time={self.time}{flags}"
        return ET.tostring(el, encoding="unicode")


# ---------------------------------------------------------------------------
# VEV
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VEVKey:
    """The vacuum expectation value ``<O>`` of one operator."""

    operator: Operator

    def __post_init__(self):
        _check_operator(self.operator, "VEV operator")

    def __str__(self) -> str:
        return f"VEV {self.operator.coordinate}"

    @classmethod
    def from_sigmond(cls, text: str) -> VEVKey:
        el = _as_element(text)
        if el.tag != "VEV":
            raise ValueError(f"expected <VEV>, got <{el.tag}>")
        ops = [child for child in el if child.tag in OPERATOR_TAGS]
        if len(ops) != 1:
            raise ValueError("<VEV> must hold exactly one short operator tag")
        return cls(operator_from_xml(ops[0]))

    def to_sigmond(self) -> str:
        el = ET.Element("VEV")
        el.append(self.operator.to_xml())
        return ET.tostring(el, encoding="unicode")


# ---------------------------------------------------------------------------
# Key dispatch
# ---------------------------------------------------------------------------

CORRELATOR_TAG = "CorrT"
VEV_TAG = "VEV"


def key_tag(name: str) -> str | None:
    """The root tag of an XML observable name, or ``None`` if it is not XML."""
    if not name.startswith("<"):
        return None
    end = len(name)
    for c in (">", "/", " "):
        k = name.find(c, 1)
        if 0 < k < end:
            end = k
    return name[1:end]


def parse_key(name: str) -> CorrelatorKey | VEVKey:
    """Parse an observable name holding a ``<CorrT>`` or ``<VEV>`` key.

    Operator labels are memoized, so the same few labels recurring at every
    time and cell are parsed once. Names without XML escapes skip the XML
    parser entirely.
    """
    if name.startswith("<CorrT>") and name.endswith("</CorrT>") and "&" not in name:
        return _parse_key_text(name[7:-8])
    tag = key_tag(name)
    if tag == CORRELATOR_TAG:
        return CorrelatorKey.from_sigmond(name)
    if tag == VEV_TAG:
        return VEVKey.from_sigmond(name)
    raise ValueError(f"not a <CorrT> or <VEV> key: {name[:80]!r}")


# ---------------------------------------------------------------------------
# Matrix basis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelatorMatrixSpec:
    """An ordered operator basis plus the requested assembly rules.

    ``hermitian=True`` asks for Hermitian completion from both ordered
    directions; ``subtract_vev=True`` asks for connected correlators. The
    supplied order is kept: row/column ``i`` is ``operators[i]``. Operators may
    be given as short strings (see :func:`parse_operator`).

    Rejected: an empty basis, duplicates, the vacuum, operators from
    different channels, and a mix of forward and backward operators.
    """

    operators: tuple[Operator, ...]
    hermitian: bool = True
    subtract_vev: bool = False

    def __post_init__(self):
        from .operators import parse_operator

        ops = tuple(parse_operator(op) for op in self.operators)
        object.__setattr__(self, "operators", ops)
        object.__setattr__(self, "hermitian", bool(self.hermitian))
        object.__setattr__(self, "subtract_vev", bool(self.subtract_vev))
        if not ops:
            raise ValueError("a correlation matrix needs at least one operator")
        for op in ops:
            _check_operator(op, "basis operator")
        seen, dupes = set(), []
        for op in ops:
            if op in seen:
                dupes.append(op.coordinate)
            seen.add(op)
        if dupes:
            raise ValueError(f"duplicate basis operators: {dupes}")
        channels: dict[Channel, list[str]] = {}
        for op in ops:
            channels.setdefault(op.channel, []).append(op.coordinate)
        if len(channels) > 1:
            detail = "\n".join(f"  {ch}: {names}" for ch, names in channels.items())
            raise ValueError(f"basis operators span several channels:\n{detail}")
        if len({op.backwards for op in ops}) > 1:
            raise ValueError("a basis cannot mix forward and backward (bwd) operators")

    def __len__(self) -> int:
        return len(self.operators)

    @property
    def channel(self) -> Channel:
        return self.operators[0].channel

    @property
    def coordinates(self) -> tuple[str, ...]:
        return tuple(op.coordinate for op in self.operators)

    def position(self, op: Operator) -> int:
        try:
            return self.operators.index(op)
        except ValueError:
            raise KeyError(f"{op.coordinate} is not in the basis") from None

    def with_operators(self, operators: Sequence[Operator]) -> CorrelatorMatrixSpec:
        return CorrelatorMatrixSpec(tuple(operators), self.hermitian, self.subtract_vev)

    def to_sigmond_xml(self) -> str:
        """``<CorrelatorMatrixInfo>`` text (short operator tags, basis order), for Sigmond input."""
        el = ET.Element("CorrelatorMatrixInfo")
        for op in self.operators:
            el.append(op.to_xml())
        if self.hermitian:
            ET.SubElement(el, "HermitianMatrix")
        if self.subtract_vev:
            ET.SubElement(el, "SubtractVEV")
        return ET.tostring(el, encoding="unicode")
