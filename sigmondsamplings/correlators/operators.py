"""
Immutable labels for Sigmond operators.

Two operator kinds mirror Sigmond's ``BasicLapHOperatorInfo`` ("BL") and
``GenIrrepOperatorInfo`` ("GI"):

- :class:`BasicLapHOperator` -- a symmetry-projected single- or multi-hadron
  operator built from :class:`Hadron` constituents (or the vacuum).
- :class:`GenIrrepOperator` -- a superposition (rotated or improved operator),
  known only by its channel quantum numbers plus a name and an index.

Both are frozen dataclasses. Defaults are resolved at construction, so
``str(op)`` is Sigmond's canonical short form, and two labels are equal exactly
when their canonical forms are (the short form encodes every field, as
Sigmond's own integer encoding does). ``op.coordinate`` (``BL{...}`` /
``GI{...}``) is that form wrapped by kind; it is computed once and is the
array coordinate and the equality/hash key. Sigmond's bit-field limits are
checked only on export (:meth:`to_sigmond`), not in Python.

Only the short forms are read and written: Sigmond's HDF5 keys always use them
(``MCObsInfo::serialize`` writes the short form). The name tables live in
:mod:`slat.sigmond_names`. This module is pure Python so labels stay cheap to
import.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from typing import Literal

from slat import get_irrep_latex_str
from slat.constants import latex_to_unicode
from slat.sigmond_names import (
    HADRON_FLAVORS,
    ISOSPINS,
    LG_IRREPS,
    SPATIAL_TYPES,
    TETRAQUARK_LONG,
    TETRAQUARK_SHORT,
    UNDISPLACED_SPATIAL,
)

__all__ = [
    "Hadron",
    "BasicLapHOperator",
    "GenIrrepOperator",
    "Operator",
    "Channel",
    "SigmondParseError",
    "parse_operator",
    "operator_from_xml",
]

_IRREPS = frozenset(LG_IRREPS)
_SPATIAL = frozenset(SPATIAL_TYPES)

# Sigmond bit-field limits, enforced only when a label is exported to Sigmond.
MAX_MOMENTUM_COMPONENT = 0x7F
MAX_BL_ROW = 0xF
MAX_SPATIAL_ID = 0x3FF
MAX_DISPLACEMENT = 0x7
MAX_CG_ID = 0x3F
MAX_HADRONS = 6
MAX_GI_ROW = 0xF
MAX_GI_NAME = 50  # checked C++ encoder; the reference note says 24
MAX_GI_INDEX = 32767
MAX_STRANGENESS = 0x7
MAX_MOMENTUM_SQUARED = 0xFFFFFF
MAX_TIME = 0xFFFFFF


class SigmondParseError(ValueError):
    """A Sigmond label that could not be parsed; records where and why."""

    def __init__(self, text: str, position: int, reason: str):
        self.text = text
        self.position = position
        self.reason = reason
        pointer = " " * max(position, 0) + "^"
        super().__init__(f"{reason} at position {position}:\n  {text}\n  {pointer}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_MOMENTUM_RE = re.compile(r"^P=\(([+-]?\d+),([+-]?\d+),([+-]?\d+)\)$")
_PSQ_RE = re.compile(r"^PSQ=(\d+)$")
_UINT_RE = re.compile(r"^\d+$")
_INT_RE = re.compile(r"^[+-]?\d+$")


def _tokens(text: str, offset: int = 0) -> list[tuple[str, int]]:
    """Whitespace-separated tokens with their start positions in the full text."""
    return [(m.group(), m.start() + offset) for m in re.finditer(r"\S+", text)]


def _int(value, what: str) -> int:
    if isinstance(value, bool) or getattr(value, "dtype", None) == "bool" or int(value) != value:
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _momentum_tuple(value) -> tuple[int, int, int]:
    p = tuple(_int(v, "momentum component") for v in value)
    if len(p) != 3:
        raise ValueError(f"momentum must have three components, got {value!r}")
    return p  # type: ignore[return-value]


def _format_momentum(p: tuple[int, int, int]) -> str:
    return f"P=({p[0]},{p[1]},{p[2]})"


def _strip_iso(name: str) -> str:
    return name[3:] if name.startswith("iso") else name


def _check_isospin(isospin: str) -> str:
    isospin = _strip_iso(str(isospin))
    if isospin not in ISOSPINS:
        raise ValueError(f"unknown isospin {isospin!r}; expected one of {sorted(ISOSPINS)}")
    return isospin


def _check_irrep(irrep: str) -> str:
    if irrep not in _IRREPS:
        raise ValueError(f"unknown irrep {irrep!r}")
    return irrep


def _limit(value: int, bound: int, what: str) -> None:
    if value > bound:
        raise ValueError(f"{what} {value} exceeds Sigmond's limit of {bound}")


def _set(obj, name: str, value) -> None:
    object.__setattr__(obj, name, value)


class _Label:
    """Equality, hashing and ordering by the canonical coordinate (set in ``_finish``)."""

    __slots__ = ()

    def _finish(self, prefix: str) -> None:
        coordinate = f"{prefix}{{{self}}}"
        _set(self, "_coordinate", coordinate)
        _set(self, "_hash", hash(coordinate))

    @property
    def coordinate(self) -> str:
        """``BL{...}`` / ``GI{...}``: the canonical short form, tagged by kind."""
        return self._coordinate

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, _Label):
            return NotImplemented
        return self._hash == other._hash and self._coordinate == other._coordinate

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other) -> bool:
        if not isinstance(other, _Label):
            return NotImplemented
        return self._coordinate < other._coordinate

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    def __reduce__(self):
        # Rebuild from fields, including the process-local cached hash.
        return (
            _restore_label,
            (type(self), {f.name: getattr(self, f.name) for f in fields(self) if f.init}),
        )


# Reconstruct fields so caches (including the process-local hash) are rebuilt.
def _restore_label(cls, values):
    return cls(**values)


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Channel:
    """Quantum numbers shared by every operator of a correlation-matrix basis.

    Comparison is exact: an unspecified field (``momentum=None`` for a
    ``PSQ``-only label, ``row=None`` for a row-averaged one) is a distinct
    value, not a wildcard.
    """

    isospin: str
    strangeness: int
    momentum: tuple[int, int, int] | None
    momentum_squared: int
    irrep: str
    row: int | None

    def __str__(self) -> str:
        mom = _format_momentum(self.momentum) if self.momentum else f"PSQ={self.momentum_squared}"
        row = f"_{self.row}" if self.row is not None else ""
        return f"iso{self.isospin} S={self.strangeness} {mom} {self.irrep}{row}"

    def to_sector(self):
        """The isospin and strangeness as a :class:`~sigmondsamplings.info.SectorInfo`."""
        from ..info import SectorInfo

        return SectorInfo(S=self.strangeness, twoI=ISOSPINS[self.isospin])


# ---------------------------------------------------------------------------
# Hadron
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hadron:
    """One constituent of a :class:`BasicLapHOperator` (in its reference term).

    ``displacement=None`` resolves to Sigmond's default: 0 for ``SS``/``VI``
    and glueballs, 3 for mesons, 2 otherwise. Tetraquarks take the long flavor
    name (e.g. ``"isosinglet_eta_eta"``) plus ``color=+1`` or ``-1``; the
    short names (``"tquuuu1p"``) are also accepted and converted.
    """

    flavor: str
    momentum: tuple[int, int, int]
    irrep: str
    spatial_type: str
    spatial_id: int = 0
    displacement: int | None = None
    color: int | None = None

    def __post_init__(self):
        flavor = str(self.flavor)
        color = self.color
        if flavor[:2] == "tq" and flavor[:-1] in TETRAQUARK_LONG:
            short_color = {"p": 1, "m": -1}.get(flavor[-1])
            if short_color is None:
                raise ValueError(f"tetraquark flavor {flavor!r} must end in 'p' or 'm'")
            if color is not None and color != short_color:
                raise ValueError(f"color {color} contradicts tetraquark flavor {flavor!r}")
            flavor, color = TETRAQUARK_LONG[flavor[:-1]], short_color
        if flavor not in HADRON_FLAVORS:
            raise ValueError(f"unknown flavor {flavor!r}")
        _set(self, "flavor", flavor)

        if self.is_tetraquark:
            if color not in (1, -1):
                raise ValueError(f"tetraquark {flavor!r} needs color=+1 or -1, got {color!r}")
        elif color is not None:
            raise ValueError(f"only tetraquarks carry a color type, not {flavor!r}")
        _set(self, "color", color)

        _set(self, "momentum", _momentum_tuple(self.momentum))
        irrep = _check_irrep(str(self.irrep))
        if self.is_boson and irrep[0] not in "ABET":
            raise ValueError(f"bosonic hadron {flavor!r} needs an A/B/E/T irrep, got {irrep!r}")
        if self.is_fermion and irrep[0] not in "GFH":
            raise ValueError(f"fermionic hadron {flavor!r} needs a G/F/H irrep, got {irrep!r}")

        if self.spatial_type not in _SPATIAL:
            raise ValueError(f"unknown spatial type {self.spatial_type!r}")
        spatial_id = _int(self.spatial_id, "spatial_id")
        if spatial_id < 0:
            raise ValueError("spatial_id must be nonnegative")

        displacement = self.displacement
        if self.is_glueball:
            if spatial_id != 0 or displacement not in (None, 0):
                raise ValueError("a glueball has no spatial id or displacement")
            displacement = 0
        elif self.spatial_type in UNDISPLACED_SPATIAL:
            if displacement not in (None, 0):
                raise ValueError(f"{self.spatial_type} operators have zero displacement")
            displacement = 0
        elif displacement is None:
            displacement = 3 if self.is_meson else 2
        else:
            displacement = _int(displacement, "displacement")
            if displacement < 1:
                raise ValueError(
                    f"a displaced {self.spatial_type} operator needs displacement >= 1"
                )
        _set(self, "spatial_id", spatial_id)
        _set(self, "displacement", displacement)

    # -- classification ------------------------------------------------------

    @property
    def kind(self) -> str:
        """``"glueball"``, ``"meson"``, ``"baryon"`` or ``"tetraquark"``."""
        return HADRON_FLAVORS[self.flavor].kind

    @property
    def is_glueball(self) -> bool:
        return self.kind == "glueball"

    @property
    def is_meson(self) -> bool:
        return self.kind == "meson"

    @property
    def is_baryon(self) -> bool:
        return self.kind == "baryon"

    @property
    def is_tetraquark(self) -> bool:
        return self.kind == "tetraquark"

    @property
    def is_boson(self) -> bool:
        return self.is_glueball or self.is_meson

    @property
    def is_fermion(self) -> bool:
        return self.is_baryon

    @property
    def strangeness(self) -> int:
        return HADRON_FLAVORS[self.flavor].strangeness

    @property
    def isospin(self) -> str:
        return HADRON_FLAVORS[self.flavor].isospin

    @property
    def default_displacement(self) -> int:
        """The displacement Sigmond omits from the short form (its ``dld``).

        Sigmond's ``shortwrite_hadron`` shifts a tetraquark's flavor code by 12
        (color +1) or 24 (color -1) and then masks it to 5 bits when deciding
        whether the hadron is a meson, so some tetraquarks omit ``D3`` instead
        of ``D2``. Sigmond looks HDF5 records up by their exact key text, so
        SS reproduces this only in compatibility output. Python strings retain
        the displacement because the parser defaults tetraquarks to D2.
        """
        if self.spatial_type in UNDISPLACED_SPATIAL:
            return 0
        if self.is_tetraquark:
            shifted = HADRON_FLAVORS[self.flavor].code + (12 if self.color == 1 else 24)
            return 3 if 2 <= (shifted & 0x1F) <= 6 else 2
        return 3 if self.is_meson else 2

    @property
    def short_flavor(self) -> str:
        """Flavor as written in a short string (tetraquarks use ``tq...p/m``)."""
        if self.is_tetraquark:
            return TETRAQUARK_SHORT[self.flavor] + ("p" if self.color == 1 else "m")
        return self.flavor

    @property
    def latex_str(self) -> str:
        """e.g. ``\\pi(0,0,1)``: the flavor symbol and the momentum."""
        p = ",".join(str(c) for c in self.momentum)
        return rf"{HADRON_FLAVORS[self.flavor].latex}({p})"

    # -- output --------------------------------------------------------------

    def _short_body(self, row: int | None, *, compatibility: bool = False) -> str:
        """``P=(..) irrep[_row] type[_id [Dn]]``, as ``shortwrite_hadron``."""
        text = f"{_format_momentum(self.momentum)} {self.irrep}"
        if row is not None:
            text += f"_{row}"
        text += f" {self.spatial_type}"
        if not self.is_glueball:
            text += f"_{self.spatial_id}"
            default = (
                self.default_displacement
                if compatibility or self.spatial_type in UNDISPLACED_SPATIAL
                else (3 if self.is_meson else 2)
            )
            if self.displacement != default:
                text += f" D{self.displacement}"
        return text

    def _check_sigmond(self) -> None:
        for p in self.momentum:
            _limit(abs(p), MAX_MOMENTUM_COMPONENT, "momentum component")
        _limit(self.spatial_id, MAX_SPATIAL_ID, "spatial id")
        _limit(self.displacement, MAX_DISPLACEMENT, "displacement")


# ---------------------------------------------------------------------------
# BasicLapHOperator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class BasicLapHOperator(_Label):
    """A basic LapH (BL) operator: the vacuum, one hadron, or several.

    For one hadron, ``irrep`` is the hadron's irrep (filled in when omitted)
    and ``isospin`` follows from the flavor, so it is stored as ``None``.
    For several hadrons, the total ``isospin``, ``irrep`` and ``row`` are
    required and ``iso_cg`` / ``lg_cg`` are the Clebsch-Gordan occurrence
    ids. ``BasicLapHOperator.vacuum()`` is the one zero-hadron label.
    """

    hadrons: tuple[Hadron, ...]
    irrep: str | None = None
    row: int | None = None
    isospin: str | None = None
    iso_cg: int = 0
    lg_cg: int = 0
    backwards: bool = False
    _coordinate: str = field(init=False)
    _hash: int = field(init=False)

    def __post_init__(self):
        self._validate()
        self._finish("BL")

    def _validate(self) -> None:
        hadrons = tuple(self.hadrons)
        if not all(isinstance(h, Hadron) for h in hadrons):
            raise TypeError("hadrons must be Hadron objects")
        _set(self, "hadrons", hadrons)
        iso_cg = _int(self.iso_cg, "iso_cg")
        lg_cg = _int(self.lg_cg, "lg_cg")
        if iso_cg < 0 or lg_cg < 0:
            raise ValueError("Clebsch-Gordan ids must be nonnegative")
        _set(self, "iso_cg", iso_cg)
        _set(self, "lg_cg", lg_cg)
        _set(self, "backwards", bool(self.backwards))
        n = len(hadrons)

        if n == 0:
            if (self.irrep, self.row, self.isospin) != (None, None, None) or iso_cg or lg_cg:
                raise ValueError("the vacuum carries no irrep, row, isospin or CG ids")
            if self.backwards:
                raise ValueError("the vacuum cannot be backwards")
            return

        row = _int(self.row, "row") if self.row is not None else None
        if row is None or row < 1:
            raise ValueError(f"a non-vacuum basic operator needs a positive row, got {self.row!r}")
        _set(self, "row", row)

        if n == 1:
            h = hadrons[0]
            if self.irrep is not None and self.irrep != h.irrep:
                raise ValueError(f"irrep {self.irrep!r} differs from the hadron's {h.irrep!r}")
            _set(self, "irrep", h.irrep)
            if self.isospin is not None and _strip_iso(self.isospin) != h.isospin:
                raise ValueError(f"isospin {self.isospin!r} contradicts flavor {h.flavor!r}")
            _set(self, "isospin", None)
            if iso_cg or lg_cg:
                raise ValueError("a single-hadron operator has no Clebsch-Gordan ids")
            if h.is_glueball and self.backwards:
                raise ValueError("a glueball cannot be backwards")
            return

        if self.isospin is None:
            raise ValueError("a multi-hadron operator needs an explicit total isospin")
        _set(self, "isospin", _check_isospin(self.isospin))
        if self.irrep is None:
            raise ValueError("a multi-hadron operator needs a total irrep")
        _set(self, "irrep", _check_irrep(str(self.irrep)))
        for h in hadrons:
            if h.is_glueball or h.is_tetraquark:
                raise ValueError(f"{h.flavor!r} cannot be a constituent of a multi-hadron operator")

    # -- constructors --------------------------------------------------------

    @classmethod
    def vacuum(cls) -> BasicLapHOperator:
        return cls(())

    @classmethod
    def single(
        cls,
        flavor: str,
        momentum,
        irrep: str,
        row: int,
        spatial_type: str,
        spatial_id: int = 0,
        displacement: int | None = None,
        *,
        color: int | None = None,
        backwards: bool = False,
    ) -> BasicLapHOperator:
        """A one-hadron operator (the common case)."""
        hadron = Hadron(flavor, momentum, irrep, spatial_type, spatial_id, displacement, color)
        return cls((hadron,), row=row, backwards=backwards)

    @classmethod
    def from_sigmond(cls, text: str) -> BasicLapHOperator:
        """Parse Sigmond's short BL string (an optional ``BL{...}`` wrapper is accepted)."""
        return _parse_cached("BL", _unwrap(text, "BL"))

    # -- derived quantum numbers --------------------------------------------

    @property
    def n_hadrons(self) -> int:
        return len(self.hadrons)

    @property
    def is_vacuum(self) -> bool:
        return not self.hadrons

    @property
    def momentum(self) -> tuple[int, int, int]:
        return tuple(sum(h.momentum[k] for h in self.hadrons) for k in range(3))  # type: ignore[return-value]

    @property
    def momentum_squared(self) -> int:
        return sum(p * p for p in self.momentum)

    @property
    def strangeness(self) -> int:
        return sum(h.strangeness for h in self.hadrons)

    @property
    def total_isospin(self) -> str:
        if self.is_vacuum:
            return "singlet"
        return self.isospin if self.isospin is not None else self.hadrons[0].isospin

    @property
    def channel(self) -> Channel:
        return Channel(
            isospin=self.total_isospin,
            strangeness=self.strangeness,
            momentum=self.momentum,
            momentum_squared=self.momentum_squared,
            irrep=self.irrep if self.irrep is not None else "A1gp",
            row=self.row,
        )

    def with_backwards(self, backwards: bool = True) -> BasicLapHOperator:
        return replace(self, backwards=backwards)

    @property
    def latex_str(self) -> str:
        """A compact label, e.g. ``\\pi(0,0,0)\\,A_{1u}^{-}\\,\\mathrm{SD}_5``."""
        if self.is_vacuum:
            return r"\mathrm{vacuum}"
        if self.n_hadrons == 1:
            h = self.hadrons[0]
            parts = [h.latex_str, get_irrep_latex_str(h.irrep)]
            if not h.is_glueball:
                parts.append(rf"\mathrm{{{h.spatial_type}}}_{{{h.spatial_id}}}")
            else:
                parts.append(rf"\mathrm{{{h.spatial_type}}}")
            return r"\,".join(parts)
        hadrons = r"\,".join(h.latex_str for h in self.hadrons)
        return rf"{hadrons}\,{get_irrep_latex_str(self.irrep)}"

    @property
    def compact_str(self) -> str:
        """A short plain-text label for plots, e.g. ``ππ[(0,0,1),(0,0,-1)] CG1``.

        Flavor symbols and momenta, plus what else distinguishes operators of
        one channel: the spatial type and id of a single hadron, and for several
        hadrons the Clebsch-Gordan ids and any non-``SS_0`` spatial parts.
        Shorter than ``str(op)`` but not guaranteed unique; hover text and
        tables use the full form.
        """
        if self.is_vacuum:
            return "vacuum"

        def symbol(h: Hadron) -> str:
            return latex_to_unicode(HADRON_FLAVORS[h.flavor].latex)

        def momentum(h: Hadron) -> str:
            return "({},{},{})".format(*h.momentum)

        def spatial(h: Hadron) -> str:
            return h.spatial_type if h.is_glueball else f"{h.spatial_type}{h.spatial_id}"

        if self.n_hadrons == 1:
            h = self.hadrons[0]
            text = f"{symbol(h)}{momentum(h)} {spatial(h)}"
            return text + " bwd" if self.backwards else text
        text = "".join(symbol(h) for h in self.hadrons)
        text += "[" + ",".join(momentum(h) for h in self.hadrons) + "]"
        if any((h.spatial_type, h.spatial_id) != ("SS", 0) for h in self.hadrons):
            text += " " + ",".join(spatial(h) for h in self.hadrons)
        if self.lg_cg:
            text += f" CG{self.lg_cg}"
        if self.iso_cg:
            text += f" isoCG{self.iso_cg}"
        return text + " bwd" if self.backwards else text

    # -- output --------------------------------------------------------------

    def __str__(self) -> str:
        return self._short_string()

    def _short_string(self, *, compatibility: bool = False) -> str:
        n = self.n_hadrons
        if n == 0:
            return ""
        if n == 1:
            h = self.hadrons[0]
            text = f"{h.short_flavor} {h._short_body(self.row, compatibility=compatibility)}"
            return text + " bwd" if self.backwards else text
        iso = f"iso{self.isospin}" + (f"-{self.iso_cg}" if self.iso_cg else "")
        flavors = "_".join(h.flavor for h in self.hadrons)
        total = f"{self.irrep}_{self.row}" + (f" CG_{self.lg_cg}" if self.lg_cg else "")
        if self.backwards:
            total += " bwd"
        parts = " ".join(
            f"[{h._short_body(None, compatibility=compatibility)}]" for h in self.hadrons
        )
        return f"{iso}_{flavors} {total} {parts}"

    def _check_sigmond(self) -> None:
        _limit(self.n_hadrons, MAX_HADRONS, "number of hadrons")
        if self.row is not None:
            _limit(self.row, MAX_BL_ROW, "irrep row")
        _limit(self.iso_cg, MAX_CG_ID, "isospin CG id")
        _limit(self.lg_cg, MAX_CG_ID, "little-group CG id")
        for h in self.hadrons:
            h._check_sigmond()

    def to_sigmond(self) -> str:
        """Canonical short string, after checking Sigmond's encoding limits."""
        self._check_sigmond()
        return self._short_string(compatibility=True)

    def to_xml(self) -> ET.Element:
        """Lossless Python XML; use ``to_sigmond()`` for compatibility key text."""
        el = ET.Element("BLOperatorString")
        el.text = str(self)
        return el


# ---------------------------------------------------------------------------
# GenIrrepOperator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class GenIrrepOperator(_Label):
    """A general irrep (GI) operator: channel quantum numbers plus name and index.

    Give ``momentum`` (definite) or ``momentum_squared`` (``PSQ``-only); a
    definite momentum also fills ``momentum_squared``, and ``PSQ=0`` becomes
    ``momentum=(0,0,0)``. ``row=None`` is a row-averaged label.
    """

    isospin: str
    strangeness: int
    irrep: str
    name: str
    index: int = 0
    momentum: tuple[int, int, int] | None = None
    momentum_squared: int | None = None
    row: int | None = None
    _coordinate: str = field(init=False)
    _hash: int = field(init=False)

    def __post_init__(self):
        _set(self, "isospin", _check_isospin(self.isospin))
        _set(self, "strangeness", _int(self.strangeness, "strangeness"))
        _set(self, "irrep", _check_irrep(str(self.irrep)))
        name = str(self.name)
        if not name or any(c.isspace() for c in name) or "{" in name or "}" in name:
            raise ValueError(f"GI name must be non-empty without whitespace or braces: {name!r}")
        _set(self, "name", name)
        index = _int(self.index, "index")
        if index < 0:
            raise ValueError("GI index must be nonnegative")
        _set(self, "index", index)

        momentum, psq = self.momentum, self.momentum_squared
        if momentum is None and psq is None:
            raise ValueError("give momentum or momentum_squared")
        if psq is not None:
            psq = _int(psq, "momentum_squared")
            if psq < 0:
                raise ValueError("momentum_squared must be nonnegative")
        if momentum is not None:
            momentum = _momentum_tuple(momentum)
            p2 = sum(p * p for p in momentum)
            if psq is not None and psq != p2:
                raise ValueError(f"momentum {momentum} does not have PSQ={psq}")
            psq = p2
        elif psq == 0:
            momentum = (0, 0, 0)
        _set(self, "momentum", momentum)
        _set(self, "momentum_squared", psq)

        row = self.row
        if row is not None:
            row = _int(row, "row")
            if row < 0:
                raise ValueError("row must be nonnegative")
            row = row or None  # Sigmond's row 0 means "no row"
        _set(self, "row", row)
        self._finish("GI")

    @classmethod
    def from_sigmond(cls, text: str) -> GenIrrepOperator:
        """Parse Sigmond's short GI string (an optional ``GI{...}`` wrapper is accepted)."""
        return _parse_cached("GI", _unwrap(text, "GI"))

    @property
    def has_definite_momentum(self) -> bool:
        return self.momentum is not None

    @property
    def channel(self) -> Channel:
        return Channel(
            isospin=self.isospin,
            strangeness=self.strangeness,
            momentum=self.momentum,
            momentum_squared=self.momentum_squared,
            irrep=self.irrep,
            row=self.row,
        )

    @property
    def is_vacuum(self) -> bool:
        return False

    @property
    def backwards(self) -> bool:
        return False

    @classmethod
    def from_channel(cls, channel: Channel, *, name: str, index: int = 0) -> GenIrrepOperator:
        return cls(
            channel.isospin,
            channel.strangeness,
            channel.irrep,
            name,
            index,
            channel.momentum,
            channel.momentum_squared,
            channel.row,
        )

    def with_index(self, index: int) -> GenIrrepOperator:
        return replace(self, index=index)

    @property
    def latex_str(self) -> str:
        """e.g. ``\\mathrm{ROT}_{2}\\,A_{1u}^{-}``."""
        return rf"\mathrm{{{self.name}}}_{{{self.index}}}\,{get_irrep_latex_str(self.irrep)}"

    @property
    def compact_str(self) -> str:
        """A short plain-text label for plots, e.g. ``ROT 2``."""
        return f"{self.name} {self.index}"

    def __str__(self) -> str:
        mom = (
            _format_momentum(self.momentum)
            if self.momentum is not None
            else f"PSQ={self.momentum_squared}"
        )
        irrep = f"{self.irrep}_{self.row}" if self.row is not None else self.irrep
        return f"iso{self.isospin} S={self.strangeness} {mom} {irrep} {self.name} {self.index}"

    def _check_sigmond(self) -> None:
        _limit(abs(self.strangeness), MAX_STRANGENESS, "|strangeness|")
        if self.momentum is not None:
            for p in self.momentum:
                _limit(abs(p), MAX_MOMENTUM_COMPONENT, "momentum component")
        _limit(self.momentum_squared, MAX_MOMENTUM_SQUARED, "momentum squared")
        if self.row is not None:
            _limit(self.row, MAX_GI_ROW, "irrep row")
        _limit(len(self.name), MAX_GI_NAME, "GI name length")
        _limit(self.index, MAX_GI_INDEX, "GI index")

    def to_sigmond(self) -> str:
        """Canonical short string, after checking Sigmond's encoding limits."""
        self._check_sigmond()
        return str(self)

    def to_xml(self) -> ET.Element:
        """``<GIOperatorString>...</GIOperatorString>``."""
        el = ET.Element("GIOperatorString")
        el.text = self.to_sigmond()
        return el


Operator = BasicLapHOperator | GenIrrepOperator
"""Either operator kind (a typing alias, not a wrapper class)."""


# ---------------------------------------------------------------------------
# Short-form parsing
# ---------------------------------------------------------------------------


def _unwrap(text: str, prefix: str) -> str:
    stripped = text.strip()
    if stripped.startswith(prefix + "{"):
        if not stripped.endswith("}"):
            raise SigmondParseError(text, len(text), f"unterminated {prefix}{{...}} wrapper")
        return stripped[len(prefix) + 1 : -1]
    return text


@lru_cache(maxsize=65536)
def _parse_cached(kind: str, body: str) -> Operator:
    """Parse one operator body. The same few labels recur at every time and
    matrix cell of a file, so parses are memoized (labels are immutable)."""
    return _parse_basic(body) if kind == "BL" else _parse_general(body)


def _parse_momentum(tok: str, pos: int, text: str) -> tuple[int, int, int]:
    m = _MOMENTUM_RE.match(tok)
    if not m:
        raise SigmondParseError(text, pos, f"expected momentum 'P=(x,y,z)', got {tok!r}")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _parse_irrep_row(tok: str, pos: int, text: str, row_required: bool) -> tuple[str, int | None]:
    parts = tok.split("_")
    if len(parts) > 2 or (row_required and len(parts) != 2):
        want = "'irrep_row'" if row_required else "'irrep' or 'irrep_row'"
        raise SigmondParseError(text, pos, f"expected {want}, got {tok!r}")
    irrep = parts[0]
    if irrep not in _IRREPS:
        raise SigmondParseError(text, pos, f"unknown irrep {irrep!r}")
    if len(parts) == 1:
        return irrep, None
    if not _UINT_RE.match(parts[1]):
        raise SigmondParseError(text, pos + len(irrep) + 1, f"invalid irrep row {parts[1]!r}")
    return irrep, int(parts[1])


def _parse_spatial(tok: str, pos: int, text: str) -> tuple[str, int]:
    parts = tok.split("_")
    if len(parts) != 2:
        raise SigmondParseError(text, pos, f"expected 'type_id', got {tok!r}")
    sptype, spid = parts
    if sptype not in _SPATIAL:
        raise SigmondParseError(text, pos, f"unknown spatial type {sptype!r}")
    if not _UINT_RE.match(spid):
        raise SigmondParseError(text, pos + len(sptype) + 1, f"invalid spatial id {spid!r}")
    return sptype, int(spid)


def _parse_displacement(tok: str, pos: int, text: str) -> int:
    if not re.match(r"^D\d+$", tok):
        raise SigmondParseError(text, pos, f"expected displacement 'D<n>', got {tok!r}")
    return int(tok[1:])


def _build(text: str, pos: int, factory, *args, **kwargs):
    """Run a label constructor, reporting its validation error at ``pos``."""
    try:
        return factory(*args, **kwargs)
    except (TypeError, ValueError) as e:
        if isinstance(e, SigmondParseError):
            raise
        raise SigmondParseError(text, pos, str(e)) from None


def _parse_basic(text: str) -> BasicLapHOperator:
    toks = _tokens(text)
    if not toks:
        return BasicLapHOperator.vacuum()
    head = toks[0][0]
    if head == "glueball":
        return _parse_glueball(text, toks)
    if "_" not in head:
        return _parse_single_hadron(text, toks)
    return _parse_multi_hadron(text)


def _parse_glueball(text: str, toks) -> BasicLapHOperator:
    if len(toks) != 4:
        pos = toks[4][1] if len(toks) > 4 else len(text)
        raise SigmondParseError(text, pos, "a glueball string has exactly four tokens")
    mom = _parse_momentum(*toks[1], text)
    irrep, row = _parse_irrep_row(*toks[2], text, row_required=True)
    sptype, sp_pos = toks[3]
    if sptype not in _SPATIAL:
        raise SigmondParseError(text, sp_pos, f"unknown spatial type {sptype!r}")
    hadron = _build(text, toks[0][1], Hadron, "glueball", mom, irrep, sptype)
    return _build(text, toks[2][1], BasicLapHOperator, (hadron,), row=row)


def _parse_single_hadron(text: str, toks) -> BasicLapHOperator:
    flavor, fpos = toks[0]
    if flavor not in HADRON_FLAVORS and not (flavor[:2] == "tq" and flavor[:-1] in TETRAQUARK_LONG):
        raise SigmondParseError(text, fpos, f"unknown flavor {flavor!r}")
    if len(toks) < 4:
        raise SigmondParseError(text, len(text), "a single-hadron string needs at least 4 tokens")
    mom = _parse_momentum(*toks[1], text)
    irrep, row = _parse_irrep_row(*toks[2], text, row_required=True)
    sptype, spid = _parse_spatial(*toks[3], text)
    displacement, backwards = None, False
    for tok, pos in toks[4:]:
        if tok == "bwd" and not backwards:
            backwards = True
        elif tok.startswith("D") and displacement is None:
            displacement = _parse_displacement(tok, pos, text)
        else:
            raise SigmondParseError(text, pos, f"unexpected token {tok!r}")
    hadron = _build(text, fpos, Hadron, flavor, mom, irrep, sptype, spid, displacement)
    return _build(text, fpos, BasicLapHOperator, (hadron,), row=row, backwards=backwards)


def _parse_multi_hadron(text: str) -> BasicLapHOperator:
    first_bracket = text.find("[")
    if first_bracket < 0:
        raise SigmondParseError(text, len(text), "a multi-hadron string needs '[...]' hadrons")
    head_toks = _tokens(text[:first_bracket])
    if not 2 <= len(head_toks) <= 4:
        raise SigmondParseError(
            text, 0, "expected 'iso<I>_<flavors> <irrep>_<row> [CG_n] [bwd]' before the hadrons"
        )
    names, names_pos = head_toks[0]
    pieces = names.split("_")
    iso_spec, flavors = pieces[0], pieces[1:]

    iso_cg = 0
    if "-" in iso_spec:
        iso_spec, cg_text = iso_spec.split("-", 1)
        if not _UINT_RE.match(cg_text):
            raise SigmondParseError(text, names_pos, f"invalid isospin CG id {cg_text!r}")
        iso_cg = int(cg_text)
    isospin = _strip_iso(iso_spec)
    if isospin not in ISOSPINS:
        raise SigmondParseError(text, names_pos, f"unknown total isospin {iso_spec!r}")

    irrep, row = _parse_irrep_row(*head_toks[1], text, row_required=True)
    lg_cg, backwards = None, False
    for tok, pos in head_toks[2:]:
        if tok == "bwd" and not backwards:
            backwards = True
        elif tok.startswith("CG_") and lg_cg is None and _UINT_RE.match(tok[3:]):
            lg_cg = int(tok[3:])
        else:
            raise SigmondParseError(text, pos, f"unexpected token {tok!r}")

    hadrons = []
    pos = first_bracket
    for k, flavor in enumerate(flavors):
        rest = text[pos:]
        if not rest.lstrip().startswith("["):
            raise SigmondParseError(text, pos, f"expected '[' for hadron {k + 1}")
        open_at = pos + len(rest) - len(rest.lstrip())
        close_at = text.find("]", open_at)
        if close_at < 0:
            raise SigmondParseError(text, open_at, "unterminated '['")
        body = text[open_at + 1 : close_at]
        if "[" in body:
            raise SigmondParseError(text, open_at + 1 + body.index("["), "nested '['")
        htoks = _tokens(body, offset=open_at + 1)
        if len(htoks) not in (3, 4):
            raise SigmondParseError(
                text, open_at, "expected '[P=(x,y,z) irrep type_id [D<n>]]' for a hadron"
            )
        if flavor.startswith("t") or flavor not in HADRON_FLAVORS:
            raise SigmondParseError(text, names_pos, f"invalid constituent flavor {flavor!r}")
        mom = _parse_momentum(*htoks[0], text)
        h_irrep, h_row = _parse_irrep_row(*htoks[1], text, row_required=False)
        if h_row is not None:
            raise SigmondParseError(text, htoks[1][1], "constituent irreps carry no row")
        sptype, spid = _parse_spatial(*htoks[2], text)
        disp = _parse_displacement(*htoks[3], text) if len(htoks) == 4 else None
        hadrons.append(_build(text, open_at, Hadron, flavor, mom, h_irrep, sptype, spid, disp))
        pos = close_at + 1

    if text[pos:].strip():
        raise SigmondParseError(
            text, pos + len(text[pos:]) - len(text[pos:].lstrip()), "unexpected text after hadrons"
        )
    if len(hadrons) < 2:
        raise SigmondParseError(text, names_pos, "a multi-hadron string names at least two hadrons")
    return _build(
        text,
        0,
        BasicLapHOperator,
        tuple(hadrons),
        irrep=irrep,
        row=row,
        isospin=isospin,
        iso_cg=iso_cg,
        lg_cg=lg_cg or 0,
        backwards=backwards,
    )


def _parse_general(text: str) -> GenIrrepOperator:
    toks = _tokens(text)
    if not 4 <= len(toks) <= 6:
        raise SigmondParseError(
            text, 0, "expected 'iso<I> [S=n] P=(x,y,z)|PSQ=n irrep[_row] name [index]'"
        )
    iso, iso_pos = toks[0]
    isospin = _strip_iso(iso)
    if isospin not in ISOSPINS:
        raise SigmondParseError(text, iso_pos, f"unknown isospin {iso!r}")
    k = 1
    strangeness = 0
    if toks[k][0].startswith("S="):
        s_tok, s_pos = toks[k]
        if not _INT_RE.match(s_tok[2:]):
            raise SigmondParseError(text, s_pos, f"invalid strangeness {s_tok!r}")
        strangeness = int(s_tok[2:])
        k += 1
    mom_tok, mom_pos = toks[k]
    momentum, psq = None, None
    if mom_tok.startswith("PSQ="):
        m = _PSQ_RE.match(mom_tok)
        if not m:
            raise SigmondParseError(text, mom_pos, f"invalid momentum squared {mom_tok!r}")
        psq = int(m.group(1))
    else:
        momentum = _parse_momentum(mom_tok, mom_pos, text)
    k += 1
    if len(toks) < k + 2:
        raise SigmondParseError(text, len(text), "missing irrep or ID name")
    irrep, row = _parse_irrep_row(*toks[k], text, row_required=False)
    name, name_pos = toks[k + 1]
    k += 2
    index = 0
    if k < len(toks):
        idx_tok, idx_pos = toks[k]
        if not _UINT_RE.match(idx_tok):
            raise SigmondParseError(text, idx_pos, f"invalid ID index {idx_tok!r}")
        index = int(idx_tok)
        k += 1
    if k < len(toks):
        raise SigmondParseError(text, toks[k][1], f"unexpected token {toks[k][0]!r}")
    return _build(
        text,
        name_pos,
        GenIrrepOperator,
        isospin,
        strangeness,
        irrep,
        name,
        index,
        momentum=momentum,
        momentum_squared=psq,
        row=row,
    )


def parse_operator(
    text: str | Operator, kind: Literal["auto", "basic", "general"] = "auto"
) -> Operator:
    """Parse a short operator string of either kind (an operator is returned as is).

    ``BL{...}`` / ``GI{...}`` wrappers decide the kind. Otherwise ``kind`` may
    be ``"basic"`` or ``"general"``; ``"auto"`` dispatches by the head token.
    An explicit kind must match an existing operator or a tagged string.
    """
    if kind not in ("auto", "basic", "general"):
        raise ValueError(f"kind must be 'auto', 'basic' or 'general', got {kind!r}")
    if isinstance(text, _Label):
        actual = "basic" if isinstance(text, BasicLapHOperator) else "general"
        if kind not in ("auto", actual):
            raise TypeError(f"expected {kind} operator, got {actual}")
        return text
    stripped = text.strip()
    tagged = (
        "basic" if stripped.startswith("BL{") else "general" if stripped.startswith("GI{") else None
    )
    if tagged is not None:
        if kind not in ("auto", tagged):
            raise ValueError(f"{tagged} tag conflicts with kind={kind!r}")
        kind = tagged
    if kind == "auto":
        head = stripped.split(maxsplit=1)[0] if stripped else ""
        kind = "general" if _strip_iso(head) in ISOSPINS else "basic"
    parser = BasicLapHOperator if kind == "basic" else GenIrrepOperator
    return parser.from_sigmond(stripped)


# ---------------------------------------------------------------------------
# Short XML tags
# ---------------------------------------------------------------------------

_BL_TAGS = ("OperatorString", "BLOperatorString")
_GI_TAGS = ("GIOperatorString",)
OPERATOR_TAGS = _BL_TAGS + _GI_TAGS


def _as_element(xml: str | ET.Element) -> ET.Element:
    if not isinstance(xml, str):
        return xml
    try:
        return ET.fromstring(xml.strip())
    except ET.ParseError as e:
        raise ValueError(f"invalid XML {xml[:80]!r}: {e}") from None


def operator_from_xml(xml: str | ET.Element) -> Operator:
    """An operator from ``<BLOperatorString>``, ``<OperatorString>`` or ``<GIOperatorString>``."""
    el = _as_element(xml)
    if el.tag in _BL_TAGS:
        return BasicLapHOperator.from_sigmond(el.text or "")
    if el.tag in _GI_TAGS:
        return GenIrrepOperator.from_sigmond(el.text or "")
    raise ValueError(f"<{el.tag}> is not a short operator tag; expected one of {OPERATOR_TAGS}")
