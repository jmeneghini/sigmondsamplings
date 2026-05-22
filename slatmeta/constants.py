"""Shared constants and lightweight formatting helpers for SLAT packages."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = [
    "VERBOSITY_MAP",
    "ENERGY_TYPE_LATEX_MAP",
    "ENERGY_TYPES",
    "SHIFT_ENERGY_TYPES",
    "NI_PAIR_ENERGY_TYPES",
    "PARTICLE_LATEX_MAP",
    "IRREP_LATEX_MAP",
    "QC_MATRIX_LATEX_MAP",
    "QC_TYPE_LATEX_MAP",
    "ELL_SPEC_MAP",
    "COLORS",
    "MARKERS",
    "IndexedCycle",
    "create_spectro_bidict",
    "_half_int_latex",
    "get_energy_type_latex_str",
    "get_particle_latex_str",
    "get_all_particle_latex_mappings",
    "get_irrep_latex_str",
    "get_all_irrep_latex_mappings",
    "get_spectro_str",
    "get_twoJ_L_twoS_from_spectro",
    "get_spectro_latex_str",
    "get_qc_matrix_latex_str",
    "get_all_qcmatrix_latex_mappings",
    "get_qc_type_latex_str",
    "get_all_qctype_latex_mappings",
    "wrap_str_in_determinant",
]

T = TypeVar("T")


class _BiMap(dict):
    """Tiny bidirectional mapping for the spectroscopic notation table."""

    @property
    def inverse(self) -> dict:
        return {value: key for key, value in self.items()}


VERBOSITY_MAP = {
    "low": "L",
    "medium": "M",
    "high": "H",
}

ENERGY_TYPE_LATEX_MAP: dict[str, str] = {
    "ecm": r"E_{\mathrm{cm}}",
    "elab": r"E_{\mathrm{lab}}",
    "delab": r"\Delta E_{\mathrm{lab}}",
    "decm": r"\Delta E_{\mathrm{cm}}",
    "qcmsq": r"q^{*2}/m_{\mathrm{ref}}^{2}",
}

ENERGY_TYPES: set[str] = {"ecm", "elab", "decm", "delab", "qcmsq"}
SHIFT_ENERGY_TYPES: set[str] = {"delab", "decm"}
NI_PAIR_ENERGY_TYPES: set[str] = {"delab", "decm", "qcmsq"}


def get_energy_type_latex_str(energy_type_name: str, index: int = None) -> str:
    """Return the LaTeX label for an energy type, falling back to the input."""
    tex_str = ENERGY_TYPE_LATEX_MAP.get(energy_type_name, energy_type_name)
    if index is not None:
        tex_str = tex_str + rf"^{{{index}}}"
    return tex_str


PARTICLE_LATEX_MAP: dict[str, str] = {
    "pion": r"\pi",
    "pi": r"\pi",
    "pi+": r"\pi^+",
    "pi-": r"\pi^-",
    "pi0": r"\pi^0",
    "eta": r"\eta",
    "kaon": r"K",
    "K": r"K",
    "k": r"\bar{K}",
    "K+": r"K^+",
    "K-": r"K^-",
    "K0": r"K^0",
    "rho": r"\rho",
    "rho+": r"\rho^+",
    "rho-": r"\rho^-",
    "rho0": r"\rho^0",
    "omega": r"\omega",
    "phi": r"\phi",
    "nucleon": r"N",
    "N": r"N",
    "proton": r"p",
    "neutron": r"n",
    "sigma": r"\Sigma",
    "S": r"\Sigma",
    "lambda": r"\Lambda",
    "L": r"\Lambda",
    "delta": r"\Delta",
    "xi": r"\Xi",
    "X": r"\Xi",
}


def get_particle_latex_str(particle_name: str) -> str:
    """Return the LaTeX label for a particle, falling back to the input."""
    return PARTICLE_LATEX_MAP.get(particle_name, particle_name)


def get_all_particle_latex_mappings() -> dict[str, str]:
    """Return a copy of all particle LaTeX mappings."""
    return PARTICLE_LATEX_MAP.copy()


IRREP_LATEX_MAP: dict[str, str] = {
    "A1g": r"A_{1g}",
    "A2g": r"A_{2g}",
    "A1u": r"A_{1u}",
    "A2u": r"A_{2u}",
    "Eg": r"E_{g}",
    "Eu": r"E_{u}",
    "T1g": r"T_{1g}",
    "T1u": r"T_{1u}",
    "T2g": r"T_{2g}",
    "T2u": r"T_{2u}",
    "A1": r"A_1",
    "A2": r"A_2",
    "B1": r"B_1",
    "B2": r"B_2",
    "E": r"E",
    "F1": r"F_1",
    "F2": r"F_2",
    "G": r"G",
    "G1": r"G_1",
    "G1g": r"G_{1g}",
    "G1u": r"G_{1u}",
    "G2": r"G_2",
    "Hg": r"H_{g}",
    "Hu": r"H_{u}",
}

_latex_format_addp = {
    irrep + "p": latex_format[:-1] + "^{+}$"
    for irrep, latex_format in IRREP_LATEX_MAP.items()
}
_latex_format_addm = {
    irrep + "m": latex_format[:-1] + "^{-}$"
    for irrep, latex_format in IRREP_LATEX_MAP.items()
}
IRREP_LATEX_MAP.update(_latex_format_addp)
IRREP_LATEX_MAP.update(_latex_format_addm)


def get_irrep_latex_str(irrep_name: str) -> str:
    """Return the LaTeX label for an irrep, falling back to the input."""
    return IRREP_LATEX_MAP.get(irrep_name, irrep_name)


def get_all_irrep_latex_mappings() -> dict[str, str]:
    """Return a copy of all irrep LaTeX mappings."""
    return IRREP_LATEX_MAP.copy()


QC_MATRIX_LATEX_MAP: dict[str, str] = {
    "ktilde": r"\widetilde{K}",
    "ktildeinv": r"\widetilde{K}^{-1}",
    "b": r"B",
    "stilde": r"\widetilde{S}",
    "stildeinv": r"\widetilde{S}^{-1}",
    "cb": r"C_B",
}


def create_spectro_bidict(limit=15):
    prefix = ["s", "p", "d", "f"]
    alphabetical = [char for char in string.ascii_lowercase[6:] if char != "j"]
    full_sequence = prefix + alphabetical
    return _BiMap({idx: symbol.upper() for idx, symbol in enumerate(full_sequence[: limit + 1])})


ELL_SPEC_MAP = create_spectro_bidict(10)


def _half_int_latex(n_times_two: int, use_frac: bool = False) -> str:
    """Convert a half-integer, given as twice the value, to a LaTeX string."""
    if n_times_two % 2 == 0:
        return str(n_times_two // 2)
    if use_frac:
        return rf"\frac{{{n_times_two}}}{{2}}"
    return rf"{n_times_two}/2"


def get_spectro_str(twoJ, L, twoS) -> str:
    """Return spectroscopic notation for quantum numbers J, L, and S."""
    if L not in ELL_SPEC_MAP:
        raise ValueError(
            f"Unsupported L value: {L}. Supported range is 0 to {max(ELL_SPEC_MAP.keys())}."
        )

    S_str = f"{int(twoS) + 1}"
    L_str = ELL_SPEC_MAP[int(L)]
    J_str = _half_int_latex(twoJ, use_frac=False)

    return f"{S_str}{L_str}{J_str}"


def get_twoJ_L_twoS_from_spectro(spectro_str: str) -> tuple[int, int, int]:
    """Parse spectroscopic notation into ``(2J, L, 2S)``."""
    match = re.match(r"(\d*)([a-zA-Z])(.*)", spectro_str.strip())
    if not match:
        raise ValueError(f"Could not parse notation: {spectro_str}")

    mult_str, l_symbol, j_str = match.groups()

    L = ELL_SPEC_MAP.inverse.get(l_symbol.upper())
    if L is None:
        raise ValueError(f"Invalid symbol: {l_symbol}")

    multiplicity = int(mult_str) if mult_str else 1
    two_s = multiplicity - 1

    j_str = j_str.strip()
    if "/" in j_str:
        num, den = j_str.split("/")
        two_j = (2 * int(num)) // int(den)
    elif j_str:
        two_j = int(j_str) * 2
    else:
        two_j = 0

    return two_j, L, two_s


def get_spectro_latex_str(twoJ, L, twoS) -> str:
    """Return spectroscopic LaTeX notation for quantum numbers J, L, and S."""
    if L not in ELL_SPEC_MAP:
        raise ValueError(
            f"Unsupported L value: {L}. Supported range is 0 to {max(ELL_SPEC_MAP.keys())}."
        )

    S_str = f"{int(twoS) + 1}"
    L_str = ELL_SPEC_MAP[int(L)]

    return rf"{{}}^{{{S_str}}}{L_str}" + (
        f"_ {{{_half_int_latex(twoJ, use_frac=False)}}}" if twoJ is not None else ""
    )


def get_qc_matrix_latex_str(qc_matrix_name: str) -> str:
    """Return the LaTeX label for a QC matrix, falling back to the input."""
    return QC_MATRIX_LATEX_MAP.get(qc_matrix_name, qc_matrix_name)


def get_all_qcmatrix_latex_mappings() -> dict[str, str]:
    """Return a copy of all QC matrix LaTeX mappings."""
    return QC_MATRIX_LATEX_MAP.copy()


QC_TYPE_LATEX_MAP: dict[str, str] = {
    "ktilde_b": rf"1 - {get_qc_matrix_latex_str('ktilde')} {get_qc_matrix_latex_str('b')}",
    "ktildeinv_b": rf"{get_qc_matrix_latex_str('ktildeinv')} - {get_qc_matrix_latex_str('b')}",
    "stildeinv_cb": rf"{get_qc_matrix_latex_str('stildeinv')} + {get_qc_matrix_latex_str('cb')}",
    "stilde_cb": rf"1 + {get_qc_matrix_latex_str('stilde')} {get_qc_matrix_latex_str('cb')}",
}


def get_qc_type_latex_str(qc_type_name: str, show_det: bool = True) -> str:
    """Return the LaTeX label for a QC type, optionally determinant-wrapped."""
    main_str = QC_TYPE_LATEX_MAP.get(qc_type_name, qc_type_name)
    return wrap_str_in_determinant(main_str) if show_det else main_str


def get_all_qctype_latex_mappings(show_det: bool = True) -> dict[str, str]:
    """Return a copy of all QC type LaTeX mappings."""
    if show_det:
        return {key: wrap_str_in_determinant(val) for key, val in QC_TYPE_LATEX_MAP.items()}
    return QC_TYPE_LATEX_MAP.copy()


def wrap_str_in_determinant(expr: str) -> str:
    """Wrap an expression in determinant notation."""
    return rf"\det\left({expr}\right)"


COLORS = [
    "#d60000",  # Red
    "#8c3bff",  # Purple
    "#018700",  # Green
    "#00acc6",  # Cyan
    "#ffa52f",  # Orange
    "#6b004f",  # Dark Plum
    "#97ff00",  # Lime
    "#ff7ed1",  # Pink
    "#5e56ff",  # Blue-Purple
    "#c500ff",  # Magenta
]
COLORS += COLORS

MARKERS = [
    "o",
    "s",
    "D",
    "v",
    "^",
    "<",
    ">",
    "o",
    "s",
    "D",
    "v",
    "^",
    "*",
    "x",
    "+",
    "o",
    "s",
    "D",
    "v",
    "^",
    "*",
    "x",
    "+",
]
MARKERS += MARKERS


class IndexedCycle(Iterator[T]):
    """A bounded, restartable cycle with stateful access."""

    __slots__ = ("items", "index")

    def __init__(self, iterable: Iterable[T]):
        self.items: list[T] = list(iterable)
        if not self.items:
            raise ValueError("IndexedCycle requires a non-empty iterable")
        self.index: int = 0

    def __next__(self) -> T:
        value = self.items[self.index]
        self.index = (self.index + 1) % len(self.items)
        return value

    def __iter__(self) -> IndexedCycle[T]:
        return self

    def get_current(self) -> T:
        return self.items[self.index]

    def get_state(self) -> int:
        return self.index

    def set_state(self, index: int) -> None:
        self.index = index % len(self.items)

    def __len__(self) -> int:
        return len(self.items)
