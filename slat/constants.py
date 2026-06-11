"""Shared constants and lightweight formatting helpers for SLAT packages."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = [
    "VERBOSITY_MAP",
    "ENERGY_TYPE_LATEX_MAP",
    "ENERGY_TYPE_UNICODE_MAP",
    "ENERGY_TYPES",
    "SHIFT_ENERGY_TYPES",
    "NI_PAIR_ENERGY_TYPES",
    "PARTICLE_LATEX_MAP",
    "PARTICLE_UNICODE_MAP",
    "IRREP_LATEX_MAP",
    "IRREP_UNICODE_MAP",
    "QC_MATRIX_LATEX_MAP",
    "QC_MATRIX_UNICODE_MAP",
    "QC_TYPE_LATEX_MAP",
    "QC_TYPE_UNICODE_MAP",
    "ELL_SPEC_MAP",
    "COLORS",
    "MARKERS",
    "IndexedCycle",
    "create_spectro_bidict",
    "_half_int_latex",
    "latex_to_unicode",
    "get_energy_type_latex_str",
    "get_energy_type_unicode_str",
    "get_particle_latex_str",
    "get_particle_unicode_str",
    "get_all_particle_latex_mappings",
    "get_all_particle_unicode_mappings",
    "get_irrep_latex_str",
    "get_irrep_unicode_str",
    "get_all_irrep_latex_mappings",
    "get_all_irrep_unicode_mappings",
    "get_spectro_str",
    "get_twoJ_L_twoS_from_spectro",
    "get_spectro_latex_str",
    "get_spectro_unicode_str",
    "get_qc_matrix_latex_str",
    "get_qc_matrix_unicode_str",
    "get_all_qcmatrix_latex_mappings",
    "get_all_qcmatrix_unicode_mappings",
    "get_qc_type_latex_str",
    "get_qc_type_unicode_str",
    "get_all_qctype_latex_mappings",
    "get_all_qctype_unicode_mappings",
    "wrap_str_in_determinant",
]

T = TypeVar("T")


class _BiMap(dict):
    """Tiny bidirectional mapping for the spectroscopic notation table."""

    @property
    def inverse(self) -> dict:
        return {value: key for key, value in self.items()}

_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "\u2070",
        "1": "\u00b9",
        "2": "\u00b2",
        "3": "\u00b3",
        "4": "\u2074",
        "5": "\u2075",
        "6": "\u2076",
        "7": "\u2077",
        "8": "\u2078",
        "9": "\u2079",
        "+": "\u207a",
        "-": "\u207b",
        "/": "\u141f",
        "=": "\u207c",
        "(": "\u207d",
        ")": "\u207e",
        "n": "\u207f",
    }
)
_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "\u2080",
        "1": "\u2081",
        "2": "\u2082",
        "3": "\u2083",
        "4": "\u2084",
        "5": "\u2085",
        "6": "\u2086",
        "7": "\u2087",
        "8": "\u2088",
        "9": "\u2089",
        "+": "\u208a",
        "-": "\u208b",
        "=": "\u208c",
        "(": "\u208d",
        ")": "\u208e",
        "a": "\u2090",
        "e": "\u2091",
        "h": "\u2095",
        "i": "\u1d62",
        "j": "\u2c7c",
        "k": "\u2096",
        "l": "\u2097",
        "m": "\u2098",
        "n": "\u2099",
        "o": "\u2092",
        "p": "\u209a",
        "r": "\u1d63",
        "s": "\u209b",
        "t": "\u209c",
        "u": "\u1d64",
        "v": "\u1d65",
        "x": "\u2093",
    }
)
_LATEX_COMMAND_UNICODE_MAP = {
    r"\Delta": "\u0394",
    r"\Gamma": "\u0393",
    r"\Lambda": "\u039b",
    r"\Omega": "\u03a9",
    r"\Sigma": "\u03a3",
    r"\Xi": "\u039e",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\eta": "\u03b7",
    r"\gamma": "\u03b3",
    r"\kappa": "\u03ba",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\nu": "\u03bd",
    r"\omega": "\u03c9",
    r"\phi": "\u03c6",
    r"\pi": "\u03c0",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\theta": "\u03b8",
    r"\xi": "\u03be",
    r"\det": "det",
}
_LATEX_SPACE_COMMAND_RE = re.compile(r"\\[,;:! ]")
_LATEX_TEXT_COMMAND_RE = re.compile(r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}")
_LATEX_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_ACCENT_RE = re.compile(r"\\(bar|tilde|widetilde)\s*\{([^{}]+)\}")
_LATEX_SCRIPT_RE = re.compile(r"([_^])\s*(?:\{([^{}]*)\}|([^\\{}\s]))")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")


def _unicode_accent(command: str, value: str) -> str:
    mark = "\u0304" if command == "bar" else "\u0303"
    return "".join(char + mark for char in value)


def _unicode_script(match: re.Match[str]) -> str:
    marker, braced, plain = match.groups()
    value = braced if braced is not None else plain
    table = _SUPERSCRIPT_TRANSLATION if marker == "^" else _SUBSCRIPT_TRANSLATION
    return value.translate(table)


def latex_to_unicode(latex: str) -> str:
    """Best-effort conversion from the package's lightweight LaTeX labels to Unicode."""
    text = str(latex)
    text = text.replace("$", "")
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_TEXT_COMMAND_RE.sub(lambda match: match.group(1), text)
        text = _LATEX_FRAC_RE.sub(lambda match: f"{match.group(1)}/{match.group(2)}", text)
        text = _LATEX_ACCENT_RE.sub(
            lambda match: _unicode_accent(match.group(1), match.group(2)), text
        )
        text = _LATEX_SCRIPT_RE.sub(_unicode_script, text)
    text = _LATEX_SPACE_COMMAND_RE.sub("", text)
    for command, replacement in sorted(
        _LATEX_COMMAND_UNICODE_MAP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(command, replacement)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = _LATEX_COMMAND_RE.sub(lambda match: match.group(0)[1:], text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


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


def get_energy_type_unicode_str(energy_type_name: str, index: int = None) -> str:
    """Return the Unicode label for an energy type, falling back to the input."""
    return latex_to_unicode(get_energy_type_latex_str(energy_type_name, index))


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


def get_particle_unicode_str(particle_name: str) -> str:
    """Return the Unicode label for a particle, falling back to the input."""
    return latex_to_unicode(get_particle_latex_str(particle_name))


def get_all_particle_latex_mappings() -> dict[str, str]:
    """Return a copy of all particle LaTeX mappings."""
    return PARTICLE_LATEX_MAP.copy()


def get_all_particle_unicode_mappings() -> dict[str, str]:
    """Return all particle labels converted to Unicode."""
    return {key: latex_to_unicode(value) for key, value in PARTICLE_LATEX_MAP.items()}


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


def get_irrep_unicode_str(irrep_name: str) -> str:
    """Return the Unicode label for an irrep, falling back to the input."""
    if irrep_name.endswith(("p", "m")):
        base = irrep_name[:-1]
        if base in IRREP_LATEX_MAP:
            suffix = "⁺" if irrep_name.endswith("p") else "⁻"
            return f"{latex_to_unicode(IRREP_LATEX_MAP[base])}{suffix}"
    return latex_to_unicode(get_irrep_latex_str(irrep_name))


def get_all_irrep_latex_mappings() -> dict[str, str]:
    """Return a copy of all irrep LaTeX mappings."""
    return IRREP_LATEX_MAP.copy()


def get_all_irrep_unicode_mappings() -> dict[str, str]:
    """Return all irrep labels converted to Unicode."""
    return {key: get_irrep_unicode_str(key) for key in IRREP_LATEX_MAP}


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


def get_spectro_unicode_str(twoJ, L, twoS) -> str:
    """Return spectroscopic Unicode notation for quantum numbers J, L, and S."""
    return latex_to_unicode(get_spectro_latex_str(twoJ, L, twoS))


def get_qc_matrix_latex_str(qc_matrix_name: str) -> str:
    """Return the LaTeX label for a QC matrix, falling back to the input."""
    return QC_MATRIX_LATEX_MAP.get(qc_matrix_name, qc_matrix_name)


def get_qc_matrix_unicode_str(qc_matrix_name: str) -> str:
    """Return the Unicode label for a QC matrix, falling back to the input."""
    return latex_to_unicode(get_qc_matrix_latex_str(qc_matrix_name))


def get_all_qcmatrix_latex_mappings() -> dict[str, str]:
    """Return a copy of all QC matrix LaTeX mappings."""
    return QC_MATRIX_LATEX_MAP.copy()


def get_all_qcmatrix_unicode_mappings() -> dict[str, str]:
    """Return all QC matrix labels converted to Unicode."""
    return {key: latex_to_unicode(value) for key, value in QC_MATRIX_LATEX_MAP.items()}


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


def get_qc_type_unicode_str(qc_type_name: str, show_det: bool = True) -> str:
    """Return the Unicode label for a QC type, optionally determinant-wrapped."""
    return latex_to_unicode(get_qc_type_latex_str(qc_type_name, show_det))


def get_all_qctype_latex_mappings(show_det: bool = True) -> dict[str, str]:
    """Return a copy of all QC type LaTeX mappings."""
    if show_det:
        return {key: wrap_str_in_determinant(val) for key, val in QC_TYPE_LATEX_MAP.items()}
    return QC_TYPE_LATEX_MAP.copy()


def get_all_qctype_unicode_mappings(show_det: bool = True) -> dict[str, str]:
    """Return all QC type labels converted to Unicode."""
    return {
        key: latex_to_unicode(value)
        for key, value in get_all_qctype_latex_mappings(show_det).items()
    }


ENERGY_TYPE_UNICODE_MAP: dict[str, str] = {
    key: latex_to_unicode(value) for key, value in ENERGY_TYPE_LATEX_MAP.items()
}
PARTICLE_UNICODE_MAP: dict[str, str] = get_all_particle_unicode_mappings()
IRREP_UNICODE_MAP: dict[str, str] = get_all_irrep_unicode_mappings()
QC_MATRIX_UNICODE_MAP: dict[str, str] = get_all_qcmatrix_unicode_mappings()
QC_TYPE_UNICODE_MAP: dict[str, str] = get_all_qctype_unicode_mappings(show_det=False)


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
