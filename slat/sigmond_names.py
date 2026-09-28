"""
Names used in Sigmond operator labels: hadron flavors, little-group irreps,
isospins and spatial operator types.

The names are copied from the ``Encoder`` tables in Sigmond's
``observables/basic_laph_operator_info.cc`` and ``gen_irrep_operator_info.cc``.
The flavor codes are kept because one detail of Sigmond's short-form output
depends on them (which tetraquark displacement is omitted).

Note that Sigmond's ``omega`` flavor is the baryon :math:`\\Omega`, not the
meson that :data:`slat.PARTICLE_LATEX_MAP` calls ``omega``; use
:data:`HADRON_FLAVORS` for operator labels.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HadronFlavor",
    "HADRON_FLAVORS",
    "TETRAQUARK_SHORT",
    "TETRAQUARK_LONG",
    "LG_IRREPS",
    "ISOSPINS",
    "SPATIAL_TYPES",
    "UNDISPLACED_SPATIAL",
]


@dataclass(frozen=True)
class HadronFlavor:
    """One hadron flavor of a basic LapH operator."""

    code: int  # Sigmond's 5-bit flavor code
    kind: str  # "glueball", "meson", "baryon" or "tetraquark"
    strangeness: int
    isospin: str  # e.g. "triplet"
    latex: str


def _flavor(code, kind, strangeness, isospin, latex) -> HadronFlavor:
    return HadronFlavor(code, kind, strangeness, isospin, latex)


# Tetraquark long flavor -> short-string stem. Short strings append "p" for
# colour type +1 and "m" for colour type -1 (e.g. "tquuuu1p").
TETRAQUARK_SHORT: dict[str, str] = {
    "isosinglet_eta_eta": "tquuuu1",
    "isotriplet_eta_pion": "tquudu3",
    "isosinglet_pion_pion": "tqdudu1",
    "isotriplet_pion_pion": "tqdudu3",
    "isoquintet_pion_pion": "tqdudu5",
    "isodoublet_kaon_eta": "tqsuuu2",
    "isodoublet_kaon_pion": "tqsudu2",
    "isoquartet_kaon_pion": "tqsudu4",
    "isotriplet_phi_pion": "tqssdu3",
    "isosinglet_eta_phi": "tquuss1",
    "isodoublet_kaon_phi": "tqsuss2",
    "isosinglet_phi_phi": "tqssss1",
}
TETRAQUARK_LONG: dict[str, str] = {v: k for k, v in TETRAQUARK_SHORT.items()}

HADRON_FLAVORS: dict[str, HadronFlavor] = {
    "glueball": _flavor(1, "glueball", 0, "singlet", "G"),
    "pion": _flavor(2, "meson", 0, "triplet", r"\pi"),
    "eta": _flavor(3, "meson", 0, "singlet", r"\eta"),
    "phi": _flavor(4, "meson", 0, "singlet", r"\phi"),
    "kaon": _flavor(5, "meson", 1, "doublet", "K"),
    "kbar": _flavor(6, "meson", -1, "doublet", r"\bar{K}"),
    "nucleon": _flavor(7, "baryon", 0, "doublet", "N"),
    "delta": _flavor(8, "baryon", 0, "quartet", r"\Delta"),
    "sigma": _flavor(9, "baryon", -1, "triplet", r"\Sigma"),
    "lambda": _flavor(10, "baryon", -1, "singlet", r"\Lambda"),
    "xi": _flavor(11, "baryon", -2, "doublet", r"\Xi"),
    "omega": _flavor(12, "baryon", -3, "singlet", r"\Omega"),
    **{
        long: _flavor(
            13 + k,
            "tetraquark",
            1 if "kaon" in long else 0,
            long.split("_")[0][3:],
            rf"\mathrm{{{short}}}",
        )
        for k, (long, short) in enumerate(TETRAQUARK_SHORT.items())
    },
}

LG_IRREPS: tuple[str, ...] = tuple(
    (
        "A1gp A1 A1g A1gm A1m A1p A1u A1um A1up A2 A2g A2gm A2gp A2m A2p A2u A2um A2up "
        "B1 B1m B1p B2 B2m B2p E Eg Egm Egp Em Ep Eu Eum Eup F1 F2 G G1 G1g G1u G2 G2g G2u "
        "Hg Hu T1g T1gm T1gp T1u T1um T1up T2g T2gm T2gp T2u T2um T2up"
    ).split()
)
"""Little-group irreps in Sigmond's encoder order (the index is Sigmond's code)."""

ISOSPINS: dict[str, int] = {
    "singlet": 0,
    "doublet": 1,
    "triplet": 2,
    "quartet": 3,
    "quintet": 4,
    "sextet": 5,
    "septet": 6,
}
"""Isospin names to ``2I`` (also Sigmond's isospin code)."""

SPATIAL_TYPES: tuple[str, ...] = (
    "SS",
    "SD",
    "LSD",
    "TSD",
    "DDI",
    "DDL",
    "TDT",
    "TDU",
    "TDO",
    "TrEig",
    "TrW1Eig",
    "TrW2Eig",
    "VI",
    "QDX",
    "Plaq",
)
UNDISPLACED_SPATIAL = frozenset({"SS", "VI"})
"""Spatial types whose displacement is always zero."""
