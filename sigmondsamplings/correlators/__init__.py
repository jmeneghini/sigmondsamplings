"""
Operators, correlators and correlation matrices for Sigmond data.

Labels (:mod:`.operators`, :mod:`.keys`) and the correlator/VEV observable
infos (:mod:`.observables`) are pure Python and import eagerly. The data layer
(:mod:`.index`, :mod:`.matrix`) and the calculators (:mod:`.effective`), which
need NumPy and xarray, are imported on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .effective import EffectiveEnergyObsInfo as EffectiveEnergyObsInfo
    from .effective import EffectiveEnergyWarning as EffectiveEnergyWarning
    from .effective import effective_energy as effective_energy
    from .effective import effective_time_offset as effective_time_offset
    from .index import CorrelatorIndex as CorrelatorIndex
    from .matrix import CorrelatorMatrix as CorrelatorMatrix
    from .matrix import HermiticityCheck as HermiticityCheck
    from .matrix import Transformation as Transformation

from .keys import (
    CorrelatorAtTimeInfo,
    CorrelatorInfo,
    CorrelatorMatrixInfo,
    VEVInfo,
    parse_key,
)
from .observables import CorrelatorObsInfo, VEVObsInfo
from .operators import (
    BasicLapHOperator,
    Channel,
    GenIrrepOperator,
    Hadron,
    Operator,
    SigmondParseError,
    operator_from_xml,
    parse_operator,
)

_LAZY = {
    "CorrelatorIndex": "index",
    "CorrelatorMatrix": "matrix",
    "EffectiveEnergyObsInfo": "effective",
    "EffectiveEnergyWarning": "effective",
    "HermiticityCheck": "matrix",
    "Transformation": "matrix",
    "effective_energy": "effective",
    "effective_time_offset": "effective",
}

__all__ = [
    "BasicLapHOperator",
    "Channel",
    "CorrelatorAtTimeInfo",
    "CorrelatorInfo",
    "CorrelatorMatrixInfo",
    "CorrelatorObsInfo",
    "GenIrrepOperator",
    "Hadron",
    "Operator",
    "SigmondParseError",
    "VEVInfo",
    "VEVObsInfo",
    "operator_from_xml",
    "parse_key",
    "parse_operator",
    *_LAZY,
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
