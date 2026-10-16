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
    from .effective import EffectiveEnergyInfo as EffectiveEnergyInfo
    from .effective import EffectiveEnergyWarning as EffectiveEnergyWarning
    from .effective import effective_energy as effective_energy
    from .effective import effective_time_offset as effective_time_offset
    from .index import CorrelatorIndex as CorrelatorIndex
    from .matrix import CorrelatorMatrix as CorrelatorMatrix
    from .matrix import HermiticityCheck as HermiticityCheck
    from .matrix import Transformation as Transformation
    from .pivot import (
        GEVPSolution as GEVPSolution,
    )
    from .pivot import (
        NullSpaceCheck as NullSpaceCheck,
    )
    from .pivot import (
        OffDiagonalCheck as OffDiagonalCheck,
    )
    from .pivot import (
        PivotDiagnostics as PivotDiagnostics,
    )
    from .pivot import (
        PivotError as PivotError,
    )
    from .pivot import (
        PivotSettings as PivotSettings,
    )
    from .pivot import (
        PivotWarning as PivotWarning,
    )
    from .pivot import (
        SinglePivot as SinglePivot,
    )
    from .pivot import (
        solve_gevp as solve_gevp,
    )

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
    **dict.fromkeys(("SinglePivot", "GEVPSolution", "NullSpaceCheck", "OffDiagonalCheck",
                     "PivotDiagnostics", "PivotError", "PivotSettings", "PivotWarning",
                     "solve_gevp"), "pivot"),
    "CorrelatorIndex": "index",
    "CorrelatorMatrix": "matrix",
    "EffectiveEnergyInfo": "effective",
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
