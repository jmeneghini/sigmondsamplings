"""
Operators, correlators and correlation matrices for Sigmond data.

Labels (:mod:`.operators`, :mod:`.keys`) and the correlator/VEV observable
infos (:mod:`.observables`) are pure Python and import eagerly. The data layer
(:mod:`.index`, :mod:`.matrix`), which needs NumPy and xarray, is imported on
first use.
"""

from __future__ import annotations

from .keys import CorrelatorKey, CorrelatorMatrixSpec, CorrelatorPair, VEVKey, parse_key
from .observables import CorrelatorObsInfo, VEVObsInfo
from .operators import (
    BasicOperator,
    Channel,
    GeneralOperator,
    Hadron,
    Operator,
    SigmondParseError,
    operator_from_xml,
    parse_operator,
)

_LAZY = {
    "CorrelatorIndex": "index",
    "CorrelatorMatrix": "matrix",
    "HermiticityCheck": "matrix",
    "Transformation": "matrix",
}

__all__ = [
    "BasicOperator",
    "Channel",
    "CorrelatorKey",
    "CorrelatorMatrixSpec",
    "CorrelatorObsInfo",
    "CorrelatorPair",
    "GeneralOperator",
    "Hadron",
    "Operator",
    "SigmondParseError",
    "VEVKey",
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
