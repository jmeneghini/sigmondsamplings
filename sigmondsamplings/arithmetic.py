"""Small immutable arithmetic descriptors; no live operand graph or sample identity."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from numbers import Integral, Real

import numpy as np

from .info import INDEP_ENSEMBLE, ObservableInfo, SectorInfo, register_obs_kind
from .metadata import (
    VERSION, AtObsInfo, CorrelatorSource, _json, _version, ensemble_record,
    generated_name, identity_json, observable_record,
)


def _identifier(record):
    return hashlib.sha256(identity_json(record).encode()).hexdigest()


def constant_record(value):
    """Exact scalar arguments and deterministic numeric-array value fingerprints."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int)):
        return dict(kind=type(value).__name__, value=value)
    if isinstance(value, float):
        return dict(kind="float", value=value.hex())
    if isinstance(value, complex):
        return dict(kind="complex", real=value.real.hex(), imag=value.imag.hex())
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("Arithmetic constants must be numeric")
    array = np.ascontiguousarray(array)
    return dict(kind="array", shape=list(array.shape), dtype=array.dtype.str,
                fingerprint=hashlib.sha256(array.tobytes()).hexdigest())


def _expression(operation, labels):
    args = [f"({label})" for label in labels]
    symbols = dict(add="+", subtract="-", multiply=r"\,", divide="/", power="^")
    if operation in symbols and len(args) == 2:
        return f"{args[0]}{symbols[operation]}{args[1]}"
    if operation == "negative":
        return "-" + args[0]
    if operation == "positive":
        return "+" + args[0]
    return rf"\mathrm{{{operation}}}" + "(" + ",".join(args) + ")"


@dataclass(frozen=True, eq=False)
class DerivedElement:
    """Ordered operand identifiers and arguments, separate from presentation."""
    operation: str
    operands: tuple[str, ...]
    sector_info: SectorInfo = field(default_factory=SectorInfo)
    labels: tuple[str, ...] = field(default=(), compare=False, hash=False)
    time_offset: float | None = 0.0
    time_extent: int | None = None

    def __post_init__(self):
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("Derived operation must be a nonempty string")
        object.__setattr__(self, "operands", tuple(_json(json.loads(item)) for item in self.operands))
        object.__setattr__(self, "labels", tuple(self.labels))
        if not isinstance(self.sector_info, SectorInfo):
            raise TypeError("Derived sector must be a SectorInfo")
        if self.labels and (len(self.labels) != len(self.operands) or
                            any(not isinstance(label, str) for label in self.labels)):
            raise ValueError("Derived operand labels must match the operands")
        if self.time_offset is not None:
            if isinstance(self.time_offset, bool) or not isinstance(self.time_offset, Real) or not math.isfinite(self.time_offset):
                raise ValueError("Derived time offset must be finite or unknown")
            object.__setattr__(self, "time_offset", float(self.time_offset))
        if self.time_extent is not None:
            if isinstance(self.time_extent, bool) or not isinstance(self.time_extent, Integral) or self.time_extent <= 0:
                raise ValueError("Derived time extent must be a positive integer or unknown")
            object.__setattr__(self, "time_extent", int(self.time_extent))

    @property
    def rot_idx(self):
        return None

    @property
    def latex_str(self):
        return _expression(self.operation, self.labels or tuple("?" for _ in self.operands))

    def to_record(self):
        return dict(element_kind="derived", metadata_version=VERSION, operation=self.operation,
                    operands=[json.loads(item) for item in self.operands], sector_info=self.sector_info.to_dict(),
                    time_offset=self.time_offset, time_extent=self.time_extent)

    @classmethod
    def from_record(cls, record):
        _version(record, required=True)
        return cls(record["operation"], tuple(_json(item) for item in record["operands"]),
                   SectorInfo(**record["sector_info"]), time_offset=record.get("time_offset", 0.0),
                   time_extent=record.get("time_extent"))

    def __eq__(self, other):
        return isinstance(other, DerivedElement) and identity_json(self.to_record()) == identity_json(other.to_record())

    def __hash__(self):
        return hash(identity_json(self.to_record()))


for _field in fields(SectorInfo):
    setattr(DerivedElement, _field.name, property(lambda self, name=_field.name: getattr(self.sector_info, name)))
DerivedElement.strangeness = property(lambda self: self.sector_info.S)
DerivedElement.isospin = property(ObservableInfo.isospin.fget)


@register_obs_kind("derived")
class DerivedObsInfo(ObservableInfo):
    """Arithmetic result, with source context but no energy or pivot claims."""
    def __init__(self, derivation, *, ensemble_info=INDEP_ENSEMBLE, latex_str=None):
        if not isinstance(derivation, DerivedElement):
            raise TypeError("derivation must be a DerivedElement")
        self._derivation = derivation
        super().__init__("", ensemble_info=ensemble_info, sector_info=derivation.sector_info, latex_str=latex_str)
        del self._sector_info

    @property
    def sector_info(self):
        return self.derivation.sector_info

    @property
    def derivation(self):
        return self._derivation

    @property
    def identity_record(self):
        return dict(obs_kind="derived", metadata_version=VERSION, derivation=self.derivation.to_record(),
                    context=dict(index=self.index, op_type=self.op_type, re_im=self.re_im,
                                 ensemble=ensemble_record(self.ensemble_info)))

    @property
    def name(self):
        return generated_name("Derived", self.identity_record)

    @name.setter
    def name(self, value):
        if value and value != self.name:
            raise ValueError("Derived names are generated from their identity records")

    @property
    def latex_str(self):
        return self.derivation.latex_str if self._latex_str is None else self._latex_str

    @latex_str.setter
    def latex_str(self, value):
        self._latex_str = value

    def to_attrs(self):
        attrs = dict(obs_kind=self.obs_kind, metadata_version=VERSION,
                     derivation=self.derivation.to_record(), operand_labels=list(self.derivation.labels),
                     identity_record=self.identity_record)
        if self._latex_str is not None:
            attrs["latex_str"] = self._latex_str
        return attrs

    @classmethod
    def from_attrs(cls, base, attrs):
        from dataclasses import replace
        _version(attrs, required=True)
        derivation = DerivedElement.from_record(attrs["derivation"])
        derivation = replace(derivation, labels=tuple(attrs.get("operand_labels", ())))
        result = cls(derivation, ensemble_info=base.ensemble_info, latex_str=attrs.get("latex_str"))
        result.index, result.op_type, result.re_im = base.index, base.op_type, base.re_im
        if result.name != base.name or identity_json(attrs["identity_record"]) != identity_json(result.identity_record):
            raise ValueError("Derived storage name or identity record contradicts metadata")
        return result

    def __eq__(self, other):
        return isinstance(other, DerivedObsInfo) and identity_json(self.identity_record) == identity_json(other.identity_record)

    def __hash__(self):
        return hash(identity_json(self.identity_record))


def arithmetic_info(operation, inputs):
    """One policy for scalar and broadcast-cell ufunc results."""
    infos = [item for item in inputs if isinstance(item, ObservableInfo)]
    sectors = [info.sector_info for info in infos]
    sector = SectorInfo(**{f.name: getattr(sectors[0], f.name) if all(
        getattr(s, f.name) == getattr(sectors[0], f.name) for s in sectors) else None
        for f in fields(SectorInfo)})
    ensemble = infos[0].ensemble_info if all(info.ensemble_info == infos[0].ensemble_info for info in infos) else INDEP_ENSEMBLE
    times = [info.time for info in infos if getattr(info, "time", None) is not None]
    at_time = bool(times) and all(time == times[0] for time in times)
    records, labels = [], []
    for item in inputs:
        if isinstance(item, ObservableInfo):
            if at_time and isinstance(item, AtObsInfo):
                record = dict(element=item.element.to_record(), context=dict(index=item.index,
                              re_im=item.re_im, op_type=item.op_type, ensemble=ensemble_record(item.ensemble_info)))
                label = item._latex_str if item._latex_str is not None else item.element.latex_str
            elif at_time and getattr(item, "key", None) is not None and hasattr(item, "hermitian"):
                source = CorrelatorSource(item.element, item.hermitian, item.subtract_vev,
                                         item.re_im, item.sink_rotation, item.source_rotation)
                record = dict(element=source.to_record(), context=dict(index=item.index,
                              re_im=item.re_im, op_type=item.op_type, ensemble=ensemble_record(item.ensemble_info)))
                label = item.latex_str
            else:
                record, label = observable_record(item), item.latex_str
            records.append(_json(dict(kind="observable", identifier=_identifier(record))))
            labels.append(label)
        else:
            records.append(_json(constant_record(item)))
            labels.append(str(item) if np.ndim(item) == 0 else r"\mathrm{array}")
    timed_infos = [info for info in infos if getattr(info, "time", None) is not None]
    offsets = [getattr(getattr(info, "element", None), "time_offset", 0.0) for info in timed_infos]
    extents = [getattr(getattr(info, "element", None), "time_extent", info.ensemble_info.temporal_extent)
               for info in timed_infos]
    offset = offsets[0] if offsets and all(value == offsets[0] for value in offsets) else (None if offsets else 0.0)
    extent = extents[0] if extents and all(value == extents[0] for value in extents) else None
    element = DerivedElement(operation, tuple(records), sector, tuple(labels), offset, extent)
    return AtObsInfo(element, "time", times[0], ensemble_info=ensemble) if at_time else DerivedObsInfo(element, ensemble_info=ensemble)
