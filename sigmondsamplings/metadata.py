"""Immutable element snapshots and the concrete SS nested metadata codec.

Storage identities are versioned, deterministic records. Labels are presentation
only; metadata equality and hashing never inspect sample buffers.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from numbers import Integral
from typing import TYPE_CHECKING

from .info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SectorInfo, register_obs_kind

if TYPE_CHECKING:
    from .correlators.keys import CorrelatorInfo

VERSION = 1


def _scalar(value):
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"coordinate/metadata value must be a finite JSON scalar, got {value!r}")


def _normalized(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata record keys must be strings")
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    return _scalar(value)


def _json(value):
    return json.dumps(_normalized(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _typed(value):
    """Preserve numeric types and exact floating values in persistent identity."""
    value = _normalized(value)
    if isinstance(value, dict):
        return ["map", [[key, _typed(value[key])] for key in sorted(value)]]
    if isinstance(value, list):
        return ["list", [_typed(item) for item in value]]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, int) and not isinstance(value, bool):
        return ["int", str(value)]
    if isinstance(value, bool):
        return ["bool", value]
    if value is None:
        return ["null"]
    return ["str", value]


def identity_json(record):
    return _json(_typed(record))


def generated_name(prefix, record):
    return f"{prefix}_ID{hashlib.sha256(identity_json(record).encode('utf-8')).hexdigest()[:32]}"


def _version(record, *, required=False):
    if required and "metadata_version" not in record:
        raise ValueError("Missing metadata version")
    version = record.get("metadata_version", VERSION)
    if type(version) is not int or version != VERSION:
        raise ValueError(f"Unsupported metadata version: {version!r}")


def ensemble_record(ensemble):
    return _normalized(dict(name=ensemble.name, num_measurements=ensemble.num_measurements,
                num_bins=ensemble.num_bins, spatial_extent=ensemble.spatial_extent,
                temporal_extent=ensemble.temporal_extent, num_streams=ensemble.num_streams or 1,
                tweak_info=ensemble.tweak_info))


def observable_record(info):
    attrs = dict(info.to_attrs())
    for label_field in ("latex_str", "element_label", "source_label", "operand_labels", "element_operand_labels", "prediction_label", "prediction_coordinate_label"):
        attrs.pop(label_field, None)
    return dict(name=info.name, index=info.index, op_type=info.op_type, re_im=info.re_im,
                ensemble=ensemble_record(info.ensemble_info), metadata=attrs)


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class RotationBinding:
    pivot_id: str
    index: int

    def __post_init__(self):
        if not isinstance(self.pivot_id, str) or not self.pivot_id:
            raise ValueError("pivot_id must be a nonempty string")
        object.__setattr__(self, "index", _nonnegative(self.index, "rotation index"))

    def to_record(self):
        return {"pivot_id": self.pivot_id, "index": self.index}


def binding_from_record(record):
    return None if record is None else RotationBinding(**record)


@dataclass(frozen=True)
class CorrelatorSource:
    correlator: CorrelatorInfo
    hermitian: bool
    subtract_vev: bool
    re_im: str
    sink_rotation: RotationBinding | None = None
    source_rotation: RotationBinding | None = None

    def __post_init__(self):
        from .correlators.keys import CorrelatorInfo

        if not isinstance(self.correlator, CorrelatorInfo):
            raise TypeError("correlator source requires a frozen CorrelatorInfo")
        if type(self.hermitian) is not bool or type(self.subtract_vev) is not bool:
            raise TypeError("correlator flags must be boolean")
        if self.re_im not in ("re", "im"):
            raise ValueError("source component must be re or im")
        for binding in (self.sink_rotation, self.source_rotation):
            if binding is not None and not isinstance(binding, RotationBinding):
                raise TypeError("rotation bindings must be RotationBinding records")

    @property
    def sector_info(self):
        a, b = self.correlator.sink.channel.to_sector(), self.correlator.source.channel.to_sector()
        return a if a == b else SectorInfo()

    @property
    def rot_idx(self):
        if (self.correlator.sink == self.correlator.source and self.sink_rotation is not None
                and self.sink_rotation == self.source_rotation):
            return self.sink_rotation.index
        return None

    def to_record(self):
        return dict(source_kind="correlator", correlator=self.correlator.to_sigmond(),
                    hermitian=self.hermitian, subtract_vev=self.subtract_vev, re_im=self.re_im,
                    sink_rotation=None if self.sink_rotation is None else self.sink_rotation.to_record(),
                    source_rotation=None if self.source_rotation is None else self.source_rotation.to_record())


@dataclass(frozen=True, eq=False)
class SeriesInfo:
    """A source identity plus named non-time coordinates; no live source object."""
    parent_identity: str
    coordinates: tuple = ()
    sector_info: SectorInfo = field(default_factory=SectorInfo)
    label: str | None = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        # Freeze/normalize nested identities supplied by callers as well.
        object.__setattr__(self, "parent_identity", _json(json.loads(self.parent_identity)))
        coords = tuple((str(name), _scalar(value)) for name, value in self.coordinates)
        if len({name for name, _ in coords}) != len(coords):
            raise ValueError("series coordinate names must be unique")
        object.__setattr__(self, "coordinates", coords)
        if not isinstance(self.sector_info, SectorInfo):
            raise TypeError("series sector_info must be a SectorInfo")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("series labels must be strings")

    @property
    def rot_idx(self):
        return None

    @property
    def latex_str(self):
        return r"\mathrm{series}" if self.label is None else self.label

    def to_record(self):
        return dict(element_kind="series", metadata_version=VERSION,
                    parent_identity=json.loads(self.parent_identity), coordinates=dict(self.coordinates),
                    sector_info=self.sector_info.to_dict())

    def __eq__(self, other):
        return isinstance(other, SeriesInfo) and identity_json(self.to_record()) == identity_json(other.to_record())

    def __hash__(self):
        return hash(identity_json(self.to_record()))


def source_from_record(record):
    _version(record)
    kind = record.get("source_kind")
    if kind == "correlator":
        from .correlators.keys import CorrelatorInfo

        return CorrelatorSource(CorrelatorInfo.from_sigmond(record["correlator"]),
                                record["hermitian"], record["subtract_vev"], record["re_im"],
                                binding_from_record(record.get("sink_rotation")),
                                binding_from_record(record.get("source_rotation")))
    if kind == "series":
        return element_from_record({**record, "element_kind": "series"})
    raise ValueError(f"Unsupported source metadata kind: {kind!r}")


@dataclass(frozen=True)
class EnergyTarget:
    """Optional known lab-energy target, snapshotted without mutable infos."""
    sector_info: SectorInfo
    ensemble_identity: str
    rot_idx: int | None = None
    ref_particle: str | None = None

    def __post_init__(self):
        if not isinstance(self.sector_info, SectorInfo):
            raise TypeError("target sector_info must be a SectorInfo")
        object.__setattr__(self, "ensemble_identity", _json(json.loads(self.ensemble_identity)))
        if self.rot_idx is not None:
            object.__setattr__(self, "rot_idx", _nonnegative(self.rot_idx, "rot_idx"))
        if self.ref_particle is not None and not isinstance(self.ref_particle, str):
            raise TypeError("target ref_particle must be a string")

    @classmethod
    def from_energy(cls, info):
        return cls(info.sector_info, _json(ensemble_record(info.ensemble_info)), info.rot_idx, info.ref_particle)

    @property
    def energy(self):
        from .energy_levels import EnergyObsInfo

        return EnergyObsInfo(sector_info=self.sector_info,
                             name="effective_energy_target" if self.sector_info.psq is None or self.sector_info.irrep is None else None,
                             ensemble_info=EnsembleInfo(**json.loads(self.ensemble_identity)),
                             energy_type="elab", rot_idx=self.rot_idx, ref_particle=self.ref_particle)

    def to_record(self):
        return dict(sector_info=self.sector_info.to_dict(),
                    ensemble=json.loads(self.ensemble_identity), rot_idx=self.rot_idx, ref_particle=self.ref_particle)


@dataclass(frozen=True)
class EffectiveEnergyInfo:
    source: CorrelatorSource | SeriesInfo
    kind: str = "forward"
    step: int = 1
    time_extent: int | None = None
    subtract_const: float = 0.0
    target: EnergyTarget | None = None

    def __post_init__(self):
        from .correlators.effective import effective_time_offset

        if not isinstance(self.source, (CorrelatorSource, SeriesInfo)):
            raise TypeError("effective-energy source must be an immutable source snapshot")
        effective_time_offset(self.kind, self.step)
        step = _nonnegative(self.step, "step")
        if step == 0:
            raise ValueError("step must be positive")
        object.__setattr__(self, "step", step)
        if self.time_extent is not None:
            extent = _nonnegative(self.time_extent, "time_extent")
            if extent == 0:
                raise ValueError("time_extent must be positive")
            object.__setattr__(self, "time_extent", extent)
        const = float(self.subtract_const)
        if not math.isfinite(const):
            raise ValueError("subtract_const must be finite")
        object.__setattr__(self, "subtract_const", const)
        if self.target is not None and not isinstance(self.target, EnergyTarget):
            raise TypeError("target must be an immutable EnergyTarget")

    @property
    def sector_info(self):
        return self.source.sector_info

    @property
    def rot_idx(self):
        return self.source.rot_idx

    @property
    def energy(self):
        return None if self.target is None else self.target.energy

    @property
    def time_offset(self):
        from .correlators.effective import effective_time_offset

        return effective_time_offset(self.kind, self.step)

    @property
    def latex_str(self):
        from .labels import effective_energy_label

        return effective_energy_label(rot_idx=self.rot_idx,
                                      ref_particle=None if self.target is None else self.target.ref_particle)

    def to_record(self):
        source = self.source.to_record()
        if isinstance(self.source, SeriesInfo):
            source = {**source, "source_kind": "series"}
            source.pop("element_kind")
        record = dict(element_kind="effective_energy", metadata_version=VERSION,
                    sector_info=self.sector_info.to_dict(), source=source, kind=self.kind,
                    step=self.step, time_extent=self.time_extent, subtract_const=self.subtract_const,
                    target=None if self.target is None else self.target.to_record())
        if isinstance(self.source, SeriesInfo):
            # The generic source already owns its explicitly supplied sector.
            record.pop("sector_info")
        return record


def element_from_record(record):
    _version(record)
    kind = record.get("element_kind")
    if kind == "derived":
        from .arithmetic import DerivedElement
        return DerivedElement.from_record(record)
    if kind == "prediction":
        from .predictions import PredictionSeriesInfo
        return PredictionSeriesInfo.from_record(record)
    if kind == "series":
        return SeriesInfo(_json(record["parent_identity"]), tuple(record.get("coordinates", {}).items()),
                          SectorInfo(**record.get("sector_info", {})))
    if kind == "effective_energy":
        target = record.get("target")
        if target is not None:
            _version(target)
        target = None if target is None else EnergyTarget(SectorInfo(**target["sector_info"]),
                                                          _json(target["ensemble"]), target.get("rot_idx"), target.get("ref_particle"))
        element = EffectiveEnergyInfo(source_from_record(record["source"]), record["kind"],
                                      record["step"], record.get("time_extent"), record["subtract_const"], target)
        if "sector_info" in record and SectorInfo(**record["sector_info"]) != element.sector_info:
            raise ValueError("Effective-energy sector annotation contradicts its source")
        return element
    raise ValueError(f"Unsupported element metadata kind: {kind!r}")


@register_obs_kind("at")
class AtObsInfo(ObservableInfo):
    """One value of an immutable element at a typed coordinate."""
    def __init__(self, element, coord, value, *, ensemble_info=INDEP_ENSEMBLE,
                 index=0, op_type="n", re_im="re", latex_str=None):
        from .arithmetic import DerivedElement
        from .predictions import PredictionSeriesInfo
        if not isinstance(element, (EffectiveEnergyInfo, SeriesInfo, DerivedElement, PredictionSeriesInfo)):
            raise TypeError("AtObsInfo needs a registered immutable element")
        if isinstance(element, PredictionSeriesInfo) and element.dim != coord:
            raise ValueError("Prediction coordinate contradicts its series dimension")
        if not isinstance(coord, str) or not coord:
            raise ValueError("coord must be a nonempty string")
        self._element, self._coord, self._value = element, coord, _scalar(value)
        if self._value is None:
            raise ValueError("at-coordinate values must be specified")
        super().__init__("", index, op_type, re_im, ensemble_info, latex_str)
        del self._sector_info
        self.name = generated_name("EffEn" if isinstance(element, EffectiveEnergyInfo) else "At", self.identity_record)

    @property
    def name(self):
        canonical = identity_json(self.identity_record)
        cached = getattr(self, "_name_cache", None)
        if cached is None or cached[0] != canonical:
            prefix = "EffEn" if isinstance(self.element, EffectiveEnergyInfo) else "At"
            self._name_cache = (canonical, generated_name(prefix, self.identity_record))
        return self._name_cache[1]

    @name.setter
    def name(self, value):
        if value != "" and value != self.name:
            raise ValueError("AtObsInfo names are generated from their identity records")

    @property
    def element(self):
        return self._element

    @property
    def coord(self):
        return self._coord

    @property
    def value(self):
        return self._value

    @property
    def time(self):
        return self.value if self.coord == "time" else None

    @property
    def sector_info(self):
        return self.element.sector_info

    @property
    def rot_idx(self):
        return self.element.rot_idx

    @property
    def identity_record(self):
        return dict(obs_kind="at", metadata_version=VERSION, coord=self.coord, value=self.value,
                    element=self.element.to_record(), context=dict(index=self.index, op_type=self.op_type,
                    re_im=self.re_im, ensemble=ensemble_record(self.ensemble_info)))

    @property
    def latex_str(self):
        if self._latex_str is not None:
            return self._latex_str
        from .labels import coordinate_label

        return coordinate_label(self.element.latex_str, self.value)

    @latex_str.setter
    def latex_str(self, value):
        self._latex_str = value

    def __eq__(self, other):
        return isinstance(other, AtObsInfo) and identity_json(self.identity_record) == identity_json(other.identity_record)

    def __hash__(self):
        return hash(identity_json(self.identity_record))

    def to_attrs(self):
        attrs = dict(obs_kind=self.obs_kind, metadata_version=VERSION, coord=self.coord, value=self.value,
                     element=self.element.to_record(), identity_record=self.identity_record)
        if self._latex_str is not None:
            attrs["latex_str"] = self._latex_str
        from .arithmetic import DerivedElement
        if isinstance(self.element, DerivedElement):
            attrs["element_operand_labels"] = list(self.element.labels)
        from .predictions import PredictionSeriesInfo
        if isinstance(self.element, PredictionSeriesInfo):
            if self.element.label is not None:
                attrs["prediction_label"] = self.element.label
            if self.element.coordinate_label is not None:
                attrs["prediction_coordinate_label"] = self.element.coordinate_label
        if isinstance(self.element, SeriesInfo) and self.element.label is not None:
            attrs["element_label"] = self.element.label
        if (isinstance(self.element, EffectiveEnergyInfo) and isinstance(self.element.source, SeriesInfo)
                and self.element.source.label is not None):
            attrs["source_label"] = self.element.source.label
        return attrs

    def public_attributes(self):
        return dict(super().public_attributes(), coord=self.coord, value=self.value, time=self.time)

    @classmethod
    def from_attrs(cls, base, attrs):
        _version(attrs, required=True)
        element = element_from_record(attrs["element"])
        from .arithmetic import DerivedElement
        if isinstance(element, DerivedElement):
            from dataclasses import replace
            element = replace(element, labels=tuple(attrs.get("element_operand_labels", ())))
        from .predictions import PredictionSeriesInfo
        if isinstance(element, PredictionSeriesInfo):
            from dataclasses import replace
            element = replace(element, label=attrs.get("prediction_label"),
                              coordinate_label=attrs.get("prediction_coordinate_label"))
        if isinstance(element, SeriesInfo) and attrs.get("element_label") is not None:
            from dataclasses import replace
            element = replace(element, label=attrs["element_label"])
        if (isinstance(element, EffectiveEnergyInfo) and isinstance(element.source, SeriesInfo)
                and attrs.get("source_label") is not None):
            from dataclasses import replace
            element = replace(element, source=replace(element.source, label=attrs["source_label"]))
        result = cls(element, attrs["coord"], attrs["value"], ensemble_info=base.ensemble_info,
                     index=base.index, op_type=base.op_type, re_im=base.re_im, latex_str=attrs.get("latex_str"))
        if "identity_record" in attrs and identity_json(attrs["identity_record"]) != identity_json(result.identity_record):
            raise ValueError("At metadata identity record contradicts its fields or ensemble context")
        if result.name != base.name:
            raise ValueError("Generated at-coordinate storage name contradicts its identity record")
        return result


def source_elements(data):
    """Validate source metadata along time and return one snapshot per other position."""
    import numpy as np
    from .correlators.observables import CorrelatorObsInfo
    from .sampling_array import ArrayElementObsInfo

    axis = data.dims.index("time")
    dims = tuple(dim for dim in data.dims if dim != "time")
    shape = tuple(data.shape[data.dims.index(dim)] for dim in dims)
    infos = np.moveaxis(data.infos, axis, -1)
    if not infos.shape[-1]:
        raise ValueError("empty time axis has no element metadata")
    output = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        coords = tuple((dim, _scalar(data.coords[dim][i]) if dim in data.coords else i)
                       for dim, i in zip(dims, idx))
        established = None
        context = None
        for t, info in enumerate(infos[idx]):
            if isinstance(info, AtObsInfo) and info.coord != "time":
                raise ValueError("at-coordinate metadata does not describe the time axis")
            cell_context = identity_json(dict(index=info.index, op_type=info.op_type, re_im=info.re_im,
                                             ensemble=ensemble_record(info.ensemble_info)))
            if isinstance(info, CorrelatorObsInfo):
                if info.key is None:
                    raise ValueError(f"cannot establish source from opaque correlator key: {info.key_error}")
                source = CorrelatorSource(info.element, info.hermitian, info.subtract_vev, info.re_im,
                                          info.sink_rotation, info.source_rotation)
            else:
                parent = info.parent if isinstance(info, ArrayElementObsInfo) else info
                record = info.element.to_record() if isinstance(info, AtObsInfo) else observable_record(parent)
                source = SeriesInfo(_json(record), coords, info.sector_info, parent.latex_str)
            if hasattr(info, "time") and info.time is not None and info.time != data.coords["time"][t]:
                raise ValueError("cell time metadata disagrees with time coordinates")
            if established is None:
                established, context = source, cell_context
            elif identity_json(source.to_record()) != identity_json(established.to_record()) or context != cell_context:
                raise ValueError(f"inconsistent source element/context along time at {dict(coords)}")
        output[idx] = established
    return output


def _sector_view(name):
    return property(lambda info: getattr(info.sector_info, name))


# A small structural convention, without an element inheritance hierarchy.
for _record_type in (CorrelatorSource, SeriesInfo, EffectiveEnergyInfo, EnergyTarget):
    for _sector_field in fields(SectorInfo):
        setattr(_record_type, _sector_field.name, _sector_view(_sector_field.name))
    _record_type.strangeness = _sector_view("S")
    _record_type.isospin = property(ObservableInfo.isospin.fget)
