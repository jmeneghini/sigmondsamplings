"""Immutable sampled-prediction series, independent of a model callable."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, fields
from numbers import Real

import numpy as np

from .info import INDEP_ENSEMBLE, ObservableInfo, SectorInfo
from .metadata import (
    VERSION, AtObsInfo, EnergyTarget, _json, _scalar, _version, identity_json,
    observable_record,
)


@dataclass(frozen=True, eq=False)
class PredictionSeriesInfo:
    """A fixed prediction-set identity and optional explicit physical context.

    Labels are presentation only. No model, callable, parameter container or
    sample buffer is retained by this descriptor.
    """
    series_id: str
    dim: str = "time"
    sector_info: SectorInfo = field(default_factory=SectorInfo)
    target: EnergyTarget | None = None
    label: str | None = field(default=None, compare=False, hash=False)
    coordinate_label: str | None = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        if not isinstance(self.series_id, str) or not re.fullmatch(r"[0-9a-f]{64}", self.series_id):
            raise ValueError("prediction series_id must be a SHA-256 identifier")
        if not isinstance(self.dim, str) or not self.dim or self.dim == "sample":
            raise ValueError("prediction dimension must be a nonempty spatial dimension")
        if not isinstance(self.sector_info, SectorInfo):
            raise TypeError("prediction sector must be a SectorInfo")
        if self.target is not None:
            if not isinstance(self.target, EnergyTarget):
                raise TypeError("prediction target must be an immutable EnergyTarget")
            if not self.sector_info.compatible_with(self.target.sector_info):
                raise ValueError("prediction sector contradicts its explicit target")
        if any(value is not None and not isinstance(value, str) for value in (self.label, self.coordinate_label)):
            raise TypeError("prediction labels must be strings")

    @property
    def rot_idx(self):
        # A prediction is not itself a fitted level or a pivot mode.
        return None

    @property
    def latex_str(self):
        return r"\mathrm{prediction}({VAR})" if self.label is None else self.label

    def to_record(self):
        return dict(element_kind="prediction", metadata_version=VERSION, series_id=self.series_id,
                    dim=self.dim, coordinate_convention="numeric", sector_info=self.sector_info.to_dict(),
                    target=None if self.target is None else self.target.to_record())

    @classmethod
    def from_record(cls, record):
        _version(record, required=True)
        if record.get("coordinate_convention") != "numeric":
            raise ValueError("Unsupported prediction coordinate convention")
        target = record.get("target")
        if target is not None:
            _version(target)
            target = EnergyTarget(SectorInfo(**target["sector_info"]), _json(target["ensemble"]),
                                  target.get("rot_idx"), target.get("ref_particle"))
        return cls(record["series_id"], record["dim"], SectorInfo(**record["sector_info"]), target)

    def __eq__(self, other):
        return isinstance(other, PredictionSeriesInfo) and identity_json(self.to_record()) == identity_json(other.to_record())

    def __hash__(self):
        return hash(identity_json(self.to_record()))


for _field in fields(SectorInfo):
    setattr(PredictionSeriesInfo, _field.name, property(lambda self, name=_field.name: getattr(self.sector_info, name)))
PredictionSeriesInfo.strangeness = property(lambda self: self.sector_info.S)
PredictionSeriesInfo.isospin = property(ObservableInfo.isospin.fget)


def _normalized_buffer(values):
    """Canonical float64/complex128 little-endian C-order bytes, with one NaN encoding."""
    values = np.asarray(values)
    if values.dtype.kind not in "biufc":
        raise TypeError("Predictions must have numeric values")
    array = np.array(values, dtype="<c16" if np.iscomplexobj(values) else "<f8", order="C", copy=True)
    if np.iscomplexobj(array):
        array.real[np.isnan(array.real)] = np.nan
        array.imag[np.isnan(array.imag)] = np.nan
    else:
        array[np.isnan(array)] = np.nan
    return array


def _prediction_descriptor(predictions, coordinates, dim, *, sector_info=None, target=None,
                           label=None, coordinate_label=None):
    """Generate the ID once; subsequent metadata hashing never reads predictions."""
    sector = SectorInfo() if sector_info is None else sector_info
    if not isinstance(sector, SectorInfo):
        raise TypeError("prediction sector must be a SectorInfo")
    if target is not None and not isinstance(target, EnergyTarget):
        raise TypeError("prediction target must be an immutable EnergyTarget")
    if target is not None and sector_info is None:
        sector = target.sector_info
    buffer = _normalized_buffer(np.stack([p.data for p in predictions]))
    sampling = predictions[0].sampling_info
    convention = None if sampling is None else dict(method=sampling.method, num_resamplings=sampling.num_resamplings,
                                                     seed=sampling.seed, boot_skip=sampling.boot_skip,
                                                     extra_params=sampling.extra_params)
    record = dict(metadata_version=VERSION, dim=dim, coordinates=[_scalar(c) for c in coordinates],
                  shape=list(buffer.shape), dtype=buffer.dtype.str,
                  buffer_fingerprint=hashlib.sha256(buffer.tobytes(order="C")).hexdigest(),
                  sampling=convention, inputs=[observable_record(p.observable_info) for p in predictions],
                  sector_info=sector.to_dict(), target=None if target is None else target.to_record())
    series_id = hashlib.sha256(identity_json(record).encode()).hexdigest()
    return PredictionSeriesInfo(series_id, dim, sector, target, label, coordinate_label)


def _wrap_predictions(predictions, coordinates, dim, descriptor):
    """Wrap existing scalar predictions with at-coordinate cells; never reevaluate."""
    from .data import SigmondData
    from .sampling_array import ArrayObsInfo
    predictions = list(predictions)
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 1 or len(predictions) != len(coordinates) or not predictions:
        raise ValueError("Predictions require a nonempty matching one-dimensional coordinate array")
    if not isinstance(descriptor, PredictionSeriesInfo) or descriptor.dim != dim:
        raise ValueError("Prediction descriptor dimension does not match coordinates")
    if not all(isinstance(c, Real) and not isinstance(c, (bool, np.bool_)) and np.isfinite(c) for c in coordinates):
        raise ValueError("Prediction coordinates must be finite real numbers")
    if len(set(coordinates.tolist())) != len(coordinates):
        raise ValueError("Prediction coordinates must be unique")
    if any(p.ndim for p in predictions):
        raise ValueError("Prediction wrapping requires scalar samplings")
    cells = [p.with_observable_info(AtObsInfo(descriptor, dim, c, ensemble_info=p.ensemble_info,
                        latex_str=("" if descriptor.label == "" else
                            getattr(p.observable_info, "_latex_str", None) if descriptor.label is None else None)))
             for p,c in zip(predictions, coordinates)]
    ensemble = predictions[0].ensemble_info if all(p.ensemble_info == predictions[0].ensemble_info for p in predictions) else INDEP_ENSEMBLE
    parent = ArrayObsInfo("prediction_" + descriptor.series_id, ensemble_info=ensemble,
                          sector_info=descriptor.sector_info, latex_str=descriptor.label)
    return SigmondData.from_samplings(cells, obs_info=parent, dim_names=(dim,), coords={dim:coordinates})
