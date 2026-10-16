"""
Observable metadata for correlators and VEVs.

The loader gives every ``<CorrT>`` observable a :class:`CorrelatorObsInfo` and
every ``<VEV>`` observable a :class:`VEVObsInfo`, so correlators take part in
the ordinary collection queries::

    obs.filter(obs_kind="correlator", time=8, irrep="A1um")
    obs.group_by("correlator")
    ss query corrs.hdf5 --where obs_kind=correlator psq=0 --list-attrs

The typed key is parsed on first use (labels are memoized, so this is cheap).
A name that does not parse, e.g. one using a flavor or irrep that SS does not
know yet, still loads: its ``key`` is ``None`` and :attr:`key_error` says why,
and the quantum-number attributes are ``None``.

The name is the key's canonical text when the info is built from a key, and
is generated only when first needed (a large assembled matrix creates one info
per cell).
"""

from __future__ import annotations

from ..info import (
    INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SectorInfo, register_obs_kind, sector_from_attrs,
)
from .keys import (
    CORRELATOR_TAG,
    VEV_TAG,
    CorrelatorAtTimeInfo,
    CorrelatorInfo,
    VEVInfo,
    key_tag,
    parse_key,
)
from .operators import Channel, Operator
from ..metadata import RotationBinding, binding_from_record

__all__ = ["CorrelatorObsInfo", "VEVObsInfo", "OverlapObsInfo", "observable_info_for"]


class _KeyedObsInfo(ObservableInfo):
    """An observable whose name is a typed Sigmond key (parsed lazily)."""

    _key_type: type = object

    def __init__(
        self,
        name: str | None = None,
        index: int = 0,
        op_type: str = "n",
        re_im: str = "re",
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        latex_str: str | None = None,
        *,
        key: CorrelatorAtTimeInfo | VEVInfo | None = None,
    ):
        if name is None and key is None:
            raise ValueError("give a name or a key")
        if key is not None and not isinstance(key, self._key_type):
            raise TypeError(f"key must be a {self._key_type.__name__}, got {key!r}")
        if name is not None and key is not None and parse_key(name) != key:
            raise ValueError("name and key identify different observables")
        super().__init__(name, index, op_type, re_im, ensemble_info, latex_str)
        self._key = key
        del self._sector_info

    @property
    def sector_info(self) -> SectorInfo:
        channel = self.channel
        return channel.to_sector() if channel is not None else SectorInfo()

    def to_attrs(self):
        attrs = {"obs_kind": self.obs_kind, "sector_info": self.sector_info.to_dict()}
        if self._latex_str is not None:
            attrs["latex_str"] = self._latex_str
        return attrs

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self._key.to_sigmond()
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        # A renamed observable is re-parsed on next use.
        self._name = value
        self._key = None
        self._key_error = None
        if hasattr(self, "_sink_rotation"):
            self._sink_rotation = self._source_rotation = None

    @property
    def key(self) -> CorrelatorAtTimeInfo | VEVInfo | None:
        """The parsed key, or ``None`` if the name is not a valid key."""
        if self._key is None and self._key_error is None:
            try:
                key = parse_key(self._name)
                if not isinstance(key, self._key_type):
                    raise ValueError(f"not a {self._key_type.__name__}")
                self._key = key
            except (ValueError, TypeError) as e:
                self._key_error = e
        return self._key

    @property
    def key_error(self) -> Exception | None:
        """Why the name did not parse (``None`` if it did)."""
        _ = self.key
        return self._key_error

    @classmethod
    def from_attrs(cls, base: ObservableInfo, attrs) -> _KeyedObsInfo:
        result = cls(base.name, base.index, base.op_type, base.re_im, base.ensemble_info)
        annotated = sector_from_attrs(attrs)
        if result.key is not None and annotated != SectorInfo() and annotated != result.sector_info:
            raise ValueError("Sector annotation contradicts native key")
        if attrs.get("latex_str") is not None:
            result.latex_str = attrs["latex_str"]
        return result

    def _get(self, fn):
        key = self.key
        return None if key is None else fn(key)

    def __repr__(self) -> str:
        key = self.key
        shown = str(key) if key is not None else self.name
        return f"{type(self).__name__}({shown!r}, ensemble='{self.ensemble_info}')"


@register_obs_kind("correlator", "corr")
class CorrelatorObsInfo(_KeyedObsInfo):
    """A ``<CorrT>`` observable: one correlator ``C_ij(t)``.

    Attributes follow the key: ``sink``, ``source``, ``correlator``, ``time``,
    ``hermitian`` and ``subtract_vev`` (the stored-key flags), and the
    common physical sector. Both source and sink channels remain available;
    ``momentum`` and the legacy ``channel`` accessor describe the sink.
    """

    _key_type = CorrelatorAtTimeInfo

    def __init__(self, *args, sink_rotation=None, source_rotation=None, **kwargs):
        super().__init__(*args, **kwargs)
        for binding in (sink_rotation, source_rotation):
            if binding is not None and not isinstance(binding, RotationBinding):
                raise TypeError("rotation bindings must be RotationBinding records")
        self._sink_rotation, self._source_rotation = sink_rotation, source_rotation

    @property
    def element(self) -> CorrelatorInfo | None:
        return self.correlator

    @property
    def sink_rotation(self):
        return self._sink_rotation

    @property
    def source_rotation(self):
        return self._source_rotation

    @property
    def rot_idx(self):
        if self.key is not None and self.sink == self.source and self.sink_rotation is not None and self.sink_rotation == self.source_rotation:
            return self.sink_rotation.index
        return None

    def with_rotation(self, sink, source):
        for binding in (sink, source):
            if binding is not None and not isinstance(binding, RotationBinding):
                raise TypeError("rotation bindings must be RotationBinding records")
        result = self.copy()
        result._sink_rotation, result._source_rotation = sink, source
        return result

    def to_attrs(self):
        return dict(super().to_attrs(),
                    sink_rotation=None if self.sink_rotation is None else self.sink_rotation.to_record(),
                    source_rotation=None if self.source_rotation is None else self.source_rotation.to_record())

    @classmethod
    def from_attrs(cls, base, attrs):
        result = super().from_attrs(base, attrs)
        return result.with_rotation(binding_from_record(attrs.get("sink_rotation")),
                                    binding_from_record(attrs.get("source_rotation")))

    def __eq__(self, other):
        return (isinstance(other, CorrelatorObsInfo) and super().__eq__(other)
                and self.sink_rotation == other.sink_rotation and self.source_rotation == other.source_rotation)

    def __hash__(self):
        return hash((super().__hash__(), self.sink_rotation, self.source_rotation))

    @property
    def sink(self) -> Operator | None:
        return self._get(lambda k: k.sink)

    @property
    def source(self) -> Operator | None:
        return self._get(lambda k: k.source)

    @property
    def correlator(self) -> CorrelatorInfo | None:
        return self._get(lambda k: k.correlator)

    @property
    def time(self) -> int | None:
        return self._get(lambda k: k.time)

    @property
    def hermitian(self) -> bool | None:
        return self._get(lambda k: k.hermitian)

    @property
    def subtract_vev(self) -> bool | None:
        return self._get(lambda k: k.subtract_vev)

    @property
    def channel(self) -> Channel | None:
        return self._get(lambda k: k.sink.channel)

    @property
    def sink_channel(self) -> Channel | None:
        return self._get(lambda k: k.sink.channel)

    @property
    def source_channel(self) -> Channel | None:
        return self._get(lambda k: k.source.channel)

    @property
    def sector_info(self) -> SectorInfo:
        if self.sink_channel is None or self.source_channel is None:
            return SectorInfo()
        sink, source = self.sink_channel.to_sector(), self.source_channel.to_sector()
        return sink if sink == source else SectorInfo()

    @property
    def momentum(self) -> tuple[int, int, int] | None:
        return self._get(lambda k: k.sink.channel.momentum)

    @property
    def latex_str(self) -> str:
        if self._latex_str is not None:
            return self._latex_str
        key = self.key
        if key is None:
            return super().latex_str
        from ..labels import correlator_label

        return correlator_label(self.element, key.time, rot_idx=self.rot_idx)

    @latex_str.setter
    def latex_str(self, value: str) -> None:
        self._latex_str = value


@register_obs_kind("vev")
class VEVObsInfo(_KeyedObsInfo):
    """A ``<VEV>`` observable: the vacuum expectation value of ``operator``."""

    _key_type = VEVInfo

    @property
    def operator(self) -> Operator | None:
        return self._get(lambda k: k.operator)

    @property
    def channel(self) -> Channel | None:
        return self._get(lambda k: k.operator.channel)

    @property
    def latex_str(self) -> str:
        if self._latex_str is not None:
            return self._latex_str
        key = self.key
        if key is None:
            return super().latex_str
        return rf"\langle {key.operator.latex_str} \rangle"

    @latex_str.setter
    def latex_str(self, value: str) -> None:
        self._latex_str = value


_BY_TAG = {CORRELATOR_TAG: CorrelatorObsInfo, VEV_TAG: VEVObsInfo}


def observable_info_for(
    name: str, re_im: str = "re", ensemble_info: EnsembleInfo = INDEP_ENSEMBLE
) -> ObservableInfo:
    """The info type for a raw XML observable name: correlator, VEV, or plain."""
    cls = _BY_TAG.get(key_tag(name), ObservableInfo)
    return cls(name, 0, "n", re_im, ensemble_info)


def key_of(info: ObservableInfo) -> CorrelatorAtTimeInfo | VEVInfo | None:
    """The typed key of any info whose name is a ``<CorrT>``/``<VEV>`` key, else ``None``.

    Raises ``ValueError`` for a plain info whose name looks like a key but does
    not parse (keyed infos report that through ``key_error`` instead).
    """
    if isinstance(info, _KeyedObsInfo):
        return info.key
    name = info.name
    if isinstance(name, str) and key_tag(name) in _BY_TAG:
        return parse_key(name)
    return None


@register_obs_kind("pivot_overlap")
class OverlapObsInfo(ObservableInfo):
    """A derived single-pivot overlap cell, with durable labels/conventions.

    Named like Sigmond's own Z-factor output: one name per rotated level,
    ``Z2_<pivot id>_<level>`` (``Z2N_...`` when row-normalized), with the
    operator's basis position as the observable index. The pivot id keeps
    overlaps of different pivots distinct in one file; the labels themselves
    travel in the attributes.
    """

    def __init__(self, operator, level, *, pivot_id, normalize, norm_time, metric_time,
                 diagonalize_time, ensemble_info, name=None, index=0, op_type="n", re_im="re",
                 sector_info=None):
        from .operators import parse_operator

        self.operator = parse_operator(operator)
        self.level = parse_operator(level)
        self.pivot_id = str(pivot_id)
        self.normalize = bool(normalize)
        self.norm_time = int(norm_time)
        self.metric_time = int(metric_time)
        self.diagonalize_time = int(diagonalize_time)
        name = name or f"Z2{'N' if self.normalize else ''}_{self.pivot_id}_{self.level.index}"
        super().__init__(name, index, op_type, re_im, ensemble_info, sector_info=sector_info)

    def to_attrs(self):
        return dict(super().to_attrs(), operator=self.operator.coordinate,
                    level=self.level.coordinate, pivot_id=self.pivot_id, normalize=self.normalize,
                    norm_time=self.norm_time, metric_time=self.metric_time,
                    diagonalize_time=self.diagonalize_time)

    @classmethod
    def from_attrs(cls, base, attrs):
        return cls(attrs["operator"], attrs["level"], pivot_id=attrs["pivot_id"],
                   normalize=attrs["normalize"],
                   norm_time=attrs["norm_time"], metric_time=attrs["metric_time"],
                   diagonalize_time=attrs["diagonalize_time"],
                   ensemble_info=base.ensemble_info, name=base.name, index=base.index,
                   op_type=base.op_type, re_im=base.re_im,
                   sector_info=sector_from_attrs(attrs, default=base.sector_info))
