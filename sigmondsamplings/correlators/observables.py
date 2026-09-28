"""
Observable metadata for correlators and VEVs.

The loader gives every ``<CorrT>`` observable a :class:`CorrelatorObsInfo` and
every ``<VEV>`` observable a :class:`VEVObsInfo`, so correlators take part in
the ordinary collection queries::

    obs.filter(obs_kind="correlator", time=8, irrep="A1um")
    obs.group_by("pair")
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

from ..info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, register_obs_kind
from .keys import CORRELATOR_TAG, VEV_TAG, CorrelatorKey, VEVKey, key_tag, parse_key

__all__ = ["CorrelatorObsInfo", "VEVObsInfo", "observable_info_for"]


class _KeyedObsInfo(ObservableInfo):
    """An observable whose name is a typed Sigmond key (parsed lazily)."""

    _key_type: type = object
    _reconstructs_own_label = True

    def __init__(
        self,
        name: str | None = None,
        index: int = 0,
        op_type: str = "n",
        re_im: str = "re",
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        latex_str: str | None = None,
        *,
        key=None,
    ):
        if name is None and key is None:
            raise ValueError("give a name or a key")
        if key is not None and not isinstance(key, self._key_type):
            raise TypeError(f"key must be a {self._key_type.__name__}, got {key!r}")
        super().__init__(name, index, op_type, re_im, ensemble_info, latex_str)
        self._key = key

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

    @property
    def key(self):
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
        return cls(base.name, base.index, base.op_type, base.re_im, base.ensemble_info)

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

    Attributes follow the key: ``sink``, ``source``, ``pair``, ``time``,
    ``hermitian`` and ``subtract_vev`` (the stored-key flags), and the
    channel's ``irrep``, ``psq``, ``momentum``, ``isospin`` and ``strangeness``
    (from the sink).
    """

    _key_type = CorrelatorKey

    sink = property(lambda self: self._get(lambda k: k.sink))
    source = property(lambda self: self._get(lambda k: k.source))
    pair = property(lambda self: self._get(lambda k: k.pair))
    time = property(lambda self: self._get(lambda k: k.time))
    hermitian = property(lambda self: self._get(lambda k: k.hermitian))
    subtract_vev = property(lambda self: self._get(lambda k: k.subtract_vev))
    channel = property(lambda self: self._get(lambda k: k.sink.channel))
    irrep = property(lambda self: self._get(lambda k: k.sink.channel.irrep))
    psq = property(lambda self: self._get(lambda k: k.sink.channel.momentum_squared))
    momentum = property(lambda self: self._get(lambda k: k.sink.channel.momentum))
    isospin = property(lambda self: self._get(lambda k: k.sink.channel.isospin))
    strangeness = property(lambda self: self._get(lambda k: k.sink.channel.strangeness))

    @property
    def latex_str(self) -> str:
        if self._latex_str:
            return self._latex_str
        key = self.key
        if key is None:
            return super().latex_str
        return rf"C[{key.sink.latex_str};\,{key.source.latex_str}]({key.time})"

    @latex_str.setter
    def latex_str(self, value: str) -> None:
        self._latex_str = value


@register_obs_kind("vev")
class VEVObsInfo(_KeyedObsInfo):
    """A ``<VEV>`` observable: the vacuum expectation value of ``operator``."""

    _key_type = VEVKey

    operator = property(lambda self: self._get(lambda k: k.operator))
    channel = property(lambda self: self._get(lambda k: k.operator.channel))
    irrep = property(lambda self: self._get(lambda k: k.operator.channel.irrep))
    psq = property(lambda self: self._get(lambda k: k.operator.channel.momentum_squared))

    @property
    def latex_str(self) -> str:
        if self._latex_str:
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


def key_of(info: ObservableInfo) -> CorrelatorKey | VEVKey | None:
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
