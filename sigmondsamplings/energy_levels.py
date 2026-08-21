"""
Energy level specializations for ObservableInfo and SigmondSampling.

Energy levels are identified by a momentum-irrep sector, an energy type, and -
for non-interacting levels - the particles in the decay channel. This module
provides :class:`EnergyObsInfo` and its single hadron specialization
:class:`SHEnergyObsInfo`, along with the machinery to recover those attributes
from an observable name (:func:`parse_energy_attributes`,
:func:`create_energy_obs_info`) or from HDF5 attrs (:func:`energy_obs_from_attrs`).
"""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from slat import (
    IRREP_LATEX_MAP,
    PARTICLE_LATEX_MAP,
    get_energy_type_latex_str,
    get_irrep_latex_str,
    get_particle_latex_str,
    resolve_particle_name,
)

from .info import (
    INDEP_ENSEMBLE,
    EnsembleInfo,
    ObservableInfo,
    obs_kind_class,
    register_obs_kind,
)

logger = logging.getLogger(__name__)

ENERGY_TYPES = ("elab", "ecm", "delab", "decm", "qcmsq")
"""tuple of str: Every recognized energy type."""

SHIFT_ENERGY_TYPES = ("delab", "decm")
"""tuple of str: The energy types measured as a shift from a non-interacting level."""


class Particle:
    """
    A particle with an optional momentum specification.

    Parameters
    ----------
    name : str
        Particle name (e.g. ``'pi'``, ``'rho'``, ``'K'``). Normalized to its
        canonical spelling by :func:`slat.resolve_particle_name`.
    psq : int, optional
        Momentum integer squared (:math:`d^2`) for this particle. Required for
        non-interacting pair specifications in shift energy types.

    Raises
    ------
    ValueError
        If ``name`` is not a recognized particle, or if ``psq`` is negative.

    Notes
    -----
    Because ``name`` is canonicalized, ``Particle('PI')`` and ``Particle('pi')``
    are the same particle and produce the same canonical observable name (and
    therefore the same HDF5 key).

    Examples
    --------
    >>> pi_at_rest = Particle('pi', psq=0)
    >>> rho_moving = Particle('rho', psq=1)  # momentum d^2 = 1
    >>> generic_pi = Particle('pi')  # no momentum, for non-shift types
    """

    def __init__(self, name: str, psq: int | None = None):
        if PARTICLE_LATEX_MAP:
            canonical = resolve_particle_name(name)
            if canonical is None:
                raise ValueError(f"Invalid particle name: {name}")
            name = canonical
        if psq is not None and psq < 0:
            raise ValueError(f"psq must be non-negative, got {psq}")

        self.name = name
        self.psq = psq

    @classmethod
    def from_string(cls, particle_str: str) -> "Particle":
        """
        Create a Particle from its string representation.

        Parameters
        ----------
        particle_str : str
            Name optionally followed by a parenthesized momentum, e.g. ``'pi'``
            (no momentum), ``'rho(1)'`` (:math:`d^2 = 1`), or ``'pi+(2)'``
            (charge suffixes are part of the name).

        Returns
        -------
        Particle
            The parsed particle.

        Raises
        ------
        ValueError
            If ``particle_str`` does not match the ``name`` / ``name(psq)`` form,
            or if the name is not a recognized particle.

        Notes
        -----
        The name itself is validated by :func:`slat.resolve_particle_name` rather
        than by the pattern, so every spelling in ``PARTICLE_LATEX_MAP``
        round-trips through :meth:`__str__`.
        """
        pattern = r"^([^()]+)(?:\((\d+)\))?$"
        match = re.match(pattern, particle_str)
        if not match:
            raise ValueError(f"Invalid particle string: {particle_str}")

        name = match.group(1)
        psq_str = match.group(2)
        psq = int(psq_str) if psq_str is not None else None

        return cls(name, psq)

    @property
    def has_momentum(self) -> bool:
        """bool: Whether a momentum was specified for this particle."""
        return self.psq is not None

    def latex_str(self) -> str:
        """
        Render the particle as LaTeX.

        Returns
        -------
        str
            The particle's LaTeX label, suffixed with ``(d^2=psq)`` when a
            momentum is specified.
        """
        if not PARTICLE_LATEX_MAP:
            return self.name
        base_latex = get_particle_latex_str(self.name)
        if self.has_momentum:
            return f"{base_latex}(d^2={self.psq})"
        return base_latex

    def __str__(self) -> str:
        """str: Round-trippable form (inverse of :meth:`from_string`): ``'pi'`` or ``'pi(1)'``."""
        if self.has_momentum:
            return f"{self.name}({self.psq})"
        return self.name

    def __repr__(self) -> str:
        if self.has_momentum:
            return f"Particle('{self.name}', psq={self.psq})"
        return f"Particle('{self.name}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            # Allow comparison with string for backward compatibility
            return self.name == (resolve_particle_name(other) or other)
        if not isinstance(other, Particle):
            return False
        return self.name == other.name and self.psq == other.psq

    @property
    def _momentum_key(self) -> tuple[int, int]:
        """tuple of int: Ordering key placing momentum-less particles first."""
        return (0, 0) if self.psq is None else (1, self.psq)

    def __lt__(self, other) -> bool:
        """Order by momentum; particles without momentum sort first."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self._momentum_key < other._momentum_key

    def __gt__(self, other) -> bool:
        """Order by momentum; particles with momentum sort after those without."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self._momentum_key > other._momentum_key

    def __le__(self, other) -> bool:
        """Less than or equal comparison based on momentum."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison based on momentum."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self.__gt__(other) or self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.name, self.psq))


_DEFAULT_DELIMITERS = r"[_\.\s/|]"
"""str: Regex character class for the delimiters separating tokens in a name."""


class _BoundaryPatterns:
    """
    Builds regex patterns anchored on delimiter boundaries.

    Parameters
    ----------
    delimiters : str, optional
        Regex character class matching the delimiters that separate tokens in an
        observable name.
    """

    def __init__(self, delimiters: str = _DEFAULT_DELIMITERS):
        self._delimiters = delimiters

    def wrap(self, pattern: str) -> str:
        """
        Wrap a pattern so it only matches a whole delimited token.

        Parameters
        ----------
        pattern : str
            Regex fragment to anchor.

        Returns
        -------
        str
            ``pattern`` preceded by a start-or-delimiter group and followed by a
            delimiter-or-end group.
        """
        return f"(?:^|{self._delimiters}){pattern}(?:{self._delimiters}|$)"


def _parse_energy_type(name: str, bounds: _BoundaryPatterns) -> str | None:
    """
    Extract the energy type from an observable name.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the match on.

    Returns
    -------
    str or None
        The matched member of :data:`ENERGY_TYPES`, lowercased, or ``None``.
    """
    pattern = bounds.wrap("(" + "|".join(ENERGY_TYPES) + ")")
    match = re.search(pattern, name, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _parse_psq(name: str, bounds: _BoundaryPatterns) -> int | None:
    """
    Extract the PSQ value from an observable name.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the match on.

    Returns
    -------
    int or None
        The momentum squared parsed from a ``PSQ4``/``psq4``/``P4``/``p4`` or
        ``PSQ=4`` token, or ``None`` if no such token is present.
    """
    # Try PSQ4, psq4, P4, p4 formats
    pattern = bounds.wrap(r"(?:psq|p)(\d+)")
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try PSQ=4 format
    pattern_eq = bounds.wrap(r"PSQ=(\d+)")
    match = re.search(pattern_eq, name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_level_index(name: str, bounds: _BoundaryPatterns) -> int | None:
    """
    Extract the level index from an observable name.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the match on.

    Returns
    -------
    int or None
        The first standalone number, or ``None`` if there is none. PSQ digits are
        attached to their ``PSQ``/``P`` prefix and so are never standalone.
    """
    match = re.search(bounds.wrap(r"(\d+)"), name)
    return int(match.group(1)) if match else None


def _parse_reference_mode(name: str, bounds: _BoundaryPatterns) -> str | None:
    """
    Detect whether an observable name indicates reference mode.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the match on.

    Returns
    -------
    str or None
        ``'ref'`` if a ``ref`` token is present, otherwise ``None``.
    """
    pattern = bounds.wrap("ref")
    return "ref" if re.search(pattern, name, re.IGNORECASE) else None


def _parse_irrep(name: str, bounds: _BoundaryPatterns) -> str | None:
    """
    Extract the irrep from an observable name.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the match on.

    Returns
    -------
    str or None
        The matched irrep, respelled with the exact case used by
        ``IRREP_LATEX_MAP``, or ``None`` if no irrep is present.
    """
    if not IRREP_LATEX_MAP:
        return None

    irrep_names = list(IRREP_LATEX_MAP.keys())
    pattern = bounds.wrap("(" + "|".join(re.escape(i) for i in irrep_names) + ")")
    match = re.search(pattern, name, re.IGNORECASE)

    if not match:
        return None

    # Find exact case match from shared label constants.
    matched_irrep = match.group(1)
    for irrep_key in irrep_names:
        if irrep_key.lower() == matched_irrep.lower():
            return irrep_key
    return matched_irrep


def _particle_alternation() -> str:
    """
    Build a regex alternation over the particle vocabulary.

    Returns
    -------
    str
        A ``|``-joined alternation of every name in ``PARTICLE_LATEX_MAP``.

    Notes
    -----
    Multi-character names match case-insensitively via a scoped ``(?i:...)``
    group; single-letter abbreviations stay case-sensitive so a stray ``n``/``s``
    token is not read as a particle and ``K``/``k`` stay distinct. Longest names
    come first so ``pion`` wins over ``pi``.
    """
    names = sorted(PARTICLE_LATEX_MAP, key=lambda n: len(n), reverse=True)
    return "|".join(
        f"(?i:{re.escape(n)})" if len(n) > 1 else re.escape(n) for n in names
    )


def _parse_particles(name: str, bounds: _BoundaryPatterns) -> tuple[list[str], int | None]:
    """
    Extract particle names, and any PSQ carried by ``particle(psq)`` notation.

    Parameters
    ----------
    name : str
        Observable name to search.
    bounds : _BoundaryPatterns
        Delimiter boundaries to anchor the matches on.

    Returns
    -------
    particles : list of str
        Canonicalized particle names, in order of appearance and deduplicated.
    psq_from_particle : int or None
        The PSQ value found in ``particle(N)`` notation, or ``None`` if no
        particle carried one.
    """
    if not PARTICLE_LATEX_MAP:
        return [], None

    particles = []
    psq_from_particle = None

    # Match particle(psq) pattern to extract both particle and optional PSQ
    pattern = bounds.wrap("(" + _particle_alternation() + r")(?:\((\d+)\))?")

    for match in re.finditer(pattern, name):
        # Canonicalize so case variants collapse onto one spelling.
        particle = resolve_particle_name(match.group(1))
        psq_str = match.group(2)

        if particle is not None and particle not in particles:
            particles.append(particle)

        if psq_str is not None:
            psq_from_particle = int(psq_str)

    return particles, psq_from_particle


def parse_energy_attributes(name: str, delimiters: str = _DEFAULT_DELIMITERS) -> dict[str, Any]:
    """
    Parse energy level attributes out of an observable name.

    Parameters
    ----------
    name : str
        Observable name to parse.
    delimiters : str, optional
        Regex character class for the delimiters separating tokens. Defaults to
        underscore, dot, whitespace, slash, and pipe.

    Returns
    -------
    dict
        Any of the keys ``'energy_type'``, ``'psq'``, ``'level_index'``,
        ``'ref_particle'``, ``'irrep'``, and ``'particles'`` that could be parsed.
        Keys whose attribute is absent from ``name`` are omitted entirely.

    Notes
    -----
    A PSQ given in ``particle(N)`` notation takes precedence over one given as a
    standalone ``PSQ``/``P`` token.
    """
    bounds = _BoundaryPatterns(delimiters)
    result = {}

    # Parse each attribute using specialized functions
    if energy_type := _parse_energy_type(name, bounds):
        result["energy_type"] = energy_type

    # Check for None explicitly since PSQ can be 0 (which is falsy)
    psq = _parse_psq(name, bounds)
    if psq is not None:
        result["psq"] = psq

    # Check for None explicitly since level_index can be 0 (which is falsy)
    level_index = _parse_level_index(name, bounds)
    if level_index is not None:
        result["level_index"] = level_index

    if ref_particle := _parse_reference_mode(name, bounds):
        result["ref_particle"] = ref_particle

    if irrep := _parse_irrep(name, bounds):
        result["irrep"] = irrep

    # Parse particles (may also extract PSQ from particle notation)
    particles, psq_from_particle = _parse_particles(name, bounds)
    if particles:
        result["particles"] = particles
    if psq_from_particle is not None:
        result["psq"] = psq_from_particle  # Override with particle-specific PSQ

    return result


def _generate_latex_str(
    energy_type: str | None = None,
    irrep: str | None = None,
    psq: int | None = None,
    particles: Sequence[Particle] | None = None,
    level_index: int | None = None,
    ref_particle: str | None = None,
    include_irrep: bool = True,
    include_psq: bool = True,
    include_particles: bool = True,
    include_level_index: bool = True,
    is_single_hadron: bool = False,
) -> str:
    r"""
    Generate the LaTeX label for an energy level observable.

    Parameters
    ----------
    energy_type : str, optional
        Member of :data:`ENERGY_TYPES`. Falls back to a bare ``E`` if omitted.
    irrep : str, optional
        Irrep label.
    psq : int, optional
        Momentum squared.
    particles : sequence of Particle, optional
        Non-interacting pair particles.
    level_index : int, optional
        Level index, rendered as a superscript on the energy symbol.
    ref_particle : str, optional
        Particle to divide by. ``'ref'`` renders as a generic
        :math:`M_{\text{ref}}`; any other value renders as that particle's mass.
        When omitted, the label is prefixed with the lattice spacing ``a_t``.
    include_irrep, include_psq, include_particles, include_level_index : bool, optional
        Whether to render each attribute. All default to ``True``.
    is_single_hadron : bool, optional
        Render the single hadron special case (see Notes).

    Returns
    -------
    str
        The LaTeX label: a base expression, a ``~`` separator, and a run of
        parenthesized attribute groups.

    Notes
    -----
    A single hadron at rest is a mass rather than an energy, so it renders as
    :math:`M_{\text{particle}}` with neither PSQ nor the particle repeated in
    the trailing attribute groups.
    """
    particle_names = [p.name for p in particles] if particles else []

    # Determine base expression
    if is_single_hadron and psq == 0 and particle_names:
        # Special case: Single hadron PSQ=0 uses m_particle, with neither PSQ nor
        # the particle repeated in further_info.
        base_expr = f"M_{{{get_particle_latex_str(particle_names[0])}}}"
        include_psq = False
        include_particles = False
    else:
        # Standard energy expression
        base_expr = (
            get_energy_type_latex_str(energy_type, level_index if include_level_index else None)
            if energy_type
            else "E"
        )
    if ref_particle is not None:
        # Reference mode: E_type/M_ref
        ref_latex = (
            r"M_{\text{ref}}"
            if ref_particle == "ref"
            else f"M_{{{get_particle_latex_str(ref_particle)}}}"
        )
        base_expr = f"{base_expr}/{ref_latex}"
    else:
        # Standard mode: a_t E_type
        base_expr = f"a_t {base_expr}"

    # Build further info (unified for all cases)
    further_info = []
    if include_irrep and irrep:
        further_info.append("(" + get_irrep_latex_str(irrep) + ")")
    if include_psq and psq is not None:
        further_info.append(f"(P^2={psq})")
    if include_particles and particle_names:
        particle_strs = [get_particle_latex_str(p) for p in particle_names]
        further_info.append(f"({','.join(particle_strs)})")

    return base_expr + "~" + "".join(further_info)


def _shared_energy_attrs(base: ObservableInfo, attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Identity and energy fields common to every energy type, as constructor kwargs."""
    return dict(
        name=base.name,
        index=base.index,
        ensemble_info=base.ensemble_info,
        irrep=attrs.get("irrep"),
        psq=int(attrs["psq"]) if "psq" in attrs else None,
        energy_type=attrs.get("energy_type"),
        ref_particle=attrs.get("ref_particle"),
    )


def _ni_pairs_from_attrs(attrs: Mapping[str, Any]) -> list["Particle"]:
    """Parse the ``ni_pairs`` attr back into particles (empty when absent)."""
    return [Particle.from_string(str(s)) for s in attrs.get("ni_pairs", [])]


@register_obs_kind("energy")
class EnergyObsInfo(ObservableInfo):
    """
    Observable info for an energy level, carrying irrep, PSQ, and energy type.

    Parameters
    ----------
    name : str, optional
        Observable name. Defaults to :attr:`canonical_name`, which requires
        ``irrep``, ``psq``, and ``energy_type`` to be given.
    index : int, optional
        Observable index.
    ensemble_info : EnsembleInfo, optional
        Ensemble this observable belongs to.
    irrep : str, optional
        Irrep label; must be a key of ``IRREP_LATEX_MAP``.
    psq : int, optional
        Momentum squared; must be non-negative.
    energy_type : str, optional
        Member of :data:`ENERGY_TYPES`.
    particles : sequence of Particle, optional
        Non-interacting pair particles, each with its own momentum. Required for
        the shift energy types (see :attr:`needs_ni_pair`).
    level_index : int, optional
        Index of this level within its momentum-irrep sector.
    ref_particle : str, optional
        Particle whose mass this level is expressed relative to, or the literal
        ``'ref'`` for a generic reference.

    Raises
    ------
    ValueError
        If any argument fails validation, or if ``name`` is omitted and the
        remaining attributes are too incomplete to build a canonical name. The
        message repeats every input to make the failure diagnosable in bulk loads.

    See Also
    --------
    SHEnergyObsInfo : The single hadron specialization.
    create_energy_obs_info : Build one of these by parsing an observable name.
    """

    is_energy = True
    _reconstructs_own_label = True

    _repr_fields = ("irrep", "psq", "energy_type", "level_index", "ref_particle", "particles")

    def __init__(
        self,
        name: str | None = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        irrep: str | None = None,
        psq: int | None = None,
        energy_type: str | None = None,
        particles: Sequence[Particle] | None = None,
        level_index: int | None = None,
        ref_particle: str | None = None,
    ):
        try:
            # Validate inputs
            if energy_type and energy_type not in ENERGY_TYPES:
                raise ValueError(f"Invalid energy_type: {energy_type}")
            if psq is not None and psq < 0:
                raise ValueError(f"PSQ must be non-negative: {psq}")
            if irrep and irrep not in IRREP_LATEX_MAP:
                raise ValueError(f"Invalid irrep: {irrep}")
            for i, particle in enumerate(particles or ()):
                if not isinstance(particle, Particle):
                    raise TypeError(f"particles[{i}] must be Particle object, got {type(particle)}")

            if ref_particle and ref_particle != "ref" and PARTICLE_LATEX_MAP:
                canonical_ref = resolve_particle_name(ref_particle)
                if canonical_ref is None:
                    raise ValueError(f"Invalid ref_particle: {ref_particle}")
                ref_particle = canonical_ref

            # Store attributes for canonical name generation
            self.irrep = irrep
            self.psq = psq
            self.energy_type = energy_type
            self.particles = tuple(particles or ())
            self.level_index = level_index
            self.ref_particle = ref_particle

            # Auto-generate name if needed using canonical form
            super().__init__(
                name if name is not None else self.canonical_name,
                index,
                "n",
                "re",
                ensemble_info,
            )
        except Exception as e:
            raise ValueError(
                f"Error initializing EnergyObsInfo: {e}. Inputs were: "
                f"name={name}, index={index}, ensemble_info={ensemble_info}, "
                f"irrep={irrep}, psq={psq}, energy_type={energy_type}, "
                f"particles={particles}, level_index={level_index}, ref_particle={ref_particle}"
            ) from e

    @property
    def _default_latex_params(self) -> dict:
        """dict: Default keyword arguments for :func:`_generate_latex_str`."""
        return {
            "energy_type": self.energy_type,
            "irrep": self.irrep,
            "psq": self.psq,
            "particles": self.particles,
            "level_index": self.level_index,
            "ref_particle": self.ref_particle,
        }

    @property
    def latex_str(self) -> str:
        """str: LaTeX representation, generated from the current attributes."""
        return _generate_latex_str(**self._default_latex_params)

    def specify_latex_str(self, **kwargs) -> str:
        """
        Generate a LaTeX label, overriding some of this observable's attributes.

        Parameters
        ----------
        **kwargs
            Any parameter of :func:`_generate_latex_str` - ``energy_type``,
            ``irrep``, ``psq``, ``particles``, ``level_index``, ``ref_particle``,
            ``include_irrep``, ``include_psq``, ``include_particles``,
            ``include_level_index``, or ``is_single_hadron``. Anything not given
            falls back to this observable's own value.

        Returns
        -------
        str
            The LaTeX label.

        See Also
        --------
        latex_str : The same label with no overrides.
        """
        # override defaults with any provided kwargs
        kwargs_in = self._default_latex_params.copy()
        kwargs_in.update(kwargs)
        return _generate_latex_str(**kwargs_in)

    @classmethod
    def from_observable_info(cls, obs_info: ObservableInfo, **energy_kwargs) -> "EnergyObsInfo":
        """
        Create an EnergyObsInfo from an existing ObservableInfo.

        Parameters
        ----------
        obs_info : ObservableInfo
            Supplies ``name``, ``index``, and ``ensemble_info``.
        **energy_kwargs
            Energy attributes to attach, as accepted by the constructor.

        Returns
        -------
        EnergyObsInfo
            The new observable info.
        """
        return cls(
            name=obs_info.name,
            index=obs_info.index,
            ensemble_info=obs_info.ensemble_info,
            **energy_kwargs,
        )

    @property
    def is_ref(self) -> bool:
        """bool: Whether this energy level is expressed relative to a reference mass."""
        return self.ref_particle is not None

    @property
    def sector(self) -> tuple[int, str] | None:
        """tuple of (int, str) or None: Momentum-irrep sector, or ``None`` if either is unset."""
        if self.psq is None or self.irrep is None:
            return None
        return (self.psq, self.irrep)

    @property
    def is_shift_type(self) -> bool:
        """bool: Whether this is a shift energy type (see :data:`SHIFT_ENERGY_TYPES`)."""
        return self.energy_type in SHIFT_ENERGY_TYPES

    @property
    def needs_ni_pair(self) -> bool:
        """
        bool: Whether this energy level requires a non-interacting pair.

        True for the shift energy types and for ``qcmsq``, where only the decay
        channel is relevant.
        """
        return self.is_shift_type or self.energy_type == "qcmsq"

    def to_attrs(self) -> dict[str, Any]:
        """
        Flatten the energy metadata into HDF5-friendly dataset attrs.

        Returns
        -------
        dict
            ``obs_kind`` plus every set energy attribute; ``None`` and empty
            values are omitted. The non-interacting pair particles - which cannot
            be recovered from the canonical name - are serialized as ``ni_pairs``.

        See Also
        --------
        energy_obs_from_attrs : The inverse.
        """
        attrs: dict[str, Any] = {"obs_kind": self.obs_kind}
        for key in ("irrep", "psq", "energy_type", "level_index", "ref_particle"):
            value = getattr(self, key)
            if value is not None:
                attrs[key] = value
        if self.particles:
            attrs["ni_pairs"] = [str(p) for p in self.particles]
        return attrs

    @classmethod
    def from_attrs(cls, base: ObservableInfo, attrs: Mapping[str, Any]) -> "EnergyObsInfo":
        """
        Rebuild this energy type from the attrs :meth:`to_attrs` wrote.

        Parameters
        ----------
        base : ObservableInfo
            Supplies ``name``, ``index``, and ``ensemble_info``.
        attrs : Mapping
            Dataset attrs; any mapping, including an ``h5py`` attrs view.

        Returns
        -------
        EnergyObsInfo
            The reconstructed observable.
        """
        return cls(
            **_shared_energy_attrs(base, attrs),
            level_index=int(attrs["level_index"]) if "level_index" in attrs else None,
            particles=_ni_pairs_from_attrs(attrs),
        )

    @property
    def canonical_name(self) -> str:
        """
        str: Canonical name, ``PSQ{psq}_{irrep}_{energy_type}[_{level_index}][_ref]``.

        Raises
        ------
        ValueError
            If ``irrep``, ``psq``, or ``energy_type`` is unset.
        """
        if not (self.irrep and self.psq is not None and self.energy_type):
            raise ValueError("Cannot generate canonical name: missing irrep, psq, or energy_type")

        parts = [f"PSQ{self.psq}", self.irrep, self.energy_type]
        if self.level_index is not None:
            parts.append(str(self.level_index))
        if self.ref_particle is not None:
            parts.append("ref")
        return "_".join(parts)

    def update_name(self, strict: bool = True) -> bool:
        """
        Rename this observable to match :attr:`canonical_name`.

        Parameters
        ----------
        strict : bool, optional
            When ``True`` (the default), propagate the :exc:`ValueError` raised by
            :attr:`canonical_name` for attributes too incomplete to name. When
            ``False``, log a warning and keep the existing name - used by bulk
            edits, where one unnameable observable must not abort the whole pass.

        Returns
        -------
        bool
            Whether the name was updated.

        Raises
        ------
        ValueError
            If the attributes are too incomplete to name and ``strict`` is ``True``.
        """
        try:
            self.name = self.canonical_name
        except ValueError as exc:
            if strict:
                raise
            logger.warning(f"Keeping name {self.name!r}: {exc}")
            return False
        return True

    def __eq__(self, other):
        """Compare for equality, including the energy-specific attributes."""
        if not isinstance(other, EnergyObsInfo):
            return False
        return (
            super().__eq__(other)
            and self.irrep == other.irrep
            and self.psq == other.psq
            and self.energy_type == other.energy_type
            and self.particles == other.particles
            and self.level_index == other.level_index
            and self.ref_particle == other.ref_particle
        )

    def __hash__(self):
        """int: Hash over the base observable identity and the energy attributes."""
        return hash(
            (
                super().__hash__(),
                self.irrep,
                self.psq,
                self.energy_type,
                self.particles or None,
                self.level_index,
                self.ref_particle,
            )
        )

    def __repr__(self):
        parts = [f"name='{self.name}'", f"index={self.index}"]
        parts += [
            f"{field}={getattr(self, field)!r}"
            for field in self._repr_fields
            if getattr(self, field) not in (None, "", ())
        ]
        return f"{type(self).__name__}({', '.join(parts)})"


@register_obs_kind("energy_single_hadron", "energy_sh")
class SHEnergyObsInfo(EnergyObsInfo):
    """
    Observable info for a single hadron energy level.

    Parameters
    ----------
    name : str, optional
        Observable name. Defaults to :attr:`canonical_name`, which requires
        ``psq`` and ``particle`` to be given.
    index : int, optional
        Observable index.
    ensemble_info : EnsembleInfo, optional
        Ensemble this observable belongs to.
    irrep : str, optional
        Irrep label; must be a key of ``IRREP_LATEX_MAP``.
    psq : int, optional
        Momentum squared; must be non-negative. Also becomes the momentum of
        ``particle``.
    energy_type : str, optional
        Member of :data:`ENERGY_TYPES`, excluding the shift types and ``qcmsq``.
    particle : str, optional
        Name of the hadron, in any spelling accepted by
        :func:`slat.resolve_particle_name`.
    ref_particle : str, optional
        Particle whose mass this level is expressed relative to, or the literal
        ``'ref'`` for a generic reference.

    Raises
    ------
    ValueError
        If ``energy_type`` is a shift type or ``qcmsq``, or if any argument fails
        the validation inherited from :class:`EnergyObsInfo`.
    """

    is_single_hadron = True

    _repr_fields = ("irrep", "psq", "energy_type", "particle", "ref_particle")

    def __init__(
        self,
        name: str | None = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        irrep: str | None = None,
        psq: int | None = None,
        energy_type: str | None = None,
        particle: str | None = None,
        ref_particle: str | None = None,
    ):
        # Single hadron constraints
        if energy_type in (*SHIFT_ENERGY_TYPES, "qcmsq"):
            raise ValueError(f"Single hadron cannot use energy type '{energy_type}'")

        # Convert particle string to Particle object with momentum
        super().__init__(
            name,
            index,
            ensemble_info,
            irrep,
            psq,
            energy_type,
            [Particle(particle, psq=psq)] if particle else [],
            level_index=None,
            ref_particle=ref_particle,
        )

    @property
    def _default_latex_params(self) -> dict:
        """dict: Parent defaults, plus the single hadron rendering flags."""
        return {
            **super()._default_latex_params,
            "include_level_index": False,
            "is_single_hadron": True,
        }

    @property
    def particle(self) -> str | None:
        """str or None: Name of the single hadron. Setting it re-applies :attr:`psq`."""
        return self.particles[0].name if self.particles else None

    @particle.setter
    def particle(self, value: str):
        self.particles = (Particle(value, psq=self.psq),) if value else ()

    @classmethod
    def from_attrs(cls, base: ObservableInfo, attrs: Mapping[str, Any]) -> "SHEnergyObsInfo":
        """
        Rebuild a single hadron energy level from its dataset attrs.

        The hadron is stored in ``ni_pairs`` as a one-element list, so it is read
        back from there rather than from a dedicated ``particle`` attr.
        """
        particles = _ni_pairs_from_attrs(attrs)
        return cls(
            **_shared_energy_attrs(base, attrs),
            particle=particles[0].name if particles else None,
        )

    @property
    def canonical_name(self) -> str:
        """
        str: Canonical name, ``PSQ{psq}_{particle}[_ref]``.

        Raises
        ------
        ValueError
            If ``psq`` or ``particle`` is unset.
        """
        if not (self.psq is not None and self.particle):
            raise ValueError("Cannot generate canonical name: missing psq or particle")
        name = f"PSQ{self.psq}_{self.particle}"
        if self.ref_particle is not None:
            name += "_ref"
        return name


def energy_obs_from_attrs(base: ObservableInfo, attrs: Mapping[str, Any]) -> EnergyObsInfo:
    """
    Rebuild an energy observable from HDF5 dataset attrs.

    Thin wrapper over the registry: looks up ``attrs['obs_kind']`` and defers to
    that class's :meth:`~EnergyObsInfo.from_attrs`.

    Parameters
    ----------
    base : ObservableInfo
        Supplies the identifying ``name``, ``index``, and ``ensemble_info``, so
        the result round-trips and stays groupable with its Re/Im partner.
    attrs : Mapping
        The energy attributes, as written by :meth:`EnergyObsInfo.to_attrs`. Any
        mapping will do, including an ``h5py`` attrs view.

    Returns
    -------
    EnergyObsInfo
        An instance of the class registered for the ``obs_kind`` tag, falling
        back to :class:`EnergyObsInfo` for an unrecognized tag.

    See Also
    --------
    EnergyObsInfo.to_attrs : The inverse.
    """
    cls = obs_kind_class(attrs.get("obs_kind"))
    if cls is None or not issubclass(cls, EnergyObsInfo):
        cls = EnergyObsInfo
    return cls.from_attrs(base, attrs)


def detect_energy_level_type(parsed_attributes: dict) -> str:
    """
    Detect which energy level class a set of parsed attributes describes.

    Parameters
    ----------
    parsed_attributes : dict
        Attributes as returned by :func:`parse_energy_attributes`.

    Returns
    -------
    {'single_hadron', 'multi_hadron', 'unknown'}
        ``'unknown'`` if nothing in ``parsed_attributes`` looks like an energy
        level; ``'single_hadron'`` if exactly one particle was parsed and no level
        index was; ``'multi_hadron'`` otherwise.
    """
    # Check if it looks like an energy level
    if not any(key in parsed_attributes for key in ["energy_type", "irrep", "psq", "particles"]):
        return "unknown"

    # If level index is given, it is multi-hadron
    if parsed_attributes.get("level_index") is not None:
        return "multi_hadron"

    # Single particle suggests single hadron
    particles = parsed_attributes.get("particles", [])
    return "single_hadron" if len(particles) == 1 else "multi_hadron"


def _unrecognized_particle_hint(name: str, delimiters: str = _DEFAULT_DELIMITERS) -> str:
    """
    Explain why no particle was found in an observable name.

    Parameters
    ----------
    name : str
        Observable name that failed to yield a particle.
    delimiters : str, optional
        Regex character class for the delimiters separating tokens.

    Returns
    -------
    str
        A sentence to append to an error message, or ``''`` if the particle
        vocabulary is empty.

    Notes
    -----
    The generic "missing irrep, psq, or energy_type" failure points at the wrong
    field when the real problem is an unrecognized particle token, which after
    case-insensitive resolution can only be a wrong-case single-letter
    abbreviation or a genuine typo.
    """
    if not PARTICLE_LATEX_MAP:
        return ""
    abbreviations = [n for n in PARTICLE_LATEX_MAP if len(n) == 1]
    for token in filter(None, re.split(delimiters, name)):
        for known in abbreviations:
            if token != known and token.lower() == known.lower():
                return (
                    f" Found '{token}' but single-letter particle abbreviations are"
                    f" case-sensitive - did you mean '{known}'?"
                )
    return (
        f" No recognized particle in '{name}'."
        f" Known particles: {', '.join(PARTICLE_LATEX_MAP)}."
    )


def create_energy_obs_info(
    obs_info: ObservableInfo, force_type: str = "auto", **manual_overrides
) -> EnergyObsInfo:
    """
    Build the appropriate energy level observable info by parsing a name.

    Parameters
    ----------
    obs_info : ObservableInfo
        Observable whose ``name`` is parsed, and whose ``index`` and
        ``ensemble_info`` are carried over.
    force_type : {'auto', 'single_hadron', 'multi_hadron'}, optional
        Which class to build. ``'auto'`` (the default) decides via
        :func:`detect_energy_level_type`.
    **manual_overrides
        Attributes that override whatever was parsed from the name.

    Returns
    -------
    EnergyObsInfo
        An :class:`SHEnergyObsInfo` or an :class:`EnergyObsInfo`. A single hadron
        that fails to construct is retried as a multi-hadron level.

    Raises
    ------
    ValueError
        If the type is unrecognized or construction fails. The message repeats the
        detected type and the parsed attributes, and adds a hint when the likely
        cause is an unrecognized particle token.
    """
    # TODO: this will need updated when we need anisotropy observable
    # Parse once and reuse
    parsed = parse_energy_attributes(obs_info.name)
    parsed.update(manual_overrides)

    # Determine energy level type
    energy_type = detect_energy_level_type(parsed) if force_type == "auto" else force_type

    try:
        if energy_type == "single_hadron":
            try:
                return _create_single_hadron_obs(obs_info, parsed)
            except ValueError:
                return _create_multi_hadron_obs(obs_info, parsed)
        if energy_type != "multi_hadron":
            raise ValueError(f"Unrecognized energy level type: {energy_type}")
        return _create_multi_hadron_obs(obs_info, parsed)
    except Exception as e:
        hint = "" if parsed.get("particles") else _unrecognized_particle_hint(obs_info.name)
        raise ValueError(
            f"Error creating energy observable info.{hint} Underlying error: {e}. "
            f"Detected type: {energy_type}, Parsed attributes: {parsed}, "
            f"ObservableInfo: {obs_info}"
        ) from e


def _create_single_hadron_obs(obs_info: ObservableInfo, parsed: dict) -> SHEnergyObsInfo:
    """
    Create a single hadron observable from parsed attributes.

    Parameters
    ----------
    obs_info : ObservableInfo
        Supplies ``index`` and ``ensemble_info``.
    parsed : dict
        Attributes from :func:`parse_energy_attributes`. Left unmodified; only the
        first entry of ``'particles'`` is used.

    Returns
    -------
    SHEnergyObsInfo
        The new observable info.
    """
    parsed = dict(parsed)
    particles = parsed.pop("particles", None)
    return SHEnergyObsInfo(
        index=obs_info.index,
        ensemble_info=obs_info.ensemble_info,
        particle=particles[0] if particles else None,
        **parsed,
    )


def _create_multi_hadron_obs(obs_info: ObservableInfo, parsed: dict) -> EnergyObsInfo:
    """
    Create a multi-hadron observable from parsed attributes.

    Parameters
    ----------
    obs_info : ObservableInfo
        Supplies ``index`` and ``ensemble_info``.
    parsed : dict
        Attributes from :func:`parse_energy_attributes`. Left unmodified; the
        ``'particles'`` names are promoted to :class:`Particle` objects.

    Returns
    -------
    EnergyObsInfo
        The new observable info.
    """
    parsed = dict(parsed)
    # Convert particle strings to Particle objects
    parsed["particles"] = [Particle(p) for p in parsed.get("particles", [])]
    return EnergyObsInfo(index=obs_info.index, ensemble_info=obs_info.ensemble_info, **parsed)
