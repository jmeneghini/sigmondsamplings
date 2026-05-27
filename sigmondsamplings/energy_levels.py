"""Energy level specializations for ObservableInfo and SigmondSampling."""

import re
from typing import Any

from slatmeta import (
    IRREP_LATEX_MAP,
    PARTICLE_LATEX_MAP,
    get_energy_type_latex_str,
    get_irrep_latex_str,
    get_particle_latex_str,
)

from .info import DEFAULT_ENSEMBLE, EnsembleInfo, ObservableInfo


class Particle:
    """
    Represents a particle with optional momentum specification.

    Parameters:
    -----------
    name : str
        Particle name (e.g., 'pi', 'rho', 'K')
    psq : int, optional
        Momentum integer squared (d^2) for this particle.
        Required for non-interacting pair specifications in shift energy types.

    Examples:
    ---------
    >>> # Particle at rest
    >>> pi_at_rest = Particle('pi', psq=0)
    >>>
    >>> # Particle with momentum d^2 = 1
    >>> rho_moving = Particle('rho', psq=1)
    >>>
    >>> # Particle without momentum specification (for non-shift types)
    >>> generic_pi = Particle('pi')
    """

    def __init__(self, name: str, psq: int | None = None):
        """Initialize particle with name and optional momentum"""
        if name not in PARTICLE_LATEX_MAP and PARTICLE_LATEX_MAP:
            raise ValueError(f"Invalid particle name: {name}")
        if psq is not None and psq < 0:
            raise ValueError(f"psq must be non-negative, got {psq}")

        self.name = name
        self.psq = psq

    @classmethod
    def from_string(cls, particle_str: str) -> "Particle":
        """
        Create Particle from string representation.

        Examples of valid strings:
        - "pi" (no momentum)
        - "rho(1)" (momentum d^2=1)
        """
        pattern = r"^([a-zA-Z0-9]+)(?:\((\d+)\))?$"
        match = re.match(pattern, particle_str)
        if not match:
            raise ValueError(f"Invalid particle string: {particle_str}")

        name = match.group(1)
        psq_str = match.group(2)
        psq = int(psq_str) if psq_str is not None else None

        return cls(name, psq)

    @property
    def has_momentum(self) -> bool:
        """Check if momentum is specified"""
        return self.psq is not None

    def latex_str(self) -> str:
        """Get LaTeX representation of particle"""
        if not PARTICLE_LATEX_MAP:
            return self.name
        base_latex = get_particle_latex_str(self.name)
        if self.has_momentum:
            return f"{base_latex}(d^2={self.psq})"
        return base_latex

    def __str__(self) -> str:
        if self.has_momentum:
            return f"{self.name}(d^2={self.psq})"
        return self.name

    def __repr__(self) -> str:
        if self.has_momentum:
            return f"Particle('{self.name}', psq={self.psq})"
        return f"Particle('{self.name}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            # Allow comparison with string for backward compatibility
            return self.name == other
        if not isinstance(other, Particle):
            return False
        return self.name == other.name and self.psq == other.psq

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        """
        Less than comparison based on momentum.

        Particles without momentum (psq=None) are considered less than those with momentum.
        If both have momentum, compares psq values.
        """
        if not isinstance(other, Particle):
            return NotImplemented

        # Handle None cases
        if self.psq is None and other.psq is None:
            return False  # Equal in terms of momentum
        if self.psq is None:
            return True  # None is less than any number
        if other.psq is None:
            return False  # Any number is greater than None

        return self.psq < other.psq

    def __le__(self, other) -> bool:
        """Less than or equal comparison based on momentum."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other) -> bool:
        """
        Greater than comparison based on momentum.

        Particles with momentum are considered greater than those without.
        If both have momentum, compares psq values.
        """
        if not isinstance(other, Particle):
            return NotImplemented

        # Handle None cases
        if self.psq is None and other.psq is None:
            return False  # Equal in terms of momentum
        if self.psq is None:
            return False  # None is less than any number
        if other.psq is None:
            return True  # Any number is greater than None

        return self.psq > other.psq

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison based on momentum."""
        if not isinstance(other, Particle):
            return NotImplemented
        return self.__gt__(other) or self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.name, self.psq))


class _BoundaryPatterns:
    """Helper class for creating regex patterns with delimiter boundaries."""

    def __init__(self, delimiters: str = r"[_\.\s/]"):
        self.start = f"(?:^|{delimiters})"
        self.end = f"(?:{delimiters}|$)"

    def wrap(self, pattern: str) -> str:
        """Wrap pattern with delimiter boundaries."""
        return f"{self.start}{pattern}{self.end}"


def _parse_energy_type(name: str, bounds: _BoundaryPatterns) -> str | None:
    """Extract energy type (elab, ecm, delab, decm, qcmsq) from name."""
    pattern = bounds.wrap("(elab|ecm|delab|decm|qcmsq)")
    match = re.search(pattern, name, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _parse_psq(name: str, bounds: _BoundaryPatterns) -> int | None:
    """Extract PSQ value from name, supporting multiple formats."""
    # Try PSQ4, psq4, P4, p4 formats
    pattern = bounds.wrap(r"(?:psq|p)(\d+)")
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try PSQ=4 format
    pattern_eq = bounds.wrap(r"PSQ=(\d+)")
    match = re.search(pattern_eq, name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_level_index(
    name: str, bounds: _BoundaryPatterns, psq_match_span: tuple | None = None
) -> int | None:
    """Extract level index (standalone number not part of PSQ)."""
    pattern = bounds.wrap(r"(\d+)")
    for match in re.finditer(pattern, name):
        # Skip if this is the PSQ number
        if psq_match_span and match.span() == psq_match_span:
            continue
        # Return first standalone number
        return int(match.group(1))
    return None


def _parse_reference_mode(name: str, bounds: _BoundaryPatterns) -> str | None:
    """Check if name indicates reference mode."""
    pattern = bounds.wrap("ref")
    return "ref" if re.search(pattern, name, re.IGNORECASE) else None


def _parse_irrep(name: str, bounds: _BoundaryPatterns) -> str | None:
    """Extract irrep, preserving exact case from shared label constants."""
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


def _parse_particles(name: str, bounds: _BoundaryPatterns) -> tuple[list[str], int | None]:
    """
    Extract particle names and optional PSQ from particle(psq) notation.

    Returns:
        Tuple of (particle_list, psq_from_particle) where psq_from_particle
        is the PSQ value if found in particle(N) notation, else None.
    """
    if not PARTICLE_LATEX_MAP:
        return [], None

    particle_names = list(PARTICLE_LATEX_MAP.keys())
    particles = []
    psq_from_particle = None

    # Match particle(psq) pattern to extract both particle and optional PSQ
    pattern = bounds.wrap("(" + "|".join(re.escape(p) for p in particle_names) + r")(?:\((\d+)\))?")

    for match in re.finditer(pattern, name):
        particle = match.group(1)
        psq_str = match.group(2)

        if particle not in particles:
            particles.append(particle)

        if psq_str is not None:
            psq_from_particle = int(psq_str)

    return particles, psq_from_particle


def parse_energy_attributes(name: str, delimiters: str = r"[_\.\s/|]") -> dict[str, Any]:
    """
    Parse energy level attributes from observable name with flexible delimiters.

    Args:
        name: Observable name to parse
        delimiters: Regex character class for delimiters (default: underscore, dash, dot, space, slash)

    Returns:
        Dictionary with parsed attributes
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


def _is_single_hadron_mass(
    is_single_hadron: bool,
    psq: int,
) -> bool:
    """Check if this is the special case: single hadron PSQ=0 mass."""
    return is_single_hadron and psq == 0


def _generate_latex_str(
    energy_type: str = None,
    irrep: str = None,
    psq: int = None,
    particles: list[Particle] | None = None,
    level_index: int = None,
    ref_particle: str = None,
    include_irrep: bool = True,
    include_psq: bool = True,
    include_particles: bool = True,
    include_level_index: bool = True,
    is_single_hadron: bool = False,
) -> str:
    """Generate LaTeX string for energy level observable."""
    particle_names = [p.name for p in particles] if particles else []

    # Determine base expression
    if _is_single_hadron_mass(is_single_hadron, psq):
        # Special case: Single hadron PSQ=0 uses m_particle
        base_expr = f"M_{{{get_particle_latex_str(particle_names[0])}}}"
        # For mass, don't include PSQ or particles in further_info
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


class EnergyObsInfo(ObservableInfo):
    """
    Observable info for energy levels with irrep, PSQ, and energy type.

    Uses Particle objects with momentum specifications for shift energy types (delab, decm).
    """

    def __init__(
        self,
        name: str = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        irrep: str = None,
        psq: int = None,
        energy_type: str = None,
        particles: list[Particle] | None = None,
        level_index: int = None,
        ref_particle: str = None,
    ):
        try:
            # Validate inputs
            if energy_type and energy_type not in ["elab", "ecm", "delab", "decm", "qcmsq"]:
                raise ValueError(f"Invalid energy_type: {energy_type}")
            if psq is not None and psq < 0:
                raise ValueError(f"PSQ must be non-negative: {psq}")
            if irrep and irrep not in IRREP_LATEX_MAP:
                raise ValueError(f"Invalid irrep: {irrep}")

            # Validate particle list
            particle_list = particles or []
            if particles:
                for i, p in enumerate(particles):
                    if not isinstance(p, Particle):
                        raise TypeError(f"particles[{i}] must be Particle object, got {type(p)}")

            if (
                ref_particle
                and ref_particle != "ref"
                and ref_particle not in PARTICLE_LATEX_MAP
                and PARTICLE_LATEX_MAP
            ):
                raise ValueError(f"Invalid ref_particle: {ref_particle}")

            # Store attributes for canonical name generation
            self.irrep = irrep
            self.psq = psq
            self.energy_type = energy_type
            self.particles = tuple(particle_list)
            self.level_index = level_index
            self.ref_particle = ref_particle

            # Auto-generate name if needed using canonical form
            if name is None:
                name = self.canonical_name

            super().__init__(name, index, "n", "re", ensemble_info, latex_str=None)
        except Exception as e:
            raise ValueError(
                f"Error initializing EnergyObsInfo: {e}. Inputs were: "
                f"name={name}, index={index}, ensemble_info={ensemble_info}, "
                f"irrep={irrep}, psq={psq}, energy_type={energy_type}, "
                f"particles={particles}, level_index={level_index}, ref_particle={ref_particle}"
            ) from e

    @property
    def _default_latex_params(self) -> dict:
        """Default parameters for LaTeX generation."""
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
        """Generate LaTeX representation dynamically."""
        return _generate_latex_str(**self._default_latex_params)

    def specify_latex_str(self, **kwargs) -> str:
        """Generate LaTeX string with options to include/exclude certain attributes.
        For kwargs, one can choose from
             irrep: str = None,
             psq: int = None,
             particles: list[Particle] | None = None,
             level_index: int = None,
             ref_particle: str = None,
             include_irrep: bool = True,
             include_psq: bool = True,
             include_particles: bool = True,
             include_level_index: bool = True,
             is_single_hadron: bool = False
        """
        # override defaults with any provided kwargs
        kwargs_in = self._default_latex_params.copy()
        kwargs_in.update(kwargs)
        return _generate_latex_str(**kwargs_in)

    @classmethod
    def from_observable_info(cls, obs_info: ObservableInfo, **energy_kwargs) -> "EnergyObsInfo":
        """Create EnergyObsInfo from existing ObservableInfo."""
        return cls(
            name=obs_info.name,
            index=obs_info.index,
            ensemble_info=obs_info.ensemble_info,
            **energy_kwargs,
        )

    @property
    def is_ref(self) -> bool:
        """Check if this energy level is in reference mode."""
        return self.ref_particle is not None

    @property
    def is_shift_type(self) -> bool:
        """Check if this energy level is a shift energy type."""
        return self.energy_type in ["delab", "decm"] if self.energy_type else False

    @property
    def needs_ni_pair(self) -> bool:
        """Determine if this energy level requires non-interacting pair specification. For qcmsq, only the decay channel is relevant."""
        return self.is_shift_type or self.energy_type == "qcmsq"

    @property
    def canonical_name(self) -> str:
        """Generate canonical form: PSQ{psq}_{irrep}_{energy_type}_{level_idx} + _ref (if true)."""
        if not (self.irrep and self.psq is not None and self.energy_type):
            raise ValueError("Cannot generate canonical name: missing irrep, psq, or energy_type")

        parts = [f"PSQ{self.psq}", self.irrep, self.energy_type]
        if self.level_index is not None:
            parts.append(str(self.level_index))
        if self.ref_particle is not None:
            parts.append("ref")
        return "_".join(parts)

    def update_name(self):
        """Update name to match canonical form based on current attributes."""
        self.name = self.canonical_name

    def __eq__(self, other):
        """Check equality including energy-specific attributes."""
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
        """Make EnergyObsInfo hashable including energy-specific attributes."""
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
        if self.irrep:
            parts.append(f"irrep='{self.irrep}'")
        if self.psq is not None:
            parts.append(f"psq={self.psq}")
        if self.energy_type:
            parts.append(f"energy_type='{self.energy_type}'")
        if self.level_index is not None:
            parts.append(f"level_index={self.level_index}")
        if self.ref_particle is not None:
            parts.append(f"ref_particle='{self.ref_particle}'")
        if self.particles:
            parts.append(f"particles={self.particles}")
        return f"EnergyObsInfo({', '.join(parts)})"


class SHEnergyObsInfo(EnergyObsInfo):
    """Single hadron energy level observable info."""

    def __init__(
        self,
        name: str = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        irrep: str = None,
        psq: int = None,
        energy_type: str = None,
        particle: str = None,
        ref_particle: str = None,
    ):
        # Single hadron constraints
        if energy_type in ["delab", "decm", "qcmsq"]:
            raise ValueError(f"Single hadron cannot use energy type '{energy_type}'")

        # Convert particle string to Particle object with momentum
        particles = [Particle(particle, psq=psq)] if particle else []

        # Let parent handle name generation via canonical_name property
        super().__init__(
            name,
            index,
            ensemble_info,
            irrep,
            psq,
            energy_type,
            particles,
            level_index=None,
            ref_particle=ref_particle,
        )

    @property
    def _default_latex_params(self) -> dict:
        """Default parameters for LaTeX generation specific to single hadron."""
        return {
            "energy_type": self.energy_type,
            "irrep": self.irrep,
            "psq": self.psq,
            "particles": self.particles,
            "ref_particle": self.ref_particle,
            "include_level_index": False,
            "is_single_hadron": True,
        }

    @property
    def particle(self) -> str | None:
        """Get the single particle name."""
        return self.particles[0].name if self.particles else None

    @particle.setter
    def particle(self, value: str):
        """Set the single particle name."""
        self.particles = (Particle(value, psq=self.psq),) if value else ()

    @property
    def canonical_name(self) -> str:
        """Generate canonical form: PSQ{psq}_{particle_name}."""
        if not (self.psq is not None and self.particle):
            raise ValueError("Cannot generate canonical name: missing psq or particle")
        name = f"PSQ{self.psq}_{self.particle}"
        if self.ref_particle is not None:
            name += "_ref"
        return name

    def __repr__(self):
        parts = [f"name='{self.name}'", f"index={self.index}"]
        if self.irrep:
            parts.append(f"irrep='{self.irrep}'")
        if self.psq is not None:
            parts.append(f"psq={self.psq}")
        if self.energy_type:
            parts.append(f"energy_type='{self.energy_type}'")
        if self.particle:
            parts.append(f"particle='{self.particle}'")
        if self.ref_particle:
            parts.append(f"ref_particle='{self.ref_particle}'")
        return f"SHEnergyObsInfo({', '.join(parts)})"


def detect_energy_level_type(parsed_attributes: dict) -> str:
    """Detect energy level type from parsed attributes."""
    # Check if it looks like an energy level
    if not any(key in parsed_attributes for key in ["energy_type", "irrep", "psq", "particles"]):
        return "unknown"

    # If level index is given, it is multi-hadron
    if parsed_attributes.get("level_index") is not None:
        return "multi_hadron"

    # Single particle suggests single hadron
    particles = parsed_attributes.get("particles", [])
    return "single_hadron" if len(particles) == 1 else "multi_hadron"


def create_energy_obs_info(obs_info: ObservableInfo, force_type: str = "auto", **manual_overrides):
    """Factory function to create appropriate energy level ObservableInfo."""
    # Parse once and reuse
    parsed = parse_energy_attributes(obs_info.name)
    parsed.update(manual_overrides)

    # Determine energy level type
    if force_type == "auto":
        energy_type = detect_energy_level_type(parsed)
    else:
        energy_type = force_type

    try:
        # Create appropriate energy level observable
        if energy_type == "single_hadron":
            try:
                return _create_single_hadron_obs(obs_info, parsed)
            except ValueError:
                return _create_multi_hadron_obs(obs_info, parsed)
        elif energy_type == "multi_hadron":
            return _create_multi_hadron_obs(obs_info, parsed)
        else:
            raise ValueError(f"Unrecognized energy level type: {energy_type}")
        # TODO: this will need updated when we need anisotropy observable
    except Exception as e:
        raise ValueError(
            f"Error creating energy observable info: {e}. "
            f"Detected type: {energy_type}, Parsed attributes: {parsed}, "
            f"ObservableInfo: {obs_info}"
        ) from e


def _create_single_hadron_obs(obs_info: ObservableInfo, parsed: dict) -> SHEnergyObsInfo:
    """Create single hadron observable from parsed attributes."""
    # Extract single particle
    particle = None
    if "particles" in parsed:
        particle = parsed["particles"][0] if parsed["particles"] else None
        parsed = {k: v for k, v in parsed.items() if k != "particles"}

    return SHEnergyObsInfo(
        index=obs_info.index,
        ensemble_info=obs_info.ensemble_info,
        particle=particle,
        **parsed,
    )


def _create_multi_hadron_obs(obs_info: ObservableInfo, parsed: dict) -> EnergyObsInfo:
    """Create multi-hadron observable from parsed attributes."""
    # Convert particle strings to Particle objects
    if "particles" in parsed:
        parsed = parsed.copy()
        parsed["particles"] = [Particle(p) for p in parsed["particles"]]
    return EnergyObsInfo(index=obs_info.index, ensemble_info=obs_info.ensemble_info, **parsed)
