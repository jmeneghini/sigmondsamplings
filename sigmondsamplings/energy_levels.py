"""
Energy level specializations for ObservableInfo and SigmondSampling.
"""

import re
from typing import Optional, List, Dict, Any
from .sampling import ObservableInfo, EnsembleInfo, DEFAULT_ENSEMBLE

# Import KBfit constants (required)
try:
    from kbfit.constants import (
        PARTICLE_LATEX_MAP,
        IRREP_LATEX_MAP,
        get_particle_latex_str,
        get_irrep_latex_str,
    )
except ImportError:
    # warning only; KBfit not strictly required for this module
    PARTICLE_LATEX_MAP = {}
    IRREP_LATEX_MAP = {} 
    print("Warning: KBfit not installed; particle and irrep LaTeX mappings will be unavailable.")


def get_energy_type_latex_str(energy_type: str, level_index: int = None) -> str:
    """Get LaTeX representation of energy type with optional level index."""
    latex_map = {
        "elab": r"E_{\text{lab}}",
        "ecm": r"E_{\text{cm}}",
        "delab": r"\Delta E_{\text{lab}}",
        "decm": r"\Delta E_{\text{cm}}",
    }
    base_latex = latex_map.get(energy_type.lower(), energy_type)

    if level_index is not None:
        base_latex = base_latex.replace("}", f", {level_index}}}")

    return base_latex


def parse_energy_attributes(
    name: str, delimiters: str = r"[_\.\s/]"
) -> Dict[str, Any]:
    """
    Parse energy level attributes from observable name with flexible delimiters.

    Args:
        name: Observable name to parse
        delimiters: Regex character class for delimiters (default: underscore, dash, dot, space, slash)

    Returns:
        Dictionary with parsed attributes
    """
    result = {}

    # Create delimiter boundary patterns
    start_bound = f"(?:^|{delimiters})"
    end_bound = f"(?:{delimiters}|$)"

    # Energy type (case-insensitive)
    energy_pattern = f"{start_bound}(elab|ecm|delab|decm){end_bound}"
    energy_match = re.search(energy_pattern, name, re.IGNORECASE)
    if energy_match:
        result["energy_type"] = energy_match.group(1).lower()

    # PSQ (case-insensitive) - allow various formats: PSQ4, psq4, P4, p4
    psq_pattern = f"{start_bound}(?:psq|p)(\\d+){end_bound}"
    psq_match = re.search(psq_pattern, name, re.IGNORECASE)
    if psq_match:
        result["psq"] = int(psq_match.group(1))
        
    # Check for PSQ=int pattern
    else:
        psq_pattern_eq = f"{start_bound}PSQ=(\\d+){end_bound}"
        psq_match_eq = re.search(psq_pattern_eq, name, re.IGNORECASE)
        if psq_match_eq:
            result["psq"] = int(psq_match_eq.group(1))

    # Level index - standalone number (not part of PSQ or other patterns)
    # Find all standalone numbers and pick the one that's not PSQ
    level_pattern = f"{start_bound}(\\d+){end_bound}"
    for match in re.finditer(level_pattern, name):
        number = int(match.group(1))
        # Skip if this number is already captured as PSQ
        if psq_match and match.span() == psq_match.span():
            continue
        
        # Take the first standalone number as level index
        result["level_index"] = number
        break

    # Reference mode
    ref_pattern = f"{start_bound}ref{end_bound}"
    if re.search(ref_pattern, name, re.IGNORECASE):
        result["ref_particle"] = "ref"

    # Irrep (case-insensitive but preserve exact case from KBfit)
    irrep_names = list(IRREP_LATEX_MAP.keys())
    irrep_pattern = (
        f"{start_bound}("
        + "|".join(re.escape(i) for i in irrep_names)
        + f"){end_bound}"
    )
    irrep_match = re.search(irrep_pattern, name, re.IGNORECASE)
    if irrep_match:
        # Find the exact match from KBfit constants
        matched_irrep = irrep_match.group(1)
        for irrep_key in irrep_names:
            if irrep_key.lower() == matched_irrep.lower():
                result["irrep"] = irrep_key
                break
        else:
            result["irrep"] = matched_irrep

    # Particles (case-sensitive for exact matches)
    particle_names = list(PARTICLE_LATEX_MAP.keys())
    particle_pattern = (
        f"{start_bound}("
        + "|".join(re.escape(p) for p in particle_names)
        + f"){end_bound}"
    )
    particles = []
    for match in re.finditer(particle_pattern, name):
        particle = match.group(1)
        if particle not in particles:
            particles.append(particle)
    if particles:
        result["particles"] = particles
        
    # recheck for particle(psq) patter
    particle_psq_pattern = (
        f"{start_bound}("
        + "|".join(re.escape(p) for p in particle_names)
        + r")(?:\((\d+)\))?"
        + f"{end_bound}"
    )
    for match in re.finditer(particle_psq_pattern, name):
        particle = match.group(1)
        psq_str = match.group(2)
        if particle not in particles:
            particles.append(particle)
        if psq_str is not None:
            result["psq"] = int(psq_str)
    if particles:
        result["particles"] = particles

    return result


def _generate_latex_str(
    energy_type: str = None,
    irrep: str = None,
    psq: int = None,
    particles: List[str] = None,
    level_index: int = None,
    ref_particle: str = None,
    include_irrep: bool = True,
    include_psq: bool = True,
    include_particles: bool = True,
    include_level_index: bool = True,
    is_single_hadron: bool = False,
) -> str:
    """Generate LaTeX string for energy level observable."""

    # Special case: Single hadron PSQ=0 uses m_particle
    if (
        is_single_hadron
        and psq == 0
        and particles
        and len(particles) == 1
        and energy_type == "elab"
        and ref_particle is None
    ):
        base_expr = f"m_{{{get_particle_latex_str(particles[0])}}}"
        further_info = []
        if include_irrep and irrep:
            further_info.append(get_irrep_latex_str(irrep))
        return (
            f"{base_expr} \\text{{ {' '.join(further_info)} }}"
            if further_info
            else base_expr
        )

    # Build energy expression
    if ref_particle is not None:
        # Reference mode: E_type/m_ref
        energy_latex = (
            get_energy_type_latex_str(
                energy_type, level_index if include_level_index else None
            )
            if energy_type
            else "E"
        )
        ref_latex = (
            r"m_{\text{ref}}"
            if ref_particle == "ref"
            else f"m_{{{get_particle_latex_str(ref_particle)}}}"
        )
        base_expr = f"{energy_latex}/{ref_latex}"
    else:
        # Standard mode: a_t E_type
        energy_latex = (
            get_energy_type_latex_str(
                energy_type, level_index if include_level_index else None
            )
            if energy_type
            else "E"
        )
        base_expr = f"a_t {energy_latex}"

    # Build further info
    further_info = []
    if include_irrep and irrep:
        further_info.append(get_irrep_latex_str(irrep))
    if include_psq and psq is not None:
        further_info.append(f"PSQ={psq}")
    if include_particles and particles:
        particle_strs = [get_particle_latex_str(p) for p in particles]
        further_info.append(f"({','.join(particle_strs)})")

    return (
        f"{base_expr} \\text{{ {' '.join(further_info)} }}"
        if further_info
        else base_expr
    )


class EnergyObsInfo(ObservableInfo):
    """Observable info for energy levels with irrep, PSQ, and energy type."""

    def __init__(
        self,
        name: str = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        latex_str: str = None,
        irrep: str = None,
        psq: int = None,
        energy_type: str = None,
        particles: List[str] = None,
        level_index: int = None,
        ref_particle: str = None,
    ):

        # Validate inputs
        if energy_type and energy_type not in ["elab", "ecm", "delab", "decm"]:
            raise ValueError(f"Invalid energy_type: {energy_type}")
        if psq is not None and psq < 0:
            raise ValueError(f"PSQ must be non-negative: {psq}")
        if irrep and irrep not in IRREP_LATEX_MAP:
            raise ValueError(f"Invalid irrep: {irrep}")
        if particles:
            for particle in particles:
                if particle not in PARTICLE_LATEX_MAP:
                    raise ValueError(f"Invalid particle: {particle}")
        if (
            ref_particle
            and ref_particle != "ref"
            and ref_particle not in PARTICLE_LATEX_MAP
        ):
            raise ValueError(f"Invalid ref_particle: {ref_particle}")

        # Store attributes for canonical name generation
        self.irrep = irrep
        self.psq = psq
        self.energy_type = energy_type
        self.particles = particles or []
        self.level_index = level_index
        self.ref_particle = ref_particle

        # Auto-generate name if needed using canonical form
        if name is None:
            if not (irrep and psq is not None and energy_type):
                raise ValueError(
                    "Must provide either name or (irrep, psq, energy_type)"
                )
            name = self.canonical_name

        # Auto-generate LaTeX if needed
        if latex_str is None:
            latex_str = _generate_latex_str(
                energy_type=energy_type,
                irrep=irrep,
                psq=psq,
                particles=particles,
                level_index=level_index,
                ref_particle=ref_particle,
            )

        super().__init__(name, index, "n", "re", ensemble_info, latex_str)

    @classmethod
    def from_observable_info(
        cls, obs_info: ObservableInfo, **energy_kwargs
    ) -> "EnergyObsInfo":
        """Create EnergyObsInfo from existing ObservableInfo."""
        return cls(
            name=obs_info.name,
            index=obs_info.index,
            ensemble_info=obs_info.ensemble_info,
            latex_str=obs_info.latex_str,
            **energy_kwargs,
        )

    def update_latex_str(
        self,
        include_irrep: bool = True,
        include_psq: bool = True,
        include_particles: bool = True,
        include_level_index: bool = True,
    ):
        """Update LaTeX representation with customizable components."""
        self.latex_str = _generate_latex_str(
            energy_type=self.energy_type,
            irrep=self.irrep,
            psq=self.psq,
            particles=self.particles,
            level_index=self.level_index,
            ref_particle=self.ref_particle,
            include_irrep=include_irrep,
            include_psq=include_psq,
            include_particles=include_particles,
            include_level_index=include_level_index,
        )

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
        return hash((
            super().__hash__(),
            self.irrep,
            self.psq,
            self.energy_type,
            tuple(self.particles) if self.particles else None,
            self.level_index,
            self.ref_particle
        ))

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

    @property
    def canonical_name(self) -> str:
        """Generate canonical form: PSQ{psq}_{irrep}_{energy_type}_{level_idx} + _ref (if true)."""
        if not (self.irrep and self.psq is not None and self.energy_type):
            raise ValueError(
                "Cannot generate canonical name: missing irrep, psq, or energy_type"
            )

        parts = [f"PSQ{self.psq}", self.irrep, self.energy_type]
        if self.level_index is not None:
            parts.append(str(self.level_index))
        if self.ref_particle is not None:
            parts.append("ref")
        return "_".join(parts)


class SHEnergyObsInfo(EnergyObsInfo):
    """Single hadron energy level observable info."""

    def __init__(
        self,
        name: str = None,
        index: int = 0,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
        latex_str: str = None,
        irrep: str = None,
        psq: int = None,
        energy_type: str = None,
        particle: str = None,
        ref_particle: str = None,
    ):

        # Single hadron constraints
        if energy_type in ["delab", "decm"]:
            raise ValueError(f"Single hadron cannot use energy type '{energy_type}'")

        # Store attributes first for canonical name generation
        particles = [particle] if particle else []

        # Auto-generate name if needed using canonical form
        if name is None:
            if not (psq is not None and particle):
                raise ValueError(
                    "Must provide either name or (psq, particle) for single hadron canonical name"
                )
            # Temporarily set attributes to use canonical_name property
            self.psq = psq
            self.particles = particles
            self.ref_particle = ref_particle
            name = self.canonical_name

        # Auto-generate LaTeX if needed
        if latex_str is None:
            particles = [particle] if particle else []
            latex_str = _generate_latex_str(
                energy_type=energy_type,
                irrep=irrep,
                psq=psq,
                particles=particles,
                ref_particle=ref_particle,
                include_level_index=False,
                is_single_hadron=True,
            )

        particles = [particle] if particle else []
        super().__init__(
            name,
            index,
            ensemble_info,
            latex_str,
            irrep,
            psq,
            energy_type,
            particles,
            level_index=None,
            ref_particle=ref_particle,
        )

    @property
    def particle(self) -> Optional[str]:
        """Get the single particle name."""
        return self.particles[0] if self.particles else None

    @particle.setter
    def particle(self, value: str):
        """Set the single particle name."""
        self.particles = [value] if value else []

    def update_latex_str(
        self,
        include_irrep: bool = True,
        include_psq: bool = True,
        include_particles: bool = True,
    ):
        """Update LaTeX representation for single hadron."""
        self.latex_str = _generate_latex_str(
            energy_type=self.energy_type,
            irrep=self.irrep,
            psq=self.psq,
            particles=self.particles,
            ref_particle=self.ref_particle,
            include_irrep=include_irrep,
            include_psq=include_psq,
            include_particles=include_particles,
            include_level_index=False,
            is_single_hadron=True,
        )

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

    @property
    def canonical_name(self) -> str:
        """Generate canonical form: PSQ{psq}_{particle_name}."""
        if not (self.psq is not None and self.particle):
            raise ValueError("Cannot generate canonical name: missing psq or particle")
        name = f"PSQ{self.psq}_{self.particle}"
        if self.ref_particle is not None:
            name += "_ref"
        return name


def detect_energy_level_type(parsed_attributes: dict) -> str:
    """Detect energy level type from parsed attributes."""
    # Check if it looks like an energy level
    if not any(key in parsed_attributes for key in ["energy_type", "irrep", "psq"]):
        return "unknown"

    # If level index is given, it is multi-hadron
    if parsed_attributes.get("level_index") is not None:
        return "multi_hadron"

    # Single particle suggests single hadron
    particles = parsed_attributes.get("particles", [])
    return "single_hadron" if len(particles) == 1 else "multi_hadron"


def create_energy_obs_info(
    obs_info: ObservableInfo, force_type: str = "auto", **manual_overrides
):
    """Factory function to create appropriate energy level ObservableInfo."""
    # Parse once and reuse
    parsed = parse_energy_attributes(obs_info.name)
    parsed.update(manual_overrides)

    # Determine energy level type
    if force_type == "auto":
        energy_type = detect_energy_level_type(parsed)
    else:
        energy_type = force_type

    # Create appropriate energy level observable
    if energy_type == "single_hadron":
        try:
            return _create_single_hadron_obs(obs_info, parsed)
        except ValueError:
            return _create_multi_hadron_obs(obs_info, parsed)
    elif energy_type == "multi_hadron":
        return _create_multi_hadron_obs(obs_info, parsed)
    else:
        return obs_info


def _create_single_hadron_obs(
    obs_info: ObservableInfo, parsed: dict
) -> SHEnergyObsInfo:
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
    return EnergyObsInfo(
        index=obs_info.index, ensemble_info=obs_info.ensemble_info, **parsed
    )
