"""Read and write PyCalQ non-interacting-level assignments."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from ..energy_levels import Particle

ShiftParticleKey = tuple[str, int, int]
ShiftParticleMap = dict[ShiftParticleKey, list[Particle]]

_SECTOR_PATTERN = re.compile(r"^(?P<irrep>.+?)\s+PSQ=(?P<psq>\d+)$")


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping for {context}, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string keys for {context}")
    return value


def _find_named_mapping(node: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Find a uniquely named nested mapping in a YAML document."""
    matches: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        if name in value:
            matches.append(_as_mapping(value[name], name))
        for child in value.values():
            visit(child)

    visit(node)
    if len(matches) > 1:
        raise ValueError(f"Found multiple {name!r} mappings in PyCalQ YAML")
    return matches[0] if matches else None


def _looks_like_sector_mapping(node: Mapping[str, Any]) -> bool:
    return any(_SECTOR_PATTERN.fullmatch(key) for key in node)


def _find_sector_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    named = _find_named_mapping(config, "non_interacting_levels")
    if named is not None:
        return named

    current = config
    while not _looks_like_sector_mapping(current):
        if len(current) != 1:
            raise ValueError(
                "Could not locate a unique PyCalQ non_interacting_levels mapping"
            )
        current = _as_mapping(next(iter(current.values())), "PyCalQ YAML wrapper")
    return current


def read_shift_particles(
    yml_path: str | Path,
    *,
    allowed_sectors: Collection[tuple[int, str]] | None = None,
) -> ShiftParticleMap:
    """Read ``(irrep, psq, level_index) -> particles`` assignments from PyCalQ YAML."""
    with Path(yml_path).open() as handle:
        config = _as_mapping(yaml.safe_load(handle), "PyCalQ YAML root")

    sector_mapping = _find_sector_mapping(config)
    assignments: ShiftParticleMap = {}

    for sector_key, raw_levels in sector_mapping.items():
        match = _SECTOR_PATTERN.fullmatch(sector_key)
        if match is None:
            continue

        irrep = match.group("irrep")
        psq = int(match.group("psq"))
        if allowed_sectors is not None and (psq, irrep) not in allowed_sectors:
            continue
        if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)):
            raise ValueError(f"Expected a level list for sector {sector_key!r}")

        for level_index, raw_particles in enumerate(raw_levels):
            if not isinstance(raw_particles, Sequence) or isinstance(
                raw_particles, (str, bytes)
            ):
                raise ValueError(
                    f"Expected a particle list for sector {sector_key!r}, level {level_index}"
                )
            if not all(isinstance(particle, str) for particle in raw_particles):
                raise ValueError(
                    f"Expected particle strings for sector {sector_key!r}, level {level_index}"
                )
            if not raw_particles:
                # PyCalQ uses empty entries to pad sparse level-index lists.
                continue
            particle_names = cast(Sequence[str], raw_particles)
            assignments[(irrep, psq, level_index)] = [
                Particle.from_string(particle) for particle in particle_names
            ]

    return assignments


def write_shift_particles(
    yml_path: str | Path,
    assignments: Mapping[ShiftParticleKey, Sequence[Particle]],
) -> None:
    """Write assignments using PyCalQ's ``non_interacting_levels`` structure."""
    sectors: dict[str, dict[int, list[str]]] = {}
    for (irrep, psq, level_index), particles in assignments.items():
        sector_key = f"{irrep} PSQ={psq}"
        sectors.setdefault(sector_key, {})[level_index] = [str(particle) for particle in particles]

    non_interacting_levels = {
        sector_key: [levels.get(index, []) for index in range(max(levels) + 1)]
        for sector_key, levels in sorted(sectors.items())
    }
    output = {"fit_spectrum": {"non_interacting_levels": non_interacting_levels}}

    with Path(yml_path).open("w") as handle:
        yaml.safe_dump(output, handle, default_flow_style=False, sort_keys=False)
