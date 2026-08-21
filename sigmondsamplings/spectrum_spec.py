"""Spec/Resolved config for energy-level spectrum selections.

Follows the project Spec/Resolved config pattern (see ``docs/design/config-design.md``):
a spectrum selection is modeled twice.

* :class:`SpectrumSpec` / :class:`SectorSpec` are the permissive TOML authoring
  surface. A sector names its levels with exactly one of ``levels`` (an explicit
  list of indices, in which unsorted/duplicate entries are tolerated) or
  ``n_levels`` (a count selecting the first ``N`` levels, ``[0, ..., N-1]``).
* :class:`SpectrumResolved` / :class:`SectorResolved` are the canonical form every
  ``levels`` an explicit, sorted, duplicate-free list and every ``(psq, irrep)``
  sector unique consumed by collection filtering via
  :meth:`SpectrumResolved.allowed_keys`.

``SpectrumSpec.resolve()`` is the single boundary that collapses the authoring
shorthand. The TOML form is a flat array of tables under ``[[spectrum]]``::

    [[spectrum]]
    psq = 0
    irrep = "A1g"
    levels = [0, 1, 2]

    [[spectrum]]
    psq = 1
    irrep = "E"
    levels = [0]
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import Field, model_validator

from slat import StrictModel

__all__ = [
    "SectorSpec",
    "SectorResolved",
    "SpectrumSpec",
    "SpectrumResolved",
]


Psq = Annotated[
    int,
    Field(
        ge=0,
        description="Total momentum squared, ``d^2``, of the sector.",
        examples=[0, 1, 2],
    ),
]

IrrepName = Annotated[
    str,
    Field(
        min_length=1,
        description="Little-group irreducible-representation label.",
        examples=["A1g", "E", "T1u"],
    ),
]

LevelIndex = Annotated[
    int,
    Field(
        ge=0,
        description="Zero-based energy-level index within a sector.",
        examples=[0, 1, 2],
    ),
]


def _check_unique_sectors(sectors: list[SectorSpec | SectorResolved]) -> None:
    """Reject a spectrum that names the same ``(psq, irrep)`` sector twice."""
    keys = [(s.psq, s.irrep) for s in sectors]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate (psq, irrep) sectors in spectrum: {duplicates}")


# ---------------------------------------------------------------------------
# Resolved (canonical) layer
# ---------------------------------------------------------------------------


class SectorResolved(StrictModel):
    """One canonical ``(psq, irrep)`` sector with an explicit level-index list."""

    psq: Psq
    irrep: IrrepName
    levels: list[LevelIndex] = Field(
        min_length=1,
        description="Sorted, duplicate-free zero-based level indices kept in this sector.",
        examples=[[0], [0, 1, 2]],
    )

    @model_validator(mode="after")
    def _check_levels(self) -> Self:
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(
                f"levels must be unique in sector (psq={self.psq}, irrep={self.irrep!r})"
            )
        if list(self.levels) != sorted(self.levels):
            raise ValueError(
                f"levels must be sorted in sector (psq={self.psq}, irrep={self.irrep!r})"
            )
        return self

    def keys(self) -> list[tuple[int, str, int]]:
        """The ``(psq, irrep, level)`` triples this sector selects."""
        return [(self.psq, self.irrep, level) for level in self.levels]

    def to_spec(self) -> SectorSpec:
        """Inverse of :meth:`SectorSpec.resolve` for editing/round-tripping."""
        return SectorSpec(psq=self.psq, irrep=self.irrep, levels=list(self.levels))


class SpectrumResolved(StrictModel):
    """Canonical spectrum selection: an ordered set of resolved sectors."""

    spectrum: list[SectorResolved] = Field(
        default_factory=list,
        description="Resolved sectors, each with an explicit level-index list.",
    )

    @model_validator(mode="after")
    def _check_unique_sectors(self) -> Self:
        _check_unique_sectors(self.spectrum)
        return self

    def allowed_keys(self) -> set[tuple[int, str, int]]:
        """All ``(psq, irrep, level)`` triples this spectrum selects."""
        return {key for sector in self.spectrum for key in sector.keys()}

    def to_spec(self) -> SpectrumSpec:
        """Inverse of :meth:`SpectrumSpec.resolve` for editing/round-tripping."""
        return SpectrumSpec(spectrum=[sector.to_spec() for sector in self.spectrum])


# ---------------------------------------------------------------------------
# Spec (authoring) layer
# ---------------------------------------------------------------------------


class SectorSpec(StrictModel):
    """Authoring spec for one ``(psq, irrep)`` sector.

    Provide exactly one of ``levels`` (an explicit list of indices) or ``n_levels``
    (a count selecting the first ``N`` levels, ``[0, ..., N-1]``). :meth:`resolve`
    normalizes either form to a sorted, duplicate-free list.
    """

    psq: Psq
    irrep: IrrepName
    levels: list[LevelIndex] | None = Field(
        default=None,
        min_length=1,
        description="Explicit zero-based level indices to keep.",
        examples=[[0], [0, 1, 2]],
    )
    n_levels: int | None = Field(
        default=None,
        gt=0,
        description="Number of leading levels to keep, selecting ``[0, ..., n_levels - 1]``.",
        examples=[1, 3],
    )

    @model_validator(mode="after")
    def _check_levels_source(self) -> Self:
        if (self.levels is None) == (self.n_levels is None):
            raise ValueError(
                f"sector (psq={self.psq}, irrep={self.irrep!r}) requires exactly one "
                "of 'levels' or 'n_levels'"
            )
        return self

    def resolve(self) -> SectorResolved:
        """Expand the authoring form into a canonical sorted, unique level list."""
        if self.n_levels is not None:
            levels = list(range(self.n_levels))
        else:
            levels = sorted(set(self.levels or []))
        return SectorResolved(psq=self.psq, irrep=self.irrep, levels=levels)


class SpectrumSpec(StrictModel):
    """Authorable spectrum selection: an ordered set of :class:`SectorSpec`.

    Round-trips to/from the flat ``[[spectrum]]`` array-of-tables TOML form.
    :meth:`resolve` collapses authoring shorthand into a :class:`SpectrumResolved`.
    """

    __toml_tag__: ClassVar[str | None] = None

    spectrum: list[SectorSpec] = Field(
        default_factory=list,
        description="Sectors to keep, each naming its (psq, irrep) and level selection.",
        examples=[
            [
                {"psq": 0, "irrep": "A1g", "levels": [0, 1, 2]},
                {"psq": 1, "irrep": "E", "levels": [0]},
            ]
        ],
    )

    def resolve(self) -> SpectrumResolved:
        """Expand every sector into the canonical spectrum."""
        return SpectrumResolved(spectrum=[sector.resolve() for sector in self.spectrum])
