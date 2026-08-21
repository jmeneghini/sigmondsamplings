"""Spec/Resolved config for observable-set edits.

Follows the project Spec/Resolved config pattern (see :mod:`sigmondsamplings.spectrum_spec`):
an edit is modeled twice.

* The ``*Op`` classes plus :class:`EditSpec` are the permissive TOML authoring surface.
  Paths may be relative, particle names may use any accepted spelling, and a scope is
  written inline.
* :class:`EditResolved` is the canonical form the executor consumes: every path made
  absolute and checked, every particle name canonicalized, every op in the order it
  will run.

:meth:`EditSpec.resolve` is the single boundary between the two. The TOML form is a flat
array of tables under ``[[edit]]``::

    [[edit]]
    op = "tag-energy"
    ni_yml = "ni.yml"

    [[edit]]
    op = "set"
    where = { psq = 0 }
    attrs = { irrep = "A1g" }
    rename = true

    [[edit]]
    op = "drop"
    where = { level_index = [4, 5] }

    [[edit]]
    op = "add-ref"
    particle = "N"

Ops run in the order written. The ``ss edit`` flag form builds the same object in a
fixed canonical order (interpret, annotate, derive, membership), so flag order on the
command line never changes the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from slat import USE_TOML_TAG, StrictModel, dump_toml, resolve_particle_name

__all__ = [
    "AddRefOp",
    "DropOp",
    "EditOp",
    "EditResolved",
    "EditSpec",
    "KeepOp",
    "ScopedOp",
    "SetOp",
    "SetRefOp",
    "TagEnergyOp",
]


ParticleName = Annotated[
    str,
    Field(
        min_length=1,
        description="Particle name, in any spelling accepted by slat.resolve_particle_name.",
        examples=["N", "pi", "K"],
    ),
]


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class ScopedOp(StrictModel):
    """Base for ops that apply to a subset of the collection.

    Carries the four clauses of the ``ss query`` filter language, spelled exactly as
    the matching CLI flags so a filter vetted with ``ss query`` transfers verbatim.
    They combine with AND and are resolved by
    :func:`sigmondsamplings.selection.filter_collection`. Leaving all four unset scopes
    the op to the whole collection.
    """

    where: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Attribute filters as {attr: value}. A list value is a membership test."
        ),
        examples=[{"psq": 0}, {"irrep": ["A1g", "T1u"], "energy_type": "elab"}],
    )
    contains: str | None = Field(
        default=None,
        min_length=1,
        description="Keep observables whose name contains this substring.",
    )
    regex: str | None = Field(
        default=None,
        min_length=1,
        description="Keep observables whose name matches this regular expression.",
    )
    spec: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Path to a spectrum TOML, as written by `ss query energy --save`; keeps "
            "only the (psq, irrep, level) levels it names."
        ),
        examples=["spectrum.toml"],
    )

    @property
    def is_scoped(self) -> bool:
        """Whether this op constrains its scope at all."""
        return bool(self.where or self.contains or self.regex or self.spec)

    def scope(self, collection):
        """Return the subset of *collection* this op applies to.

        The result wraps the same observable objects as *collection*, so mutating it
        in place is visible in *collection* too - which is what lets a scoped edit
        avoid a split-and-rejoin.
        """
        from .selection import filter_collection

        return filter_collection(
            collection,
            where=self.where,
            contains=self.contains,
            regex=self.regex,
            spec=self.spec,
        )

    def _resolved_scope_fields(self, *, base_dir: Path) -> dict[str, Any]:
        """Scope fields with ``spec`` made absolute and checked, for ``model_copy``."""
        if self.spec is None:
            return {}
        path = Path(self.spec)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Spectrum spec TOML does not exist: {path}")
        return {"spec": str(path)}


def _canonical_particle(name: str) -> str:
    """Canonicalize a particle name, rejecting one slat does not recognize."""
    resolved = resolve_particle_name(name)
    if resolved is None:
        raise ValueError(f"Unknown particle name {name!r}")
    return resolved


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


class TagEnergyOp(StrictModel):
    """Interpret observables as energy levels, optionally applying NI pair assignments.

    Observables that cannot be read as energy levels are left as they are and pass
    through to the output untouched. Every energy-aware op needs this to have run.
    Unscoped by design: interpretation applies to the whole collection.
    """

    op: Literal["tag-energy"]
    ni_yml: str | None = Field(
        default=None,
        min_length=1,
        description="PyCalQ YAML with non-interacting pair assignments.",
    )
    skip_missing_particles: bool = Field(
        default=True,
        description="Skip observables naming particles slat does not recognize.",
    )

    def resolve(self, *, base_dir: Path) -> TagEnergyOp:
        if self.ni_yml is None:
            return self.model_copy()
        path = Path(self.ni_yml)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"NI pair YAML does not exist: {path}")
        return self.model_copy(update={"ni_yml": str(path)})


class SetOp(ScopedOp):
    """Set observable attributes on the scope, in place."""

    op: Literal["set"]
    attrs: dict[str, Any] = Field(
        min_length=1,
        description="Attribute values to set on each observable in the scope.",
        examples=[{"irrep": "A1g"}, {"psq": 2, "energy_type": "ecm"}],
    )
    rename: bool = Field(
        default=False,
        description=(
            "Resync each name to its canonical form afterwards. Observables whose "
            "attributes are too incomplete to name keep their existing name."
        ),
    )

    def resolve(self, *, base_dir: Path) -> SetOp:
        return self.model_copy(update=self._resolved_scope_fields(base_dir=base_dir))


class SetRefOp(ScopedOp):
    """Tag levels already in reference mode with the reference particle's name.

    Only affects levels with ``is_ref`` true. To *create* reference levels, use
    :class:`AddRefOp`.
    """

    op: Literal["set-ref"]
    particle: ParticleName

    def resolve(self, *, base_dir: Path) -> SetRefOp:
        return self.model_copy(
            update={
                "particle": _canonical_particle(self.particle),
                **self._resolved_scope_fields(base_dir=base_dir),
            }
        )


class AddRefOp(ScopedOp):
    """Derive reference levels, dividing each level in scope by a single-hadron mass.

    The single-hadron observable supplying the denominator is looked up in the *whole*
    collection, not just the scope, so a scope can name the levels to divide without
    having to include the reference particle itself. Idempotent: levels whose reference
    observable already exists are skipped.
    """

    op: Literal["add-ref"]
    particle: ParticleName
    psq: int = Field(
        default=0,
        ge=0,
        description="Momentum frame of the single-hadron observable used as denominator.",
    )

    def resolve(self, *, base_dir: Path) -> AddRefOp:
        return self.model_copy(
            update={
                "particle": _canonical_particle(self.particle),
                **self._resolved_scope_fields(base_dir=base_dir),
            }
        )


class KeepOp(ScopedOp):
    """Write only the scope; everything outside it is dropped."""

    op: Literal["keep"]

    @model_validator(mode="after")
    def _check_scope(self) -> Self:
        if not self.is_scoped:
            raise ValueError("'keep' needs a scope; it would otherwise be a no-op")
        return self

    def resolve(self, *, base_dir: Path) -> KeepOp:
        return self.model_copy(update=self._resolved_scope_fields(base_dir=base_dir))


class DropOp(ScopedOp):
    """Write everything except the scope."""

    op: Literal["drop"]

    @model_validator(mode="after")
    def _check_scope(self) -> Self:
        if not self.is_scoped:
            raise ValueError("'drop' needs a scope; it would otherwise empty the file")
        return self

    def resolve(self, *, base_dir: Path) -> DropOp:
        return self.model_copy(update=self._resolved_scope_fields(base_dir=base_dir))


EditOp = Annotated[
    TagEnergyOp | SetOp | SetRefOp | AddRefOp | KeepOp | DropOp,
    Field(discriminator="op"),
]

# Ops that need observables to have been interpreted as energy levels first.
ENERGY_OPS = (SetRefOp, AddRefOp)


# ---------------------------------------------------------------------------
# Resolved (canonical) layer
# ---------------------------------------------------------------------------


class EditResolved(StrictModel):
    """Canonical edit program: ops in execution order, every path checked."""

    edit: list[EditOp] = Field(
        default_factory=list,
        description="Resolved ops, applied in order.",
    )


# ---------------------------------------------------------------------------
# Spec (authoring) layer
# ---------------------------------------------------------------------------


class EditSpec(StrictModel):
    """Authorable edit program, round-tripping the ``[[edit]]`` array-of-tables form."""

    __toml_tag__: ClassVar[str | None] = None

    edit: list[EditOp] = Field(
        default_factory=list,
        description="Ops to apply, in order.",
        examples=[
            [
                {"op": "tag-energy", "ni_yml": "ni.yml"},
                {"op": "add-ref", "particle": "N"},
            ]
        ],
    )

    @model_validator(mode="after")
    def _check_energy_ops_are_reachable(self) -> Self:
        """Reject an energy-aware op with no preceding tag-energy to interpret for it."""
        tagged = False
        for op in self.edit:
            if isinstance(op, TagEnergyOp):
                tagged = True
            elif isinstance(op, ENERGY_OPS) and not tagged:
                raise ValueError(
                    f"'{op.op}' needs energy levels; add a 'tag-energy' op before it"
                )
        return self

    def resolve(self, *, base_dir: Path) -> EditResolved:
        """Resolve every op against *base_dir*, producing the canonical program."""
        return EditResolved(edit=[op.resolve(base_dir=base_dir) for op in self.edit])

    def to_toml(
        self,
        dest: Any = None,
        *,
        table: Any = USE_TOML_TAG,
        comment_unset: bool = False,
    ) -> str:
        """Serialize the ``[[edit]]`` array of tables.

        Overrides the base dump to omit fields left at their default, so a recipe
        written by ``ss edit --save-recipe`` shows only what the edit actually asked
        for rather than every unscoped ``where`` and untouched flag. Values read back
        identically, since an absent key resolves to that same default.
        """
        tag = type(self).__toml_tag__ if table is USE_TOML_TAG else table
        return dump_toml(
            self.model_dump(mode="json", exclude_defaults=True),
            dest,
            table=tag,
            comment_unset=comment_unset,
        )
