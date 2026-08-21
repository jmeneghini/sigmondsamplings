"""Parameter description layer for optimagic-based fitting.

This module follows the project Spec/Resolved config pattern (see
``docs/design/config-design.md``): each concept is modeled twice.

* **Spec** (:class:`ParamSpec`, :class:`ConstraintSpec`, :class:`ParamSetSpec`) is
  the permissive authoring/TOML surface. Parameter names are keys in the
  ``values`` mapping; individual records accept shorthand (``bounds``/
  ``soft_bounds`` pairs, an omitted ``initial``) and round-trip compactly.
* **Resolved** (:class:`ParamResolved`, :class:`ConstraintResolved`,
  :class:`ParamSetResolved`) is the canonical form: ``initial`` is required, bound
  channels are explicit, and constraints carry a uniform ``param_groups`` selector
  shape. ``ParamSetSpec.resolve()`` is the single boundary that collapses authoring
  ambiguity.

Everything is modeled the way optimagic likes it: a **dict PyTree** keyed by
parameter name. The optimagic adapters (:meth:`ParamSetResolved.bounds`,
:meth:`ParamSetResolved.constraints_for_optimagic`) build ``optimagic.Bounds`` and
``optimagic.*Constraint`` objects whose selectors reference parameters by name, so a
fit driver can pass them straight into ``optimagic.minimize`` alongside
:meth:`ParamSetResolved.start_values`. These runtime objects are constructed only
from the Resolved layer.

optimagic is an optional dependency, imported lazily by the adapter methods.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from slat import StrictModel

from ..info import ObservableInfo

if TYPE_CHECKING:
    import optimagic as om

__all__ = [
    "ConstraintKind",
    "ConstraintSpec",
    "ConstraintResolved",
    "ParamSpec",
    "ParamResolved",
    "ParamSetSpec",
    "ParamSetResolved",
]

#: Constraint kinds mapped to optimagic constraint classes. ``nonlinear`` is
#: intentionally excluded — it needs a Python callable and so cannot live in TOML.
ConstraintKind = Literal[
    "fixed",
    "increasing",
    "decreasing",
    "equality",
    "linear",
    "probability",
    "pairwise_equality",
]

#: ``kind`` strings whose constraint takes multiple selectors (``selectors=...``)
#: rather than a single ``selector=...``.
_MULTI_SELECTOR_KINDS = frozenset({"pairwise_equality"})

#: Extra (non-selector) constraint fields forwarded to a constraint class when that
#: class declares them. Drawn from ``optimagic.LinearConstraint``.
_CONSTRAINT_EXTRA_FIELDS = ("weights", "lower_bound", "upper_bound", "value")

#: Per-parameter bound channels, in the order ``optimagic.Bounds`` declares them.
_BOUND_CHANNELS = ("lower", "upper", "soft_lower", "soft_upper")


ParamName = Annotated[
    str,
    Field(
        min_length=1,
        description="Non-empty fit-parameter identifier (the dict-PyTree key).",
        examples=["A", "m", "energy"],
    ),
]


def _import_optimagic() -> om:
    try:
        import optimagic as om
    except ImportError as exc:  # pragma: no cover - exercised only without optimagic
        raise ImportError(
            "the fitting parameter layer requires the optional 'optimagic' package"
        ) from exc
    return om


def _constraint_classes(om: om) -> dict[str, type]:
    """Map :data:`ConstraintKind` strings to optimagic constraint classes."""
    return {
        "fixed": om.FixedConstraint,
        "increasing": om.IncreasingConstraint,
        "decreasing": om.DecreasingConstraint,
        "equality": om.EqualityConstraint,
        "linear": om.LinearConstraint,
        "probability": om.ProbabilityConstraint,
        "pairwise_equality": om.PairwiseEqualityConstraint,
    }


def _name_selector(names: Sequence[str]) -> Callable[[dict[str, float]], np.ndarray]:
    """Build a selector picking ``names`` (in order) from a dict-PyTree params."""
    ordered = list(names)
    return lambda params: np.array([params[name] for name in ordered])


def _check_bound_order(
    name: str,
    lower: float | None,
    upper: float | None,
    soft_lower: float | None,
    soft_upper: float | None,
) -> None:
    """Reject inverted box bounds. Shared by the Spec and Resolved param layers."""
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(
            f"lower bound exceeds upper bound for {name!r}: {lower} > {upper}"
        )
    if soft_lower is not None and soft_upper is not None and soft_lower > soft_upper:
        raise ValueError(
            f"soft_lower exceeds soft_upper for {name!r}: {soft_lower} > {soft_upper}"
        )


def _check_param_set_consistency(
    names: Sequence[str], constraints: Iterable[ConstraintSpec | ConstraintResolved]
) -> None:
    """Enforce a non-empty parameter mapping and known constraint references."""
    if not names:
        raise ValueError("a parameter set must contain at least one parameter")
    known = set(names)
    for constraint in constraints:
        unknown = sorted(set(constraint.referenced_names()) - known)
        if unknown:
            raise ValueError(
                f"constraint {constraint.kind!r} references unknown parameter(s): {unknown}"
            )


# ---------------------------------------------------------------------------
# Resolved (canonical) layer — runtime/optimagic objects are built from here.
# ---------------------------------------------------------------------------


class ParamResolved(StrictModel):
    """One fully resolved fit parameter.

    The containing :class:`ParamSetResolved.values` mapping owns the parameter
    name. ``initial`` is always present and each bound channel is explicit
    (``None`` meaning unbounded). The bound fields mirror ``optimagic.Bounds``;
    ``fixed=True`` pins the keyed parameter at its start value.
    """

    initial: float = Field(description="Resolved initial value (start guess).")
    lower: float | None = None
    upper: float | None = None
    soft_lower: float | None = None
    soft_upper: float | None = None
    fixed: bool = False
    latex: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _check_bound_order(
            "parameter", self.lower, self.upper, self.soft_lower, self.soft_upper
        )
        return self

    def observable_info(self, name: str) -> ObservableInfo:
        """Build the output observable metadata for this fit parameter."""
        return ObservableInfo(name=name, latex_str=self.latex)

    def to_spec(self) -> ParamSpec:
        """Inverse of :meth:`ParamSpec.resolve` for editing/round-tripping."""
        return ParamSpec(**self.model_dump())


class ConstraintResolved(StrictModel):
    """One fully resolved cross-parameter constraint.

    Canonical counterpart of :class:`ConstraintSpec`: every constraint stores its
    selected parameters as ``param_groups`` (a list of name groups). Single-selector
    kinds (``increasing``, ``decreasing``, ``equality``, ``linear``, ``probability``,
    ``fixed``) carry exactly one group; ``pairwise_equality`` carries one group per
    selector. The ``weights``/``lower_bound``/``upper_bound``/``value`` fields are
    forwarded only to constraint classes that declare them (i.e. linear).
    """

    kind: ConstraintKind
    param_groups: list[list[ParamName]] = Field(
        min_length=1,
        description="One name group per selector; single-selector kinds carry one group.",
    )
    weights: float | list[float] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _check_groups(self) -> Self:
        if any(len(group) < 1 for group in self.param_groups):
            raise ValueError("each entry of 'param_groups' must name at least one parameter")
        if self.kind not in _MULTI_SELECTOR_KINDS and len(self.param_groups) != 1:
            raise ValueError(
                f"constraint kind {self.kind!r} takes a single selector group"
            )
        return self

    def referenced_names(self) -> list[str]:
        """All parameter names this constraint selects, for validation."""
        return [name for group in self.param_groups for name in group]

    def build(self):
        """Construct the optimagic constraint object for this resolved spec."""
        om = _import_optimagic()
        cls = _constraint_classes(om)[self.kind]
        if self.kind in _MULTI_SELECTOR_KINDS:
            selectors = [_name_selector(group) for group in self.param_groups]
            return cls(selectors=selectors)

        (group,) = self.param_groups
        field_names = {f.name for f in dataclasses.fields(cls)}
        extra = {
            key: getattr(self, key)
            for key in _CONSTRAINT_EXTRA_FIELDS
            if key in field_names and getattr(self, key) is not None
        }
        return cls(selector=_name_selector(group), **extra)

    def to_spec(self) -> ConstraintSpec:
        """Inverse of :meth:`ConstraintSpec.resolve` for editing/round-tripping."""
        extra = {key: getattr(self, key) for key in _CONSTRAINT_EXTRA_FIELDS}
        if self.kind in _MULTI_SELECTOR_KINDS:
            return ConstraintSpec(kind=self.kind, param_groups=self.param_groups, **extra)
        (group,) = self.param_groups
        return ConstraintSpec(kind=self.kind, params=group, **extra)


class ParamSetResolved(StrictModel):
    """Canonical, total parameter set: the runtime/optimagic-facing layer.

    ``values`` is the canonical ordered mapping from parameter name to its resolved
    controls. The model provides the three pieces an optimagic fit needs:
    :meth:`start_values` (the params PyTree), :meth:`bounds` (an ``optimagic.Bounds``),
    and :meth:`constraints_for_optimagic` (per-parameter fixes plus the declared
    :class:`ConstraintResolved` constraints).
    """

    values: dict[ParamName, ParamResolved]
    constraints: list[ConstraintResolved] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        _check_param_set_consistency(self.names, self.constraints)
        return self

    @property
    def names(self) -> list[str]:
        """Parameter names in declared order."""
        return list(self.values)

    def start_values(self) -> dict[str, float]:
        """Initial-value PyTree ``{name: initial}`` (``initial`` is always set)."""
        return {name: float(spec.initial) for name, spec in self.values.items()}

    def bounds(self) -> om.Bounds | None:
        """Build an ``optimagic.Bounds`` from the per-parameter bound fields.

        Each bound channel (``lower``/``upper``/``soft_lower``/``soft_upper``) becomes
        a dict holding only the parameters that set it. Returns ``None`` when no
        parameter declares any bound.
        """
        om = _import_optimagic()
        kwargs: dict[str, dict[str, float]] = {}
        for channel in _BOUND_CHANNELS:
            mapping = {
                name: float(getattr(spec, channel))
                for name, spec in self.values.items()
                if getattr(spec, channel) is not None
            }
            if mapping:
                kwargs[channel] = mapping
        if not kwargs:
            return None
        return om.Bounds(**kwargs)

    def constraints_for_optimagic(self) -> list:
        """Optimagic constraint objects: per-parameter fixes plus declared constraints."""
        om = _import_optimagic()
        constraints = [
            om.FixedConstraint(selector=_name_selector([name]))
            for name, spec in self.values.items()
            if spec.fixed
        ]
        constraints.extend(constraint.build() for constraint in self.constraints)
        return constraints

    def to_spec(self) -> ParamSetSpec:
        """Inverse of :meth:`ParamSetSpec.resolve` for editing/round-tripping."""
        return ParamSetSpec(
            values={name: spec.to_spec() for name, spec in self.values.items()},
            constraints=[constraint.to_spec() for constraint in self.constraints],
        )


# ---------------------------------------------------------------------------
# Spec (authoring) layer — permissive TOML surface.
# ---------------------------------------------------------------------------


def _expand_bound_pair(
    data: dict[str, Any], pair_key: str, lower_key: str, upper_key: str
) -> None:
    """Expand a ``[lower, upper]`` shorthand into its two explicit bound fields."""
    if pair_key not in data:
        return
    pair = data.pop(pair_key)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"{pair_key!r} must be a [{lower_key}, {upper_key}] pair")
    for key, value in ((lower_key, pair[0]), (upper_key, pair[1])):
        if data.get(key) is not None:
            raise ValueError(f"specify either {pair_key!r} or {key!r}, not both")
        data[key] = value


class ParamSpec(StrictModel):
    """Authoring controls for one keyed fit parameter.

    Accepts bound shorthand: ``bounds = [lower, upper]`` and
    ``soft_bounds = [soft_lower, soft_upper]`` expand into the explicit per-channel
    fields (you may not give both a pair and its expanded field). ``initial`` may be
    omitted while authoring; :meth:`resolve` requires it.
    """

    initial: float | None = None
    lower: float | None = None
    upper: float | None = None
    soft_lower: float | None = None
    soft_upper: float | None = None
    fixed: bool = False
    latex: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        _expand_bound_pair(normalized, "bounds", "lower", "upper")
        _expand_bound_pair(normalized, "soft_bounds", "soft_lower", "soft_upper")
        return normalized

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _check_bound_order(
            "parameter", self.lower, self.upper, self.soft_lower, self.soft_upper
        )
        return self

    def resolve(self, name: str) -> ParamResolved:
        """Expand into a canonical :class:`ParamResolved` (requires ``initial``)."""
        if self.initial is None:
            raise ValueError(f"initial value is missing for parameter {name!r}")
        return ParamResolved(**self.model_dump())


class ConstraintSpec(StrictModel):
    """Authoring spec for one cross-parameter constraint.

    ``params`` lists the (ordered) parameter names the constraint acts on for the
    single-selector kinds (``increasing``, ``decreasing``, ``equality``, ``linear``,
    ``probability``, ``fixed``). ``pairwise_equality`` instead uses ``param_groups``,
    one selector per group. The ``weights``/``lower_bound``/``upper_bound``/``value``
    fields are forwarded only to constraint classes that declare them (i.e. linear).
    """

    kind: ConstraintKind
    params: list[ParamName] | None = None
    param_groups: list[list[ParamName]] | None = None
    weights: float | list[float] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _check_selectors(self) -> Self:
        if self.kind in _MULTI_SELECTOR_KINDS:
            if not self.param_groups:
                raise ValueError(f"constraint kind {self.kind!r} requires 'param_groups'")
            if any(len(group) < 1 for group in self.param_groups):
                raise ValueError("each entry of 'param_groups' must name at least one parameter")
        else:
            if not self.params:
                raise ValueError(f"constraint kind {self.kind!r} requires 'params'")
        return self

    def referenced_names(self) -> list[str]:
        """All parameter names this constraint selects, for validation."""
        if self.param_groups:
            return [name for group in self.param_groups for name in group]
        return list(self.params or [])

    def resolve(self) -> ConstraintResolved:
        """Canonicalize the selector to ``param_groups`` form."""
        groups = (
            [list(group) for group in self.param_groups or []]
            if self.kind in _MULTI_SELECTOR_KINDS
            else [list(self.params or [])]
        )
        return ConstraintResolved(
            kind=self.kind,
            param_groups=groups,
            weights=self.weights,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            value=self.value,
        )


class ParamSetSpec(StrictModel):
    """An ordered ``name -> ParamSpec`` mapping plus cross-parameter constraints.

    :meth:`resolve` collapses authoring shorthand into a :class:`ParamSetResolved`,
    the layer the optimagic fit driver consumes.
    """

    __toml_tag__: ClassVar[str | None] = "params"

    values: dict[ParamName, ParamSpec]
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        _check_param_set_consistency(self.names, self.constraints)
        return self

    @property
    def names(self) -> list[str]:
        """Parameter names in declared order."""
        return list(self.values)

    def resolve(self) -> ParamSetResolved:
        """Expand every parameter and constraint into the canonical set."""
        return ParamSetResolved(
            values={name: spec.resolve(name) for name, spec in self.values.items()},
            constraints=[constraint.resolve() for constraint in self.constraints],
        )

    def add_constraint(
        self,
        kind: ConstraintKind,
        params: Sequence[str] | None = None,
        *,
        param_groups: Sequence[Sequence[str]] | None = None,
        **extra: Any,
    ) -> ConstraintSpec:
        """Append a validated :class:`ConstraintSpec` and return it.

        The programmatic escape hatch for constraints that are awkward to hand-write
        in TOML. Re-runs validation so unknown parameter names are rejected immediately.
        """
        spec = ConstraintSpec(
            kind=kind,
            params=list(params) if params is not None else None,
            param_groups=[list(group) for group in param_groups]
            if param_groups is not None
            else None,
            **extra,
        )
        self.constraints = [*self.constraints, spec]
        return spec
