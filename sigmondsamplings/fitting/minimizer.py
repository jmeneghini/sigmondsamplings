"""optimagic minimizer selection as a strict, TOML-round-trippable config.

The only backend is optimagic, so :class:`MinimizerConfig` is essentially a thin
serializer over optimagic's own algorithm settings: ``algorithm`` names an entry of
``optimagic.algorithms.AVAILABLE_ALGORITHMS`` and ``options`` carries the overrides
forwarded to that algorithm's settings dataclass. Validation, the list of selectable
algorithms, default settings, and capability flags are all read straight from
optimagic rather than re-declared here.

optimagic is imported lazily so a config can still be loaded/serialized where the
optional package is absent; it is required to validate options eagerly and to
:meth:`MinimizerConfig.build` the algorithm.
"""

from __future__ import annotations
import json

from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from slat import StrictModel

if TYPE_CHECKING:
    from optimagic.optimization.algorithm import Algorithm

__all__ = [
    "MinimizerConfig",
    "algorithm_capabilities",
    "algorithm_settings",
    "available_algorithms",
]

#: Default algorithm — gradient-based, bound-aware, always available with optimagic.
DEFAULT_ALGORITHM = "scipy_lbfgsb"

#: ``minimize()``-level option groups carried as nested dicts, each validated
#: against the matching optimagic options dataclass exactly like ``options`` is
#: validated against the algorithm settings dataclass.
_OPTION_CLASS_NAMES: dict[str, str] = {
    "scaling": "ScalingOptions",
    "multistart": "MultistartOptions",
    "numdiff": "NumdiffOptions",
}


def _available_algorithms() -> dict[str, Any]:
    try:
        from optimagic.algorithms import AVAILABLE_ALGORITHMS
    except ImportError as exc:  # pragma: no cover - exercised only without optimagic
        raise ImportError(
            "the fitting minimizer layer requires the optional 'optimagic' package"
        ) from exc
    return AVAILABLE_ALGORITHMS


def _algorithm_class(algorithm: str) -> Any:
    """Return the optimagic ``AlgorithmMeta`` for ``algorithm`` or raise."""
    registry = _available_algorithms()
    try:
        return registry[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"unknown optimagic algorithm {algorithm!r}; "
            f"available: {sorted(registry)}"
        ) from exc


def _clean_enums(mapping: dict[str, Any]) -> dict[str, Any]:
    """Replace enum values with their ``.value`` so the dict is JSON/TOML-native."""
    return {key: getattr(value, "value", value) for key, value in mapping.items()}


def _jsonify_settings(mapping: dict[str, Any]) -> dict[str, Any]:
    """Make a resolved optimagic-settings dict TOML-round-trippable.

    Cleans enums, normalizes tuples to lists (TOML has no tuple), and drops
    ``None`` values: a ``None`` here means "use optimagic's own default", so an
    absent key reloads to the same default rather than to the lossy ``false``
    sentinel a nested ``None`` would otherwise become.
    """
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in _clean_enums(mapping).items()
        if value is not None
    }


def _option_class(group: str) -> Any:
    """Return the optimagic options dataclass for an option ``group`` or raise."""
    try:
        import optimagic as om
    except ImportError as exc:  # pragma: no cover - exercised only without optimagic
        raise ImportError(
            "the fitting minimizer layer requires the optional 'optimagic' package"
        ) from exc
    return getattr(om, _OPTION_CLASS_NAMES[group])


def _coerce_option_mapping(cls: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    """Coerce list values to tuples for tuple-typed fields of ``cls``.

    optimagic option dataclasses declare some settings as tuples (e.g.
    ``MultistartOptions.mixing_weight_bounds``); TOML/JSON render tuples as lists,
    so a hand-written or round-tripped mapping arrives with lists. Convert those
    back so construction and TOML round-trips both succeed.
    """
    tuple_fields = {f.name for f in fields(cls) if str(f.type).lower().startswith("tuple")}
    if not tuple_fields:
        return mapping
    return {
        key: tuple(value) if key in tuple_fields and isinstance(value, list) else value
        for key, value in mapping.items()
    }


def _build_option(group: str, mapping: dict[str, Any]) -> Any:
    """Construct the optimagic options dataclass for ``group`` from a mapping."""
    cls = _option_class(group)
    return cls(**_coerce_option_mapping(cls, mapping))


def _validate_option_mapping(group: str, mapping: dict[str, Any]) -> None:
    """Eagerly validate one option ``group`` by constructing its optimagic dataclass."""
    cls = _option_class(group)
    try:
        _build_option(group, mapping)
    except (TypeError, ValueError) as exc:
        valid = sorted(f.name for f in fields(cls))
        raise ValueError(
            f"invalid {group} options: {exc}. Valid {group} option keys: {valid}"
        ) from exc


def available_algorithms() -> list[str]:
    """Sorted names of optimagic algorithms whose dependencies are installed."""
    return sorted(_available_algorithms())


def algorithm_settings(algorithm: str) -> dict[str, Any]:
    """Default settings (option name -> default) for an optimagic ``algorithm``."""
    return _clean_enums(asdict(_algorithm_class(algorithm)()))


def algorithm_capabilities(algorithm: str) -> dict[str, Any]:
    """Capability flags (``algo_info``) for an optimagic ``algorithm``."""
    return _clean_enums(asdict(_algorithm_class(algorithm)().algo_info))


class MinimizerConfig(StrictModel):
    """Selects an optimagic algorithm and its option overrides for a fit.

    A thin, TOML-round-trippable serializer over optimagic's algorithm and
    ``minimize()``-level settings; see each field for what it carries. When
    optimagic is importable, :attr:`algorithm`, :attr:`options`, and the option
    groups are all checked on construction by instantiating the matching optimagic
    dataclass, so an unknown algorithm name or option key fails with a field-level
    message drawn from optimagic itself. (optimagic's dataclasses do not
    range/type-check individual values at construction; those surface at solve time.)

    Example::

        MinimizerConfig(
            algorithm="scipy_lbfgsb",
            options={"stopping_maxiter": 500},
            numdiff={"n_cores": 4},
            n_opt_cores=4,
        )
    """

    __toml_tag__: ClassVar[str | None] = "minimizer"

    algorithm: str = Field(
        default=DEFAULT_ALGORITHM,
        description=(
            "Name of the optimagic algorithm to use — a key of "
            "optimagic.algorithms.AVAILABLE_ALGORITHMS (the installed subset is "
            "listed by available_algorithms()). Defaults to 'scipy_lbfgsb' "
            "(gradient-based, bound-aware, always available)."
        ),
        examples=["scipy_lbfgsb", "scipy_neldermead"],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/algorithms.html"}
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Overrides forwarded to the algorithm's optimagic settings dataclass; "
            "keys not given keep optimagic's defaults. Valid keys depend on "
            "'algorithm' and are reported by algorithm_settings()."
        ),
        examples=[{"stopping_maxiter": 500, "convergence_ftol_rel": 1e-9}],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/algorithms.html"}
    )
    scaling: dict[str, Any] | None = Field(
        default=None,
        description=(
            "optimagic minimize()-level scaling options. None leaves scaling off "
            "(optimagic's default); a dict enables it and is forwarded to "
            "optimagic.ScalingOptions."
        ),
        examples=[{"method": "start_values", "magnitude": 1.0}],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/how_to/how_to_scaling.html"}
    )
    multistart: dict[str, Any] | None = Field(
        default=None,
        description=(
            "optimagic minimize()-level multistart options. None leaves multistart "
            "off (optimagic's default); a dict enables it and is forwarded to "
            "optimagic's MultistartOptions."
        ),
        examples=[{"n_samples": 20, "n_cores": 4}],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"}
    )
    numdiff: dict[str, Any] | None = Field(
        default=None,
        description=(
            "optimagic minimize()-level numerical-differentiation options. None uses "
            "optimagic's own numdiff defaults; a dict overrides them via optimagic's "
            "NumdiffOptions."
        ),
        examples=[{"n_cores": 4}],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"}
    )
    error_handling: Literal["raise", "continue"] = Field(
        default="raise",
        description=(
            "optimagic error handling when the criterion raises: 'raise' (default) "
            "or 'continue' (substitute a penalty value and keep going, e.g. to ride "
            "out occasional NaNs during multistart)."
        ),
        examples=["raise", "continue"],
        json_schema_extra={"docs_url": "https://optimagic.readthedocs.io/en/latest/how_to/how_to_errors_during_optimization.html"}
    )
    n_opt_cores: int = Field(
        default=1,
        description=(
            "Total cores a single optimagic fit consumes — parallel finite "
            "differences (set via 'numdiff'), multistart, and any algorithm-specific "
            "parallelism combined. The fit driver trusts this number to budget the "
            "outer resample pool (outer_workers * n_opt_cores * blas_threads <= cpus); "
            "it does not configure optimagic's parallelism for you."
        ),
        examples=[1, 4],
    )

    @model_validator(mode="after")
    def _validate_against_optimagic(self) -> MinimizerConfig:
        # Validate only when optimagic is present; otherwise the config stays
        # loadable/serializable and build() raises the clear ImportError later.
        try:
            algo_cls = _algorithm_class(self.algorithm)
        except ImportError:
            return self
        try:
            algo_cls(**self.options)
        except (TypeError, ValueError) as exc:
            valid = sorted(f.name for f in fields(algo_cls))
            raise ValueError(
                f"invalid options for optimagic algorithm {self.algorithm!r}: {exc}. "
                f"Valid option keys: {valid}"
            ) from exc
        for group in _OPTION_CLASS_NAMES:
            mapping = getattr(self, group)
            if mapping is not None:
                _validate_option_mapping(group, mapping)
        if self.n_opt_cores < 1:
            raise ValueError("n_opt_cores must be a positive integer")
        return self

    def build(self) -> Algorithm:
        """Construct the configured optimagic ``Algorithm`` instance."""
        return _algorithm_class(self.algorithm)(**self.options)

    def build_scaling(self) -> Any:
        """``ScalingOptions`` when ``scaling`` is set, else ``False`` (disabled)."""
        return _build_option("scaling", self.scaling) if self.scaling is not None else False

    def build_multistart(self) -> Any:
        """``MultistartOptions`` when ``multistart`` is set, else ``False``."""
        if self.multistart is None:
            return False
        return _build_option("multistart", self.multistart)

    def build_numdiff(self) -> Any:
        """``NumdiffOptions`` when ``numdiff`` is set, else ``None`` (optimagic default)."""
        return _build_option("numdiff", self.numdiff) if self.numdiff is not None else None

    def minimize_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`optimagic.minimize` from this config.

        Builds the algorithm and the enabled ``minimize()``-level option groups.
        Parallelism (numdiff ``n_cores``, multistart cores, …) is taken verbatim
        from the option groups; the driver budgets around :attr:`n_opt_cores`.
        """
        kwargs: dict[str, Any] = {
            "algorithm": self.build(),
            "error_handling": self.error_handling,
        }
        scaling = self.build_scaling()
        if scaling is not False:
            kwargs["scaling"] = scaling
        multistart = self.build_multistart()
        if multistart is not False:
            kwargs["multistart"] = multistart
        numdiff = self.build_numdiff()
        if numdiff is not None:
            kwargs["numdiff_options"] = numdiff
        return kwargs

    def resolved_settings(self) -> dict[str, Any]:
        """Effective settings: algorithm defaults merged with ``options``.

        TOML-native (see :func:`_jsonify_settings`): enums cleaned, tuples listed,
        ``None`` defaults dropped (absent ⇒ optimagic's default).
        """
        return _jsonify_settings(asdict(self.build()))

    def _resolved_group(self, group: str) -> dict[str, Any] | None:
        """Full settings for an option ``group`` (defaults included) or ``None``."""
        mapping = getattr(self, group)
        if mapping is None:
            return None
        return _jsonify_settings(asdict(_build_option(group, mapping)))

    def resolved(self) -> MinimizerConfig:
        """Return a copy whose option groups hold their full resolved settings.

        Expands ``options`` and every enabled ``minimize()``-level group
        (``scaling``/``multistart``/``numdiff``) to all of optimagic's settings
        (defaults included), so :meth:`to_toml` writes a fully explicit,
        self-documenting config. Groups left unset (``None``) stay unset.
        """
        update: dict[str, Any] = {"options": self.resolved_settings()}
        for group in _OPTION_CLASS_NAMES:
            update[group] = self._resolved_group(group)
        return self.model_copy(update=update)

    def capabilities(self) -> dict[str, Any]:
        """Capability flags (``algo_info``) for the selected algorithm."""
        return algorithm_capabilities(self.algorithm)

    @staticmethod
    def available_algorithms() -> list[str]:
        """Sorted names of selectable optimagic algorithms."""
        return available_algorithms()
