"""Shared observable-selection language for the query and edit front-ends.

``ss query -w psq=0`` and ``ss edit -w psq=0`` mean exactly the same thing, because
both resolve through :func:`filter_collection` here. This module owns the four
selection clauses and the ``attr=value`` parsing that backs the CLI spelling:

``where``
    Attribute equality/membership, as a mapping or as ``attr=value`` strings.
``contains`` / ``regex``
    Substring and regular-expression matches against the observable name.
``spec``
    A spectrum TOML naming ``(psq, irrep, levels)`` sectors, as written by
    ``ss query energy --save``.

Clauses combine with AND. The parsing helpers previously lived in
``sigmondsamplings.cli.query``; they moved here so library code (the edit executor)
can use them without importing from ``cli``. ``cli.query`` re-exports them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from difflib import get_close_matches
from typing import Any

from .info import canonical_obs_kind

logger = logging.getLogger(__name__)

__all__ = [
    "filter_collection",
    "obs_attrs",
    "parse_where_specs",
    "resolve_where",
    "unknown_attr_message",
]


def filter_collection(
    collection,
    *,
    where: Mapping[str, Any] | Iterable[str] | None = None,
    contains: str | None = None,
    regex: str | None = None,
    spec: str | None = None,
):
    """Return the subset of *collection* matching every provided clause.

    The returned collection wraps the *same* observable objects as *collection*
    (``ObservableCollection.filter`` does not copy), so in-place mutation of the
    result is visible in the original. Callers wanting independence must ``copy()``.

    Args:
        collection: Any :class:`~sigmondsamplings.observable_collection.ObservableCollection`.
        where: ``{attr: value}`` mapping, or an iterable of ``"attr=value"`` strings.
            List values are membership tests.
        contains: Keep observables whose name contains this substring.
        regex: Keep observables whose name matches this regular expression.
        spec: Path to a spectrum TOML; keeps only the ``(psq, irrep, level)`` levels
            it names. Requires an energy-level collection.

    Raises:
        ValueError: For an unparseable spec string, an unknown attribute, or a
            ``spec`` clause on a collection that is not energy-typed.
    """
    filters = resolve_where(where, collection)
    if filters:
        collection = collection.filter(**filters)

    if contains:
        collection = collection.filter(
            predicate=lambda obs_info: contains in str(obs_info.name)
        )

    if regex:
        pattern = re.compile(regex)
        collection = collection.filter(
            predicate=lambda obs_info: pattern.search(str(obs_info.name)) is not None
        )

    if spec:
        filter_from_toml = getattr(collection, "filter_from_toml", None)
        if filter_from_toml is None:
            raise ValueError(
                "A spectrum spec selection requires an energy-level collection; "
                f"got {type(collection).__name__}. Interpret energy attrs first."
            )
        collection = filter_from_toml(str(spec))

    return collection


def resolve_where(
    where: Mapping[str, Any] | Iterable[str] | None,
    collection=None,
) -> dict[str, Any]:
    """Normalize a ``where`` clause into ``collection.filter`` kwargs.

    Accepts the mapping form used by recipe TOML (values already typed) and the
    ``"attr=value"`` string form used by the CLI (values coerced by
    :func:`_parse_scalar`). Both go through the same attribute validation and
    tuple-value normalization.
    """
    if where is None:
        return {}
    if isinstance(where, Mapping):
        return _resolve_where_mapping(where, collection)
    return parse_where_specs(where, collection)


def parse_where_specs(specs: Iterable[str] | None, collection=None) -> dict[str, Any]:
    """Parse repeated ``attr=value`` specs into collection filter kwargs."""
    filters: dict[str, Any] = {}
    # Available attribute names are identical across observables of a type, so scan
    # the collection once here and reuse the set for every spec rather than
    # rescanning all observables per spec.
    available = _available_attrs(collection) if collection is not None else None
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"Invalid --where {spec!r}; expected attr=value")
        attr, raw_value = spec.split("=", 1)
        attr = attr.strip()
        if not attr:
            raise ValueError(f"Invalid --where {spec!r}; attribute is empty")
        _check_attr(attr, available)
        value = _parse_where_value(raw_value.strip())
        if collection is not None:
            value = _normalize_filter_value(collection, attr, value)
        _merge_filter(filters, attr, value)
    return filters


def _resolve_where_mapping(where: Mapping[str, Any], collection=None) -> dict[str, Any]:
    """Validate an already-typed ``{attr: value}`` mapping from a recipe."""
    available = _available_attrs(collection) if collection is not None else None
    filters: dict[str, Any] = {}
    for attr, value in where.items():
        _check_attr(attr, available)
        if collection is not None:
            value = _normalize_filter_value(collection, attr, value)
        filters[attr] = value
    return filters


def _check_attr(attr: str, available: set[str] | None) -> None:
    if available is not None and attr not in available:
        raise ValueError(unknown_attr_message(attr, available))


def _parse_where_value(value: str) -> Any:
    if "," in value:
        return [_parse_scalar(part.strip()) for part in value.split(",") if part.strip()]
    return _parse_scalar(value)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _merge_filter(filters: dict[str, Any], attr: str, value: Any) -> None:
    if attr not in filters:
        filters[attr] = value
        return

    old_value = filters[attr]
    old_values = old_value if isinstance(old_value, list) else [old_value]
    new_values = value if isinstance(value, list) else [value]
    filters[attr] = old_values + new_values


def _normalize_obs_kind_value(collection, value: Any) -> Any:
    """Resolve ``obs_kind`` aliases (e.g. ``energy_sh``) to the canonical tag.

    ``obs_kind`` names one exact class, so a filter on it is deliberately not
    hierarchical: ``obs_kind=energy`` excludes single hadrons. That trips people up,
    so warn and point at the boolean facet, which is the hierarchical form.
    """
    values = value if isinstance(value, list) else [value]
    resolved = [canonical_obs_kind(item) for item in values]
    if "energy" in resolved and any(
        getattr(sampling.observable_info, "is_single_hadron", False) for sampling in collection
    ):
        logger.warning(
            "obs_kind=energy matches multi-hadron levels only; single hadrons are "
            "obs_kind=energy_single_hadron. Use is_energy=true for every energy level."
        )
    return resolved if isinstance(value, list) else resolved[0]


def _normalize_filter_value(collection, attr: str, value: Any) -> Any:
    if attr == "obs_kind":
        return _normalize_obs_kind_value(collection, value)
    sample = _first_non_none_attr(collection, attr)
    if isinstance(sample, tuple):
        if isinstance(value, str) and ":" in value:
            return _parse_tuple_value(value)
        if isinstance(value, list):
            if all(isinstance(item, str) and ":" in item for item in value):
                return [_parse_tuple_value(item) for item in value]
            if all(isinstance(item, (list, tuple)) for item in value):
                return [tuple(item) for item in value]
            raise ValueError(
                f"Tuple-valued filter '{attr}' must use ':' between tuple fields "
                "and ',' between values, e.g. sector=0:A1g,1:T1u"
            )
    return value


def _parse_tuple_value(value: str) -> tuple[Any, ...]:
    return tuple(_parse_scalar(part.strip()) for part in value.split(":"))


def _first_non_none_attr(collection, attr: str) -> Any:
    for sampling in collection:
        value = _get_attr(sampling, attr)
        if value is not None:
            return value
    return None


def _available_attrs(collection) -> set[str]:
    attrs: set[str] = set()
    for sampling in collection:
        for target in (sampling, sampling.observable_info, sampling.sampling_info):
            attrs.update(name for name in dir(target) if not name.startswith("_"))
            if hasattr(target, "__dict__"):
                attrs.update(name for name in vars(target) if not name.startswith("_"))
    return attrs


def obs_attrs(collection) -> set[str]:
    """Attribute names available on the observable metadata of *collection*.

    Narrower than :func:`_available_attrs`, which also reports sampling-level names:
    only observable metadata is writable, so this is what validates a ``--set``.
    """
    attrs: set[str] = set()
    for sampling in collection:
        target = sampling.observable_info
        attrs.update(name for name in dir(target) if not name.startswith("_"))
        if hasattr(target, "__dict__"):
            attrs.update(name for name in vars(target) if not name.startswith("_"))
    return attrs


def unknown_attr_message(attr: str, available: set[str], *, label: str = "--where") -> str:
    """Error text for an unrecognized attribute name, with a did-you-mean suggestion."""
    available_sorted = sorted(available)
    aliases = {
        "obs_type": "obs_kind",
        "kind": "obs_kind",
        "energy": "is_energy",
        "single_hadron": "is_single_hadron",
        "is_sh": "is_single_hadron",
    }
    alias = aliases.get(attr)
    matches = [alias] if alias in available else get_close_matches(attr, available_sorted, n=1)
    suffix = f" Did you mean {matches[0]!r}?" if matches else ""
    sample = ", ".join(available_sorted[:12])
    if len(available_sorted) > 12:
        sample += ", ..."
    return f"Unknown {label} attribute {attr!r}.{suffix} Available attributes include: {sample}"


def _get_attr(sampling, attr: str) -> Any:
    for target in (sampling, sampling.observable_info, sampling.sampling_info):
        if hasattr(target, attr):
            return getattr(target, attr)
    return None
