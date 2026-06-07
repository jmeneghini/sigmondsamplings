"""Collection-oriented query helpers for the ``ss-query`` CLI."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigmondsamplings.loader import SigmondLoader, is_hdf5_file, verify_sigmond_hdf5


@dataclass(frozen=True)
class QuerySpec:
    """Normalized query options shared by raw and energy views."""

    where: tuple[str, ...] = ()
    contains: str | None = None
    regex: str | None = None
    sort: str | None = None
    reverse: bool = False
    limit: int | None = None


def load_collection(
    filename: str | Path,
    *,
    hdf5_path: str | None = None,
    energy: bool = False,
    lazy: bool = True,
):
    """Load raw observables or the energy-level view for a Sigmond file."""
    filename = str(filename)
    use_lazy = lazy and is_hdf5_file(filename)
    loader = SigmondLoader(filename, hdf5_path=hdf5_path, lazy=use_lazy)
    if energy:
        return loader.energy_observables(return_type="list")
    collection = loader.observables
    collection.return_type = "list"
    return collection


def hdf5_paths(filename: str | Path) -> list[str]:
    """Return Sigmond HDF5 data paths in ``filename``."""
    is_valid, _file_kind, paths = verify_sigmond_hdf5(str(filename))
    if not is_valid:
        raise ValueError(f"{filename} is not a valid Sigmond HDF5 file")
    return paths or []


def file_info(filename: str | Path, *, hdf5_path: str | None = None) -> dict[str, Any]:
    """Return a compact info record for a Sigmond file."""
    collection = load_collection(filename, hdf5_path=hdf5_path, lazy=True)
    sampling_info = collection.shared_attr("sampling_info")
    ensemble_info = collection.shared_attr("ensemble_info")
    return {
        "file": str(filename),
        "count": len(collection),
        "sampling_info": sampling_info,
        "ensemble_info": ensemble_info,
    }


def parse_columns(columns: str | None) -> list[str] | None:
    """Parse a comma-separated column/attribute list."""
    if columns is None:
        return None
    parsed = [part.strip() for part in columns.split(",") if part.strip()]
    return parsed or None


def parse_attrs(attrs: str) -> list[str]:
    """Parse one or more comma-separated attribute names."""
    parsed = parse_columns(attrs)
    if not parsed:
        raise ValueError("Expected at least one attribute")
    return parsed


def parse_where_specs(specs: Iterable[str] | None, collection=None) -> dict[str, Any]:
    """Parse repeated ``attr=value`` specs into collection filter kwargs."""
    filters: dict[str, Any] = {}
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"Invalid --where {spec!r}; expected attr=value")
        attr, raw_value = spec.split("=", 1)
        attr = attr.strip()
        if not attr:
            raise ValueError(f"Invalid --where {spec!r}; attribute is empty")
        value = _parse_where_value(raw_value.strip())
        if collection is not None:
            value = _normalize_filter_value(collection, attr, value)
        _merge_filter(filters, attr, value)
    return filters


def apply_query(collection, spec: QuerySpec):
    """Apply generic CLI query options using collection methods."""
    filters = parse_where_specs(spec.where, collection)
    if filters:
        collection = collection.filter(**filters)

    if spec.contains:
        collection = collection.filter(
            predicate=lambda obs_info: spec.contains in str(obs_info.name)
        )

    if spec.regex:
        pattern = re.compile(spec.regex)
        collection = collection.filter(
            predicate=lambda obs_info: pattern.search(str(obs_info.name)) is not None
        )

    if spec.sort:
        attrs = parse_attrs(spec.sort)
        key: str | list[str] = attrs[0] if len(attrs) == 1 else attrs
        collection = collection.sort(key, reverse=spec.reverse, nulls_last=True)

    if spec.limit is not None:
        collection = collection[: spec.limit]

    return collection


def unique_records(collection, attrs: str) -> list[dict[str, str]]:
    """Return records for unique values of one or more attributes."""
    attr_list = parse_attrs(attrs)
    values = collection.unique(_composite_getter(attr_list))
    return [{"value": format_value(value)} for value in (values or [])]


def group_records(collection, attrs: str) -> list[dict[str, str | int]]:
    """Return one record per group with the group key and count."""
    attr_list = parse_attrs(attrs)
    groups = collection.group_by(_composite_getter(attr_list))
    return [
        {"group": format_value(key), "count": len(group)}
        for key, group in sorted(groups.items(), key=lambda item: format_value(item[0]))
    ]


def collection_dataframe(
    collection,
    columns: Sequence[str] | None = None,
    *,
    excluded_attrs: Iterable[str] | None = None,
):
    """Build a dataframe for display/export, augmenting requested computed attrs."""
    df = collection.to_dataframe(excluded_attrs=excluded_attrs)
    if columns:
        for column in columns:
            if column not in df.columns:
                values = attr_values(collection, column)
                if len(values) == len(df):
                    df[column] = values
    return df


def column_records(
    collection,
    preferred_front: Sequence[str],
    *,
    excluded_attrs: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Return display records for all available dataframe columns."""
    df = collection_dataframe(
        collection,
        columns=preferred_front,
        excluded_attrs=excluded_attrs,
    )
    df = select_columns(df, None, preferred_front)
    return [{"column": str(column)} for column in df.columns]


def attr_values(collection, attr: str) -> list[Any]:
    """Collect an attribute from sampling, observable-info, or sampling-info objects."""
    return [_get_attr(sampling, attr) for sampling in collection]


def select_columns(df, columns: Sequence[str] | None, preferred_front: Sequence[str]):
    """Select requested columns or move preferred columns to the front."""
    if columns:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {', '.join(missing)}")
        return df.loc[:, list(columns)]

    front = [column for column in preferred_front if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df.loc[:, front + rest]


def format_value(value: Any) -> str:
    """Format values consistently for terminal, CSV, and JSON records."""
    if isinstance(value, tuple):
        return ",".join(format_value(item) for item in value)
    if isinstance(value, list):
        return ",".join(format_value(item) for item in value)
    return "" if value is None else str(value)


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


def _normalize_filter_value(collection, attr: str, value: Any) -> Any:
    sample = _first_non_none_attr(collection, attr)
    if isinstance(sample, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _first_non_none_attr(collection, attr: str) -> Any:
    for sampling in collection:
        value = _get_attr(sampling, attr)
        if value is not None:
            return value
    return None


def _composite_getter(attrs: Sequence[str]):
    if len(attrs) == 1:
        attr = attrs[0]
        return lambda sampling: _get_attr(sampling, attr)
    return lambda sampling: tuple(_get_attr(sampling, attr) for attr in attrs)


def _get_attr(sampling, attr: str) -> Any:
    for target in (sampling, sampling.observable_info, sampling.sampling_info):
        if hasattr(target, attr):
            return getattr(target, attr)
    return None
