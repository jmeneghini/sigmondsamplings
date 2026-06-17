"""Collection-oriented query helpers for the ``ss-query`` CLI."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from sigmondsamplings.io.loader import SigmondLoader, is_hdf5_file, verify_sigmond_hdf5


@dataclass(frozen=True)
class QuerySpec:
    """Normalized query options shared by raw and energy views."""

    where: tuple[str, ...] = ()
    contains: str | None = None
    regex: str | None = None
    sort: str | None = None
    reverse: bool = False
    limit: int | None = None
    default_sort: tuple[str, ...] | None = None


# Output/display defaults shared by every front-end (ss-query and kb).
QUERY_FORMATS = ("table", "json", "csv")
RAW_FRONT = ("name", "data_str", "mean", "error")
ENERGY_FRONT = (
    "name",
    "sector",
    "energy_type",
    "psq",
    "irrep",
    "level_index",
    "data_str",
    "mean",
    "error",
)
ENERGY_DEFAULT_SORT = ("obs_kind", "sector", "level_index", "energy_type")


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
        if available is not None and attr not in available:
            raise ValueError(_unknown_attr_message(attr, available))
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
    elif spec.default_sort:
        collection = collection.sort(list(spec.default_sort), nulls_last=True)

    if spec.limit is not None:
        collection = collection[: spec.limit]

    return collection


def run_query_view(
    collection,
    *,
    energy: bool,
    spec: QuerySpec,
    unique: str | None = None,
    group: str | None = None,
    list_attrs: bool = False,
    plot: str | None = None,
    plot_spectrum: bool = False,
    plot_output: str | Path | None = None,
    no_gui: bool = False,
    plot_obs_index: int | None = None,
    plot_panels: str | None = None,
    latex: bool = False,
    columns: str | None = None,
    exclude: str | None = None,
    fmt: str = "table",
    save: str | Path | None = None,
):
    """Apply *spec* to *collection*, then render, plot, or save the result.

    The data-source-agnostic core of the query CLI: ``ss-query`` hands it a
    file-loaded collection, while other front-ends (e.g. a project-wide
    multi-ensemble collection) pass their own. Performs the requested terminal
    action (table/json/csv render, plot, or ``--save`` spectrum config) and
    returns the filtered collection.

    Raises :class:`ValueError` for invalid option combinations so each caller can
    translate it into its own CLI error type. Plotting imports are deferred so
    callers that never plot do not pull in matplotlib.
    """
    from .render import render_dataframe, render_records

    if fmt not in QUERY_FORMATS:
        raise ValueError(f"--format must be one of: {', '.join(QUERY_FORMATS)}")

    collection = apply_query(collection, spec)

    if save is not None and not energy:
        raise ValueError("--save writes an energy spectrum config and requires the energy view.")
    if plot_spectrum and not energy:
        raise ValueError("--plot-spectrum is only available for energy queries.")
    if plot and plot_spectrum:
        raise ValueError("Use either --plot or --plot-spectrum, not both.")
    plot_options_used = (
        plot_output is not None
        or no_gui
        or plot_obs_index is not None
        or plot_panels is not None
        or latex
    )
    if plot_options_used and not (plot or plot_spectrum):
        raise ValueError("Plot options require --plot or --plot-spectrum.")
    if plot_spectrum and (plot_obs_index is not None or plot_panels is not None):
        raise ValueError("--plot-obs-index and --plot-panels are only valid with --plot.")

    mode_count = sum(
        bool(value) for value in (unique, group, list_attrs, plot, plot_spectrum, save)
    )
    if mode_count > 1:
        raise ValueError(
            "Use only one of --unique, --group, --list-attrs, --plot, "
            "--plot-spectrum, or --save."
        )

    if save is not None:
        collection.save_spec(str(save))
        return collection

    if unique:
        render_records(unique_records(collection, unique), fmt=fmt)
        return collection

    if group:
        render_records(group_records(collection, group), fmt=fmt)
        return collection

    selected_columns = parse_columns(columns)
    excluded_attrs = parse_columns(exclude)
    preferred_front = ENERGY_FRONT if energy else RAW_FRONT

    if list_attrs:
        render_records(
            attr_records(collection, preferred_front, excluded_attrs=excluded_attrs),
            fmt=fmt,
        )
        return collection

    if plot or plot_spectrum:
        if len(collection) == 0:
            raise ValueError(
                "Query matched no observables; cannot plot. "
                "Check --where, --contains, and --regex filters."
            )
        from .plot import render_generic_plot, render_spectrum_plot

        try:
            if plot:
                render_generic_plot(
                    collection,
                    method=plot,
                    output=plot_output,
                    show=not no_gui,
                    obs_index=plot_obs_index,
                    panels=plot_panels,
                    latex=latex,
                )
            else:
                render_spectrum_plot(
                    collection, output=plot_output, show=not no_gui, latex=latex
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not render plot: {exc}") from exc
        return collection

    df = collection_dataframe(
        collection,
        selected_columns or preferred_front,
        excluded_attrs=excluded_attrs,
    )
    df = select_columns(df, selected_columns, preferred_front)
    render_dataframe(df, fmt=fmt)
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


def attr_records(
    collection,
    preferred_front: Sequence[str],
    *,
    excluded_attrs: Iterable[str] | None = None,
) -> list[dict[str, str | int]]:
    """Return one record per available attribute with its non-null occurrence count."""
    df = collection_dataframe(
        collection,
        columns=preferred_front,
        excluded_attrs=excluded_attrs,
    )
    df = select_columns(df, None, preferred_front)
    return [
        {"attribute": str(column), "count": int(df[column].notna().sum())}
        for column in df.columns
    ]


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
    if isinstance(sample, tuple):
        if isinstance(value, str) and ":" in value:
            return _parse_tuple_value(value)
        if isinstance(value, list):
            if all(isinstance(item, str) and ":" in item for item in value):
                return [_parse_tuple_value(item) for item in value]
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


def _unknown_attr_message(attr: str, available: set[str]) -> str:
    available_sorted = sorted(available)
    aliases = {"obs_type": "obs_kind"}
    alias = aliases.get(attr)
    matches = [alias] if alias in available else get_close_matches(attr, available_sorted, n=1)
    suffix = f" Did you mean {matches[0]!r}?" if matches else ""
    sample = ", ".join(available_sorted[:12])
    if len(available_sorted) > 12:
        sample += ", ..."
    return f"Unknown --where attribute {attr!r}.{suffix} Available attributes include: {sample}"


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
