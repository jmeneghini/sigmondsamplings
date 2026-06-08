"""Typer entry point for ``ss-query``."""

from __future__ import annotations

from pathlib import Path

import typer

from .query import (
    QuerySpec,
    apply_query,
    column_records,
    collection_dataframe,
    file_info,
    group_records,
    hdf5_paths,
    load_collection,
    parse_columns,
    select_columns,
    unique_records,
)
from .plot import GENERIC_PLOT_METHODS, render_generic_plot, render_spectrum_plot
from .render import render_dataframe, render_records


app = typer.Typer(
    help="View and query Sigmond sampling and bins files.",
    no_args_is_help=True,
)

RAW_FRONT = ("name", "data_str", "mean", "error")
ENERGY_FRONT = ("name", "sector", "energy_type", "psq", "irrep", "level_index", "data_str", "mean", "error")
FORMATS = ["table", "json", "csv"]


@app.command()
def paths(file: Path) -> None:
    """List Sigmond HDF5 data paths."""
    records = [{"path": path} for path in hdf5_paths(file)]
    render_records(records, fmt="table")


@app.command()
def info(
    file: Path,
    hdf5_path: str | None = typer.Option(None, "--hdf5-path", help="Root path inside HDF5 file."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Show a compact file summary."""
    _check_format(fmt)
    render_records([file_info(file, hdf5_path=hdf5_path)], fmt=fmt)


@app.command()
def obs(
    file: Path,
    where: list[str] | None = typer.Option(None, "--where", "-w", help="Filter as attr=value. Repeat or use comma values for membership."),
    unique: str | None = typer.Option(None, "--unique", help="Show unique values for attr or attr,attr."),
    group: str | None = typer.Option(None, "--group", help="Group by attr or attr,attr and show counts."),
    list_columns: bool = typer.Option(False, "--list-columns", help="List available dataframe columns after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    columns: str | None = typer.Option(None, "--columns", help="Comma-separated display columns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated ObservableInfo attrs to omit from automatic columns."),
    contains: str | None = typer.Option(None, "--contains", help="Keep observables whose name contains text."),
    regex: str | None = typer.Option(None, "--regex", help="Keep observables whose name matches regex."),
    sort: str | None = typer.Option(None, "--sort", help="Sort by attr or attr,attr."),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Maximum rows to show."),
    hdf5_path: str | None = typer.Option(None, "--hdf5-path", help="Root path inside HDF5 file."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query raw observables."""
    _query_view(
        file,
        energy=False,
        hdf5_path=hdf5_path,
        where=where,
        unique=unique,
        group=group,
        list_columns=list_columns,
        plot=plot,
        plot_spectrum=False,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        columns=columns,
        exclude=exclude,
        contains=contains,
        regex=regex,
        sort=sort,
        reverse=reverse,
        limit=limit,
        fmt=fmt,
    )


@app.command()
def energy(
    file: Path,
    where: list[str] | None = typer.Option(None, "--where", "-w", help="Filter as attr=value. Repeat or use comma values for membership."),
    unique: str | None = typer.Option(None, "--unique", help="Show unique values for attr or attr,attr."),
    group: str | None = typer.Option(None, "--group", help="Group by attr or attr,attr and show counts."),
    list_columns: bool = typer.Option(False, "--list-columns", help="List available dataframe columns after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_spectrum: bool = typer.Option(False, "--plot-spectrum", help="Render the queried energy collection with SectorSpectrumPlotter."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    columns: str | None = typer.Option(None, "--columns", help="Comma-separated display columns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated ObservableInfo attrs to omit from automatic columns."),
    contains: str | None = typer.Option(None, "--contains", help="Keep observables whose name contains text."),
    regex: str | None = typer.Option(None, "--regex", help="Keep observables whose name matches regex."),
    sort: str | None = typer.Option(None, "--sort", help="Sort by attr or attr,attr."),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Maximum rows to show."),
    hdf5_path: str | None = typer.Option(None, "--hdf5-path", help="Root path inside HDF5 file."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query energy observables."""
    _query_view(
        file,
        energy=True,
        hdf5_path=hdf5_path,
        where=where,
        unique=unique,
        group=group,
        list_columns=list_columns,
        plot=plot,
        plot_spectrum=plot_spectrum,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        columns=columns,
        exclude=exclude,
        contains=contains,
        regex=regex,
        sort=sort,
        reverse=reverse,
        limit=limit,
        fmt=fmt,
    )


def _query_view(
    file: Path,
    *,
    energy: bool,
    hdf5_path: str | None,
    where: list[str] | None,
    unique: str | None,
    group: str | None,
    list_columns: bool,
    plot: str | None,
    plot_spectrum: bool,
    plot_output: Path | None,
    no_gui: bool,
    plot_obs_index: int | None,
    plot_panels: str | None,
    columns: str | None,
    exclude: str | None,
    contains: str | None,
    regex: str | None,
    sort: str | None,
    reverse: bool,
    limit: int | None,
    fmt: str,
) -> None:
    _check_format(fmt)
    collection = load_collection(file, hdf5_path=hdf5_path, energy=energy)
    try:
        collection = apply_query(
            collection,
            QuerySpec(
                where=tuple(where or ()),
                contains=contains,
                regex=regex,
                sort=sort,
                reverse=reverse,
                limit=limit,
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if plot_spectrum and not energy:
        raise typer.BadParameter("--plot-spectrum is only available for energy queries.")
    if plot and plot_spectrum:
        raise typer.BadParameter("Use either --plot or --plot-spectrum, not both.")
    plot_options_used = (
        plot_output is not None
        or no_gui
        or plot_obs_index is not None
        or plot_panels is not None
    )
    if plot_options_used and not (plot or plot_spectrum):
        raise typer.BadParameter(
            "Plot options require --plot or --plot-spectrum."
        )
    if plot_spectrum and (plot_obs_index is not None or plot_panels is not None):
        raise typer.BadParameter(
            "--plot-obs-index and --plot-panels are only valid with --plot."
        )

    mode_count = sum(bool(value) for value in (unique, group, list_columns, plot, plot_spectrum))
    if mode_count > 1:
        raise typer.BadParameter(
            "Use only one of --unique, --group, --list-columns, --plot, or --plot-spectrum."
        )

    if unique:
        render_records(unique_records(collection, unique), fmt=fmt)
        return

    if group:
        render_records(group_records(collection, group), fmt=fmt)
        return

    selected_columns = parse_columns(columns)
    excluded_attrs = parse_columns(exclude)
    preferred_front = ENERGY_FRONT if energy else RAW_FRONT
    if list_columns:
        render_records(
            column_records(collection, preferred_front, excluded_attrs=excluded_attrs),
            fmt=fmt,
        )
        return

    if plot:
        _require_nonempty_collection(collection)
        try:
            render_generic_plot(
                collection,
                method=plot,
                output=plot_output,
                show=not no_gui,
                obs_index=plot_obs_index,
                panels=plot_panels,
            )
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(f"Could not render plot: {exc}") from exc
        return

    if plot_spectrum:
        _require_nonempty_collection(collection)
        try:
            render_spectrum_plot(collection, output=plot_output, show=not no_gui)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(f"Could not render spectrum plot: {exc}") from exc
        return

    df = collection_dataframe(
        collection,
        selected_columns or preferred_front,
        excluded_attrs=excluded_attrs,
    )
    df = select_columns(df, selected_columns, preferred_front)
    render_dataframe(df, fmt=fmt)


def _check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise typer.BadParameter(f"--format must be one of: {', '.join(FORMATS)}")


def _require_nonempty_collection(collection) -> None:
    if len(collection) == 0:
        raise typer.BadParameter(
            "Query matched no observables; cannot plot. "
            "Check --where, --contains, and --regex filters."
        )


def main() -> None:
    """Console-script wrapper."""
    app()


if __name__ == "__main__":
    main()
