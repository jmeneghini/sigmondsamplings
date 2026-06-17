"""Typer entry point for ``ss-query``."""

from __future__ import annotations

from pathlib import Path

import typer

from .plot import GENERIC_PLOT_METHODS
from .query import (
    ENERGY_DEFAULT_SORT,
    QUERY_FORMATS,
    QuerySpec,
    file_info,
    hdf5_paths,
    load_collection,
    run_query_view,
)
from .render import render_records

app = typer.Typer(
    help="View and query Sigmond sampling and bins files.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


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
    list_attrs: bool = typer.Option(False, "--list-attrs", help="List available attributes with their occurrence counts after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
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
        list_attrs=list_attrs,
        plot=plot,
        plot_spectrum=False,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        latex=latex,
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
    list_attrs: bool = typer.Option(False, "--list-attrs", help="List available attributes with their occurrence counts after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_spectrum: bool = typer.Option(False, "--plot-spectrum", help="Render the queried energy collection with SectorSpectrumPlotter."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
    columns: str | None = typer.Option(None, "--columns", help="Comma-separated display columns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated ObservableInfo attrs to omit from automatic columns."),
    contains: str | None = typer.Option(None, "--contains", help="Keep observables whose name contains text."),
    regex: str | None = typer.Option(None, "--regex", help="Keep observables whose name matches regex."),
    sort: str | None = typer.Option(None, "--sort", help="Sort by attr or attr,attr."),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Maximum rows to show."),
    save: Path | None = typer.Option(None, "--save", help="Write the queried spectrum to a spec TOML instead of displaying it."),
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
        list_attrs=list_attrs,
        plot=plot,
        plot_spectrum=plot_spectrum,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        latex=latex,
        columns=columns,
        exclude=exclude,
        contains=contains,
        regex=regex,
        sort=sort,
        reverse=reverse,
        limit=limit,
        fmt=fmt,
        save=save,
    )


def _query_view(
    file: Path,
    *,
    energy: bool,
    hdf5_path: str | None,
    where: list[str] | None,
    unique: str | None,
    group: str | None,
    list_attrs: bool,
    plot: str | None,
    plot_spectrum: bool,
    plot_output: Path | None,
    no_gui: bool,
    plot_obs_index: int | None,
    plot_panels: str | None,
    latex: bool,
    columns: str | None,
    exclude: str | None,
    contains: str | None,
    regex: str | None,
    sort: str | None,
    reverse: bool,
    limit: int | None,
    fmt: str,
    save: Path | None = None,
) -> None:
    collection = load_collection(file, hdf5_path=hdf5_path, energy=energy)
    spec = QuerySpec(
        where=tuple(where or ()),
        contains=contains,
        regex=regex,
        sort=sort,
        reverse=reverse,
        limit=limit,
        default_sort=ENERGY_DEFAULT_SORT if energy else None,
    )
    try:
        run_query_view(
            collection,
            energy=energy,
            spec=spec,
            unique=unique,
            group=group,
            list_attrs=list_attrs,
            plot=plot,
            plot_spectrum=plot_spectrum,
            plot_output=plot_output,
            no_gui=no_gui,
            plot_obs_index=plot_obs_index,
            plot_panels=plot_panels,
            latex=latex,
            columns=columns,
            exclude=exclude,
            fmt=fmt,
            save=save,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _check_format(fmt: str) -> None:
    if fmt not in QUERY_FORMATS:
        raise typer.BadParameter(f"--format must be one of: {', '.join(QUERY_FORMATS)}")


def main() -> None:
    """Console-script wrapper."""
    app()


if __name__ == "__main__":
    main()
