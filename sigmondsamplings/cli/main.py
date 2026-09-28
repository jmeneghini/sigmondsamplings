"""Typer entry point for the unified ``ss`` CLI.

Two areas: ``ss query …`` (read/inspect Sigmond files) and the write commands
``ss convert`` / ``ss combine`` / ``ss edit``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .edit import edit
from .plot import GENERIC_PLOT_METHODS
from .query import (
    ENERGY_DEFAULT_SORT,
    QUERY_FORMATS,
    QuerySpec,
    file_info,
    load_collection,
    root_groups,
    run_query_view,
)
from .render import render_records
from .write import combine, convert

app = typer.Typer(
    help="Inspect, convert, and combine Sigmond sampling and bins files.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

query_app = typer.Typer(
    help="View and query Sigmond sampling and bins files.",
    no_args_is_help=True,
)
app.add_typer(query_app, name="query")

# Write-side commands share the root app alongside the query group.
app.command()(convert)
app.command()(combine)
app.command()(edit)


@query_app.command()
def groups(file: Path) -> None:
    """List Sigmond HDF5 root groups."""
    records = [{"group": group} for group in root_groups(file)]
    render_records(records, fmt="table")


@query_app.command()
def info(
    file: Path,
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Show a compact file summary."""
    _check_format(fmt)
    render_records([file_info(file, group=group)], fmt=fmt)


@query_app.command()
def obs(
    file: Path,
    where: list[str] | None = typer.Option(None, "--where", "-w", help="Filter as attr=value. Repeat or use comma values for membership."),
    unique: str | None = typer.Option(None, "--unique", help="Show unique values for attr or attr,attr."),
    group_by: str | None = typer.Option(None, "--group-by", help="Group by attr or attr,attr and show counts."),
    list_attrs: bool = typer.Option(False, "--list-attrs", help="List available attributes with their occurrence counts after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    plot_backend: str | None = typer.Option(None, "--plot-backend", help="Plotting backend: matplotlib (default) or plotly."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
    columns: str | None = typer.Option(None, "--columns", help="Comma-separated display columns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated ObservableInfo attrs to omit from automatic columns."),
    contains: str | None = typer.Option(None, "--contains", help="Keep observables whose name contains text."),
    regex: str | None = typer.Option(None, "--regex", help="Keep observables whose name matches regex."),
    sort: str | None = typer.Option(None, "--sort", help="Sort by attr or attr,attr."),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Maximum rows to show."),
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query raw observables."""
    _query_view(
        file,
        energy=False,
        group=group,
        where=where,
        unique=unique,
        group_by=group_by,
        list_attrs=list_attrs,
        plot=plot,
        plot_spectrum=False,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        plot_backend=plot_backend,
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


@query_app.command()
def energy(
    file: Path,
    where: list[str] | None = typer.Option(None, "--where", "-w", help="Filter as attr=value. Repeat or use comma values for membership."),
    unique: str | None = typer.Option(None, "--unique", help="Show unique values for attr or attr,attr."),
    group_by: str | None = typer.Option(None, "--group-by", help="Group by attr or attr,attr and show counts."),
    list_attrs: bool = typer.Option(False, "--list-attrs", help="List available attributes with their occurrence counts after filtering."),
    plot: str | None = typer.Option(None, "--plot", help=f"Render plot method: {', '.join(GENERIC_PLOT_METHODS)}."),
    plot_spectrum: bool = typer.Option(False, "--plot-spectrum", help="Render the queried energy collection with SectorSpectrumPlotter."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_obs_index: int | None = typer.Option(None, "--plot-obs-index", min=0, help="Observable index for histogram/summary plots."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels."),
    plot_backend: str | None = typer.Option(None, "--plot-backend", help="Plotting backend: matplotlib (default) or plotly."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
    columns: str | None = typer.Option(None, "--columns", help="Comma-separated display columns."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated ObservableInfo attrs to omit from automatic columns."),
    contains: str | None = typer.Option(None, "--contains", help="Keep observables whose name contains text."),
    regex: str | None = typer.Option(None, "--regex", help="Keep observables whose name matches regex."),
    sort: str | None = typer.Option(None, "--sort", help="Sort by attr or attr,attr."),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Maximum rows to show."),
    save: Path | None = typer.Option(None, "--save", help="Write the queried spectrum to a spec TOML instead of displaying it."),
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query energy observables."""
    _query_view(
        file,
        energy=True,
        group=group,
        where=where,
        unique=unique,
        group_by=group_by,
        list_attrs=list_attrs,
        plot=plot,
        plot_spectrum=plot_spectrum,
        plot_output=plot_output,
        no_gui=no_gui,
        plot_obs_index=plot_obs_index,
        plot_panels=plot_panels,
        plot_backend=plot_backend,
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


@query_app.command()
def corr(
    file: Path,
    operators: bool = typer.Option(False, "--operators", help="List operators, numbered for --sink/--source."),
    channels: bool = typer.Option(False, "--channels", help="List channels, numbered for --matrix."),
    vevs: bool = typer.Option(False, "--vevs", help="List stored VEVs with their values."),
    sink: str | None = typer.Option(None, "--sink", help="Sink operator (number from --operators, or short string)."),
    source: str | None = typer.Option(None, "--source", help="Source operator (number from --operators, or short string)."),
    matrix: str | None = typer.Option(None, "--matrix", help="Show the correlation matrix of a channel (number from --channels, or label substring)."),
    times: str | None = typer.Option(None, "--times", help="Comma-separated times for --sink/--source (default: all stored)."),
    ops: str | None = typer.Option(None, "--ops", help="Operators for --matrix, as numbers from --operators (e.g. 0-6,8; default: the whole channel)."),
    time: int | None = typer.Option(None, "--time", help="Time for --matrix (default: the first stored)."),
    hermitian: bool | None = typer.Option(None, "--hermitian/--no-hermitian", help="Use (or derive) hermitian-averaged correlators. Default: on for --matrix; for --sink/--source, off unless the pair is stored only as HermMat."),
    subtract_vev: bool = typer.Option(False, "--subtract-vev", help="Use (or derive) VEV-subtracted correlators (bins need --sampling)."),
    sampling: str | None = typer.Option(None, "--sampling", help="Resample bins before display: 'jackknife' or 'bootstrap:N[:SEED]'."),
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query correlators: stored series by default, or operators, channels, VEVs, one correlator, or a matrix."""
    from . import corr as views

    _check_format(fmt)
    pair = sink is not None or source is not None
    modes = sum(bool(m) for m in (operators, channels, vevs, pair, matrix))
    if modes > 1:
        raise typer.BadParameter(
            "Use only one of --operators, --channels, --vevs, --sink/--source, or --matrix."
        )
    if pair and (sink is None or source is None):
        raise typer.BadParameter("--sink and --source must be given together.")
    if times is not None and not pair:
        raise typer.BadParameter("--times requires --sink/--source.")
    if time is not None and not matrix:
        raise typer.BadParameter("--time requires --matrix.")
    if ops is not None and not matrix:
        raise typer.BadParameter("--ops requires --matrix.")
    if (hermitian is not None or subtract_vev or sampling) and not (pair or matrix):
        raise typer.BadParameter(
            "--hermitian, --subtract-vev and --sampling require --sink/--source or --matrix."
        )

    try:
        index = views.load_index(file, group=group)
        sampling_info = views.parse_sampling(index, sampling) if sampling else None
        if operators:
            records = views.operator_records(index)
        elif channels:
            records = views.channel_records(index)
        elif vevs:
            records = views.vev_records(index)
        elif pair:
            time_list = [int(t) for t in times.split(",")] if times else None
            records = views.correlator_records(
                index,
                sink,
                source,
                times=time_list,
                hermitian=hermitian,
                subtract_vev=subtract_vev,
                sampling_info=sampling_info,
            )
        elif matrix:
            shown, records = views.matrix_records(
                index,
                matrix,
                time,
                hermitian=True if hermitian is None else hermitian,
                subtract_vev=subtract_vev,
                sampling_info=sampling_info,
                operators=views.parse_operator_numbers(ops) if ops is not None else None,
            )
            if fmt == "table":
                typer.echo(f"time = {shown}")
        else:
            records = views.series_records(index)
    except (KeyError, ValueError) as exc:
        message = str(exc).strip("'\"").replace(
            "set hermitian=/subtract_vev= to match", "use --hermitian/--subtract-vev to match"
        )
        raise typer.BadParameter(message) from exc
    render_records(records, fmt=fmt)


def _query_view(
    file: Path,
    *,
    energy: bool,
    group: str | None,
    where: list[str] | None,
    unique: str | None,
    group_by: str | None,
    list_attrs: bool,
    plot: str | None,
    plot_spectrum: bool,
    plot_output: Path | None,
    no_gui: bool,
    plot_obs_index: int | None,
    plot_panels: str | None,
    plot_backend: str | None,
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
    collection = load_collection(file, group=group, energy=energy)
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
            group_by=group_by,
            list_attrs=list_attrs,
            plot=plot,
            plot_spectrum=plot_spectrum,
            plot_output=plot_output,
            no_gui=no_gui,
            plot_obs_index=plot_obs_index,
            plot_panels=plot_panels,
            plot_backend=plot_backend,
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
