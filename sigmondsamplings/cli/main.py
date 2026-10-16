"""Typer entry point for the unified ``ss`` CLI.

Two areas: ``ss query …`` (read/inspect Sigmond files) and the write commands
``ss convert`` / ``ss combine`` / ``ss edit`` / ``ss pivot``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .edit import edit
from .plot import CORRELATOR_PLOT_METHODS, GENERIC_PLOT_METHODS, MATRIX_PLOT_METHODS
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
    time: int | None = typer.Option(None, "--time", help="Time for --matrix (default: the first complete time of the selected basis)."),
    hermitian: bool | None = typer.Option(None, "--hermitian/--no-hermitian", help="Use (or derive) hermitian-averaged correlators. Default: on for --matrix; for --sink/--source, off unless the correlator is stored only as HermMat."),
    subtract_vev: bool = typer.Option(False, "--subtract-vev", help="Use (or derive) VEV-subtracted correlators (bins need --sampling)."),
    key_form: str | None = typer.Option(None, "--key-form", help="Stored form: plain, HermMat, SubVEV, or HermMat SubVEV."),
    sampling: str | None = typer.Option(None, "--sampling", help="Resample bins before display: 'jackknife' or 'bootstrap:N[:SEED]'."),
    plot: str | None = typer.Option(None, "--plot", help=f"Plot the selection instead of listing it: {', '.join(CORRELATOR_PLOT_METHODS)}."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_panels: str | None = typer.Option(None, "--plot-panels", help="Comma-separated summary panels (single-correlator summary)."),
    plot_backend: str | None = typer.Option(None, "--plot-backend", help="Plotting backend: matplotlib (default) or plotly."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
    eff_kind: str = typer.Option("forward", "--eff-kind", help="Effective-energy kind: forward, symmetric, forward-const or symmetric-const."),
    eff_step: int = typer.Option(1, "--eff-step", min=1, help="Effective-energy time step."),
    time_extent: int | None = typer.Option(None, "--time-extent", min=1, help="Lattice time extent T for effective energies (default: the ensemble's)."),
    norm_time: int | None = typer.Option(None, "--norm-time", help="Normalize matrix plots at this time (grid, heatmap)."),
    plot_value: str | None = typer.Option(None, "--plot-value", help="Heatmap value: normalized (default), significance or hermiticity."),
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Query correlators: stored series by default, or operators, channels, VEVs, one correlator, or a matrix."""
    from . import corr as views

    _check_format(fmt)
    single = sink is not None or source is not None
    modes = sum(bool(m) for m in (operators, channels, vevs, single, matrix))
    if modes > 1:
        raise typer.BadParameter(
            "Use only one of --operators, --channels, --vevs, --sink/--source, or --matrix."
        )
    if single and (sink is None or source is None):
        raise typer.BadParameter("--sink and --source must be given together.")
    if times is not None and not single:
        raise typer.BadParameter("--times requires --sink/--source.")
    if time is not None and not matrix:
        raise typer.BadParameter("--time requires --matrix.")
    plot_options_used = (
        plot_output is not None
        or no_gui
        or plot_panels is not None
        or plot_backend is not None
        or latex
        or eff_kind != "forward"
        or eff_step != 1
        or time_extent is not None
        or norm_time is not None
        or plot_value is not None
    )
    if plot_options_used and plot is None:
        raise typer.BadParameter("Plot options require --plot.")
    if plot is not None:
        if plot not in CORRELATOR_PLOT_METHODS:
            raise typer.BadParameter(
                f"Unknown --plot {plot!r}. Expected one of: {', '.join(CORRELATOR_PLOT_METHODS)}"
            )
        if not (single or matrix):
            raise typer.BadParameter("--plot requires --sink/--source or --matrix.")
        if plot_value is not None and plot != "heatmap":
            raise typer.BadParameter("--plot-value applies to --plot heatmap.")
        if matrix and time is not None and plot != "heatmap":
            raise typer.BadParameter("--time applies to the heatmap plot.")
    if ops is not None and not matrix:
        raise typer.BadParameter("--ops requires --matrix.")
    if (hermitian is not None or subtract_vev or sampling or key_form is not None) and not (single or matrix):
        raise typer.BadParameter(
            "--hermitian, --subtract-vev and --sampling require --sink/--source or --matrix."
        )

    try:
        index = views.load_index(file, group=group)
        sampling_info = views.parse_sampling(index, sampling) if sampling else None
        if plot is not None:
            _plot_correlators(
                views,
                index,
                plot,
                single=single,
                sink=sink,
                source=source,
                times=times,
                matrix=matrix,
                ops=ops,
                time=time,
                hermitian=hermitian,
                subtract_vev=subtract_vev,
                key_form=key_form,
                sampling_info=sampling_info,
                output=plot_output,
                show=not no_gui,
                panels=plot_panels,
                backend=plot_backend,
                latex=latex,
                eff_kind=eff_kind,
                eff_step=eff_step,
                time_extent=time_extent,
                norm_time=norm_time,
                value=plot_value,
            )
            return
        if operators:
            records = views.operator_records(index)
        elif channels:
            records = views.channel_records(index)
        elif vevs:
            records = views.vev_records(index)
        elif single:
            time_list = [int(t) for t in times.split(",")] if times else None
            records = views.correlator_records(
                index,
                sink,
                source,
                times=time_list,
                hermitian=hermitian,
                subtract_vev=subtract_vev,
                key_form=key_form,
                sampling_info=sampling_info,
            )
        elif matrix:
            shown, records = views.matrix_records(
                index,
                matrix,
                time,
                hermitian=True if hermitian is None else hermitian,
                subtract_vev=subtract_vev,
                key_form=key_form,
                sampling_info=sampling_info,
                operators=views.parse_operator_numbers(ops) if ops is not None else None,
            )
            if fmt == "table":
                typer.echo(f"time = {shown}")
        else:
            records = views.series_records(index)
    except (KeyError, ValueError, OSError) as exc:
        message = (
            str(exc)
            .strip("'\"")
            .replace(
                "set hermitian=/subtract_vev= to match", "use --hermitian/--subtract-vev to match"
            )
            .replace("pass time_extent= explicitly", "pass --time-extent")
        )
        raise typer.BadParameter(message) from exc
    render_records(records, fmt=fmt)


_PIVOT_PLOT_METHODS = tuple(m for m in MATRIX_PLOT_METHODS if m != "coverage")


@app.command()
def pivot(
    file: Path,
    matrix: str = typer.Option(..., "--matrix", help="Channel to pivot (number from `ss query corr --channels`, or label substring)."),
    norm_time: int = typer.Option(..., "--norm-time", help="Normalization time tN."),
    metric_time: int = typer.Option(..., "--metric-time", help="Metric time t0."),
    diagonalize_time: int = typer.Option(..., "--diagonalize-time", help="Diagonalization time tD."),
    ops: str | None = typer.Option(None, "--ops", help="Basis operators, as numbers from `ss query corr --operators` (e.g. 0-6,8; default: the whole channel)."),
    min_inv_condition: float = typer.Option(1 / 150, "--min-inv-condition", help="Metric and reduced-matrix eigenvalue cut."),
    numeric_type: str = typer.Option("complex", "--numeric-type", help="Sigmond build convention: complex or real."),
    subtract_vev: bool = typer.Option(False, "--subtract-vev", help="Build on VEV-subtracted correlators (bins need --sampling)."),
    key_form: str | None = typer.Option(None, "--key-form", help="Stored form: plain, HermMat, SubVEV, or HermMat SubVEV."),
    sampling: str | None = typer.Option(None, "--sampling", help="Resample bins first: 'jackknife' or 'bootstrap:N[:SEED]'."),
    write_pivot: Path | None = typer.Option(None, "--write-pivot", help="Write the reusable SS-native pivot to this HDF5 file."),
    pivot_group: str = typer.Option("pivot", "--pivot-group", help="HDF5 group for --write-pivot."),
    write_rotated: Path | None = typer.Option(None, "--write-rotated", help="Write the rotated correlators to this HDF5 file."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output groups."),
    diagonal_only_before_time: int | None = typer.Option(None, "--diagonal-only-before-time", help="Export only diagonal keys before this time."),
    remove_off_diagonal: bool = typer.Option(False, "--remove-off-diagonal", help="Export only diagonal correlator keys."),
    upper_triangle_only: bool = typer.Option(False, "--upper-triangle-only", help="Export only the upper triangle in rotated level order."),
    plot: str | None = typer.Option(None, "--plot", help=f"Plot the rotated matrix instead of reporting: {', '.join(_PIVOT_PLOT_METHODS)}."),
    plot_output: Path | None = typer.Option(None, "--plot-output", help="Write plot to this path."),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not display the plot GUI."),
    plot_backend: str | None = typer.Option(None, "--plot-backend", help="Plotting backend: matplotlib (default) or plotly."),
    latex: bool = typer.Option(False, "--latex", help="Render plot text with matplotlib's LaTeX engine."),
    eff_kind: str = typer.Option("forward", "--eff-kind", help="Effective-energy kind: forward, symmetric, forward-const or symmetric-const."),
    eff_step: int = typer.Option(1, "--eff-step", min=1, help="Effective-energy time step."),
    time_extent: int | None = typer.Option(None, "--time-extent", min=1, help="Lattice time extent T for effective energies (default: the ensemble's)."),
    time: int | None = typer.Option(None, "--time", help="Time for --plot heatmap."),
    plot_value: str | None = typer.Option(None, "--plot-value", help="Heatmap value: normalized (default), significance or hermiticity."),
    group: str | None = typer.Option(None, "--group", help="HDF5 root group to read."),
    fmt: str = typer.Option("table", "--format", help="Output format: table, json, or csv."),
) -> None:
    """Build a fixed full-sample GEVP pivot for one channel and rotate it.

    Reports the pivot and its off-diagonal check, or plots the rotated matrix;
    optionally writes the pivot and the rotated correlators.
    """
    from sigmondsamplings.correlators import SinglePivot

    from . import corr as views

    _check_format(fmt)
    if (diagonal_only_before_time is not None or remove_off_diagonal or upper_triangle_only) and write_rotated is None:
        raise typer.BadParameter("Export filters require --write-rotated.")
    plot_options_used = (plot_output is not None or no_gui or plot_backend is not None or latex
                         or eff_kind != "forward" or eff_step != 1 or time_extent is not None
                         or time is not None or plot_value is not None)
    if plot_options_used and plot is None:
        raise typer.BadParameter("Plot options require --plot.")
    if plot is not None and plot not in _PIVOT_PLOT_METHODS:
        raise typer.BadParameter(f"Unknown --plot {plot!r}. Expected one of: {', '.join(_PIVOT_PLOT_METHODS)}")
    if (plot_value is not None or time is not None) and plot != "heatmap":
        raise typer.BadParameter("--time and --plot-value apply to --plot heatmap.")
    try:
        index = views.load_index(file, group=group)
        sampling_info = views.parse_sampling(index, sampling) if sampling else None
        if index.is_bins and subtract_vev and sampling_info is None:
            raise ValueError("connected pivot construction from bins requires --sampling")
        basis = views.channel_operators(index, matrix, views.parse_operator_numbers(ops) if ops else None)
        assembled = index.matrix(basis, hermitian=True, subtract_vev=subtract_vev,
                                 sampling_info=sampling_info, key_form=key_form,
                                 include_vevs=subtract_vev and all(op in index.vev_operators for op in basis))
        built = SinglePivot.from_matrix(assembled, norm_time=norm_time, metric_time=metric_time,
                                        diagonalize_time=diagonalize_time,
                                        min_inv_condition=min_inv_condition, numeric_type=numeric_type)
        rotated = built.rotate(assembled)
        if write_rotated is not None:
            rotated.write(write_rotated, overwrite=overwrite,
                          diagonal_only_before_time=diagonal_only_before_time,
                          remove_off_diagonal=remove_off_diagonal, upper_triangle_only=upper_triangle_only)
        if write_pivot is not None:
            built.write(write_pivot, group=pivot_group, overwrite=overwrite)
        if plot is not None:
            from .plot import render_correlator_plot

            if rotated.is_bins:
                raise ValueError("pivot plots of bins require --sampling")
            render_correlator_plot(rotated, method=plot, labels=[str(op) for op in rotated.operators],
                                   time=time, output=plot_output, show=not no_gui, panels=None,
                                   backend=plot_backend, latex=latex, eff_kind=eff_kind,
                                   eff_step=eff_step, time_extent=time_extent, norm_time=None,
                                   value=plot_value)
            return
        records = [{"pivot": built.summary(), "off_diagonal": built.off_diagonal_check(rotated).summary()}]
    except (KeyError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc).strip("'\"")) from exc
    render_records(records, fmt=fmt)


def _plot_correlators(
    views,
    index,
    method: str,
    *,
    single: bool,
    sink,
    source,
    times,
    matrix,
    ops,
    time,
    hermitian,
    subtract_vev,
    key_form,
    sampling_info,
    **render,
) -> None:
    """Build what ``ss query corr`` selected and hand it to the correlator renderer."""
    from .plot import render_correlator_plot

    if index.is_bins and sampling_info is None and method != "coverage":
        raise ValueError(
            "plots of bins need --sampling (e.g. --sampling jackknife): errors and effective "
            "energies are computed from samplings"
        )
    if single:
        data = views.single_correlator(
            index,
            sink,
            source,
            times=[int(t) for t in times.split(",")] if times else None,
            hermitian=hermitian,
            subtract_vev=subtract_vev,
            key_form=key_form,
            sampling_info=sampling_info,
        )
        render_correlator_plot(data, method=method, **render)
        return
    numbers = views.parse_operator_numbers(ops) if ops is not None else None
    flags = {"hermitian": True if hermitian is None else hermitian, "subtract_vev": subtract_vev, "key_form": key_form}
    if method == "coverage":
        data = views.channel_coverage(index, matrix, operators=numbers, **flags)
        labels = views.operator_numbers(index, data.attrs["operators"])
    else:
        data = views.channel_matrix(
            index, matrix, operators=numbers, sampling_info=sampling_info, **flags
        )
        labels = views.operator_numbers(index, data.operators)
    render_correlator_plot(data, method=method, labels=labels, time=time, **render)


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
