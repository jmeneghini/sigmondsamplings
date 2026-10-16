"""Plot renderers for queried ``ss query`` collections."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from slat.plotting.lifecycle import close_figure, figure_factory, retain_figure

from ..plot_contracts import get_plot_definition, plot_definitions

GENERIC_PLOT_METHODS = tuple(d.name for d in plot_definitions(family="sampling"))


@figure_factory
def render_generic_plot(
    collection,
    *,
    method: str,
    output: Path | None = None,
    show: bool = True,
    obs_index: int | None = None,
    panels: str | None = None,
    latex: bool = False,
    backend: str | None = None,
) -> None:
    """Render a generic SamplingPlotter method from a queried collection."""
    definition = get_plot_definition(method)
    options = {}
    if method == "histogram":
        options["sampling"] = 0 if obs_index is None else obs_index
    elif method == "summary":
        if panels is not None:
            options["panels"] = _parse_panels(panels)
        if obs_index is not None:
            options["obs_index"] = obs_index
    elif panels is not None or obs_index is not None:
        raise ValueError("panels/obs_index are not options for this method")
    with _backend_context(backend), _latex_context(latex):
        result = definition.make(collection, options=options, backend=backend)
        _finish_plot(result.figure, output=output, show=show)


@figure_factory
def render_spectrum_plot(
    collection,
    *,
    output: Path | None = None,
    show: bool = True,
    latex: bool = False,
    backend: str | None = None,
) -> None:
    """Render a SectorSpectrumPlotter plot from a queried energy collection."""
    from sigmondsamplings.plotting import SectorSpectrumPlotter

    with _backend_context(backend), _latex_context(latex):
        target = SectorSpectrumPlotter(collection).plot()
        _finish_plot(target, output=output, show=show)


#: ``ss query corr --plot`` methods for one correlator (``--sink/--source``).
PAIR_PLOT_METHODS = ("correlator", "effective-energy", "summary")
#: ``ss query corr --plot`` methods for a channel matrix (``--matrix``).
MATRIX_PLOT_METHODS = ("diagonal", "effective-energy", "grid", "heatmap", "coverage")
CORRELATOR_PLOT_METHODS = tuple(dict.fromkeys(PAIR_PLOT_METHODS + MATRIX_PLOT_METHODS))
EFFECTIVE_KINDS = ("forward", "symmetric", "forward-const", "symmetric-const")
HEATMAP_VALUES = ("normalized", "significance", "hermiticity")


@figure_factory
def render_correlator_plot(
    data,
    *,
    method: str,
    output: Path | None = None,
    show: bool = True,
    latex: bool = False,
    backend: str | None = None,
    panels: str | None = None,
    labels=None,
    eff_kind: str = "forward",
    eff_step: int = 1,
    time_extent: int | None = None,
    norm_time: int | None = None,
    value: str | None = None,
    time: int | None = None,
) -> None:
    """Render a correlator plot for ``ss query corr --plot``.

    ``data`` is what the query selected: a 1-D correlator (``SigmondData``)
    for ``--sink/--source``, a ``CorrelatorMatrix`` for ``--matrix``, or the
    coverage ``DataArray`` for ``--plot coverage``. ``labels`` label the
    matrix operators (the CLI passes their ``--operators`` numbers).
    """
    from sigmondsamplings.correlators.matrix import CorrelatorMatrix
    from sigmondsamplings.plotting import (
        CorrelatorMatrixPlotter,
        CorrelatorPlotter,
        plot_coverage,
    )

    kind = _effective_kind(eff_kind)
    if value is not None and value not in HEATMAP_VALUES:
        raise ValueError(f"--plot-value must be one of: {', '.join(HEATMAP_VALUES)}")
    effective = {"kind": kind, "step": eff_step, "time_extent": time_extent}

    with _backend_context(backend), _latex_context(latex):
        if isinstance(data, CorrelatorMatrix):
            if method not in MATRIX_PLOT_METHODS:
                raise ValueError(
                    f"--plot {method} is not a matrix plot; with --matrix use one of: "
                    f"{', '.join(MATRIX_PLOT_METHODS)}"
                )
            plotter = CorrelatorMatrixPlotter(
                data, labels=labels if labels is not None else "index"
            )
            shown = (
                time
                if time is not None
                else (norm_time if norm_time is not None else data.times[0])
            )
            if method == "diagonal":
                target = plotter.plot_diagonal()
            elif method == "effective-energy":
                target = plotter.plot_diagonal(effective=True, **effective)
            elif method == "grid":
                what = "normalized" if norm_time is not None else "correlator"
                target = plotter.plot_grid(what=what, norm_time=norm_time)
            else:  # heatmap
                target = plotter.plot_heatmap(
                    shown, value=value or "normalized", norm_time=norm_time
                )
        elif method == "coverage":
            target = plot_coverage(data, labels=labels if labels is not None else "index")
        else:
            if method not in PAIR_PLOT_METHODS:
                raise ValueError(
                    f"--plot {method} needs --matrix; with --sink/--source use one of: "
                    f"{', '.join(PAIR_PLOT_METHODS)}"
                )
            plotter = CorrelatorPlotter(data)
            if method == "correlator":
                target = plotter.plot()
            elif method == "effective-energy":
                target = plotter.plot_effective_energy(**effective)
            else:  # summary
                chosen = _parse_panels(panels)
                target = plotter.plot_summary(
                    panels=chosen or ("correlator", "effective_energy"), **effective
                )

        _finish_plot(target, output=output, show=show)


def _effective_kind(text: str) -> str:
    kind = text.strip().lower().replace("_", "-")
    if kind not in EFFECTIVE_KINDS:
        raise ValueError(f"--eff-kind must be one of: {', '.join(EFFECTIVE_KINDS)}")
    return kind.replace("-", "_")


@contextmanager
def _backend_context(backend: str | None):
    """Select the plotting backend for this render, if one was requested."""
    import slat.plotting as slp

    if backend is None:
        yield
        return
    if backend not in slp.available():
        raise ValueError(
            f"Backend {backend!r} is not available here. Available: {slp.available()}"
        )
    with slp.backend(backend):
        yield


@contextmanager
def _latex_context(latex: bool):
    """Render with matplotlib's TeX text engine when ``latex`` is set."""
    import slat.plotting as slp

    if not latex:
        yield
        return
    if slp.current() != "matplotlib":
        # text.usetex has no plotly analogue; MathJax already handles the math.
        yield
        return

    import matplotlib.pyplot as plt

    with plt.rc_context({"text.usetex": True}):
        yield


def _finish_plot(target, *, output: Path | None, show: bool) -> None:
    import slat.plotting as slp

    fig = target.figure if isinstance(target, slp.Axes) else target
    if output is not None:
        fig.save(output)
    if show:
        fig.show()
        retain_figure(fig)
    else:
        close_figure(fig)


def _parse_panels(panels: str | None) -> list[str] | None:
    if panels is None:
        return None
    # Panel names are spelled with hyphens on the command line, underscores in Python.
    parsed = [panel.strip().replace("-", "_") for panel in panels.split(",") if panel.strip()]
    return parsed or None
