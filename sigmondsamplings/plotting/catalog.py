"""Execution of the lightweight plot catalog (no KBfit dependencies)."""

from __future__ import annotations

from importlib.util import find_spec

import slat.plotting as slp
from slat.plotting.lifecycle import track_figure, track_native_figure

from ..plot_contracts import DiagnosticStyle, PlotInputError, PlotResult


def _plotter(data):
    from .sampling import SamplingPlotter

    return (
        data
        if isinstance(data, SamplingPlotter)
        else getattr(data, "plotter", None) or SamplingPlotter(data)
    )


def validate_input(definition, data, options):
    name = definition.name
    if definition.family == "diagnostic":
        if find_spec("optimagic") is None:
            raise slp.UnsupportedFeature("Fit diagnostics require optimagic")
        # Loaded HDF5 results require an external trace; live OptimizeResults
        # carry history. Duck-typed diagnostic providers implement the contract.
        if hasattr(data, "optimize_result"):
            result = data.optimize_result
            if not getattr(result, "history", None) and getattr(data, "trace_path", None) is None:
                raise PlotInputError("Fit diagnostics need optimization history or a trace_path")
        return
    if name == "corner" and find_spec("corner") is None:
        raise slp.UnsupportedFeature("The corner plot requires the optional corner package")
    stats = _plotter(data).stats
    if stats is None or not len(stats):
        raise PlotInputError("Sampling diagnostics require nonempty scalar samplings")
    if any(s.ndim != 0 or s.is_complex for s in stats):
        raise PlotInputError("Sampling diagnostics require real scalar samplings")
    index = (
        getattr(options, "sampling", None)
        if name == "histogram"
        else getattr(options, "obs_index", None)
    )
    if index is not None and index >= len(stats):
        raise PlotInputError(f"observable index {index} is out of range")
    methods = {s.sampling_info.method for s in stats}
    if name == "bootstrap-intervals" and methods != {"bootstrap"}:
        raise PlotInputError("bootstrap intervals require bootstrap samplings for every observable")


def make_plot(definition, data, *, options=None, style=None, backend=None):
    options = definition.options_model.model_validate(options or {})
    style = DiagnosticStyle.model_validate(style or {})
    backend = backend or slp.current()
    if backend not in definition.backends:
        raise slp.UnsupportedFeature(f"{definition.name} does not support backend {backend!r}")
    target = data if definition.family == "diagnostic" else _plotter(data)
    definition.validate_input(target, options)
    effective = options.model_dump(mode="json", exclude_none=True)
    geometry = style.figure
    with slp.figure_scope() as scope, slp.backend(backend), slp.style_context(dpi=geometry.dpi):
        kwargs = options.model_dump(exclude_none=True)
        if geometry.figsize is not None and definition.takes_figsize:
            kwargs["figsize"] = geometry.figsize
        made = getattr(target, definition.method)(**kwargs)
        if not isinstance(made, slp.Axes | slp.Figure):
            if backend == "matplotlib":
                import matplotlib.pyplot as plt
                track_native_figure(made, plt.close)
            else:
                track_native_figure(made, _clear_plotly)
        figure = slp.as_figure(made)
        # Native figures (corner, optimagic) bypass slp.figure; let the scope
        # close them too if anything below fails.
        if not any(tracked is figure for tracked in scope.figures):
            track_figure(figure)
        if geometry.figsize is not None and not definition.takes_figsize:
            figure.set_size(*geometry.figsize)
        synthetic = getattr(data, "is_synthetic", None)
        origin = {True: "synthetic", False: "resampled"}.get(synthetic, "unknown")
        if synthetic is True:
            figure.suptitle("Synthetic parameter samplings")
        statistics = definition.statistics(options, target)
        scope.retain(figure)
        return PlotResult(
            figure,
            effective,
            style.model_dump(mode="json"),
            metadata={"sampling_origin": origin, "statistics": statistics},
        )


def _clear_plotly(figure):
    figure.data = ()
