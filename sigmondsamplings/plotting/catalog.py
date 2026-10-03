"""Execution of the lightweight plot catalog (no KBfit dependencies)."""

from __future__ import annotations

from importlib.util import find_spec

import slat.plotting as slp

from ..plot_contracts import DiagnosticStyle, PlotInputError, PlotResult

_METHODS = {
    "histogram": "plot_sampling_histogram",
    "errorbar": "plot_sampling_errorbar",
    "correlation": "plot_correlation_matrix",
    "eff-sample-size": "plot_effective_sample_size",
    "summary": "plot_stats_summary",
    "bootstrap-intervals": "plot_bootstrap_intervals",
    "corner": "plot_corner",
}


def _plotter(data):
    from .sampling import SamplingPlotter

    return (
        data
        if isinstance(data, SamplingPlotter)
        else getattr(data, "plotter", None) or SamplingPlotter(data)
    )


def validate_input(name, family, data, options):
    if family == "diagnostic":
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
    made = None
    try:
        with slp.figure_scope() as scope, slp.backend(backend), slp.style_context(dpi=geometry.dpi):
            if definition.family == "diagnostic":
                made = getattr(data, f"{definition.name}_plot")()
            else:
                kwargs = options.model_dump(exclude_none=True)
                if geometry.figsize is not None and definition.name != "corner":
                    kwargs["figsize"] = geometry.figsize
                made = getattr(target, _METHODS[definition.name])(**kwargs)
            wrapped = slp.wrap(made)
            figure = wrapped.figure if isinstance(wrapped, slp.Axes) else wrapped
            if geometry.figsize is not None and (
                definition.family == "diagnostic" or definition.name == "corner"
            ):
                width, height = geometry.figsize
                if figure.backend == "matplotlib":
                    figure.native.set_size_inches(width, height)
                else:
                    figure.native.update_layout(
                        width=width * geometry.dpi, height=height * geometry.dpi
                    )
            synthetic = getattr(data, "is_synthetic", None)
            origin = (
                "synthetic"
                if synthetic is True
                else "resampled"
                if synthetic is False
                else "unknown"
            )
            if origin == "synthetic":
                figure.suptitle("Synthetic parameter samplings")
            statistics = {}
            if definition.family == "sampling":
                statistics = {"central": "mean", "error_metric": "error"}
                if definition.name == "bootstrap-intervals":
                    statistics["error_metric"] = "confidence_interval"
                    statistics["confidence_levels"] = effective["confidence_levels"]
                elif (
                    definition.name == "histogram"
                    and target.stats[0].sampling_info.method == "bootstrap"
                ):
                    statistics["error_metric"] = "confidence_interval"
                    statistics["confidence_level"] = effective["confidence_level"]
                elif definition.name in ("correlation", "eff-sample-size", "corner"):
                    statistics = {}
            scope.retain(figure)
            return PlotResult(
                figure,
                effective,
                style.model_dump(mode="json"),
                metadata={"sampling_origin": origin, "statistics": statistics},
            )
    except BaseException:
        if made is not None:
            wrapped = slp.wrap(made)
            (wrapped.figure if isinstance(wrapped, slp.Axes) else wrapped).close()
        raise
