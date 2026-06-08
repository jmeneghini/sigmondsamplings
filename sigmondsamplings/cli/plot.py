"""Plot renderers for queried ``ss-query`` collections."""

from __future__ import annotations

from pathlib import Path


GENERIC_PLOT_METHODS = (
    "histogram",
    "errorbar",
    "correlation",
    "eff-sample-size",
    "summary",
    "bootstrap-intervals",
    "corner",
)


def render_generic_plot(
    collection,
    *,
    method: str,
    output: Path | None = None,
    show: bool = True,
    obs_index: int | None = None,
    panels: str | None = None,
) -> None:
    """Render a generic SamplingPlotter method from a queried collection."""
    if method not in GENERIC_PLOT_METHODS:
        raise ValueError(
            f"Unknown plot method {method!r}. Expected one of: "
            f"{', '.join(GENERIC_PLOT_METHODS)}"
        )

    from sigmondsamplings.plotter import SamplingPlotter

    plotter = SamplingPlotter(collection)
    if method == "histogram":
        target = plotter.plot_sampling_histogram(sampling=0 if obs_index is None else obs_index)
    elif method == "errorbar":
        target = plotter.plot_sampling_errorbar()
    elif method == "correlation":
        target = plotter.plot_correlation_matrix()
    elif method == "eff-sample-size":
        target = plotter.plot_effective_sample_size()
    elif method == "summary":
        target = plotter.plot_stats_summary(
            panels=_parse_panels(panels),
            obs_index=obs_index,
        )
    elif method == "bootstrap-intervals":
        target = plotter.plot_bootstrap_intervals()
    else:
        target = plotter.plot_corner()

    _finish_plot(target, output=output, show=show)


def render_spectrum_plot(
    collection,
    *,
    output: Path | None = None,
    show: bool = True,
) -> None:
    """Render a SectorSpectrumPlotter plot from a queried energy collection."""
    from sigmondsamplings.spectrum_plotter import SectorSpectrumPlotter

    ax = SectorSpectrumPlotter(collection).plot()
    _finish_plot(ax, output=output, show=show)


def _finish_plot(target, *, output: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = target.figure if hasattr(target, "figure") else target
    if output is not None:
        fig.savefig(output)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_panels(panels: str | None) -> list[str] | None:
    if panels is None:
        return None
    parsed = [panel.strip() for panel in panels.split(",") if panel.strip()]
    return parsed or None
