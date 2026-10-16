"""Plotting for sigmondsamplings.

The backend-neutral core lives in :mod:`slat.plotting`; this package holds the
domain plotters built on it and re-exports the core so callers need only one
import::

    from sigmondsamplings import plotting as ssp

    with ssp.backend("plotly"):
        axes = ssp.SectorSpectrumPlotter(levels).plot()
        axes.figure.save("spectrum.html")

Matplotlib is the publication backend, plotly the exploration/report one. See
``docs/plotting_api_design.md`` for what is portable and what is not.
"""

from __future__ import annotations

from slat.plotting import (
    Axes,
    Figure,
    Mark,
    PlotStyle,
    Recorder,
    UnsupportedFeature,
    available,
    backend,
    current,
    figure,
    get_style,
    record,
    set_style,
    style_context,
    use,
    wrap,
)

from ..plot_contracts import (
    PlotDefinition,
    PlotResult,
    UncertaintyOptions,
    get_plot_definition,
    plot_definitions,
)
from ._util import energy_level_hover_text, full_sample_and_bounds, ni_level_hover_text
from .correlators import (
    CorrelatorMatrixPlotter,
    CorrelatorPlotter,
    operator_labels,
    plot_coverage,
)
from .markers import FitOverlay, HMarker
from .overlaps import OverlapPlotter
from .sampling import SamplingPlotter
from .series import SeriesPlotter, SeriesStyle
from .spectrum import (
    PreparedSpectrum,
    SectorSpectrumPlotter,
    SpectrumPlotter,
    SpectrumStyle,
    draw_spectrum,
    prepare_spectrum,
)

__all__ = [
    "PlotDefinition", "PlotResult", "get_plot_definition", "plot_definitions",
    "UncertaintyOptions", "PreparedSpectrum", "prepare_spectrum", "draw_spectrum",
    "Axes",
    "CorrelatorMatrixPlotter",
    "CorrelatorPlotter",
    "Figure",
    "FitOverlay",
    "HMarker",
    "Mark",
    "PlotStyle",
    "Recorder",
    "SamplingPlotter",
    "OverlapPlotter",
    "SectorSpectrumPlotter",
    "SeriesPlotter",
    "SeriesStyle",
    "SpectrumPlotter",
    "SpectrumStyle",
    "UnsupportedFeature",
    "available",
    "backend",
    "current",
    "figure",
    "energy_level_hover_text",
    "ni_level_hover_text",
    "full_sample_and_bounds",
    "get_style",
    "operator_labels",
    "plot_coverage",
    "record",
    "set_style",
    "style_context",
    "use",
    "wrap",
]
