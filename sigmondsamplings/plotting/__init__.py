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
    FEATURES,
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

from .sampling import SamplingPlotter
from .spectrum import HMarker, SectorSpectrumPlotter, SpectrumPlotter, SpectrumStyle

__all__ = [
    "FEATURES",
    "Axes",
    "Figure",
    "HMarker",
    "Mark",
    "PlotStyle",
    "Recorder",
    "SamplingPlotter",
    "SectorSpectrumPlotter",
    "SpectrumPlotter",
    "SpectrumStyle",
    "UnsupportedFeature",
    "available",
    "backend",
    "current",
    "figure",
    "get_style",
    "record",
    "set_style",
    "style_context",
    "use",
    "wrap",
]
