"""Shared style configuration.

One small dataclass, translated by each backend. This is deliberately *not* a
bridge between matplotlib's rcParams and plotly's templates -- it carries only
the handful of settings both backends need to agree on.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

from ..constants import COLORS, MARKERS, IndexedCycle

__all__ = ["PlotStyle", "get_style", "set_style", "style_context"]


@dataclass(frozen=True)
class PlotStyle:
    """Settings both backends honour.

    Attributes:
        colors: Palette used by ``color_cycle()``. Frontends pull explicit
            colors from it rather than relying on backend auto-cycling.
        markers: Default SeriesPlotter marker palette. An explicit
            SeriesStyle.marker_cycle overrides it.
        figsize: Default figure size in inches. Plotly lays figures out at a
            fixed 100 px per inch (``_plotly.PX_PER_INCH``).
        dpi: Dots per inch of raster exports, on both backends. It does not
            change a plotly figure's layout or its HTML size.
        grid / grid_alpha: Whether new axes start with a grid, and its opacity.
            Frontends never force a grid on; set ``PlotStyle(grid=True)`` or
            ``style_context(grid=True)`` to get one.
        label_fontsize / tick_fontsize / legend_fontsize / title_fontsize:
            In points. ``None`` takes matplotlib's default size, which the
            plotly backend reproduces. Axis labels default to 15 pt, 1.5x
            matplotlib's 10 pt, on both backends.
    """

    colors: Sequence[str] = field(default_factory=lambda: tuple(COLORS))
    markers: Sequence[str] = field(default_factory=lambda: tuple(MARKERS))
    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: float = 100.0
    grid: bool = False
    grid_alpha: float = 0.3
    label_fontsize: float | None = 15.0
    tick_fontsize: float | None = None
    legend_fontsize: float | None = None
    title_fontsize: float | None = None

    def color_cycle(self) -> IndexedCycle[str]:
        """A fresh, restartable cycle over :attr:`colors`."""
        return IndexedCycle(self.colors)

    def replace(self, **changes) -> PlotStyle:
        """A copy with ``changes`` applied."""
        return replace(self, **changes)


_style = PlotStyle()


def get_style() -> PlotStyle:
    """The active default style."""
    return _style


def set_style(style: PlotStyle | None = None, **changes) -> PlotStyle:
    """Replace the active default style; returns the previous one."""
    global _style
    previous = _style
    base = style if style is not None else _style
    _style = base.replace(**changes) if changes else base
    return previous


@contextlib.contextmanager
def style_context(style: PlotStyle | None = None, **changes) -> Iterator[PlotStyle]:
    """Apply a style for the duration of a block.

    ::

        with style_context(colors=["#000", "#f00"]):
            fig = figure()
    """
    previous = set_style(style, **changes)
    try:
        yield get_style()
    finally:
        set_style(previous)


def resolve_colors(colors: Iterable[str] | None, style: PlotStyle) -> IndexedCycle[str]:
    """A color cycle from an explicit palette, falling back to the style's."""
    return IndexedCycle(colors) if colors is not None else style.color_cycle()
