"""Shared style configuration.

One small dataclass, translated by each backend. This is deliberately *not* a
bridge between matplotlib's rcParams and plotly's templates -- it carries only
the handful of settings both backends need to agree on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from ..constants import COLORS, MARKERS, IndexedCycle

__all__ = ["PlotStyle", "get_style", "set_style", "style_context"]


@dataclass(frozen=True)
class PlotStyle:
    """Settings both backends honour.

    Attributes:
        colors: Palette used by ``color_cycle()``. Frontends pull explicit
            colors from it rather than relying on backend auto-cycling.
        markers: Marker palette used by ``marker_cycle()``.
        figsize: Default figure size in inches. Plotly converts via ``dpi``.
        dpi: Dots per inch. Sets raster export resolution and the
            inches-to-pixels conversion for plotly.
        grid / grid_alpha: Default grid visibility and opacity.
        label_fontsize / tick_fontsize / legend_fontsize / title_fontsize:
            ``None`` falls back to each backend's own default.
        mathtext: When False, LaTeX in labels is flattened to Unicode. Backends
            that cannot render math in tick labels and legend entries (plotly)
            do this regardless for those two places.
    """

    colors: Sequence[str] = field(default_factory=lambda: tuple(COLORS))
    markers: Sequence[str] = field(default_factory=lambda: tuple(MARKERS))
    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: float = 100.0
    grid: bool = True
    grid_alpha: float = 0.3
    label_fontsize: float | None = None
    tick_fontsize: float | None = None
    legend_fontsize: float | None = None
    title_fontsize: float | None = None
    mathtext: bool = True

    def color_cycle(self) -> IndexedCycle[str]:
        """A fresh, restartable cycle over :attr:`colors`."""
        return IndexedCycle(self.colors)

    def marker_cycle(self) -> IndexedCycle[str]:
        """A fresh, restartable cycle over :attr:`markers`."""
        return IndexedCycle(self.markers)

    def replace(self, **changes) -> PlotStyle:
        """A copy with ``changes`` applied."""
        return replace(self, **changes)

    @property
    def pixel_size(self) -> tuple[int, int]:
        """:attr:`figsize` converted to pixels at :attr:`dpi`."""
        return (round(self.figsize[0] * self.dpi), round(self.figsize[1] * self.dpi))


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


class style_context:
    """Context manager applying a style for the duration of a block.

    ::

        with style_context(colors=["#000", "#f00"]):
            fig = figure()
    """

    def __init__(self, style: PlotStyle | None = None, **changes):
        self._style = style
        self._changes = changes
        self._previous: PlotStyle | None = None

    def __enter__(self) -> PlotStyle:
        self._previous = set_style(self._style, **self._changes)
        return get_style()

    def __exit__(self, *exc) -> None:
        if self._previous is not None:
            set_style(self._previous)


def resolve_colors(colors: Iterable[str] | None, style: PlotStyle) -> IndexedCycle[str]:
    """A color cycle from an explicit palette, falling back to the style's."""
    return IndexedCycle(colors) if colors is not None else style.color_cycle()
