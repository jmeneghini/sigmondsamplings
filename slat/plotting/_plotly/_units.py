"""Sizes, fonts and stacking: matplotlib defaults in plotly units."""

from __future__ import annotations

#: The layout scale: plotly figures are laid out at this many pixels per inch
#: whatever ``style.dpi`` is. ``style.dpi`` only sets the resolution of raster
#: exports, as ``savefig(dpi=)`` does on matplotlib.
PX_PER_INCH = 100.0

#: matplotlib's default sizes in points (the rcParams named alongside), used
#: wherever a size is None so that both backends draw the same figure.
_DEFAULT_PT = {
    "linewidth": 1.5,  # lines.linewidth
    "markersize": 6.0,  # lines.markersize
    "markeredge": 1.0,  # lines.markeredgewidth
    "capsize": 0.0,  # errorbar.capsize
    "edgewidth": 1.0,  # patch.linewidth
    "font": 10.0,  # font.size: ticks, axis labels, legend, text
    "title": 12.0,  # axes.titlesize and figure.titlesize ("large")
    "titlepad": 6.0,  # axes.titlepad
    "labelpad": 4.0,  # axes.labelpad
    "spine": 0.8,  # axes.linewidth
    "ticklen": 3.5,  # xtick.major.size
    "grid": 0.8,  # grid.linewidth
}
_GRID_COLOR = "#b0b0b0"  # grid.color

#: A sans-serif close to matplotlib's DejaVu Sans. plotly's default, Open Sans
#: falling back to Verdana, draws a Greek pi that reads as Cyrillic "п", and
#: Greek letters are common in tick labels and legend entries here.
_FONT_FAMILY = "Arial, Helvetica, sans-serif"

# matplotlib's default zorders: images 0, patches 1, lines and collections 2.
_Z_IMAGE, _Z_PATCH, _Z_LINE = 0.0, 1.0, 2.0

# Line height as a multiple of the font size, for reserving room for text.
_LEADING = 1.25


def _px(points: float | None) -> float | None:
    """Typographic points as plotly layout pixels."""
    return None if points is None else points * PX_PER_INCH / 72.0


def _size(points: float | None, role: str) -> float:
    """``points`` in pixels, or matplotlib's default for ``role`` when None."""
    return _px(_DEFAULT_PT[role] if points is None else points)


def _suffix(index: int) -> str:
    """plotly's name suffix for the ``index``-th subplot (1-based): ``""`` for the first."""
    return "" if index == 1 else str(index)


def _z(zorder: float | None, default: float) -> int:
    """plotly's integer zorder for a matplotlib one, keeping tenths."""
    return round(10 * (default if zorder is None else zorder))
