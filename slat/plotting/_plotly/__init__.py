"""plotly backend.

The figure is drawn to match matplotlib's defaults: sizes given in points are
converted to pixels at :data:`PX_PER_INCH`, unset sizes take matplotlib's
default values, and each subplot has its own legend. Differences that remain:

* ``zorder`` is honoured between traces, but an axis line (a plotly shape)
  can only go under every trace or over every trace.
* :meth:`PlotlyAxes.marker_extent` is an estimate, and returns ``0.0`` until
  the y limits have been set.
* LaTeX in tick labels and legend entries is converted to HTML scripts and
  Unicode, because MathJax does not render reliably in those positions.
  Titles, axis labels and annotations use MathJax.
* Created grids and wrapped primary Cartesian cells retain trace-free cells through ``Figure.save``, ``embed_html``
  and ``show``. These append minimal null traces when needed; direct native
  exports bypass this step. Wrapping preserves native layouts and exposes their
  primary Cartesian cells; secondary and non-Cartesian axes stay native.
"""

from ._axes import PlotlyAxes
from ._figure import PlotlyFigure, make_figure, wrap
from ._units import PX_PER_INCH

__all__ = ["PX_PER_INCH", "PlotlyAxes", "PlotlyFigure", "make_figure", "wrap"]
