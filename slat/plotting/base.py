"""Backend-neutral drawing surfaces.

The two abstractions are :class:`Axes` (one set of axes -- the thing marks are
drawn onto) and :class:`Figure` (a grid of them, plus output). Backends
subclass both.

The mark vocabulary is deliberately small and closed. Anything not expressible
with it is either composed in numpy by the caller (histograms, ellipses) or
reached through :attr:`Axes.native`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "FEATURES",
    "Axes",
    "Coords",
    "Figure",
    "Origin",
    "UnsupportedFeature",
]

Coords = Literal["data", "axes"]
Origin = Literal["upper", "lower"]

#: Optional behaviours a backend may or may not provide. Query with
#: :meth:`Axes.supports` before relying on one.
FEATURES = frozenset(
    {
        "capsize",  # errorbar caps are drawn
        "contour",  # Axes.contour is implemented
        "hover",  # per-point hover text is honoured
        "mathtext",  # LaTeX renders in tick labels and legend entries
        "pixel_metrics",  # Axes.marker_extent is exact rather than estimated
        "raster_export",  # Figure.save can write png/pdf/svg
        "zorder",  # zorder= is honoured (plotly orders by insertion)
    }
)


class UnsupportedFeature(NotImplementedError):
    """Raised when a mark or option has no meaning on the active backend."""


class Axes(ABC):
    """One set of axes.

    Every mark accepts ``native=``: a dict forwarded verbatim to the underlying
    backend call. It is the escape hatch for backend-specific options and is by
    definition not portable.

    Colors are always explicit. Neither backend's automatic color cycling is
    used, because matplotlib cycles per-Axes and plotly cycles per-trace and the
    two will never agree; take colors from ``style.color_cycle()`` instead.
    """

    #: Features this backend provides; see :data:`FEATURES`.
    supported: frozenset[str] = frozenset()

    #: Backend name ("matplotlib", "plotly", "recording").
    backend: str = ""

    @property
    @abstractmethod
    def figure(self) -> Figure:
        """The Figure this Axes belongs to."""

    @property
    @abstractmethod
    def native(self) -> Any:
        """The underlying backend object.

        ``plt.Axes`` for matplotlib, ``(go.Figure, row, col)`` for plotly.
        """

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    @abstractmethod
    def line(
        self,
        x,
        y,
        *,
        color: str | None = None,
        width: float | None = None,
        style: str = "-",
        marker: str | None = None,
        markersize: float | None = None,
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A connected line through ``(x, y)``."""

    @abstractmethod
    def points(
        self,
        x,
        y,
        *,
        color: str | None = None,
        size: float | None = None,
        marker: str = "o",
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Unconnected markers. ``size`` is a marker diameter in points."""

    @abstractmethod
    def errorbar(
        self,
        x,
        y,
        *,
        yerr=None,
        xerr=None,
        color: str | None = None,
        marker: str | None = "o",
        markersize: float | None = None,
        capsize: float | None = None,
        width: float | None = None,
        style: str | None = None,
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Points with error bars.

        ``style=None`` (the default) draws markers only, matching the
        ``fmt="o"`` idiom. ``yerr``/``xerr`` accept a scalar, a 1-D array, or a
        ``(2, N)`` array of asymmetric ``(lower, upper)`` offsets.
        """

    @abstractmethod
    def band(
        self,
        x,
        lo,
        hi,
        *,
        color: str | None = None,
        alpha: float = 0.25,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A filled region between ``lo`` and ``hi`` over ``x``."""

    @abstractmethod
    def bar(
        self,
        x,
        height,
        *,
        width: float | Sequence[float] = 0.8,
        color: str | None = None,
        label: str | None = None,
        alpha: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Vertical bars centred on ``x``. ``width`` may be per-bar."""

    @abstractmethod
    def polygon(
        self,
        x,
        y,
        *,
        facecolor: str | None = None,
        edgecolor: str | None = None,
        linewidth: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A closed filled shape. Ellipses and boxes are built on this."""

    @abstractmethod
    def hlines(
        self,
        y,
        xmin,
        xmax,
        *,
        color: str | None = None,
        style: str = "-",
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Horizontal line segments. ``y`` may be a scalar or an array."""

    @abstractmethod
    def vlines(
        self,
        x,
        ymin,
        ymax,
        *,
        color: str | None = None,
        style: str = "-",
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Vertical line segments. ``x`` may be a scalar or an array."""

    @abstractmethod
    def axhline(
        self,
        y: float,
        *,
        color: str | None = None,
        style: str = "-",
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A horizontal line spanning the full axes width."""

    @abstractmethod
    def axvline(
        self,
        x: float,
        *,
        color: str | None = None,
        style: str = "-",
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A vertical line spanning the full axes height."""

    @abstractmethod
    def heatmap(
        self,
        values,
        *,
        x=None,
        y=None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        origin: Origin = "lower",
        colorbar: bool = True,
        colorbar_label: str | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A 2-D array as colored cells.

        ``x``/``y`` are cell centres; when omitted, integer indices are used.
        ``origin="upper"`` puts ``values[0]`` in the top row, which is the
        convention for correlation matrices.
        """

    @abstractmethod
    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        coords: Coords | tuple[Coords, Coords] = "data",
        ha: str = "left",
        va: str = "bottom",
        fontsize: float | None = None,
        color: str | None = None,
        alpha: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Text at ``(x, y)``, in data or axes-fraction coordinates.

        ``coords`` may be a ``(xcoords, ycoords)`` pair to mix the two, e.g.
        ``("data", "axes")`` for a label pinned below a particular x position.
        """

    def contour(
        self,
        x,
        y,
        z,
        *,
        levels: Sequence[float],
        labels: bool = False,
        colors: str | Sequence[str] | None = None,
        filled: bool = False,
        linewidths: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Contour lines of ``z`` over the ``(x, y)`` grid.

        Optional: backends may raise :class:`UnsupportedFeature`. Check with
        ``axes.supports("contour")``.
        """
        raise UnsupportedFeature(f"{self.backend} backend does not implement contour()")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @abstractmethod
    def set(
        self,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xscale: str | None = None,
        yscale: str | None = None,
    ) -> None:
        """Set axis labels, title, limits and scales. ``None`` leaves unchanged."""

    @abstractmethod
    def ticks(
        self,
        axis: Literal["x", "y"],
        positions: Sequence[float],
        labels: Sequence[str] | None = None,
        *,
        rotation: float = 0.0,
        fontsize: float | None = None,
    ) -> None:
        """Place explicit ticks, optionally with text labels."""

    @abstractmethod
    def grid(
        self,
        show: bool = True,
        *,
        axis: Literal["x", "y", "both"] = "both",
        alpha: float | None = None,
    ) -> None:
        """Toggle grid lines."""

    @abstractmethod
    def legend(
        self,
        show: bool = True,
        *,
        loc: str | None = None,
        ncol: int = 1,
        frameon: bool = False,
        fontsize: float | None = None,
    ) -> None:
        """Show a legend built from the labels passed to marks.

        Only marks given a ``label`` appear. On plotly the legend is a
        figure-level object, so ``loc`` positions it relative to the whole
        figure rather than this Axes.
        """

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def supports(self, feature: str) -> bool:
        """Whether this backend provides ``feature`` (see :data:`FEATURES`)."""
        if feature not in FEATURES:
            raise ValueError(f"Unknown feature {feature!r}. Known: {sorted(FEATURES)}")
        return feature in self.supported

    @abstractmethod
    def point_size(
        self,
        points: float,
        axis: Literal["x", "y"] = "y",
        *,
        units: Coords = "data",
    ) -> float:
        """Size of ``points`` typographic points along ``axis``.

        Returned in data units (``units="data"``) or as a fraction of the axes
        (``units="axes"``), signed to follow the sign of ``points``. This is the
        one genuinely leaky part of the
        abstraction: exact on matplotlib via the data transform, estimated on
        plotly from the axis range and subplot height, and ``0.0`` in data units
        when the limits have not been set -- so set limits first. Check
        ``axes.supports("pixel_metrics")`` if the difference matters.
        """

    def marker_extent(self, markersize: float) -> float:
        """Vertical size of a ``markersize``-point marker, in y data units."""
        return abs(self.point_size(markersize, "y", units="data"))


class Figure(ABC):
    """A grid of :class:`Axes` plus output."""

    #: Backend name.
    backend: str = ""

    @property
    @abstractmethod
    def native(self) -> Any:
        """The underlying ``plt.Figure`` or ``go.Figure``."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """``(nrows, ncols)``."""

    @abstractmethod
    def axes(self, row: int = 0, col: int = 0) -> Axes:
        """The Axes at ``(row, col)``, zero-indexed."""

    @property
    def flat(self) -> list[Axes]:
        """All Axes in row-major order."""
        nrows, ncols = self.shape
        return [self.axes(r, c) for r in range(nrows) for c in range(ncols)]

    @abstractmethod
    def hide(self, row: int, col: int) -> None:
        """Hide an unused cell of the grid."""

    @abstractmethod
    def suptitle(self, text: str, **kwargs: Any) -> None:
        """Set an overall figure title."""

    @abstractmethod
    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        """Write the figure out.

        Both backends write ``.png``/``.pdf``/``.svg``; plotly additionally
        writes ``.html`` and requires ``kaleido`` for the raster formats.
        """

    @abstractmethod
    def embed_html(self, *, dpi: float | None = None) -> str:
        """An HTML fragment for this figure.

        A base64 ``<img>`` on matplotlib, an interactive ``<div>`` on plotly.
        This is what lets report generators stay backend-agnostic.
        """

    @abstractmethod
    def show(self) -> None:
        """Display the figure."""

    @abstractmethod
    def close(self) -> None:
        """Release the figure's resources."""
