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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from .style import PlotStyle, get_style

__all__ = [
    "Axes",
    "Coords",
    "Figure",
    "FigureSpec",
    "Origin",
    "UnsupportedFeature",
]

Coords = Literal["data", "axes"]
Origin = Literal["upper", "lower"]


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

    The public methods carry the signatures, defaults and documentation. Each
    calls a protected hook of the same name (``line`` -> ``_line``) with every
    argument passed by keyword, so a backend implements only the hooks.
    Line widths and marker sizes are in points; ``None`` takes matplotlib's
    default on every backend.
    """

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

    @property
    def style(self) -> PlotStyle:
        """The style of this Axes' figure.

        Frontends drawing on a given Axes read their defaults (palette, grid)
        here rather than from the process-wide :func:`get_style`.
        """
        return self.figure.style

    @contextmanager
    def legend_group(self) -> Iterator[None]:
        """Draw a composed item with one interactive legend toggle.

        Give its first visible trace mark a label. Plotly groups trace marks
        in this scope, with a fresh identity local to this Axes; nested scopes
        create separate items. Axis-line shapes keep their own legend items
        (including their hover traces). Static and recording backends keep
        their usual legends.
        """
        with self._legend_group():
            yield

    @contextmanager
    def _legend_group(self) -> Iterator[None]:
        yield

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    def line(
        self,
        x,
        y,
        *,
        color: str | None = None,
        linewidth: float | None = None,
        linestyle: str = "-",
        marker: str | None = None,
        markersize: float | None = None,
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        fill: bool = True,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A connected line through ``(x, y)``.

        ``fill=False`` draws the markers (if any) hollow.
        """
        self._line(
            x,
            y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            label=label,
            alpha=alpha,
            zorder=zorder,
            hover=hover,
            fill=fill,
            native=native,
        )

    def points(
        self,
        x,
        y,
        *,
        color: str | None = None,
        markersize: float | None = None,
        marker: str = "o",
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        fill: bool = True,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Unconnected markers. ``markersize`` is a marker diameter in points.

        ``fill=False`` draws them hollow, outlined in ``color``.
        """
        self._points(
            x,
            y,
            color=color,
            markersize=markersize,
            marker=marker,
            label=label,
            alpha=alpha,
            zorder=zorder,
            hover=hover,
            fill=fill,
            native=native,
        )

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
        linewidth: float | None = None,
        linestyle: str | None = None,
        label: str | None = None,
        alpha: float | None = None,
        zorder: float | None = None,
        hover: Sequence[str] | None = None,
        fill: bool = True,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Points with error bars.

        ``linewidth`` is the width of the error bars. ``linestyle=None`` (the
        default) draws markers only, matching the ``fmt="o"`` idiom.
        ``yerr``/``xerr`` accept a scalar, a 1-D array, or a ``(2, N)`` array
        of asymmetric ``(lower, upper)`` offsets. ``fill=False`` draws the
        markers hollow.
        """
        self._errorbar(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            color=color,
            marker=marker,
            markersize=markersize,
            capsize=capsize,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            alpha=alpha,
            zorder=zorder,
            hover=hover,
            fill=fill,
            native=native,
        )

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
        self._band(x, lo, hi, color=color, alpha=alpha, label=label, zorder=zorder, native=native)

    def bar(
        self,
        x,
        height,
        *,
        width: float | Sequence[float] = 0.8,
        color: str | None = None,
        label: str | None = None,
        alpha: float | None = None,
        hover: Sequence[str] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Vertical bars centred on ``x``. ``width`` is in data units and may be per-bar."""
        self._bar(
            x,
            height,
            width=width,
            color=color,
            label=label,
            alpha=alpha,
            hover=hover,
            native=native,
        )

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
        self._polygon(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
        )

    def hlines(
        self,
        y,
        xmin,
        xmax,
        *,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        hover: str | Sequence[str] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Horizontal segments; hover is shared text or one string per segment.

        Static backends ignore hover text. ``y`` may be a scalar or an array.
        """
        self._hlines(
            y,
            xmin,
            xmax,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            hover=hover,
            native=native,
        )

    def vlines(
        self,
        x,
        ymin,
        ymax,
        *,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Vertical line segments. ``x`` may be a scalar or an array."""
        self._vlines(
            x,
            ymin,
            ymax,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
        )

    def axhline(
        self,
        y: float,
        *,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        hover: str | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A horizontal line spanning the full axes width. Static backends ignore hover."""
        self._axhline(
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            hover=hover,
            native=native,
        )

    def axvline(
        self,
        x: float,
        *,
        ymin: float = 0.0,
        ymax: float = 1.0,
        color: str | None = None,
        linestyle: str = "-",
        linewidth: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
        zorder: float | None = None,
        hover: str | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A vertical line with endpoints in axes fractions.

        Defaults span the full height. Endpoints outside [0, 1] extend beyond
        the axes, e.g. a group separator continuing below the x-axis. Static
        backends ignore hover.
        """
        self._axvline(
            x,
            ymin=ymin,
            ymax=ymax,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            hover=hover,
            native=native,
        )

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
        hover: Sequence[Sequence[str]] | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """A 2-D array as colored cells.

        ``x``/``y`` are cell centres; when omitted, integer indices are used.
        ``origin="upper"`` puts ``values[0]`` in the top row, which is the
        convention for correlation matrices. It does so by reversing the y
        axis, so a later ``set(ylim=(lo, hi))`` undoes it on every backend;
        pass ``ylim=(hi, lo)`` to keep row 0 at the top. ``hover`` is a 2-D array of
        strings shaped like ``values``, shown per cell on interactive backends.

        NaN cells render empty (no color) on every backend. Frontends rely on
        this to show missing entries, so it is part of the contract.
        """
        self._heatmap(
            values,
            x=x,
            y=y,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin=origin,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            hover=hover,
            native=native,
        )

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
        self._text(
            x,
            y,
            s,
            coords=coords,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            native=native,
        )

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
        linewidth: float | None = None,
        native: dict[str, Any] | None = None,
    ) -> None:
        """Contour lines of ``z`` over the ``(x, y)`` grid.

        Levels must be finite and strictly increasing. With ``filled=True``,
        adjacent levels bound filled bands; values outside them are unfilled.
        Labels apply to lines only.

        Optional: backends may raise :class:`UnsupportedFeature`.
        """
        levels = tuple(float(level) for level in levels)
        minimum = 2 if filled else 1
        if (len(levels) < minimum or not all(isfinite(level) for level in levels)
                or any(a >= b for a, b in zip(levels, levels[1:]))):
            raise ValueError(f"contour levels must contain at least {minimum} finite, increasing values")
        if colors is not None and not isinstance(colors, str):
            colors = tuple(colors)
            if not colors:
                raise ValueError("contour colors must be nonempty")
        self._contour(
            x,
            y,
            z,
            levels=levels,
            labels=labels,
            colors=colors,
            filled=filled,
            linewidth=linewidth,
            native=native,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_limits(
        self, axis: Literal["x", "y"], *, explicit_only: bool = False
    ) -> tuple[float, float] | None:
        """Current limits in data units, or None if unknown/automatic.

        With ``explicit_only``, return None when the axis uses automatic
        limits. This lets frontends respect ranges set by their caller.
        """
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        return self._limits(axis, explicit_only)

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
        label_fontsize: float | None = None,
    ) -> None:
        """Set axis labels, title, limits and scales. ``None`` leaves unchanged.

        ``label_fontsize`` overrides the style's axis-label size for this call.
        """
        self._set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xlim=xlim,
            ylim=ylim,
            xscale=xscale,
            yscale=yscale,
            label_fontsize=label_fontsize,
        )

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
        self._ticks(axis, positions, labels, rotation=rotation, fontsize=fontsize)

    def grid(
        self,
        show: bool = True,
        *,
        axis: Literal["x", "y", "both"] = "both",
        alpha: float | None = None,
    ) -> None:
        """Toggle grid lines. ``alpha`` defaults to the style's ``grid_alpha``."""
        self._grid(show, axis=axis, alpha=alpha)

    def legend(
        self,
        show: bool = True,
        *,
        loc: str | None = None,
        ncol: int = 1,
        frameon: bool = False,
        fontsize: float | None = None,
    ) -> None:
        """Show this Axes' legend, built from the labels passed to its marks.

        Only marks given a ``label`` appear, and no legend is shown until this
        is called. ``loc`` takes matplotlib's names; ``"best"`` (the default)
        is upper right on backends that cannot measure overlap.
        """
        self._legend(show, loc=loc, ncol=ncol, frameon=frameon, fontsize=fontsize)

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
        abstraction: measured at the axes midpoint via matplotlib's data transform,
        estimated on plotly from the inner plotting dimensions and axis range, and ``0.0`` in data units
        when the limits have not been set -- so set limits first. On nonlinear
        axes this is a local offset, not a uniform additive data distance.
        """

    def marker_extent(self, markersize: float) -> float:
        """Vertical size of a ``markersize``-point marker, in y data units."""
        return abs(self.point_size(markersize, "y", units="data"))

    # ------------------------------------------------------------------
    # Backend hooks: keyword-only, every argument resolved by the public
    # method above. Documented there.
    # ------------------------------------------------------------------

    @abstractmethod
    def _line(
        self, x, y, *, color, linewidth, linestyle, marker, markersize, label, alpha, zorder,
        hover, fill, native,
    ) -> None: ...

    @abstractmethod
    def _points(
        self, x, y, *, color, markersize, marker, label, alpha, zorder, hover, fill, native
    ) -> None: ...

    @abstractmethod
    def _errorbar(
        self, x, y, *, yerr, xerr, color, marker, markersize, capsize, linewidth, linestyle,
        label, alpha, zorder, hover, fill, native,
    ) -> None: ...

    @abstractmethod
    def _band(self, x, lo, hi, *, color, alpha, label, zorder, native) -> None: ...

    @abstractmethod
    def _bar(self, x, height, *, width, color, label, alpha, hover, native) -> None: ...

    @abstractmethod
    def _polygon(
        self, x, y, *, facecolor, edgecolor, linewidth, alpha, label, zorder, native
    ) -> None: ...

    @abstractmethod
    def _hlines(
        self, y, xmin, xmax, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None: ...

    @abstractmethod
    def _vlines(
        self, x, ymin, ymax, *, color, linestyle, linewidth, alpha, label, zorder, native
    ) -> None: ...

    @abstractmethod
    def _axhline(
        self, y, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None: ...

    @abstractmethod
    def _axvline(
        self, x, *, ymin, ymax, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None: ...

    @abstractmethod
    def _heatmap(
        self, values, *, x, y, cmap, vmin, vmax, origin, colorbar, colorbar_label, hover, native
    ) -> None: ...

    @abstractmethod
    def _text(self, x, y, s, *, coords, ha, va, fontsize, color, alpha, native) -> None: ...

    def _contour(self, x, y, z, *, levels, labels, colors, filled, linewidth, native) -> None:
        raise UnsupportedFeature(f"{self.backend} backend does not implement contour()")

    def _limits(
        self, axis: Literal["x", "y"], explicit_only: bool
    ) -> tuple[float, float] | None:
        return None

    @abstractmethod
    def _set(
        self, *, xlabel, ylabel, title, xlim, ylim, xscale, yscale, label_fontsize
    ) -> None: ...

    @abstractmethod
    def _ticks(self, axis, positions, labels, *, rotation, fontsize) -> None: ...

    @abstractmethod
    def _grid(self, show, *, axis, alpha) -> None: ...

    @abstractmethod
    def _legend(self, show, *, loc, ncol, frameon, fontsize) -> None: ...


@dataclass(frozen=True)
class FigureSpec:
    """What :func:`slat.plotting.figure` asks a backend's ``make_figure`` for.

    ``style`` is already resolved, with any requested figure size applied to
    ``style.figsize``. ``constrained_layout`` is matplotlib's layout engine;
    other backends ignore it.
    """

    rows: int = 1
    cols: int = 1
    style: PlotStyle = field(default_factory=get_style)
    height_ratios: tuple[float, ...] | None = None
    width_ratios: tuple[float, ...] | None = None
    sharex: bool = False
    sharey: bool = False
    constrained_layout: bool = True


class Figure(ABC):
    """A grid of :class:`Axes` plus output.

    Backends keep the figure's :class:`PlotStyle` as ``_style``.
    """

    #: Backend name.
    backend: str = ""

    _style: PlotStyle

    @property
    def style(self) -> PlotStyle:
        """The style this figure was made or wrapped with."""
        return self._style

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
    def suptitle(self, text: str, *, fontsize: float | None = None) -> None:
        """Set an overall figure title. ``fontsize`` is in points."""

    @abstractmethod
    def set_size(self, width: float, height: float) -> None:
        """Resize the figure to ``width`` x ``height`` inches, after drawing if need be."""

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
