"""matplotlib backend."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import FEATURES, Axes, Coords, Figure, Origin
from .style import PlotStyle, get_style

__all__ = ["MatplotlibAxes", "MatplotlibFigure", "make_figure", "wrap"]

# matplotlib does everything.
_SUPPORTED = FEATURES


def _merge(base: dict[str, Any], native: dict[str, Any] | None) -> dict[str, Any]:
    """Drop None-valued entries, then let ``native`` override what remains."""
    out = {k: v for k, v in base.items() if v is not None}
    if native:
        out.update(native)
    return out


class MatplotlibAxes(Axes):
    """An :class:`~slat.plotting.base.Axes` backed by ``matplotlib.axes.Axes``."""

    backend = "matplotlib"
    supported = _SUPPORTED

    def __init__(self, ax, figure: MatplotlibFigure, style: PlotStyle):
        self._ax = ax
        self._figure = figure
        self._style = style

    @property
    def figure(self) -> MatplotlibFigure:
        return self._figure

    @property
    def native(self):
        return self._ax

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    def line(
        self,
        x,
        y,
        *,
        color=None,
        width=None,
        style="-",
        marker=None,
        markersize=None,
        label=None,
        alpha=None,
        zorder=None,
        hover=None,
        native=None,
    ) -> None:
        # hover is plotly-only; matplotlib silently ignores it.
        kwargs = _merge(
            {
                "color": color,
                "linewidth": width,
                "linestyle": style,
                "marker": marker,
                "markersize": markersize,
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.plot(x, y, **kwargs)

    def points(
        self,
        x,
        y,
        *,
        color=None,
        size=None,
        marker="o",
        label=None,
        alpha=None,
        zorder=None,
        hover=None,
        native=None,
    ) -> None:
        # `size` is a diameter in points; scatter wants an area.
        kwargs = _merge(
            {
                "color": color,
                "s": None if size is None else size**2,
                "marker": marker,
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.scatter(x, y, **kwargs)

    def errorbar(
        self,
        x,
        y,
        *,
        yerr=None,
        xerr=None,
        color=None,
        marker="o",
        markersize=None,
        capsize=None,
        width=None,
        style=None,
        label=None,
        alpha=None,
        zorder=None,
        hover=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "yerr": yerr,
                "xerr": xerr,
                "color": color,
                "marker": marker,
                "markersize": markersize,
                "capsize": capsize,
                "elinewidth": width,
                # style=None means markers only, matching the fmt="o" idiom.
                "linestyle": style if style is not None else "none",
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.errorbar(x, y, **kwargs)

    def band(self, x, lo, hi, *, color=None, alpha=0.25, label=None, zorder=None, native=None):
        kwargs = _merge(
            {"color": color, "alpha": alpha, "label": label, "zorder": zorder},
            native,
        )
        self._ax.fill_between(x, lo, hi, **kwargs)

    def bar(self, x, height, *, width=0.8, color=None, label=None, alpha=None, native=None):
        kwargs = _merge(
            {"width": width, "color": color, "label": label, "alpha": alpha},
            native,
        )
        self._ax.bar(x, height, **kwargs)

    def polygon(
        self,
        x,
        y,
        *,
        facecolor=None,
        edgecolor=None,
        linewidth=None,
        alpha=None,
        label=None,
        zorder=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.fill(x, y, **kwargs)

    def hlines(
        self,
        y,
        xmin,
        xmax,
        *,
        color=None,
        style="-",
        width=None,
        alpha=None,
        label=None,
        zorder=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "colors": color,
                "linestyles": style,
                "linewidths": width,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.hlines(y, xmin, xmax, **kwargs)

    def vlines(
        self,
        x,
        ymin,
        ymax,
        *,
        color=None,
        style="-",
        width=None,
        alpha=None,
        label=None,
        zorder=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "colors": color,
                "linestyles": style,
                "linewidths": width,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.vlines(x, ymin, ymax, **kwargs)

    def axhline(
        self,
        y,
        *,
        color=None,
        style="-",
        width=None,
        alpha=None,
        label=None,
        zorder=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "color": color,
                "linestyle": style,
                "linewidth": width,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.axhline(y, **kwargs)

    def axvline(
        self,
        x,
        *,
        color=None,
        style="-",
        width=None,
        alpha=None,
        label=None,
        zorder=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "color": color,
                "linestyle": style,
                "linewidth": width,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.axvline(x, **kwargs)

    def heatmap(
        self,
        values,
        *,
        x=None,
        y=None,
        cmap="viridis",
        vmin=None,
        vmax=None,
        origin: Origin = "lower",
        colorbar=True,
        colorbar_label=None,
        native=None,
    ) -> None:
        values = np.asarray(values)
        kwargs = _merge({"cmap": cmap, "vmin": vmin, "vmax": vmax}, native)

        if x is None and y is None:
            mappable = self._ax.imshow(values, origin=origin, aspect="auto", **kwargs)
        else:
            nrows, ncols = values.shape
            xs = np.arange(ncols) if x is None else np.asarray(x)
            ys = np.arange(nrows) if y is None else np.asarray(y)
            grid = values if origin == "lower" else values[::-1]
            mappable = self._ax.pcolormesh(xs, ys, grid, shading="auto", **kwargs)

        if colorbar:
            self._figure.native.colorbar(mappable, ax=self._ax, label=colorbar_label)

    def text(
        self,
        x,
        y,
        s,
        *,
        coords: Coords | tuple[Coords, Coords] = "data",
        ha="left",
        va="bottom",
        fontsize=None,
        color=None,
        alpha=None,
        native=None,
    ) -> None:
        kwargs = _merge(
            {
                "ha": ha,
                "va": va,
                "fontsize": fontsize,
                "color": color,
                "alpha": alpha,
                "transform": self._transform(coords),
                "clip_on": False,
            },
            native,
        )
        self._ax.text(x, y, s, **kwargs)

    def _transform(self, coords: Coords | tuple[Coords, Coords]):
        xc, yc = (coords, coords) if isinstance(coords, str) else coords
        if xc == yc:
            return self._ax.transAxes if xc == "axes" else self._ax.transData
        from matplotlib.transforms import blended_transform_factory

        return blended_transform_factory(
            self._ax.transAxes if xc == "axes" else self._ax.transData,
            self._ax.transAxes if yc == "axes" else self._ax.transData,
        )

    def contour(
        self,
        x,
        y,
        z,
        *,
        levels,
        labels=False,
        colors=None,
        filled=False,
        linewidths=None,
        native=None,
    ) -> None:
        kwargs = _merge({"levels": list(levels), "colors": colors}, native)
        if filled:
            self._ax.contourf(x, y, z, **kwargs)
            return
        cs = self._ax.contour(x, y, z, linewidths=linewidths, **kwargs)
        if labels:
            self._ax.clabel(cs, inline=True, fontsize=self._style.tick_fontsize or 8)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set(
        self,
        *,
        xlabel=None,
        ylabel=None,
        title=None,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
    ) -> None:
        fs = self._style.label_fontsize
        if xlabel is not None:
            self._ax.set_xlabel(xlabel, fontsize=fs)
        if ylabel is not None:
            self._ax.set_ylabel(ylabel, fontsize=fs)
        if title is not None:
            self._ax.set_title(title, fontsize=self._style.title_fontsize)
        if xlim is not None:
            self._ax.set_xlim(*xlim)
        if ylim is not None:
            self._ax.set_ylim(*ylim)
        if xscale is not None:
            self._ax.set_xscale(xscale)
        if yscale is not None:
            self._ax.set_yscale(yscale)

    def ticks(
        self,
        axis: Literal["x", "y"],
        positions: Sequence[float],
        labels: Sequence[str] | None = None,
        *,
        rotation: float = 0.0,
        fontsize: float | None = None,
    ) -> None:
        fontsize = fontsize if fontsize is not None else self._style.tick_fontsize
        setter = self._ax.set_xticks if axis == "x" else self._ax.set_yticks
        label_setter = self._ax.set_xticklabels if axis == "x" else self._ax.set_yticklabels
        setter(list(positions))
        if labels is not None:
            kwargs: dict[str, Any] = {"rotation": rotation, "fontsize": fontsize}
            if axis == "x" and rotation:
                kwargs["ha"] = "right"
            label_setter(list(labels), **{k: v for k, v in kwargs.items() if v is not None})

    def grid(self, show=True, *, axis="both", alpha=None) -> None:
        alpha = alpha if alpha is not None else self._style.grid_alpha
        self._ax.grid(show, axis=axis, alpha=alpha)

    def legend(self, show=True, *, loc=None, ncol=1, frameon=False, fontsize=None) -> None:
        if not show:
            legend = self._ax.get_legend()
            if legend is not None:
                legend.remove()
            return
        # Matching matplotlib's own behaviour: no labelled artists, no legend.
        _, labels = self._ax.get_legend_handles_labels()
        if not labels:
            return
        fontsize = fontsize if fontsize is not None else self._style.legend_fontsize
        kwargs = _merge({"loc": loc, "ncol": ncol, "frameon": frameon, "fontsize": fontsize}, None)
        self._ax.legend(**kwargs)

    def point_size(self, points: float, axis="y", *, units: Coords = "data") -> float:
        """Exact: points -> pixels -> data units (or axes fraction) via the transform."""
        pixels = points * self._figure.native.dpi / 72.0
        if units == "axes":
            bbox = self._ax.get_window_extent()
            extent = bbox.width if axis == "x" else bbox.height
            return pixels / extent if extent else 0.0
        inverse = self._ax.transData.inverted()
        origin = inverse.transform((0, 0))
        if axis == "x":
            return inverse.transform((pixels, 0))[0] - origin[0]
        return inverse.transform((0, pixels))[1] - origin[1]


class MatplotlibFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` backed by ``matplotlib.figure.Figure``."""

    backend = "matplotlib"

    def __init__(self, fig, axes_grid, style: PlotStyle):
        self._fig = fig
        self._axes = [[MatplotlibAxes(ax, self, style) for ax in row] for row in axes_grid]
        self._style = style

    @property
    def native(self):
        return self._fig

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._axes), len(self._axes[0]))

    def axes(self, row: int = 0, col: int = 0) -> MatplotlibAxes:
        return self._axes[row][col]

    def hide(self, row: int, col: int) -> None:
        self._axes[row][col].native.set_visible(False)

    def suptitle(self, text: str, **kwargs: Any) -> None:
        self._fig.suptitle(text, **kwargs)

    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        self._fig.savefig(str(path), dpi=dpi or self._style.dpi, bbox_inches="tight")

    def embed_html(self, *, dpi: float | None = None) -> str:
        buffer = io.BytesIO()
        self._fig.savefig(buffer, format="png", dpi=dpi or self._style.dpi, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;">'

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)


def make_figure(
    rows: int = 1,
    cols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    height_ratios: Sequence[float] | None = None,
    width_ratios: Sequence[float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    style: PlotStyle | None = None,
    constrained_layout: bool = True,
) -> MatplotlibFigure:
    """Create a matplotlib figure with a ``rows`` x ``cols`` grid of axes."""
    import matplotlib.pyplot as plt

    style = style or get_style()
    fig, axes_array = plt.subplots(
        rows,
        cols,
        figsize=figsize or style.figsize,
        dpi=style.dpi,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw={
            k: v
            for k, v in (("height_ratios", height_ratios), ("width_ratios", width_ratios))
            if v is not None
        },
        constrained_layout=constrained_layout,
    )
    return MatplotlibFigure(fig, axes_array, style)


def wrap(obj, *, style: PlotStyle | None = None) -> MatplotlibAxes | MatplotlibFigure:
    """Adopt an existing ``plt.Axes`` or ``plt.Figure``.

    This is what keeps the legacy ``ax=`` keyword working: callers holding a
    raw matplotlib Axes can hand it straight to the new API.
    """
    import matplotlib.axes
    import matplotlib.figure

    style = style or get_style()
    if isinstance(obj, matplotlib.axes.Axes):
        figure = MatplotlibFigure(obj.figure, [[obj]], style)
        return figure.axes(0, 0)
    if isinstance(obj, matplotlib.figure.Figure):
        axes_list = obj.get_axes() or [obj.add_subplot(1, 1, 1)]
        return MatplotlibFigure(obj, [axes_list], style)
    raise TypeError(f"Cannot wrap {type(obj).__name__} with the matplotlib backend")
