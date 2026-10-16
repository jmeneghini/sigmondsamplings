"""matplotlib backend."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from .base import Axes, Coords, Figure, FigureSpec
from .style import PlotStyle, get_style

__all__ = ["MatplotlibAxes", "MatplotlibFigure", "make_figure", "wrap"]


def _merge(base: dict[str, Any], native: dict[str, Any] | None) -> dict[str, Any]:
    """Drop None-valued entries, then let ``native`` override what remains."""
    out = {k: v for k, v in base.items() if v is not None}
    if native:
        out.update(native)
    return out


class MatplotlibAxes(Axes):
    """An :class:`~slat.plotting.base.Axes` backed by ``matplotlib.axes.Axes``."""

    backend = "matplotlib"

    def __init__(self, ax, figure: MatplotlibFigure):
        self._ax = ax
        self._figure = figure

    @property
    def figure(self) -> MatplotlibFigure:
        return self._figure

    @property
    def native(self):
        return self._ax

    # ------------------------------------------------------------------
    # Marks (hover is plotly-only; matplotlib ignores it)
    # ------------------------------------------------------------------

    def _line(
        self, x, y, *, color, linewidth, linestyle, marker, markersize, label, alpha, zorder,
        hover, fill, native,
    ) -> None:
        kwargs = _merge(
            {
                "color": color,
                "linewidth": linewidth,
                "linestyle": linestyle,
                "marker": marker,
                "markersize": markersize,
                "markerfacecolor": None if fill else "none",
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.plot(x, y, **kwargs)

    def _points(
        self, x, y, *, color, markersize, marker, label, alpha, zorder, hover, fill, native
    ) -> None:
        # `markersize` is a diameter in points; scatter wants an area. With
        # facecolors="none", scatter takes `color` as the edge color only.
        kwargs = _merge(
            {
                "color": color,
                "facecolors": None if fill else "none",
                "s": None if markersize is None else markersize**2,
                "marker": marker,
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.scatter(x, y, **kwargs)

    def _errorbar(
        self, x, y, *, yerr, xerr, color, marker, markersize, capsize, linewidth, linestyle,
        label, alpha, zorder, hover, fill, native,
    ) -> None:
        kwargs = _merge(
            {
                "yerr": yerr,
                "xerr": xerr,
                "color": color,
                "marker": marker,
                "markersize": markersize,
                "markerfacecolor": None if fill else "none",
                "capsize": capsize,
                "elinewidth": linewidth,
                # linestyle=None means markers only, matching the fmt="o" idiom.
                "linestyle": linestyle if linestyle is not None else "none",
                "label": label,
                "alpha": alpha,
                "zorder": zorder,
            },
            native,
        )
        self._ax.errorbar(x, y, **kwargs)

    def _band(self, x, lo, hi, *, color, alpha, label, zorder, native) -> None:
        kwargs = _merge(
            {"color": color, "alpha": alpha, "label": label, "zorder": zorder},
            native,
        )
        self._ax.fill_between(x, lo, hi, **kwargs)

    def _bar(self, x, height, *, width, color, label, alpha, hover, native) -> None:
        kwargs = _merge(
            {"width": width, "color": color, "label": label, "alpha": alpha},
            native,
        )
        self._ax.bar(x, height, **kwargs)

    def _polygon(
        self, x, y, *, facecolor, edgecolor, linewidth, alpha, label, zorder, native
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

    @staticmethod
    def _collection_kwargs(color, linestyle, linewidth, alpha, label, zorder, native):
        """hlines/vlines draw a LineCollection, whose properties are plural."""
        return _merge(
            {
                "colors": color,
                "linestyles": linestyle,
                "linewidths": linewidth,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )

    def _hlines(
        self, y, xmin, xmax, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        kwargs = self._collection_kwargs(color, linestyle, linewidth, alpha, label, zorder, native)
        self._ax.hlines(y, xmin, xmax, **kwargs)

    def _vlines(
        self, x, ymin, ymax, *, color, linestyle, linewidth, alpha, label, zorder, native
    ) -> None:
        kwargs = self._collection_kwargs(color, linestyle, linewidth, alpha, label, zorder, native)
        self._ax.vlines(x, ymin, ymax, **kwargs)

    def _axhline(
        self, y, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        kwargs = _merge(
            {
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.axhline(y, **kwargs)

    def _axvline(
        self, x, *, ymin, ymax, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        kwargs = _merge(
            {
                "ymin": ymin,
                "ymax": ymax,
                "clip_on": not (ymin < 0.0 or ymax > 1.0),
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
            },
            native,
        )
        self._ax.axvline(x, **kwargs)

    def _heatmap(
        self, values, *, x, y, cmap, vmin, vmax, origin, colorbar, colorbar_label, hover, native
    ) -> None:
        # NaN cells are masked by imshow/pcolormesh and drawn in the colormap's
        # "bad" color, transparent by default.
        values = np.asarray(values)
        kwargs = _merge({"cmap": cmap, "vmin": vmin, "vmax": vmax}, native)

        if x is None and y is None:
            mappable = self._ax.imshow(values, origin=origin, aspect="auto", **kwargs)
        else:
            nrows, ncols = values.shape
            xs = np.arange(ncols) if x is None else np.asarray(x)
            ys = np.arange(nrows) if y is None else np.asarray(y)
            mappable = self._ax.pcolormesh(xs, ys, values, shading="auto", **kwargs)
            if origin == "upper" and not self._ax.yaxis_inverted():
                # Flip the axis, not the values, so values[i] stays at y[i].
                self._ax.invert_yaxis()

        if colorbar:
            self._figure.native.colorbar(mappable, ax=self._ax, label=colorbar_label)

    def _text(self, x, y, s, *, coords, ha, va, fontsize, color, alpha, native) -> None:
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

    def _contour(self, x, y, z, *, levels, labels, colors, filled, linewidth, native) -> None:
        kwargs = _merge({"levels": list(levels), "colors": colors}, native)
        if filled:
            self._ax.contourf(x, y, z, **kwargs)
            return
        cs = self._ax.contour(x, y, z, linewidths=linewidth, **kwargs)
        if labels:
            self._ax.clabel(cs, inline=True, fontsize=self.style.tick_fontsize or 8)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _limits(self, axis, explicit_only):
        if explicit_only and getattr(self._ax, f"get_autoscale{axis}_on")():
            return None
        return tuple(getattr(self._ax, f"get_{axis}lim")())

    def _set(self, *, xlabel, ylabel, title, xlim, ylim, xscale, yscale, label_fontsize) -> None:
        fs = label_fontsize if label_fontsize is not None else self.style.label_fontsize
        if xlabel is not None:
            self._ax.set_xlabel(xlabel, fontsize=fs)
        if ylabel is not None:
            self._ax.set_ylabel(ylabel, fontsize=fs)
        if title is not None:
            self._ax.set_title(title, fontsize=self.style.title_fontsize)
        if xlim is not None:
            self._ax.set_xlim(*xlim)
        if ylim is not None:
            self._ax.set_ylim(*ylim)
        if xscale is not None:
            self._ax.set_xscale(xscale)
        if yscale is not None:
            self._ax.set_yscale(yscale)

    def _ticks(self, axis, positions, labels, *, rotation, fontsize) -> None:
        fontsize = fontsize if fontsize is not None else self.style.tick_fontsize
        setter = self._ax.set_xticks if axis == "x" else self._ax.set_yticks
        label_setter = self._ax.set_xticklabels if axis == "x" else self._ax.set_yticklabels
        setter(list(positions))
        if labels is not None:
            kwargs: dict[str, Any] = {"rotation": rotation, "fontsize": fontsize}
            if axis == "x" and rotation:
                kwargs["ha"] = "right"
            label_setter(list(labels), **{k: v for k, v in kwargs.items() if v is not None})

    def _grid(self, show, *, axis, alpha) -> None:
        if not show:
            # Any line property passed with False turns the grid back on.
            self._ax.grid(False, axis=axis)
            return
        alpha = alpha if alpha is not None else self.style.grid_alpha
        self._ax.grid(True, axis=axis, alpha=alpha)

    def _legend(self, show, *, loc, ncol, frameon, fontsize) -> None:
        if not show:
            legend = self._ax.get_legend()
            if legend is not None:
                legend.remove()
            return
        # Matching matplotlib's own behaviour: no labelled artists, no legend.
        _, labels = self._ax.get_legend_handles_labels()
        if not labels:
            return
        fontsize = fontsize if fontsize is not None else self.style.legend_fontsize
        kwargs = {"loc": loc, "ncol": ncol, "frameon": frameon, "fontsize": fontsize}
        self._ax.legend(**{k: v for k, v in kwargs.items() if v is not None})

    def point_size(self, points: float, axis="y", *, units: Coords = "data") -> float:
        """Exact: points -> pixels -> data units (or axes fraction) via the transform."""
        pixels = points * self._figure.native.dpi / 72.0
        if units == "axes":
            bbox = self._ax.get_window_extent()
            extent = bbox.width if axis == "x" else bbox.height
            return pixels / extent if extent else 0.0
        bbox = self._ax.get_window_extent()
        centre = np.array([bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2])
        displaced = centre.copy()
        index = 0 if axis == "x" else 1
        displaced[index] += abs(pixels)
        inverse = self._ax.transData.inverted()
        offset = inverse.transform(displaced)[index] - inverse.transform(centre)[index]
        return float(np.copysign(abs(offset), points))


class MatplotlibFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` backed by ``matplotlib.figure.Figure``."""

    backend = "matplotlib"

    def __init__(self, fig, axes_grid, style: PlotStyle):
        self._fig = fig
        self._style = style
        self._axes = [[MatplotlibAxes(ax, self) for ax in row] for row in axes_grid]

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

    def suptitle(self, text: str, *, fontsize: float | None = None) -> None:
        fontsize = fontsize if fontsize is not None else self.style.title_fontsize
        self._fig.suptitle(text, **({} if fontsize is None else {"fontsize": fontsize}))

    def set_size(self, width: float, height: float) -> None:
        self._fig.set_size_inches(width, height)
        self._style = self._style.replace(figsize=(width, height))

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


def make_figure(spec: FigureSpec) -> MatplotlibFigure:
    """Create a matplotlib figure with a ``spec.rows`` x ``spec.cols`` grid of axes."""
    import matplotlib.pyplot as plt

    style = spec.style
    fig, axes_array = plt.subplots(
        spec.rows,
        spec.cols,
        figsize=style.figsize,
        dpi=style.dpi,
        squeeze=False,
        sharex=spec.sharex,
        sharey=spec.sharey,
        gridspec_kw={
            k: v
            for k, v in (("height_ratios", spec.height_ratios), ("width_ratios", spec.width_ratios))
            if v is not None
        },
        constrained_layout=spec.constrained_layout,
    )
    if style.grid:
        for ax in axes_array.flat:
            ax.grid(True, alpha=style.grid_alpha)
    return MatplotlibFigure(fig, axes_array, style)


def wrap(obj, *, style: PlotStyle | None = None) -> MatplotlibAxes | MatplotlibFigure:
    """Adopt an existing ``plt.Axes`` or ``plt.Figure``.

    Callers holding a raw matplotlib Axes can pass it as ``axes=`` to any plot
    function.
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
