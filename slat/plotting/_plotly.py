"""plotly backend.

Differences from matplotlib that callers should know about:

* ``zorder`` is ignored -- plotly draws in trace-insertion order.
* The legend is a figure-level object, so ``Axes.legend(loc=...)`` positions it
  relative to the whole figure.
* :meth:`PlotlyAxes.marker_extent` is an estimate, and returns ``0.0`` until
  the y limits have been set.
* LaTeX in tick labels and legend entries is flattened to Unicode, because
  MathJax does not render reliably in those positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import Axes, Coords, Figure, Origin
from .style import PlotStyle, get_style
from .text import to_display

__all__ = ["PlotlyAxes", "PlotlyFigure", "make_figure", "wrap"]

# No pixel_metrics (estimated only) and no zorder (insertion-ordered).
_SUPPORTED = frozenset({"capsize", "contour", "hover", "mathtext", "raster_export"})

_MARKERS = {
    "o": "circle",
    "s": "square",
    "D": "diamond",
    "d": "diamond-tall",
    "v": "triangle-down",
    "^": "triangle-up",
    "<": "triangle-left",
    ">": "triangle-right",
    "*": "star",
    "x": "x-thin",
    "+": "cross-thin",
    ".": "circle",
    "p": "pentagon",
    "h": "hexagon",
}

_DASHES = {
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
    "none": "solid",
}

_COLORSCALES = {
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "magma": "Magma",
    "cividis": "Cividis",
    "rdbu": "RdBu",
    "rdbu_r": "RdBu_r",
    "coolwarm": "RdBu_r",
    "gray": "Greys",
    "grey": "Greys",
    "jet": "Jet",
}

# Enough of the CSS palette to cover what the frontends actually pass. Anything
# missing still renders (plotly knows CSS names) but cannot take an alpha.
_NAMED_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "lightgrey": (211, 211, 211),
    "lightgray": (211, 211, 211),
    "darkgrey": (169, 169, 169),
    "darkgray": (169, 169, 169),
}


def _to_rgb(color: str) -> tuple[int, int, int] | None:
    """Best-effort conversion of a matplotlib-style color to an RGB triple."""
    if not isinstance(color, str):
        return None
    text = color.strip()
    if text.startswith("#"):
        body = text[1:]
        if len(body) == 3:
            body = "".join(ch * 2 for ch in body)
        if len(body) == 6:
            return tuple(int(body[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        return None
    lowered = text.lower()
    if lowered in _NAMED_RGB:
        return _NAMED_RGB[lowered]
    try:
        # matplotlib spells greyscale as a float-valued string, e.g. "0.35".
        level = float(text)
    except ValueError:
        return None
    if 0.0 <= level <= 1.0:
        value = round(level * 255)
        return (value, value, value)
    return None


def _color(color: str | None, alpha: float | None = None) -> str | None:
    """Render a color for plotly, folding ``alpha`` into an rgba() string."""
    if color is None:
        return None
    if alpha is None or alpha >= 1.0:
        rgb = _to_rgb(color)
        return color if rgb is None else f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    rgb = _to_rgb(color)
    if rgb is None:
        # Unknown name: keep the color and lose the transparency.
        return color
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _marker(symbol: str | None) -> str | None:
    return None if symbol is None else _MARKERS.get(symbol, symbol)


def _dash(style: str | None) -> str | None:
    return None if style is None else _DASHES.get(style, style)


def _colorscale(cmap: str) -> str:
    return _COLORSCALES.get(cmap.lower(), cmap)


def _clean(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not None}


def _err(values, count: int) -> dict[str, Any] | None:
    """Translate a scalar / 1-D / (2, N) error spec into a plotly error dict."""
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return {"type": "data", "array": np.full(count, float(array)), "visible": True}
    if array.ndim == 2:
        return {
            "type": "data",
            "array": array[1],
            "arrayminus": array[0],
            "symmetric": False,
            "visible": True,
        }
    return {"type": "data", "array": array, "visible": True}


#: mpl legend ``loc`` strings mapped to plotly legend anchors.
_LEGEND_LOC = {
    "upper right": (0.99, 0.99, "right", "top"),
    "upper left": (0.01, 0.99, "left", "top"),
    "lower right": (0.99, 0.01, "right", "bottom"),
    "lower left": (0.01, 0.01, "left", "bottom"),
    "upper center": (0.5, 0.99, "center", "top"),
    "lower center": (0.5, 0.01, "center", "bottom"),
    "center right": (0.99, 0.5, "right", "middle"),
    "center left": (0.01, 0.5, "left", "middle"),
    "center": (0.5, 0.5, "center", "middle"),
    "best": (0.99, 0.99, "right", "top"),
}


class PlotlyAxes(Axes):
    """An :class:`~slat.plotting.base.Axes` backed by one plotly subplot."""

    backend = "plotly"
    supported = _SUPPORTED

    def __init__(self, figure: PlotlyFigure, row: int, col: int, style: PlotStyle):
        self._figure = figure
        # plotly's row/col are 1-indexed; ours are 0-indexed.
        self._row = row + 1
        self._col = col + 1
        self._style = style
        _, ncols = figure.shape
        index = row * ncols + col + 1
        self._suffix = "" if index == 1 else str(index)

    @property
    def figure(self) -> PlotlyFigure:
        return self._figure

    @property
    def native(self):
        """``(go.Figure, row, col)`` with plotly's 1-indexed row/col."""
        return (self._figure.native, self._row, self._col)

    @property
    def _fig(self):
        return self._figure.native

    @property
    def _xref(self) -> str:
        return f"x{self._suffix}"

    @property
    def _yref(self) -> str:
        return f"y{self._suffix}"

    @staticmethod
    def _hover(hover: Sequence[str] | None) -> dict[str, Any]:
        """Per-point hover text -- the main thing plotly buys over matplotlib."""
        if hover is None:
            return {}
        return {"hovertext": list(hover), "hoverinfo": "text"}

    def _add(self, trace, native: dict[str, Any] | None) -> None:
        if native:
            trace.update(**native)
        self._fig.add_trace(trace, row=self._row, col=self._col)

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
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines+markers" if marker else "lines",
            line=_clean({"color": _color(color, alpha), "width": width, "dash": _dash(style)}),
            marker=_clean({"symbol": _marker(marker), "size": markersize}),
            name=label,
            showlegend=label is not None,
            **self._hover(hover),
        )
        self._add(trace, native)

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
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=_clean({"symbol": _marker(marker), "size": size, "color": _color(color, alpha)}),
            name=label,
            showlegend=label is not None,
            **self._hover(hover),
        )
        self._add(trace, native)

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
        import plotly.graph_objects as go

        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        rendered = _color(color, alpha)

        error_y = _err(yerr, len(y))
        error_x = _err(xerr, len(x))
        for spec in (error_y, error_x):
            if spec is not None:
                spec.update(_clean({"color": rendered, "thickness": width, "width": capsize}))

        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers" if style is None else "lines+markers",
            marker=_clean({"symbol": _marker(marker), "size": markersize, "color": rendered}),
            line=_clean({"color": rendered, "dash": _dash(style)}),
            error_y=error_y,
            error_x=error_x,
            name=label,
            showlegend=label is not None,
            **self._hover(hover),
        )
        self._add(trace, native)

    def band(self, x, lo, hi, *, color=None, alpha=0.25, label=None, zorder=None, native=None):
        x = np.asarray(x)
        self._filled(
            np.concatenate([x, x[::-1]]),
            np.concatenate([np.asarray(hi), np.asarray(lo)[::-1]]),
            facecolor=color,
            alpha=alpha,
            label=label,
            native=native,
        )

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
        self._filled(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            native=native,
        )

    def _filled(
        self,
        x,
        y,
        *,
        facecolor=None,
        edgecolor=None,
        linewidth=None,
        alpha=None,
        label=None,
        native=None,
    ) -> None:
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=_color(facecolor, alpha),
            line=_clean(
                {
                    "color": _color(edgecolor) if edgecolor else "rgba(0,0,0,0)",
                    "width": linewidth if edgecolor else 0,
                }
            ),
            name=label,
            showlegend=label is not None,
            hoverinfo="skip",
        )
        self._add(trace, native)

    def bar(self, x, height, *, width=0.8, color=None, label=None, alpha=None, native=None):
        import plotly.graph_objects as go

        trace = go.Bar(
            x=x,
            y=height,
            width=width,
            marker=_clean({"color": _color(color, alpha)}),
            name=label,
            showlegend=label is not None,
        )
        self._add(trace, native)

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
        ys, x0s, x1s = np.broadcast_arrays(
            np.atleast_1d(y), np.atleast_1d(xmin), np.atleast_1d(xmax)
        )
        xs: list[float | None] = []
        vals: list[float | None] = []
        for yi, a, b in zip(ys, x0s, x1s):
            xs += [a, b, None]
            vals += [yi, yi, None]
        self._segments(xs, vals, color, style, width, alpha, label, native)

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
        xs_in, y0s, y1s = np.broadcast_arrays(
            np.atleast_1d(x), np.atleast_1d(ymin), np.atleast_1d(ymax)
        )
        xs: list[float | None] = []
        vals: list[float | None] = []
        for xi, a, b in zip(xs_in, y0s, y1s):
            xs += [xi, xi, None]
            vals += [a, b, None]
        self._segments(xs, vals, color, style, width, alpha, label, native)

    def _segments(self, xs, ys, color, style, width, alpha, label, native) -> None:
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            connectgaps=False,
            line=_clean({"color": _color(color, alpha), "width": width, "dash": _dash(style)}),
            name=label,
            showlegend=label is not None,
            hoverinfo="skip",
        )
        self._add(trace, native)

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
        self._axline("h", y, color, style, width, alpha, label, native)

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
        self._axline("v", x, color, style, width, alpha, label, native)

    def _axline(self, kind, value, color, style, width, alpha, label, native) -> None:
        line = _clean({"color": _color(color, alpha), "width": width, "dash": _dash(style)})
        kwargs: dict[str, Any] = {"line": line, "row": self._row, "col": self._col}
        if label is not None:
            kwargs["annotation_text"] = label
        if native:
            kwargs.update(native)
        if kind == "h":
            self._fig.add_hline(y=value, **kwargs)
        else:
            self._fig.add_vline(x=value, **kwargs)

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
        import plotly.graph_objects as go

        values = np.asarray(values)
        trace = go.Heatmap(
            z=values,
            x=x,
            y=y,
            colorscale=_colorscale(cmap),
            zmin=vmin,
            zmax=vmax,
            showscale=colorbar,
            colorbar=_clean({"title": colorbar_label}) if colorbar else None,
        )
        self._add(trace, native)
        if origin == "upper":
            # Match imshow: row 0 at the top.
            self._fig.update_yaxes(autorange="reversed", row=self._row, col=self._col)

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
        xc, yc = (coords, coords) if isinstance(coords, str) else coords
        kwargs = _clean(
            {
                "x": x,
                "y": y,
                "text": s,
                "xref": f"{self._xref}{' domain' if xc == 'axes' else ''}",
                "yref": f"{self._yref}{' domain' if yc == 'axes' else ''}",
                "showarrow": False,
                "xanchor": {"left": "left", "right": "right", "center": "center"}.get(ha, "left"),
                "yanchor": {"bottom": "bottom", "top": "top", "center": "middle"}.get(va, "bottom"),
                "font": _clean({"size": fontsize, "color": _color(color, alpha)}) or None,
            }
        )
        if native:
            kwargs.update(native)
        self._fig.add_annotation(**kwargs)
        if xc == "axes":
            self._reserve_margin(x, "x", s, fontsize)
        if yc == "axes":
            self._reserve_margin(y, "y", s, fontsize)

    def _reserve_margin(
        self, value: float, axis: Literal["x", "y"], text: str, fontsize: float | None
    ) -> None:
        """Grow the figure margin so an out-of-domain annotation stays visible.

        matplotlib's layout engine makes room for text placed outside the axes;
        plotly clips it against the paper instead. Spectrum plots put their
        outer-group labels below the axes, and right-edge labels sit past the
        right domain edge, so reserve the space on whichever side is used.

        The text allowance is an estimate -- measuring a string needs a renderer,
        which is exactly what is not available at build time.
        """
        if 0.0 <= value <= 1.0:
            return
        overshoot = -value if value < 0 else value - 1.0
        size = fontsize or self._style.label_fontsize or 12.0
        if axis == "x":
            extent = self._fig.layout.width or self._style.pixel_size[0]
            # Width scales with the string; 0.62 em per character is a rough mean.
            allowance = min(240.0, 8.0 + 0.62 * size * len(str(text)))
            side = "l" if value < 0 else "r"
        else:
            extent = self._fig.layout.height or self._style.pixel_size[1]
            allowance = 2.0 * size  # one line, plus breathing room
            side = "b" if value < 0 else "t"

        needed = round(overshoot * extent + allowance)
        if needed > (self._fig.layout.margin[side] or 0):
            self._fig.update_layout(margin={side: needed})

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
        import plotly.graph_objects as go

        color = colors if isinstance(colors, str) else None
        # plotly contours are uniformly spaced, so emit one trace per level.
        for i, level in enumerate(levels):
            level_color = color
            if color is None and isinstance(colors, Sequence) and not isinstance(colors, str):
                level_color = colors[i % len(colors)]
            trace = go.Contour(
                x=x,
                y=y,
                z=z,
                contours=_clean(
                    {
                        "start": level,
                        "end": level,
                        "size": 0,
                        "coloring": "fill" if filled else "lines",
                        "showlabels": labels,
                    }
                ),
                line=_clean({"color": _color(level_color), "width": linewidths}),
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            )
            self._add(trace, native)

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
        font = _clean({"size": self._style.label_fontsize})
        x_updates = _clean(
            {
                "title_text": xlabel,
                "range": list(xlim) if xlim else None,
                "type": _scale(xscale),
            }
        )
        y_updates = _clean(
            {
                "title_text": ylabel,
                "range": list(ylim) if ylim else None,
                "type": _scale(yscale),
            }
        )
        if xlabel is not None and font:
            x_updates["title_font"] = font
        if ylabel is not None and font:
            y_updates["title_font"] = font
        if x_updates:
            self._fig.update_xaxes(**x_updates, row=self._row, col=self._col)
        if y_updates:
            self._fig.update_yaxes(**y_updates, row=self._row, col=self._col)

        if title is not None:
            if self._figure.shape == (1, 1):
                self._fig.update_layout(
                    title=_clean(
                        {
                            "text": title,
                            "font": _clean({"size": self._style.title_fontsize}) or None,
                        }
                    )
                )
            else:
                # Per-subplot titles are annotations in plotly.
                self.text(
                    0.5,
                    1.02,
                    title,
                    coords="axes",
                    ha="center",
                    va="bottom",
                    fontsize=self._style.title_fontsize,
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
        fontsize = fontsize if fontsize is not None else self._style.tick_fontsize
        updates: dict[str, Any] = {"tickmode": "array", "tickvals": list(positions)}
        if labels is not None:
            # MathJax is unreliable in tick labels; fall back to Unicode.
            updates["ticktext"] = [to_display(label, mathtext=False) for label in labels]
        if rotation:
            # matplotlib rotates counter-clockwise, plotly clockwise.
            updates["tickangle"] = -rotation
        if fontsize is not None:
            updates["tickfont"] = {"size": fontsize}
        updater = self._fig.update_xaxes if axis == "x" else self._fig.update_yaxes
        updater(**updates, row=self._row, col=self._col)

    def grid(self, show=True, *, axis="both", alpha=None) -> None:
        alpha = alpha if alpha is not None else self._style.grid_alpha
        updates = {"showgrid": show, "gridcolor": _color("grey", alpha)}
        if axis in ("x", "both"):
            self._fig.update_xaxes(**updates, row=self._row, col=self._col)
        if axis in ("y", "both"):
            self._fig.update_yaxes(**updates, row=self._row, col=self._col)

    def legend(self, show=True, *, loc=None, ncol=1, frameon=False, fontsize=None) -> None:
        # plotly's legend is figure-level, so this configures the whole figure.
        if not show:
            self._fig.update_layout(showlegend=False)
            return
        fontsize = fontsize if fontsize is not None else self._style.legend_fontsize
        legend = _clean(
            {
                "font": _clean({"size": fontsize}) or None,
                "borderwidth": 1 if frameon else 0,
                # plotly has no column count; horizontal is the closest analogue.
                "orientation": "h" if ncol > 1 else "v",
            }
        )
        if loc in _LEGEND_LOC:
            x, y, xanchor, yanchor = _LEGEND_LOC[loc]
            legend.update({"x": x, "y": y, "xanchor": xanchor, "yanchor": yanchor})
        self._fig.update_layout(showlegend=True, legend=legend)

    def point_size(self, points: float, axis="y", *, units: Coords = "data") -> float:
        """Estimated from the axis range and the subplot's pixel extent.

        Returns ``0.0`` in data units when the range has not been set, since
        there is nothing to scale against before plotly lays the figure out.
        """
        target = self._fig.layout[f"{axis}axis{self._suffix}"]
        domain = target.domain or (0.0, 1.0)
        figure_px = self._style.pixel_size[0 if axis == "x" else 1]
        total = (self._fig.layout.width if axis == "x" else self._fig.layout.height) or figure_px
        extent_px = total * (domain[1] - domain[0])
        if extent_px <= 0:
            return 0.0
        pixels = points * self._style.dpi / 72.0
        if units == "axes":
            return pixels / extent_px
        if target.range is None:
            return 0.0
        span = abs(target.range[1] - target.range[0])
        return span * pixels / extent_px


def _scale(scale: str | None) -> str | None:
    if scale is None:
        return None
    return {"log": "log", "linear": "linear"}.get(scale, scale)


class PlotlyFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` backed by ``plotly.graph_objects.Figure``."""

    backend = "plotly"

    def __init__(self, fig, rows: int, cols: int, style: PlotStyle):
        self._fig = fig
        self._shape = (rows, cols)
        self._style = style
        self._axes = [[PlotlyAxes(self, r, c, style) for c in range(cols)] for r in range(rows)]

    @property
    def native(self):
        return self._fig

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def axes(self, row: int = 0, col: int = 0) -> PlotlyAxes:
        return self._axes[row][col]

    def hide(self, row: int, col: int) -> None:
        self._fig.update_xaxes(visible=False, row=row + 1, col=col + 1)
        self._fig.update_yaxes(visible=False, row=row + 1, col=col + 1)

    def suptitle(self, text: str, **kwargs: Any) -> None:
        self._fig.update_layout(title={"text": text, "x": 0.5, "xanchor": "center", **kwargs})

    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        path = Path(path)
        if path.suffix.lower() in (".html", ".htm"):
            self._fig.write_html(str(path))
            return
        try:
            scale = (dpi or self._style.dpi) / 100.0
            self._fig.write_image(str(path), scale=scale)
        except Exception as exc:  # pragma: no cover - depends on kaleido
            raise RuntimeError(
                f"Writing {path.suffix} with the plotly backend needs kaleido "
                "(pip install kaleido). Use a .html path to avoid it."
            ) from exc

    def embed_html(self, *, dpi: float | None = None) -> str:
        return self._fig.to_html(full_html=False, include_plotlyjs="cdn")

    def show(self) -> None:
        self._fig.show()

    def close(self) -> None:
        self._fig.data = ()


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
) -> PlotlyFigure:
    """Create a plotly figure with a ``rows`` x ``cols`` grid of subplots."""
    from plotly.subplots import make_subplots

    style = style or get_style()
    if figsize is not None:
        style = style.replace(figsize=figsize)
    width, height = style.pixel_size

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=sharex,
        shared_yaxes=sharey,
        # make_subplots wants top-to-bottom row heights, matching gridspec.
        row_heights=list(height_ratios) if height_ratios is not None else None,
        column_widths=list(width_ratios) if width_ratios is not None else None,
    )
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 50, "b": 60},
    )
    return PlotlyFigure(fig, rows, cols, style)


def wrap(obj, *, style: PlotStyle | None = None) -> PlotlyAxes | PlotlyFigure:
    """Adopt an existing ``go.Figure`` as a single-cell :class:`PlotlyFigure`."""
    import plotly.graph_objects as go

    if isinstance(obj, go.Figure):
        return PlotlyFigure(obj, 1, 1, style or get_style())
    raise TypeError(f"Cannot wrap {type(obj).__name__} with the plotly backend")
