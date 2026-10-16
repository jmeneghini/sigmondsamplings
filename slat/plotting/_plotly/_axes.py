"""PlotlyAxes: one plotly subplot behind the backend-neutral Axes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..base import Axes, Coords
from ..text import line_count, to_html, to_plotly
from ._convert import (
    _LEGEND_LOC,
    _NO_LINE,
    _clean,
    _color,
    _colorscale,
    _dash,
    _err,
    _hover_text,
    _label,
    _line_spec,
    _marker_spec,
    _opacity,
)
from ._units import _GRID_COLOR, _LEADING, _Z_IMAGE, _Z_LINE, _Z_PATCH, _px, _size, _suffix, _z

if TYPE_CHECKING:
    from ._figure import PlotlyFigure


class PlotlyAxes(Axes):
    """An :class:`~slat.plotting.base.Axes` backed by one plotly subplot."""

    backend = "plotly"

    def __init__(self, figure: PlotlyFigure, row: int, col: int, refs=None):
        self._figure = figure
        # plotly's row/col are 1-indexed; ours are 0-indexed.
        self._row = row + 1
        self._col = col + 1
        _, ncols = figure.shape
        self._suffix = _suffix(row * ncols + col + 1)
        self._refs = refs or (f"x{self._suffix}", f"y{self._suffix}")
        # Each axes has its own legend, as on matplotlib.
        self._legend_name = f"legend{self._suffix}"
        # Kept so the legend and colorbar follow the subplot when rows move.
        self._legend_anchor: tuple[float, float] | None = None
        self._colorbar_index: int | None = None
        # Hidden [0, 1] overlay axes carrying axis-line hover targets, by letter.
        self._hover_axes: dict[str, str] = {}
        self._group_count = 0
        self._group: dict[str, Any] | None = None

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
        return self._refs[0]

    @property
    def _yref(self) -> str:
        return self._refs[1]

    def _axis_key(self, axis):
        ref = self._xref if axis == "x" else self._yref
        return f"{axis}axis{ref[1:]}"

    def _update_axis(self, axis, **updates):
        self._fig.update_layout({self._axis_key(axis): updates})

    def _layout_axis(self, key):
        # Native traces can reference axes whose layout has no explicit entry.
        # Read defaults without inserting layout state until drawing/configuring.
        try:
            return self._fig.layout[key]
        except KeyError:
            import plotly.graph_objects as go
            return go.layout.XAxis() if key.startswith("x") else go.layout.YAxis()

    def _matched_axes(self, axis):
        """Connected native matches, including reverse links; tolerate cycles."""
        keys = [key for key in self._fig.layout.to_plotly_json()
                if re.fullmatch(rf"{axis}axis\d*", key)]
        start = self._axis_key(axis)
        pending, seen = [start], set()
        while pending:
            key = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            target = self._layout_axis(key)
            yield target
            if target.matches:
                pending.append(f"{axis}axis{target.matches[1:]}")
            ref = axis + key[len(axis + "axis"):]
            pending.extend(other for other in keys if self._fig.layout[other].matches == ref)

    def _range_axis(self, axis):
        targets = list(self._matched_axes(axis))
        return next((target for target in targets
                     if target.range is not None and not target.autorange), targets[0])

    def _domain(self, axis: Literal["x", "y"]) -> tuple[float, float]:
        return tuple(self._layout_axis(self._axis_key(axis)).domain or (0.0, 1.0))

    @staticmethod
    def _hover(hover: Sequence[str] | None) -> dict[str, Any]:
        """Per-point hover text -- the main thing plotly buys over matplotlib."""
        if hover is None:
            return {}
        return {"hovertext": [_hover_text(text) for text in hover], "hoverinfo": "text"}

    def _legend_entry(self, label: str | None, *, grouped: bool = True) -> dict[str, Any]:
        """Legend attributes shared by traces and axis-line shapes."""
        entry = {
            "name": _label(label),
            "showlegend": label is not None,
            "legend": self._legend_name,
            # Entries in drawing order, traces and axis lines alike.
            "legendrank": self._figure._next_rank(),
        }
        if grouped and self._group is not None:
            entry["legendgroup"] = self._group["name"]
            entry["showlegend"] = label is not None and not self._group["labelled"]
            self._group["labelled"] |= label is not None
        return entry

    @contextmanager
    def _legend_group(self):
        self._group_count += 1
        previous = self._group
        self._group = {
            "name": f"slat-{self._legend_name}-{self._group_count}", "labelled": False
        }
        try:
            yield
        finally:
            self._group = previous

    def _entry(self, label: str | None, zorder: float | None, default_z: float) -> dict[str, Any]:
        """Legend and stacking attributes shared by every trace."""
        return {**self._legend_entry(label), "zorder": _z(zorder, default_z)}

    def _add(self, trace, native: dict[str, Any] | None) -> None:
        trace.update(xaxis=self._xref, yaxis=self._yref)
        if native:
            trace.update(**native)
        self._fig.add_trace(trace)

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    def _line(
        self, x, y, *, color, linewidth, linestyle, marker, markersize, label, alpha, zorder,
        hover, fill, native,
    ) -> None:
        import plotly.graph_objects as go

        no_line = linestyle in _NO_LINE
        mode = ("markers" if marker else "none") if no_line else (
            "lines+markers" if marker else "lines"
        )
        trace = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            line=_clean(
                {
                    "color": _color(color, alpha),
                    "width": _size(linewidth, "linewidth"),
                    "dash": None if no_line else _dash(linestyle),
                }
            ),
            marker=_marker_spec(marker, fill, _size(markersize, "markersize")),
            opacity=_opacity(color, alpha),
            **self._entry(label, zorder, _Z_LINE),
            **self._hover(hover),
        )
        self._add(trace, native)

    def _points(
        self, x, y, *, color, markersize, marker, label, alpha, zorder, hover, fill, native
    ) -> None:
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=_marker_spec(marker, fill, _size(markersize, "markersize"), _color(color, alpha)),
            opacity=_opacity(color, alpha),
            **self._entry(label, zorder, _Z_LINE),
            **self._hover(hover),
        )
        self._add(trace, native)

    def _errorbar(
        self, x, y, *, yerr, xerr, color, marker, markersize, capsize, linewidth, linestyle,
        label, alpha, zorder, hover, fill, native,
    ) -> None:
        import plotly.graph_objects as go

        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        rendered = _color(color, alpha)

        error_y = _err(yerr, len(y))
        error_x = _err(xerr, len(x))
        for spec in (error_y, error_x):
            if spec is not None:
                # plotly's cap width is a half-length, as matplotlib's capsize is.
                spec.update(
                    _clean(
                        {
                            "color": rendered,
                            "thickness": _size(linewidth, "linewidth"),
                            "width": _size(capsize, "capsize"),
                        }
                    )
                )

        lines = linestyle is not None and linestyle not in _NO_LINE
        trace = go.Scatter(
            x=x,
            y=y,
            mode=("lines+markers" if marker else "lines") if lines else (
                "markers" if marker else "none"
            ),
            marker=_marker_spec(marker, fill, _size(markersize, "markersize"), rendered),
            line=_clean(
                {
                    "color": rendered,
                    "width": _size(None, "linewidth"),
                    "dash": _dash(linestyle) if lines else None,
                }
            ),
            error_y=error_y,
            error_x=error_x,
            opacity=_opacity(color, alpha),
            **self._entry(label, zorder, _Z_LINE),
            **self._hover(hover),
        )
        self._add(trace, native)

    def _band(self, x, lo, hi, *, color, alpha, label, zorder, native) -> None:
        x = np.asarray(x)
        self._filled(
            np.concatenate([x, x[::-1]]),
            np.concatenate([np.asarray(hi), np.asarray(lo)[::-1]]),
            facecolor=color,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
        )

    def _polygon(
        self, x, y, *, facecolor, edgecolor, linewidth, alpha, label, zorder, native
    ) -> None:
        self._filled(
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
        zorder=None,
        native=None,
    ) -> None:
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=_color(facecolor, alpha),
            opacity=_opacity(facecolor, alpha),
            line=_clean(
                {
                    "color": _color(edgecolor) if edgecolor else "rgba(0,0,0,0)",
                    "width": _size(linewidth, "edgewidth") if edgecolor else 0,
                }
            ),
            hoverinfo="skip",
            **self._entry(label, zorder, _Z_PATCH),
        )
        self._add(trace, native)

    def _bar(self, x, height, *, width, color, label, alpha, hover, native) -> None:
        import plotly.graph_objects as go

        trace = go.Bar(
            x=x,
            y=height,
            width=width,
            # No outline: matplotlib bars have none, the plotly template's are white.
            marker=_clean({"color": _color(color, alpha), "line": {"width": 0}}),
            opacity=_opacity(color, alpha),
            **self._entry(label, None, _Z_PATCH),
            **self._hover(hover),
        )
        self._add(trace, native)

    def _hlines(
        self, y, xmin, xmax, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        ys, x0s, x1s = np.broadcast_arrays(
            np.atleast_1d(y), np.atleast_1d(xmin), np.atleast_1d(xmax)
        )
        labels: list[str] | None = None
        if hover is not None:
            labels = [hover] * len(ys) if isinstance(hover, str) else list(hover)
            if len(labels) != len(ys):
                raise ValueError("hlines hover must have one string per segment")
            labels = [_hover_text(text) for text in labels]
        # Hover has two paths. Here, a midpoint gives each segment an interior
        # target with no JavaScript (fig.show(), notebooks); in saved HTML the
        # post-script rebuilds dense targets from ``slat_hover_segments`` on zoom.
        xs: list[float | None] = []
        vals: list[float | None] = []
        hovertext: list[str] = []
        segments: list[list] = []
        for i, (yi, a, b) in enumerate(zip(ys, x0s, x1s)):
            if labels is None:
                xs += [a, b, None]
                vals += [yi, yi, None]
                continue
            xs += [a, (a + b) / 2, b, None]
            vals += [yi, yi, yi, None]
            hovertext += [labels[i]] * 4
            segments.append([float(yi), float(a), float(b), labels[i]])
        self._segments(
            xs,
            vals,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
            hover=hovertext if labels is not None else None,
            segments=segments if labels is not None else None,
        )

    def _vlines(
        self, x, ymin, ymax, *, color, linestyle, linewidth, alpha, label, zorder, native
    ) -> None:
        xs_in, y0s, y1s = np.broadcast_arrays(
            np.atleast_1d(x), np.atleast_1d(ymin), np.atleast_1d(ymax)
        )
        xs: list[float | None] = []
        vals: list[float | None] = []
        for xi, a, b in zip(xs_in, y0s, y1s):
            xs += [xi, xi, None]
            vals += [a, b, None]
        self._segments(
            xs,
            vals,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
        )

    def _segments(
        self,
        xs,
        ys,
        *,
        color,
        linestyle,
        linewidth,
        alpha,
        label,
        zorder,
        native,
        hover=None,
        segments=None,
    ) -> None:
        import plotly.graph_objects as go

        trace = go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            connectgaps=False,
            line=_line_spec(color, alpha, linewidth, linestyle),
            opacity=_opacity(color, alpha),
            hoverinfo="text" if hover is not None else "skip",
            hovertext=hover,
            meta={"slat_hover_segments": segments} if segments is not None else None,
            **self._entry(label, zorder, _Z_LINE),
        )
        self._add(trace, native)

    def _axhline(
        self, y, *, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        self._axline(
            "h",
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
            hover=hover,
        )

    def _axvline(
        self, x, *, ymin, ymax, color, linestyle, linewidth, alpha, label, zorder, hover, native
    ) -> None:
        native = {"y0": ymin, "y1": ymax, **(native or {})}
        self._axline(
            "v",
            x,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
            native=native,
            hover=hover,
        )

    def _axline(
        self, kind, value, *, color, linestyle, linewidth, alpha, label, zorder, native, hover
    ) -> None:
        line = _line_spec(color, alpha, linewidth, linestyle)
        opacity = _opacity(color, alpha)
        kwargs: dict[str, Any] = _clean(
            {
                "line": line,
                # Explicit: plotly's templates default shapes to 0.3 opacity.
                "opacity": 1.0 if opacity is None else opacity,
                # A shape cannot interleave with traces. Like matplotlib's
                # gridlines (zorder 1.5) it goes under them or over everything.
                "layer": "below" if (_Z_LINE if zorder is None else zorder) < 1.5 else "above",
                # Shapes carry legend entries natively.
                **self._legend_entry(label, grouped=False),
                "type": "line",
                "xref": f"{self._xref} domain" if kind == "h" else self._xref,
                "yref": self._yref if kind == "h" else f"{self._yref} domain",
                "x0": 0.0 if kind == "h" else value,
                "x1": 1.0 if kind == "h" else value,
                "y0": value if kind == "h" else 0.0,
                "y1": value if kind == "h" else 1.0,
            }
        )
        if hover is not None:
            # Toggling the legend entry hides the hover target with the line.
            kwargs.setdefault("legendgroup", f"slat-{self._legend_name}-axline-{kwargs['legendrank']}")
        if native:
            kwargs.update(native)
        self._fig.add_shape(**kwargs)
        if hover is not None:
            if kind == "h":
                span = (kwargs.get("x0", 0.0), kwargs.get("x1", 1.0))
            else:
                span = (kwargs["y0"], kwargs["y1"])
            self._axline_hover(kind, value, span, hover, line.get("color"), kwargs["legendgroup"])

    def _axline_hover(self, kind, value, span, hover, color, group) -> None:
        """An invisible trace along an axis line, so the line answers hover.

        A shape cannot carry hover text. The trace's position along the line
        is on a hidden overlay axis fixed to [0, 1], so it spans the line at
        any zoom without depending on the data range.
        """
        import plotly.graph_objects as go

        along = self._hover_axis("y" if kind == "v" else "x")
        fractions = np.linspace(span[0], span[1], 41)  # a target every 2.5%
        fixed = np.full_like(fractions, value, dtype=float)
        x, y = (fixed, fractions) if kind == "v" else (fractions, fixed)
        trace = go.Scatter(
            x=x,
            y=y,
            xaxis=self._xref if kind == "v" else along,
            yaxis=along if kind == "v" else self._yref,
            mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hovertext=[_hover_text(hover)] * len(fractions),
            hoverinfo="text",
            hoverlabel=_clean({"bgcolor": color}),
            showlegend=False,
            legend=self._legend_name,
            legendgroup=group,
        )
        self._fig.add_trace(trace)

    def _hover_axis(self, letter: str) -> str:
        """This subplot's hidden ``letter`` overlay axis with a fixed [0, 1] range."""
        if letter not in self._hover_axes:
            pattern = re.compile(rf"{letter}axis(\d*)")
            used = [
                int(match.group(1) or 1)
                for key in self._fig.layout.to_plotly_json()
                if (match := pattern.fullmatch(key))
            ]
            index = max(used, default=1) + 1
            base, other = (self._yref, self._xref) if letter == "y" else (self._xref, self._yref)
            self._fig.layout[f"{letter}axis{index}"] = {
                "overlaying": base,
                "anchor": other,
                "range": [0, 1],
                "fixedrange": True,
                "visible": False,
            }
            self._hover_axes[letter] = f"{letter}{index}"
        return self._hover_axes[letter]

    def _heatmap(
        self, values, *, x, y, cmap, vmin, vmax, origin, colorbar, colorbar_label, hover, native
    ) -> None:
        import plotly.graph_objects as go

        # NaN cells are gaps in a plotly heatmap: no color, as on matplotlib.
        values = np.asarray(values)
        text = {}
        if hover is not None:
            text = {"text": [[_hover_text(t) for t in row] for row in hover], "hoverinfo": "text"}
        trace = go.Heatmap(
            z=values,
            x=x,
            y=y,
            colorscale=_colorscale(cmap),
            zmin=vmin,
            zmax=vmax,
            # Plotly otherwise ignores a lone bound and autoscales both ends.
            zauto=False if vmin is not None or vmax is not None else None,
            showscale=colorbar,
            colorbar=self._colorbar(colorbar_label) if colorbar else None,
            zorder=_z(None, _Z_IMAGE),
            **text,
        )
        self._add(trace, native)
        if colorbar:
            self._colorbar_index = len(self._fig.data) - 1
        if origin == "upper":
            # Match imshow: row 0 at the top.
            self._update_axis("y", autorange="reversed")

    def _colorbar(self, label: str | None) -> dict[str, Any]:
        """A colorbar beside this subplot, taking its room from the subplot.

        This is what ``fig.colorbar(ax=...)`` does on matplotlib; a plotly
        colorbar otherwise spans the whole figure height at the right edge.
        """
        width, height = self._figure._pixels()
        x0, x1 = self._domain("x")
        y0, y1 = self._domain("y")
        font = _size(self.style.tick_fontsize, "font")
        # matplotlib's colorbar aspect is 20: thickness is length / 20.
        thickness = min(30.0, max(8.0, (y1 - y0) * height / 20.0))
        gap = 10.0
        ticks = _size(None, "ticklen") + 3.0 * font  # tick marks plus ~5 digits of label
        title = _LEADING * font if label else 0.0
        steal = min((gap + thickness + ticks + title) / width, 0.4 * (x1 - x0))
        self._fig.update_layout({self._axis_key("x"): {"domain": [x0, x1 - steal]}})
        return _clean(
            {
                "title": _clean({"text": to_plotly(label), "side": "right"}) if label else None,
                "x": x1 - steal + gap / width,
                "xref": "paper",
                "xanchor": "left",
                "y": (y0 + y1) / 2,
                "yref": "paper",
                "yanchor": "middle",
                "len": y1 - y0,
                "lenmode": "fraction",
                "thickness": thickness,
                "thicknessmode": "pixels",
                "outlinewidth": _size(None, "spine"),
                "ticks": "outside",
                "ticklen": _size(None, "ticklen"),
                "tickfont": {"size": font},
            }
        )

    def _text(self, x, y, s, *, coords, ha, va, fontsize, color, alpha, native) -> None:
        xc, yc = (coords, coords) if isinstance(coords, str) else coords
        size = _size(fontsize, "font")
        kwargs = _clean(
            {
                "x": x,
                "y": y,
                "text": to_plotly(s),
                "xref": f"{self._xref}{' domain' if xc == 'axes' else ''}",
                "yref": f"{self._yref}{' domain' if yc == 'axes' else ''}",
                "showarrow": False,
                "xanchor": ha if ha in ("left", "right", "center") else "left",
                "yanchor": {"bottom": "bottom", "top": "top", "center": "middle"}.get(va, "bottom"),
                "font": _clean({"size": size, "color": _color(color, alpha)}),
            }
        )
        if native:
            kwargs.update(native)
        self._fig.add_annotation(**kwargs)
        if xc == "axes":
            self._reserve_margin(x, "x", s, size)
        if yc == "axes":
            self._reserve_margin(y, "y", s, size)

    def _reserve_margin(
        self, value: float, axis: Literal["x", "y"], text: str, size: float
    ) -> None:
        """Grow the figure margin so an out-of-domain annotation stays visible.

        matplotlib's layout engine makes room for text placed outside the axes;
        plotly clips it against the paper instead. Spectrum plots put their
        outer-group labels below the axes, and right-edge labels sit past the
        right domain edge, so reserve the space on whichever side is used.

        The text allowance is an estimate -- measuring a string needs a renderer,
        which is exactly what is not available at build time. ``size`` is the
        font size in pixels.
        """
        if 0.0 <= value <= 1.0:
            return
        overshoot = -value if value < 0 else value - 1.0
        width, height = self._figure._pixels()
        if axis == "x":
            # Width scales with the string; 0.62 em per character is a rough mean.
            allowance = min(240.0, 8.0 + 0.62 * size * len(str(text)))
            side = "l" if value < 0 else "r"
            extent = width
        else:
            allowance = 2.0 * size  # one line, plus breathing room
            side = "b" if value < 0 else "t"
            extent = height
        self._figure._reserve(side, overshoot * extent + allowance)

    def _contour(self, x, y, z, *, levels, labels, colors, filled, linewidth, native) -> None:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale

        values = [(a + b) / 2 for a, b in zip(levels, levels[1:])] if filled else levels
        if colors is None:
            span = levels[-1] - levels[0]
            palette = sample_colorscale("Viridis", [(v - levels[0]) / span if span else 0.5
                                                   for v in values])
        else:
            palette = (colors,) if isinstance(colors, str) else colors
        # One trace per irregular line or band. Constraint contours shade the
        # *invalid* region: allowing the outside (][) fills the inside interval.
        for i, value in enumerate(values):
            color = _color(palette[i % len(palette)])
            contours = ({"type": "constraint", "operation": "][",
                         "value": [levels[i], levels[i + 1]], "showlines": False}
                        if filled else
                        {"start": value, "end": value, "size": 1,
                         "coloring": "lines", "showlabels": labels})
            trace = go.Contour(
                x=x,
                y=y,
                z=z,
                autocontour=False,
                contours=contours,
                fillcolor=color if filled else None,
                # Line coloring uses a constant scale, as Plotly otherwise
                # takes its line color from the z colorscale instead.
                colorscale=[[0, color], [1, color]],
                line={"color": color, "width": _size(linewidth, "linewidth")},
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
                zorder=_z(None, _Z_PATCH if filled else _Z_LINE),
            )
            self._add(trace, native)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _limits(self, axis, explicit_only):
        target = self._range_axis(axis)
        if target.range is None or (explicit_only and target.autorange):
            return None
        limits = target.range
        if target.type == "log":
            limits = tuple(10.0 ** value for value in limits)
        return tuple(limits)

    def _set(self, *, xlabel, ylabel, title, xlim, ylim, xscale, yscale, label_fontsize) -> None:
        if label_fontsize is None:
            label_fontsize = self.style.label_fontsize
        size = _size(label_fontsize, "font")
        for axis, label, lim, scale in (
            ("x", xlabel, xlim, xscale), ("y", ylabel, ylim, yscale),
        ):
            # Preserve data-unit ranges when only the scale changes.
            if lim is None and scale is not None:
                lim = self.get_limits(axis, explicit_only=True)
            updates = _clean(
                {
                    "title_text": to_plotly(label),
                    "range": self._range(axis, lim, scale),
                    "type": scale,
                }
            )
            if label is not None:
                updates["title_font"] = {"size": size}
            if lim is not None:
                updates["autorange"] = False
            if updates:
                self._update_axis(axis, **updates)
                shared = {key: value for key, value in updates.items()
                          if key in ("range", "type", "autorange")}
                for target in self._matched_axes(axis):
                    target.update(**shared)
        if title is not None:
            self._title(title)

    def _title(self, text: str) -> None:
        """The axes title, centred above this subplot as matplotlib's is.

        Always an annotation, even on a 1x1 figure, so the figure title stays
        free for :meth:`PlotlyFigure.suptitle`. Setting it again replaces it;
        ``""`` clears it.
        """
        name = f"slat-title{self._suffix}"
        self._fig.layout.annotations = [
            a for a in self._fig.layout.annotations if a.name != name
        ]
        if not text:
            return
        size = _size(self.style.title_fontsize, "title")
        pad = _size(None, "titlepad")
        self._fig.add_annotation(
            x=0.5,
            y=1.0,
            xref=f"{self._xref} domain",
            yref=f"{self._yref} domain",
            text=to_plotly(text),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=pad,
            font={"size": size},
            name=name,
        )
        # A top-row title sits in the figure margin, a lower one in the gap
        # above its row; either grows to fit it.
        lines = line_count(text)
        needed = pad + lines * _LEADING * size + 4.0
        if self._row == 1:
            self._figure._reserve("t", needed)
        else:
            self._figure._reserve_row_gap(self._row - 1, needed)

    def _range(self, axis: Literal["x", "y"], lim, scale: str | None) -> list[float] | None:
        """Limits as plotly wants them: log10 of the data limits on a log axis.

        The axis type is the one being set in the same call, else the current one,
        so set ``yscale="log"`` before or together with ``ylim``.
        """
        if lim is None:
            return None
        kind = scale or self._range_axis(axis).type
        if kind != "log":
            return list(lim)
        lo, hi = lim
        if lo <= 0 or hi <= 0:
            raise ValueError(f"log-scale {axis} limits must be positive, got {tuple(lim)}")
        return [float(np.log10(lo)), float(np.log10(hi))]

    def _ticks(self, axis, positions, labels, *, rotation, fontsize) -> None:
        fontsize = fontsize if fontsize is not None else self.style.tick_fontsize
        updates: dict[str, Any] = {
            "tickmode": "array",
            "tickvals": list(positions),
            "tickfont": {"size": _size(fontsize, "font")},
            # matplotlib rotates counter-clockwise, plotly clockwise.
            "tickangle": -rotation,
        }
        if labels is not None:
            # MathJax is unreliable in tick labels; fall back to Unicode.
            updates["ticktext"] = [to_html(label) for label in labels]
        self._update_axis(axis, **updates)

    def _grid(self, show, *, axis, alpha) -> None:
        alpha = alpha if alpha is not None else self.style.grid_alpha
        updates = {
            "showgrid": show,
            "gridcolor": _color(_GRID_COLOR, alpha),
            "gridwidth": _size(None, "grid"),
        }
        if axis in ("x", "both"):
            self._update_axis("x", **updates)
        if axis in ("y", "both"):
            self._update_axis("y", **updates)

    def _legend(self, show, *, loc, ncol, frameon, fontsize) -> None:
        """This subplot's own legend, placed inside it as on matplotlib."""
        if not show:
            self._legend_anchor = None
            self._fig.update_layout({self._legend_name: {"visible": False}})
            return
        fontsize = fontsize if fontsize is not None else self.style.legend_fontsize
        fx, fy, xanchor, yanchor = _LEGEND_LOC.get(loc or "best", _LEGEND_LOC["best"])
        self._legend_anchor = (fx, fy)
        legend = {
            "visible": True,
            "groupclick": "togglegroup",
            "tracegroupgap": 0 if self._group_count else None,
            **self._legend_position(),
            "xref": "paper",
            "yref": "paper",
            "xanchor": xanchor,
            "yanchor": yanchor,
            "font": {"size": _size(fontsize, "font")},
            # matplotlib's frame: white at 0.8 alpha with a light grey edge.
            "bgcolor": "rgba(255,255,255,0.8)" if frameon else "rgba(255,255,255,0)",
            "bordercolor": "#cccccc",
            "borderwidth": 1 if frameon else 0,
            # plotly has no column count; horizontal is the closest analogue.
            "orientation": "h" if ncol > 1 else "v",
        }
        self._fig.update_layout({self._legend_name: legend})

    def _legend_position(self) -> dict[str, float]:
        fx, fy = self._legend_anchor
        x0, x1 = self._domain("x")
        y0, y1 = self._domain("y")
        return {"x": x0 + fx * (x1 - x0), "y": y0 + fy * (y1 - y0)}

    def _place(self) -> None:
        """Re-anchor the legend and colorbar after the subplot's domain moved."""
        if self._legend_anchor is not None:
            self._fig.update_layout({self._legend_name: self._legend_position()})
        if self._colorbar_index is not None:
            y0, y1 = self._domain("y")
            self._fig.data[self._colorbar_index].colorbar.update(y=(y0 + y1) / 2, len=y1 - y0)

    def point_size(self, points: float, axis="y", *, units: Coords = "data") -> float:
        """Estimate at the axis midpoint using the inner plotting dimensions.

        On log axes this is a local data offset at the geometric midpoint,
        rather than an additive exponent span. Set data limits first.
        """
        domain = self._domain(axis)
        total = self._figure._paper_pixels()[0 if axis == "x" else 1]
        extent_px = total * abs(domain[1] - domain[0])
        if extent_px <= 0:
            return 0.0
        fraction = _px(points) / extent_px
        if units == "axes":
            return fraction
        target = self._range_axis(axis)
        if target.range is None:
            return 0.0
        span = abs(target.range[1] - target.range[0])
        if target.type == "log":
            midpoint = sum(target.range) / 2
            offset = 10.0 ** (midpoint + span * abs(fraction)) - 10.0 ** midpoint
            return float(np.copysign(offset, points))
        return span * fraction
