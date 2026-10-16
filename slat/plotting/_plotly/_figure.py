"""PlotlyFigure, figure creation, and the subplot layout engine."""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..base import Figure, FigureSpec
from ..style import PlotStyle, get_style
from ..text import line_count, to_plotly
from ._axes import PlotlyAxes
from ._convert import _color
from ._units import _FONT_FAMILY, _GRID_COLOR, _LEADING, PX_PER_INCH, _size, _suffix

# Run after Plotly has resolved autoranges. Keep initial sizes and axis spans so
# repeated zooms do not compound the scaling, and each subplot scales locally.
# Plotly substitutes the ``{plot_id}`` placeholder in the text.
_POST_SCRIPT = files(__package__).joinpath("post_script.js").read_text()

#: Default margins in pixels. Titles and out-of-domain text grow them.
_MARGIN = {"l": 70, "r": 30, "t": 20, "b": 60}


class PlotlyFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` backed by ``plotly.graph_objects.Figure``."""

    backend = "plotly"

    def __init__(
        self,
        fig,
        rows: int,
        cols: int,
        style: PlotStyle,
        *,
        row_layout: tuple[Sequence[float], float] | None = None,
        axis_refs=None,
    ):
        self._fig = fig
        self._shape = (rows, cols)
        self._style = style
        self._axes = [
            [PlotlyAxes(self, r, c, None if axis_refs is None else axis_refs[r][c])
             if axis_refs is None or axis_refs[r][c] is not None else None
             for c in range(cols)] for r in range(rows)
        ]
        margin = fig.layout.margin
        # A wrapped figure keeps its own margins as the base.
        self._base_margin = {
            side: margin[side] if margin[side] is not None else (100 if side == "t" else 80)
            for side in "lrtb"
        }
        self._reserved = dict.fromkeys("lrtb", 0.0)
        self._suptitle_px = 0.0
        self._rank = 0
        # (row height ratios, pixels between rows before titles); None leaves
        # the domains to plotly, as for a wrapped figure.
        self._row_layout = row_layout
        self._row_title_px = [0.0] * rows
        self._layout_rows()

    @property
    def native(self):
        return self._fig

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def axes(self, row: int = 0, col: int = 0) -> PlotlyAxes:
        axes = self._axes[row][col]
        if axes is None:
            raise ValueError("this native grid cell has no Cartesian drawing axes")
        return axes

    @property
    def flat(self):
        return [axes for row in self._axes for axes in row if axes is not None]

    def _pixels(self) -> tuple[float, float]:
        """The layout size in pixels."""
        layout = self._fig.layout
        # Wrapped native figures retain Plotly's default export dimensions;
        # created figures already carry the style's explicit pixel dimensions.
        default = (700, 450) if self._row_layout is None else (
            self._style.figsize[0] * PX_PER_INCH, self._style.figsize[1] * PX_PER_INCH,
        )
        return (
            layout.width or default[0],
            layout.height or default[1],
        )

    def _next_rank(self) -> int:
        self._rank += 1
        return self._rank

    def _paper_pixels(self):
        width, height = self._pixels()
        margin = self._fig.layout.margin
        margins = {side: margin[side] if margin[side] is not None else self._base_margin[side]
                   for side in "lrtb"}
        return (width - margins["l"] - margins["r"],
                height - margins["t"] - margins["b"])

    def _reserve(self, side: str, pixels: float) -> None:
        """Make the ``side`` margin at least ``pixels`` (plus the figure title, on top)."""
        if pixels > self._reserved[side]:
            self._reserved[side] = pixels
            self._apply_margins()

    def _apply_margins(self) -> None:
        margin = {side: max(self._base_margin[side], self._reserved[side]) for side in "lrtb"}
        margin["t"] += self._suptitle_px
        self._fig.update_layout(margin={side: round(value) for side, value in margin.items()})
        self._layout_rows()

    def _reserve_row_gap(self, row: int, pixels: float) -> None:
        """Make the gap above ``row`` hold ``pixels`` of title."""
        if pixels > self._row_title_px[row]:
            self._row_title_px[row] = pixels
            self._layout_rows()

    def _layout_rows(self) -> None:
        """Set the row domains from pixel gaps, as constrained layout sizes them.

        Each gap holds the upper row's x axis and the lower row's title, so a
        row without a title sits close under the one above.
        """
        rows, cols = self._shape
        if self._row_layout is None or rows == 1:
            return
        ratios, below = self._row_layout
        _, height = self._pixels()
        margin = self._fig.layout.margin
        paper = height - (margin.t or 0) - (margin.b or 0)
        gaps = [below + self._row_title_px[r] for r in range(1, rows)]
        total = sum(gaps)
        if total > 0.5 * paper:  # the cells keep at least half the height
            gaps = [gap * 0.5 * paper / total for gap in gaps]
            total = 0.5 * paper
        cell = (paper - total) / sum(ratios)
        top = 1.0
        domains = {}
        for r in range(rows):
            extent = cell * ratios[r] / paper
            for c in range(cols):
                index = r * cols + c + 1
                domains[f"yaxis{_suffix(index)}"] = {
                    "domain": [max(0.0, top - extent), top]
                }
            if r < rows - 1:
                top -= extent + gaps[r] / paper
        self._fig.update_layout(domains)
        for axes in self.flat:
            axes._place()

    def set_size(self, width: float, height: float) -> None:
        """Resize the layout and re-place rows, legends and colorbars.

        A colorbar's thickness, and the width it took from its subplot, were
        fixed when its heatmap was drawn and are not recomputed.
        """
        self._style = self._style.replace(figsize=(width, height))
        self._fig.update_layout(
            width=round(width * PX_PER_INCH), height=round(height * PX_PER_INCH)
        )
        self._apply_margins()

    def hide(self, row: int, col: int) -> None:
        axes = self.axes(row, col)
        axes._update_axis("x", visible=False)
        axes._update_axis("y", visible=False)
        frame = _frame_name(row * self._shape[1] + col + 1)
        self._fig.layout.shapes = [s for s in self._fig.layout.shapes if s.name != frame]

    def suptitle(self, text: str, *, fontsize: float | None = None) -> None:
        """The figure title, above every axes title, with room made for it."""
        fontsize = fontsize if fontsize is not None else self.style.title_fontsize
        size = _size(fontsize, "title")
        pad = _size(None, "titlepad")
        lines = line_count(text)
        self._fig.update_layout(
            title={
                "text": to_plotly(text),
                "x": 0.5,
                "xref": "container",
                "xanchor": "center",
                "y": 1.0,
                "yref": "container",
                "yanchor": "top",
                "pad": {"t": pad},
                "font": {"size": size},
            }
        )
        self._suptitle_px = 2 * pad + lines * _LEADING * size
        self._apply_margins()

    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        self._preserve_empty_cells()
        path = Path(path)
        if path.suffix.lower() in (".html", ".htm"):
            self._fig.write_html(
                str(path), include_mathjax="cdn", post_script=_POST_SCRIPT
            )
            return
        try:
            # The layout is PX_PER_INCH pixels per inch; the export is at dpi.
            scale = (dpi or self._style.dpi) / PX_PER_INCH
            self._fig.write_image(str(path), scale=scale)
        except Exception as exc:  # pragma: no cover - depends on kaleido
            raise RuntimeError(
                f"Writing {path.suffix} with the plotly backend needs kaleido "
                "(pip install kaleido). Use a .html path to avoid it."
            ) from exc

    def embed_html(self, *, dpi: float | None = None) -> str:
        self._preserve_empty_cells()
        return self._fig.to_html(
            full_html=False, include_plotlyjs="cdn", include_mathjax="cdn",
            post_script=_POST_SCRIPT,
        )

    def show(self) -> None:
        self._preserve_empty_cells()
        self._fig.show()

    def _preserve_empty_cells(self) -> None:
        """Keep trace-free cells in Plotly.js without moving existing traces.

        Titles, text and shapes alone do not retain their axes. Append empty
        traces only at output time, so drawing indices stay unchanged. Null
        coordinates supply no autorange values, hover targets or legend items.
        Wrapped primary Cartesian cells use the same preservation step.
        """
        import plotly.graph_objects as go

        referenced = {
            (trace.xaxis or "x", trace.yaxis or "y")
            for trace in self._fig.data
            if hasattr(trace, "xaxis") and hasattr(trace, "yaxis") and trace.visible is not False
        }
        for axes in self.flat:
            pair = (axes._xref, axes._yref)
            visible = any(
                axes._layout_axis(axes._axis_key(letter)).visible is not False
                for letter in "xy"
            )
            if visible and pair not in referenced:
                self._fig.add_trace(go.Scatter(
                    x=[None], y=[None], xaxis=pair[0], yaxis=pair[1],
                    mode="none", showlegend=False, hoverinfo="skip",
                    meta={"slat_empty_cell": True},
                ))

    def close(self) -> None:
        self._fig.data = ()


def _row_gap(sharex: bool, style: PlotStyle) -> float:
    """Pixels between rows before any title: the upper row's x ticks and label."""
    if sharex:
        return 10.0
    ticks = _LEADING * _size(style.tick_fontsize, "font")
    label = _LEADING * _size(style.label_fontsize, "font")
    return _size(None, "ticklen") + ticks + label + _size(None, "labelpad") + 10.0


def _column_spacing(cols: int, width: float, sharey: bool, style: PlotStyle) -> dict[str, float]:
    """The gap between columns, as a fraction, sized for y tick labels and the y label.

    make_subplots' default is a fixed fraction of the figure, far wider than
    matplotlib's constrained layout. Row gaps are set later, in pixels, by
    :meth:`PlotlyFigure._layout_rows`.
    """
    if cols == 1:
        return {}
    ticks = 4 * 0.62 * _size(style.tick_fontsize, "font")  # ~4 characters of tick label
    label = _LEADING * _size(style.label_fontsize, "font")
    yaxis = _size(None, "ticklen") + ticks + label + _size(None, "labelpad") + 10
    # make_subplots refuses gaps that leave a cell no room.
    return {"horizontal_spacing": min((15.0 if sharey else yaxis) / width, 0.5 / (cols - 1))}


def make_figure(spec: FigureSpec) -> PlotlyFigure:
    """Create a plotly figure with a ``spec.rows`` x ``spec.cols`` grid of subplots.

    Laid out at :data:`PX_PER_INCH` with matplotlib's default sizes and
    look: a black frame, outward ticks, no grid unless ``style.grid``, and a
    legend per subplot shown only by :meth:`PlotlyAxes.legend`.
    """
    from plotly.subplots import make_subplots

    rows, cols, style = spec.rows, spec.cols, spec.style
    width = round(style.figsize[0] * PX_PER_INCH)
    height = round(style.figsize[1] * PX_PER_INCH)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=spec.sharex,
        shared_yaxes=spec.sharey,
        # make_subplots wants top-to-bottom row heights, matching gridspec.
        row_heights=list(spec.height_ratios) if spec.height_ratios is not None else None,
        column_widths=list(spec.width_ratios) if spec.width_ratios is not None else None,
        **_column_spacing(cols, width, spec.sharey, style),
    )
    hidden_legends = {
        f"legend{_suffix(i)}": {"visible": False} for i in range(1, rows * cols + 1)
    }
    fig.update_layout(
        width=width,
        height=height,
        template="simple_white",
        margin=_MARGIN,
        font={"family": _FONT_FAMILY, "size": _size(None, "font"), "color": "black"},
        # Legends are per subplot and start hidden; Axes.legend shows one.
        showlegend=True,
        **hidden_legends,
    )
    spine = _size(None, "spine")
    fig.update_xaxes(title_standoff=_size(None, "labelpad"), **_axis_look(style, spine))
    fig.update_yaxes(title_standoff=_size(None, "labelpad"), **_axis_look(style, spine))
    for index in range(1, rows * cols + 1):
        fig.add_shape(**_frame(index, spine))
    ratios = list(spec.height_ratios) if spec.height_ratios is not None else [1.0] * rows
    return PlotlyFigure(fig, rows, cols, style, row_layout=(ratios, _row_gap(spec.sharex, style)))


def _frame_name(index: int) -> str:
    return f"slat-frame{_suffix(index)}"


def _frame(index: int, width: float) -> dict[str, Any]:
    """matplotlib's closed black frame (its spines) around one subplot.

    A shape rather than the axis lines: plotly.js stops drawing axis lines once
    a subplot's traces have more than one zorder, which every plot here does.
    """
    suffix = _suffix(index)
    return {
        "type": "rect",
        "xref": f"x{suffix} domain",
        "yref": f"y{suffix} domain",
        "x0": 0,
        "x1": 1,
        "y0": 0,
        "y1": 1,
        "line": {"color": "black", "width": width},
        "fillcolor": "rgba(0,0,0,0)",
        "layer": "above",
        "opacity": 1.0,
        "name": _frame_name(index),
    }


def _axis_look(style: PlotStyle, spine: float) -> dict[str, Any]:
    """matplotlib's default axes: outward ticks. The frame is :func:`_frame`."""
    return {
        "showline": False,
        "ticks": "outside",
        "ticklen": _size(None, "ticklen"),
        "tickwidth": spine,
        "tickcolor": "black",
        "showgrid": style.grid,
        "gridcolor": _color(_GRID_COLOR, style.grid_alpha),
        "gridwidth": _size(None, "grid"),
        "zeroline": False,
        # 10^-4 rather than plotly's SI "100µ", as matplotlib labels it.
        "exponentformat": "power",
    }


def wrap(obj, *, style: PlotStyle | None = None) -> PlotlyAxes | PlotlyFigure:
    """Adopt a native figure, preserving layout and primary Cartesian axes.

    Native subplot metadata retains its grid (including empty cells). Without
    it, Cartesian cells are indexed by their native domains, top to bottom and
    left to right. Secondary/overlay axes remain available through ``native``.
    Non-Cartesian cells can be exported but are not portable drawing surfaces.
    """
    import plotly.graph_objects as go

    if isinstance(obj, go.Figure):
        refs = _native_axes(obj)
        return PlotlyFigure(obj, len(refs), len(refs[0]), style or get_style(), axis_refs=refs)
    raise TypeError(f"Cannot wrap {type(obj).__name__} with the plotly backend")


def _native_axes(fig):
    import plotly.graph_objects as go

    grid = getattr(fig, "_grid_ref", None)
    if grid is not None:
        return [[(cell[0].trace_kwargs["xaxis"], cell[0].trace_kwargs["yaxis"])
                 if cell and cell[0].subplot_type == "xy" else None
                 for cell in row] for row in grid]
    layout = fig.layout.to_plotly_json()
    pairs = {(trace.xaxis or "x", trace.yaxis or "y") for trace in fig.data
             if hasattr(trace, "xaxis") and hasattr(trace, "yaxis")}
    keys = sorted({key for key in layout if re.fullmatch(r"xaxis\d*", key)} |
                  {"xaxis" + xref[1:] for xref, _ in pairs})
    if not keys:
        return [[("x", "y") if not fig.data or any(hasattr(t, "xaxis") for t in fig.data)
                 else None]]
    cells = []
    for key in keys:
        xaxis = go.layout.XAxis(layout.get(key, {}))
        if xaxis.overlaying:
            continue
        xref = "x" + key[5:]
        candidates = {yref for trace_x, yref in pairs if trace_x == xref
                      and not layout.get("yaxis" + yref[1:], {}).get("overlaying")}
        if xaxis.anchor and xaxis.anchor.startswith("y"):
            yref = xaxis.anchor
        elif len(candidates) == 1:
            yref = candidates.pop()
        elif len(candidates) > 1:
            raise ValueError("native Cartesian trace axes are ambiguous; configure anchors or use native")
        else:
            yref = "y" + key[5:]
        yaxis = go.layout.YAxis(layout.get("yaxis" + yref[1:], {}))
        if yaxis.overlaying:
            continue
        cells.append((tuple(xaxis.domain or (0, 1)), tuple(yaxis.domain or (0, 1)), (xref, yref)))
    if not cells:
        return [[None]]
    columns = sorted({cell[0] for cell in cells})
    rows = sorted({cell[1] for cell in cells}, key=lambda domain: domain[1], reverse=True)
    refs = [[None] * len(columns) for _ in rows]
    for xdomain, ydomain, pair in cells:
        r, c = rows.index(ydomain), columns.index(xdomain)
        if refs[r][c] is not None:
            raise ValueError("native Cartesian axes with identical domains are ambiguous; use native")
        refs[r][c] = pair
    return refs
