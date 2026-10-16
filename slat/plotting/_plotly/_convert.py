"""Translation of matplotlib-style colors, markers, dashes and labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..text import to_html
from ._units import _size

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
    "gray": "Greys_r",
    "grey": "Greys_r",
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
    rgb = _to_rgb(color)
    if rgb is None:
        # Unknown name: keep the color and lose any transparency.
        return color
    if alpha is None or alpha >= 1.0:
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _opacity(color: str | None, alpha: float | None) -> float | None:
    """Trace opacity for an ``alpha`` that :func:`_color` cannot fold into a color.

    With no color there is no rgba() to carry it, and an unknown color name
    drops it, so the whole trace takes it instead -- as matplotlib applies
    ``alpha`` to the whole artist.
    """
    if alpha is None or alpha >= 1.0:
        return None
    if color is not None and _to_rgb(color) is not None:
        return None
    return alpha


def _marker(symbol: str | None, fill: bool = True) -> str | None:
    """A plotly symbol name; ``fill=False`` picks its ``-open`` (hollow) variant."""
    if symbol is None:
        return None
    name = _MARKERS.get(symbol, symbol)
    if not fill and not name.endswith("-open"):
        name = f"{name}-open"
    return name


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


#: mpl legend ``loc`` strings mapped to anchors within one subplot's domain.
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

#: Line styles that draw no line.
_NO_LINE = frozenset({"none", "None", "", " "})


def _label(text: str | None) -> str | None:
    """A legend entry: HTML scripts, since MathJax is unreliable in legends."""
    return None if text is None else _hover_text(text)


def _hover_text(text: str) -> str:
    """Hover text with math as HTML scripts, which hover labels render.

    Converted per ``<br>`` line: :func:`to_html` rewrites ``_x`` anywhere in a
    string holding a ``$...$`` span, so a plain line such as ``Ecm/M_N`` must
    not share a conversion with a line that has math. A newline is a line break
    too, since plotly hover labels ignore ``\\n``.
    """
    lines = str(text).replace("\n", "<br>").split("<br>")
    return "<br>".join(to_html(line) for line in lines)


def _line_spec(color, alpha, linewidth, linestyle) -> dict[str, Any]:
    """A segment or axis line. A ``"none"`` linestyle keeps the mark at zero width,
    since matplotlib still gives an undrawn line its legend entry and hover."""
    if linestyle in _NO_LINE:
        return _clean({"color": _color(color, alpha), "width": 0})
    return _clean(
        {
            "color": _color(color, alpha),
            "width": _size(linewidth, "linewidth"),
            "dash": _dash(linestyle),
        }
    )


def _marker_spec(symbol, fill, size, color=None) -> dict[str, Any]:
    spec = _clean({"symbol": _marker(symbol, fill), "size": size, "color": color})
    if not fill:
        # Open symbols are stroked; matplotlib's edge is 1 pt.
        spec["line"] = {"width": _size(None, "markeredge")}
    return spec
