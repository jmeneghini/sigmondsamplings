"""Backend-neutral plotting for SLAT packages.

Plot code is written once against :class:`Axes` and rendered by whichever
backend is active. The split the API is organised around:

* **matplotlib** -- publication figures (PDF/PGF, ``text.usetex``, exact sizing)
* **plotly** -- exploration and HTML reports (hover, zoom, ``embed_html``)

The goal is not parity between the two. It is that the plots worth having in
both places get written once.

::

    import slat.plotting as slp

    fig = slp.figure(rows=2, height_ratios=[3, 1], sharex=True)
    top = fig.axes(0, 0)
    top.line(x, y, color="#d60000", label="data")
    top.set(ylabel=r"$E_{\\mathrm{cm}}$")
    fig.save("out.pdf")

Selecting a backend::

    slp.use("plotly")                    # process default
    with slp.backend("plotly"):  ...     # scoped
    fig = slp.figure(backend="plotly")   # single call

Both matplotlib and plotly are optional dependencies, imported only when their
backend is first used.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any

from .base import FEATURES, Axes, Coords, Figure, Origin, UnsupportedFeature
from .recording import Mark, Recorder, record
from .style import PlotStyle, get_style, set_style, style_context
from .text import ensure_math, strip_math, to_display

__all__ = [
    "FEATURES",
    "Axes",
    "Coords",
    "Figure",
    "Mark",
    "Origin",
    "PlotStyle",
    "Recorder",
    "UnsupportedFeature",
    "available",
    "backend",
    "current",
    "ensure_math",
    "figure",
    "get_style",
    "record",
    "set_style",
    "strip_math",
    "style_context",
    "to_display",
    "use",
    "wrap",
]

_MODULES = {
    "matplotlib": "._mpl",
    "plotly": "._plotly",
    "recording": ".recording",
}

_REQUIRES = {"matplotlib": "matplotlib", "plotly": "plotly"}

_current = "matplotlib"
_loaded: dict[str, ModuleType] = {}


def available() -> list[str]:
    """Backend names that can actually be used in this environment."""
    names = []
    for name, requirement in (("matplotlib", "matplotlib"), ("plotly", "plotly")):
        if importlib.util.find_spec(requirement) is not None:
            names.append(name)
    names.append("recording")
    return names


def current() -> str:
    """The name of the active backend."""
    return _current


def use(name: str) -> str:
    """Set the process-wide default backend; returns the previous name."""
    if name not in _MODULES:
        raise ValueError(f"Unknown backend {name!r}. Known: {sorted(_MODULES)}")
    global _current
    previous, _current = _current, name
    return previous


@contextlib.contextmanager
def backend(name: str) -> Iterator[str]:
    """Use ``name`` as the backend for the duration of a block."""
    previous = use(name)
    try:
        yield name
    finally:
        use(previous)


def _module(name: str | None) -> ModuleType:
    name = name or _current
    if name not in _MODULES:
        raise ValueError(f"Unknown backend {name!r}. Known: {sorted(_MODULES)}")
    if name not in _loaded:
        requirement = _REQUIRES.get(name)
        if requirement is not None and importlib.util.find_spec(requirement) is None:
            raise ImportError(
                f"The {name!r} backend needs {requirement}. "
                f"Install it, or pick another backend: {available()}"
            )
        _loaded[name] = importlib.import_module(_MODULES[name], __package__)
    return _loaded[name]


def figure(
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
    backend: str | None = None,
) -> Figure:
    """Create a figure with a ``rows`` x ``cols`` grid of axes.

    Args:
        rows, cols: Grid shape.
        figsize: Size in inches. Plotly converts it to pixels via ``style.dpi``.
        height_ratios: Relative row heights, top to bottom.
        width_ratios: Relative column widths, left to right.
        sharex, sharey: Share axis ranges across the grid.
        style: Style override; defaults to :func:`get_style`.
        constrained_layout: matplotlib layout engine; ignored by plotly.
        backend: Backend name; defaults to the active one.
    """
    return _module(backend).make_figure(
        rows,
        cols,
        figsize=figsize,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        sharex=sharex,
        sharey=sharey,
        style=style,
        constrained_layout=constrained_layout,
    )


def wrap(obj: Any, *, style: PlotStyle | None = None, backend: str | None = None) -> Axes | Figure:
    """Adopt an existing native object as an :class:`Axes` or :class:`Figure`.

    This is what keeps legacy ``ax=`` keywords working: a caller holding a raw
    ``plt.Axes`` can hand it to a plot function that now speaks this API.
    Already-wrapped objects pass through unchanged.
    """
    if isinstance(obj, Axes | Figure):
        return obj
    if backend is None:
        # Identify by type so wrap() works regardless of the active backend.
        for name in ("matplotlib", "plotly"):
            module = _loaded.get(name)
            if module is None and importlib.util.find_spec(_REQUIRES[name]) is None:
                continue
            with contextlib.suppress(TypeError):
                return _module(name).wrap(obj, style=style)
        raise TypeError(f"No backend can wrap {type(obj).__name__}")
    return _module(backend).wrap(obj, style=style)
