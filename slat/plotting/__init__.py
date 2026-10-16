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
from collections.abc import Iterable, Iterator, Sequence
from types import ModuleType
from typing import Any

from .base import Axes, Coords, Figure, FigureSpec, Origin, UnsupportedFeature
from .export import BackendExports, ExportCapability, export_capabilities
from .lifecycle import figure_scope, track_figure
from .recording import Mark, Recorder, record
from .style import PlotStyle, get_style, set_style, style_context
from .text import ensure_math

__all__ = [
    "export_capabilities",
    "BackendExports",
    "ExportCapability",
    "Axes",
    "Coords",
    "Figure",
    "Mark",
    "Origin",
    "PlotStyle",
    "Recorder",
    "UnsupportedFeature",
    "as_figure",
    "available",
    "backend",
    "current",
    "ensure_math",
    "figure",
    "figure_scope",
    "get_style",
    "record",
    "set_style",
    "show",
    "style_context",
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
    return [
        name
        for name in _MODULES
        if name not in _REQUIRES or importlib.util.find_spec(_REQUIRES[name]) is not None
    ]


def current() -> str:
    """The name of the active backend."""
    return _current


def _require(name: str) -> None:
    """Fail unless ``name`` is a known backend whose dependency is installed."""
    if name not in _MODULES:
        raise ValueError(f"Unknown backend {name!r}. Known: {sorted(_MODULES)}")
    requirement = _REQUIRES.get(name)
    if requirement is not None and importlib.util.find_spec(requirement) is None:
        raise ImportError(
            f"The {name!r} backend needs {requirement}. "
            f"Install it, or pick another backend: {available()}"
        )


def use(name: str) -> str:
    """Set the process-wide default backend; returns the previous name.

    An unknown name raises ``ValueError``, a missing dependency ``ImportError``,
    here rather than at the first figure.
    """
    _require(name)
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
    if name not in _loaded:
        # figure(backend=...) and wrap(backend=...) arrive here without use().
        _require(name)
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
        figsize: Size in inches. Plotly lays it out at 100 px per inch.
        height_ratios: Relative row heights, top to bottom.
        width_ratios: Relative column widths, left to right.
        sharex, sharey: Share axis ranges across the grid.
        style: Style override; defaults to :func:`get_style`.
        constrained_layout: matplotlib layout engine; ignored by plotly.
        backend: Backend name; defaults to the active one.
    """
    style = style or get_style()
    if figsize is not None:
        style = style.replace(figsize=tuple(figsize))
    spec = FigureSpec(
        rows=rows,
        cols=cols,
        style=style,
        height_ratios=None if height_ratios is None else tuple(height_ratios),
        width_ratios=None if width_ratios is None else tuple(width_ratios),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=constrained_layout,
    )
    return track_figure(_module(backend).make_figure(spec))


def wrap(obj: Any, *, style: PlotStyle | None = None, backend: str | None = None) -> Axes | Figure:
    """Adopt an existing native object as an :class:`Axes` or :class:`Figure`.

    A caller holding a raw ``plt.Axes`` can hand it as ``axes=`` to a plot
    function that speaks this API. Already-wrapped objects pass through unchanged.
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


def as_figure(obj: Any, *, style: PlotStyle | None = None) -> Figure:
    """The :class:`Figure` behind whatever a plot function returned.

    Accepts an :class:`Axes` (its figure), a :class:`Figure`, or a native
    figure or axes that :func:`wrap` can adopt.
    """
    wrapped = wrap(obj, style=style)
    return wrapped.figure if isinstance(wrapped, Axes) else wrapped


def show(figures: Iterable[Any]) -> None:
    """Display figures, each normalised with :func:`as_figure`.

    ``plt.show()`` displays every open matplotlib figure and blocks, so it is
    called once for all of them, last; every other figure is shown on its own.
    """
    normalised = [as_figure(obj) for obj in figures]
    matplotlib = [fig for fig in normalised if fig.backend == "matplotlib"]
    for fig in normalised:
        if fig.backend != "matplotlib":
            fig.show()
    if matplotlib:
        matplotlib[0].show()
