"""Recording backend, for testing plot code without rendering anything.

Every mark and configuration call is appended to a list instead of drawn, so
tests can assert on the *structure* of a plot with neither matplotlib nor
plotly installed::

    with plotting.record() as rec:
        plot_fit_result(x, y, model)

    assert rec.kinds() == ["band", "line", "line", "errorbar"]
    assert rec.first("band").kwargs["alpha"] == 0.22

This catches the class of bug where a frontend silently stops emitting an
element -- which a smoke test that only checks "it didn't raise" will miss.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import FEATURES, Axes, Figure
from .style import PlotStyle, get_style

__all__ = ["Mark", "Recorder", "RecordingAxes", "RecordingFigure", "make_figure", "record", "wrap"]


@dataclass
class Mark:
    """One recorded call."""

    kind: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    cell: tuple[int, int] = (0, 0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        keys = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items() if v is not None)
        return f"Mark({self.kind}, cell={self.cell}, {keys})"


class Recorder:
    """Collects :class:`Mark` objects from every figure created while active."""

    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self.figures: list[RecordingFigure] = []

    def add(self, mark: Mark) -> None:
        self.marks.append(mark)

    def kinds(self) -> list[str]:
        """The ``kind`` of every recorded mark, in order."""
        return [m.kind for m in self.marks]

    def of_kind(self, kind: str) -> list[Mark]:
        """Every recorded mark of a given kind."""
        return [m for m in self.marks if m.kind == kind]

    def first(self, kind: str) -> Mark:
        """The first mark of a given kind. Raises if there is none."""
        for mark in self.marks:
            if mark.kind == kind:
                return mark
        raise AssertionError(f"No {kind!r} mark recorded. Got: {self.kinds()}")

    def count(self, kind: str) -> int:
        """How many marks of a given kind were recorded."""
        return sum(1 for m in self.marks if m.kind == kind)

    def labels(self) -> list[str]:
        """Every non-None ``label`` passed to a mark, in order."""
        return [m.kwargs["label"] for m in self.marks if m.kwargs.get("label") is not None]


_active: Recorder | None = None


@contextlib.contextmanager
def record() -> Iterator[Recorder]:
    """Activate the recording backend and yield the :class:`Recorder`."""
    from . import use

    global _active
    previous_recorder, _active = _active, Recorder()
    previous_backend = use("recording")
    try:
        yield _active
    finally:
        use(previous_backend)
        _active = previous_recorder


class RecordingAxes(Axes):
    """An :class:`~slat.plotting.base.Axes` that records instead of drawing."""

    backend = "recording"
    supported = FEATURES

    def __init__(self, figure: RecordingFigure, row: int, col: int):
        self._figure = figure
        self._cell = (row, col)

    @property
    def figure(self) -> RecordingFigure:
        return self._figure

    @property
    def native(self):
        """The list of marks recorded on this cell."""
        return [m for m in self._figure.marks if m.cell == self._cell]

    def _record(self, kind: str, *args: Any, **kwargs: Any) -> None:
        mark = Mark(kind, args, kwargs, self._cell)
        self._figure.marks.append(mark)
        if _active is not None:
            _active.add(mark)

    def line(self, x, y, **kwargs) -> None:
        self._record("line", x, y, **kwargs)

    def points(self, x, y, **kwargs) -> None:
        self._record("points", x, y, **kwargs)

    def errorbar(self, x, y, **kwargs) -> None:
        self._record("errorbar", x, y, **kwargs)

    def band(self, x, lo, hi, **kwargs) -> None:
        self._record("band", x, lo, hi, **kwargs)

    def bar(self, x, height, **kwargs) -> None:
        self._record("bar", x, height, **kwargs)

    def polygon(self, x, y, **kwargs) -> None:
        self._record("polygon", x, y, **kwargs)

    def hlines(self, y, xmin, xmax, **kwargs) -> None:
        self._record("hlines", y, xmin, xmax, **kwargs)

    def vlines(self, x, ymin, ymax, **kwargs) -> None:
        self._record("vlines", x, ymin, ymax, **kwargs)

    def axhline(self, y, **kwargs) -> None:
        self._record("axhline", y, **kwargs)

    def axvline(self, x, **kwargs) -> None:
        self._record("axvline", x, **kwargs)

    def heatmap(self, values, **kwargs) -> None:
        self._record("heatmap", values, **kwargs)

    def text(self, x, y, s, **kwargs) -> None:
        self._record("text", x, y, s, **kwargs)

    def contour(self, x, y, z, **kwargs) -> None:
        self._record("contour", x, y, z, **kwargs)

    def set(self, **kwargs) -> None:
        self._record("set", **kwargs)

    def ticks(self, axis, positions, labels=None, **kwargs) -> None:
        self._record("ticks", axis, positions, labels, **kwargs)

    def grid(self, show=True, **kwargs) -> None:
        self._record("grid", show, **kwargs)

    def legend(self, show=True, **kwargs) -> None:
        self._record("legend", show, **kwargs)

    def point_size(self, points: float, axis="y", *, units="data") -> float:
        self._record("point_size", points, axis=axis, units=units)
        return 0.0


class RecordingFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` that records instead of drawing."""

    backend = "recording"

    def __init__(self, rows: int, cols: int, style: PlotStyle):
        self.marks: list[Mark] = []
        self.style = style
        self._shape = (rows, cols)
        self._axes = [[RecordingAxes(self, r, c) for c in range(cols)] for r in range(rows)]
        self.hidden: list[tuple[int, int]] = []
        self.title: str | None = None
        self.saved: list[Path] = []
        if _active is not None:
            _active.figures.append(self)

    @property
    def native(self):
        return self.marks

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def axes(self, row: int = 0, col: int = 0) -> RecordingAxes:
        return self._axes[row][col]

    def hide(self, row: int, col: int) -> None:
        self.hidden.append((row, col))

    def suptitle(self, text: str, **kwargs: Any) -> None:
        self.title = text

    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        self.saved.append(Path(path))

    def embed_html(self, *, dpi: float | None = None) -> str:
        return f"<!-- recording backend: {len(self.marks)} marks -->"

    def show(self) -> None:
        pass

    def close(self) -> None:
        pass


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
) -> RecordingFigure:
    """Create a recording figure with a ``rows`` x ``cols`` grid."""
    style = style or get_style()
    if figsize is not None:
        style = style.replace(figsize=figsize)
    return RecordingFigure(rows, cols, style)


def wrap(obj, *, style: PlotStyle | None = None):
    """Pass through an already-recording Axes or Figure."""
    if isinstance(obj, RecordingAxes | RecordingFigure):
        return obj
    raise TypeError(f"Cannot wrap {type(obj).__name__} with the recording backend")
