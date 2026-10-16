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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Axes, Figure, FigureSpec
from .style import PlotStyle

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

    # Each hook records its mark under the public name, with every argument
    # resolved: defaults appear in ``Mark.kwargs`` as the backends receive them.

    def _line(self, x, y, **kwargs) -> None:
        self._record("line", x, y, **kwargs)

    def _points(self, x, y, **kwargs) -> None:
        self._record("points", x, y, **kwargs)

    def _errorbar(self, x, y, **kwargs) -> None:
        self._record("errorbar", x, y, **kwargs)

    def _band(self, x, lo, hi, **kwargs) -> None:
        self._record("band", x, lo, hi, **kwargs)

    def _bar(self, x, height, **kwargs) -> None:
        self._record("bar", x, height, **kwargs)

    def _polygon(self, x, y, **kwargs) -> None:
        self._record("polygon", x, y, **kwargs)

    def _hlines(self, y, xmin, xmax, **kwargs) -> None:
        self._record("hlines", y, xmin, xmax, **kwargs)

    def _vlines(self, x, ymin, ymax, **kwargs) -> None:
        self._record("vlines", x, ymin, ymax, **kwargs)

    def _axhline(self, y, **kwargs) -> None:
        self._record("axhline", y, **kwargs)

    def _axvline(self, x, **kwargs) -> None:
        self._record("axvline", x, **kwargs)

    def _heatmap(self, values, **kwargs) -> None:
        self._record("heatmap", values, **kwargs)

    def _text(self, x, y, s, **kwargs) -> None:
        self._record("text", x, y, s, **kwargs)

    def _contour(self, x, y, z, **kwargs) -> None:
        self._record("contour", x, y, z, **kwargs)

    def _set(self, **kwargs) -> None:
        self._record("set", **kwargs)

    def _limits(self, axis, explicit_only):
        for mark in reversed(self.native):
            if mark.kind == "set" and mark.kwargs.get(f"{axis}lim") is not None:
                return tuple(mark.kwargs[f"{axis}lim"])
        return None

    def _ticks(self, axis, positions, labels, **kwargs) -> None:
        self._record("ticks", axis, positions, labels, **kwargs)

    def _grid(self, show, **kwargs) -> None:
        self._record("grid", show, **kwargs)

    def _legend(self, show, **kwargs) -> None:
        self._record("legend", show, **kwargs)

    def point_size(self, points: float, axis="y", *, units="data") -> float:
        self._record("point_size", points, axis=axis, units=units)
        return 0.0


class RecordingFigure(Figure):
    """A :class:`~slat.plotting.base.Figure` that records instead of drawing."""

    backend = "recording"

    def __init__(self, rows: int, cols: int, style: PlotStyle):
        self.marks: list[Mark] = []
        self._style = style
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

    def suptitle(self, text: str, *, fontsize: float | None = None) -> None:
        self.title = text

    def set_size(self, width: float, height: float) -> None:
        self._style = self._style.replace(figsize=(width, height))

    def save(self, path: str | Path, *, dpi: float | None = None) -> None:
        self.saved.append(Path(path))

    def embed_html(self, *, dpi: float | None = None) -> str:
        return f"<!-- recording backend: {len(self.marks)} marks -->"

    def show(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_figure(spec: FigureSpec) -> RecordingFigure:
    """Create a recording figure with a ``spec.rows`` x ``spec.cols`` grid."""
    return RecordingFigure(spec.rows, spec.cols, spec.style)


def wrap(obj, *, style: PlotStyle | None = None):
    """Pass through an already-recording Axes or Figure."""
    if isinstance(obj, RecordingAxes | RecordingFigure):
        return obj
    raise TypeError(f"Cannot wrap {type(obj).__name__} with the recording backend")
