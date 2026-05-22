"""Color and marker palettes for sigmondsamplings plots.

Mirrors the design used in kbfit: a small package-level ``COLORS`` /
``MARKERS`` list and an :class:`IndexedCycle` helper for cycling through them
with stateful access (peek current, save / restore state).

Plotters should resolve their working palette via ``rcparams`` so that callers
can override globally::

    sigmondsamplings.rc["plot.colors"] = ["#000000", "#ff0000"]

When the rc value is ``None`` (the default), the palettes defined here are
used.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = ["COLORS", "MARKERS", "IndexedCycle"]

T = TypeVar("T")


COLORS: list[str] = [
    "#d60000",  # Red
    "#8c3bff",  # Purple
    "#018700",  # Green
    "#00acc6",  # Cyan
    "#ffa52f",  # Orange
    "#6b004f",  # Dark Plum
    "#97ff00",  # Lime
    "#ff7ed1",  # Pink
    "#5e56ff",  # Blue-Purple
    "#c500ff",  # Magenta
]
COLORS += COLORS  # double up so simple sequential plots don't wrap visibly


MARKERS: list[str] = [
    "o",
    "s",
    "D",
    "v",
    "^",
    "<",
    ">",
    "o",
    "s",
    "D",
    "v",
    "^",
    "*",
    "x",
    "+",
    "o",
    "s",
    "D",
    "v",
    "^",
    "*",
    "x",
    "+",
]
MARKERS += MARKERS


class IndexedCycle(Iterator[T]):
    """A bounded, restartable cycle with stateful access.

    Unlike :func:`itertools.cycle`, this exposes the current item via
    :meth:`get_current` without advancing, and lets callers snapshot / restore
    the position via :meth:`get_state` / :meth:`set_state`.
    """

    __slots__ = ("items", "index")

    def __init__(self, iterable: Iterable[T]):
        self.items: list[T] = list(iterable)
        if not self.items:
            raise ValueError("IndexedCycle requires a non-empty iterable")
        self.index: int = 0

    def __next__(self) -> T:
        value = self.items[self.index]
        self.index = (self.index + 1) % len(self.items)
        return value

    def __iter__(self) -> IndexedCycle[T]:
        return self

    def get_current(self) -> T:
        return self.items[self.index]

    def get_state(self) -> int:
        return self.index

    def set_state(self, index: int) -> None:
        self.index = index % len(self.items)

    def __len__(self) -> int:
        return len(self.items)
