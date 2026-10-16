"""Ownership scopes for figures allocated by plotting factories."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

_current_scope: ContextVar = ContextVar("slat_figure_scope", default=None)


def _same(left, right):
    return getattr(left, "native", left) is getattr(right, "native", right)


@dataclass
class FigureScope:
    figures: list[Any] = field(default_factory=list)
    retained: list[Any] = field(default_factory=list)
    parent: Any = None

    def retain(self, figure):
        self.retained.append(figure)
        # Replace a native cleanup handle with its adopted drawing surface.
        for i, owned in enumerate(self.figures):
            if _same(owned, figure):
                self.figures[i] = figure
        return figure


@dataclass
class _NativeFigure:
    native: Any
    closer: Any

    def close(self):
        self.closer(self.native)


def track_native_figure(figure, closer):
    """Own a native allocation before adoption, using its native close function."""
    track_figure(_NativeFigure(figure, closer))
    return figure


def track_figure(figure):
    scope = _current_scope.get()
    if scope is not None and not any(_same(figure, owned) for owned in scope.figures):
        scope.figures.append(figure)
    return figure


def retain_figure(figure):
    scope = _current_scope.get()
    if scope is not None:
        scope.retain(figure)
    return figure


def close_figure(figure):
    """Close once and remove this allocation from all active ownership scopes."""
    scope = _current_scope.get()
    while scope is not None:
        scope.figures[:] = [owned for owned in scope.figures if not _same(owned, figure)]
        scope.retained[:] = [kept for kept in scope.retained if not _same(kept, figure)]
        scope = scope.parent
    figure.close()


@contextmanager
def figure_scope():
    """Close owned allocations on failure; transfer retained ones to the parent.

    Retaining caller-owned axes does not adopt them. Only allocations tracked
    inside the operation are eligible for cleanup or transfer.
    """
    scope = FigureScope(parent=_current_scope.get())
    token = _current_scope.set(scope)
    failed = True
    try:
        yield scope
        failed = False
    finally:
        _current_scope.reset(token)
        for figure in scope.figures:
            keep = not failed and any(_same(figure, retained) for retained in scope.retained)
            if keep:
                if scope.parent is not None:
                    track_figure(figure)
            else:
                figure.close()


def figure_factory(function):
    """Give a public drawing operation one scope, preserving its return type."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from .base import Axes, Figure

        with figure_scope() as scope:
            result = function(*args, **kwargs)
            if isinstance(result, Axes):
                scope.retain(result.figure)
            elif isinstance(result, Figure):
                scope.retain(result)
            return result
    return wrapped
