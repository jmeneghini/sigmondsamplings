"""Ownership scope for figures allocated by a plotting factory."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current_scope: ContextVar = ContextVar("slat_figure_scope", default=None)


@dataclass
class FigureScope:
    figures: list[Any] = field(default_factory=list)
    retained: list[Any] = field(default_factory=list)

    def retain(self, figure):
        self.retained.append(figure)
        return figure


def track_figure(figure):
    scope = _current_scope.get()
    if scope is not None:
        scope.figures.append(figure)
    return figure


@contextmanager
def figure_scope():
    """Close allocated figures on failure; transfer retained figures on success."""
    scope = FigureScope()
    token = _current_scope.set(scope)
    failed = True
    try:
        yield scope
        failed = False
    finally:
        _current_scope.reset(token)
        for figure in scope.figures:
            if failed or not any(figure is retained for retained in scope.retained):
                figure.close()
