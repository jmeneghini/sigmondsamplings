"""Shared helpers for the sigmondsamplings plot frontends."""

from __future__ import annotations

import warnings
from typing import Any

import slat.plotting as slp

__all__ = ["latex_label", "require_backend", "resolve_axes"]


def resolve_axes(
    axes: slp.Axes | None = None,
    ax: Any = None,
    *,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    style: slp.PlotStyle | None = None,
) -> slp.Axes:
    """Return the Axes to draw on, creating a figure when none was given.

    ``ax`` is the pre-backend spelling and is accepted for compatibility: a raw
    ``matplotlib.axes.Axes`` handed in that way is wrapped transparently.
    """
    if ax is not None:
        if axes is not None:
            raise TypeError("Pass either `axes` or the legacy `ax`, not both")
        warnings.warn(
            "The `ax` argument is deprecated; pass `axes` instead. "
            "A raw matplotlib Axes is still accepted and will be wrapped.",
            DeprecationWarning,
            stacklevel=3,
        )
        axes = ax
    if axes is not None:
        return slp.wrap(axes, style=style)
    return slp.figure(figsize=figsize, style=style, backend=backend).axes()


def require_backend(axes: slp.Axes, name: str, what: str) -> None:
    """Raise unless ``axes`` uses backend ``name``.

    For plots that are genuinely single-backend rather than merely unported.
    """
    if axes.backend != name:
        raise slp.UnsupportedFeature(
            f"{what} is {name}-only; the active backend is {axes.backend!r}"
        )


def latex_label(text: str | None) -> str | None:
    """Wrap a bare LaTeX fragment in ``$...$``."""
    return slp.ensure_math(text)
