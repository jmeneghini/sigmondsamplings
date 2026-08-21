"""Label text handling across backends.

matplotlib renders ``$...$`` with mathtext (or a real TeX engine). plotly
renders it with MathJax, which works in titles and axis titles but is
unreliable in *tick labels and legend entries* -- exactly where irrep names
live in spectrum plots. Those two places therefore fall back to the Unicode
approximation from :func:`slat.constants.latex_to_unicode`.
"""

from __future__ import annotations

from ..constants import latex_to_unicode

__all__ = ["ensure_math", "has_math", "strip_math", "to_display"]


def has_math(text: str | None) -> bool:
    """Whether ``text`` contains a ``$...$`` math span."""
    return bool(text) and text.count("$") >= 2


def strip_math(text: str | None) -> str | None:
    """Remove surrounding ``$`` delimiters, leaving the LaTeX body."""
    if text is None:
        return None
    return text.strip().strip("$")


def ensure_math(text: str | None) -> str | None:
    """Wrap ``text`` in ``$...$`` unless it already contains a math span."""
    if text is None or has_math(text):
        return text
    return f"${text}$"


def to_display(text: str | None, *, mathtext: bool) -> str | None:
    """Render a label for a backend.

    With ``mathtext=True`` the string passes through untouched, for a backend
    that renders ``$...$`` in this position. With ``mathtext=False`` any LaTeX
    is flattened to Unicode.
    """
    if text is None:
        return None
    if mathtext:
        return text
    return latex_to_unicode(text)
