"""Label text handling across backends.

matplotlib renders ``$...$`` with mathtext (or a real TeX engine). plotly
renders it with MathJax, which works in titles and axis titles but is
unreliable in *tick labels and legend entries* -- exactly where irrep names
live in spectrum plots. Unicode provides a plain-text approximation; HTML
subscript and superscript tags preserve grouped scripts in Plotly ticks.
"""

from __future__ import annotations

import re

from ..constants import latex_to_unicode

__all__ = ["ensure_math", "has_math", "strip_math", "to_display", "to_html"]


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


def to_html(text: str | None) -> str | None:
    """Convert lightweight math labels to HTML, preserving whole script groups.

    Plotly supports ``sub``/``sup`` tags in ticks, including letters such as
    ``g`` that have no Unicode subscript equivalent.
    """
    if text is None or not has_math(text):
        return text
    text = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}", r"\1", text)

    def script(match: re.Match[str]) -> str:
        marker, braced, plain = match.groups()
        tag = "sub" if marker == "_" else "sup"
        value = braced if braced is not None else plain
        return f"<{tag}>{latex_to_unicode(value)}</{tag}>"

    text = re.sub(r"([_^])\s*(?:\{([^{}]*)\}|([^\\{}\s]))", script, text)
    return latex_to_unicode(text)
