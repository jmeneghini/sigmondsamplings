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

__all__ = ["ensure_math", "has_math", "line_count", "to_html", "to_plotly"]


def has_math(text: str | None) -> bool:
    """Whether ``text`` contains a ``$...$`` math span."""
    return bool(text) and text.count("$") >= 2


def ensure_math(text: str | None) -> str | None:
    """Wrap ``text`` in ``$...$`` unless it already contains a math span."""
    if text is None or has_math(text):
        return text
    return f"${text}$"


def to_html(text: str | None) -> str | None:
    """Convert lightweight math labels to HTML, preserving whole script groups.

    Plotly supports ``sub``/``sup`` tags in ticks, including letters such as
    ``g`` that have no Unicode subscript equivalent.
    """
    if text is None or not has_math(text):
        return text
    # Hold escaped literals out of the script regex, so the ``\_`` in a name
    # such as ``\text{eta\_ref}`` is not read as a subscript.
    literals = {f"\ue000{i}\ue001": ch for i, ch in enumerate("_%&#")}
    for placeholder, ch in literals.items():
        text = text.replace(f"\\{ch}", placeholder)
    text = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}", r"\1", text)

    def script(match: re.Match[str]) -> str:
        marker, braced, plain = match.groups()
        tag = "sub" if marker == "_" else "sup"
        value = braced if braced is not None else plain
        return f"<{tag}>{latex_to_unicode(value)}</{tag}>"

    text = latex_to_unicode(re.sub(r"([_^])\s*(?:\{([^{}]*)\}|([^\\{}\s]))", script, text))
    for placeholder, ch in literals.items():
        text = text.replace(placeholder, ch)
    return text


_DOLLAR = re.compile(r"(?<!\\)\$")
_TEXT_COMMAND = re.compile(r"\\text\s*\{([^{}]*)\}")
_LINE_BREAK = re.compile(r"\n|<br\s*/?>")


def line_count(text: str) -> int:
    """Lines in a label, counting ``\\n`` and ``<br>`` breaks alike."""
    return len(_LINE_BREAK.split(str(text)))


def to_plotly(text: str | None) -> str | None:
    """A title, axis label or annotation for plotly, which renders math with MathJax.

    MathJax typesets a string only when it is one math span, and silently drops
    any plain text around it, so ``"$68\\%$ CI"`` would lose ``" CI"``. Mixed
    strings are therefore merged into a single span with the plain parts in
    ``\\text{}``, where MathJax prints ``%``, ``&``, ``#`` and backslashes
    literally. Two things cannot go in that span, and fall back to
    :func:`to_html` instead:

    * a line break (``\\n`` or ``<br>``): MathJax cannot break a line inside
      math, and shows ``<br>`` as ``\\lt br\\gt``. Each line is converted on its own.
    * a brace in the plain text: an unbalanced one blanks the whole label.

    Newlines become ``<br>``, which is plotly's line break.
    """
    if text is None:
        return None
    lines = _LINE_BREAK.split(str(text))
    if not has_math(str(text)):
        return "<br>".join(lines)
    if len(lines) > 1:
        return "<br>".join(to_html(line) for line in lines)
    parts = _DOLLAR.split(lines[0])
    plain = parts[::2]
    if len(parts) % 2 == 0 or any(ch in part for part in plain for ch in "{}"):
        # An unpaired $, or plain text that would break the merged span.
        return to_html(lines[0])
    pieces = []
    for i, part in enumerate(parts):
        if i % 2:
            # MathJax's text mode prints \_ literally; a bare _ is already literal.
            pieces.append(_TEXT_COMMAND.sub(lambda m: r"\text{" + m[1].replace(r"\_", "_") + "}", part))
        elif part:
            pieces.append(r"\text{" + part + "}")
    return "$" + "".join(pieces) + "$"
