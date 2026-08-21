"""
toml_config.py
--------------
Shared TOML configuration utilities for the Sigmond and Luscher Analysis Toolkit.

Provides the low-level ``load_toml``/``load_toml_file``/``dump_toml`` helpers
and :class:`TomlConfigModel`, a Pydantic base that adds TOML round-tripping
under a canonical section tag. TOML has no null type, so ``None`` values are
simply omitted on dump (absent key reads back as ``None``); pass
``comment_unset=True`` to advertise unset optionals as commented ``# key =``
lines instead. These are consumed by downstream packages (e.g. ``kbfit``) so
config-serialization logic lives in one place.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import tomlkit
from pydantic import BaseModel, ConfigDict

__all__ = [
    "load_toml",
    "load_toml_file",
    "dump_toml",
    "TomlConfigModel",
    "StrictModel",
    "USE_TOML_TAG",
]


def _select_table(data: dict, table: str | None) -> dict:
    # Optionally return a single named sub-table instead of the whole document.
    if table is None:
        return data
    if table not in data:
        raise KeyError(f"TOML table '{table}' not found in document")
    return data[table]


def load_toml(source: str | bytes, table: str | None = None) -> dict:
    """
    Parse inline TOML text into a plain ``dict``.

    Parameters
    ----------
    source:
        TOML document as a ``str`` (or UTF-8 ``bytes``). Use
        :func:`load_toml_file` to read from a path.
    table:
        Optional top-level key. When given, the corresponding sub-table is
        returned instead of the whole document, letting one document hold the
        configs for several objects under named sections.

    Returns
    -------
    dict
        The parsed TOML mapping (or the selected sub-table).
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return _select_table(tomlkit.loads(source).unwrap(), table)


def load_toml_file(path: str | PathLike, table: str | None = None) -> dict:
    """
    Parse a TOML file into a plain ``dict``.

    File counterpart to :func:`load_toml`; *table* selects a named sub-table.
    A missing file raises :class:`FileNotFoundError`.
    """
    with open(path, "rb") as f:
        return _select_table(tomlkit.load(f).unwrap(), table)


def _to_toml_value(value: Any, *, comment_unset: bool) -> Any:
    # Convert a non-``None`` JSON value into its tomlkit counterpart, recursing
    # into tables and arrays-of-tables so nested ``None`` is handled uniformly.
    if isinstance(value, dict):
        return _to_toml_container(value, comment_unset=comment_unset)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, dict) for item in value):
            aot = tomlkit.aot()
            for item in value:
                aot.append(_to_toml_container(item, comment_unset=comment_unset))
            return aot
        # Scalar array: TOML has no null, and silently dropping a ``None``
        # element would change the array's length/positions, so fail loudly.
        if any(item is None for item in value):
            raise ValueError("TOML arrays cannot hold None")
        return list(value)
    return value


def _to_toml_container(mapping: dict, *, comment_unset: bool, _top: bool = False):
    # Build a tomlkit document/table from a plain mapping. TOML has no null
    # type, so ``None`` values are dropped; with ``comment_unset`` they are
    # emitted as commented ``# key =`` lines instead, so the document advertises
    # which optional keys exist while still reading back as ``None`` (the key is
    # absent to the parser).
    container = tomlkit.document() if _top else tomlkit.table()
    for key, value in mapping.items():
        if value is None:
            if comment_unset:
                container.add(tomlkit.comment(f"{key} ="))
            continue
        container[key] = _to_toml_value(value, comment_unset=comment_unset)
    return container


def dump_toml(
    data: dict,
    dest: str | bytes | PathLike | None = None,
    *,
    table: str | None = None,
    comment_unset: bool = False,
) -> str:
    """
    Serialize a plain mapping to a TOML string (and optionally write it).

    Counterpart to :func:`load_toml`. TOML has no null type, so ``None`` values
    are omitted; an absent key reads back as ``None`` (the field default).

    Parameters
    ----------
    data:
        JSON-compatible mapping, typically ``model.model_dump(mode="json")``.
    dest:
        Optional path to write the TOML text to. The string is always returned.
    table:
        Optional top-level key to nest the document under, mirroring the
        ``table`` argument of :func:`load_toml` so several configs can share one
        file under named sections.
    comment_unset:
        When ``True``, ``None`` values are written as commented ``# key =``
        lines instead of being dropped, so the document advertises which
        optional keys exist (used for the resolved-run snapshot). The commented
        line is absent to the parser, so it still reads back as ``None``.
    """
    payload = {table: dict(data)} if table is not None else dict(data)
    document = _to_toml_container(payload, comment_unset=comment_unset, _top=True)
    text = tomlkit.dumps(document)
    if dest is not None:
        Path(dest).write_text(text)
    return text


class _UseTomlTag:
    """Sentinel for the default ``table`` argument: use the model's tag."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "USE_TOML_TAG"


# Default ``table`` for TomlConfigModel.to_toml/from_toml: resolve to the
# model's ``__toml_tag__``. Distinct from ``table=None`` (flat, no section).
USE_TOML_TAG = _UseTomlTag()


class TomlConfigModel(BaseModel):
    """
    Pydantic base adding TOML round-tripping under a canonical section tag.

    Subclasses set ``__toml_tag__`` to the section name they write to and read
    from by default (e.g. ``"minimizer"``), so several configs can share one
    file under named sections::

        class MinimizerInfo(TomlConfigModel):
            __toml_tag__ = "minimizer"

    ``to_toml`` nests the document under that tag; ``from_toml`` looks for it,
    falling back to the whole document when the tag is absent (so a standalone,
    untagged file still loads). Pass an explicit ``table`` to target a different
    section, or ``table=None`` to write/read a flat document with no nesting.
    A subclass that leaves ``__toml_tag__`` as ``None`` round-trips flat.

    TOML has no null type, so ``None`` fields are omitted on dump and read back
    as ``None`` from their absence; nullable fields should therefore default to
    ``None`` (a non-``None`` default cannot round-trip an explicit ``None``).
    """

    # Default section for to_toml/from_toml; ``None`` means write/read flat.
    __toml_tag__: ClassVar[str | None] = None
    __toml_dict__: ClassVar[dict | None] = None

    @classmethod
    def _validate_section(cls, doc: dict, table: Any):
        # Pick the section *table* selects, then validate it into an instance.
        # ``USE_TOML_TAG`` uses ``__toml_tag__`` if present, else the whole doc;
        # ``None`` reads flat; any other value names an explicit sub-table.
        if table is USE_TOML_TAG:
            tag = cls.__toml_tag__
            section = doc[tag] if tag is not None and tag in doc else doc
        else:
            section = _select_table(doc, table)
        return cls.model_validate(section)

    @classmethod
    def from_toml(cls, source: str | bytes, table: Any = USE_TOML_TAG):
        """
        Create an instance from an inline TOML string (or UTF-8 bytes).

        With the default ``table``, the model's ``__toml_tag__`` section is used
        if present, else the whole document. Pass an explicit section name, or
        ``None`` to read the document flat. Use :meth:`from_file` to read a path.
        """
        return cls._validate_section(load_toml(source), table)

    @classmethod
    def from_file(cls, path: str | PathLike, table: Any = USE_TOML_TAG):
        """
        Create an instance from a TOML file. *table* behaves as in :meth:`from_toml`.
        """
        return cls._validate_section(load_toml_file(path), table)

    def to_toml(
        self,
        dest: str | bytes | PathLike | None = None,
        *,
        table: Any = USE_TOML_TAG,
        comment_unset: bool = False,
    ) -> str:
        """
        Serialize to a TOML string, optionally writing it to *dest*.

        With the default ``table``, the document is nested under the model's
        ``__toml_tag__`` (flat if that is ``None``). Pass an explicit section
        name to override, or ``None`` to force a flat document. ``None`` fields
        are omitted (TOML has no null) and read back as ``None`` by
        :meth:`from_toml`; pass ``comment_unset=True`` to instead advertise them
        as commented ``# key =`` lines.
        """
        tag = type(self).__toml_tag__ if table is USE_TOML_TAG else table
        # Dump *with* None (no exclude_none): ``comment_unset`` needs to see
        # which keys are unset to advertise them, so None-stripping is a
        # rendering choice made by dump_toml, not a model-level exclusion.
        out_dict = self.__toml_dict__ or self.model_dump(mode="json")
        return dump_toml(
            out_dict, dest, table=tag, comment_unset=comment_unset
        )


class StrictModel(TomlConfigModel):
    """Shared base for all project-layer configs: unknown keys are load-time
    errors, so typos in hand-edited TOML fail with a field-level message."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
