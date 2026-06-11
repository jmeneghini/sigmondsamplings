"""
toml_config.py
--------------
Shared TOML configuration utilities for the Sigmond and Luscher Analysis Toolkit.

Provides the low-level ``load_toml``/``load_toml_file``/``dump_toml`` helpers
(including the ``false`` <-> ``None`` sentinel used because TOML has no null
type) and :class:`TomlConfigModel`, a Pydantic base that adds TOML
round-tripping under a canonical section tag. These are consumed by downstream
packages (e.g. ``kbfit``) so config-serialization logic lives in one place.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, get_args

import tomlkit
from pydantic import BaseModel, model_validator

__all__ = [
    "load_toml",
    "load_toml_file",
    "dump_toml",
    "coerce_false_to_none",
    "TomlConfigModel",
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


def _none_to_false(obj: Any) -> Any:
    # TOML has no null type; the configs encode ``None`` as ``false`` so a
    # field whose default is *not* ``None`` (e.g. ``fast_zeta``) can still be
    # explicitly disabled and round-trip. ``coerce_false_to_none`` is the
    # inverse, applied on load.
    if isinstance(obj, dict):
        return {key: _none_to_false(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_none_to_false(value) for value in obj]
    return False if obj is None else obj


def dump_toml(
    data: dict,
    dest: str | bytes | PathLike | None = None,
    *,
    table: str | None = None,
) -> str:
    """
    Serialize a plain mapping to a TOML string (and optionally write it).

    Counterpart to :func:`load_toml`. ``None`` values are encoded as ``false``
    (recursively) because TOML has no null type; :func:`coerce_false_to_none`
    reverses this on load for nullable, non-boolean fields.

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
    """
    payload = {table: dict(data)} if table is not None else dict(data)
    text = tomlkit.dumps(_none_to_false(payload))
    if dest is not None:
        Path(dest).write_text(text)
    return text


def _allows_none_excludes_bool(annotation: Any) -> bool:
    # True for unions like ``str | None`` / ``FastZetaConfig | None`` but not for
    # a plain ``bool`` field, so the ``false`` sentinel is only re-read as
    # ``None`` where ``None`` is actually a valid value.
    args = get_args(annotation)
    return bool(args) and type(None) in args and bool not in args


def coerce_false_to_none(cls, data: Any) -> Any:
    """
    Pydantic ``mode="before"`` helper: map the ``false`` TOML sentinel to ``None``.

    Intended for use in a ``model_validator(mode="before")`` on configs loaded
    from TOML (see :func:`dump_toml`). Only fields whose annotation admits
    ``None`` and is not a plain ``bool`` are converted, leaving genuine boolean
    fields untouched. Non-dict inputs are returned unchanged.
    """
    if not isinstance(data, dict):
        return data
    fields = getattr(cls, "model_fields", {})
    out = dict(data)
    for name, field in fields.items():
        alias = getattr(field, "alias", None)
        key = name if name in out else (alias if alias and alias in out else None)
        if key is not None and out[key] is False and _allows_none_excludes_bool(field.annotation):
            out[key] = None
    return out


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

    Also installs the ``false``-sentinel ``None`` coercion (see
    :func:`coerce_false_to_none`) shared by every TOML-loaded config.
    """

    # Default section for to_toml/from_toml; ``None`` means write/read flat.
    __toml_tag__: ClassVar[str | None] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_false_to_none(cls, data: Any) -> Any:
        # Accept the TOML ``false`` sentinel as ``None`` for nullable fields.
        return coerce_false_to_none(cls, data)

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
        self, dest: str | bytes | PathLike | None = None, *, table: Any = USE_TOML_TAG
    ) -> str:
        """
        Serialize to a TOML string, optionally writing it to *dest*.

        With the default ``table``, the document is nested under the model's
        ``__toml_tag__`` (flat if that is ``None``). Pass an explicit section
        name to override, or ``None`` to force a flat document. ``None`` fields
        are written as the ``false`` sentinel (TOML has no null) and read back
        as ``None`` by :meth:`from_toml`.
        """
        tag = type(self).__toml_tag__ if table is USE_TOML_TAG else table
        return dump_toml(self.model_dump(mode="json"), dest, table=tag)
