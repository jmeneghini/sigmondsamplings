"""Global runtime configuration for sigmondsamplings, modelled after matplotlib's rcParams.

Designed to look and feel like ``kbfit.rcparams``, with a different set of keys
tailored to sigmondsamplings plotting and the ``KnownEnsembles`` database.

Usage
-----
Read / write individual keys::

    import sigmondsamplings as sg
    sg.rc["plot.confidence_level"] = 0.95
    sg.rc["plot.central_metric"] = "full_sample_value"

Temporarily override with a context manager::

    with sg.rc_context({"plot.confidence_level": 0.95}):
        fig, ax = sg.plot_fit_result(...)

Load from a TOML file::

    sg.rc_file("~/.sigmondsamplings/config.toml")

Reset to package defaults::

    sg.rc_defaults()

Auto-loading
------------
On package import, sigmondsamplings looks for ``$SIGMONDSAMPLINGS_RC`` first,
then ``~/.sigmondsamplings/config.toml``, and applies whichever it finds.
Writes via :func:`rc_save` round-trip back to that same file so settings
persist across sessions.

Available keys
--------------
plot.colors             None  (None → ``colors.COLORS``)
plot.markers            None  (None → ``colors.MARKERS``)
plot.central_metric     "full_sample_value"   SigmondSampling attr used for the central point/line
plot.error_metric       "error"               'error' or 'confidence_interval'
plot.confidence_level   0.68
plot.metrics_include    None  (None → all)    Iterable[str] of metric names to render
plot.metrics_exclude    ()                    Iterable[str] of metric names to skip
ensembles.xml_file      None                  Path to the KnownEnsembles XML database
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - older Python
    import tomli as tomllib

__all__ = [
    "rc",
    "rc_context",
    "rc_defaults",
    "rc_file",
    "rc_save",
    "user_config_path",
    "VALID_CENTRAL_METRICS",
    "VALID_ERROR_METRICS",
]


# Valid choices for metric-style keys. Tested in __setitem__ below so a typo
# fails loudly at assignment, not deep inside a plot call.
VALID_CENTRAL_METRICS: frozenset[str] = frozenset(
    {
        "full_sample_value",
        "mean",
        "bias_corrected_mean",
        "median",
    }
)
VALID_ERROR_METRICS: frozenset[str] = frozenset({"error", "confidence_interval"})


_DEFAULTS: dict[str, Any] = {
    # ── Palette ───────────────────────────────────────────────────────────
    "plot.colors": None,  # None → colors.COLORS
    "plot.markers": None,  # None → colors.MARKERS
    # ── Central value / error semantics ───────────────────────────────────
    "plot.central_metric": "full_sample_value",
    "plot.error_metric": "error",
    "plot.confidence_level": 0.68,
    # ── Metric selection (for multi-panel summary plots) ─────────────────
    "plot.metrics_include": None,  # None → all
    "plot.metrics_exclude": (),
    # ── Ensembles ─────────────────────────────────────────────────────────
    "ensembles.xml_file": None,
}


_VALIDATORS: dict[str, Any] = {
    "plot.central_metric": lambda v: v in VALID_CENTRAL_METRICS,
    "plot.error_metric": lambda v: v in VALID_ERROR_METRICS,
    "plot.confidence_level": lambda v: isinstance(v, (int, float)) and 0.0 < v < 1.0,
}


_DEFAULT_CONFIG_DIR = Path.home() / ".sigmondsamplings"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.toml"
_LEGACY_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config"  # old plain-text XML path


def user_config_path() -> Path:
    """Return the user-level config file path.

    Honors ``$SIGMONDSAMPLINGS_RC`` if set, otherwise
    ``~/.sigmondsamplings/config.toml``.
    """
    env = os.environ.get("SIGMONDSAMPLINGS_RC")
    if env:
        return Path(env).expanduser()
    return _DEFAULT_CONFIG_FILE


class RcParams(dict):
    """Dict subclass that validates keys against the known defaults."""

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in _DEFAULTS:
            raise KeyError(
                f"Unknown sigmondsamplings rc key: {key!r}. Valid keys:\n  "
                + "\n  ".join(sorted(_DEFAULTS))
            )
        validator = _VALIDATORS.get(key)
        if validator is not None and value is not None and not validator(value):
            raise ValueError(f"Invalid value {value!r} for rc key {key!r}")
        super().__setitem__(key, value)

    def reset_to_defaults(self) -> None:
        self.clear()
        self.update(deepcopy(_DEFAULTS))

    def __repr__(self) -> str:
        lines = [f"sigmondsamplings.rc  ({len(self)} keys)"]
        for k in sorted(self):
            lines.append(f"  {k}: {self[k]!r}")
        return "\n".join(lines)


#: Module-level singleton — import and mutate directly.
rc = RcParams(deepcopy(_DEFAULTS))


@contextmanager
def rc_context(overrides: dict[str, Any]) -> Iterator[None]:
    """Context manager for temporary rc overrides."""
    saved = {k: rc[k] for k in overrides}
    try:
        for k, v in overrides.items():
            rc[k] = v
        yield
    finally:
        for k, v in saved.items():
            rc[k] = v


def rc_defaults() -> None:
    """Reset all rc parameters to their package defaults."""
    rc.reset_to_defaults()


def _flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a TOML dict-of-sections into dotted ``rc`` keys."""
    flat: dict[str, Any] = {}
    for section, entries in data.items():
        if not isinstance(entries, dict):
            raise ValueError(
                f"Expected a TOML table for section [{section!r}], got {type(entries).__name__}"
            )
        for key, value in entries.items():
            dotted = f"{section}.{key}"
            # Match the tuple defaults so equality / hashing stays predictable.
            if isinstance(value, list):
                value = tuple(value)
            flat[dotted] = value
    return flat


def rc_file(path: str | Path) -> None:
    """Load rc parameters from a TOML file."""
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for dotted, value in _flatten_toml(data).items():
        rc[dotted] = value


def rc_save(path: str | Path | None = None) -> Path:
    """Persist any non-default rc values to a TOML file.

    Args:
        path: Destination. Defaults to :func:`user_config_path`.

    Returns:
        The path written to.
    """
    target = Path(path).expanduser() if path is not None else user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Group dotted keys into [section]\nkey = value blocks. Only emit keys that
    # differ from the defaults to keep the file readable.
    sections: dict[str, dict[str, Any]] = {}
    for key, value in rc.items():
        if value == _DEFAULTS.get(key):
            continue
        section, _, suffix = key.partition(".")
        sections.setdefault(section, {})[suffix] = value

    lines: list[str] = []
    for section in sorted(sections):
        lines.append(f"[{section}]")
        for suffix in sorted(sections[section]):
            lines.append(f"{suffix} = {_toml_literal(sections[section][suffix])}")
        lines.append("")

    target.write_text("\n".join(lines).rstrip() + "\n")
    return target


def _toml_literal(value: Any) -> str:
    """Render a Python value as a TOML literal.

    Handles the subset we actually store: bool, int, float, str, Path, tuple,
    list. Falls back to ``repr`` for anything else (and lets TOML reject it on
    the next round-trip).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        return _quote_str(str(value))
    if isinstance(value, str):
        return _quote_str(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    return _quote_str(repr(value))


def _quote_str(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _maybe_migrate_legacy_config() -> None:
    """One-shot migration from ``~/.sigmondsamplings/config`` (plain XML path)
    to the new TOML config. Idempotent — if the TOML already exists or the
    legacy file doesn't, this is a no-op.
    """
    if _DEFAULT_CONFIG_FILE.exists() or not _LEGACY_CONFIG_FILE.exists():
        return
    try:
        xml_path = _LEGACY_CONFIG_FILE.read_text().strip()
    except OSError:
        return
    if not xml_path:
        return
    rc["ensembles.xml_file"] = xml_path
    rc_save(_DEFAULT_CONFIG_FILE)


def _auto_load() -> None:
    """Apply the user's TOML config if present. Called on package import."""
    _maybe_migrate_legacy_config()
    path = user_config_path()
    if path.exists():
        rc_file(path)


_auto_load()
