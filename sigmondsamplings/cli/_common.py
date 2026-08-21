"""Shared option definitions and helpers for the ``ss`` CLI write commands.

These factories keep the ``--group`` / ``--in-group`` / ``--out-group`` /
``--overwrite`` options (and the input/output guards) defined once instead of
repeated across ``convert`` / ``combine`` / ``edit``. Each factory returns
a fresh ``typer.Option`` so it can be used directly as a parameter default.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..io.loader import DEFAULT_GROUP
from ..io.writer import SigmondWriter


def read_group_option():
    """``--group``: the HDF5 root group to read."""
    return typer.Option(None, "--group", help="HDF5 root group to read (default: auto-detect).")


def in_group_option():
    """``--in-group``: which root group to read from a multi-group HDF5 input."""
    return typer.Option(
        None,
        "--in-group",
        help="Root group to read from a multi-group HDF5 input (default: auto-detect).",
    )


def out_group_option(default: str | None = DEFAULT_GROUP):
    """``--out-group``: the root group to write the output under."""
    return typer.Option(default, "--out-group", help="HDF5 root group to write the output under.")


def overwrite_option():
    """``--overwrite`` / ``-f``: replace an existing output file."""
    return typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite (and back up) an existing output file.",
    )


def require_input(path: str | Path) -> Path:
    """Return ``path`` as a Path, raising a CLI error if it does not exist."""
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"Input file {p} does not exist")
    return p


def guard_output(input_file: str | Path, output_file: str | Path, overwrite: bool) -> Path:
    """Resolve the HDF5 output path and reject an existing file unless overwriting."""
    out = SigmondWriter.hdf5_output_path(input_file, output_file)
    if out.exists() and not overwrite:
        raise typer.BadParameter(
            f"Output file {out} already exists. Use --overwrite/-f to overwrite."
        )
    return out
