"""Write-side commands for the ``ss`` CLI: ``convert`` and ``combine``.

Thin typer wrappers over the library functions in ``sigmondsamplings.io``;
shared option definitions and the input/output guards live in ``_common``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..io.combine import combine_files
from ..io.convert import convert_to_hdf5
from ._common import (
    guard_output,
    in_group_option,
    out_group_option,
    overwrite_option,
    require_input,
)


def convert(
    file: Path = typer.Argument(..., help="Input Sigmond file (.smp, .bins, .fstream, .hdf5)."),
    output: Path = typer.Argument(..., help="Output HDF5 file."),
    in_group: str | None = in_group_option(),
    out_group: str = out_group_option(),
    overwrite: bool = overwrite_option(),
) -> None:
    """Convert a Sigmond file to HDF5, preserving samplings vs bins."""
    require_input(file)
    guard_output(file, output, overwrite)
    try:
        result = convert_to_hdf5(str(file), str(output), in_group=in_group, out_group=out_group)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Converted {file} -> {result}")


def combine(
    files: list[Path] = typer.Argument(..., help="Input Sigmond files (.smp, .fstream, .hdf5)."),
    output: Path = typer.Option(..., "-o", "--output", help="Output HDF5 file."),
    out_group: str = out_group_option(),
    base_path: Path | None = typer.Option(
        None, "--base-path", help="Base directory for resolving relative input paths."
    ),
    overwrite: bool = overwrite_option(),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print detailed progress."),
) -> None:
    """Combine multiple Sigmond files into a single HDF5 file."""
    try:
        result = combine_files(
            [str(f) for f in files],
            str(output),
            group=out_group,
            base_path=str(base_path) if base_path else None,
            verbose=verbose,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Combined {len(files)} files -> {result}")
