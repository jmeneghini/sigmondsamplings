"""Write-side commands for the ``ss`` CLI: ``convert``, ``combine``, ``energy-tag``.

Thin typer wrappers over the library functions in ``sigmondsamplings.io``;
shared option definitions and the input/output guards live in ``_common``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..io.combine import combine_files
from ..io.convert import convert_to_hdf5
from ..io.energy_tag import add_energy_attrs
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


def energy_tag(
    file: Path = typer.Argument(..., help="Input Sigmond samplings file (.smp or .hdf5)."),
    output: Path = typer.Argument(..., help="Output HDF5 file."),
    ni_yml: Path | None = typer.Option(
        None, "--ni-yml", help="PyCalQ YAML with non-interacting pair assignments."
    ),
    ref_particle: str | None = typer.Option(
        None, "--ref-particle", help="Reference particle for reference-mode levels (E/M_ref)."
    ),
    in_group: str | None = in_group_option(),
    out_group: str | None = out_group_option(default=None),
    overwrite: bool = overwrite_option(),
) -> None:
    """Tag energy observables with self-describing attrs (and optional NI pairs)."""
    require_input(file)
    if ni_yml is not None:
        require_input(ni_yml)
    guard_output(file, output, overwrite)
    try:
        result = add_energy_attrs(
            str(file),
            str(output),
            ni_yml=str(ni_yml) if ni_yml else None,
            ref_particle=ref_particle,
            in_group=in_group,
            out_group=out_group,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Tagged energy attrs {file} -> {result}")
