"""The ``ss edit`` command: change an observable set on the way to a new file.

Two front doors onto one core. Flags cover the one-shot case and desugar into an
:class:`~sigmondsamplings.edit_spec.EditSpec` in a fixed canonical order, so flag order
on the command line never changes the result. ``--recipe`` reads that same object from
TOML, which is the only form able to scope two different operations differently in one
pass. ``--save-recipe`` writes back what a flag invocation desugared to.

Selection (``--where``/``--contains``/``--regex``/``--spec``) is the ``ss query`` filter
language, resolved by the same code. A scope says which observables an operation
*touches*; everything else passes through to the output untouched. Only ``--only`` and
``--drop`` change which observables reach the output.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..edit_spec import (
    AddRefOp,
    DropOp,
    EditSpec,
    KeepOp,
    SetOp,
    SetRefOp,
    TagEnergyOp,
)
from ..selection import parse_where_specs
from ._common import (
    guard_output,
    in_group_option,
    out_group_option,
    overwrite_option,
    require_input,
)


def edit(
    file: Path = typer.Argument(..., help="Input Sigmond file (.smp, .bins, .fstream, .hdf5)."),
    output: Path = typer.Argument(..., help="Output HDF5 file."),
    # selection
    where: list[str] | None = typer.Option(
        None, "--where", "-w",
        help="Scope as attr=value. Repeat or use comma values for membership.",
    ),
    contains: str | None = typer.Option(
        None, "--contains", help="Scope to observables whose name contains text."
    ),
    regex: str | None = typer.Option(
        None, "--regex", help="Scope to observables whose name matches regex."
    ),
    spec: Path | None = typer.Option(
        None, "--spec",
        help="Scope to the (psq, irrep, levels) in a spectrum TOML, as written by "
             "`ss query energy --save`.",
    ),
    # membership
    only: bool = typer.Option(
        False, "--only", help="Write only the scope, dropping everything outside it."
    ),
    drop: bool = typer.Option(
        False, "--drop", help="Write everything except the scope."
    ),
    # operations
    tag_energy: bool = typer.Option(
        False, "--tag-energy", help="Interpret observables as energy levels first."
    ),
    ni_yml: Path | None = typer.Option(
        None, "--ni-yml",
        help="PyCalQ YAML with non-interacting pair assignments (implies --tag-energy).",
    ),
    ref_particle: str | None = typer.Option(
        None, "--ref-particle", help="Tag existing reference-mode levels with this particle."
    ),
    add_ref: str | None = typer.Option(
        None, "--add-ref", help="Add E/M_ref levels dividing the scope by this particle's mass."
    ),
    ref_psq: int = typer.Option(
        0, "--ref-psq", min=0, help="Momentum frame of the --add-ref particle."
    ),
    set_: list[str] | None = typer.Option(
        None, "--set", help="Set an attribute on the scope, as attr=value. Repeatable."
    ),
    rename: bool = typer.Option(
        False, "--rename", help="Resync names to their canonical form after --set."
    ),
    # recipe
    recipe: Path | None = typer.Option(
        None, "--recipe", help="Read the edit program from a TOML file instead of flags."
    ),
    save_recipe: Path | None = typer.Option(
        None, "--save-recipe", help="Write the edit program this invocation ran to a TOML file."
    ),
    # io
    in_group: str | None = in_group_option(),
    out_group: str | None = out_group_option(default=None),
    overwrite: bool = overwrite_option(),
) -> None:
    """Edit an observable set: retag, rewrite attrs, add reference levels, prune."""
    from ..io.edit import edit_file

    require_input(file)
    if spec is not None:
        require_input(spec)
    if ni_yml is not None:
        require_input(ni_yml)
    guard_output(file, output, overwrite)

    if recipe is not None:
        require_input(recipe)
        if _flag_ops_given(locals()):
            raise typer.BadParameter(
                "--recipe cannot be combined with the operation flags; "
                "put every op in the TOML."
            )
        edit_spec = EditSpec.from_file(recipe)
        base_dir = recipe.parent
    else:
        edit_spec = _spec_from_flags(
            where=where, contains=contains, regex=regex, spec=spec,
            only=only, drop=drop,
            tag_energy=tag_energy, ni_yml=ni_yml, ref_particle=ref_particle,
            add_ref=add_ref, ref_psq=ref_psq, set_=set_, rename=rename,
        )
        base_dir = Path.cwd()

    if save_recipe is not None:
        edit_spec.to_toml(save_recipe)

    try:
        resolved = edit_spec.resolve(base_dir=base_dir)
        result = edit_file(
            str(file), str(output), resolved,
            in_group=in_group, out_group=out_group, overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Edited {file} -> {result}")


_FLAG_OPS = (
    "where", "contains", "regex", "spec", "only", "drop",
    "tag_energy", "ni_yml", "ref_particle", "add_ref", "set_", "rename",
)


def _flag_ops_given(values: dict) -> bool:
    """Whether any operation/selection flag was used (``--recipe`` forbids them)."""
    return any(values.get(name) for name in _FLAG_OPS)


def _spec_from_flags(
    *,
    where: list[str] | None,
    contains: str | None,
    regex: str | None,
    spec: Path | None,
    only: bool,
    drop: bool,
    tag_energy: bool,
    ni_yml: Path | None,
    ref_particle: str | None,
    add_ref: str | None,
    ref_psq: int,
    set_: list[str] | None,
    rename: bool,
) -> EditSpec:
    """Desugar the flags into an ordered edit program.

    The order is fixed here rather than taken from the command line: interpret, then
    annotate, then derive, then decide membership. Deciding membership last is what
    lets ``--only`` coexist with an operator whose input lives outside the scope.
    """
    if only and drop:
        raise typer.BadParameter("Use either --only or --drop, not both.")
    if rename and not set_:
        raise typer.BadParameter(
            "--rename applies to --set; give at least one --set attr=value."
        )

    try:
        scope = {
            "where": parse_where_specs(where),
            "contains": contains,
            "regex": regex,
            "spec": str(spec) if spec else None,
        }
        attrs = parse_where_specs(set_) if set_ else {}
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    operations = bool(set_ or ref_particle or add_ref)
    membership = only or drop
    if any(scope.values()) and not (operations or membership):
        raise typer.BadParameter(
            "A scope on its own changes nothing. Add --only to keep just the scope, "
            "--drop to remove it, or an operation to apply to it."
        )

    ops = []

    # 1. interpret
    if tag_energy or ni_yml:
        ops.append(TagEnergyOp(op="tag-energy", ni_yml=str(ni_yml) if ni_yml else None))

    # 2. annotate
    if set_:
        ops.append(SetOp(op="set", attrs=attrs, rename=rename, **scope))
    if ref_particle:
        ops.append(SetRefOp(op="set-ref", particle=ref_particle, **scope))

    # 3. derive
    if add_ref:
        ops.append(AddRefOp(op="add-ref", particle=add_ref, psq=ref_psq, **scope))

    # 4. membership
    if only:
        ops.append(KeepOp(op="keep", **scope))
    elif drop:
        ops.append(DropOp(op="drop", **scope))

    if not ops:
        raise typer.BadParameter(
            "Nothing to do. Give an operation (--tag-energy, --set, --ref-particle, "
            "--add-ref) or a membership flag (--only, --drop)."
        )

    try:
        return EditSpec(edit=ops)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
