"""The ``ss edit`` command: flag desugaring, recipe equivalence, and guards."""

from __future__ import annotations

from typer.testing import CliRunner

from sigmondsamplings.cli.main import app
from sigmondsamplings.edit_spec import EditSpec
from sigmondsamplings.io.loader import SigmondLoader

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, ["edit", *[str(arg) for arg in args]])


def names(path) -> set[str]:
    return {s.observable_info.name for s in SigmondLoader(str(path)).observables}


def obs_named(path, name: str):
    return next(
        s.observable_info
        for s in SigmondLoader(str(path)).observables
        if s.observable_info.name == name
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_tag_energy_writes_the_attrs(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    result = invoke(samplings_file, out, "--tag-energy")
    assert result.exit_code == 0, result.output
    assert obs_named(out, "PSQ0_A1g_elab_0").psq == 0


def test_add_ref_adds_reference_levels(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    assert invoke(samplings_file, out, "--tag-energy", "--add-ref", "N").exit_code == 0
    assert obs_named(out, "PSQ1_E_elab_0_ref").ref_particle == "N"


def test_scoped_set_with_rename(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    result = invoke(
        samplings_file, out, "--tag-energy", "-w", "psq=1", "--set", "irrep=A2", "--rename"
    )
    assert result.exit_code == 0, result.output
    assert "PSQ1_A2_elab_0" in names(out)
    assert "PSQ0_A1g_elab_0" in names(out)


def test_only_prunes_to_the_scope(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    assert invoke(samplings_file, out, "--tag-energy", "-w", "psq=0", "--only").exit_code == 0
    assert names(out) == {"PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1", "PSQ0_N", "PSQ0_pi"}


def test_drop_removes_the_scope(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    assert invoke(samplings_file, out, "--tag-energy", "-w", "level_index=1", "--drop").exit_code == 0
    assert "PSQ0_A1g_elab_1" not in names(out)


def test_spec_selector_pairs_with_query_save(samplings_file, tmp_path):
    """The TOML `ss query energy --save` writes is what `--spec` consumes."""
    tagged = tmp_path / "tagged.hdf5"
    assert invoke(samplings_file, tagged, "--tag-energy").exit_code == 0

    spectrum = tmp_path / "spectrum.toml"
    saved = runner.invoke(app, ["query", "energy", str(tagged), "--save", str(spectrum)])
    assert saved.exit_code == 0, saved.output

    out = tmp_path / "out.hdf5"
    assert invoke(tagged, out, "--spec", spectrum, "--only").exit_code == 0
    assert names(out) == {"PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1", "PSQ1_E_elab_0"}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


def test_save_recipe_records_the_desugared_program(samplings_file, tmp_path):
    recipe = tmp_path / "recipe.toml"
    result = invoke(
        samplings_file, tmp_path / "out.hdf5", "--tag-energy", "--add-ref", "N",
        "--save-recipe", recipe,
    )
    assert result.exit_code == 0, result.output
    assert [op.op for op in EditSpec.from_file(recipe).edit] == ["tag-energy", "add-ref"]


def test_a_saved_recipe_replays_to_the_same_file(samplings_file, tmp_path):
    """The load-bearing test: both front doors run the same core."""
    recipe = tmp_path / "recipe.toml"
    from_flags = tmp_path / "flags.hdf5"
    assert invoke(
        samplings_file, from_flags, "--tag-energy", "-w", "psq=1", "--add-ref", "N",
        "--save-recipe", recipe,
    ).exit_code == 0

    from_recipe = tmp_path / "replay.hdf5"
    assert invoke(samplings_file, from_recipe, "--recipe", recipe).exit_code == 0

    assert names(from_flags) == names(from_recipe)


def test_flag_order_does_not_change_the_result(samplings_file, tmp_path):
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    assert invoke(samplings_file, a, "--tag-energy", "--add-ref", "N", "-w", "psq=1").exit_code == 0
    assert invoke(samplings_file, b, "-w", "psq=1", "--add-ref", "N", "--tag-energy").exit_code == 0
    assert names(a) == names(b)


def test_recipe_scopes_two_sets_differently(samplings_file, tmp_path):
    """What the flag form cannot express in one pass."""
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        '[[edit]]\nop = "tag-energy"\n\n'
        '[[edit]]\nop = "set"\nwhere = { psq = 0, level_index = 0 }\n'
        'attrs = { energy_type = "ecm" }\nrename = true\n\n'
        '[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\nrename = true\n'
    )
    out = tmp_path / "out.hdf5"
    assert invoke(samplings_file, out, "--recipe", recipe).exit_code == 0
    assert "PSQ0_A1g_ecm_0" in names(out)
    assert "PSQ1_A2_elab_0" in names(out)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_recipe_and_operation_flags_are_mutually_exclusive(samplings_file, tmp_path):
    recipe = tmp_path / "recipe.toml"
    recipe.write_text('[[edit]]\nop = "tag-energy"\n')
    result = invoke(samplings_file, tmp_path / "out.hdf5", "--recipe", recipe, "--add-ref", "N")
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_only_and_drop_are_mutually_exclusive(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5", "-w", "psq=0", "--only", "--drop")
    assert result.exit_code != 0
    assert "not both" in result.output


def test_a_scope_with_no_operation_is_refused(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5", "-w", "psq=0")
    assert result.exit_code != 0
    assert "changes nothing" in result.output


def test_no_operation_at_all_is_refused(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5")
    assert result.exit_code != 0
    assert "Nothing to do" in result.output


def test_rename_without_set_is_refused(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5", "--tag-energy", "--rename")
    assert result.exit_code != 0
    assert "--rename applies to --set" in result.output


def test_add_ref_without_tag_energy_is_refused(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5", "--add-ref", "N")
    assert result.exit_code != 0
    assert "needs energy levels" in result.output


def test_unknown_set_attribute_is_refused(samplings_file, tmp_path):
    result = invoke(samplings_file, tmp_path / "out.hdf5", "--tag-energy", "--set", "psqq=2")
    assert result.exit_code != 0
    assert "Did you mean 'psq'?" in result.output


def test_existing_output_needs_overwrite(samplings_file, tmp_path):
    out = tmp_path / "out.hdf5"
    assert invoke(samplings_file, out, "--tag-energy").exit_code == 0
    result = invoke(samplings_file, out, "--tag-energy")
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert invoke(samplings_file, out, "--tag-energy", "-f").exit_code == 0


def test_missing_input_is_refused(tmp_path):
    result = invoke(tmp_path / "nope.hdf5", tmp_path / "out.hdf5", "--tag-energy")
    assert result.exit_code != 0
    assert "does not exist" in result.output
