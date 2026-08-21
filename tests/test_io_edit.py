"""The edit executor: scope semantics, operators, and the file round trip."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sigmondsamplings.edit_spec import EditSpec
from sigmondsamplings.energy_levels import EnergyObsInfo
from sigmondsamplings.io.edit import apply_edits, edit_file
from sigmondsamplings.io.loader import SigmondLoader

TAG = '[[edit]]\nop = "tag-energy"\n'


def run(collection, recipe: str, base_dir: Path | None = None):
    """Apply a TOML recipe to *collection*."""
    spec = EditSpec.from_toml(recipe).resolve(base_dir=base_dir or Path.cwd())
    return apply_edits(collection, spec)


def names(collection) -> set[str]:
    return {sampling.observable_info.name for sampling in collection}


def obs_named(collection, name: str):
    return next(s.observable_info for s in collection if s.observable_info.name == name)


# ---------------------------------------------------------------------------
# The aliasing the executor is built on
# ---------------------------------------------------------------------------


def test_filter_returns_a_view_over_the_same_observables(observables):
    """The premise of every scoped op: no split and rejoin is needed."""
    scope = observables.filter(name="junk_obs")
    scope.obs.update(name="renamed")
    assert "renamed" in names(observables)


# ---------------------------------------------------------------------------
# tag-energy
# ---------------------------------------------------------------------------


def test_tag_energy_interprets_what_it_can(observables):
    out = run(observables, TAG)
    assert isinstance(obs_named(out, "PSQ0_A1g_elab_0"), EnergyObsInfo)
    assert obs_named(out, "PSQ0_A1g_elab_0").psq == 0
    assert obs_named(out, "PSQ0_A1g_elab_0").irrep == "A1g"


def test_tag_energy_passes_non_energy_observables_through(observables):
    out = run(observables, TAG)
    assert names(out) == names(observables) | {"junk_obs"}
    assert not isinstance(obs_named(out, "junk_obs"), EnergyObsInfo)


def test_tag_energy_does_not_lose_observables(observables):
    before = len(observables)
    assert len(run(observables, TAG)) == before


def test_re_tagging_preserves_attrs_the_name_cannot_carry(observables):
    """A second tag-energy must not re-parse names over existing metadata."""
    once = run(observables, TAG + '\n[[edit]]\nop = "add-ref"\nparticle = "N"\n')
    twice = run(once, TAG)
    assert obs_named(twice, "PSQ0_A1g_elab_0_ref").ref_particle == "N"


# ---------------------------------------------------------------------------
# Scope semantics
# ---------------------------------------------------------------------------


def test_scope_leaves_observables_outside_it_untouched(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\n')
    assert obs_named(out, "PSQ1_E_elab_0").irrep == "A2"
    assert obs_named(out, "PSQ0_A1g_elab_0").irrep == "A1g"


def test_scope_does_not_change_output_membership(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\n')
    assert len(out) == len(observables)


def test_only_writes_just_the_scope(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "keep"\nwhere = { psq = 0 }\n')
    assert names(out) == {"PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1", "PSQ0_N", "PSQ0_pi"}


def test_drop_removes_just_the_scope(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "drop"\nwhere = { level_index = 1 }\n')
    assert "PSQ0_A1g_elab_1" not in names(out)
    assert "PSQ0_A1g_elab_0" in names(out)


def test_spec_selector_keeps_only_the_named_levels(observables, tmp_path):
    (tmp_path / "spectrum.toml").write_text(
        '[[spectrum]]\npsq = 0\nirrep = "A1g"\nlevels = [0]\n'
    )
    out = run(
        observables,
        TAG + '\n[[edit]]\nop = "keep"\nspec = "spectrum.toml"\n',
        base_dir=tmp_path,
    )
    assert names(out) == {"PSQ0_A1g_elab_0"}


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


def test_set_rewrites_the_attribute(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\n')
    assert obs_named(out, "PSQ1_E_elab_0").irrep == "A2"


def test_set_alone_leaves_the_name_stale(observables):
    """Names only resync when asked; the attr and the name are separate."""
    out = run(observables, TAG + '\n[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\n')
    assert "PSQ1_E_elab_0" in names(out)


def test_rename_resyncs_the_canonical_name(observables):
    out = run(
        observables,
        TAG + '\n[[edit]]\nop = "set"\nwhere = { psq = 1 }\nattrs = { irrep = "A2" }\nrename = true\n',
    )
    assert "PSQ1_A2_elab_0" in names(out)
    assert "PSQ1_E_elab_0" not in names(out)


def test_rename_keeps_names_it_cannot_build(observables):
    """A plain observable has no canonical form; it must not abort the pass."""
    out = run(
        observables,
        TAG + '\n[[edit]]\nop = "set"\nattrs = { index = 3 }\nrename = true\n',
    )
    assert "junk_obs" in names(out)
    assert obs_named(out, "junk_obs").index == 3


def test_set_rejects_an_unknown_attribute(observables):
    with pytest.raises(ValueError, match="Unknown --set attribute 'psqq'"):
        run(observables, TAG + '\n[[edit]]\nop = "set"\nattrs = { psqq = 2 }\n')


def test_set_rejects_a_read_only_attribute(observables):
    with pytest.raises(ValueError, match="Cannot set that attribute"):
        run(observables, TAG + '\n[[edit]]\nop = "set"\nattrs = { is_ref = true }\n')


# ---------------------------------------------------------------------------
# add-ref / set-ref
# ---------------------------------------------------------------------------


ADD_REF = TAG + '\n[[edit]]\nop = "add-ref"\nparticle = "N"\n'


def test_add_ref_creates_a_reference_level_per_level(observables):
    before = len(observables)
    out = run(observables, ADD_REF)
    assert len(out) == before + 5  # every energy level; junk_obs is not one
    assert obs_named(out, "PSQ0_A1g_elab_0_ref").ref_particle == "N"


def test_add_ref_is_idempotent(observables):
    spec = EditSpec.from_toml(ADD_REF).resolve(base_dir=Path.cwd())
    once = apply_edits(observables, spec)
    twice = apply_edits(once, spec)
    assert len(twice) == len(once)


def test_add_ref_honours_its_scope(observables):
    out = run(observables, TAG + '\n[[edit]]\nop = "add-ref"\nparticle = "N"\nwhere = { psq = 1 }\n')
    assert "PSQ1_E_elab_0_ref" in names(out)
    assert "PSQ0_A1g_elab_0_ref" not in names(out)


def test_add_ref_resolves_its_particle_outside_the_scope(observables):
    """The nucleon is at psq=0, but the scope is psq=1; the lookup must still find it."""
    out = run(observables, TAG + '\n[[edit]]\nop = "add-ref"\nparticle = "N"\nwhere = { psq = 1 }\n')
    assert obs_named(out, "PSQ1_E_elab_0_ref").ref_particle == "N"


def test_add_ref_divides_by_the_reference_mass(observables):
    out = run(observables, ADD_REF)
    level = next(s for s in out if s.observable_info.name == "PSQ1_E_elab_0")
    ref = next(s for s in out if s.observable_info.name == "PSQ1_E_elab_0_ref")
    nucleon = next(s for s in out if s.observable_info.name == "PSQ0_N")
    assert ref.full_sample_value == pytest.approx(
        level.full_sample_value / nucleon.full_sample_value
    )


def test_add_ref_reports_a_particle_it_cannot_find(observables):
    with pytest.raises(ValueError, match="No single-hadron observable for particle 'K'"):
        run(observables, TAG + '\n[[edit]]\nop = "add-ref"\nparticle = "K"\n')


def test_energy_op_on_an_untagged_collection_is_refused(observables):
    """The spec-level guard only orders ops; this is the runtime one."""
    from sigmondsamplings.edit_spec import AddRefOp, EditResolved

    spec = EditResolved(edit=[AddRefOp(op="add-ref", particle="N")])
    with pytest.raises(ValueError, match="Run a 'tag-energy' op first"):
        apply_edits(observables, spec)


def test_set_ref_tags_only_reference_mode_levels(observables):
    out = run(observables, ADD_REF + '\n[[edit]]\nop = "set-ref"\nparticle = "pi"\n')
    assert obs_named(out, "PSQ0_A1g_elab_0_ref").ref_particle == "pi"
    assert obs_named(out, "PSQ0_A1g_elab_0").ref_particle is None


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_round_trips_through_hdf5(samplings_file, tmp_path):
    spec = EditSpec.from_toml(ADD_REF).resolve(base_dir=Path.cwd())
    out = edit_file(str(samplings_file), str(tmp_path / "out.hdf5"), spec)

    reloaded = SigmondLoader(str(out)).observables
    assert obs_named(reloaded, "PSQ0_A1g_elab_0_ref").ref_particle == "N"
    assert "junk_obs" in names(reloaded)


def test_editing_a_written_file_again_is_stable(samplings_file, tmp_path):
    spec = EditSpec.from_toml(ADD_REF).resolve(base_dir=Path.cwd())
    first = edit_file(str(samplings_file), str(tmp_path / "a.hdf5"), spec)
    second = edit_file(str(first), str(tmp_path / "b.hdf5"), spec)
    assert len(SigmondLoader(str(second)).observables) == len(
        SigmondLoader(str(first)).observables
    )


def test_edit_file_refuses_to_write_an_empty_result(samplings_file, tmp_path):
    spec = EditSpec.from_toml(
        TAG + '\n[[edit]]\nop = "keep"\nwhere = { name = "nothing_matches" }\n'
    ).resolve(base_dir=Path.cwd())
    with pytest.raises(ValueError, match="removed every observable"):
        edit_file(str(samplings_file), str(tmp_path / "out.hdf5"), spec)


def test_h5_output_extension_is_preserved(samplings_file, tmp_path):
    spec = EditSpec.from_toml(TAG).resolve(base_dir=Path.cwd())
    out = tmp_path / "energy_attrs.h5"
    assert edit_file(str(samplings_file), str(out), spec) == out
    assert out.exists()


def test_missing_output_extension_inherits_the_input_extension(samplings_file, tmp_path):
    """Without a suffix on the output, the input's HDF5 suffix carries over."""
    input_h5 = tmp_path / "energy_input.h5"
    shutil.copy2(samplings_file, input_h5)
    spec = EditSpec.from_toml(TAG).resolve(base_dir=Path.cwd())

    result = edit_file(str(input_h5), str(tmp_path / "energy_attrs"), spec)

    assert result == tmp_path / "energy_attrs.h5"
    assert result.exists()
