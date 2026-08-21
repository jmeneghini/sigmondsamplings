"""EditSpec authoring surface: op discrimination, validation, TOML round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sigmondsamplings.edit_spec import (
    AddRefOp,
    DropOp,
    EditSpec,
    KeepOp,
    SetOp,
    SetRefOp,
    TagEnergyOp,
)

RECIPE = """
[[edit]]
op = "tag-energy"

[[edit]]
op = "set"
where = { psq = 0 }
attrs = { irrep = "A1g" }
rename = true

[[edit]]
op = "add-ref"
particle = "N"

[[edit]]
op = "drop"
where = { level_index = [4, 5] }
"""


def test_ops_are_discriminated_by_their_op_field():
    spec = EditSpec.from_toml(RECIPE)
    assert [type(op) for op in spec.edit] == [TagEnergyOp, SetOp, AddRefOp, DropOp]


def test_ops_keep_their_authored_order():
    spec = EditSpec.from_toml(RECIPE)
    assert [op.op for op in spec.edit] == ["tag-energy", "set", "add-ref", "drop"]


def test_scope_clauses_land_on_the_op_itself():
    set_op = EditSpec.from_toml(RECIPE).edit[1]
    assert set_op.where == {"psq": 0}
    assert set_op.attrs == {"irrep": "A1g"}
    assert set_op.rename is True
    assert set_op.is_scoped


def test_op_without_scope_clauses_is_unscoped():
    add_ref = EditSpec.from_toml(RECIPE).edit[2]
    assert not add_ref.is_scoped


def test_toml_round_trip_is_exact():
    spec = EditSpec.from_toml(RECIPE)
    assert EditSpec.from_toml(spec.to_toml()) == spec


def test_dump_omits_fields_left_at_their_default():
    dumped = EditSpec(edit=[TagEnergyOp(op="tag-energy")]).to_toml()
    assert "skip_missing_particles" not in dumped
    assert "ni_yml" not in dumped
    assert 'op = "tag-energy"' in dumped


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EditSpec.from_toml('[[edit]]\nop = "tag-energy"\nni_ymll = "x.yml"\n')


def test_unknown_op_is_rejected():
    with pytest.raises(ValidationError):
        EditSpec.from_toml('[[edit]]\nop = "teleport"\n')


@pytest.mark.parametrize("op_name", ["set-ref", "add-ref"])
def test_energy_op_without_a_preceding_tag_energy_is_rejected(op_name):
    with pytest.raises(ValidationError, match="needs energy levels"):
        EditSpec.from_toml(f'[[edit]]\nop = "{op_name}"\nparticle = "N"\n')


@pytest.mark.parametrize("op_name", ["set-ref", "add-ref"])
def test_energy_op_after_tag_energy_is_accepted(op_name):
    spec = EditSpec.from_toml(
        f'[[edit]]\nop = "tag-energy"\n\n[[edit]]\nop = "{op_name}"\nparticle = "N"\n'
    )
    assert len(spec.edit) == 2


@pytest.mark.parametrize("op_name", ["keep", "drop"])
def test_membership_op_needs_a_scope(op_name):
    with pytest.raises(ValidationError, match="needs a scope"):
        EditSpec.from_toml(f'[[edit]]\nop = "{op_name}"\n')


def test_set_needs_at_least_one_attr():
    with pytest.raises(ValidationError):
        EditSpec.from_toml('[[edit]]\nop = "set"\nattrs = {}\n')


def test_resolve_canonicalizes_the_particle_name():
    spec = EditSpec(
        edit=[TagEnergyOp(op="tag-energy"), AddRefOp(op="add-ref", particle="PI")]
    )
    assert spec.resolve(base_dir=Path.cwd()).edit[1].particle == "pi"


def test_resolve_rejects_an_unknown_particle():
    spec = EditSpec(
        edit=[TagEnergyOp(op="tag-energy"), SetRefOp(op="set-ref", particle="quark")]
    )
    with pytest.raises(ValueError, match="Unknown particle name"):
        spec.resolve(base_dir=Path.cwd())


def test_resolve_makes_a_relative_spec_path_absolute(tmp_path):
    (tmp_path / "spectrum.toml").write_text('[[spectrum]]\npsq = 0\nirrep = "A1g"\nlevels = [0]\n')
    resolved = EditSpec(edit=[KeepOp(op="keep", spec="spectrum.toml")]).resolve(base_dir=tmp_path)
    assert Path(resolved.edit[0].spec) == (tmp_path / "spectrum.toml").resolve()


def test_resolve_rejects_a_missing_spec_path(tmp_path):
    spec = EditSpec(edit=[KeepOp(op="keep", spec="nope.toml")])
    with pytest.raises(FileNotFoundError, match="Spectrum spec TOML does not exist"):
        spec.resolve(base_dir=tmp_path)


def test_resolve_rejects_a_missing_ni_yml(tmp_path):
    spec = EditSpec(edit=[TagEnergyOp(op="tag-energy", ni_yml="nope.yml")])
    with pytest.raises(FileNotFoundError, match="NI pair YAML does not exist"):
        spec.resolve(base_dir=tmp_path)
