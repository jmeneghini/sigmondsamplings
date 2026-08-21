"""Collection attribute-writing API: replace (copies) vs update (in place)."""

from __future__ import annotations

import pytest

from sigmondsamplings.energy_levels import EnergyObsInfo


def test_replace_leaves_the_original_untouched(observables):
    new = observables.obs.replace(name="renamed")
    assert set(new.obs.name) == {"renamed"}
    assert "PSQ0_N" in set(observables.obs.name)


def test_update_writes_through_to_the_original(observables):
    observables.obs.update(name="renamed")
    assert set(observables.obs.name) == {"renamed"}


def test_update_on_a_filtered_scope_reaches_the_parent(observables):
    observables.filter(name="junk_obs").obs.update(index=7)
    assert next(s.observable_info.index for s in observables if s.observable_info.name == "junk_obs") == 7


def test_update_accepts_the_same_value_forms_as_replace(observables):
    observables.obs.update(name=lambda obs: f"x_{obs.name}")
    assert all(name.startswith("x_") for name in observables.obs.name)

    observables.obs.update(index=list(range(len(observables))))
    assert list(observables.obs.index) == list(range(len(observables)))


def test_update_rejects_a_mismatched_list_length(observables):
    with pytest.raises(ValueError, match="must match collection length"):
        observables.obs.update(index=[1, 2])


def test_update_rejects_an_empty_call(observables):
    with pytest.raises(ValueError, match="at least one attribute"):
        observables.obs.update()


def test_update_invalidates_the_shared_attr_cache(observables):
    assert observables.shared_attr("index") == 0
    observables.obs.update(index=5)
    assert observables.shared_attr("index") == 5


# ---------------------------------------------------------------------------
# update_name
# ---------------------------------------------------------------------------


def _energy_info(**kwargs) -> EnergyObsInfo:
    defaults = dict(name="orig", index=0, irrep="A1g", psq=0, energy_type="elab", level_index=0)
    return EnergyObsInfo(**{**defaults, **kwargs})


def test_update_name_builds_the_canonical_name():
    info = _energy_info()
    assert info.update_name() is True
    assert info.name == "PSQ0_A1g_elab_0"


def test_update_name_raises_on_incomplete_attrs_by_default():
    info = _energy_info(irrep=None)
    with pytest.raises(ValueError, match="Cannot generate canonical name"):
        info.update_name()


def test_update_name_non_strict_keeps_the_existing_name():
    info = _energy_info(irrep=None)
    assert info.update_name(strict=False) is False
    assert info.name == "orig"
