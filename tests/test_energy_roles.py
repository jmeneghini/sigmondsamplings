"""Explicit energy roles, legacy projections and copied physical level ordering."""
from dataclasses import replace

import numpy as np
import pytest

from sigmondsamplings.energy_levels import (
    EnergyObsInfo, SHEnergyObsInfo, Particle, create_energy_obs_info, energy_obs_from_attrs,
)
from sigmondsamplings.energy_level_collection import (
    SingleEnsembleEnergyCollection, MultiEnsembleEnergyCollection,
)
from sigmondsamplings.info import ObservableInfo, SectorInfo, EnsembleInfo, SamplingInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.sampling import SigmondSampling

SECTOR = SectorInfo(psq=0, irrep="A1g")
ENS = EnsembleInfo("roles", 64)
SAMP = SamplingInfo("bootstrap", 2, seed=7)
PARTS = (Particle("pi", 1), Particle("pi", 1), Particle("K", 0))


def energy(value=1.0, *, sector=SECTOR, ensemble=ENS, **kwargs):
    info = EnergyObsInfo(ensemble_info=ensemble, sector_info=sector,
                         name=kwargs.pop("name", f"fitted_{value}"),
                         energy_type=kwargs.pop("energy_type", "elab"), **kwargs)
    return SigmondSampling(np.array([value, 9., -2.]), info, SAMP)


def test_single_hadron_authority_alias_and_legacy_projection():
    info = SHEnergyObsInfo(hadron="PI", psq=0)
    assert info.hadron == info.particle == "pi"
    assert info.ni_particles == () and not info.non_interacting
    assert info.particles == (Particle("pi", 0),)
    info.psq = 2
    info.particle = "K"
    assert info.hadron == "K" and info.particles == (Particle("K", 2),)
    with pytest.raises(AttributeError):
        info.particles = PARTS
    assert info.public_attributes()["hadron"] == "K"
    assert info.to_attrs()["ni_pairs"] == ["K(2)"]


@pytest.mark.parametrize("kwargs", [
    dict(hadron="pi"), dict(particles=PARTS, ni_particles=PARTS[:1]),
    dict(non_interacting=True), dict(ni_particles=PARTS, non_interacting=True, energy_type="ecm"),
    dict(name="elab_ni_pi(0)", ni_particles=PARTS, non_interacting=False),
])
def test_invalid_plain_energy_roles_reject(kwargs):
    with pytest.raises(ValueError):
        EnergyObsInfo(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(particle="pi", hadron="K"), dict(ni_particles=PARTS),
    dict(non_interacting=True), dict(energy_type="decm"), dict(energy_type="qcmsq"),
    dict(particles=PARTS), dict(particles=[Particle("pi", 1)], psq=0),
])
def test_invalid_single_hadron_roles_reject(kwargs):
    with pytest.raises(ValueError):
        SHEnergyObsInfo(name="sh", **kwargs)


@pytest.mark.parametrize("psq", [None, 0, 2])
@pytest.mark.parametrize("ref", [None, "ref", "pi"])
def test_ni_names_keep_total_momentum_order_and_repetitions(psq, ref):
    info = EnergyObsInfo(psq=psq, ni_particles=PARTS, non_interacting=True,
                         ref_particle=ref, rot_idx=3)
    expected = ("" if psq is None else f"PSQ{psq}_") + "elab_ni_pi(1)_pi(1)_K(0)_ROT3"
    if ref:
        expected += "_ref"
    assert info.name == expected
    parsed = create_energy_obs_info(ObservableInfo(expected))
    assert type(parsed) is EnergyObsInfo and parsed.non_interacting
    assert parsed.ni_particles == PARTS and parsed.psq == psq
    assert parsed.rot_idx == 3 and parsed.level_index is None
    assert parsed.canonical_name == expected


def test_legacy_roles_use_kind_type_and_marker_instead_of_constituent_count():
    base = ObservableInfo("legacy")
    common = dict(psq=0, irrep="A1g", ni_pairs=["pi(0)"], energy_type="decm")
    shift = energy_obs_from_attrs(base, dict(common, obs_kind="energy"))
    assert type(shift) is EnergyObsInfo and shift.ni_particles == (Particle("pi", 0),)
    assert not shift.non_interacting and shift.hadron is None
    sh = energy_obs_from_attrs(base, dict(common, obs_kind="energy_sh", energy_type="elab"))
    assert sh.hadron == "pi" and sh.ni_particles == ()
    ni = energy_obs_from_attrs(ObservableInfo("elab_ni_pi(0)"),
                              dict(common, obs_kind="energy", energy_type="elab"))
    assert ni.non_interacting and not ni.is_single_hadron
    parsed = create_energy_obs_info(ObservableInfo("PSQ0_A1g_decm_0_pi(1)_pi(1)"))
    assert parsed.psq == 0 and parsed.ni_particles == PARTS[:2]
    with pytest.raises(ValueError):
        create_energy_obs_info(ObservableInfo("ecm_ni_pi(0)"))


def test_interacting_view_excludes_explicit_ni_sums():
    fitted = energy()
    ni = fitted.with_observable_info(EnergyObsInfo(ni_particles=PARTS, non_interacting=True, ensemble_info=ENS))
    sh = fitted.with_observable_info(SHEnergyObsInfo(hadron="pi", psq=0, ensemble_info=ENS))
    collection = SingleEnsembleEnergyCollection([fitted, ni, sh])
    assert list(collection.interacting_spectra) == [fitted]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("kind", ["sh", "ni", "shift", "interacting"])
def test_roles_round_trip_codecs_and_hdf5(tmp_path, lazy, kind):
    infos = dict(
        sh=SHEnergyObsInfo(hadron="pi", sector_info=SECTOR, ensemble_info=ENS),
        ni=EnergyObsInfo(ni_particles=PARTS, non_interacting=True, ref_particle="pi", ensemble_info=ENS),
        shift=EnergyObsInfo(ni_particles=PARTS, energy_type="decm", sector_info=SECTOR, ensemble_info=ENS),
        interacting=EnergyObsInfo(energy_type="elab", sector_info=SECTOR, ensemble_info=ENS),
    )
    info = infos[kind]
    attrs = info.to_attrs()
    restored = energy_obs_from_attrs(ObservableInfo(info.name, ensemble_info=ENS), attrs)
    assert type(restored) is type(info) and restored == info
    for key in ("hadron", "ni_particles", "non_interacting", "energy_type", "particles"):
        assert key in restored.public_attributes()
    path = tmp_path / "roles.hdf5"
    sampling = SigmondSampling(np.array([1., .9, 1.1]), info, SAMP)
    SigmondWriter(create_backups=False).write_hdf5(str(path), [sampling], group="g")
    loaded = SigmondLoader(str(path), lazy=lazy).observables[0]
    assert loaded.observable_info == info
    np.testing.assert_array_equal(loaded.data, sampling.data)


@pytest.mark.parametrize("attrs", [
    dict(obs_kind="energy", ni_pairs=["pi(0)"], ni_particles=["K(0)"]),
    dict(obs_kind="energy_sh", ni_pairs=["pi(0)"], hadron="K"),
    dict(obs_kind="energy_sh", hadron="pi", ni_particles=["pi(0)"]),
    dict(obs_kind="energy_sh", hadron="pi", non_interacting=True),
    dict(obs_kind="energy", non_interacting="false"),
])
def test_conflicting_role_codecs_reject(attrs):
    with pytest.raises(ValueError):
        energy_obs_from_attrs(ObservableInfo("legacy"), attrs)


@pytest.mark.parametrize("collection_type", [SingleEnsembleEnergyCollection, MultiEnsembleEnergyCollection])
def test_ranked_copies_use_stable_full_sample_order_and_preserve_buffers(collection_type):
    original = [energy(3, rot_idx=0), energy(1, rot_idx=1), energy(1, rot_idx=2)]
    collection = collection_type(original, return_type="list")
    ranked = collection.with_level_indices()
    assert type(ranked) is collection_type and ranked._return_type == "list"
    assert [s.observable_info.level_index for s in ranked] == [2, 0, 1]
    assert [s.observable_info.rot_idx for s in ranked] == [0, 1, 2]
    for source, result in zip(original, ranked):
        assert source.observable_info.level_index is None
        assert result.observable_info is not source.observable_info
        assert result.data is source.data
        assert result.observable_info.name == result.observable_info.canonical_name
        np.testing.assert_array_equal(result.data, source.data)
    ranked[0].observable_info.irrep = "E"
    assert original[0].observable_info.irrep == "A1g"


def test_ranks_group_by_exact_sector_and_ensemble():
    collection = MultiEnsembleEnergyCollection([
        energy(4), energy(1), energy(5, sector=replace(SECTOR, twoI=0)),
        energy(6, ensemble=EnsembleInfo("other", 64)),
    ])
    assert [s.observable_info.level_index for s in collection.with_level_indices()] == [1, 0, 0, 0]


def test_ranked_sector_address_collisions_remain_a_writer_error(tmp_path):
    collection = SingleEnsembleEnergyCollection([
        energy(1), energy(2, sector=replace(SECTOR, twoI=0)),
    ])
    ranked = collection.with_level_indices()
    # Exact sectors rank independently; their legacy addresses can still collide.
    assert [s.observable_info.level_index for s in ranked] == [0, 0]
    assert ranked[0].observable_info != ranked[1].observable_info
    with pytest.raises(ValueError):
        SigmondWriter(create_backups=False).write_hdf5(str(tmp_path / "collision.hdf5"), ranked)


@pytest.mark.parametrize("case", ["indices", "unknown", "nan", "inf", "complex", "mixed_type", "mixed_ref", "rotation", "sh", "ni"])
def test_incomparable_or_ambiguous_ranking_rejects_without_mutation(case):
    samplings = [energy(2), energy(1)]
    info = samplings[1].observable_info
    if case == "indices": info.level_index = 4
    if case == "unknown": info.psq = None
    if case == "nan": samplings[1].data[0] = np.nan
    if case == "inf": samplings[1].data[0] = np.inf
    if case == "complex": samplings[1].data = samplings[1].data.astype(complex)
    if case == "mixed_type": info.energy_type = "ecm"
    if case == "mixed_ref": info.ref_particle = "pi"
    if case == "rotation":
        for sampling in samplings: sampling.observable_info.rot_idx = 2
    if case == "sh": samplings[1] = samplings[1].with_observable_info(SHEnergyObsInfo(hadron="pi", psq=0, ensemble_info=ENS))
    if case == "ni": samplings[1] = samplings[1].with_observable_info(EnergyObsInfo(ni_particles=PARTS, non_interacting=True, ensemble_info=ENS))
    collection = SingleEnsembleEnergyCollection(samplings)
    before = [sampling.observable_info.copy() for sampling in collection]
    with pytest.raises(ValueError): collection.with_level_indices()
    assert [sampling.observable_info for sampling in collection] == before
    if case == "indices":
        assert [s.observable_info.level_index for s in collection.with_level_indices(overwrite=True)] == [1, 0]
        assert info.level_index == 4
