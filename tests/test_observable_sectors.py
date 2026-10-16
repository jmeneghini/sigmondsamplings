"""Shared-sector authority, boundary codecs and discoverable analysis facets."""

from dataclasses import fields, replace

import h5py
import numpy as np
import pytest
from correlator_data import ENS, JACK, OPS, ckey

from sigmondsamplings.correlators import CorrelatorObsInfo, VEVInfo, VEVObsInfo
from sigmondsamplings.energy_levels import EnergyObsInfo, SHEnergyObsInfo, create_energy_obs_info
from sigmondsamplings.info import ObservableInfo, SectorInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.obsmeta import fields_for
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.sampling_array import ArrayObsInfo, SigmondDataArray
from sigmondsamplings.cli.query import attr_records

SECTOR = SectorInfo(B=0, Q=0, S=0, C=0, twoI=2, twoI3=0, twoJ=0,
                    Gpar=-1, Cpar=1, par=-1, psq=0, irrep="A1um")


def sampling(info):
    return SigmondSampling(np.arange(JACK.num_resamplings + 1, dtype=float), info, JACK)


@pytest.mark.parametrize("field", [field.name for field in fields(SectorInfo)])
def test_sector_fields_participate_in_exact_identity_and_partial_compatibility(field):
    partial = replace(SECTOR, **{field: None})
    assert partial != SECTOR
    assert len({partial, SECTOR}) == 2
    assert partial.compatible_with(SECTOR) and SECTOR.compatible_with(partial)
    a = ObservableInfo("same", sector_info=SECTOR)
    b = ObservableInfo("same", sector_info=partial)
    assert a != b and len({a, b}) == 2
    assert getattr(a, field) == getattr(SECTOR, field)
    energy = EnergyObsInfo("same", energy_type="elab", sector_info=SECTOR)
    assert energy != EnergyObsInfo("same", energy_type="elab", sector_info=partial)


@pytest.mark.parametrize("psq", [-1, 1.0, 1.5, True, "1"])
def test_invalid_psq_rejects_every_constructor(psq):
    for construct in (lambda: SectorInfo(psq=psq), lambda: ObservableInfo("x", psq=psq),
                      lambda: EnergyObsInfo("x", psq=psq)):
        with pytest.raises(ValueError):
            construct()


def test_sector_validation_display_and_unknown_values():
    assert SectorInfo(psq=np.int64(0)).psq == 0
    with pytest.raises(ValueError, match="irrep"):
        SectorInfo(irrep="not-an-irrep")
    a, b = SectorInfo(psq=0), SectorInfo(psq=1)
    assert not a.compatible_with(b) and not b.compatible_with(a)
    assert ObservableInfo("x").sector_info == SectorInfo()
    assert ObservableInfo("x").isospin is None
    assert "PSQ=0" in str(SECTOR) and "A1um" in repr(SECTOR)
    assert "d" in SECTOR._repr_latex_()


@pytest.mark.parametrize("kind", [ObservableInfo, EnergyObsInfo, ArrayObsInfo])
def test_normalization_and_copied_edits(kind):
    a = kind("first", sector_info=SECTOR, psq=0, irrep="A1um")
    b = a.copy()
    assert b.sector_info is SECTOR
    assert a.isospin == "triplet" and a.strangeness == a.S == 0
    b.psq = 2
    b.strangeness = -1
    b.Q = 1
    assert a.sector_info is SECTOR
    assert b.sector_info == replace(SECTOR, psq=2, S=-1, Q=1)
    assert not {"psq", "irrep", "strangeness", "twoI"}.intersection(vars(a))
    for kwargs in ({"psq": 1}, {"irrep": "E"}):
        with pytest.raises(ValueError, match="Conflicting sector"):
            kind("bad", sector_info=SECTOR, **kwargs)


def test_channel_conversion_and_explicit_energy_context():
    channel = OPS[0].channel
    sector = channel.to_sector()
    assert sector == SectorInfo(twoI=2, S=0, psq=0, irrep="A1um")
    assert sector.B is None and sector.Q is None
    energy = EnergyObsInfo.from_channel(channel, ensemble_info=ENS, rot_idx=2)
    assert energy.sector_info == sector and energy.sector == (0, "A1um")
    assert energy.rot_idx == 2 and energy.level_index is None
    assert energy.name == "PSQ0_A1um_elab_ROT2"
    assert "ROT" in energy.latex_str
    assert create_energy_obs_info(energy).sector_info == sector
    assert create_energy_obs_info(energy).rot_idx == 2
    with pytest.raises(TypeError):
        EnergyObsInfo.from_channel(channel)


@pytest.mark.parametrize("rot_idx", [-1, True, 1.2])
def test_rotation_index_rejects_invalid_values(rot_idx):
    with pytest.raises(ValueError, match="rot_idx"):
        EnergyObsInfo.from_channel(OPS[0].channel, ensemble_info=ENS, rot_idx=rot_idx)


def test_keyed_sectors_remain_read_only_and_preserve_both_channels():
    corr = CorrelatorObsInfo(key=ckey(0, 1, 3), ensemble_info=ENS)
    vev = VEVObsInfo(key=VEVInfo(OPS[0]), ensemble_info=ENS)
    for info in (corr, vev):
        assert info.sector_info == OPS[0].channel.to_sector()
        assert info.isospin == "triplet" and info.strangeness == 0
        assert "_sector_info" not in vars(info)
        for facet, value in (("psq", 2), ("S", -1), ("sector_info", SectorInfo())):
            with pytest.raises(AttributeError):
                setattr(info, facet, value)
    assert corr.sink_channel == OPS[0].channel
    assert corr.source_channel == OPS[1].channel
    corr.name = corr.name.replace("A1um_1", "A9_1")
    assert corr.key is None and corr.key_error is not None
    assert corr.sector_info == SectorInfo() and corr.irrep is None


def test_sector_survives_analysis_filters_copies_and_summaries():
    collection = ObservableCollection([sampling(ObservableInfo("a", ensemble_info=ENS, sector_info=SECTOR))])
    assert len(collection.filter(S=0, twoI=2, psq=0, irrep="A1um")) == 1
    assert len(collection.filter(strangeness=0, isospin="triplet")) == 1
    changed = collection.obs.replace(Q=1)
    assert changed[0].observable_info.Q == 1 and collection[0].observable_info.Q == 0
    frame = collection.to_dataframe()
    for field in fields(SectorInfo):
        assert frame.iloc[0][field.name] == getattr(SECTOR, field.name)
    records = {r["attribute"]: r["count"] for r in attr_records(collection, ["name"])}
    assert records["twoI"] == records["psq"] == records["strangeness"] == 1
    parent = ArrayObsInfo("array", ensemble_info=ENS, sector_info=SECTOR)
    array = SigmondDataArray.from_data(np.zeros((2, JACK.num_resamplings + 1)), parent, JACK)
    assert array[0].observable_info.sector_info == SECTOR


@pytest.mark.parametrize("kind", ["ordinary", "energy", "single_hadron", "correlator", "vev", "opaque"])
@pytest.mark.parametrize("lazy", [False, True])
def test_sectors_round_trip_in_ss_hdf5(tmp_path, kind, lazy):
    infos = {
        "ordinary": ObservableInfo("ordinary", sector_info=SECTOR, ensemble_info=ENS),
        "energy": EnergyObsInfo(sector_info=SECTOR, ensemble_info=ENS, energy_type="elab", rot_idx=2),
        "single_hadron": SHEnergyObsInfo(sector_info=SECTOR, ensemble_info=ENS, particle="pi"),
        "correlator": CorrelatorObsInfo(key=ckey(0, 1, 3), ensemble_info=ENS),
        "vev": VEVObsInfo(key=VEVInfo(OPS[0]), ensemble_info=ENS),
        "opaque": CorrelatorObsInfo(ckey(0, 1, 3).to_sigmond().replace("A1um_1", "A9_1"), ensemble_info=ENS),
    }
    info = infos[kind]
    path = tmp_path / "sector.hdf5"
    original = sampling(info)
    SigmondWriter(create_backups=False).write_hdf5(str(path), [original], group="g")
    restored = SigmondLoader(str(path), lazy=lazy).observables[0]
    assert type(restored.observable_info) is type(info)
    assert restored.observable_info == info
    assert restored.observable_info.sector_info == info.sector_info
    np.testing.assert_array_equal(restored.data, original.data)
    if kind == "opaque":
        assert restored.observable_info.key is None and restored.observable_info.key_error is not None
    if kind in ("ordinary", "energy", "single_hadron"):
        attrs = info.to_attrs()
        assert attrs["sector_info"] == SECTOR.to_dict()
        assert attrs["psq"] == 0 and attrs["irrep"] == "A1um"


@pytest.mark.parametrize("kind", [ObservableInfo, EnergyObsInfo, SHEnergyObsInfo])
def test_legacy_and_nested_codecs_agree_and_conflicts_reject(kind):
    attrs = {"psq": 0, "irrep": "A1um", "energy_type": "elab", "ni_pairs": ["pi(0)"]}
    base = ObservableInfo("legacy", ensemble_info=ENS)
    legacy = kind.from_attrs(base, attrs)
    assert legacy.sector_info == SectorInfo(psq=0, irrep="A1um")
    nested = kind.from_attrs(base, {**attrs, "sector_info": SECTOR.to_dict()})
    assert nested.sector_info == SECTOR
    with pytest.raises(ValueError, match="Conflicting sector"):
        kind.from_attrs(base, {**attrs, "psq": 1, "sector_info": SECTOR.to_dict()})


def test_recursive_numpy_metadata_normalization():
    meta = fields_for(np.zeros(2), {"sector_info": {"twoI": np.int64(2)},
                                   "nested": [np.array([1, 2]), {"x": np.float64(0.25)}]})
    import json
    assert json.loads(json.dumps(meta))["nested"] == [[1, 2], {"x": 0.25}]


def test_hdf5_rejects_colliding_native_storage_keys(tmp_path):
    a = sampling(ObservableInfo("same", ensemble_info=ENS, sector_info=SECTOR))
    b = sampling(ObservableInfo("same", ensemble_info=ENS, sector_info=replace(SECTOR, Q=1)))
    with pytest.raises(ValueError, match="Duplicate logical observable"):
        SigmondWriter(create_backups=False).write_hdf5(str(tmp_path / "duplicate.hdf5"), [a, b])


def test_legacy_general_aliases_and_partial_sector_hdf5(tmp_path):
    info = ObservableInfo("legacy_general", ensemble_info=ENS)
    path = tmp_path / "legacy.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [sampling(info)], group="g")
    with h5py.File(path, "a") as file:
        del file["g/ObsMeta"]
        dataset = next(iter(file["g/Values"].values()))
        dataset.attrs["strangeness"] = 0
        dataset.attrs["isospin"] = "isotriplet"
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored.sector_info == SectorInfo(S=0, twoI=2)
    with pytest.raises(ValueError, match="Conflicting sector"):
        ObservableInfo.from_attrs(info, {"strangeness": 1, "S": 0})
    with pytest.raises(ValueError, match="Conflicting sector"):
        ObservableInfo.from_attrs(info, {"isospin": "singlet", "twoI": 2})


def test_native_keys_without_ss_annotations_keep_physics_and_flags(tmp_path):
    key = ckey(0, 1, 3, herm=True)
    info = CorrelatorObsInfo(key=key, ensemble_info=ENS)
    path = tmp_path / "native.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [sampling(info)], group="g")
    with h5py.File(path, "a") as file:
        del file["g/ObsMeta"]
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored.key == key and restored.key.to_sigmond() == key.to_sigmond()
    assert restored.hermitian is True and restored.subtract_vev == key.subtract_vev
    assert restored.sector_info == info.sector_info
    contradictory = info.to_attrs()
    contradictory["sector_info"] = {**contradictory["sector_info"], "twoI": 0}
    with pytest.raises(ValueError, match="contradicts native key"):
        CorrelatorObsInfo.from_attrs(info, contradictory)


def test_rotation_names_and_labels_keep_both_indices_separate():
    energy = EnergyObsInfo(sector_info=SECTOR, energy_type="elab", rot_idx=2, level_index=4,
                           ref_particle="ref")
    assert energy.name == "PSQ0_A1um_elab_4_ROT2_ref"
    parsed = create_energy_obs_info(energy)
    assert parsed.rot_idx == 2 and parsed.level_index == 4
    assert energy.specify_latex_str() == energy.latex_str
    assert "ROT" not in energy.specify_latex_str(rot_idx=None)
    assert energy != EnergyObsInfo(energy.name, sector_info=SECTOR, energy_type="elab", rot_idx=3,
                                   level_index=4, ref_particle="ref")


def test_single_hadron_sector_edits_keep_legacy_particle_projection_consistent(tmp_path):
    original = SHEnergyObsInfo(particle="pi", sector_info=SECTOR, ensemble_info=ENS)
    changed = original.copy()
    changed.psq = 1
    changed.name = "moving_pi"
    assert original.psq == original.particles[0].psq == 0
    assert changed.psq == changed.particles[0].psq == 1
    assert changed.to_attrs()["ni_pairs"] == ["pi(1)"]
    path = tmp_path / "single_hadron.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [sampling(changed)], group="g")
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored == changed
    with pytest.raises(ValueError, match="Conflicting sector"):
        SHEnergyObsInfo.from_attrs(original, {**original.to_attrs(), "ni_pairs": ["pi(1)"]})


def test_disagreeing_source_sink_sectors_do_not_claim_a_common_sector():
    from sigmondsamplings.correlators import BasicLapHOperator, CorrelatorAtTimeInfo, CorrelatorInfo
    source = BasicLapHOperator.from_sigmond("pion P=(0,0,1) A2m_1 SS_0")
    key = CorrelatorAtTimeInfo(CorrelatorInfo(OPS[0], source), time=3)
    info = CorrelatorObsInfo(key=key)
    assert info.sector_info == SectorInfo()
    assert info.sink_channel.momentum_squared == 0
    assert info.source_channel.momentum_squared == 1
