"""Durable element/source identities through rotation, calculation and SS files."""
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from pivot_data import spectrum

from sigmondsamplings import AtObsInfo, EffectiveEnergyInfo
from sigmondsamplings.correlators import CorrelatorObsInfo, SinglePivot
from sigmondsamplings.correlators.effective import effective_energy
from sigmondsamplings.correlators.keys import CorrelatorAtTimeInfo
from sigmondsamplings.energy_levels import EnergyObsInfo
from sigmondsamplings.info import ObservableInfo, SectorInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.metadata import (
    EnergyTarget, RotationBinding, SeriesInfo, element_from_record,
)
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.sampling_array import ArrayObsInfo, SigmondDataArray


def pivot(matrix, **kwargs):
    return SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1,
                                   diagonalize_time=kwargs.pop("diagonalize_time", 3),
                                   min_inv_condition=1e-8, **kwargs)


def rotated_series():
    matrix, energies = spectrum()
    p = pivot(matrix)
    return matrix, p, p.rotate(matrix).diagonal(component="real"), energies


@pytest.mark.parametrize("lazy", [False, True])
def test_matrix_to_effective_collection_file_and_rebuilt_series(tmp_path, lazy):
    matrix, p, diagonal, energies = rotated_series()
    eff = effective_energy(diagonal)
    assert type(eff.observable_info) is ArrayObsInfo
    assert not hasattr(eff.observable_info, "time_offset")
    assert eff.dims == ("time", "operator")
    np.testing.assert_allclose(eff.data, np.broadcast_to(energies[None, :, None], eff.data.shape), atol=1e-12)
    for i in range(3):
        first = eff.infos[0, i].element
        for t in range(eff.shape[0]):
            info = eff.infos[t, i]
            assert isinstance(info, AtObsInfo) and info.element is first
            assert info.time == eff.coords["time"][t] and info.rot_idx == i
            assert info.sector_info == matrix.info.channel.to_sector()
            assert not info.is_energy and not info.is_single_hadron
            assert getattr(info, "level_index", None) is None
            assert info.element.source.sink_rotation == RotationBinding(p.identifier, i)
            assert info.element.source.source_rotation == RotationBinding(p.identifier, i)
            assert info.element.source.hermitian and not info.element.source.subtract_vev
            assert info.element.source.re_im == "re"
            assert info.element.kind == "forward" and info.element.step == 1
            assert info.element.time_extent == 16 and info.element.time_offset == .5
            assert info.element.energy is None
    collection = eff.to_collection()
    path = tmp_path / "effective.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), list(collection), group="g")
    loaded = SigmondLoader(str(path), lazy=lazy).observables
    original = {cell.observable_info.name: cell for cell in collection}
    for cell in loaded:
        prior = original[cell.observable_info.name]
        assert cell.observable_info == prior.observable_info
        assert cell.observable_info.to_attrs() == prior.observable_info.to_attrs()
        if lazy:
            assert not cell.is_materialized
        np.testing.assert_array_equal(cell.data, prior.data)
    selected = loaded.filter(irrep="A1g", psq=0, rot_idx=2)
    rebuilt = SigmondDataArray.from_collection(selected, shape_by=["time"])
    for cell in rebuilt.to_collection():
        assert cell.observable_info.rot_idx == 2
        assert cell.observable_info.element.source.sink_rotation.pivot_id == p.identifier
    np.testing.assert_array_equal(rebuilt.sel(time=eff.coords["time"]).data, eff.isel(operator=2).data)


@pytest.mark.parametrize("kind", ["forward", "symmetric", "forward_const", "symmetric_const"])
@pytest.mark.parametrize("step", [1, 2])
def test_settings_are_per_cell_and_distinguish_identities(kind, step):
    _, _, diagonal, _ = rotated_series()
    eff = effective_energy(diagonal, kind=kind, step=step, time_extent=20, subtract_const=.001, on_fail="nan")
    info = eff.infos[0, 0]
    element = info.element
    assert (element.kind, element.step, element.time_extent, element.subtract_const) == (kind, step, 20, .001)
    changed = AtObsInfo(replace(element, time_extent=22), "time", info.time, ensemble_info=info.ensemble_info)
    assert changed != info and changed.name != info.name
    assert element_from_record(element.to_record()) == element


def test_multiple_pivots_coexist_as_effective_cells_but_raw_keys_require_namespaces(tmp_path):
    matrix, _ = spectrum()
    p, q = pivot(matrix), pivot(matrix, diagonalize_time=4)
    assert p.identifier != q.identifier
    raw_p, raw_q = p.rotate(matrix), q.rotate(matrix)
    a, b = raw_p.values.infos[0, 0, 0], raw_q.values.infos[0, 0, 0]
    assert a.name == b.name and a.rot_idx == b.rot_idx == 0
    assert a != b and len({a, b}) == 2
    writer = SigmondWriter(create_backups=False)
    with pytest.raises(ValueError, match="Duplicate logical observable"):
        writer.write_hdf5(str(tmp_path / "raw.hdf5"), [raw_p.values[0, 0, 0], raw_q.values[0, 0, 0]], group="g")
    path = tmp_path / "raw_append.hdf5"
    writer.write_hdf5(str(path), [raw_p.values[0, 0, 0]], group="g")
    with pytest.raises(ValueError, match="rotation bindings differ"):
        writer.write_hdf5(str(path), [raw_q.values[0, 0, 0]], group="g", mode="a", overwrite=True)
    ea = effective_energy(raw_p.diagonal(component="real")).isel(operator=0)
    eb = effective_energy(raw_q.diagonal(component="real")).isel(operator=0)
    assert ea[0].observable_info.name != eb[0].observable_info.name
    path = tmp_path / "two_pivots.hdf5"
    writer.write_hdf5(str(path), [*ea.to_collection(), *eb.to_collection()], group="g")
    loaded = SigmondLoader(str(path)).observables
    assert len(loaded) == 2 * ea.shape[0]
    assert {cell.observable_info.element.source.sink_rotation.pivot_id for cell in loaded} == {p.identifier, q.identifier}


def test_rotation_bindings_persist_on_raw_native_keys(tmp_path):
    matrix, _ = spectrum()
    p = pivot(matrix)
    raw = p.rotate(matrix).values[2, 1, 2]
    info = raw.observable_info
    assert info.rot_idx is None
    path = tmp_path / "native_rotated.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [raw], group="g")
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored == info and restored.key == info.key
    assert restored.sink_rotation == RotationBinding(p.identifier, 1)
    assert restored.source_rotation == RotationBinding(p.identifier, 2)


def test_selection_transpose_and_exact_transforms_preserve_bindings_mixing_clears():
    matrix, _ = spectrum()
    rotated = pivot(matrix).rotate(matrix)
    assert rotated.select([rotated.operators[2], rotated.operators[0]]).values[0, 0, 0].observable_info.rot_idx == 2
    permutation = [2, 1, 0]
    selected = rotated.transform(np.eye(3)[:, permutation], [rotated.operators[i] for i in permutation])
    assert selected.values[0, 0, 0].observable_info.rot_idx == 2
    mixed = rotated.transform(np.ones((3, 2)), name="MIX")
    assert all(info.sink_rotation is None and info.source_rotation is None for info in mixed.values.infos.flat)
    assert all(info.rot_idx is None for info in matrix.values.infos.flat)
    raw_rot_name = matrix.transform(np.eye(3), name="ROT")
    assert raw_rot_name.transformation is not None
    assert all(info.rot_idx is None for info in raw_rot_name.values.infos.flat)
    eff = effective_energy(rotated.diagonal(component="real")).transpose("operator", "time")
    scalar = eff.sel(time=3).isel(operator=2)
    assert scalar.observable_info.time == 3 and scalar.observable_info.rot_idx == 2


@pytest.mark.parametrize("field", ["flags", "component", "binding"])
def test_inconsistent_explicit_sources_are_diagnosed(field):
    _, _, diagonal, _ = rotated_series()
    infos = diagonal.infos.copy()
    changed = infos[1, 0].copy()
    if field == "flags":
        changed = CorrelatorObsInfo(key=CorrelatorAtTimeInfo(changed.element, changed.time, hermitian=False),
                                    ensemble_info=changed.ensemble_info,
                                    sink_rotation=changed.sink_rotation, source_rotation=changed.source_rotation)
    elif field == "component":
        changed.re_im = "im"
    else:
        changed = changed.with_rotation(None, None)
    infos[1, 0] = changed
    with pytest.raises(ValueError, match="inconsistent source element/context"):
        effective_energy(diagonal._with_replaced_infos(infos))


def test_generic_series_snapshots_use_named_coordinates_and_are_immutable():
    matrix, _ = spectrum()
    times = np.arange(6)
    parent = ArrayObsInfo("numeric_parent", ensemble_info=matrix.values.ensemble_info)
    values = np.exp(-times[:, None, None] * np.array([.2, .5])[None, :, None]) * np.ones((1, 1, 13))
    data = SigmondDataArray.from_data(values, parent, matrix.values.sampling_info,
                                     dim_names=["time", "mode"], coords={"time": times, "mode": ["a", "b"]})
    eff = effective_energy(data)
    a, b = eff.infos[0]
    assert isinstance(a.element.source, SeriesInfo)
    assert a.element.source.coordinates == (("mode", "a"),)
    assert b.element.source.coordinates == (("mode", "b"),)
    assert a.element.sector_info == SectorInfo() and a.rot_idx is None
    before = a.name
    parent.name = "changed_parent"
    parent.psq = 3
    assert a.name == before
    with pytest.raises(FrozenInstanceError):
        a.element.kind = "symmetric"
    with pytest.raises(FrozenInstanceError):
        a.element.source.coordinates = ()


def test_at_coordinate_typed_values_labels_and_storage_names():
    element = SeriesInfo(json.dumps({"source": "snapshot"}), label=r"F_{x}({VAR})")
    a = AtObsInfo(element, "time", np.int64(8))
    b = AtObsInfo(element, "time", 8)
    floating = AtObsInfo(element, "time", 8.)
    assert a == b and hash(a) == hash(b) and a.name == b.name
    assert a != floating and a.name != floating.name
    assert a.time == 8 and a.latex_str == r"F_{x}(8)"
    before = a.name
    a.latex_str = r"\mathrm{custom}"
    assert a == b and hash(a) == hash(b) and a.name == before
    assert AtObsInfo(element, "x", .125).time is None
    assert len(a.name.split("ID")[-1]) == 32
    assert a.name != AtObsInfo(element, "time", 8, index=1).name
    with pytest.raises(ValueError, match="generated"):
        a.name = "renamed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_and_constants_reject(value):
    source = SeriesInfo('{}')
    with pytest.raises(ValueError, match="finite"):
        AtObsInfo(source, "time", value)
    with pytest.raises(ValueError, match="finite"):
        EffectiveEnergyInfo(source, subtract_const=value)
    matrix, _ = spectrum()
    with pytest.raises(ValueError, match="finite"):
        effective_energy(matrix.diagonal(component="real"), subtract_const=value)


def test_known_target_is_a_snapshot_and_reconstructs_independent_energy_infos():
    matrix, _ = spectrum()
    original = EnergyObsInfo.from_channel(matrix.info.channel, ensemble_info=matrix.values.ensemble_info,
                                          rot_idx=1)
    target = EnergyTarget.from_energy(original)
    element = EffectiveEnergyInfo(SeriesInfo('{}'), target=target)
    original.psq = 2
    one, two = element.energy, element.energy
    assert one is not two and one.psq == two.psq == 0 and one.rot_idx == 1
    assert one.level_index is None and one.energy_type == "elab"
    one.psq = 3
    assert two.psq == 0
    assert element_from_record(element.to_record()) == element


@pytest.mark.parametrize("failure", ["version", "kind", "source_kind", "sector"])
def test_nested_codec_rejects_unknown_or_contradictory_records(failure):
    _, _, diagonal, _ = rotated_series()
    record = effective_energy(diagonal).infos[0, 0].element.to_record()
    if failure == "version":
        record["metadata_version"] = 99
    elif failure == "kind":
        record["element_kind"] = "arbitrary_callable"
    elif failure == "source_kind":
        record["source"]["source_kind"] = "arbitrary_callable"
    else:
        record["sector_info"]["twoI"] = 2
    with pytest.raises(ValueError, match="Unsupported|contradicts"):
        element_from_record(record)


def test_wrapper_codec_rejects_version_kind_identity_and_name_changes():
    a = AtObsInfo(SeriesInfo('{}'), "time", 3)
    base = ObservableInfo(a.name)
    for attrs in ({**a.to_attrs(), "metadata_version": 2},
                  {**a.to_attrs(), "value": 4},
                  {**a.to_attrs(), "element": {"element_kind": "unknown"}}):
        with pytest.raises(ValueError):
            AtObsInfo.from_attrs(base, attrs)
    with pytest.raises(ValueError, match="storage name"):
        AtObsInfo.from_attrs(ObservableInfo("wrong"), a.to_attrs())
    with pytest.raises(ValueError, match="Unsupported observable"):
        SigmondLoader._apply_obs_attrs(base, {"obs_kind": "unknown", "metadata_version": 1})


def test_generated_id_collision_rejects_batch_and_append_even_with_overwrite(tmp_path, monkeypatch):
    import sigmondsamplings.metadata as metadata
    from sigmondsamplings.sampling import SigmondSampling
    matrix, _ = spectrum()
    monkeypatch.setattr(metadata, "generated_name", lambda *args: "At_ID" + "0" * 32)
    a = AtObsInfo(SeriesInfo('{}'), "time", 1, ensemble_info=matrix.values.ensemble_info)
    b = AtObsInfo(SeriesInfo('{}'), "time", 2, ensemble_info=matrix.values.ensemble_info)
    def wrap(info):
        return SigmondSampling(np.ones(13), info, matrix.values.sampling_info)
    collection = ObservableCollection([wrap(a), wrap(b)])
    assert len(collection) == 2
    writer = SigmondWriter(create_backups=False)
    path = tmp_path / "collision.hdf5"
    with pytest.raises(ValueError, match="Duplicate logical"):
        writer.write_hdf5(str(path), list(collection), group="g")
    writer.write_hdf5(str(path), [wrap(a)], group="g")
    with pytest.raises(ValueError, match="Generated storage identity collision"):
        writer.write_hdf5(str(path), [wrap(b)], group="g", mode="a", overwrite=True)


def test_snapshot_hashing_does_not_read_sample_buffers(monkeypatch):
    _, _, diagonal, _ = rotated_series()
    eff = effective_energy(diagonal)
    info = eff.infos[0, 0]
    def fail(*args, **kwargs):
        raise AssertionError("sample buffers were read")
    monkeypatch.setattr(type(eff), "_as_numpy", fail)
    assert hash(info) == hash(info.copy())
    assert info.name == info.copy().name
    assert info.to_attrs() == info.copy().to_attrs()


def test_old_effective_array_api_is_removed():
    import sigmondsamplings.correlators as correlators
    assert "EffectiveEnergyObsInfo" not in correlators.__all__
    assert not hasattr(correlators, "EffectiveEnergyObsInfo")


def test_offdiagonal_elements_preserve_both_sources_without_a_scalar_mode():
    matrix, _ = spectrum()
    p = pivot(matrix)
    rotated = p.rotate(matrix)
    data = rotated.correlator(rotated.operators[0], rotated.operators[1]).component("real")
    eff = effective_energy(data, on_fail="nan")
    element = eff.infos[0].element
    assert element.rot_idx is None and element.energy is None
    assert element.source.correlator.sink == rotated.operators[0]
    assert element.source.correlator.source == rotated.operators[1]
    assert element.source.sink_rotation == RotationBinding(p.identifier, 0)
    assert element.source.source_rotation == RotationBinding(p.identifier, 1)
    assert element.psq == 0 and element.irrep == "A1g" and element.isospin == "singlet"
    assert element.Q is None and element.strangeness == 0


def test_coordinate_and_constant_identities_do_not_round():
    source = SeriesInfo('{}')
    a = AtObsInfo(source, "x", .1)
    b = AtObsInfo(source, "x", np.nextafter(.1, 1.))
    assert a != b and a.name != b.name
    e = EffectiveEnergyInfo(source, subtract_const=.1)
    f = replace(e, subtract_const=np.nextafter(.1, 1.))
    assert AtObsInfo(e, "time", 3).name != AtObsInfo(f, "time", 3).name
    encoded = SeriesInfo(json.dumps({"number_type": "float", "hex": (.1).hex()}))
    numeric = SeriesInfo(json.dumps(.1))
    assert encoded != numeric


def test_generic_element_labels_round_trip_outside_identity(tmp_path):
    from sigmondsamplings.sampling import SigmondSampling
    matrix, _ = spectrum()
    element = SeriesInfo('{}', label=r"F_{x}({var})")
    info = AtObsInfo(element, "x", .1, ensemble_info=matrix.values.ensemble_info,
                     latex_str=r"\mathrm{override}")
    equal = AtObsInfo(replace(element, label="different label"), "x", .1, ensemble_info=info.ensemble_info)
    assert info == equal and hash(info) == hash(equal) and info.name == equal.name
    path = tmp_path / "labels.hdf5"
    data = SigmondSampling(np.ones(13), info, matrix.values.sampling_info)
    SigmondWriter(create_backups=False).write_hdf5(str(path), [data], group="g")
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored == info and restored.latex_str == info.latex_str
    assert restored.element.label == element.label


def test_duplicate_target_coordinates_reject_on_reconstruction():
    _, _, diagonal, _ = rotated_series()
    a = effective_energy(diagonal, step=1).isel(operator=0)
    b = effective_energy(diagonal, step=2).isel(operator=0)
    with pytest.raises(ValueError, match="Multiple samplings map to cell"):
        SigmondDataArray.from_collection([*a.to_collection(), *b.to_collection()], shape_by=["time"])


def test_multiple_settings_coexist_in_one_hdf5_file(tmp_path):
    _, _, diagonal, _ = rotated_series()
    arrays = [effective_energy(diagonal, kind=kind, step=step, time_extent=extent,
                               subtract_const=const, on_fail="nan").isel(operator=0)
              for kind, step, extent, const in [("forward", 1, 16, 0.), ("forward", 2, 20, .001),
                                                ("forward_const", 1, 16, 0.)]]
    items = [cell for array in arrays for cell in array.to_collection()]
    path = tmp_path / "settings.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), items, group="g")
    restored = SigmondLoader(str(path)).observables
    assert {cell.observable_info for cell in restored} == {cell.observable_info for cell in items}


def test_full_identity_persists_and_tampered_record_rejects(tmp_path):
    import h5py
    from sigmondsamplings.io import obsmeta
    _, _, diagonal, _ = rotated_series()
    cell = effective_energy(diagonal)[0, 0]
    path = tmp_path / "identity.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [cell], group="g")
    with h5py.File(path, "a") as file:
        table = obsmeta.read(file["g"])
        record = next(iter(table.values()))
        assert record["identity_record"] == cell.observable_info.identity_record
        assert "sector_info" not in record  # the wrapper forwards its element's sector
        record["element"]["step"] = 2
        obsmeta.write(file["g"], table)
    with pytest.raises(ValueError, match="identity record"):
        SigmondLoader(str(path))


def test_source_components_distinguish_effective_cells_and_round_trip(tmp_path):
    _, _, diagonal, _ = rotated_series()
    complex_data = diagonal.to_complex()
    complex_data.data.imag[...] = diagonal.data
    real = effective_energy(complex_data.component("real"))
    imag = effective_energy(complex_data.component("imag"))
    assert real.infos[0, 0].element.source.re_im == "re"
    assert imag.infos[0, 0].element.source.re_im == "im"
    assert real.infos[0, 0] != imag.infos[0, 0]
    path = tmp_path / "components.hdf5"
    items = [real[0, 0], imag[0, 0]]
    SigmondWriter(create_backups=False).write_hdf5(str(path), items, group="g")
    restored = SigmondLoader(str(path)).observables
    assert {cell.observable_info for cell in restored} == {cell.observable_info for cell in items}


def test_identity_projection_does_not_expose_mutable_ensemble_state():
    matrix, _ = spectrum()
    info = AtObsInfo(SeriesInfo('{}'), "time", 1, ensemble_info=matrix.values.ensemble_info)
    record = info.identity_record
    record["context"]["ensemble"]["tweak_info"]["rebin"] = 99
    assert info.ensemble_info.tweak_info["rebin"] == 1
    assert info.public_attributes()["time"] == 1


def test_obsmeta_rejects_conflicting_duplicate_records(tmp_path):
    import h5py
    from sigmondsamplings.io import obsmeta
    path = tmp_path / "duplicate_meta.hdf5"
    with h5py.File(path, "w") as file:
        rows = [json.dumps({"key": "same", "identity_record": {"value": value}}) for value in (1, 2)]
        file.create_dataset("ObsMeta", data=rows, dtype=h5py.string_dtype("utf-8"))
        with pytest.raises(ValueError, match="Conflicting metadata records"):
            obsmeta.read(file)


def test_opaque_or_renamed_native_keys_do_not_claim_a_rotated_diagonal():
    matrix, _ = spectrum()
    info = pivot(matrix).rotate(matrix).values[0, 0, 0].observable_info.copy()
    binding = info.sink_rotation
    assert info.rot_idx == 0
    unknown = info.name.replace("A1g_1", "A9_1")
    opaque = CorrelatorObsInfo(unknown, sink_rotation=binding, source_rotation=binding)
    assert opaque.key is None and opaque.rot_idx is None
    info.name = unknown
    assert info.sink_rotation is None and info.source_rotation is None and info.rot_idx is None


def test_version_checks_apply_to_nested_sources_and_registered_key_annotations():
    _, _, diagonal, _ = rotated_series()
    record = effective_energy(diagonal).infos[0, 0].element.to_record()
    record["source"]["metadata_version"] = 99
    with pytest.raises(ValueError, match="Unsupported metadata version"):
        element_from_record(record)
    info = diagonal.infos[0, 0]
    with pytest.raises(ValueError, match="Unsupported metadata version"):
        SigmondLoader._apply_obs_attrs(info, {**info.to_attrs(), "metadata_version": 99})


def test_effective_generic_source_label_is_a_persisted_presentation_field():
    element = EffectiveEnergyInfo(SeriesInfo('{}', label=r"C_{\mathrm{custom}}"))
    info = AtObsInfo(element, "time", 3)
    attrs = info.to_attrs()
    assert "source_label" in attrs and "source_label" not in attrs["identity_record"]
    restored = AtObsInfo.from_attrs(ObservableInfo(info.name), attrs)
    assert restored == info and restored.element.source.label == element.source.label


def test_known_target_preserves_reference_normalization():
    matrix, _ = spectrum()
    original = EnergyObsInfo(sector_info=matrix.info.channel.to_sector(),
                             ensemble_info=matrix.values.ensemble_info, energy_type="elab", ref_particle="N")
    element = EffectiveEnergyInfo(SeriesInfo('{}'), target=EnergyTarget.from_energy(original))
    restored = element_from_record(element.to_record())
    assert restored.energy.ref_particle == "N" and restored.energy.level_index is None
    assert restored.energy.ensemble_info == original.ensemble_info
