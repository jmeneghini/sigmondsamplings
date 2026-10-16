"""Arithmetic identity is independent of values and consistent for scalar/N-D data."""
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from sigmondsamplings import DerivedElement, DerivedObsInfo
from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.energy_levels import EnergyObsInfo, SHEnergyObsInfo
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo, SectorInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.metadata import AtObsInfo, SeriesInfo
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.sampling_array import SigmondDataArray

ENS = EnsembleInfo("arithmetic", 3, spatial_extent=8, temporal_extent=32)
BOOT = SamplingInfo("bootstrap", 2, seed=7)
SECTOR = SectorInfo(psq=0, irrep="A1g", twoI=0, S=0)


def sampling(info=None, values=(2., 1.8, 2.2)):
    if info is None:
        info = EnergyObsInfo(ensemble_info=ENS, sector_info=SECTOR, energy_type="elab", level_index=2, rot_idx=3)
    return SigmondSampling(np.array(values), info, BOOT)


OPERATIONS = [
    lambda e: -e, lambda e: +e, lambda e: np.conj(e), lambda e: 2*e,
    lambda e: 2/e, lambda e: e**2, lambda e: np.log(e), lambda e: np.sqrt(e),
    lambda e: abs(e), lambda e: e/e, lambda e: e+e, lambda e: e*1, lambda e: e+0,
]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_every_value_transform_derives_metadata_and_matches_scalar_cells(operation):
    source = sampling()
    array = SigmondDataArray.from_samplings([source], dim_names=["state"])
    scalar_result, array_result = operation(source), operation(array)
    result = scalar_result.observable_info
    assert isinstance(result, DerivedObsInfo) and result != source.observable_info
    assert result.sector_info == SECTOR and not result.is_energy and not result.is_single_hadron
    assert getattr(result, "level_index", None) is None and getattr(result, "rot_idx", None) is None
    assert array_result[0].observable_info == result
    np.testing.assert_array_equal(array_result[0].data, scalar_result.data)
    assert "(" in result.latex_str
    assert source.observable_info.level_index == 2 and source.observable_info.rot_idx == 3


def test_source_and_negation_coexist_and_order_and_exact_constants_affect_identity():
    e = sampling()
    assert len(ObservableCollection([e, -e])) == 2
    assert (e+1).observable_info != (1+e).observable_info
    assert (e-1).observable_info != (1-e).observable_info
    assert (e*1.000000000000001).observable_info != (e*1.000000000000002).observable_info
    assert (e*1).observable_info != (e*1.).observable_info
    assert (e*0.).observable_info != (e*-0.).observable_info
    assert (-e).observable_info == (-e.copy()).observable_info
    assert (-e).observable_info.name == (-e.copy()).observable_info.name
    assert (e*2).observable_info == (e*np.int64(2)).observable_info
    assert (e+e).observable_info != e.observable_info


def test_numpy_array_arguments_are_snapshotted_by_value():
    e = sampling()
    factors = np.array([1., 2., 3.])
    first = (e*factors).observable_info
    assert first == (e*factors.copy()).observable_info
    factors[0] = 4
    assert first != (e*factors).observable_info
    assert json.loads(first.derivation.operands[1])["kind"] == "array"
    assert first != (e*factors.astype(np.float32)).observable_info


def test_only_equal_known_sector_fields_survive_binary_operations():
    a = sampling()
    b = sampling(EnergyObsInfo("b", ensemble_info=ENS, energy_type="ecm",
                              sector_info=replace(SECTOR, psq=1, S=None)))
    sector = (a+b).observable_info.sector_info
    assert sector == replace(SECTOR, psq=None, S=None)
    assert (-b).observable_info.sector_info == b.observable_info.sector_info


def test_broadcast_and_transposed_arrays_match_corresponding_scalar_operations():
    a, b = sampling(), sampling(ObservableInfo("b", ensemble_info=ENS))
    left = SigmondDataArray.from_samplings([a,b], dim_names=["left"], coords={"left":["a","b"]})
    right = SigmondDataArray.from_samplings([b,a], dim_names=["right"], coords={"right":[4,5]})
    result = (left-right).transpose("right", "left")
    for i in range(2):
        for j in range(2):
            expected = left[i]-right[j]
            assert result[j,i].observable_info == expected.observable_info
            np.testing.assert_array_equal(result[j,i].data, expected.data)
    assert (left+a)[1].observable_info == (b+a).observable_info


def time_series():
    element = SeriesInfo('{"source":"C"}', sector_info=SECTOR, label="C({VAR})")
    cells = [sampling(AtObsInfo(element, "time", t, ensemble_info=ENS)) for t in (2,4,3)]
    return SigmondDataArray.from_collection(cells, shape_by=["time"])


@pytest.mark.parametrize("operation", [lambda a:-a, lambda a:np.log(a), lambda a:a/a, lambda a:2*a])
def test_time_aligned_arithmetic_builds_derived_elements_and_at_cells(operation):
    source = time_series()
    result = operation(source)
    assert isinstance(result.ts.elements.item(), DerivedElement)
    assert result.ts.elements.item().rot_idx is None
    for t in source.coords["time"]:
        cell = result.ts.at(t)
        assert isinstance(cell.observable_info, AtObsInfo) and cell.observable_info.time == t
        assert cell.observable_info == operation(source.ts.at(t)).observable_info
        assert cell.observable_info.sector_info == SECTOR
    window = result.ts.window(2,3)
    assert set(window.ts.times) == {2,3}
    assert not result.ts.at(2).observable_info.is_energy


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("at_time", [False, True])
def test_derived_metadata_and_labels_round_trip_in_ss_files(tmp_path, lazy, at_time):
    result = -time_series() if at_time else -sampling()
    cells = list(result.to_collection()) if at_time else [result]
    path = tmp_path / "derived.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), cells, group="g")
    restored = SigmondLoader(str(path), lazy=lazy).observables
    original = {s.observable_info.name:s for s in cells}
    for cell in restored:
        prior = original[cell.observable_info.name]
        assert cell.observable_info == prior.observable_info
        assert cell.observable_info.latex_str == prior.observable_info.latex_str
        assert cell.observable_info.to_attrs() == prior.observable_info.to_attrs()
        np.testing.assert_array_equal(cell.data, prior.data)
    if at_time:
        assert isinstance(SigmondDataArray.from_collection(restored, shape_by=["time"]).ts.elements.item(), DerivedElement)


def test_labels_never_enter_derived_identity_even_in_later_operations():
    e = sampling()
    labeled = e.copy()
    labeled.observable_info.latex_str = "custom E"
    assert (-e).observable_info == (-labeled).observable_info
    a,b = -e,-labeled
    assert a.observable_info.latex_str != b.observable_info.latex_str
    a.observable_info.latex_str = "custom result"
    assert (a*2).observable_info == (b*2).observable_info
    series = time_series()
    changed = series.obs.replace(latex_str="other C")
    assert (-series).ts.at(2).observable_info == (-changed).ts.at(2).observable_info
    assert (2*-series).ts.at(2).observable_info == (2*-changed).ts.at(2).observable_info


def test_snapshot_does_not_hold_live_input_metadata_and_frozen_elements_reject_edits():
    e = sampling()
    result = -e
    before = result.observable_info.name
    e.observable_info.level_index = 9
    assert result.observable_info.name == before
    with pytest.raises(FrozenInstanceError):
        result.observable_info.derivation.operation = "other"
    with pytest.raises(AttributeError):
        result.observable_info.psq = 2
    with pytest.raises(ValueError):
        result.observable_info.name = "fake"


def test_explicit_reference_normalization_retains_energy_role_and_input_metadata():
    e = sampling()
    mass = sampling(SHEnergyObsInfo(hadron="pi", psq=0, ensemble_info=ENS))
    ratio = e.create_ref_sampling(mass)
    assert isinstance(ratio.observable_info, EnergyObsInfo)
    assert ratio.observable_info.ref_particle == "pi" and ratio.observable_info.level_index == 2
    assert ratio.observable_info.rot_idx == 3 and ratio.observable_info.sector_info == SECTOR
    assert e.observable_info.ref_particle is None
    assert isinstance((e/mass).observable_info, DerivedObsInfo)
    assert isinstance((-e).create_ref_sampling(mass).observable_info, DerivedObsInfo)


def test_bin_statistics_on_derived_inputs_match_scalar_and_array_metadata():
    bins = SigmondBins(np.array([1.,2.,3.]), ObservableInfo("bins", ensemble_info=ENS))
    derived = -bins
    array = SigmondDataArray.from_samplings([derived], dim_names=["state"])
    for statistic in ("mean", "variance"):
        scalar = derived.resample(BOOT, statistic=statistic)
        batched = array.resample(BOOT, statistic=statistic)
        assert isinstance(scalar.observable_info, DerivedObsInfo)
        assert batched[0].observable_info == scalar.observable_info
        np.testing.assert_array_equal(batched[0].data, scalar.data)
    assert bins.observable_info.name == "bins"


def test_editor_copies_derived_metadata_and_buffers():
    result = -SigmondDataArray.from_samplings([sampling()], dim_names=["state"])
    editor = result.editor()
    editor.replace_obs(latex_str="edited")
    editor.set_data(0, np.array([7.,8.,9.]))
    edited = editor.freeze()
    assert edited[0].observable_info == result[0].observable_info
    assert edited[0].observable_info.latex_str == "edited"
    assert result[0].observable_info.latex_str != "edited"
    np.testing.assert_array_equal(result.data, [[-2.,-1.8,-2.2]])


@pytest.mark.parametrize("input_kind", ["fixed", "sampled", "sequence"])
def test_model_returning_a_parameter_does_not_rename_it(input_kind):
    from sigmondsamplings.fitting.model_func import SigmondModelFunc
    parameter = sampling(ObservableInfo("A", ensemble_info=ENS))
    model = SigmondModelFunc(lambda x, A:A, ObservableCollection([parameter]))
    values = 2 if input_kind == "fixed" else parameter if input_kind == "sampled" else [parameter, parameter]
    result = model(values)
    results = result if isinstance(result, list) else [result]
    assert parameter.observable_info.name == "A"
    assert all(s is not parameter for s in results)
    assert all(s.observable_info is not parameter.observable_info for s in results)
    for s in results:
        np.testing.assert_array_equal(s.data, parameter.data)


def test_generic_numeric_time_arrays_create_durable_timewise_derivations():
    array = SigmondDataArray.from_data(np.arange(18).reshape(3,2,3)+1, "C", BOOT,
                ensemble_info=ENS, dim_names=["time","operator"], coords={"time":[2,4,3], "operator":[0,1]})
    result = array + 1j*array
    assert all(isinstance(info, AtObsInfo) for info in result.infos.flat)
    assert result.ts.elements.shape == (2,)
    assert result.component("imag").ts.elements.shape == (2,)
    assert result.ts.at(4).isel(operator=1).observable_info.time == 4
    np.testing.assert_array_equal(result.data, (1+1j)*array.data)


def test_transformed_effective_series_keeps_time_axis_context_and_clears_pivot_claims():
    from pivot_data import spectrum
    from sigmondsamplings.correlators import SinglePivot
    matrix,_ = spectrum()
    pivot = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                   min_inv_condition=1e-8)
    eff = pivot.rotate(matrix).diagonal(component="real").ts.effective_energy()
    result = -eff
    assert result.ts.axis.offset == eff.ts.axis.offset == .5
    assert result.ts.axis.extent == eff.ts.axis.extent
    assert all(info.rot_idx is None and not info.is_energy for info in result.infos.flat)
    assert result.ts.at(result.coords["time"][0]).isel(operator=0).observable_info == (
        -eff.ts.at(eff.coords["time"][0]).isel(operator=0)).observable_info


def test_differing_source_time_offsets_reject_a_common_derived_offset():
    from sigmondsamplings.metadata import EffectiveEnergyInfo
    source = SeriesInfo('{"C":1}', sector_info=SECTOR)
    a,b = [],[]
    for t in (2,3):
        a.append(sampling(AtObsInfo(source,"time",t,ensemble_info=ENS)))
        b.append(sampling(AtObsInfo(EffectiveEnergyInfo(source,"forward",1,32,False),"time",t,ensemble_info=ENS)))
    left = SigmondDataArray.from_collection(a,shape_by=["time"])
    right = SigmondDataArray.from_collection(b,shape_by=["time"])
    result = left+right
    assert isinstance(result.ts.elements.item(), DerivedElement)
    with pytest.raises(ValueError, match="offset"):
        _ = result.ts.axis.offset


@pytest.mark.parametrize("at_time", [False,True])
def test_generated_derivation_codecs_reject_contradictions_and_unknown_versions(at_time):
    result = (-time_series()).ts.at(2) if at_time else -sampling()
    info = result.observable_info
    attrs = info.to_attrs()
    base = ObservableInfo(info.name,ensemble_info=ENS)
    attrs["metadata_version"] = 99
    with pytest.raises(ValueError,match="version"):
        type(info).from_attrs(base,attrs)
    attrs = info.to_attrs()
    key = "element" if at_time else "derivation"
    attrs[key]["operation"] = "other"
    with pytest.raises(ValueError,match="identity|name"):
        type(info).from_attrs(base,attrs)


@pytest.mark.parametrize("empty", [False,True])
def test_empty_or_mixed_parent_time_arrays_still_allow_arithmetic(empty):
    left = SigmondDataArray.from_data(np.ones((2,3)), "left", BOOT, ensemble_info=ENS,
                                    dim_names=["time"],coords={"time":[0,1]})
    right = SigmondDataArray.from_data(np.ones((2,3)), "right", BOOT, ensemble_info=ENS,
                                    dim_names=["time"],coords={"time":[2,3]})
    array = left.isel(time=[]) if empty else np.concatenate([left,right])
    result = -array
    np.testing.assert_array_equal(result.data,-array.data)
    assert result.shape == array.shape
    if empty:
        assert len(result.ts.times) == 0
    else:
        with pytest.raises(ValueError,match="inconsistent"):
            _ = result.ts.elements


def test_stats_range_buffer_fallback_has_its_own_derivation():
    from sigmondsamplings.stats import SamplingStats
    a = sampling(ObservableInfo("a",ensemble_info=ENS))
    b = sampling(ObservableInfo("b",ensemble_info=ENS))
    low,high = SamplingStats([a,b]).min_and_max_val_with_buffer(.3)
    assert isinstance(low.observable_info,DerivedObsInfo)
    assert isinstance(high.observable_info,DerivedObsInfo)
    assert low.full_sample_value < 2 < high.full_sample_value
    assert a.observable_info.name == "a" and b.observable_info.name == "b"


@pytest.mark.parametrize("scalar_numerator", [False,True])
def test_array_reference_normalization_uses_scalar_domain_metadata(scalar_numerator):
    energy = sampling()
    mass = sampling(SHEnergyObsInfo(hadron="pi",psq=0,ensemble_info=ENS))
    numerator = energy if scalar_numerator else SigmondDataArray.from_samplings([energy,energy],dim_names=["state"])
    denominator = SigmondDataArray.from_samplings([mass,mass],dim_names=["mass"]) if scalar_numerator else mass
    result = numerator.create_ref_sampling(denominator)
    expected = energy.create_ref_sampling(mass)
    for cell in result.to_collection():
        assert cell.observable_info == expected.observable_info
        np.testing.assert_array_equal(cell.data,expected.data)
    assert energy.observable_info.ref_particle is None
