"""Durable sampled predictions, fixed identities and callable-free fit wrapping."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from sigmondsamplings import PredictionSeriesInfo
from sigmondsamplings.energy_levels import EnergyObsInfo
from sigmondsamplings.fitting.model_func import SigmondModelFunc
from sigmondsamplings.fitting.result import FitResult
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo, SectorInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.metadata import AtObsInfo, EnergyTarget
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.predictions import _normalized_buffer, _prediction_descriptor, _wrap_predictions
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.sampling_array import SigmondDataArray
from sigmondsamplings.stats import SamplingStats

ENS = EnsembleInfo("predictions", 3, spatial_extent=8, temporal_extent=32)
SAMP = SamplingInfo("bootstrap", 2, seed=7)
SECTOR = SectorInfo(psq=0,irrep="A1g",twoI=0)


def samp(name="A", values=(2.,1.8,2.2), index=0, **kwargs):
    return SigmondSampling(np.array(values), ObservableInfo(name,index,ensemble_info=ENS,**kwargs),SAMP)


def model(values=(2.,1.8,2.2), label="A{VAR}+1"):
    return SigmondModelFunc(lambda x,A:A*x+1,ObservableCollection([samp(values=values)]),latex_str=label)


def result(n=4):
    predictions = [samp("theory",(i+1.,i+.9,i+1.1),index=i) for i in range(n)]
    params = SamplingStats([samp()])
    return FitResult(optimize_result=SimpleNamespace(fun=1.,params={"A":2.},n_free=1),
                     params=params,chi_squared=samp("chi2"),theory=SamplingStats(predictions),
                     plotter=None,param_names=["A"],num_data_residuals=n,num_free_params=1,
                     is_synthetic=True,x=np.arange(n)+.25,x_name="time")


@pytest.mark.parametrize("x", [2.5,[3.,1.5,2.],np.array([.5,1.5]),samp("x"),[samp("x"),samp("y",(3.,2.8,3.2))]])
def test_model_series_uses_existing_evaluation_and_keeps_call_contract(x):
    m=model()
    before=m(x)
    series=m.series(x)
    predictions=before if isinstance(before,list) else [before]
    assert series.dims == ("time",)
    assert series.shape == (len(predictions),)
    np.testing.assert_array_equal(series.data,np.stack([p.data for p in predictions]))
    assert isinstance(series.ts.elements.item(),PredictionSeriesInfo)
    assert series.ts.elements.item().sector_info == SectorInfo()
    assert all(isinstance(i,AtObsInfo) and not i.is_energy for i in series.infos)
    assert m.params[0].observable_info.name == "A"
    assert type(m(x)) is type(before)


def test_model_evaluates_only_once_per_coordinate_and_fractional_times_are_preserved():
    calls=[]
    def func(x,A):
        calls.append(x)
        return A*x
    m=SigmondModelFunc(func,ObservableCollection([samp()]))
    series=m.series([.25,.75,1.25])
    assert len(calls)==3
    assert series.ts.at(.75).observable_info.time == .75
    np.testing.assert_array_equal(series.ts.window(.5,1.).ts.times,[.75])
    with pytest.raises(ValueError,match="integer"):
        series.ts.effective_energy()
    m.series([1,2],dim="momentum")
    assert len(calls)==5


def test_prediction_sets_with_same_parameter_names_stay_distinct():
    a=model().series([1,2])
    b=model((3.,2.8,3.2)).series([1,2])
    assert a.ts.elements.item().series_id != b.ts.elements.item().series_id
    assert a[0].observable_info != b[0].observable_info
    assert len(ObservableCollection([a[0],b[0]]))==2
    assert a[0].observable_info == model().series([1,2])[0].observable_info


def test_series_labels_are_outside_identity_and_explicit_overrides_work():
    a=model(label="A{VAR}").series([1,2])
    b=model(label="B{var}").series([1,2],coordinate_label="t")
    assert a.ts.elements.item()==b.ts.elements.item()
    assert a[0].observable_info == b[0].observable_info
    assert a[0].observable_info.latex_str != b[0].observable_info.latex_str
    assert model().series([1],label="")[0].observable_info.latex_str == ""
    assert (-a)[0].observable_info == (-b)[0].observable_info


def test_physics_is_explicit_and_targets_do_not_claim_fitted_energy_or_pivot_mode():
    m=model()
    energy=EnergyObsInfo(sector_info=SECTOR,ensemble_info=ENS,energy_type="elab",rot_idx=2)
    target=EnergyTarget.from_energy(energy)
    series=m.series([1,2],target=target)
    element=series.ts.elements.item()
    assert element.sector_info==SECTOR and element.target==target
    assert element.rot_idx is None and not series[0].observable_info.is_energy
    assert getattr(series[0].observable_info,"level_index",None) is None
    with pytest.raises(ValueError,match="contradict"):
        m.series([1],sector_info=replace(SECTOR,psq=1),target=target)
    assert m.series([1],sector_info=SECTOR).ts.elements.item().target is None


@pytest.mark.parametrize("lazy",[False,True])
@pytest.mark.parametrize("physical",[False,True])
def test_prediction_cells_round_trip_and_rebuild_without_model(tmp_path,lazy,physical):
    target=EnergyTarget.from_energy(EnergyObsInfo(sector_info=SECTOR,ensemble_info=ENS,energy_type="elab")) if physical else None
    series=model().series([.25,1.5,3.],label="f({VAR})",coordinate_label="t",target=target)
    path=tmp_path/"predictions.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path),series.to_collection(),group="g")
    loaded=SigmondLoader(str(path),lazy=lazy).observables
    rebuilt=SigmondDataArray.from_collection(loaded,shape_by=["time"]).sel(time=series.ts.times)
    assert rebuilt.ts.elements.item()==series.ts.elements.item()
    assert rebuilt.ts.elements.item().label=="f({VAR})"
    assert rebuilt.ts.elements.item().coordinate_label=="t"
    for t in series.ts.times:
        assert rebuilt.ts.at(t).observable_info.to_attrs()==series.ts.at(t).observable_info.to_attrs()
    np.testing.assert_array_equal(rebuilt.data,series.data)


def test_theory_series_preserves_existing_synthetic_predictions_and_labels_without_callable():
    fit=result()
    fit.theory[0].observable_info.latex_str="custom theory"
    original=np.stack([p.data.copy() for p in fit.theory])
    wrapped=fit.theory_series()
    np.testing.assert_array_equal(wrapped.data,original)
    assert fit.is_synthetic and wrapped[0].observable_info.latex_str=="custom theory"
    assert wrapped.ts.elements.item().sector_info==SectorInfo()
    assert wrapped.ts.elements.item()==fit.theory_series(label="different").ts.elements.item()
    assert fit.theory[0].observable_info.name=="theory"
    explicit=fit.theory_series(sector_info=SECTOR)
    assert explicit.ts.elements.item().sector_info==SECTOR
    assert not explicit[0].observable_info.is_energy


def test_reloaded_fit_wraps_predictions_exactly_in_coordinate_order(tmp_path):
    fit=result(12)
    path=tmp_path/"fit.hdf5"
    fit.write_params(path)
    fit.write_theory(path)
    loaded=FitResult.from_hdf5(path,theory_file=path)
    wrapped=loaded.theory_series()
    np.testing.assert_array_equal(wrapped.data,fit.theory_series().data)
    np.testing.assert_array_equal(wrapped.ts.times,fit.x)
    assert wrapped.ts.elements.item()==fit.theory_series().ts.elements.item()


@pytest.mark.parametrize("complex_values",[False,True])
def test_prediction_normalization_fixes_dtype_byte_order_layout_and_nan_payload(complex_values):
    values=np.array([[1.,np.nan],[2.,3.]])
    values.view(np.uint64)[0,1]=0x7ff8000000000042
    if complex_values:
        values=values.astype(complex)
        values.imag[1,1]=np.nan
    expected=_normalized_buffer(values)
    swapped=_normalized_buffer(np.asfortranarray(values.astype(values.dtype.newbyteorder(">"))))
    assert expected.tobytes()==swapped.tobytes()
    standard=values.copy()
    standard.real[np.isnan(standard.real)]=np.nan
    assert expected.tobytes()==_normalized_buffer(standard).tobytes()
    assert expected.flags.c_contiguous
    assert expected.dtype.str in ("<f8","<c16")


def test_metadata_hashing_never_reads_prediction_buffers_again():
    m=model()
    series=m.series([1,2])
    element=series.ts.elements.item()
    expected=hash(element)
    name=series[0].observable_info.name
    series.data[:]=9
    m.params[0].data[:]=8
    assert hash(element)==expected and series[0].observable_info.name==name
    with pytest.raises(FrozenInstanceError):
        element.series_id="0"*64
    assert element!=m.series([1,2]).ts.elements.item()


@pytest.mark.parametrize("coordinates",[[],[1,1],[np.nan],[np.inf],[[1,2]],[True]])
def test_invalid_coordinates_reject(coordinates):
    with pytest.raises((ValueError,TypeError)):
        model().series(coordinates)


@pytest.mark.parametrize("dim",["","sample",None])
def test_invalid_prediction_dimensions_reject(dim):
    with pytest.raises(ValueError):
        model().series([1],dim=dim)


def test_prediction_codec_rejects_unknown_versions_and_contradictions():
    cell=model().series([1])[0]
    attrs=cell.observable_info.to_attrs()
    base=ObservableInfo(cell.observable_info.name,ensemble_info=ENS)
    attrs["element"]["metadata_version"]=99
    with pytest.raises(ValueError,match="version"):
        AtObsInfo.from_attrs(base,attrs)
    attrs=cell.observable_info.to_attrs()
    attrs["element"]["series_id"]="0"*64
    with pytest.raises(ValueError,match="identity"):
        AtObsInfo.from_attrs(base,attrs)
    with pytest.raises(ValueError,match="dimension"):
        AtObsInfo(cell.observable_info.element,"momentum",1)


def test_shared_label_substitution_keeps_latex_braces_and_replaces_both_tokens():
    m=model(label=r"\frac{A{VAR}}{{var}+1}")
    assert m.get_latex_str_with_var("t")==r"\frac{At}{t+1}"
    assert m.series([1])[0].observable_info.latex_str==r"\frac{A1}{1+1}"


def test_nan_resamples_and_synthetic_predictions_are_wrapped_without_replacement():
    fit=result()
    fit.optimize_result=object()  # wrapping needs neither optimizer nor model
    fit.theory[0].data[1:]=np.nan
    fit.theory[1].data[2]=np.inf
    before=np.stack([p.data.copy() for p in fit.theory])
    wrapped=fit.theory_series()
    np.testing.assert_array_equal(wrapped.data,before)
    assert np.isnan(wrapped.data[0,1:]).all()
    assert np.isinf(wrapped.data[1,2])


def test_normalized_nan_payloads_produce_the_same_series_identifier():
    a=samp(values=(np.nan,1.,2.))
    b=samp(values=(np.nan,1.,2.))
    b.data.view(np.uint64)[0]=0xfff8000000000042
    assert _prediction_descriptor([a],[1],"time")==_prediction_descriptor([b],[1],"time")


def test_complex_model_predictions_preserve_every_sample():
    parameter=SigmondSampling(np.array([1+2j,2+3j,3+4j]),ObservableInfo("A",ensemble_info=ENS),SAMP,is_complex=True)
    m=SigmondModelFunc(lambda x,A:A*x,ObservableCollection([parameter]))
    series=m.series([.5,1.5])
    assert series.is_complex
    np.testing.assert_array_equal(series.data,np.stack([p.data for p in m([.5,1.5])]))


def test_non_time_dimension_uses_matching_at_coordinates():
    series=model().series([.5,1.5],dim="momentum")
    assert series.dims==("momentum",)
    assert series[0].observable_info.coord=="momentum" and series[0].observable_info.value==.5
    assert series[0].observable_info.time is None
    with pytest.raises(ValueError,match="time"):
        _=series.ts


def test_prediction_order_metadata_rejects_missing_or_duplicate_keys(tmp_path):
    import h5py
    fit=result()
    path=tmp_path/"fit.hdf5"
    fit.write_params(path)
    fit.write_theory(path)
    with h5py.File(path,"a") as handle:
        handle["theory"].attrs["PredictionOrder"]='[]'
    with pytest.raises(ValueError,match="order"):
        FitResult.from_hdf5(path,theory_file=path)


@pytest.mark.parametrize("lazy",[False,True])
def test_complex_prediction_cells_round_trip(tmp_path,lazy):
    p=SigmondSampling(np.array([1+2j,2+3j,3+4j]),ObservableInfo("A",ensemble_info=ENS),SAMP,is_complex=True)
    series=SigmondModelFunc(lambda x,A:A*x,ObservableCollection([p])).series([1,2])
    path=tmp_path/"complex.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path),series.to_collection(),group="g")
    restored=SigmondDataArray.from_collection(SigmondLoader(str(path),lazy=lazy).observables,shape_by=["time"])
    assert restored.is_complex and restored.ts.elements.item()==series.ts.elements.item()
    np.testing.assert_array_equal(restored.data,series.data)
