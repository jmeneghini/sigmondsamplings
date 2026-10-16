"""Computed temporal behavior and labels recovered from durable cell metadata."""
from dataclasses import replace

import numpy as np
import pytest
from pivot_data import spectrum

import slat.plotting as slp
from sigmondsamplings.correlators import SinglePivot
from sigmondsamplings.data import SigmondData
from sigmondsamplings.energy_levels import EnergyObsInfo, SHEnergyObsInfo, create_energy_obs_info
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo, SectorInfo
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.labels import coordinate_label, effective_energy_label, energy_symbol, normalized_energy_symbol
from sigmondsamplings.metadata import AtObsInfo, EffectiveEnergyInfo, SeriesInfo
from sigmondsamplings.plotting import SeriesPlotter, SeriesStyle
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.sampling_array import ArrayObsInfo

ENS = EnsembleInfo("time_test", 12, temporal_extent=32)
PLAN = SamplingInfo.jackknife_for(ENS)


def numeric(times=(0, 1, 2, 3, 4, 5), *, ens=ENS):
    times = np.asarray(times)
    values = np.exp(-.3 * times.astype(float))[:, None] * np.ones((1, 13))
    return SigmondData.from_data(values, ArrayObsInfo("C", ensemble_info=ens), PLAN,
                                 dim_names=["time"], coords={"time": times})


def annotated(times=(0, 1, 2, 3, 4), *, element=None):
    element = SeriesInfo('{}', label=r"F_{x}({VAR})") if element is None else element
    data = numeric(times)
    infos = np.empty(len(times), dtype=object)
    for i, time in enumerate(times):
        infos[i] = AtObsInfo(element, "time", time, ensemble_info=ENS)
    return data._with_replaced_infos(infos)


def effective_matrix():
    matrix, _ = spectrum()
    pivot = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-8)
    return pivot.rotate(matrix).diagonal(component="real").ts.effective_energy()


@pytest.mark.parametrize("times", [[0, 0], [0, float("nan")], [float("inf")], [True], ["0", "1"], [1j]])
def test_time_coordinates_must_be_unique_finite_real_numbers(times):
    data = SigmondData.from_data(np.ones((len(times), 13)), "C", PLAN,
                                 dim_names=["time"], coords={"time": times})
    with pytest.raises(ValueError, match="unique|finite real"):
        data.ts


def test_requires_named_time_with_coordinates():
    for dims, coords in [(["x"], {"x": [0, 1]}), (["time"], {})]:
        data = SigmondData.from_data(np.ones((2, 13)), "C", PLAN, dim_names=dims, coords=coords)
        with pytest.raises(ValueError, match="named time"):
            data.ts
    with pytest.raises(ValueError, match="named time"):
        numeric()[0].ts


def test_elements_for_generic_series_are_scalar_and_named_for_other_dims():
    data = numeric()
    element = data.ts.elements
    assert element.dims == () and isinstance(element.item(), SeriesInfo)
    assert element.item().coordinates == ()
    stacked = SigmondData.stack([data, data], dim_name="series", coords=["a", "b"])
    elements = stacked.ts.elements
    assert elements.dims == ("series",)
    assert elements.sel(series="a").item().coordinates == (("series", "a"),)
    assert stacked.ts.axis.offset == 0 and stacked.ts.axis.extent == 32


def test_raw_native_elements_are_time_independent_correlator_keys():
    matrix, _ = spectrum()
    elements = matrix.values.ts.elements
    assert elements.dims == ("sink", "source")
    assert elements.values[0, 1] == matrix.values.infos[0, 0, 1].element
    assert matrix.values.ts.axis.offset == 0 and matrix.values.ts.axis.extent == 16


def test_window_is_inclusive_unsorted_and_preserves_metadata():
    data = annotated((5, 1, 4, 0, 3))
    view = data.ts
    assert list(view.times) == [5, 1, 4, 0, 3]
    window = view.window(1, 4)
    assert list(window.coords["time"]) == [1, 4, 3]
    assert window[0].observable_info is data[1].observable_info
    assert list(view.window(3, 3).coords["time"]) == [3]
    cell = view.at(3)
    assert cell.observable_info is data.sel(time=3).observable_info and cell.observable_info.time == 3
    with pytest.raises(ValueError, match="reversed"):
        view.window(4, 1)


def test_floating_coordinates_stay_numeric_without_lattice_coercion():
    data = annotated((.25, .5, 1.25, 2.5))
    assert data.ts.at(.5).observable_info.time == .5
    assert list(data.ts.window(.5, 2).coords["time"]) == [.5, 1.25]
    with slp.record() as rec:
        data.ts.plot()
    np.testing.assert_array_equal(rec.first("errorbar").args[0], data.coords["time"])
    with pytest.raises(ValueError, match="integers"):
        data.ts.effective_energy()


def test_empty_time_axis_allows_coordinate_and_window_access_only():
    data = numeric().ts.window(20, 30)
    assert data.ts.times.size == 0 and data.ts.window(0, 1).shape == (0,)
    for query in [lambda: data.ts.elements, lambda: data.ts.axis.offset, lambda: data.ts.axis.extent]:
        with pytest.raises(ValueError, match="empty time axis"):
            query()


def test_effective_views_recover_metadata_after_transpose_selection_and_hdf5(tmp_path):
    data = effective_matrix()
    assert data.ts.axis.offset == .5 and data.ts.axis.extent == 16
    assert [e.rot_idx for e in data.ts.elements.values] == [0, 1, 2]
    transposed = data.transpose("operator", "time")
    assert transposed.ts.elements.dims == ("operator",)
    assert transposed.ts.elements.values[2] == data.ts.elements.values[2]
    cell = transposed.ts.at(3).isel(operator=2)
    assert cell.observable_info.time == 3 and cell.observable_info.rot_idx == 2
    assert cell.observable_info.element.time_offset == .5
    path = tmp_path / "time.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), list(data.to_collection()), group="g")
    loaded = SigmondLoader(str(path), lazy=True).observables
    rebuilt = SigmondData.from_collection(loaded.filter(rot_idx=2), shape_by=["time"])
    assert rebuilt.ts.axis.offset == .5 and rebuilt.ts.axis.extent == 16
    assert rebuilt.ts.elements.item() == data.ts.elements.values[2]
    assert rebuilt.ts.at(3).observable_info == cell.observable_info


def test_axis_is_computed_and_the_view_has_no_independent_state():
    data = numeric().ts.effective_energy()
    view = data.ts
    assert view.axis.offset == .5
    infos = data.infos.copy()
    for i, info in enumerate(infos):
        infos[i] = AtObsInfo(replace(info.element, step=2), "time", info.time, ensemble_info=ENS)
    data._infos = infos
    assert view.axis.offset == 1 and data.ts.axis.offset == 1
    assert "ts" not in vars(data) and "axis" not in vars(data)
    times = view.times
    times[0] = 100
    assert data.coords["time"][0] == 0


@pytest.mark.parametrize("extent", [None, 20])
def test_mixed_recorded_extents_raise_even_for_known_versus_unknown(extent):
    source = SeriesInfo('{}')
    a = annotated(element=EffectiveEnergyInfo(source, time_extent=16))
    b = annotated(element=EffectiveEnergyInfo(source, time_extent=extent))
    data = SigmondData.stack([a, b], dim_name="series")
    assert data.ts.axis.offset == .5
    with pytest.raises(ValueError, match="conflicting time axis extent"):
        data.ts.axis.extent


def test_generic_extent_uses_cell_ensembles_instead_of_mixed_array_parent():
    a = numeric()
    b = numeric(ens=EnsembleInfo("unknown", 12))
    data = SigmondData.stack([a, b], dim_name="series")
    with pytest.raises(ValueError, match="conflicting time axis extent"):
        data.ts.axis.extent


def mixed_offsets():
    a = numeric().ts.effective_energy(step=1)
    b = numeric().ts.effective_energy(step=2)
    common = b.coords["time"]
    return SigmondData.stack([a.sel(time=common), b], dim_name="series", coords=["one", "two"])


def test_mixed_offsets_raise_common_query_and_plot_each_series_separately():
    data = mixed_offsets()
    with pytest.raises(ValueError, match="conflicting time axis offset"):
        data.ts.axis.offset
    assert data.ts.elements.shape == (2,)
    with slp.record() as rec:
        data.ts.plot(style=SeriesStyle(dodge=0))
    bars = rec.of_kind("errorbar")
    np.testing.assert_array_equal(bars[0].args[0], data.coords["time"] + .5)
    np.testing.assert_array_equal(bars[1].args[0], data.coords["time"] + 1)
    with slp.record() as rec:
        data.ts.plot(x_offset=0., style=SeriesStyle(dodge=0))
    for bar in rec.of_kind("errorbar"):
        np.testing.assert_array_equal(bar.args[0], data.coords["time"])


def test_non_time_plot_never_applies_a_time_offset():
    data = mixed_offsets().ts.at(1)
    data = data._from_parts(data.data, data.observable_info, data.sampling_info, False,
                            ("series",), {"series": [0, 1]}, data.infos)
    with slp.record() as rec:
        SeriesPlotter(data, x="series").plot()
    np.testing.assert_array_equal(rec.first("errorbar").args[0], [0, 1])


@pytest.mark.parametrize("problem", ["element", "component", "time"])
def test_explicit_element_or_context_disagreement_is_diagnosed(problem):
    data = annotated(element=EffectiveEnergyInfo(SeriesInfo('{}'), time_extent=32))
    infos = data.infos.copy()
    if problem == "element":
        infos[1] = AtObsInfo(replace(infos[1].element, step=2), "time", 1, ensemble_info=ENS)
    elif problem == "component":
        infos[1] = infos[1].copy()
        infos[1].re_im = "im"
    else:
        infos[1] = AtObsInfo(infos[1].element, "time", 99, ensemble_info=ENS)
    invalid = data._with_replaced_infos(infos)
    with pytest.raises(ValueError, match="inconsistent|disagrees"):
        invalid.ts.elements
    with pytest.raises(ValueError, match="inconsistent|disagrees"):
        invalid.ts.plot(x_offset=0)


def test_explicit_extent_is_recorded_without_changing_selection_coordinates():
    data = numeric().ts.effective_energy(time_extent=64)
    assert data.ts.axis.extent == 64 and data.ts.axis.offset == .5
    assert data.ts.at(2).observable_info.time == 2
    assert data.ts.elements.item().time_extent == 64
    with pytest.raises(KeyError):
        data.ts.at(2.5)


def test_shared_labels_replace_only_literal_tokens_and_preserve_braces():
    label = r"\frac{A_{0}}{{VAR}}+F_{x}({var})"
    assert coordinate_label(label, "t") == r"\frac{A_{0}}{t}+F_{x}(t)"
    assert coordinate_label(label, .125) == r"\frac{A_{0}}{0.125}+F_{x}(0.125)"
    assert energy_symbol(hadron="pi", psq=0) == r"M_{\pi}"
    assert normalized_energy_symbol("E", "ref") == r"E/M_{\text{ref}}"
    assert "ROT" in effective_energy_label(rot_idx=2)
    assert "a_t" not in effective_energy_label(ref_particle="N", lattice=True)


@pytest.mark.parametrize("kind", ["ordinary", "energy", "single_hadron", "correlator"])
@pytest.mark.parametrize("override", [r"\frac{E_{0}}{M_{\pi}}", ""])
def test_label_overrides_round_trip_and_leave_identity_unchanged(tmp_path, kind, override):
    matrix, _ = spectrum()
    info = {
        "ordinary": ObservableInfo("ordinary", ensemble_info=ENS),
        "energy": EnergyObsInfo(sector_info=SectorInfo(psq=0, irrep="A1g"), energy_type="elab", ensemble_info=ENS),
        "single_hadron": SHEnergyObsInfo(particle="pi", psq=0, ensemble_info=ENS),
        "correlator": matrix.values.infos[0, 0, 0].copy(),
    }[kind]
    initial = info.copy()
    info.latex_str = override
    assert info == initial and hash(info) == hash(initial)
    if isinstance(info, EnergyObsInfo):
        assert info.specify_latex_str() == override
        assert create_energy_obs_info(info).latex_str == override
    path = tmp_path / "override.hdf5"
    data = SigmondSampling(np.ones(13), info, PLAN)
    SigmondWriter(create_backups=False).write_hdf5(str(path), [data], group="g")
    restored = SigmondLoader(str(path)).observables[0].observable_info
    assert restored.latex_str == override and restored == info


def test_time_series_uses_symbolic_element_labels_and_output_overrides():
    data = annotated()
    assert data.ts.at(2).observable_info.latex_str == r"F_{x}(2)"
    with slp.record() as rec:
        data.ts.plot()
    assert rec.first("errorbar").kwargs["label"] == r"$F_{x}(t)$"
    for info in data.infos:
        info.latex_str = r"G_{x}(t)"
    with slp.record() as rec:
        data.ts.plot()
    assert rec.first("errorbar").kwargs["label"] == r"$G_{x}(t)$"


def test_reference_normalized_effective_label_has_no_extra_lattice_factor():
    from sigmondsamplings.metadata import EnergyTarget
    target = EnergyObsInfo("target", ensemble_info=ENS, ref_particle="N", energy_type="elab")
    element = EffectiveEnergyInfo(SeriesInfo('{}'), time_extent=32, target=EnergyTarget.from_energy(target))
    data = annotated(element=element)
    with slp.record() as rec:
        data.ts.plot()
    ylabel = next(mark.kwargs["ylabel"] for mark in rec.of_kind("set") if mark.kwargs.get("ylabel") is not None)
    assert "a_t" not in ylabel and "/M_{N}" in ylabel


def test_labels_for_multiple_rotation_modes_do_not_claim_the_first_mode():
    data = effective_matrix()
    with slp.record() as rec:
        data.ts.plot()
    ylabel = next(mark.kwargs["ylabel"] for mark in rec.of_kind("set") if mark.kwargs.get("ylabel") is not None)
    assert ylabel == r"$a_t E_{\mathrm{eff}}$"
    with slp.record() as rec:
        data.isel(operator=2).ts.plot()
    ylabel = next(mark.kwargs["ylabel"] for mark in rec.of_kind("set") if mark.kwargs.get("ylabel") is not None)
    assert "ROT" in ylabel and "2" in ylabel


def test_raw_correlator_time_view_is_recovered_without_stored_axis_state(tmp_path):
    matrix, _ = spectrum()
    corr = matrix.correlator(matrix.operators[0], matrix.operators[1])
    path = tmp_path / "raw_correlator.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), list(corr.to_collection()), group="g")
    restored = SigmondData.from_collection(SigmondLoader(str(path)).observables, shape_by=["time"])
    assert restored.ts.elements.item() == corr.ts.elements.item()
    assert restored.ts.axis.offset == 0 and restored.ts.axis.extent == 16
    assert restored.ts.at(3).observable_info.time == 3


def test_time_series_types_are_public_and_lazily_exported():
    import sigmondsamplings as ss
    assert isinstance(numeric().ts, ss.TimeSeriesView)
    assert isinstance(numeric().ts.axis, ss.TimeAxis)


@pytest.mark.parametrize("kind", [ObservableInfo, EnergyObsInfo, SHEnergyObsInfo])
def test_direct_general_and_energy_codecs_keep_overrides_outside_identity(kind):
    kwargs = {"particle": "pi"} if kind is SHEnergyObsInfo else {}
    info = kind("custom", ensemble_info=ENS, latex_str=r"A_{0}", **kwargs)
    restored = kind.from_attrs(ObservableInfo(info.name, ensemble_info=ENS), info.to_attrs())
    assert restored.latex_str == info.latex_str and restored == info
    original_hash = hash(info)
    info.latex_str = "changed"
    assert hash(info) == original_hash
