"""
Matrix assembly (key forms, Hermitian completion, selection) and VEV
subtraction / basis changes, checked sample by sample against hand-computed
arrays.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from correlator_data import BOOT, ENS, JACK, N_BINS, OPS, ckey, observable, series, vev

from sigmondsamplings.bins import _block_resample
from sigmondsamplings.correlators import (
    CorrelatorIndex,
    CorrelatorKey,
    CorrelatorMatrix,
    CorrelatorMatrixSpec,
    CorrelatorPair,
    GeneralOperator,
)
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import SamplingInfo
from sigmondsamplings.stats import SamplingStats

RNG = np.random.default_rng(17)
SPEC2 = CorrelatorMatrixSpec(OPS[:2])


def _index(items):
    return CorrelatorIndex.from_collection(items)


def _full(times, n=2, herm=False, subvev=False, sampling=JACK, cells=None):
    """Store every ordered (i, j) at ``times``; return items and the raw arrays."""
    items, raw = [], {}
    for t in times:
        for i in range(n):
            for j in range(n):
                if cells is not None and (i, j) not in cells:
                    continue
                data = series(RNG, sampling)
                raw[(t, i, j)] = data
                items.append(observable(ckey(i, j, t, herm, subvev), data, sampling))
    return items, raw


# ---------------------------------------------------------------------------
# Hermitian completion and provenance
# ---------------------------------------------------------------------------


def test_hand_computed_hermitian_2x2():
    items, raw = _full([4])
    m = _index(items).matrix(SPEC2)
    c = m.values.data[0]
    assert np.allclose(c[0, 0], raw[(4, 0, 0)])
    assert np.allclose(c[0, 1], 0.5 * (raw[(4, 0, 1)] + np.conj(raw[(4, 1, 0)])))
    assert np.allclose(c[1, 0], 0.5 * (raw[(4, 1, 0)] + np.conj(raw[(4, 0, 1)])))
    assert np.allclose(c[1, 0], np.conj(c[0, 1]))
    assert m.stored.values[0].tolist() == [[True, True], [True, True]]  # both directions: averaged
    assert m.key_form == "plain"


def test_one_direction_is_conjugated_into_the_other_cell():
    items, raw = _full([4, 5], cells={(0, 0), (1, 1), (1, 0)})
    m = _index(items).matrix(SPEC2)
    assert np.allclose(m.values.data[1, 0, 1], np.conj(raw[(5, 1, 0)]))
    assert not m.stored.values[1, 0, 1] and m.stored.values[1, 1, 0]  # conjugated from (1, 0)


def test_diagonal_keeps_complex_value_and_check_hermitian_reports_it():
    items, _ = _full([4], cells={(0, 0), (1, 1), (0, 1)})
    m = _index(items).matrix(SPEC2)
    assert m.values.is_complex
    report = m.check_hermitian(tol=1e-6)
    assert np.all(report.residual.values == 0)
    assert report.diagonal_imag.values.max() > 0
    real = m.diagonal(component="real")
    assert not real.is_complex and real.dims == ("time", "operator")
    assert all(info.re_im == "re" for info in real.infos.flat)
    assert m.diagonal().is_complex


def test_non_hermitian_reads_both_directions_independently():
    items, raw = _full([4])
    spec = CorrelatorMatrixSpec(OPS[:2], hermitian=False)
    m = _index(items).matrix(spec)
    assert np.allclose(m.values.data[0, 1, 0], raw[(4, 1, 0)])
    assert np.allclose(m.values.data[0, 0, 1], raw[(4, 0, 1)])
    items_missing = [o for o in items if o.observable_info.name != ckey(1, 0, 4).to_sigmond()]
    with pytest.raises(ValueError, match="missing") as info:
        _index(items_missing).matrix(spec, times=[4])
    assert OPS[1].coordinate in str(info.value)


def test_operators_may_be_strings():
    items, raw = _full([4])
    index = _index(items)
    m = index.matrix([str(OPS[0]), OPS[1].coordinate])
    assert m.operators == OPS[:2]
    assert np.allclose(m.sel(sink=str(OPS[0]), source=OPS[1], time=4).data, m.values.data[0, 0, 1])
    c = index.correlator(str(OPS[1]), str(OPS[0]))
    assert np.allclose(c.sel(time=4).data, raw[(4, 1, 0)])
    with pytest.raises(TypeError, match="sequence of operators"):
        index.matrix(str(OPS[0]))


def test_missing_cells_are_never_zero_filled():
    items, _ = _full([3, 4, 5])
    items = [
        o
        for o in items
        if o.observable_info.name not in (ckey(0, 1, 5).to_sigmond(), ckey(1, 0, 5).to_sigmond())
    ]
    index = _index(items)
    with pytest.raises(ValueError, match="time 5: missing"):
        index.matrix(SPEC2, times=[4, 5])
    with pytest.warns(UserWarning, match="times \\[5\\]"):
        m = index.matrix(SPEC2)
    assert m.times == (3, 4) and m.omitted_times == (5,)


def test_key_forms_are_never_mixed_or_guessed():
    herm, _ = _full([4], herm=True)
    plain, _ = _full([4, 5], herm=False)
    index = _index(herm + plain)
    with pytest.raises(ValueError, match="several key forms.*'HermMat'.*'plain'"):
        index.matrix(SPEC2)
    m = index.matrix(SPEC2, key_form="HermMat")
    assert m.key_form == "HermMat" and m.times == (4,)
    assert all(info.key.hermitian for info in m.values.infos.flat)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert index.matrix(SPEC2, key_form="plain").times == (4, 5)
    with pytest.raises(ValueError, match="not allowed"):
        index.matrix(CorrelatorMatrixSpec(OPS[:2], hermitian=False), key_form="HermMat")
    with pytest.raises(ValueError, match="key_form must be"):
        index.matrix(SPEC2, key_form="Herm")
    # Split forms: HermMat on the diagonal, unflagged off-diagonal -> no complete form.
    split = [o for o in herm if o.observable_info.sink == o.observable_info.source]
    split += [
        o
        for o in plain
        if o.observable_info.time == 4 and o.observable_info.sink != o.observable_info.source
    ]
    with pytest.raises(ValueError, match="missing"):
        _index(split).matrix(SPEC2, times=[4], key_form="HermMat")


def test_wrong_flags_say_which_forms_are_stored():
    herm, _ = _full([4], herm=True)
    with pytest.raises(ValueError, match="stored forms are 'HermMat'"):
        _index(herm).correlator(OPS[0], OPS[1])
    # Only the flipped direction is stored: still worth naming.
    flipped, _ = _full([4], herm=True, cells={(1, 0)})
    with pytest.raises(ValueError, match="stored forms are 'HermMat'"):
        _index(flipped).correlator(OPS[0], OPS[1])


def test_lazy_block_reads_match_eager_data(tmp_path):
    from sigmondsamplings import SigmondLoader, SigmondWriter

    items, _ = _full([3, 4, 5], n=3)
    items += [observable(ckey(0, 0, 6), series(RNG, complex_=False))]  # a real record too
    path = tmp_path / "c.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), items, group="c")
    spec = CorrelatorMatrixSpec(OPS)
    eager = _index(items).matrix(spec, times=[3, 4, 5])
    lazy = SigmondLoader(str(path), lazy=True).correlators.matrix(spec, times=[3, 4, 5])
    assert np.array_equal(lazy.values.data, eager.values.data)
    real = SigmondLoader(str(path), lazy=True).correlators.correlator(OPS[0], OPS[0], times=[6])
    assert not real.is_complex


def test_select_sel_and_stored_alignment():
    spec = CorrelatorMatrixSpec(OPS)
    items, raw = _full([3, 4], n=3, cells=None)
    m = _index(items).matrix(spec)
    sub = m.select([OPS[2], OPS[0]])
    assert sub.operators == (OPS[2], OPS[0])
    assert list(sub.values.coords["sink"]) == [OPS[2].coordinate, OPS[0].coordinate]
    assert np.allclose(sub.values.data[:, 0, 1], m.values.data[:, 2, 0])
    assert list(sub.stored.coords["sink"].values) == list(sub.values.coords["sink"])
    assert sub.stored.values.all()
    cell = m.sel(sink=OPS[1], source=OPS[2], time=3)
    assert np.allclose(cell.data, m.values.data[0, 1, 2])
    assert m.correlator(OPS[0], OPS[1]).dims == ("time",)
    assert m.at(4).dims == ("sink", "source")
    later = m.sel_times([4])
    assert later.times == (4,) and later.stored.sizes["time"] == 1
    with pytest.raises(KeyError):
        m.sel(sink=CorrelatorPair(OPS[0], OPS[1]).sink.with_backwards())


def test_matrix_values_use_string_coords_and_real_keys():
    items, _ = _full([4])
    m = _index(items).matrix(SPEC2)
    assert m.values.dims == ("time", "sink", "source")
    assert list(m.values.coords["sink"]) == list(SPEC2.coordinates)
    key = CorrelatorKey.from_sigmond(m.values.infos[0, 1, 0].name)
    assert key == ckey(1, 0, 4, herm=True)
    x = m.values.as_xarray()
    assert x.sel(sink=OPS[0].coordinate).dims == ("time", "source", "sample")


# ---------------------------------------------------------------------------
# VEVs
# ---------------------------------------------------------------------------


def _with_vevs(sampling, times=(4, 5), n=2, stored_subvev=False):
    items, raw = _full(times, n=n, sampling=sampling, subvev=stored_subvev)
    v = {i: series(RNG, sampling, loc=2.0) for i in range(n)}
    items += [vev(OPS[i], v[i], sampling) for i in range(n)]
    return items, raw, v


@pytest.mark.parametrize("sampling", [JACK, BOOT])
def test_connected_from_samplings(sampling):
    items, raw, v = _with_vevs(sampling)
    spec = CorrelatorMatrixSpec(OPS[:2], subtract_vev=True)
    m = _index(items).matrix(spec)
    assert m.key_form == "plain"  # derived from raw keys and VEVs
    avg = 0.5 * (raw[(5, 0, 1)] + np.conj(raw[(5, 1, 0)]))
    assert np.allclose(m.values.data[1, 0, 1], avg - v[0] * np.conj(v[1]))
    assert np.allclose(m.values.data[0, 1, 1], raw[(4, 1, 1)] - v[1] * np.conj(v[1]))
    assert np.allclose(m.vevs.data, [v[0], v[1]])
    assert CorrelatorKey.from_sigmond(m.values.infos[0, 0, 1].name).subtract_vev


def test_stored_subvev_is_used_without_double_subtraction():
    items, raw, v = _with_vevs(JACK, stored_subvev=True)
    spec = CorrelatorMatrixSpec(OPS[:2], subtract_vev=True)
    m = _index(items).matrix(spec)
    assert m.key_form == "SubVEV" and m.vevs is None
    assert np.allclose(m.values.data[0, 0, 0], raw[(4, 0, 0)])
    m2 = _index(items).matrix(spec, include_vevs=True)
    assert np.allclose(m2.vevs.data[1], v[1])


def test_stored_subvev_or_derived_is_an_explicit_choice():
    raw_items, raw, v = _with_vevs(JACK)
    sub_items, sub_raw = _full([4, 5], subvev=True)
    spec = CorrelatorMatrixSpec(OPS[:2], subtract_vev=True)
    index = _index(raw_items + sub_items)
    with pytest.raises(ValueError, match="several key forms"):
        index.matrix(spec)
    stored = index.matrix(spec, key_form="SubVEV")
    assert np.allclose(stored.values.data[0, 0, 0], sub_raw[(4, 0, 0)])
    derived = index.matrix(spec, key_form="plain")
    assert np.allclose(derived.values.data[0, 0, 0], raw[(4, 0, 0)] - v[0] * np.conj(v[0]))


def test_missing_vevs_are_all_reported_before_reading():
    items, _ = _full([4], n=3)
    items.append(vev(OPS[1], series(RNG)))
    spec = CorrelatorMatrixSpec(OPS, subtract_vev=True)
    with pytest.raises(ValueError) as info:
        _index(items).matrix(spec, times=[4])
    missing = str(info.value).split("missing for:")[-1]
    assert OPS[0].coordinate in missing and OPS[2].coordinate in missing
    assert OPS[1].coordinate not in missing
    with pytest.raises(ValueError, match="include_vevs"):
        _index(items).matrix(CorrelatorMatrixSpec(OPS), include_vevs=True)


def _correlated_bins():
    """Correlator and VEV bins sharing fluctuations, so <C - v v*> != <C> - <v><v*>."""
    base = RNG.normal(size=(2, N_BINS))
    v = {i: 1.0 + 0.8 * base[i] for i in range(2)}
    items = []
    raw = {}
    for t in (4, 5):
        for i in range(2):
            for j in range(2):
                data = v[i] * v[j] + 0.1 * RNG.normal(size=N_BINS)
                raw[(t, i, j)] = data
                items.append(observable(ckey(i, j, t), data, None))
    items += [vev(OPS[i], v[i], None) for i in range(2)]
    return items, raw, v


@pytest.mark.parametrize("plan", [BOOT, SamplingInfo("jackknife", N_BINS)])
def test_connected_from_bins_resamples_jointly_then_subtracts(plan):
    items, raw, v = _correlated_bins()
    spec = CorrelatorMatrixSpec(OPS[:2], subtract_vev=True)
    index = _index(items)
    with pytest.raises(ValueError, match="sampling_info"):
        index.matrix(spec)
    m = index.matrix(spec, sampling_info=plan)
    assert m.sampling_info == plan and not m.is_bins
    c01 = 0.5 * (raw[(4, 0, 1)] + raw[(4, 1, 0)])
    stacked = np.stack([c01, v[0], v[1]])
    res = _block_resample(stacked, ENS, plan, ["mean"])[0]
    expected = res[0] - res[1] * res[2]
    assert np.allclose(m.values.data[0, 0, 1], expected)
    binwise = c01 - v[0] * v[1]
    binwise_full = binwise.mean()
    assert not np.isclose(m.values.full_sample_value.values[0, 0, 1], binwise_full)
    assert np.isclose(m.values.data[0, 0, 1, 0], c01.mean() - v[0].mean() * v[1].mean())
    assert np.allclose(m.vevs.data, res[1:])


def test_raw_bins_stay_bins_or_resample_on_request():
    items, raw, v = _correlated_bins()
    index = _index(items)
    m = index.matrix(SPEC2, include_vevs=True)
    assert m.is_bins and m.vevs.sampling_info is None
    r = index.matrix(SPEC2, include_vevs=True, sampling_info=BOOT)
    assert r.sampling_info == BOOT and r.vevs.sampling_info == BOOT


def test_correlator_request_derives_like_a_one_pair_matrix():
    items, raw, v = _with_vevs(JACK)
    index = _index(items)
    c = index.correlator(CorrelatorPair(OPS[1], OPS[0]), hermitian=True, subtract_vev=True)
    avg = 0.5 * (raw[(4, 1, 0)] + np.conj(raw[(4, 0, 1)]))
    assert np.allclose(c.sel(time=4).data, avg - v[1] * np.conj(v[0]))
    assert c.dims == ("time",)
    key = CorrelatorKey.from_sigmond(c.infos[0].name)
    assert key.sink == OPS[1] and key.subtract_vev and key.hermitian
    plain = index.correlator(CorrelatorPair(OPS[1], OPS[0]))
    assert np.allclose(plain.sel(time=5).data, raw[(5, 1, 0)])


# ---------------------------------------------------------------------------
# Basis changes
# ---------------------------------------------------------------------------

ROT = [GeneralOperator("triplet", 0, "A1um", "ROT", k, momentum=(0, 0, 0), row=1) for k in range(3)]


def test_rectangular_complex_transform():
    items, raw, v = _with_vevs(JACK, n=3)
    m = _index(items).matrix(CorrelatorMatrixSpec(OPS), include_vevs=True)
    T = np.array([[1.0, 0.3j], [0.5 - 0.2j, 1.0], [0.1, -0.4]])
    new = m.transform(T, ROT[:2])
    expected = np.einsum("ia,tijs,jb->tabs", T.conj(), m.values.data, T)
    assert np.allclose(new.values.data, expected)
    assert np.allclose(new.vevs.data, np.einsum("ia,is->as", T.conj(), m.vevs.data))
    assert new.operators == tuple(ROT[:2])
    assert np.array_equal(new.transformation.coefficients, T)
    assert new.transformation.source_operators == OPS
    assert new.key_form is None and new.stored is None
    name = new.values.infos[0, 0, 1].name
    assert CorrelatorKey.from_sigmond(name).source == ROT[1]
    again = new.transform(np.array([[1.0], [2.0]]), ROT[2:])
    assert np.allclose(again.transformation.coefficients, T @ np.array([[1.0], [2.0]]))


def test_transform_validation():
    items, _ = _full([4])
    m = _index(items).matrix(SPEC2)
    with pytest.raises(ValueError, match="shape"):
        m.transform(np.eye(3), ROT)
    psq = GeneralOperator("triplet", 0, "A1um", "ROT", 0, momentum_squared=0)
    assert psq.has_definite_momentum  # PSQ=0 canonicalizes to P=(0,0,0)
    moving = [
        GeneralOperator("triplet", 0, "A1um", "ROT", k, momentum_squared=1, row=1) for k in range(2)
    ]
    with pytest.raises(ValueError, match="PSQ-only"):
        m.transform(np.eye(2), moving)
    wrong = [GeneralOperator("triplet", 0, "A1um", "ROT", k, momentum=(0, 0, 0)) for k in range(2)]
    with pytest.raises(ValueError, match="channel"):
        m.transform(np.eye(2), wrong)
    with pytest.raises(ValueError, match="duplicate"):
        m.transform(np.eye(2), [ROT[0], ROT[0]])
    with pytest.raises(ValueError, match="finite"):
        m.transform(np.array([[np.inf, 0], [0, 1]]), ROT[:2])


def test_to_collection_rebuilds_keys_after_arithmetic():
    items, _ = _full([4, 5])
    m = _index(items).matrix(SPEC2)
    # Same labels but a different array info: the arithmetic result gets placeholders.
    other = SigmondData.from_data(
        m.values.data,
        "other",
        m.sampling_info,
        dim_names=m.values.dims,
        coords=dict(m.values.coords),
        ensemble_info=ENS,
        is_complex=True,
    )
    doubled = CorrelatorMatrix(m.spec, m.values * 2.0 - other, stored=m.stored)
    assert "mixed_operation" in doubled.values.infos[0, 0, 0].name
    names = [o.observable_info.name for o in doubled.to_collection()]
    assert names == [info.name for info in m.values.infos.flat]
    rebuilt = CorrelatorIndex.from_collection(doubled.to_collection()).matrix(m.spec)
    assert np.allclose(rebuilt.values.data, m.values.data)


def test_constructor_checks_alignment():
    items, _ = _full([4])
    m = _index(items).matrix(SPEC2)
    with pytest.raises(ValueError, match="basis coordinates"):
        CorrelatorMatrix(CorrelatorMatrixSpec(OPS[1::-1]), m.values)
    with pytest.raises(ValueError, match="dims"):
        CorrelatorMatrix(SPEC2, m.values.transpose("sink", "time", "source"))


def test_bins_matrix_write_read_round_trip(tmp_path):
    from sigmondsamplings import SigmondLoader

    items, raw, v = _correlated_bins()
    m = _index(items).matrix(SPEC2, include_vevs=True)
    path = tmp_path / "bins.hdf5"
    m.write(path, group="bins")
    back = SigmondLoader(str(path)).correlators.matrix(SPEC2, include_vevs=True)
    assert back.is_bins and back.times == m.times
    assert np.allclose(back.values.data, m.values.data)
    assert np.allclose(back.vevs.data, m.vevs.data)


# ---------------------------------------------------------------------------
# Fitting handoff
# ---------------------------------------------------------------------------


def test_fitting_handoff_keeps_time_order():
    items, _ = _full([6, 3, 5, 4])
    m = _index(items).matrix(SPEC2)
    series_ = m.diagonal(component="real").sel(operator=OPS[0].coordinate)
    t = np.asarray(series_.coords["time"], dtype=float)
    stats = SamplingStats(series_.to_collection())
    assert list(t) == [3.0, 4.0, 5.0, 6.0]
    keys = [CorrelatorKey.from_sigmond(o.observable_info.name) for o in stats]
    assert [k.time for k in keys] == [3, 4, 5, 6]
    assert np.allclose(stats.to_numpy(), series_.data)  # rows follow the time coords


def test_fitting_from_data_takes_time_from_the_diagonal_series():
    pytest.importorskip("optimagic")
    from sigmondsamplings.fitting import CallableXModel, ParamSetSpec, ParamSpec, SamplingFit

    amp, energy, factor = 2.0, 0.4, (JACK.num_resamplings - 1) / np.sqrt(JACK.num_resamplings)
    items = []
    for t in [6, 3, 5, 4]:  # stored out of order
        for i in range(2):
            for j in range(2):
                mean = amp * np.exp(-energy * t)
                data = mean + (0.01 * mean / factor) * RNG.normal(size=JACK.num_resamplings + 1)
                data[0] = mean
                items.append(
                    observable(ckey(i, j, t), data + 1j * 1e-3 * RNG.normal(size=data.size))
                )
    m = _index(items).matrix(SPEC2)
    model = CallableXModel(lambda t, th: th[0] * np.exp(-th[1] * t), ("A", "E"))
    params = ParamSetSpec(values={"A": ParamSpec(initial=1.0), "E": ParamSpec(initial=0.1)})

    complex_series = m.diagonal().sel(operator=OPS[0].coordinate)
    with pytest.raises(ValueError, match="component='real'"):
        SamplingFit.from_data(complex_series, model, params.resolve())

    series_ = m.diagonal(component="real").sel(operator=OPS[0].coordinate)
    fit = SamplingFit.from_data(series_, model, params.resolve())
    assert fit.x_name == "time" and list(fit.x) == [3.0, 4.0, 5.0, 6.0]
    assert np.allclose(fit.objective.stats.to_numpy(), series_.data)
    best = fit.fit_one(0).params
    assert best["A"] == pytest.approx(amp, rel=1e-6)
    assert best["E"] == pytest.approx(energy, rel=1e-6)
