"""Small interaction regressions from the operators/correlators API review."""

import pickle

import numpy as np
import pytest
from correlator_data import ENS, OPS, ckey, observable, vev

import slat.plotting as slp
from sigmondsamplings import SigmondLoader, SigmondWriter
from sigmondsamplings.cli.corr import matrix_records
from sigmondsamplings.correlators import (
    BasicLapHOperator,
    CorrelatorAtTimeInfo,
    CorrelatorIndex,
    CorrelatorInfo,
    CorrelatorMatrix,
    CorrelatorObsInfo,
    GenIrrepOperator,
    effective_energy,
    operator_from_xml,
    parse_operator,
)
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.plotting import CorrelatorPlotter, SeriesPlotter
from sigmondsamplings.rcparams import rc_context

PLAN = SamplingInfo("jackknife", 3)


def data(buf, times):
    return SigmondData.from_data(
        np.asarray(buf), "C", PLAN, dim_names=("time",), coords={"time": times}, ensemble_info=ENS
    )


def matrix():
    items = [
        observable(ckey(i, j, 1), np.array([i + j + 2, 1, 2, 3]) + 1j * (i - j), PLAN)
        for i in range(2)
        for j in range(2)
    ]
    return CorrelatorIndex.from_collection(items).matrix(OPS[:2], hermitian=False)


@pytest.mark.parametrize("flavor", ["tquuuu1m", "tquudu3m", "tquuss1p", "tqsuss2p", "tqssss1p"])
@pytest.mark.parametrize("displacement", [2, 3])
def test_tetraquark_identity_roundtrips(flavor, displacement):
    op = BasicLapHOperator.from_sigmond(f"{flavor} P=(0,0,0) A1um_1 QDX_1 D{displacement}")
    for restored in (
        parse_operator(str(op)),
        parse_operator(op.coordinate),
        operator_from_xml(op.to_xml()),
        pickle.loads(pickle.dumps(op)),
    ):
        assert restored == op and hash(restored) == hash(op)
        assert restored.hadrons[0].displacement == displacement
    if displacement == 3:
        assert "D3" not in op.to_sigmond()  # preserve Sigmond's stored-key spelling


@pytest.mark.parametrize("positions", [[1, 0], [1]])
def test_transform_select_transform_keeps_original_basis(positions):
    original = matrix()
    t = np.array([[1, 2j], [3j, 4]])
    rotated = original.transform(t)
    rotated = rotated.select([rotated.operators[i] for i in positions])
    u = np.ones((len(positions), 1)) * (1 + 1j)
    result = rotated.transform(u, name="FINAL")
    composed = t[:, positions] @ u
    expected = np.einsum("ia,tijs,jb->tabs", composed.conj(), original.values.data, composed)
    np.testing.assert_allclose(result.values.data, expected)
    np.testing.assert_allclose(result.transformation.coefficients, composed)
    assert result.transformation.source_operators == original.operators


@pytest.mark.parametrize("index", [(0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 0)])
def test_invalid_diagnostics_do_not_pass(index):
    m = matrix()
    m.values.data[index] = np.nan
    assert not m.check_hermitian().ok
    assert np.isnan(m.significance(1).values[index[1:3]])


def test_nonfinite_effective_inputs_follow_failure_policy():
    corr = data([[8, 8, 8, 8], [4, np.nan, 4, 4], [2, 2, 2, 2], [1, 1, 1, 1]], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="fails"):
        effective_energy(corr, on_fail="raise")
    eff = effective_energy(corr, on_fail="nan")
    assert list(eff.coords["time"]) == [1, 2, 3]
    assert np.isnan(eff.data[:2, 1]).all()
    with pytest.warns(UserWarning):
        assert list(effective_energy(corr).coords["time"]) == [3]


@pytest.mark.parametrize("times", [[], [1], [1, 3, 5]])
def test_no_neighbours_has_domain_error(times):
    with pytest.raises(ValueError, match="no time"):
        effective_energy(data(np.ones((len(times), 4)), times))


@pytest.mark.parametrize(
    "argument", [{"step": 1.9}, {"step": True}, {"time_extent": 9.5}, {"step": np.bool_(True)}]
)
def test_fractional_or_boolean_effective_arguments_rejected(argument):
    with pytest.raises((TypeError, ValueError)):
        effective_energy(data([[2] * 4, [1] * 4], [1, 2]), **argument)


def test_sparse_forward_energy_needs_no_extent():
    corr = data([[4] * 4, [2] * 4, [1] * 4], [1, 1000000000, 1000000001])
    corr.observable_info.ensemble_info = EnsembleInfo("no_extent", 3)
    np.testing.assert_allclose(effective_energy(corr).data, np.log(2))


def test_constructor_rejects_misaligned_attachments():
    m = matrix()
    cov = CorrelatorIndex.from_collection(m.to_collection()).coverage(m.info)
    with pytest.raises(ValueError, match="aligned"):
        CorrelatorMatrix(m.info, m.values, stored=cov.assign_coords(time=[2]))
    v = SigmondData.from_data(
        np.ones((2, 3)),
        "v",
        PLAN,
        dim_names=("operator",),
        coords={"operator": m.info.coordinates},
        ensemble_info=ENS,
    )
    with pytest.raises(ValueError, match="sample length"):
        CorrelatorMatrix(m.info, m.values, v)
    v = SigmondData.from_data(
        np.ones((2, 4)),
        "v",
        PLAN,
        dim_names=("operator",),
        coords={"operator": m.info.coordinates},
        ensemble_info=EnsembleInfo("other", 3),
    )
    with pytest.raises(ValueError, match="ensemble"):
        CorrelatorMatrix(m.info, m.values, v)


def test_metadata_snapshot_refresh(tmp_path):
    path = tmp_path / "corr.h5"
    SigmondWriter(create_backups=False).write_hdf5(
        str(path), [observable(ckey(0, 0, 1), np.ones(4), PLAN)]
    )
    loader = SigmondLoader(str(path), lazy=True)
    old = loader.correlators
    key = ckey(0, 0, 11)
    loader.observables.obs.update(name=key.to_sigmond())
    assert old.times == (1,) and old.get(ckey(0, 0, 1)).observable_info.time == 1
    assert loader.refresh_correlators().times == (11,)


def test_coverage_rejects_im_only_and_splits_directions():
    bad = observable(ckey(0, 0, 1), np.ones(4), PLAN)
    bad.observable_info.re_im = "im"
    index = CorrelatorIndex.from_collection([bad])
    assert not index.coverage([OPS[0]]).values.any()
    assert index.available_bases([OPS[0]]) == []
    items = [observable(ckey(0, 0, 1), np.ones(4), PLAN), vev(OPS[0], np.ones(4), PLAN)]
    items[1].observable_info.re_im = "im"
    assert (
        not CorrelatorIndex.from_collection(items)
        .coverage([OPS[0]], subtract_vev=True)
        .values.any()
    )
    backwards = OPS[0].with_backwards()
    items = [
        observable(CorrelatorAtTimeInfo(CorrelatorInfo(op, op), 1), np.ones(4), PLAN)
        for op in (OPS[0], backwards)
    ]
    index = CorrelatorIndex.from_collection(items)
    channel = OPS[0].channel
    assert index.available_bases(channel) == [(OPS[0],), (backwards,)]
    with pytest.raises(ValueError, match="mix"):
        index.matrix(channel)


def test_projection_updates_cell_and_array_metadata():
    corr = matrix().correlator(*OPS[:2])
    projected = CorrelatorPlotter(corr, component="imag").corr
    assert not projected.is_complex and projected.observable_info.re_im == "im"
    assert all(info.re_im == "im" for info in projected.infos.flat)
    assert projected.to_real().observable_info.re_im == "re"


def test_default_cli_time_is_complete_in_selected_channel():
    other = BasicLapHOperator.from_sigmond("kaon P=(0,0,0) A1um_1 SS_0")
    items = [
        observable(CorrelatorAtTimeInfo(CorrelatorInfo(op, op), t), np.ones(4), PLAN)
        for op, t in [(OPS[0], 1), (other, 4)]
    ]
    index = CorrelatorIndex.from_collection(items)
    assert matrix_records(index, "1")[0] == 4
    assert index.matrix(other.channel).times == (4,)
    assert list(index.correlator(other).coords["time"]) == [4]


def test_conflicting_metadata_and_operator_kinds_rejected():
    with pytest.raises(ValueError):
        CorrelatorObsInfo(ckey(0, 0, 1).to_sigmond(), key=ckey(0, 0, 10))
    with pytest.raises(TypeError):
        parse_operator(OPS[0], "general")
    with pytest.raises(ValueError):
        parse_operator(OPS[0], "bad")
    assert GenIrrepOperator.from_channel(OPS[0].channel, name="ROT").channel == OPS[0].channel


def test_log_absolute_reflects_asymmetric_interval_but_not_positive_hollow():
    boot = SamplingInfo("bootstrap", 4)
    corr = SigmondData.from_data(
        np.array([[-10, -14, -13, -9, -8], [10, 6, 7, 11, 12]]),
        "C",
        boot,
        dim_names=("time",),
        coords={"time": [1, 2]},
        ensemble_info=ENS,
    )
    with (
        rc_context({"plot.error_metric": "confidence_interval", "plot.confidence_level": 0.68}),
        slp.record() as rec,
    ):
        SeriesPlotter(corr, hollow=[[False, True]]).plot(yscale="log")
    mark = rec.of_kind("errorbar")[0]
    np.testing.assert_allclose(mark.kwargs["yerr"], [[1.52, 3.52], [3.52, 1.52]])
    np.testing.assert_allclose(mark.args[1], [10, 10])
    limits = next(m.kwargs["ylim"] for m in rec.of_kind("set") if "ylim" in m.kwargs)
    assert limits[1] > 13.52


def test_connected_bins_use_independent_leave_one_out_means():
    ens = EnsembleInfo("tiny", 3)
    a, b, c = (
        np.array([1 + 1j, 2 + 2j, 4 - 1j]),
        np.array([2 - 1j, 3 + 1j, 1 + 2j]),
        np.array([10 + 2j, 20 - 1j, 30 + 3j]),
    )
    items = [
        observable(ckey(i, j, 1), c if i <= j else c.conj(), None, ens)
        for i in range(2)
        for j in range(2)
    ]
    items += [vev(OPS[0], a, None, ens), vev(OPS[1], b, None, ens)]
    m = CorrelatorIndex.from_collection(items).matrix(
        OPS[:2], subtract_vev=True, sampling_info=PLAN
    )

    def means(x):
        return np.array([x.mean()] + [np.delete(x, k).mean() for k in range(3)])

    expected = means(c) - means(a) * means(b).conj()
    np.testing.assert_allclose(m.values.data[0, 0, 1], expected)
    np.testing.assert_allclose(m.values.data[0, 1, 0], expected.conj())


def test_significance_known_jackknife_error():
    # Resamples 1, 2, 3 have jackknife variance (2/3) * (1 + 0 + 1).
    assert matrix().significance(1).values[0, 0] == pytest.approx(np.sqrt(3))


def test_nonfinite_effective_sample_does_not_remove_other_series():
    buf = np.broadcast_to(np.array([4, 2, 1])[:, None, None], (3, 2, 4)).astype(float).copy()
    buf[1, 0, 1] = np.nan
    corr = SigmondData.from_data(
        buf,
        "C",
        PLAN,
        dim_names=("time", "operator"),
        coords={"time": [1, 2, 3], "operator": ["a", "b"]},
        ensemble_info=ENS,
    )
    eff = effective_energy(corr, on_fail="nan")
    np.testing.assert_allclose(eff.data[:, 1], np.log(2))
    assert np.isnan(eff.data[:, 0, 1]).all()
