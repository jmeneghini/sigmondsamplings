"""
The metadata-only correlator index, the correlator/VEV observable infos the
loader attaches, and the lazy HDF5 bridge.
"""

from __future__ import annotations

import numpy as np
import pytest
from correlator_data import BOOT, ENS, JACK, OPS, ckey, observable, series, vev

from sigmondsamplings import SigmondLoader, SigmondWriter
from sigmondsamplings.correlators import (
    BasicLapHOperator,
    CorrelatorAtTimeInfo,
    CorrelatorIndex,
    CorrelatorInfo,
    CorrelatorMatrixInfo,
    CorrelatorObsInfo,
    VEVInfo,
    VEVObsInfo,
)
from sigmondsamplings.info import EnsembleInfo, ObservableInfo
from sigmondsamplings.lazy import LazyDataMixin, materialize, stack_data
from sigmondsamplings.sampling import SigmondSampling

RNG = np.random.default_rng(5)


@pytest.fixture(autouse=True)
def _reset_rng():
    global RNG
    RNG = np.random.default_rng(5)


def _store(times=(3, 4, 5), n=2, herm=True):
    items = [
        observable(ckey(i, j, t, herm), series(RNG))
        for t in times
        for i in range(n)
        for j in range(n)
    ]
    return items


def test_index_metadata_and_lookup():
    items = _store()
    items.append(vev(OPS[0], series(RNG, complex_=False)))
    plain = SigmondSampling(series(RNG), ObservableInfo("E0", 0, ensemble_info=ENS), JACK, True)
    items.append(plain)
    index = CorrelatorIndex.from_collection(items)
    assert index.operators == OPS[:2]
    assert index.times == (3, 4, 5)
    assert CorrelatorInfo(OPS[1], OPS[0]) in index.correlator_infos
    assert index.vev_operators == (OPS[0],)
    key = ckey(1, 0, 4, herm=True)
    assert index.get(key).observable_info == items[6].observable_info
    np.testing.assert_array_equal(index.get(key).data, items[6].data)
    assert key in index and VEVInfo(OPS[0]) in index and len(index) == 13
    assert index.series_times(key.correlator, hermitian=True) == (3, 4, 5)
    assert index.ensemble_info == ENS and index.sampling_info == JACK and not index.is_bins
    assert "4 correlators" in repr(index)
    with pytest.raises(KeyError):
        index.get(ckey(0, 1, 9, herm=True))


def test_strict_rejects_unrelated_observables():
    items = _store(times=(3,))
    items.append(SigmondSampling(series(RNG), ObservableInfo("E0", ensemble_info=ENS), JACK, True))
    CorrelatorIndex.from_collection(items)  # unrelated observables are fine by default
    with pytest.raises(ValueError, match="unrelated"):
        CorrelatorIndex.from_collection(items, strict=True)


BAD_KEY = "<CorrT>BL{pion P=(0,0,0) A1um_1 SS_0} time=3</CorrT>"


def test_unparsed_keys_are_reported_not_fatal():
    items = _store(times=(3,))
    bad = SigmondSampling(series(RNG), ObservableInfo(BAD_KEY, ensemble_info=ENS), JACK, True)
    items.append(bad)
    with pytest.warns(UserWarning, match="1 correlator/VEV key"):
        index = CorrelatorIndex.from_collection(items)
    assert [name for name, _ in index.unparsed] == [BAD_KEY]
    assert "expected an operator" in index.unparsed[0][1]
    assert index.matrix(OPS[:2]).times == (3,)  # the rest of the index is usable
    with pytest.raises(ValueError, match="1 unparsed"):
        CorrelatorIndex.from_collection(items, strict=True)


def test_correlator_infos_parse_lazily_and_tolerate_bad_names():
    info = CorrelatorObsInfo(ckey(0, 1, 3, herm=True).to_sigmond(), ensemble_info=ENS)
    assert info._key is None  # nothing parsed yet
    assert info.time == 3 and info.sink == OPS[0] and info.hermitian and not info.subtract_vev
    assert info.irrep == "A1um" and info.psq == 0 and info.obs_kind == "correlator"
    bad = CorrelatorObsInfo(BAD_KEY)
    assert bad.key is None and bad.time is None and "operator" in str(bad.key_error)
    built = CorrelatorObsInfo(key=ckey(1, 0, 7))
    assert built._name is None and built.name == ckey(1, 0, 7).to_sigmond()
    renamed = built.copy()
    renamed.name = built.name + "_x"
    assert renamed.key is None and built.key == ckey(1, 0, 7)
    vinfo = VEVObsInfo(key=VEVInfo(OPS[2]))
    assert vinfo.operator == OPS[2] and vinfo.obs_kind == "vev"


def test_duplicate_semantic_keys_differently_spelled():
    key = ckey(0, 1, 3)
    short = observable(key, series(RNG))
    # A non-default spelling of a default displacement is the same key.
    spelled = key.to_sigmond().replace("SD_5}", "SD_5 D3}")
    other = SigmondSampling(series(RNG), ObservableInfo(spelled, ensemble_info=ENS), JACK, True)
    with pytest.raises(ValueError, match="duplicate semantic key"):
        CorrelatorIndex.from_collection([short, other])


def test_mixed_ensembles_and_plans_are_rejected():
    a = observable(ckey(0, 0, 3), series(RNG))
    b = observable(ckey(0, 0, 4), series(RNG), ens=EnsembleInfo("other", 40))
    with pytest.raises(ValueError, match="ensemble"):
        CorrelatorIndex.from_collection([a, b])
    c = observable(ckey(0, 0, 4), series(RNG, BOOT), sampling_info=BOOT)
    with pytest.raises(ValueError, match="sampling plan"):
        CorrelatorIndex.from_collection([a, c])
    d = observable(ckey(0, 0, 4), series(RNG, None), sampling_info=None)
    with pytest.raises(ValueError, match="bins vs samplings"):
        CorrelatorIndex.from_collection([a, d])


def test_empty_index():
    index = CorrelatorIndex.from_collection([])
    assert index.operators == () and index.times == () and len(index) == 0


def test_channels_group_candidate_bases():
    moving = BasicLapHOperator.from_sigmond("pion P=(0,0,1) A2m_1 SS_0")
    other = CorrelatorAtTimeInfo(CorrelatorInfo(moving, moving), 3)
    items = _store(times=(3,), n=3) + [observable(other, series(RNG))]
    channels = CorrelatorIndex.from_collection(items).channels
    assert list(channels.values()) == [OPS, (moving,)]  # the pi-pi operator shares the channel
    assert list(channels) == [OPS[0].channel, moving.channel]


# ---------------------------------------------------------------------------
# HDF5 IO bridge (lazy)
# ---------------------------------------------------------------------------


@pytest.fixture
def lazy_file(tmp_path):
    items = _store(times=(2, 3, 4, 5), n=3)
    items += [vev(op, series(RNG)) for op in OPS]
    path = tmp_path / "corrs.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), items, group="corrs")
    return path, {o.observable_info.name: o for o in items}


def test_lazy_discovery_reads_no_samples(lazy_file):
    path, originals = lazy_file
    loader = SigmondLoader(str(path), lazy=True)
    index = loader.correlators
    assert loader.correlators is index  # cached
    assert set(index.operators) == set(OPS) and index.times == (2, 3, 4, 5)
    assert len(index.vev_operators) == 3
    assert not any(o.is_materialized for o in loader.observables)
    # The loader's own collection keeps the raw XML names, as typed infos.
    assert all(o.observable_info.name.startswith("<") for o in loader.observables)
    kinds = {type(o.observable_info) for o in loader.observables}
    assert kinds == {CorrelatorObsInfo, VEVObsInfo}


def test_correlators_take_part_in_collection_queries(lazy_file):
    path, _ = lazy_file
    obs = SigmondLoader(str(path), lazy=True).observables
    at3 = obs.filter(obs_kind="correlator", time=3)
    assert len(at3) == 9 and all(o.observable_info.time == 3 for o in at3)
    assert len(obs.filter(obs_kind="vev")) == 3
    assert len(obs.filter(irrep="A1um", psq=0)) == len(obs)
    assert len(obs.filter(obs_kind="correlator").group_by("correlator")) == 9
    assert not any(o.is_materialized for o in obs)


def test_file_keys_that_do_not_parse_still_load(tmp_path):
    good = observable(ckey(0, 0, 3), series(RNG))
    odd = SigmondSampling(
        series(RNG),
        ObservableInfo(good.observable_info.name.replace("A1um_1", "A9_1"), ensemble_info=ENS),
        JACK,
        True,
    )
    path = tmp_path / "odd.hdf5"
    SigmondWriter(create_backups=False).write_hdf5(str(path), [good, odd], group="g")
    loader = SigmondLoader(str(path))
    infos = [o.observable_info for o in loader.observables]
    assert all(isinstance(i, CorrelatorObsInfo) for i in infos)
    assert sorted(i.key is None for i in infos) == [False, True]
    with pytest.warns(UserWarning, match="did not parse"):
        index = loader.correlators
    assert len(index) == 1 and "unknown irrep" in index.unparsed[0][1]


def test_materialize_reads_lazy_records_into_one_block(lazy_file):
    path, originals = lazy_file
    obs = list(SigmondLoader(str(path), lazy=True).observables)[:7]
    materialize(obs + obs[:2])  # duplicates are read once
    assert all(o.is_materialized for o in obs)
    assert obs[0].data.base is not None and obs[0].data.base is obs[6].data.base  # one block
    for o in obs:
        assert np.array_equal(o.data, originals[o.observable_info.name].data)
    block = stack_data(obs)
    assert block.shape == (7, obs[0]._record.sample_length) and np.iscomplexobj(block)
    assert np.array_equal(block[3], obs[3].data)
    with pytest.raises(ValueError, match="sample lengths"):
        stack_data([obs[0], observable(ckey(0, 0, 9), series(RNG, BOOT), sampling_info=BOOT)])


def test_one_time_request_reads_only_its_cells(lazy_file):
    path, originals = lazy_file
    loader = SigmondLoader(str(path), lazy=True)
    matrix_info = CorrelatorMatrixInfo(OPS[:2])
    matrix = loader.correlators.matrix(matrix_info, times=[3])
    read = [o for o in loader.correlators._by_key.values() if isinstance(o, LazyDataMixin) and o.is_materialized]
    assert len(read) == 4  # both directions of every 2x2 cell at t=3, no VEVs
    assert matrix.times == (3,)


def test_re_im_fusion_and_orientation(lazy_file):
    path, originals = lazy_file
    index = SigmondLoader(str(path), lazy=True).correlators
    key = ckey(0, 1, 4, herm=True)
    stored = index.get(key)
    assert stored.is_complex
    assert np.array_equal(stored.data, originals[key.to_sigmond()].data)
    flipped = index.get(key.flipped())
    assert np.array_equal(flipped.data, originals[key.flipped().to_sigmond()].data)
    c = index.correlator(CorrelatorInfo(OPS[0], OPS[1]), hermitian=True)
    expected = 0.5 * (stored.data + np.conj(flipped.data))
    assert np.allclose(c.sel(time=4).data, expected)


def test_matrix_write_read_round_trip(tmp_path, lazy_file):
    path, _ = lazy_file
    index = SigmondLoader(str(path), lazy=True).correlators
    matrix_info = CorrelatorMatrixInfo(OPS, subtract_vev=True)
    matrix = index.matrix(matrix_info, times=[2, 3, 4])
    out = tmp_path / "connected.hdf5"
    matrix.write(out, group="conn")
    reloaded = SigmondLoader(str(out), lazy=True).correlators
    # Stored SubVEV: VEVs only on request.
    assert reloaded.matrix(matrix_info, times=[2, 3, 4]).vevs is None
    again = reloaded.matrix(matrix_info, times=[2, 3, 4], include_vevs=True)
    assert again.key_form == "HermMat SubVEV"
    assert again.operators == matrix.operators and again.times == matrix.times
    assert np.allclose(again.values.data, matrix.values.data)
    assert np.allclose(again.vevs.data, matrix.vevs.data)
    names = [info.name for info in again.values.infos.flat]
    assert names == [info.name for info in matrix.values.infos.flat]
    assert CorrelatorAtTimeInfo.from_sigmond(names[1]).subtract_vev


def test_correlator_example_runs(tmp_path, monkeypatch):
    """The linked matrix example runs without the optional fitting library."""
    import runpy
    from pathlib import Path

    example = Path(__file__).parents[1] / "examples" / "correlator_matrix.py"

    ops = [
        BasicLapHOperator.from_sigmond("pion P=(0,0,0) A1um_1 SD_5 D4"),
        BasicLapHOperator.from_sigmond("pion P=(0,0,0) A1um_1 SS_1"),
    ]
    items = [
        observable(CorrelatorAtTimeInfo(CorrelatorInfo(a, b), t, True), series(RNG))
        for t in range(3, 22)
        for a in ops
        for b in ops
    ]
    SigmondWriter(create_backups=False).write_hdf5(
        str(tmp_path / "corrs.hdf5"), items, group="corrs"
    )
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(example))
    assert namespace["matrix"].times == tuple(range(4, 21))
    assert len(namespace["at_t8"]) == 4
