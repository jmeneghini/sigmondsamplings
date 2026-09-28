"""
N-D primitives used by the correlator layer: transpose, rename_dims, diagonal,
apply_matrix, strict/key-function pivots and per-cell infos in from_data.
Each is checked on bins and samplings, real and complex, against plain NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling

RNG = np.random.default_rng(3)
ENS = EnsembleInfo("ens", 30)
BOOT = SamplingInfo("bootstrap", 12, seed=1)


@pytest.fixture(params=["bins-real", "bins-complex", "samp-real", "samp-complex"])
def cube(request):
    """A (time=3, sink=2, source=2) array with labeled sink/source."""
    kind, part = request.param.split("-")
    n = 30 if kind == "bins" else BOOT.num_resamplings + 1
    buf = RNG.normal(size=(3, 2, 2, n))
    if part == "complex":
        buf = buf + 1j * RNG.normal(size=buf.shape)
    info = None if kind == "bins" else BOOT
    return SigmondData.from_data(
        buf,
        "C",
        info,
        dim_names=("time", "sink", "source"),
        coords={"time": [4, 5, 6], "sink": ["a", "b"], "source": ["a", "b"]},
        ensemble_info=ENS,
        is_complex=part == "complex",
    )


def test_transpose_moves_data_and_infos(cube):
    t = cube.transpose("sink", "time", "source")
    assert t.dims == ("sink", "time", "source")
    assert np.array_equal(t.data, np.transpose(cube.data, (1, 0, 2, 3)))
    assert t.infos[1, 2, 0] is cube.infos[2, 1, 0]
    assert type(t) is type(cube)
    with pytest.raises(ValueError):
        cube.transpose("sink", "time")


def test_rename_dims_moves_coords(cube):
    r = cube.rename_dims(sink="row")
    assert r.dims == ("time", "row", "source")
    assert list(r.coords["row"]) == ["a", "b"]
    assert r.data is cube.data
    with pytest.raises(ValueError):
        cube.rename_dims(sink="source")
    with pytest.raises(KeyError):
        cube.rename_dims(nope="x")


def test_diagonal_keeps_infos_and_labels(cube):
    d = cube.diagonal("sink", "source", "operator")
    assert d.dims == ("time", "operator")
    assert list(d.coords["operator"]) == ["a", "b"]
    assert np.array_equal(d.data[:, 1], cube.data[:, 1, 1])
    assert d.infos[2, 1] is cube.infos[2, 1, 1]
    assert type(d) is type(cube)


def test_diagonal_rejects_mismatched_labels():
    buf = RNG.normal(size=(2, 2, 5))
    arr = SigmondData.from_data(
        buf,
        "x",
        BOOT.__class__("jackknife", 4),
        dim_names=("a", "b"),
        coords={"a": ["p", "q"], "b": ["q", "p"]},
    )
    with pytest.raises(ValueError, match="different coords"):
        arr.diagonal("a", "b")


def test_vectorized_isel_keeps_diagonal_labels(cube):
    idx = xr.DataArray([0, 1], dims="operator")
    d = cube.isel(sink=idx, source=idx)
    assert d.dims == ("time", "operator")
    assert list(d.coords["operator"]) == ["a", "b"]


@pytest.mark.parametrize("conj", [False, True])
def test_apply_matrix_matches_einsum(cube, conj):
    T = np.array([[1.0, 0.5j, 2.0], [0.25, -1.0, 1j]])
    out = cube.apply_matrix(T, "sink", conj=conj, coords=["x", "y", "z"])
    M = T.conj() if conj else T
    assert np.allclose(out.data, np.einsum("ia,tijs->tajs", M, cube.data))
    assert out.dims == cube.dims
    assert list(out.coords["sink"]) == ["x", "y", "z"]
    assert list(out.coords["source"]) == ["a", "b"]
    assert out.is_complex
    assert out.sampling_info == cube.sampling_info
    assert out.infos.shape == (3, 3, 2)


def test_apply_matrix_real_stays_real():
    buf = RNG.normal(size=(2, 4, 13))
    arr = SigmondSampling.from_data(buf, "v", BOOT, dim_names=("t", "op"))
    out = arr.apply_matrix(np.eye(4)[:, :2], "op", new_dim="level")
    assert not out.is_complex and out.dims == ("t", "level")
    assert np.array_equal(out.data, buf[:, :2])


def test_apply_matrix_validates(cube):
    with pytest.raises(ValueError, match="shape"):
        cube.apply_matrix(np.ones((3, 2)), "sink")
    with pytest.raises(ValueError, match="non-finite"):
        cube.apply_matrix(np.array([[np.nan, 0], [0, 1]]), "sink")
    with pytest.raises(ValueError, match="collides"):
        cube.apply_matrix(np.eye(2), "sink", new_dim="time")


@pytest.mark.filterwarnings("ignore::sigmondsamplings.analysis.NonlinearBinsWarning")
def test_matmul_over_sample_axis_is_not_available(cube):
    """A raw matmul would contract the sample axis; it must not silently work."""
    with pytest.raises(NotImplementedError):
        np.matmul(cube, cube)


def _cells(n=4, sampling=BOOT):
    out = []
    for i in range(n):
        info = ObservableInfo(f"c{i}", i, ensemble_info=ENS)
        out.append(SigmondSampling(RNG.normal(size=sampling.num_resamplings + 1), info, sampling))
    return out


def test_from_collection_key_functions():
    cells = _cells(4)
    arr = SigmondData.from_collection(
        cells, {"row": lambda info: info.index // 2, "col": lambda info: info.index % 2}
    )
    assert arr.dims == ("row", "col") and arr.shape == (2, 2)
    assert arr.infos[1, 0] is cells[2].observable_info
    assert np.array_equal(arr.data[1, 1], cells[3].data)


def test_from_collection_missing_raise_lists_cells():
    cells = _cells(3)
    with pytest.raises(ValueError, match="1 cell\\(s\\) missing"):
        SigmondData.from_collection(
            cells,
            {"row": lambda info: info.index // 2, "col": lambda info: info.index % 2},
            missing="raise",
        )
    with pytest.raises(ValueError):
        SigmondData.from_collection(cells, ["index"], missing="raise", sentinel=0.0)


def test_from_data_with_infos():
    infos = np.empty((2,), dtype=object)
    infos[0] = ObservableInfo("<CorrT>x</CorrT>", ensemble_info=ENS)
    infos[1] = ObservableInfo("<CorrT>y</CorrT>", ensemble_info=ENS)
    arr = SigmondData.from_data(RNG.normal(size=(2, 13)), "a", BOOT, dim_names=("k",), infos=infos)
    assert arr.infos[1] is infos[1] and arr.infos is not infos
    with pytest.raises(ValueError, match="shape"):
        SigmondData.from_data(RNG.normal(size=(3, 13)), "a", BOOT, infos=infos)
    bad = np.array([infos[0], "nope"], dtype=object)
    with pytest.raises(TypeError):
        SigmondData.from_data(RNG.normal(size=(2, 13)), "a", BOOT, infos=bad)


def test_bins_apply_matrix_then_resample_is_linear():
    buf = RNG.normal(size=(2, 30))
    bins = SigmondBins.from_data(buf, "v", None, dim_names=("op",), ensemble_info=ENS)
    T = np.array([[1.0, 2.0], [0.5, -1.0]])
    a = bins.apply_matrix(T, "op").resample(BOOT)
    b = bins.resample(BOOT).apply_matrix(T, "op")
    assert np.allclose(a.data, b.data)
