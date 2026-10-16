"""
Tests for the array hierarchy: SigmondDataArray -> SigmondSamplingArray /
SigmondBinsArray. Array statistics must match the per-cell scalars, and every
operation must keep the kind (bins vs samplings).
"""

from __future__ import annotations

import numpy as np
import pytest

from sigmondsamplings.analysis import NonlinearBinsWarning
from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.sampling_array import (
    SigmondBinsArray,
    SigmondDataArray,
    SigmondSamplingArray,
)

RNG = np.random.default_rng(11)
ENS = EnsembleInfo("ens", 100)
N = 40
JACK = SamplingInfo("jackknife", N)
BOOT = SamplingInfo("bootstrap", 30, seed=5)


def _obs(i: int) -> ObservableInfo:
    return ObservableInfo(name=f"o{i}", index=i, ensemble_info=ENS)


def _bins_cells(n_cells: int = 4) -> list[SigmondBins]:
    return [SigmondBins(1.0 + 0.1 * RNG.standard_normal(N), _obs(i)) for i in range(n_cells)]


def _sampling_cells(info: SamplingInfo, n_cells: int = 4) -> list[SigmondSampling]:
    n = info.num_resamplings + 1
    return [
        SigmondSampling(1.0 + 0.1 * RNG.standard_normal(n), _obs(i), info) for i in range(n_cells)
    ]


@pytest.fixture(params=["bins", "jackknife", "bootstrap"])
def cells(request):
    if request.param == "bins":
        return _bins_cells()
    return _sampling_cells(JACK if request.param == "jackknife" else BOOT)


def _expected_cls(cells):
    return SigmondBinsArray if cells[0].sampling_info is None else SigmondSamplingArray


class TestConstruction:
    def test_from_samplings_picks_subclass(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        assert type(arr) is _expected_cls(cells)

    def test_from_collection_picks_subclass(self, cells):
        arr = SigmondDataArray.from_collection(cells, shape_by=["index"])
        assert type(arr) is _expected_cls(cells)
        assert arr.shape == (len(cells),)

    def test_from_data_bins(self):
        arr = SigmondDataArray.from_data(RNG.standard_normal((2, 3, N)), "raw", None)
        assert type(arr) is SigmondBinsArray
        assert arr.num_bins == N

    def test_wrong_subclass_rejected(self):
        with pytest.raises(TypeError, match="use SigmondBins "):
            SigmondSamplingArray.from_samplings(_bins_cells())
        with pytest.raises(TypeError, match="use SigmondSampling "):
            SigmondBinsArray.from_samplings(_sampling_cells(JACK))

    def test_mixed_leaves_rejected(self):
        with pytest.raises(ValueError, match="share sampling_info"):
            SigmondDataArray.from_samplings(_bins_cells(1) + _sampling_cells(JACK, 1))


class TestStatistics:
    def test_matches_scalars(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        np.testing.assert_allclose(arr.error.values, [c.error for c in cells], rtol=1e-12)
        np.testing.assert_allclose(arr.mean.values, [c.mean for c in cells], rtol=1e-12)
        np.testing.assert_allclose(
            arr.full_sample_value.values, [c.full_sample_value for c in cells], rtol=1e-12
        )

    def test_bootstrap_extras(self):
        cells = _sampling_cells(BOOT)
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        lower, upper = arr.confidence_interval()
        np.testing.assert_allclose(lower.values, [c.confidence_interval()[0] for c in cells])
        np.testing.assert_allclose(upper.values, [c.confidence_interval()[1] for c in cells])
        np.testing.assert_allclose(arr.bootstrap_bias.values, [c.bootstrap_bias for c in cells])


class TestJoinedMetadata:
    @pytest.mark.parametrize("axis", [-3, 2])
    def test_stack_rejects_axis_outside_spatial_dims(self, cells, axis):
        array = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        with pytest.raises(ValueError, match="stack axis out of bounds"):
            np.stack([array, array], axis=axis)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("same_parent", [False, True])
    def test_stack_preserves_cells(self, cells, axis, same_parent):
        left = SigmondDataArray.from_samplings(cells[:2], obs_info="left", dim_names=["t"])
        right = SigmondDataArray.from_samplings(
            cells[2:], obs_info=left.observable_info if same_parent else "right", dim_names=["t"]
        )
        result = np.stack([left, right], axis=axis)
        expected_infos = np.stack([left.infos, right.infos], axis=axis)
        np.testing.assert_array_equal(result.data, np.stack([left.data, right.data], axis=axis % 2))
        for index in np.ndindex(result.shape):
            assert result[index].observable_info is expected_infos[index]
        assert [c.observable_info for c in result.to_collection()] == list(expected_infos.flat)
        if same_parent:
            assert result.observable_info is left.observable_info

    def test_concatenate_preserves_selection_and_collection_infos(self, cells):
        left = SigmondDataArray.from_samplings(
            cells[:2], obs_info="left", dim_names=["t"], coords={"t": [0, 1]}
        )
        right = SigmondDataArray.from_samplings(
            cells[2:], obs_info="right", dim_names=["t"], coords={"t": [2, 3]}
        )
        result = np.concatenate([left, right])
        np.testing.assert_array_equal(result.data, np.stack([c.data for c in cells]))
        for i, cell in enumerate(cells):
            assert result.sel(t=i).observable_info is cell.observable_info
            assert result.to_collection()[i].observable_info is cell.observable_info
        selected = result.sel(t=[3, 0])
        assert selected[0].observable_info is cells[3].observable_info
        assert selected[1].observable_info is cells[0].observable_info

    def test_transpose_and_diagonal_preserve_joined_cells(self, cells):
        left = SigmondDataArray.from_samplings(cells[:2], obs_info="left", dim_names=["t"])
        right = SigmondDataArray.from_samplings(cells[2:], obs_info="right", dim_names=["t"])
        joined = SigmondDataArray.stack([left, right], dim_name="row")
        transposed = joined.transpose("t", "row")
        for row in range(2):
            for t in range(2):
                assert transposed[t, row].observable_info is cells[2 * row + t].observable_info
        diagonal = transposed.diagonal("t", "row")
        assert diagonal[0].observable_info is cells[0].observable_info
        assert diagonal[1].observable_info is cells[3].observable_info

    def test_join_preserves_default_element_infos(self):
        left = SigmondDataArray.from_data(np.zeros((2, N)), "left", None, dim_names=["t"])
        right = SigmondDataArray.from_data(np.ones((2, N)), "right", None, dim_names=["t"])
        result = np.concatenate([left, right])
        for i, info in enumerate([*left.infos, *right.infos]):
            assert result[i].observable_info is info


class TestKindPreserved:
    def test_indexing_returns_cell_type(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        assert type(arr[0]) is type(cells[0])
        assert type(arr.isel(t=1)) is type(cells[0])
        assert all(type(c) is type(cells[0]) for c in arr.to_collection())

    def test_operations_keep_subclass(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        cls = _expected_cls(cells)
        assert type(arr + arr) is cls
        assert type(2.0 * arr) is cls
        assert type(arr + cells[0]) is cls
        assert type(arr.copy()) is cls
        assert type(arr[1:]) is cls
        assert type(np.concatenate([arr, arr])) is cls
        assert type(np.stack([arr, arr])) is cls
        assert type(arr.obs.replace(op_type="x")) is cls

    def test_mixing_kinds_rejected(self):
        bins = SigmondDataArray.from_samplings(_bins_cells(), dim_names=["t"])
        samps = SigmondDataArray.from_samplings(_sampling_cells(JACK), dim_names=["t"])
        with pytest.raises(TypeError, match="Cannot mix"):
            bins + samps
        with pytest.raises(TypeError, match="Cannot mix"):
            bins + _sampling_cells(JACK, 1)[0]
        with pytest.raises(ValueError):
            np.concatenate([bins, samps])

    def test_bins_array_warns_on_nonlinear(self):
        arr = SigmondDataArray.from_samplings(_bins_cells(), dim_names=["t"])
        with pytest.warns(NonlinearBinsWarning):
            arr * arr
        with pytest.warns(NonlinearBinsWarning):
            np.log(arr)


class TestBinsArrayResample:
    @pytest.mark.parametrize("info", [JACK, BOOT])
    def test_matches_scalar_resample(self, info):
        cells = _bins_cells()
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        result = arr.resample(info)
        assert type(result) is SigmondSamplingArray
        assert result.sampling_info == info
        expected = np.stack([np.asarray(c.resample(info)) for c in cells])
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12)

    def test_multiple_statistics_rename(self):
        arr = SigmondDataArray.from_samplings(_bins_cells(), dim_names=["t"])
        mean, var = arr.resample(BOOT, statistic=["mean", "variance"])
        assert var.observable_info.name.endswith("_variance")
        assert var[0].observable_info.name == "o0_variance"
        assert mean[0].observable_info.name == "o0_mean"

    def test_jackknife_count_must_match_blocks(self):
        arr = SigmondDataArray.from_samplings(_bins_cells(), dim_names=["t"])
        with pytest.raises(ValueError, match="num_resamplings"):
            arr.resample(SamplingInfo("jackknife", N - 1))


class TestUnifiedType:
    def test_single_observable_is_0d(self, cells):
        c = cells[0]
        assert c.ndim == 0 and c.shape == () and c.dims == ()
        assert c.infos[()] is c.observable_info
        with pytest.raises(TypeError, match="cannot be indexed"):
            c[0]

    def test_array_is_same_class_as_cells(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        assert isinstance(arr, type(cells[0]))
        assert arr.ndim == 1 and arr.shape == (len(cells),)

    def test_collections_reject_nd(self, cells):
        from sigmondsamplings.observable_collection import ObservableCollection

        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"])
        with pytest.raises(TypeError, match="single observables"):
            ObservableCollection([arr])

    def test_stack_single_observables(self, cells):
        arr = SigmondDataArray.stack(cells, dim_name="t")
        assert arr.shape == (len(cells),) and arr.dims == ("t",)
        assert arr[1].observable_info is cells[1].observable_info
        np.testing.assert_array_equal(np.asarray(arr), np.stack([c.data for c in cells]))

    def test_transforms_keep_labels(self, cells):
        arr = SigmondDataArray.from_samplings(cells, dim_names=["t"], coords={"t": [3, 4, 5, 6]})
        out = arr.bounded(0.0, 1.0)
        assert out.dims == ("t",) and list(out.coords["t"]) == [3, 4, 5, 6]
        assert out[0].observable_info is cells[0].observable_info


class TestLabels:
    def _arr(self, cells, t=(0, 1, 2, 3)):
        return SigmondDataArray.from_samplings(cells, dim_names=["t"], coords={"t": list(t)})

    def test_broadcast_by_dim_name(self):
        a = SigmondDataArray.from_data(RNG.standard_normal((3, N + 1)), "a", JACK, dim_names=["x"])
        b = SigmondDataArray.from_data(RNG.standard_normal((4, N + 1)), "b", JACK, dim_names=["y"])
        out = a * b
        assert out.dims == ("x", "y") and out.shape == (3, 4)
        np.testing.assert_allclose(np.asarray(out), a.data[:, None, :] * b.data[None, :, :])

    def test_misaligned_labels_raise(self, cells):
        with pytest.raises(ValueError):
            self._arr(cells) + self._arr(cells, t=(1, 2, 3, 4))

    def test_sel_slice_and_list(self, cells):
        arr = self._arr(cells)
        assert arr.sel(t=slice(1, 2)).shape == (2,)
        sub = arr.sel(t=[3, 0])
        assert [i.name for i in sub.infos] == ["o3", "o0"]
        assert arr.sel(t=2).observable_info is cells[2].observable_info

    def test_getitem_keeps_coords(self, cells):
        sub = self._arr(cells)[1:3]
        assert list(sub.coords["t"]) == [1, 2]
        assert sub[...].shape == (2,)

    def test_editor_does_not_touch_original(self, cells):
        arr = self._arr(cells)
        ed = arr.editor()
        ed.replace_obs(name="zzz")
        ed.set_data(0, np.zeros(arr.num_samples))
        new = ed.freeze()
        assert arr[0].observable_info.name == "o0"
        assert new[0].observable_info.name == "zzz"
        assert not np.allclose(arr[0].data, 0.0)


class TestVectorizedResample:
    @pytest.mark.parametrize("info", [JACK, BOOT])
    def test_mean_matches_trace_loop(self, info):
        from sigmondsamplings.bins import _block_layout, _block_resample

        data = RNG.standard_normal((3, N))
        ens = ENS
        (fast,) = _block_resample(data, ens, info, ["mean"])
        # reference: explicit traces, as the pre-vectorized implementation built them
        n_blocks, size = _block_layout(data.shape[-1], ens)
        blocks = data[:, : n_blocks * size].reshape(3, n_blocks, size)
        if info.method == "bootstrap":
            idx = np.random.RandomState(info.seed).randint(
                0, n_blocks, (info.num_resamplings, n_blocks)
            )
        else:
            idx = [np.delete(np.arange(n_blocks), i) for i in range(n_blocks)]
        ref = np.stack([blocks[:, i, :].reshape(3, -1).mean(-1) for i in idx], axis=-1)
        np.testing.assert_allclose(fast[:, 1:], ref, rtol=1e-12)
        np.testing.assert_allclose(fast[:, 0], blocks.reshape(3, -1).mean(-1), rtol=1e-12)


class TestScalarText:
    def test_str_and_repr_use_pdg_format(self, cells):
        cell = cells[0]
        assert str(cell) == cell.pdg_format()
        assert f"value={cell.pdg_format()}" in repr(cell)
        assert "±" not in str(cell) + repr(cell)
