"""
Tests for SigmondBins: construction, stats, arithmetic, Re/Im grouping,
ObservableCollection interop, and HDF5 round-trip (with CorrT observables).

Run with:  python -m pytest tests/test_bins.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.loader import SigmondLoader
from sigmondsamplings.obervable_collection import ObservableCollection
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.writer import SigmondWriter

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

N_BINS = 200
RNG = np.random.default_rng(2026)

ENS_A = EnsembleInfo("ens_A", 100)
ENS_B = EnsembleInfo("ens_B", 80)


def _make_bins(
    name: str,
    data: np.ndarray,
    ens: EnsembleInfo = ENS_A,
    is_complex: bool = False,
) -> SigmondBins:
    obs = ObservableInfo(
        name=name, index=0, op_type="n", re_im="re", ensemble_info=ens
    )
    return SigmondBins(data=data, observable_info=obs, is_complex=is_complex)


@pytest.fixture
def bins_a() -> SigmondBins:
    data = 1.0 + 0.1 * RNG.standard_normal(N_BINS)
    return _make_bins("obs_a", data)


@pytest.fixture
def bins_b() -> SigmondBins:
    data = 2.5 + 0.2 * RNG.standard_normal(N_BINS)
    return _make_bins("obs_b", data)


@pytest.fixture
def bins_other_ens() -> SigmondBins:
    data = 3.0 + 0.15 * RNG.standard_normal(N_BINS)
    return _make_bins("obs_c", data, ens=ENS_B)


@pytest.fixture
def bins_wrong_size() -> SigmondBins:
    data = 1.0 + 0.1 * RNG.standard_normal(N_BINS // 2)
    return _make_bins("obs_short", data)


# ──────────────────────────────────────────────────────────────────────────────
# Construction and metadata
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_num_bins(self, bins_a):
        assert bins_a.num_bins == N_BINS
        assert len(bins_a) == N_BINS

    def test_sampling_info_is_none(self, bins_a):
        assert bins_a.sampling_info is None

    def test_ensemble_info(self, bins_a):
        assert bins_a.ensemble_info == ENS_A

    def test_list_input_coerced(self):
        b = _make_bins("x", [1.0, 2.0, 3.0, 4.0])
        assert isinstance(b.data, np.ndarray)
        assert b.num_bins == 4

    def test_rejects_2d_data(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            _make_bins("x", np.ones((3, 3)))

    def test_complex_dtype(self):
        data = np.array([1.0 + 2.0j, 3.0 + 4.0j])
        b = _make_bins("z", data, is_complex=True)
        assert b.is_complex
        assert np.iscomplexobj(b.data)


# ──────────────────────────────────────────────────────────────────────────────
# Statistics (no resampling)
# ──────────────────────────────────────────────────────────────────────────────


class TestStats:
    def test_mean_matches_numpy(self, bins_a):
        np.testing.assert_allclose(
            bins_a.mean, np.mean(np.asarray(bins_a.data)), rtol=1e-12
        )

    def test_full_sample_value_equals_mean(self, bins_a):
        assert bins_a.full_sample_value == bins_a.mean

    def test_error_is_standard_error(self, bins_a):
        arr = np.asarray(bins_a.data)
        expected = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        np.testing.assert_allclose(bins_a.error, expected, rtol=1e-12)

    def test_error_nan_for_single_bin(self):
        b = _make_bins("one", np.array([1.0]))
        assert np.isnan(b.error)

    def test_confidence_interval_bounds_mean(self, bins_a):
        lo, hi = bins_a.confidence_interval(0.68)
        assert lo < bins_a.mean < hi

    def test_pdg_format_is_string(self, bins_a):
        s = bins_a.pdg_format()
        assert isinstance(s, str)
        assert len(s) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ──────────────────────────────────────────────────────────────────────────────


class TestArithmetic:
    def test_add_bins_scalar(self, bins_a):
        result = bins_a + 1.0
        assert isinstance(result, SigmondBins)
        np.testing.assert_allclose(
            np.asarray(result.data), np.asarray(bins_a.data) + 1.0, rtol=1e-12
        )

    def test_scalar_add_rbins(self, bins_a):
        result = 2.0 + bins_a
        assert isinstance(result, SigmondBins)
        np.testing.assert_allclose(
            np.asarray(result.data), 2.0 + np.asarray(bins_a.data), rtol=1e-12
        )

    def test_sub_two_bins(self, bins_a, bins_b):
        result = bins_a - bins_b
        assert isinstance(result, SigmondBins)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.asarray(bins_a.data) - np.asarray(bins_b.data),
            rtol=1e-12,
        )

    def test_ratio_two_bins(self, bins_a, bins_b):
        result = bins_a / bins_b
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.asarray(bins_a.data) / np.asarray(bins_b.data),
            rtol=1e-12,
        )

    def test_np_log(self, bins_a):
        result = np.log(bins_a)
        assert isinstance(result, SigmondBins)
        np.testing.assert_allclose(
            np.asarray(result.data), np.log(np.asarray(bins_a.data)), rtol=1e-12
        )

    def test_negation(self, bins_a):
        result = -bins_a
        np.testing.assert_allclose(
            np.asarray(result.data), -np.asarray(bins_a.data), rtol=1e-12
        )

    def test_incompatible_num_bins_rejected(self, bins_a, bins_wrong_size):
        with pytest.raises(ValueError, match="different number of bins"):
            _ = bins_a + bins_wrong_size

    def test_incompatible_ensemble_rejected(self, bins_a, bins_other_ens):
        with pytest.raises(ValueError, match="different ensembles"):
            _ = bins_a + bins_other_ens

    def test_mixing_with_sampling_rejected(self, bins_a):
        sinfo = SamplingInfo("bootstrap", 100, seed=1)
        samp = SigmondSampling(
            data=np.ones(101),
            observable_info=bins_a.observable_info,
            sampling_info=sinfo,
        )
        with pytest.raises(TypeError, match="Cannot mix"):
            _ = bins_a + samp


# ──────────────────────────────────────────────────────────────────────────────
# Resampling still works
# ──────────────────────────────────────────────────────────────────────────────


class TestResample:
    def test_bootstrap_returns_sampling(self, bins_a):
        sinfo = SamplingInfo("bootstrap", 64, seed=7)
        result = bins_a.resample(sinfo, statistic="mean")
        assert isinstance(result, SigmondSampling)
        assert result.sampling_info == sinfo
        # full-sample slot + N resamplings
        assert len(result.data) == 1 + 64

    def test_jackknife_returns_sampling(self, bins_a):
        sinfo = SamplingInfo("jackknife", N_BINS, seed=0)
        result = bins_a.resample(sinfo, statistic="mean")
        assert isinstance(result, SigmondSampling)


# ──────────────────────────────────────────────────────────────────────────────
# ObservableCollection interoperability
# ──────────────────────────────────────────────────────────────────────────────


class TestCollection:
    def test_build_from_bins_list(self, bins_a, bins_b):
        coll = ObservableCollection([bins_a, bins_b])
        assert len(coll) == 2

    def test_filter_by_name(self, bins_a, bins_b):
        coll = ObservableCollection([bins_a, bins_b])
        filtered = coll.filter(name="obs_a")
        assert len(filtered) == 1
        assert filtered[0] is bins_a

    def test_find(self, bins_a, bins_b):
        coll = ObservableCollection([bins_a, bins_b])
        found = coll.find(name="obs_b")
        assert found is bins_b

    def test_to_numpy_shape(self, bins_a, bins_b):
        coll = ObservableCollection([bins_a, bins_b])
        arr = coll.to_numpy()
        assert arr.shape == (2, N_BINS)

    def test_val_mean_accessor(self, bins_a, bins_b):
        coll = ObservableCollection([bins_a, bins_b])
        means = coll.val.mean
        assert len(means) == 2


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 round-trip (including CorrT-style observable names)
# ──────────────────────────────────────────────────────────────────────────────


CORRT_NAME = (
    "<CorrT>GI{isodoublet S=1 P=(0,0,0) A1g_1 ROT 11} "
    "GI{isodoublet S=1 P=(0,0,0) A1g_1 ROT 11} time=26 HermMat</CorrT>"
)


class TestRoundTrip:
    def test_write_read_plain(self, tmp_path, bins_a, bins_b):
        out = tmp_path / "plain_bins.hdf5"
        coll = ObservableCollection([bins_a, bins_b])
        coll.to_hdf5(str(out), create_backups=False, root_path="test_bins")

        assert out.exists()

        loader = SigmondLoader(str(out), hdf5_path="test_bins")
        assert loader.file_kind == "bins"
        loaded = loader.observables
        assert len(loaded) == 2

        by_name = {s.observable_info.name: s for s in loaded}
        for original in (bins_a, bins_b):
            match = by_name[original.observable_info.name]
            assert isinstance(match, SigmondBins)
            np.testing.assert_allclose(
                np.asarray(match.data), np.asarray(original.data), rtol=1e-12
            )

    def test_convert_format_preserves_bins_kind(self, tmp_path, bins_a, bins_b):
        src = tmp_path / "src_bins.hdf5"
        out = tmp_path / "converted_bins.hdf5"

        writer = SigmondWriter(create_backups=False)
        writer.write_bins_hdf5(
            str(src), [bins_a, bins_b], root_path="src_bins", overwrite=True
        )

        writer.convert_format(
            str(src),
            str(out),
            output_format="hdf5",
            hdf5_root_path="/dest_bins/",
            overwrite=True,
        )

        loader = SigmondLoader(str(out), hdf5_path="dest_bins")
        assert loader.file_kind == "bins"

        loaded = loader.observables
        assert len(loaded) == 2

        by_name = {s.observable_info.name: s for s in loaded}
        for original in (bins_a, bins_b):
            match = by_name[original.observable_info.name]
            assert isinstance(match, SigmondBins)
            np.testing.assert_allclose(
                np.asarray(match.data), np.asarray(original.data), rtol=1e-12
            )

    def test_write_read_corrt(self, tmp_path):
        data = 0.5 + 0.05 * RNG.standard_normal(N_BINS)
        bins = _make_bins(CORRT_NAME, data)
        out = tmp_path / "corrt_bins.hdf5"

        coll = ObservableCollection([bins])
        coll.to_hdf5(str(out), create_backups=False, root_path="test_bins")

        loader = SigmondLoader(str(out), hdf5_path="test_bins")
        loaded = loader.observables
        assert len(loaded) == 1

        obs = loaded[0]
        assert isinstance(obs, SigmondBins)
        # CorrT name must round-trip verbatim so sigmond_query recognizes it
        assert "CorrT" in obs.observable_info.name
        assert "time=26" in obs.observable_info.name
        np.testing.assert_allclose(
            np.asarray(obs.data), np.asarray(bins.data), rtol=1e-12
        )

    def test_mixed_collection_rejected(self, tmp_path, bins_a):
        sinfo = SamplingInfo("bootstrap", 100, seed=1)
        samp = SigmondSampling(
            data=np.ones(101),
            observable_info=ObservableInfo(
                "obs_samp", 0, "n", "re", ENS_A
            ),
            sampling_info=sinfo,
        )
        coll = ObservableCollection([bins_a, samp])
        # SingleEnsembleCollection compatibility check fires first (mixed
        # sampling metadata), and if that passes the writer's type check takes
        # over. Either way we expect a ValueError on export.
        with pytest.raises(ValueError):
            coll.to_hdf5(
                str(tmp_path / "mixed.hdf5"),
                create_backups=False,
                root_path="test_bins",
            )


# ──────────────────────────────────────────────────────────────────────────────
# fstream loading (requires sigmond_query + real bins file)
# ──────────────────────────────────────────────────────────────────────────────


TETRAQUARK_BINS = Path(
    "/home/jmeneghini/research/spectrum_analysis/tetraquarks/eff_mass_CI_plots/"
    "samplings/with_tq_single_pivot_n3_m6_d12_c200.dat"
)


@pytest.mark.skipif(
    shutil.which("sigmond_query") is None or not TETRAQUARK_BINS.exists(),
    reason="sigmond_query and the tetraquark bins file must be available",
)
class TestFstreamBins:
    def test_load_fstream_bins(self):
        loader = SigmondLoader(str(TETRAQUARK_BINS))
        assert loader.file_kind == "bins"
        obs = loader.observables
        assert len(obs) > 0
        first = obs[0]
        assert isinstance(first, SigmondBins)
        assert first.sampling_info is None
        assert first.num_bins > 0

    def test_fstream_roundtrip_to_hdf5(self, tmp_path):
        loader = SigmondLoader(str(TETRAQUARK_BINS))
        original = loader.observables
        out = tmp_path / "tq_roundtrip.hdf5"

        ObservableCollection(list(original)).to_hdf5(
            str(out), create_backups=False, root_path="test_bins"
        )

        loader2 = SigmondLoader(str(out), hdf5_path="test_bins")
        reloaded = loader2.observables
        assert len(reloaded) == len(original)

        orig_by_name = {s.observable_info.name: s for s in original}
        for s in reloaded:
            match = orig_by_name[s.observable_info.name]
            np.testing.assert_allclose(
                np.asarray(s.data), np.asarray(match.data), rtol=1e-10
            )
