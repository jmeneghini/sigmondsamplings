"""
Loader + writer tests against real Sigmond fixture files (tests/data/).

Fixtures (all subset down to small sizes from real lattice-QCD output):
  - corr_matrix_samplings.hdf5  : HDF5 samplings, complex CorrT-keyed observables
  - energy_levels_samplings.hdf5: HDF5 samplings, real scalar (Info-form) observables
  - energy_samplings_pion.smp   : fstream samplings, names containing '/'  (needs sigmond_query)
  - tetraquark_bins.hdf5        : HDF5 bins, complex CorrT-keyed observables

Run with:  python -m pytest tests/test_loader_writer.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.io.loader import DEFAULT_GROUP, SigmondLoader
from sigmondsamplings.io.writer import SigmondWriter
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.sampling import SigmondSampling

DATA_DIR = Path(__file__).parent / "data"
CORR_HDF5 = DATA_DIR / "corr_matrix_samplings.hdf5"
ENERGY_HDF5 = DATA_DIR / "energy_levels_samplings.hdf5"
PION_SMP = DATA_DIR / "energy_samplings_pion.smp"
BINS_HDF5 = DATA_DIR / "tetraquark_bins.hdf5"

needs_sigmond_query = pytest.mark.skipif(
    shutil.which("sigmond_query") is None, reason="sigmond_query not available"
)


def _copy_to(tmp_path: Path, src: Path) -> Path:
    """Copy a committed fixture into tmp_path so mutating tests never touch it."""
    dst = tmp_path / src.name
    shutil.copy2(src, dst)
    return dst


def _by_name(observables) -> dict[str, object]:
    return {o.observable_info.name: o for o in observables}


# Root-group names inside the assembled multi-path fixture.
CORR_PATH = "isotriplet_S0_A1gm_1_P0"
LEVELS_PATH = "samplings"


@pytest.fixture
def multi_path_hdf5(tmp_path) -> Path:
    """
    Build an HDF5 file holding two root groups under one global /Info group.

    Real Sigmond/IOMap files accumulate many root groups (see the format spec);
    the writer only ever emits one, so we assemble the multi-path case at the
    h5py level by copying two committed single-path fixtures together. Both are
    samplings files, so the shared /Info/FIdentifier stays consistent.
    """
    out = tmp_path / "multi_root.hdf5"
    with (
        h5py.File(CORR_HDF5, "r") as a,
        h5py.File(ENERGY_HDF5, "r") as b,
        h5py.File(out, "w") as dst,
    ):
        a.copy(a["Info"], dst, "Info")
        a.copy(a[CORR_PATH], dst, CORR_PATH)
        b.copy(b["samplings"], dst, LEVELS_PATH)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadCorrMatrixHDF5:
    """Complex, CorrT-keyed samplings (the case that previously round-tripped wrong)."""

    def test_kind_and_count(self):
        ld = SigmondLoader(str(CORR_HDF5))
        assert ld.file_kind == "samplings"
        assert ld.group == "isotriplet_S0_A1gm_1_P0"
        assert len(ld.observables) == 6

    def test_all_complex_corrt(self):
        ld = SigmondLoader(str(CORR_HDF5))
        obs = list(ld.observables)
        assert all(o.is_complex for o in obs)
        assert all(o.observable_info.name.startswith("<CorrT>") for o in obs)

    def test_sampling_info(self):
        ld = SigmondLoader(str(CORR_HDF5))
        si = ld.observables.sampling_info
        assert si.method == "bootstrap"
        assert si.num_resamplings == 1000
        assert si.seed == 6754
        assert si.boot_skip == 127
        # data length = 1 full sample + num_resamplings
        assert len(list(ld.observables)[0].data) == 1001

    def test_ensemble_info(self):
        ld = SigmondLoader(str(CORR_HDF5))
        ens = ld.observables.ensemble_info
        assert ens.name == "clover_s32_t256_ud860_s743"
        assert ens.num_bins == 412

    def test_get_file_info(self):
        ld = SigmondLoader(str(CORR_HDF5))
        ens, samp, obs_infos = ld.get_file_info()
        assert ens.num_bins == 412
        assert samp.method == "bootstrap"
        assert len(obs_infos) == 6


class TestLoadEnergyLevelsHDF5:
    """Real, simple Info-form samplings."""

    def test_kind_and_count(self):
        ld = SigmondLoader(str(ENERGY_HDF5))
        assert ld.file_kind == "samplings"
        assert ld.group == "samplings"
        assert len(ld.observables) == 54

    def test_all_real_simple_names(self):
        ld = SigmondLoader(str(ENERGY_HDF5))
        obs = list(ld.observables)
        assert not any(o.is_complex for o in obs)
        assert not any(o.observable_info.name.startswith("<") for o in obs)

    def test_known_observable_present(self):
        ld = SigmondLoader(str(ENERGY_HDF5))
        assert "K(0)_elab" in _by_name(ld.observables)


class TestLoadBinsHDF5:
    """Complex, CorrT-keyed bins (no sampling info)."""

    def test_kind_and_count(self):
        ld = SigmondLoader(str(BINS_HDF5))
        assert ld.file_kind == "bins"
        assert len(ld.observables) == 5

    def test_objects_are_bins(self):
        ld = SigmondLoader(str(BINS_HDF5))
        obs = list(ld.observables)
        assert all(isinstance(o, SigmondBins) for o in obs)
        assert all(o.sampling_info is None for o in obs)
        assert all(o.is_complex for o in obs)
        assert all(o.num_bins == 2000 for o in obs)

    def test_get_file_info_sampling_is_none(self):
        ld = SigmondLoader(str(BINS_HDF5))
        _, samp, obs_infos = ld.get_file_info()
        assert samp is None
        assert len(obs_infos) == 5


@needs_sigmond_query
class TestLoadPionFstream:
    """fstream samplings loaded via sigmond_query; names contain '/'."""

    def test_kind_and_count(self):
        ld = SigmondLoader(str(PION_SMP))
        assert ld.file_kind == "samplings"
        assert ld.group is None  # fstream has no HDF5 path
        assert len(ld.observables) == 51

    def test_names_contain_slashes(self):
        ld = SigmondLoader(str(PION_SMP))
        assert any("/" in o.observable_info.name for o in ld.observables)


# ──────────────────────────────────────────────────────────────────────────────
# Writer round-trips
# ──────────────────────────────────────────────────────────────────────────────


def _assert_samplings_roundtrip(original, reloaded):
    assert len(reloaded) == len(original)
    orig = _by_name(original)
    for r in reloaded:
        assert r.observable_info.name in orig, r.observable_info.name[:60]
        match = orig[r.observable_info.name]
        assert r.is_complex == match.is_complex
        np.testing.assert_allclose(r.data, match.data, rtol=1e-12)


class TestSamplingsRoundTrip:
    def test_corrt_complex_roundtrip(self, tmp_path):
        # Regression guard: complex CorrT keys must survive write_hdf5 verbatim.
        original = list(SigmondLoader(str(CORR_HDF5)).observables)
        out = tmp_path / "corr_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), original, group="samplings", overwrite=True
        )
        reloaded = list(SigmondLoader(str(out)).observables)
        assert all(o.is_complex for o in reloaded)
        assert all(o.observable_info.name.startswith("<CorrT>") for o in reloaded)
        _assert_samplings_roundtrip(original, reloaded)

    def test_simple_real_roundtrip(self, tmp_path):
        original = list(SigmondLoader(str(ENERGY_HDF5)).observables)
        out = tmp_path / "energy_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), original, group="samplings", overwrite=True
        )
        reloaded = list(SigmondLoader(str(out)).observables)
        _assert_samplings_roundtrip(original, reloaded)


class TestEnergyAttrsRoundTrip:
    """Energy metadata (incl. non-interacting pairs) persists via dataset attrs."""

    @staticmethod
    def _write(tmp_path):
        from sigmondsamplings.energy_levels import (
            EnergyObsInfo,
            Particle,
            SHEnergyObsInfo,
        )
        from sigmondsamplings.sampling import SamplingInfo

        si = SamplingInfo("jackknife", 10)
        data = np.arange(11, dtype=float)
        mh = EnergyObsInfo(
            irrep="A1g",
            psq=0,
            energy_type="delab",
            level_index=0,
            particles=[Particle("pi", psq=0), Particle("pi", psq=1)],
        )
        sh = SHEnergyObsInfo(irrep="A1g", psq=0, energy_type="elab", particle="pi")
        samps = [
            SigmondSampling(data.copy(), mh, si, False),
            SigmondSampling(data.copy(), sh, si, False),
        ]
        out = tmp_path / "energy_attrs_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), samps, group="samplings", overwrite=True
        )
        return out

    @pytest.mark.parametrize("lazy", [False, True])
    def test_energy_types_and_ni_pairs_survive(self, tmp_path, lazy):
        from sigmondsamplings.energy_levels import (
            EnergyObsInfo,
            Particle,
            SHEnergyObsInfo,
        )

        out = self._write(tmp_path)
        by_name = _by_name(SigmondLoader(str(out), lazy=lazy).observables)

        mh = by_name["PSQ0_A1g_delab_0"].observable_info
        assert isinstance(mh, EnergyObsInfo) and not isinstance(mh, SHEnergyObsInfo)
        assert (mh.irrep, mh.psq, mh.energy_type, mh.level_index) == ("A1g", 0, "delab", 0)
        assert mh.particles == (Particle("pi", psq=0), Particle("pi", psq=1))

        sh = by_name["PSQ0_pi"].observable_info
        assert isinstance(sh, SHEnergyObsInfo)
        assert sh.particle == "pi" and sh.psq == 0

    def test_plain_observable_passes_through(self, tmp_path):
        # Untagged datasets must not be promoted to energy types.
        from sigmondsamplings.sampling import ObservableInfo, SamplingInfo

        si = SamplingInfo("jackknife", 10)
        plain = ObservableInfo("myop", 0, "n", "re")
        out = tmp_path / "plain_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), [SigmondSampling(np.arange(11.0), plain, si, False)],
            group="samplings", overwrite=True,
        )
        oi = list(SigmondLoader(str(out)).observables)[0].observable_info
        assert type(oi) is ObservableInfo

    @pytest.mark.parametrize("lazy", [False, True])
    def test_plain_observable_latex_override_roundtrips(self, tmp_path, lazy):
        from sigmondsamplings.sampling import ObservableInfo, SamplingInfo

        si = SamplingInfo("jackknife", 10)
        plain = ObservableInfo("myop", 0, "n", "re", latex_str=r"\mathcal{O}")
        out = tmp_path / "plain_latex_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), [SigmondSampling(np.arange(11.0), plain, si, False)],
            group="samplings", overwrite=True,
        )

        oi = list(SigmondLoader(str(out), lazy=lazy).observables)[0].observable_info
        assert type(oi) is ObservableInfo
        assert oi._latex_str == r"\mathcal{O}"
        assert oi.latex_str == r"\mathcal{O}"

    def test_plain_observable_default_latex_is_not_persisted(self, tmp_path):
        from sigmondsamplings.sampling import ObservableInfo, SamplingInfo

        si = SamplingInfo("jackknife", 10)
        plain = ObservableInfo("myop", 0, "n", "re")
        out = tmp_path / "plain_default_latex_rt.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), [SigmondSampling(np.arange(11.0), plain, si, False)],
            group="samplings", overwrite=True,
        )

        oi = list(SigmondLoader(str(out)).observables)[0].observable_info
        assert oi._latex_str is None
        assert oi.latex_str == r"\text{myop}"

    def test_obs_kind_dispatch_accepts_the_short_alias(self, tmp_path):
        # ``energy_sh`` is registered as an alias, so a file written with it (by hand
        # or by an older/other tool) still rebuilds the single hadron class.
        from sigmondsamplings.energy_levels import SHEnergyObsInfo
        from sigmondsamplings.io import obsmeta

        out = self._write(tmp_path)
        with h5py.File(str(out), "r+") as handle:
            group = handle["samplings"]
            meta = obsmeta.read(group)
            for fields in meta.values():
                if fields.get("obs_kind") == "energy_single_hadron":
                    fields["obs_kind"] = "energy_sh"
            obsmeta.write(group, meta)

        by_name = _by_name(SigmondLoader(str(out)).observables)
        assert isinstance(by_name["PSQ0_pi"].observable_info, SHEnergyObsInfo)

    def test_unknown_obs_kind_falls_back_to_the_parsed_type(self, tmp_path):
        # A tag from a newer writer must not abort the read.
        from sigmondsamplings.io import obsmeta

        out = self._write(tmp_path)
        with h5py.File(str(out), "r+") as handle:
            group = handle["samplings"]
            meta = obsmeta.read(group)
            for fields in meta.values():
                if "obs_kind" in fields:
                    fields["obs_kind"] = "energy_from_the_future"
            obsmeta.write(group, meta)

        loaded = _by_name(SigmondLoader(str(out)).observables)
        assert set(loaded) == {"PSQ0_A1g_delab_0", "PSQ0_pi"}


class TestBinsRoundTrip:
    def test_corrt_bins_roundtrip(self, tmp_path):
        original = list(SigmondLoader(str(BINS_HDF5)).observables)
        out = tmp_path / "bins_rt.hdf5"
        SigmondWriter(create_backups=False).write_bins_hdf5(
            str(out), original, group="bins", overwrite=True
        )
        ld = SigmondLoader(str(out))
        assert ld.file_kind == "bins"
        reloaded = list(ld.observables)
        assert all(isinstance(o, SigmondBins) and o.is_complex for o in reloaded)
        _assert_samplings_roundtrip(original, reloaded)


class TestConvertFormat:
    def test_convert_bins_hdf5_preserves_kind(self, tmp_path):
        out = tmp_path / "bins_converted.hdf5"
        SigmondWriter(create_backups=False).convert_format(
            str(BINS_HDF5), str(out), group="/dest/", overwrite=True
        )
        ld = SigmondLoader(str(out), group="dest")
        assert ld.file_kind == "bins"
        _assert_samplings_roundtrip(
            list(SigmondLoader(str(BINS_HDF5)).observables), list(ld.observables)
        )

    @needs_sigmond_query
    def test_convert_fstream_smp_to_hdf5(self, tmp_path):
        out = tmp_path / "pion_converted.hdf5"
        SigmondWriter(create_backups=False).convert_format(
            str(PION_SMP), str(out), group="/samplings/", overwrite=True
        )
        ld = SigmondLoader(str(out), group="samplings")
        assert ld.file_kind == "samplings"
        _assert_samplings_roundtrip(
            list(SigmondLoader(str(PION_SMP)).observables), list(ld.observables)
        )


# ──────────────────────────────────────────────────────────────────────────────
# In-place mutation (append / modify)
# ──────────────────────────────────────────────────────────────────────────────


def _clone_with_name(sampling: SigmondSampling, new_name: str) -> SigmondSampling:
    oi = sampling.observable_info
    from sigmondsamplings.info import ObservableInfo

    new_info = ObservableInfo(new_name, oi.index, oi.op_type, oi.re_im, oi.ensemble_info)
    return SigmondSampling(
        sampling.data.copy(), new_info, sampling.sampling_info, is_complex=sampling.is_complex
    )


class TestWriteMode:
    @staticmethod
    def _sampling(name: str, value: float = 1.0) -> SigmondSampling:
        from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo

        ensemble = EnsembleInfo("append-test", num_measurements=4, num_bins=4)
        return SigmondSampling(
            np.full(5, value),
            ObservableInfo(name, ensemble_info=ensemble),
            SamplingInfo("jackknife", 4),
        )

    @staticmethod
    def _bins(name: str) -> SigmondBins:
        from sigmondsamplings.info import EnsembleInfo, ObservableInfo

        ensemble = EnsembleInfo("append-test", num_measurements=4, num_bins=4)
        return SigmondBins(
            np.arange(4.0), ObservableInfo(name, ensemble_info=ensemble)
        )

    def test_collection_append_preserves_existing_observables(self, tmp_path):
        f = tmp_path / "append.hdf5"
        original = self._sampling("original")
        appended = self._sampling("appended")
        ObservableCollection([original]).to_hdf5(str(f), group="samplings")

        ObservableCollection([appended]).to_hdf5(
            str(f), group="samplings", mode="a"
        )

        reloaded = SigmondLoader(str(f), group="samplings").observables
        assert len(reloaded) == 2
        assert reloaded.find(name="original") is not None
        assert reloaded.find(name="appended") is not None

    def test_append_can_create_a_new_group(self, tmp_path):
        f = tmp_path / "new_group.hdf5"
        donor = self._sampling("shared")
        SigmondWriter(create_backups=False).write_hdf5(
            str(f), [donor], group="samplings"
        )

        SigmondWriter(create_backups=False).write_hdf5(
            str(f), [donor], group="other", mode="a"
        )

        assert len(SigmondLoader(str(f), group="samplings").observables) == 1
        assert len(SigmondLoader(str(f), group="other").observables) == 1

    def test_append_collision_uses_overwrite_parameter(self, tmp_path):
        f = tmp_path / "collision.hdf5"
        existing = self._sampling("collision")
        writer = SigmondWriter(create_backups=False)
        writer.write_hdf5(str(f), [existing], group="samplings")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            writer.write_hdf5(
                str(f), [existing], group="samplings", mode="a"
            )

        replacement = self._sampling("collision", value=7.0)
        writer.write_hdf5(
            str(f), [replacement], group="samplings", mode="a", overwrite=True
        )
        reloaded = SigmondLoader(str(f), group="samplings").observables.find(
            name=existing.observable_info.name, index=existing.observable_info.index
        )
        np.testing.assert_allclose(reloaded.data, replacement.data)

    def test_overwrite_removes_stale_complex_component(self, tmp_path):
        f = tmp_path / "complex_to_real.hdf5"
        original = self._sampling("representation_change")
        original.data = original.data.astype(complex) + 3.0j
        original.is_complex = True
        replacement = self._sampling("representation_change", value=7.0)
        writer = SigmondWriter(create_backups=False)
        writer.write_hdf5(str(f), [original], group="samplings")

        writer.write_hdf5(
            str(f), [replacement], group="samplings", mode="a", overwrite=True
        )

        reloaded = SigmondLoader(str(f), group="samplings").observables[0]
        assert not reloaded.is_complex
        np.testing.assert_allclose(reloaded.data, replacement.data)

    def test_duplicate_logical_observable_rejected_before_write(self, tmp_path):
        f = tmp_path / "duplicates.hdf5"
        duplicate_a = self._sampling("duplicate", value=1.0)
        duplicate_b = self._sampling("duplicate", value=2.0)

        with pytest.raises(ValueError, match="Duplicate logical observable"):
            SigmondWriter(create_backups=False).write_hdf5(
                str(f), [duplicate_a, duplicate_b]
            )

        assert not f.exists()

    def test_bins_support_append_mode(self, tmp_path):
        f = tmp_path / "bins_append.hdf5"
        original = self._bins("original_bins")
        appended = self._bins("appended_bins")
        ObservableCollection([original]).to_hdf5(str(f), group="bins")
        ObservableCollection([appended]).to_hdf5(str(f), group="bins", mode="a")

        reloaded = SigmondLoader(str(f), group="bins").observables
        assert len(reloaded) == 2
        assert reloaded.find(name="appended_bins") is not None

    def test_legacy_append_api_uses_shared_write_engine(self, tmp_path):
        f = tmp_path / "legacy_append.hdf5"
        writer = SigmondWriter(create_backups=False)
        writer.write_hdf5(str(f), [self._sampling("original")])

        writer.append_to_file(
            str(f), [self._sampling("appended")], overwrite=False, group="data"
        )

        reloaded = SigmondLoader(str(f), group="data").observables
        assert len(reloaded) == 2
        assert reloaded.find(name="appended") is not None

    def test_modify_observable_uses_shared_write_engine(self, tmp_path):
        f = tmp_path / "modify.hdf5"
        writer = SigmondWriter(create_backups=False)
        writer.write_hdf5(str(f), [self._sampling("modified")])

        writer.modify_observable(
            str(f), "modified", 0, np.full(5, 9.0), group="data"
        )

        reloaded = SigmondLoader(str(f), group="data").observables[0]
        np.testing.assert_allclose(reloaded.data, 9.0)

    def test_multi_ensemble_collection_writes_separate_groups(self, tmp_path):
        from sigmondsamplings.ensemble_collection import MultiEnsembleCollection
        from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo

        sampling_info = SamplingInfo("jackknife", 4)
        samplings = [
            SigmondSampling(
                np.arange(5.0),
                ObservableInfo(
                    f"obs_{name}",
                    ensemble_info=EnsembleInfo(name, num_measurements=4, num_bins=4),
                ),
                sampling_info,
            )
            for name in ("a", "b")
        ]
        out = tmp_path / "multi_ensemble.hdf5"

        MultiEnsembleCollection(samplings).to_hdf5(str(out), group="data")

        assert len(SigmondLoader(str(out), group="data_a").observables) == 1
        assert len(SigmondLoader(str(out), group="data_b").observables) == 1

    def test_multi_ensemble_append_preflights_all_groups(self, tmp_path):
        from sigmondsamplings.ensemble_collection import MultiEnsembleCollection
        from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo

        sampling_info = SamplingInfo("jackknife", 4)

        def sampling(ensemble_name: str, observable_name: str) -> SigmondSampling:
            return SigmondSampling(
                np.arange(5.0),
                ObservableInfo(
                    observable_name,
                    ensemble_info=EnsembleInfo(
                        ensemble_name, num_measurements=4, num_bins=4
                    ),
                ),
                sampling_info,
            )

        out = tmp_path / "multi_preflight.hdf5"
        original = MultiEnsembleCollection(
            [sampling("a", "original_a"), sampling("b", "original_b")]
        )
        original.to_hdf5(str(out), group="data", create_backups=False)

        update = MultiEnsembleCollection(
            [sampling("a", "new_a"), sampling("b", "original_b")]
        )
        with pytest.raises(FileExistsError):
            update.to_hdf5(
                str(out), group="data", mode="a", create_backups=False
            )

        assert (
            SigmondLoader(str(out), group="data_a").observables.find(name="new_a")
            is None
        )

    def test_multi_ensemble_append_creates_one_backup(self, tmp_path):
        from sigmondsamplings.ensemble_collection import MultiEnsembleCollection
        from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo

        sampling_info = SamplingInfo("jackknife", 4)

        def sampling(ensemble_name: str, observable_name: str) -> SigmondSampling:
            ensemble = EnsembleInfo(ensemble_name, num_measurements=4, num_bins=4)
            return SigmondSampling(
                np.arange(5.0),
                ObservableInfo(observable_name, ensemble_info=ensemble),
                sampling_info,
            )

        out = tmp_path / "multi_backup.hdf5"
        MultiEnsembleCollection(
            [sampling("a", "original_a"), sampling("b", "original_b")]
        ).to_hdf5(str(out), group="data", create_backups=False)

        MultiEnsembleCollection(
            [sampling("a", "new_a"), sampling("b", "new_b")]
        ).to_hdf5(str(out), group="data", mode="a", create_backups=True)

        assert Path(f"{out}.backup_001").exists()
        assert not Path(f"{out}.backup_002").exists()


class TestAppendInPlace:
    def test_append_hdf5_in_place(self, tmp_path):
        f = _copy_to(tmp_path, ENERGY_HDF5)
        n_before = len(SigmondLoader(str(f)).observables)

        donor = list(SigmondLoader(str(f)).observables)[0]
        new_obs = _clone_with_name(donor, "appended_test_obs")

        out = SigmondWriter(create_backups=False).append_to_file(str(f), [new_obs])
        assert Path(out) == f  # HDF5 is appended in place

        reloaded = SigmondLoader(str(f)).observables
        assert len(reloaded) == n_before + 1
        assert "appended_test_obs" in _by_name(reloaded)

    def test_append_complex_corrt_in_place(self, tmp_path):
        # Appending a complex CorrT obs must also keep its key intact.
        f = _copy_to(tmp_path, CORR_HDF5)
        donor = list(SigmondLoader(str(f)).observables)[0]
        new_obs = _clone_with_name(
            donor, "<CorrT>GI{test A} GI{test A} time=99 HermMat</CorrT>"
        )
        SigmondWriter(create_backups=False).append_to_file(str(f), [new_obs])

        match = _by_name(SigmondLoader(str(f)).observables).get(new_obs.observable_info.name)
        assert match is not None
        assert match.is_complex
        np.testing.assert_allclose(match.data, new_obs.data, rtol=1e-12)

    def test_append_existing_without_overwrite_raises(self, tmp_path):
        f = _copy_to(tmp_path, ENERGY_HDF5)
        existing = list(SigmondLoader(str(f)).observables)[0]
        with pytest.raises(FileExistsError):
            SigmondWriter(create_backups=False).append_to_file(
                str(f), [existing], overwrite=False
            )


class TestModifyInPlace:
    def test_modify_observable_in_place(self, tmp_path):
        f = _copy_to(tmp_path, ENERGY_HDF5)
        target = list(SigmondLoader(str(f)).observables)[0]
        name, index = target.observable_info.name, target.observable_info.index

        new_data = np.full(len(target.data), 3.14159)
        out = SigmondWriter(create_backups=False).modify_observable(str(f), name, index, new_data)
        assert Path(out) == f

        modified = SigmondLoader(str(f)).observables.find(name=name, index=index)
        np.testing.assert_allclose(modified.data, new_data, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Multiple root groups in one HDF5 file
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiRootGroupHDF5:
    def test_no_group_with_multiple_groups_raises(self, multi_path_hdf5):
        with pytest.raises(ValueError, match="Multiple groups found"):
            SigmondLoader(str(multi_path_hdf5))

    def test_unknown_group_raises_and_lists_available(self, multi_path_hdf5):
        with pytest.raises(ValueError, match="not found"):
            SigmondLoader(str(multi_path_hdf5), group="does_not_exist")

    def test_each_group_loads_its_own_group(self, multi_path_hdf5):
        corr = SigmondLoader(str(multi_path_hdf5), group=CORR_PATH)
        levels = SigmondLoader(str(multi_path_hdf5), group=LEVELS_PATH)

        assert corr.group == CORR_PATH
        assert levels.group == LEVELS_PATH
        assert len(corr.observables) == 6
        assert len(levels.observables) == 54
        # Groups are independent: complex CorrT in one, real scalars in the other.
        assert all(o.is_complex for o in corr.observables)
        assert not any(o.is_complex for o in levels.observables)

    def test_group_load_matches_standalone_fixture(self, multi_path_hdf5):
        from_multi = SigmondLoader(str(multi_path_hdf5), group=CORR_PATH).observables
        standalone = SigmondLoader(str(CORR_HDF5)).observables
        _assert_samplings_roundtrip(list(standalone), list(from_multi))

    def test_append_targets_one_group_only(self, multi_path_hdf5):
        donor = list(SigmondLoader(str(multi_path_hdf5), group=LEVELS_PATH).observables)[0]
        new_obs = _clone_with_name(donor, "added_to_levels")

        out = SigmondWriter(create_backups=False).append_to_file(
            str(multi_path_hdf5), [new_obs], group=LEVELS_PATH
        )
        assert Path(out) == multi_path_hdf5  # appended in place

        levels = SigmondLoader(str(multi_path_hdf5), group=LEVELS_PATH).observables
        corr = SigmondLoader(str(multi_path_hdf5), group=CORR_PATH).observables
        assert len(levels) == 55  # the targeted group grew
        assert "added_to_levels" in _by_name(levels)
        assert len(corr) == 6  # the other group is untouched

    def test_modify_targets_one_group_only(self, multi_path_hdf5):
        target = list(SigmondLoader(str(multi_path_hdf5), group=LEVELS_PATH).observables)[0]
        name, index = target.observable_info.name, target.observable_info.index
        new_data = np.full(len(target.data), 2.71828)

        SigmondWriter(create_backups=False).modify_observable(
            str(multi_path_hdf5), name, index, new_data, group=LEVELS_PATH
        )

        levels = SigmondLoader(str(multi_path_hdf5), group=LEVELS_PATH).observables
        corr = SigmondLoader(str(multi_path_hdf5), group=CORR_PATH).observables
        np.testing.assert_allclose(levels.find(name=name, index=index).data, new_data, rtol=1e-12)
        assert len(levels) == 54 and len(corr) == 6  # no group gained/lost observables


# ──────────────────────────────────────────────────────────────────────────────
# Nested root-group paths (the Sigmond spec's [/a/b] form) and default path
# ──────────────────────────────────────────────────────────────────────────────


class TestNestedRootPath:
    NESTED = "isotriplet/P0A1g"

    def test_write_then_autodetect_nested_path(self, tmp_path):
        original = list(SigmondLoader(str(ENERGY_HDF5)).observables)
        out = tmp_path / "nested.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), original, group=self.NESTED, overwrite=True
        )

        # A single nested root group is auto-detected (no group needed).
        ld = SigmondLoader(str(out))
        assert ld.group == self.NESTED
        _assert_samplings_roundtrip(original, list(ld.observables))

    def test_explicit_nested_path_loads(self, tmp_path):
        original = list(SigmondLoader(str(ENERGY_HDF5)).observables)
        out = tmp_path / "nested.hdf5"
        SigmondWriter(create_backups=False).write_hdf5(
            str(out), original, group=f"/{self.NESTED}/", overwrite=True
        )
        ld = SigmondLoader(str(out), group=self.NESTED)
        _assert_samplings_roundtrip(original, list(ld.observables))


class TestDefaultGroup:
    def test_write_file_uses_default_group(self, tmp_path):
        original = list(SigmondLoader(str(ENERGY_HDF5)).observables)
        out = SigmondWriter(create_backups=False).write_file(
            str(tmp_path / "out.hdf5"), original, overwrite=True
        )
        assert SigmondLoader(str(out)).group == DEFAULT_GROUP
