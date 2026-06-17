"""
Tests for the I/O operation functions in sigmondsamplings/io/ (convert, combine,
energy_tag) that back the ``ss`` CLI write commands.

These exercise the library-level functions directly against the fixtures in
tests/data/. fstream inputs are gated on sigmond_query.

Run with:  python -m pytest tests/test_scripts.py -v
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.io.combine import (
    combine_files,
    load_all_samplings,
    resolve_paths,
    validate_compatibility,
)
from sigmondsamplings.io.convert import convert_to_hdf5
from sigmondsamplings.io.energy_tag import add_energy_attrs
from sigmondsamplings.io.loader import SigmondLoader
from sigmondsamplings.sampling import SigmondSampling

DATA_DIR = Path(__file__).parent / "data"
CORR_HDF5 = DATA_DIR / "corr_matrix_samplings.hdf5"
ENERGY_HDF5 = DATA_DIR / "energy_levels_samplings.hdf5"
PION_SMP = DATA_DIR / "energy_samplings_pion.smp"
BINS_HDF5 = DATA_DIR / "tetraquark_bins.hdf5"

needs_sigmond_query = pytest.mark.skipif(
    shutil.which("sigmond_query") is None, reason="sigmond_query not available"
)

BOOT = SamplingInfo("bootstrap", 1000, 6754, 127)
ENS = EnsembleInfo("clover_s32_t256_ud860_s743", 412, 412)


def _sampling(name: str, ens: EnsembleInfo = ENS, samp: SamplingInfo = BOOT) -> SigmondSampling:
    obs = ObservableInfo(name, 0, "n", "re", ens)
    return SigmondSampling(np.arange(5, dtype=float), obs, samp)


# ──────────────────────────────────────────────────────────────────────────────
# io.convert.convert_to_hdf5
# ──────────────────────────────────────────────────────────────────────────────


class TestConvertToHDF5:
    def test_repack_hdf5_samplings(self, tmp_path):
        out = tmp_path / "repacked.hdf5"
        # Passing in_group skips the multi-group guard (which would need sigmond_query).
        convert_to_hdf5(str(ENERGY_HDF5), str(out), in_group="samplings")
        ld = SigmondLoader(str(out))
        assert ld.file_kind == "samplings"
        assert len(ld.observables) == 54

    def test_convert_bins_preserves_kind(self, tmp_path):
        out = tmp_path / "bins_out.hdf5"
        convert_to_hdf5(str(BINS_HDF5), str(out), in_group="bins")
        ld = SigmondLoader(str(out))
        assert ld.file_kind == "bins"
        assert len(ld.observables) == 5

    def test_output_extension_enforced(self, tmp_path):
        out = tmp_path / "no_ext_output"
        result = convert_to_hdf5(str(ENERGY_HDF5), str(out), in_group="samplings")
        assert Path(result).suffix == ".hdf5"

    def test_h5_output_extension_preserved(self, tmp_path):
        out = tmp_path / "repacked.h5"
        result = convert_to_hdf5(str(ENERGY_HDF5), str(out), in_group="samplings")
        assert Path(result) == out
        assert out.exists()

    def test_missing_output_extension_inherits_input_h5_extension(self, tmp_path):
        input_h5 = tmp_path / "input.h5"
        shutil.copy2(ENERGY_HDF5, input_h5)
        out = tmp_path / "repacked"

        result = convert_to_hdf5(str(input_h5), str(out), in_group="samplings")

        assert Path(result) == tmp_path / "repacked.h5"
        assert Path(result).exists()

    @needs_sigmond_query
    def test_convert_fstream_smp(self, tmp_path):
        out = tmp_path / "pion.hdf5"
        convert_to_hdf5(str(PION_SMP), str(out))
        ld = SigmondLoader(str(out))
        assert ld.file_kind == "samplings"
        assert len(ld.observables) == 51


# ──────────────────────────────────────────────────────────────────────────────
# io.combine.resolve_paths
# ──────────────────────────────────────────────────────────────────────────────


class TestResolvePaths:
    def test_absolute_path_kept(self):
        assert resolve_paths([str(ENERGY_HDF5)]) == [str(ENERGY_HDF5.resolve())]

    def test_relative_resolved_against_base(self):
        resolved = resolve_paths([ENERGY_HDF5.name], base_path=str(DATA_DIR))
        assert resolved == [str(ENERGY_HDF5.resolve())]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_paths([str(tmp_path / "does_not_exist.hdf5")])


# ──────────────────────────────────────────────────────────────────────────────
# io.combine.load_all_samplings
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadAllSamplings:
    def test_loads_keyed_by_name_and_index(self):
        loaded = load_all_samplings([str(CORR_HDF5)])
        assert len(loaded) == 6
        assert all(" " in key and key.rsplit(" ", 1)[1].isdigit() for key in loaded)

    def test_duplicate_keys_overwrite_and_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            loaded = load_all_samplings([str(ENERGY_HDF5), str(ENERGY_HDF5)])
        assert len(loaded) == 54  # second load overwrites, no growth
        assert "conflict" in caplog.text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# io.combine.validate_compatibility
# ──────────────────────────────────────────────────────────────────────────────


class TestValidateCompatibility:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No samplings"):
            validate_compatibility({})

    def test_compatible_passes(self):
        validate_compatibility({"a 0": _sampling("a"), "b 0": _sampling("b")})  # no raise

    def test_incompatible_sampling_info_raises(self):
        other = _sampling("b", samp=SamplingInfo("jackknife", 412))
        with pytest.raises(ValueError, match="Incompatible sampling information"):
            validate_compatibility({"a 0": _sampling("a"), "b 0": other})

    def test_multiple_ensembles_allowed(self):
        ens_b = EnsembleInfo("other_ensemble", 200, 200)
        samples = {"a 0": _sampling("a"), "b 0": _sampling("b", ens=ens_b)}
        validate_compatibility(samples)  # different ensembles are allowed -> no raise


# ──────────────────────────────────────────────────────────────────────────────
# io.combine.combine_files
# ──────────────────────────────────────────────────────────────────────────────


class TestCombineFiles:
    def test_combine_two_files_unions_observables(self, tmp_path):
        out = tmp_path / "combined.hdf5"
        result = combine_files([str(CORR_HDF5), str(ENERGY_HDF5)], str(out), overwrite=True)
        reloaded = SigmondLoader(str(result)).observables
        assert len(reloaded) == 6 + 54

    def test_non_hdf5_output_extension_adjusted(self, tmp_path):
        out = tmp_path / "combined.dat"
        result = combine_files([str(ENERGY_HDF5)], str(out), overwrite=True)
        assert Path(result).suffix == ".hdf5"

    def test_h5_output_extension_preserved(self, tmp_path):
        out = tmp_path / "combined.h5"
        result = combine_files([str(ENERGY_HDF5)], str(out), overwrite=True)
        assert Path(result) == out
        assert out.exists()

    def test_existing_output_without_overwrite_raises(self, tmp_path):
        out = tmp_path / "combined.hdf5"
        out.write_text("placeholder")
        with pytest.raises(FileExistsError):
            combine_files([str(ENERGY_HDF5)], str(out), overwrite=False)


# ──────────────────────────────────────────────────────────────────────────────
# io.energy_tag.add_energy_attrs
# ──────────────────────────────────────────────────────────────────────────────


class TestAddEnergyAttrs:
    def test_h5_output_extension_preserved(self, tmp_path):
        out = tmp_path / "energy_attrs.h5"
        result = add_energy_attrs(str(ENERGY_HDF5), str(out), in_group="samplings")
        assert Path(result) == out
        assert out.exists()

    def test_missing_output_extension_inherits_input_h5_extension(self, tmp_path):
        input_h5 = tmp_path / "energy_input.h5"
        shutil.copy2(ENERGY_HDF5, input_h5)
        out = tmp_path / "energy_attrs"

        result = add_energy_attrs(str(input_h5), str(out), in_group="samplings")

        assert Path(result) == tmp_path / "energy_attrs.h5"
        assert Path(result).exists()
