"""
Tests for lazy (deferred-read) HDF5 loading via ``SigmondLoader(..., lazy=True)``.

The central guarantees:
  - the index phase (construction) and every metadata query read *no* sample
    datasets, and
  - touching an observable's data reads exactly that observable's dataset(s).

Reads are instrumented by spying on ``h5py.Dataset.__getitem__`` and counting
only datasets whose path contains ``/Values/`` (so Header/Info reads, which are
metadata, are ignored).

Run with:  python -m pytest tests/test_lazy_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.energy_level_collection import SingleEnsembleEnergyCollection
from sigmondsamplings.energy_levels import EnergyObsInfo
from sigmondsamplings.lazy import LazySigmondBins, LazySigmondSampling
from sigmondsamplings.loader import SigmondLoader
from sigmondsamplings.sampling import SigmondSampling

DATA_DIR = Path(__file__).parent / "data"
CORR_HDF5 = DATA_DIR / "corr_matrix_samplings.hdf5"        # complex samplings (Re/Im fused)
ENERGY_HDF5 = DATA_DIR / "energy_levels_samplings.hdf5"    # real scalar samplings
BINS_HDF5 = DATA_DIR / "tetraquark_bins.hdf5"              # complex bins
PION_SMP = DATA_DIR / "energy_samplings_pion.smp"          # fstream


@pytest.fixture
def value_reads(monkeypatch):
    """Spy that records the names of every ``/Values/`` dataset read."""
    reads: list[str] = []
    original = h5py.Dataset.__getitem__

    def spy(self, key):
        if "/Values/" in self.name:
            reads.append(self.name)
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    return reads


# ---------------------------------------------------------------------------
# Index phase / metadata queries are read-free
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ENERGY_HDF5, CORR_HDF5, BINS_HDF5])
def test_construction_reads_no_datasets(path, value_reads):
    SigmondLoader(str(path), lazy=True)
    assert value_reads == []


@pytest.mark.parametrize("path", [ENERGY_HDF5, CORR_HDF5, BINS_HDF5])
def test_queries_read_no_datasets(path, value_reads):
    coll = SigmondLoader(str(path), lazy=True).observables
    value_reads.clear()

    name = coll.obs.name[0]
    coll.filter(name=name)
    coll.find(index=0)
    coll.sort("name")
    coll.group_by("name")
    coll.unique("name")
    _ = len(coll)
    _ = set(coll)  # hashing
    _ = coll[0] == coll[-1]  # equality
    _ = [s.is_complex for s in coll]

    assert value_reads == []


def test_bins_num_bins_and_len_are_read_free(value_reads):
    """The criticism's headline case: bins hash/len/num_bins must not materialize."""
    coll = SigmondLoader(str(BINS_HDF5), lazy=True).observables
    value_reads.clear()

    for s in coll:
        assert isinstance(s, LazySigmondBins)
        assert s.num_bins > 0
        assert len(s) == s.num_bins
        assert not s.is_materialized
    _ = set(coll)  # __hash__ routes through num_bins

    assert value_reads == []


# ---------------------------------------------------------------------------
# Materialization reads exactly the right dataset(s)
# ---------------------------------------------------------------------------


def test_real_observable_reads_single_dataset(value_reads):
    coll = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables
    s = coll[0]
    assert not s.is_materialized
    value_reads.clear()

    _ = s.mean

    assert len(value_reads) == 1
    assert s.is_materialized


def test_complex_observable_reads_re_and_im(value_reads):
    coll = SigmondLoader(str(CORR_HDF5), lazy=True).observables
    s = coll[0]
    assert s.is_complex
    value_reads.clear()

    _ = s.full_sample_value

    # one Re dataset + one Im dataset fused into the complex array
    assert len(value_reads) == 2
    assert any("Re" in n for n in value_reads)
    assert any("Im" in n for n in value_reads)


def test_data_is_cached_after_first_read(value_reads):
    s = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables[0]
    s.materialize()
    value_reads.clear()

    _ = s.data
    _ = s.mean
    _ = s.error

    assert value_reads == []  # served from cache


# ---------------------------------------------------------------------------
# Lazy and eager loads agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ENERGY_HDF5, CORR_HDF5, BINS_HDF5])
def test_lazy_matches_eager_values(path):
    eager = SigmondLoader(str(path)).observables.sort("name")
    lazy = SigmondLoader(str(path), lazy=True).observables.sort("name")

    assert eager.obs.name == lazy.obs.name
    np.testing.assert_allclose(eager.to_numpy(), lazy.to_numpy())


@pytest.mark.parametrize(
    "path,base",
    [(ENERGY_HDF5, SigmondSampling), (CORR_HDF5, SigmondSampling), (BINS_HDF5, SigmondBins)],
)
def test_lazy_objects_are_instances_of_base(path, base):
    coll = SigmondLoader(str(path), lazy=True).observables
    assert all(isinstance(s, base) for s in coll)


def test_lazy_as_energy_level_is_metadata_only(value_reads):
    s = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables[0]
    value_reads.clear()

    energy = s.as_energy_level()

    assert isinstance(energy, LazySigmondSampling)
    assert isinstance(energy.observable_info, EnergyObsInfo)
    assert energy is not s
    assert not s.is_materialized
    assert not energy.is_materialized
    assert value_reads == []


def test_lazy_collection_as_energy_levels_is_metadata_only(value_reads):
    coll = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables
    value_reads.clear()

    energy = coll.as_energy_levels()

    assert isinstance(energy, SingleEnsembleEnergyCollection)
    assert all(isinstance(s, LazySigmondSampling) for s in energy)
    assert all(isinstance(s.observable_info, EnergyObsInfo) for s in energy)
    assert all(not s.is_materialized for s in energy)
    assert value_reads == []


def test_loader_energy_observables_is_metadata_only(value_reads):
    loader = SigmondLoader(str(ENERGY_HDF5), lazy=True)
    value_reads.clear()

    energy = loader.energy_observables()

    assert isinstance(energy, SingleEnsembleEnergyCollection)
    assert all(isinstance(s, LazySigmondSampling) for s in energy)
    assert all(not s.is_materialized for s in energy)
    assert value_reads == []


def test_lazy_bins_obs_replace_is_metadata_only(value_reads):
    coll = SigmondLoader(str(BINS_HDF5), lazy=True).observables
    value_reads.clear()

    renamed = coll.obs.replace(name=lambda obs: f"{obs.name}_renamed")

    assert len(renamed) == len(coll)
    assert all(isinstance(s, LazySigmondBins) for s in renamed)
    assert all(not s.is_materialized for s in renamed)
    assert value_reads == []


def test_collection_materialize_reads_all_values(value_reads):
    coll = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables
    value_reads.clear()

    returned = coll.materialize()

    assert returned is coll
    assert all(s.is_materialized for s in coll)
    assert len(value_reads) == len(coll)


# ---------------------------------------------------------------------------
# Subclass behavior: repr, copy, arithmetic, round-trip
# ---------------------------------------------------------------------------


def test_repr_is_metadata_safe_until_materialized(value_reads):
    s = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables[0]
    value_reads.clear()
    r = repr(s)
    assert "lazy" in r
    assert value_reads == []
    s.materialize()
    assert "lazy" not in repr(s)


def test_copy_returns_eager_independent_object():
    s = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables[0]
    c = s.copy()
    assert not isinstance(c, (LazySigmondSampling, LazySigmondBins))
    assert isinstance(c, SigmondSampling)
    c.data[0] += 123.0
    assert c.data[0] != s.data[0]  # independent array


def test_arithmetic_materializes_and_matches_eager():
    eager = SigmondLoader(str(ENERGY_HDF5)).observables.sort("name")
    lazy = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables.sort("name")

    e0, e1 = eager[0], eager[1]
    l0, l1 = lazy[0], lazy[1]
    np.testing.assert_allclose((l0 + l1).data, (e0 + e1).data)


def test_lazy_roundtrips_through_hdf5(tmp_path):
    lazy = SigmondLoader(str(ENERGY_HDF5), lazy=True).observables
    out = tmp_path / "roundtrip.hdf5"
    lazy.to_hdf5(str(out), create_backups=False)  # materializes on write

    reloaded = SigmondLoader(str(out)).observables
    eager = SigmondLoader(str(ENERGY_HDF5)).observables
    np.testing.assert_allclose(
        reloaded.sort("name").to_numpy(), eager.sort("name").to_numpy()
    )


# ---------------------------------------------------------------------------
# fstream is explicitly unsupported in phase 1
# ---------------------------------------------------------------------------


def test_lazy_fstream_raises():
    with pytest.raises(NotImplementedError):
        SigmondLoader(str(PION_SMP), lazy=True)
