"""Focused coverage for energy-collection reuse and mutation invariants."""

from __future__ import annotations

import numpy as np
import pytest

from sigmondsamplings.energy_level_collection import (
    MultiEnsembleEnergyCollection,
    SingleEnsembleEnergyCollection,
)
from sigmondsamplings.energy_levels import EnergyObsInfo, Particle
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.io.pycalq import read_shift_particles, write_shift_particles
from sigmondsamplings.observable_collection import ObservableCollection
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.spectrum_spec import SectorSpec, SpectrumSpec

SAMPLING_INFO = SamplingInfo("bootstrap", 2, seed=7)


def _sampling(
    *,
    ensemble: str = "ens_A",
    energy_type: str = "elab",
    level_index: int = 0,
    ref_particle: str | None = None,
    particles: list[Particle] | None = None,
    name: str | None = None,
) -> SigmondSampling:
    info = EnergyObsInfo(
        name=name,
        ensemble_info=EnsembleInfo(ensemble, 64),
        irrep="A1g",
        psq=0,
        energy_type=energy_type,
        level_index=level_index,
        ref_particle=ref_particle,
        particles=particles,
    )
    return SigmondSampling(np.array([1.0, 0.9, 1.1]), info, SAMPLING_INFO)


def test_empty_unique_returns_an_empty_collection():
    assert ObservableCollection([]).unique("name") == []


def test_from_collection_canonicalizes_a_metadata_copy():
    source = _sampling(name="legacy_name")

    result = SingleEnsembleEnergyCollection.from_collection([source])

    assert source.observable_info.name == "legacy_name"
    assert result[0].observable_info.name == "PSQ0_A1g_elab_0"
    assert result[0] is not source
    assert result[0].data is source.data


def test_energy_mutators_invalidate_cached_metadata():
    ref_collection = SingleEnsembleEnergyCollection(
        [_sampling(ref_particle="pi", name="PSQ0_A1g_elab_0_ref")]
    )
    assert ref_collection.shared_attr("ref_particle") == "pi"
    ref_collection.set_ref("K")
    assert ref_collection.shared_attr("ref_particle") == "K"

    shift_collection = SingleEnsembleEnergyCollection(
        [_sampling(energy_type="decm", particles=[Particle("pi", psq=0)])]
    )
    original = (Particle("pi", psq=0),)
    replacement = (Particle("K", psq=0), Particle("pi", psq=0))
    assert shift_collection.shared_attr("particles") == original
    shift_collection.set_shift_particles({("A1g", 0, 0): list(replacement)})
    assert shift_collection.shared_attr("particles") == replacement


def test_multi_ensemble_base_hooks_preserve_energy_collection_types():
    collection = MultiEnsembleEnergyCollection(
        [_sampling(ensemble="ens_A"), _sampling(ensemble="ens_B")]
    )

    assert all(
        isinstance(group, SingleEnsembleEnergyCollection)
        for group in collection.by_ensemble.values()
    )
    assert isinstance(collection[:1], MultiEnsembleEnergyCollection)
    assert repr(collection).startswith("MultiEnsembleEnergyCollection(")
    assert repr(next(iter(collection.by_ensemble.values()))).startswith(
        "SingleEnsembleEnergyCollection("
    )

    rebuilt = MultiEnsembleEnergyCollection(collection.by_ensemble)
    assert len(rebuilt) == len(collection)
    assert all(
        isinstance(group, SingleEnsembleEnergyCollection)
        for group in rebuilt.by_ensemble.values()
    )


def test_resolved_spectrum_keys_drive_collection_filtering(tmp_path):
    collection = SingleEnsembleEnergyCollection(
        [_sampling(level_index=0), _sampling(level_index=1)]
    )
    spec = SpectrumSpec(spectrum=[SectorSpec(psq=0, irrep="A1g", levels=[1])])
    path = tmp_path / "spectrum.toml"
    path.write_text(spec.to_toml())

    direct = collection.filter_by_spec([(0, "A1g", [1])])
    from_file = collection.filter_from_toml(str(path))

    assert list(direct.level_indexes) == [1]
    assert list(from_file.level_indexes) == [1]


def test_pycalq_assignments_round_trip_and_filter_sectors(tmp_path):
    path = tmp_path / "levels.yml"
    assignments = {
        ("A1g", 0, 0): [Particle("pi", psq=0), Particle("K", psq=0)],
        ("E", 1, 1): [Particle("pi", psq=1)],
    }

    write_shift_particles(path, assignments)

    assert read_shift_particles(path) == assignments
    assert read_shift_particles(path, allowed_sectors={(0, "A1g")}) == {
        ("A1g", 0, 0): assignments[("A1g", 0, 0)]
    }


def test_pycalq_reader_rejects_invalid_level_shapes(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("non_interacting_levels:\n  A1g PSQ=0: pi\n")

    with pytest.raises(ValueError, match="level list"):
        read_shift_particles(path)


def test_pycalq_writer_includes_qcmsq_assignments(tmp_path):
    collection = SingleEnsembleEnergyCollection(
        [_sampling(energy_type="qcmsq", particles=[Particle("pi", psq=0)])]
    )
    path = tmp_path / "qcmsq.yml"

    collection.create_pycalq_yml_shift_particles(str(path))

    assert read_shift_particles(path) == {("A1g", 0, 0): [Particle("pi", psq=0)]}
