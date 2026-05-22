"""Load a Sigmond HDF5 samplings file and compute simple statistics."""

from __future__ import annotations

from pathlib import Path

from sigmondsamplings import SamplingStats, SigmondLoader


DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "energy_levels_samplings.hdf5"


def main() -> None:
    loader = SigmondLoader(str(DATA))
    observables = loader.observables

    print(f"file kind: {loader.file_kind}")
    print(f"hdf5 path: {loader.hdf5_path}")
    print(f"n observables: {len(observables)}")

    first = observables[0]
    print("\nfirst observable")
    print(f"  name:  {first.observable_info.name}")
    print(f"  full:  {first.full_sample_value:.8g}")
    print(f"  mean:  {first.mean:.8g}")
    print(f"  error: {first.error:.8g}")

    stats = SamplingStats(observables[:5])
    print("\nfirst 5 covariance matrix")
    print(stats.cov_matrix)

    energy = observables.find(name="K(0)_elab")
    if energy is not None:
        print("\nqueried observable")
        print(f"  {energy.observable_info.name}: {energy.pdg_format()}")


if __name__ == "__main__":
    main()
