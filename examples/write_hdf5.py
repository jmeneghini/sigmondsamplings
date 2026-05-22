"""Write a subset of loaded samplings to a new Sigmond HDF5 file."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sigmondsamplings import SigmondLoader, SigmondWriter


DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "energy_levels_samplings.hdf5"


def main() -> None:
    loader = SigmondLoader(str(DATA))
    subset = loader.observables[:5]

    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "energy_subset.hdf5"

        writer = SigmondWriter(create_backups=False)
        writer.write_hdf5(str(out), subset, root_path="samplings", overwrite=True)

        reloaded = SigmondLoader(str(out), hdf5_path="samplings")

        print(f"wrote: {out}")
        print(f"n original: {len(subset)}")
        print(f"n reloaded: {len(reloaded.observables)}")
        print(f"first reloaded: {reloaded.observables[0].observable_info.name}")


if __name__ == "__main__":
    main()
