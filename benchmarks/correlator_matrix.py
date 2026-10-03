"""
Benchmark the correlator layer on a synthetic lazy HDF5 file.

Measures the lazy load (with SS's ``ObsMeta`` table and without it, as in
files written by Sigmond itself), the metadata-only index build, a one-time
request ``index.matrix(ops, times=[t])`` and full ``T x N x N x S`` assembly.

    python benchmarks/correlator_matrix.py --ops 12 --times 48 --samples 201
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from sigmondsamplings import SigmondLoader, SigmondSampling, SigmondWriter
from sigmondsamplings.correlators import (
    BasicLapHOperator,
    CorrelatorAtTimeInfo,
    CorrelatorIndex,
    CorrelatorInfo,
)
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo


def build_file(path: Path, n_ops: int, n_times: int, n_samples: int) -> tuple:
    ens = EnsembleInfo("bench", n_samples - 1)
    info = SamplingInfo("bootstrap", n_samples - 1)
    ops = tuple(
        BasicLapHOperator.single("pion", (0, 0, 0), "A1um", 1, "SD", k) for k in range(n_ops)
    )
    rng = np.random.default_rng(0)
    items = []
    for t in range(n_times):
        for a in ops:
            for b in ops:
                key = CorrelatorAtTimeInfo(CorrelatorInfo(a, b), t, hermitian=True)
                data = rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)
                obs = ObservableInfo(key.to_sigmond(), 0, "n", "re", ens)
                items.append(SigmondSampling(data, obs, info, is_complex=True))
    SigmondWriter(create_backups=False).write_hdf5(str(path), items, group="corrs")
    return ops, len(items)


def timed(label: str, fn):
    start = time.perf_counter()
    out = fn()
    print(f"  {label:<38s} {time.perf_counter() - start:8.3f} s")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=12)
    parser.add_argument("--times", type=int, default=48)
    parser.add_argument("--samples", type=int, default=1001)
    args = parser.parse_args()

    import sigmondsamplings.correlators.matrix  # noqa: F401  (keep xarray's import out of timings)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.hdf5"
        ops, n_obs = timed(
            "write synthetic file (setup)",
            lambda: build_file(path, args.ops, args.times, args.samples),
        )
        n_datasets = 2 * n_obs
        print(
            f"  {args.ops} ops x {args.times} times x {args.samples} samples: "
            f"{n_obs} complex correlators, {n_datasets} datasets, "
            f"{path.stat().st_size / 2**20:.0f} MiB"
        )
        loader = timed(
            "lazy load (dataset names only)", lambda: SigmondLoader(str(path), lazy=True)
        )
        plain = Path(tmp) / "sigmond_layout.hdf5"
        shutil.copy(path, plain)
        with h5py.File(plain, "a") as f:
            del f["corrs/ObsMeta"]
        timed(
            "lazy load, no ObsMeta (Sigmond layout)", lambda: SigmondLoader(str(plain), lazy=True)
        )
        index = timed(
            "index build (metadata only)",
            lambda: CorrelatorIndex.from_collection(loader.observables),
        )
        assert not any(o.is_materialized for o in loader.observables)
        t_mid = args.times // 2
        timed(f"one time: matrix(times=[{t_mid}])", lambda: index.matrix(ops, times=[t_mid]))
        fresh = CorrelatorIndex.from_collection(SigmondLoader(str(path), lazy=True).observables)
        m = timed("full matrix, lazy reads (all times)", lambda: fresh.matrix(ops))
        timed("full matrix, data already in memory", lambda: fresh.matrix(ops))
        coefficients = np.eye(args.ops, dtype=complex) + 0.1j
        timed("square complex basis transform", lambda: m.transform(coefficients))

        from sigmondsamplings.correlators import effective_energy
        from sigmondsamplings.data import SigmondData

        sparse = SigmondData.from_data(
            np.broadcast_to(np.array([4.0, 2.0, 1.0])[:, None], (3, args.samples)),
            "sparse",
            m.sampling_info,
            dim_names=("time",),
            coords={"time": [1, 10**9, 10**9 + 1]},
            ensemble_info=m.values.ensemble_info,
        )
        timed("sparse forward energy (span 1e9)", lambda: effective_energy(sparse))

        import slat.plotting as slp
        from sigmondsamplings.plotting import CorrelatorMatrixPlotter

        for i in range(args.ops):
            m.values.data[t_mid, i, i, 0] = abs(m.values.data[t_mid, i, i, 0]) + 1
        with slp.record():
            timed(
                "one-time normalized heatmap",
                lambda: CorrelatorMatrixPlotter(m).plot_heatmap(t_mid),
            )

        size = m.values.data.nbytes / 2**20
        print(f"  assembled buffer {m.values.data.shape}: {size:.0f} MiB")


if __name__ == "__main__":
    main()
