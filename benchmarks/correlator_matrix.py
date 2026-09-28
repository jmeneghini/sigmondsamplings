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
    BasicOperator,
    CorrelatorIndex,
    CorrelatorKey,
    CorrelatorPair,
)
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo


def build_file(path: Path, n_ops: int, n_times: int, n_samples: int) -> tuple:
    ens = EnsembleInfo("bench", n_samples - 1)
    info = SamplingInfo("bootstrap", n_samples - 1)
    ops = tuple(BasicOperator.single("pion", (0, 0, 0), "A1um", 1, "SD", k) for k in range(n_ops))
    rng = np.random.default_rng(0)
    items = []
    for t in range(n_times):
        for a in ops:
            for b in ops:
                key = CorrelatorKey(CorrelatorPair(a, b), t, hermitian=True)
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
        timed("lazy load, no ObsMeta (Sigmond layout)", lambda: SigmondLoader(str(plain), lazy=True))
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
        size = m.values.data.nbytes / 2**20
        print(f"  assembled buffer {m.values.data.shape}: {size:.0f} MiB")


if __name__ == "__main__":
    main()
