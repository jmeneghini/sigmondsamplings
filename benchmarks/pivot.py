"""Fixed-pivot timings and process peak RSS; targets are not CI assertions.

Run each case in its own process, e.g.:
    OPENBLAS_NUM_THREADS=1 python benchmarks/pivot.py --ops 20 --samples 2001 --rank 8
Add --complex, --bins, --vevs, --improved or --noncontiguous. A full sweep uses
N={4,12,20}, S={201,2001}, T=48, ranks N and max(2,N//3).
"""
from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from sigmondsamplings.correlators import (
    CorrelatorMatrix,
    CorrelatorMatrixInfo,
    GenIrrepOperator,
    SinglePivot,
)
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.sampling_array import _contract


def make_matrix(args):
    rng = np.random.default_rng(271)
    n, samples = args.ops, args.samples
    rank = args.rank or n
    if not 1 <= rank <= n or args.times < 4 or samples < 3:
        raise ValueError('require 1 <= rank <= ops, times >= 4 and samples >= 3')
    ens = EnsembleInfo('pivot_benchmark', samples - 1, spatial_extent=16, temporal_extent=args.times, num_streams=1)
    plan = None if args.bins else SamplingInfo.bootstrap(samples - 1, seed=271)
    ops = tuple(GenIrrepOperator.from_sigmond(f'isosinglet S=0 P=(0,0,0) A1g_1 BENCH {i}') for i in range(n))
    basis = rng.normal(size=(n, n))
    if args.complex:
        basis = basis + 1j * rng.normal(size=(n, n))
    basis, _ = np.linalg.qr(basis)
    energies = np.linspace(.05, .4, rank)
    full = np.einsum('in,tn,jn->tij', basis[:, :rank], np.exp(-np.outer(np.arange(args.times), energies)), basis[:, :rank].conj())
    factor = np.concatenate([[1.], 1 + .01 * rng.normal(size=samples - 1)])
    buffer = full[..., None] * factor
    if args.noncontiguous:
        larger = np.empty(buffer.shape[:-1] + (samples * 2,), dtype=buffer.dtype)
        larger[..., ::2] = buffer
        buffer = larger[..., ::2]
    values = SigmondData.from_data(buffer, 'benchmark', plan, ensemble_info=ens,
                                  dim_names=('time', 'sink', 'source'),
                                  coords={'time': list(range(args.times)), 'sink': [op.coordinate for op in ops],
                                          'source': [op.coordinate for op in ops]}, is_complex=args.complex)
    vevs = None
    if args.vevs:
        v = np.broadcast_to(rng.normal(size=(n, 1)), (n, samples)).copy()
        vevs = SigmondData.from_data(v, 'vevs', plan, ensemble_info=ens, dim_names=('operator',),
                                    coords={'operator': [op.coordinate for op in ops]})
    matrix = CorrelatorMatrix(CorrelatorMatrixInfo(ops, True, args.vevs and not args.bins), values, vevs)
    if args.improved:
        matrix = matrix.transform(np.eye(n)[:, :max(rank, n // 2)])
    return matrix


def measure(function, repeats):
    function()  # warmup
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
        del result
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops', type=int, default=12)
    parser.add_argument('--times', type=int, default=48)
    parser.add_argument('--samples', type=int, default=201)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--complex', action='store_true')
    parser.add_argument('--bins', action='store_true')
    parser.add_argument('--vevs', action='store_true')
    parser.add_argument('--improved', action='store_true')
    parser.add_argument('--noncontiguous', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sigmond', action='store_true', help='Also time the batch oracle including input/output I/O')
    parser.add_argument('--sigmond-timeout', type=float, default=300, help='Seconds allowed per native batch run')
    args = parser.parse_args()
    from threadpoolctl import threadpool_info
    matrix = make_matrix(args)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        def construct():
            return SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-6)
        pivot = construct()
        rotated = pivot.rotate(matrix)
        results = {'configuration': vars(args), 'blas': threadpool_info(), 'seconds': {}}
        results['seconds']['build_including_metric_diagnostics'] = measure(construct, args.repeats)
        coefficients = pivot.transformation
        from sigmondsamplings.correlators import solve_gevp
        from sigmondsamplings.correlators._kernels import rotate_matrix
        full = np.asarray(matrix.values.full_sample_value)
        scale = 1 / np.sqrt(abs(full[1].diagonal().real))
        A, B = [c * scale[:, None] * scale[None, :] for c in (full[3], full[1])]
        results['seconds']['full_sample_solve'] = measure(
            lambda: solve_gevp(A, B, min_inv_condition=1e-6), args.repeats)
        results['seconds']['new_contraction_kernel'] = measure(
            lambda: rotate_matrix(matrix.values.data, coefficients), args.repeats)
        results['seconds']['copy'] = measure(lambda: matrix.values.data.copy(), args.repeats)
        results['seconds']['old_two_contractions'] = measure(
            lambda: _contract(_contract(matrix.values.data, coefficients.conj(), 1), coefficients, 2), args.repeats)
        results['seconds']['rotation_full_matrix'] = measure(lambda: pivot.rotate(matrix), args.repeats)
        results['seconds']['collection_shared_keys'] = measure(
            lambda: rotated.to_collection(diagonal_only_before_time=3, upper_triangle_only=True), args.repeats)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rotated.hdf5'
            results['seconds']['write_shared_keys'] = measure(
                lambda: rotated.write(path, overwrite=True, diagonal_only_before_time=3, upper_triangle_only=True), args.repeats)
            results['seconds']['end_to_end_shared_keys'] = measure(
                lambda: construct().rotate(matrix).write(path, overwrite=True, diagonal_only_before_time=3, upper_triangle_only=True), args.repeats)
            if args.sigmond:
                import sys
                sys.path.insert(0, str(Path(__file__).parents[1] / 'tests'))
                from sigmond_oracle import rotate
                native_timings = []
                def native():
                    timing = {}
                    result = rotate(matrix, Path(temp) / 'sigmond', cutoff=1e-6,
                                    rotate_mode='bins' if matrix.is_bins else
                                    'samplings' if matrix.info.subtract_vev else 'samplings_unsubt',
                                    timings=timing, timeout=args.sigmond_timeout)
                    native_timings.append(timing)
                    print(f"Sigmond batch: {timing['batch_process']:.3f}s", file=sys.stderr, flush=True)
                    return result
                results['seconds']['sigmond_input_build_rotation_write'] = measure(
                    native, args.repeats)
                results['sigmond_seconds'] = {key: float(np.median([r[key] for r in native_timings[1:]]))
                                             for key in native_timings[0]}
                results['sigmond_scope'] = ('batch_process includes process startup, input reading, diagnostics, '
                                            'pivot writing and rotation/output; SS starts from an assembled resident matrix')
                from sigmondsamplings.io.loader import SigmondLoader
                def load_native_input():
                    index = SigmondLoader(Path(temp) / 'sigmond' / 'input.hdf5', group='input').correlators
                    return index.matrix(matrix.operators, times=matrix.times,
                                        subtract_vev=matrix.info.subtract_vev, key_form='HermMat')
                if not matrix.is_bins and not matrix.info.subtract_vev and not args.improved:
                    def file_pipeline():
                        loaded = load_native_input()
                        built = SinglePivot.from_matrix(loaded, norm_time=1, metric_time=1,
                                                        diagonalize_time=3, min_inv_condition=1e-6)
                        transformed = built.rotate(loaded)
                        built.off_diagonal_check(transformed)
                        built.write(Path(temp) / 'ss-pivot.hdf5', overwrite=True)
                        transformed.write(path, overwrite=True, diagonal_only_before_time=3,
                                          upper_triangle_only=True)
                    results['seconds']['ss_read_assemble'] = measure(load_native_input, args.repeats)
                    results['seconds']['ss_from_same_input_file'] = measure(file_pipeline, args.repeats)
        results['retained_rank'] = pivot.gevp.rank
        results['input_bytes'] = matrix.values.data.nbytes
        results['output_bytes'] = rotated.values.data.nbytes
        results['peak_rss_bytes_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        results['peak_rss_scope'] = 'whole benchmark process, including fixture construction, warmups, and old-kernel comparison'
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
