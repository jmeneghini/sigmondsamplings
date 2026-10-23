"""Install-once payloads and chunked dispatch in ``fitting._execution.run_jobs``.

A process pool used to receive the whole per-index callable with every job. The
resample worker's objective holds one evaluator per index, so that cost grew as
N^2 over a run. These tests pin the replacement: the payload is unpickled once per
worker process, and jobs carry only chunks of indices.
"""

import os
import uuid

import numpy as np
import pytest

from sigmondsamplings.fitting import evaluate_chi2_function_scan
from sigmondsamplings.fitting._execution import (
    _PoolInfo,
    _resolve_chunksize,
    executor_context,
    run_jobs,
)


class _UnpickleTracker:
    """Returns which process ran each index and which unpickled copy ran it.

    ``token`` is regenerated whenever the object is unpickled, so the number of
    distinct tokens seen is the number of times the payload crossed into a
    worker.
    """

    def __init__(self):
        self.token = "original"

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.token = uuid.uuid4().hex

    def __call__(self, i):
        return os.getpid(), self.token, i * i


def _square_or_fail_odd(i):
    if i % 2:
        raise ValueError(f"odd {i}")
    return i * i


def _die(i):
    os._exit(1)


def _chi2(params):
    return params[0] + 2 * params[1]


def _collect(fn, indices, executor, **kw):
    return {i: (value, error) for i, value, error in run_jobs(fn, indices, executor=executor, **kw)}


# ---------------------------------------------------------------------------
# chunk sizing
# ---------------------------------------------------------------------------


def test_auto_chunksize_is_one_without_a_process_pool():
    assert _resolve_chunksize("auto", 1000, None) == 1


def test_auto_chunksize_targets_four_chunks_per_worker():
    pool = _PoolInfo(payload=None, max_workers=4)
    assert _resolve_chunksize("auto", 1000, pool) == 63  # ceil(1000 / 16)
    assert _resolve_chunksize("auto", 3, pool) == 1


def test_explicit_chunksize_is_validated():
    assert _resolve_chunksize(7, 10, None) == 7
    with pytest.raises(ValueError):
        _resolve_chunksize(0, 10, None)


# ---------------------------------------------------------------------------
# process pool
# ---------------------------------------------------------------------------


def test_installed_payload_is_unpickled_once_per_worker_process():
    tracker = _UnpickleTracker()
    with executor_context("process", max_workers=2, payload=tracker, mp_context="spawn") as ex:
        got = list(run_jobs(tracker, range(40), executor=ex))

    assert sorted(i for i, _, _ in got) == list(range(40))
    assert all(value[2] == i * i for i, value, _ in got)
    pids = {value[0] for _, value, _ in got}
    tokens = {value[1] for _, value, _ in got}
    assert "original" not in tokens
    assert len(tokens) == len(pids) <= 2


def test_other_callable_is_shipped_once_per_chunk():
    tracker = _UnpickleTracker()
    with executor_context("process", max_workers=2, mp_context="spawn") as ex:
        got = list(run_jobs(tracker, range(40), executor=ex, chunksize=10))

    assert sorted(i for i, _, _ in got) == list(range(40))
    assert len({value[1] for _, value, _ in got}) == 4


def test_record_policy_reports_errors_per_index_within_a_chunk():
    with executor_context(
        "process", max_workers=2, payload=_square_or_fail_odd, mp_context="spawn"
    ) as ex:
        got = _collect(_square_or_fail_odd, range(10), ex, chunksize=5)

    assert sorted(got) == list(range(10))
    for i, (value, error) in got.items():
        if i % 2:
            assert value is None and "odd" in error
        else:
            assert value == i * i and error is None


def test_raise_policy_names_the_failing_index():
    with executor_context(
        "process", max_workers=2, payload=_square_or_fail_odd, mp_context="spawn"
    ) as ex:
        with pytest.raises(ValueError) as info:
            list(run_jobs(_square_or_fail_odd, [2, 4, 7], executor=ex, error_policy="raise", chunksize=3))

    assert any("raised at index 7" in note for note in info.value.__notes__)


def test_lost_job_records_every_index_in_its_chunk():
    with executor_context("process", max_workers=1, payload=_die, mp_context="spawn") as ex:
        got = _collect(_die, range(6), ex, chunksize=3)

    assert sorted(got) == list(range(6))
    assert all(value is None and error for value, error in got.values())


# ---------------------------------------------------------------------------
# thread pool and callers
# ---------------------------------------------------------------------------


def test_thread_pool_runs_chunks_in_shared_memory():
    tracker = _UnpickleTracker()
    with executor_context("thread", max_workers=2, payload=tracker) as ex:
        got = list(run_jobs(tracker, range(12), executor=ex, chunksize=4))

    assert sorted(i for i, _, _ in got) == list(range(12))
    assert {value[1] for _, value, _ in got} == {"original"}


def test_chi2_scan_process_backend_matches_serial():
    values = np.arange(12, dtype=float)[:, None]
    serial = evaluate_chi2_function_scan(_chi2, [0], values, fixed_params={1: 5.0})
    pooled = evaluate_chi2_function_scan(
        _chi2, [0], values, fixed_params={1: 5.0}, backend="process", num_workers=2
    )
    np.testing.assert_allclose(pooled.chi2_values, serial.chi2_values)


_T = np.arange(1, 9, dtype=float)


def _decay(theta):
    return theta[0] * np.exp(-theta[1] * _T)


def test_fit_indices_process_backend_matches_serial():
    pytest.importorskip("optimagic")
    from sigmondsamplings.fitting import CallableModel, MinimizerConfig, ParamSetSpec, ParamSpec, SamplingFit
    from sigmondsamplings.info import ObservableInfo, SamplingInfo
    from sigmondsamplings.sampling import SigmondSampling
    from sigmondsamplings.stats import SamplingStats

    rng = np.random.default_rng(0)
    info = SamplingInfo("bootstrap", 40, seed=1)
    samps = []
    for i, t in enumerate(_T):
        mean = 1.25 * np.exp(-0.31 * t)
        data = np.concatenate([[mean], rng.normal(mean, 0.02 * mean + 1e-3, 40)])
        samps.append(SigmondSampling(data, ObservableInfo(name=f"y{i}", index=i), info))
    params = ParamSetSpec(values={"A": ParamSpec(initial=1.0), "E": ParamSpec(initial=0.5)}).resolve()
    fit = SamplingFit.from_model(
        SamplingStats(samps),
        CallableModel(_decay, ["A", "E"]),
        params,
        minimizer=MinimizerConfig(algorithm="scipy_ls_trf"),
    )
    central = fit.prepare_resampled()
    indices = range(1, 21)

    serial = fit.fit_indices(indices, central=central, progress=False)
    pooled = fit.fit_indices(
        indices,
        central=central,
        progress=False,
        backend="process",
        num_workers=2,
        mp_context="spawn",
        chunksize=3,
    )

    np.testing.assert_array_equal(pooled.indices, serial.indices)
    np.testing.assert_allclose(pooled.params, serial.params, rtol=1e-10)
    np.testing.assert_allclose(pooled.chi2, serial.chi2, rtol=1e-10)
    assert pooled.success.all()
