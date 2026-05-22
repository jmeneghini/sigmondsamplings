import os
import threading

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sigmondsamplings.fit import (
    SamplingFit,
    default_num_workers,
    evaluate_chi2_function_scan,
    evaluate_chi2_scan,
    set_thread_counts,
)
from sigmondsamplings.fit_plotter import plot_chi2_function_1d, plot_chi2_function_2d
from sigmondsamplings.info import SamplingInfo
from sigmondsamplings.model_func import SigmondModelFunc, polynomial_model


def test_default_num_workers_accounts_for_blas_and_openmp(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    assert default_num_workers() == 2
    assert default_num_workers(num_blas_threads=1, num_openmp_threads=2) == 8


def test_run_temporarily_sets_blas_and_openmp_thread_env(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    seen = []

    def fit_at_resamp(resamp_idx, x0):
        seen.append(
            (
                resamp_idx,
                os.environ["OPENBLAS_NUM_THREADS"],
                os.environ["OMP_NUM_THREADS"],
            )
        )
        return np.asarray([x0[0] + resamp_idx]), float(resamp_idx)

    fit = SamplingFit(SamplingInfo("bootstrap", 1))
    fit.run(
        fit_at_resamp,
        ["a"],
        x0=np.asarray([1.0]),
        progress=False,
        num_blas_threads=1,
        num_openmp_threads=2,
    )

    assert seen == [(0, "1", "2"), (1, "1", "2")]
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_run_thread_backend_records_worker_results():
    thread_ids = set()

    def fit_at_resamp(resamp_idx, x0):
        thread_ids.add(threading.get_ident())
        return np.asarray([x0[0] + resamp_idx]), float(resamp_idx)

    fit = SamplingFit(SamplingInfo("bootstrap", 4))
    result = fit.run(
        fit_at_resamp,
        ["a"],
        x0=np.asarray([1.0]),
        backend="thread",
        num_workers=2,
        progress=False,
    )

    assert result.n_failed == 0
    np.testing.assert_allclose(result.param("a").data, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(result.chi_squared.data, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert thread_ids


def test_run_records_failed_resamples():
    def fit_at_resamp(resamp_idx, x0):
        if resamp_idx == 2:
            raise RuntimeError("bad resample")
        return np.asarray([x0[0] + resamp_idx]), float(resamp_idx)

    fit = SamplingFit(SamplingInfo("bootstrap", 3))
    result = fit.run(
        fit_at_resamp,
        ["a"],
        x0=np.asarray([1.0]),
        progress=False,
    )

    assert result.n_failed == 1
    assert 2 in result.failures
    assert "bad resample" in result.failures[2]
    np.testing.assert_allclose(result.param("a").data, [1.0, 2.0, 1.0, 4.0])
    assert np.isnan(result.chi_squared.data[2])


def test_fit_result_params_are_model_func_ready_collection():
    def fit_at_resamp(resamp_idx, x0):
        return np.asarray([2.0 + 0.1 * resamp_idx, 0.5 + 0.01 * resamp_idx]), 1.0

    def linear_model(x, slope, intercept):
        return slope * x + intercept

    fit = SamplingFit(SamplingInfo("bootstrap", 2))
    result = fit.run(
        fit_at_resamp,
        ["slope", "intercept"],
        x0=np.asarray([1.0, 0.0]),
        progress=False,
    )

    model = SigmondModelFunc(
        linear_model,
        result.params,
        latex_str=r"m {VAR} + b",
        independent_var_latex="x",
    )

    y = model(2.0)

    assert list(result.params.obs.name) == ["slope", "intercept"]
    assert y.full_sample_value == 4.5


def test_polynomial_model_accepts_varargs_signature():
    model = polynomial_model(2, SamplingInfo("bootstrap", 2))
    assert len(model.parameter_infos) == 3


def test_run_raises_failed_resamples_when_requested():
    def fit_at_resamp(resamp_idx, x0):
        if resamp_idx == 1:
            raise RuntimeError("bad resample")
        return np.asarray([x0[0] + resamp_idx]), float(resamp_idx)

    fit = SamplingFit(SamplingInfo("bootstrap", 1))
    with pytest.raises(RuntimeError, match="bad resample"):
        fit.run(
            fit_at_resamp,
            ["a"],
            x0=np.asarray([1.0]),
            progress=False,
            error_policy="raise",
        )


def test_set_thread_counts_updates_common_env_vars(monkeypatch):
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    set_thread_counts(num_blas_threads=2, num_openmp_threads=3)

    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["BLIS_NUM_THREADS"] == "2"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "2"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_evaluate_chi2_scan_thread_backend_preserves_order():
    class DummyStats:
        def chi_squared(self, theory, use_corr=True, resamp_idx=0):
            return float(theory[0] + 10 * use_corr + resamp_idx)

    def prediction(params):
        return np.asarray([params[0] + 2 * params[1]])

    scan = evaluate_chi2_scan(
        DummyStats(),
        prediction,
        [0],
        np.asarray([[1.0], [2.0], [3.0]]),
        fixed_params={1: 5.0},
        backend="thread",
        num_workers=2,
        use_correlation=False,
        resamp_idx=7,
    )

    np.testing.assert_allclose(scan.param_stack, [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    np.testing.assert_allclose(scan.chi2_values, [18.0, 19.0, 20.0])


def test_evaluate_chi2_function_scan_thread_backend_preserves_order():
    def chi2(params):
        return params[0] + 2 * params[1]

    scan = evaluate_chi2_function_scan(
        chi2,
        [0],
        np.asarray([[1.0], [2.0], [3.0]]),
        fixed_params={1: 5.0},
        backend="thread",
        num_workers=2,
    )

    np.testing.assert_allclose(scan.param_stack, [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    np.testing.assert_allclose(scan.chi2_values, [11.0, 12.0, 13.0])


def test_plot_chi2_function_wrappers_smoke():
    def chi2(params):
        return float((params[0] - 1.0) ** 2 + (params[1] + 2.0) ** 2)

    fig_1d, ax_1d = plot_chi2_function_1d(
        chi2,
        0,
        (0.0, 2.0),
        n_points=5,
        fixed_params={1: -2.0},
        n_total_params=2,
    )
    fig_2d, ax_2d = plot_chi2_function_2d(
        chi2,
        (0, 1),
        ((0.0, 2.0), (-3.0, -1.0)),
        n_points=(4, 4),
    )

    assert ax_1d.figure is fig_1d
    assert ax_2d.figure is fig_2d
