"""Multi-sampling fit driver.

Provides a thin orchestration layer over a user-supplied
``fit_at_resamp(resamp_idx, x0) -> (params, chisq)`` callable.  The driver:

* runs the full-sample fit (``resamp_idx=0``) in the calling process,
* optionally loops over the remaining resamplings in parallel with a thread
  or process pool,
* packages the per-parameter results as :class:`SigmondSampling` objects and
  hands back a :class:`SamplingPlotter` pre-loaded with them.

The fit function itself is agnostic to where it runs (process pool, thread
pool, in-process).  Callers that need per-worker state (e.g. a persistent
C++ chi-square object) can pass a worker initializer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .info import DEFAULT_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo
from .obervable_collection import ObservableCollection
from .plotter import SamplingPlotter
from .sampling import SigmondSampling
from .stats import SamplingStats

logger = logging.getLogger(__name__)

FitAtResamp = Callable[[int, np.ndarray], tuple[np.ndarray, float]]
ProgressKind = bool | str  # True/False/"auto"/"notebook"/"terminal"
FitBackend = Literal["serial", "thread", "process"]
ErrorPolicy = Literal["record", "raise"]

BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
OPENMP_THREAD_ENV_VARS = ("OMP_NUM_THREADS",)


@dataclass
class SamplingFitResult:
    """Output of :meth:`SamplingFit.run`."""

    params: ObservableCollection
    chi_squared: SigmondSampling
    full_sample_params: np.ndarray
    full_sample_chisq: float
    n_fits: int
    n_failed: int
    failures: dict[int, str]
    plotter: SamplingPlotter

    def __repr__(self) -> str:
        return (
            f"SamplingFitResult(n_params={len(self.params)}, "
            f"n_fits={self.n_fits}, n_failed={self.n_failed}, "
            f"chi2_full={self.full_sample_chisq:.6g})"
        )

    def model_func(
        self,
        func: Callable,
        latex_str: str | None = None,
        independent_var_latex: str | None = None,
    ):
        """Build a SigmondModelFunc using this result's parameter collection."""
        from .model_func import SigmondModelFunc

        return SigmondModelFunc(
            func,
            self.params,
            latex_str=latex_str,
            independent_var_latex=independent_var_latex,
        )

    def param(self, name: str) -> SigmondSampling:
        """Return a fitted parameter sampling by observable name."""
        return self.params.find(name=name)


@dataclass(frozen=True)
class FitExecutionConfig:
    """Execution settings for per-resampling fits."""

    backend: FitBackend = "serial"
    num_workers: int | Literal["auto"] | None = "auto"
    num_blas_threads: int | None = None
    num_openmp_threads: int | None = None
    worker_initializer: Callable | None = None
    worker_initargs: tuple = ()


@dataclass(frozen=True)
class Chi2Scan:
    """Parameter stack and chi-squared values for a scan."""

    param_stack: np.ndarray
    chi2_values: np.ndarray


def evaluate_chi2_scan(
    stats: SamplingStats,
    prediction_func: Callable[[np.ndarray], np.ndarray],
    varying_indices: Iterable[int],
    varying_values: np.ndarray,
    *,
    fixed_params: dict[int, float] | None = None,
    n_total_params: int | None = None,
    use_correlation: bool = True,
    resamp_idx: int = 0,
    backend: FitBackend = "serial",
    num_workers: int | Literal["auto"] | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
) -> Chi2Scan:
    """Evaluate chi² over rows of varied parameter values."""
    config = FitExecutionConfig(
        backend=_validate_backend(backend),
        num_workers=num_workers,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    varying_indices = list(varying_indices)
    fixed_params = dict(fixed_params or {})
    param_stack = _build_chi2_scan_param_stack(
        varying_indices,
        varying_values,
        fixed_params=fixed_params,
        n_total_params=n_total_params,
    )
    with _thread_count_context(
        num_blas_threads=config.num_blas_threads,
        num_openmp_threads=config.num_openmp_threads,
    ):
        chi2_values = _evaluate_chi2_param_stack(
            stats,
            prediction_func,
            param_stack,
            use_correlation=use_correlation,
            resamp_idx=resamp_idx,
            config=config,
            progress=progress,
        )
    return Chi2Scan(param_stack=param_stack, chi2_values=chi2_values)


def evaluate_chi2_function_scan(
    chi2_func: Callable[[np.ndarray], float],
    varying_indices: Iterable[int],
    varying_values: np.ndarray,
    *,
    fixed_params: dict[int, float] | None = None,
    n_total_params: int | None = None,
    backend: FitBackend = "serial",
    num_workers: int | Literal["auto"] | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
) -> Chi2Scan:
    """Evaluate a direct ``chi2_func(params)`` over a parameter scan."""
    config = FitExecutionConfig(
        backend=_validate_backend(backend),
        num_workers=num_workers,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    param_stack = _build_chi2_scan_param_stack(
        varying_indices,
        varying_values,
        fixed_params=fixed_params,
        n_total_params=n_total_params,
    )
    with _thread_count_context(
        num_blas_threads=config.num_blas_threads,
        num_openmp_threads=config.num_openmp_threads,
    ):
        chi2_values = _evaluate_chi2_function_stack(
            chi2_func,
            param_stack,
            config=config,
            progress=progress,
        )
    return Chi2Scan(param_stack=param_stack, chi2_values=chi2_values)


class SamplingFit:
    """Orchestrate a ``(params, resamp_idx) -> chisq`` minimiser over samplings.

    Parameters
    ----------
    sampling_info:
        Sampling info shared by all generated :class:`SigmondSampling` outputs.
    ensemble_info:
        Ensemble tag attached to the output observables.  Defaults to
        :data:`DEFAULT_ENSEMBLE` when fits aggregate across ensembles.
    """

    def __init__(
        self,
        sampling_info: SamplingInfo,
        ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE,
    ):
        self.sampling_info = sampling_info
        self.ensemble_info = ensemble_info

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        fit_at_resamp: FitAtResamp,
        param_names: Iterable[str],
        n_resamplings: int | None = None,
        x0: np.ndarray | None = None,
        full_sample_only: bool = False,
        backend: FitBackend = "serial",
        num_workers: int | Literal["auto"] | None = "auto",
        progress: ProgressKind = True,
        num_blas_threads: int | None = None,
        num_openmp_threads: int | None = None,
        worker_initializer: Callable | None = None,
        worker_initargs: tuple = (),
        error_policy: ErrorPolicy = "record",
        full_sample_result: tuple[np.ndarray, float] | None = None,
    ) -> SamplingFitResult:
        """Run the full-sample fit and (optionally) the per-resampling fits.

        Parameters
        ----------
        fit_at_resamp:
            ``f(resamp_idx, x0) -> (best_params, chisq)``.  Must be picklable
            when using ``backend="process"``.
        param_names:
            Parameter names — drives the output ``params`` dict keys.
        n_resamplings:
            Number of resamplings to fit (1..N).  Defaults to
            ``sampling_info.num_resamplings``.
        x0:
            Initial parameter guess for the full-sample fit.  Per-resample
            fits always reuse the full-sample best fit as ``x0``.
        full_sample_only:
            If ``True`` only the full-sample fit (``resamp_idx=0``) is run;
            the returned parameter resamples are filled with the full-sample
            values and per-resample chi-squared values are set to ``NaN``.
        backend:
            Per-resample execution backend: ``"serial"``, ``"thread"``, or
            ``"process"``.  The full-sample fit always runs in the calling
            process unless ``full_sample_result`` is provided.
        num_workers:
            Number of workers for ``"thread"`` and ``"process"`` backends.
            ``"auto"`` uses :func:`default_num_workers`.
        progress:
            Controls the per-resample progress bar.  ``True`` / ``"auto"``
            auto-detects (marimo if running inside a marimo notebook,
            otherwise :mod:`tqdm.auto` which picks notebook tqdm under
            IPython and terminal tqdm elsewhere); ``"notebook"`` forces
            :mod:`tqdm.notebook`; ``"terminal"`` / ``"text"`` forces
            :mod:`tqdm.std`; ``"marimo"`` forces ``marimo.status.progress_bar``;
            ``False`` disables.
        full_sample_result:
            ``(params, chisq)`` already obtained for the full sample.  When
            provided, ``fit_at_resamp`` is *not* called at ``resamp_idx=0`` —
            useful when the caller cannot invoke ``fit_at_resamp`` in the
            current process (e.g. it relies on per-worker state).
        num_blas_threads:
            If set, temporarily export common BLAS thread-count environment
            variables (``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``, etc.)
            while the fits are submitted/run.
        num_openmp_threads:
            If set, temporarily export ``OMP_NUM_THREADS`` while the fits are
            submitted/run.
        worker_initializer:
            Optional callable run once in each thread/process worker.
        worker_initargs:
            Arguments passed to ``worker_initializer``.
        error_policy:
            ``"record"`` stores failed resamples in ``result.failures`` and
            substitutes full-sample parameters; ``"raise"`` propagates the
            first failed resample exception.
        """
        config = FitExecutionConfig(
            backend=_validate_backend(backend),
            num_workers=num_workers,
            num_blas_threads=num_blas_threads,
            num_openmp_threads=num_openmp_threads,
            worker_initializer=worker_initializer,
            worker_initargs=worker_initargs,
        )
        with _thread_count_context(
            num_blas_threads=config.num_blas_threads,
            num_openmp_threads=config.num_openmp_threads,
        ):
            return self._run_impl(
                fit_at_resamp=fit_at_resamp,
                param_names=param_names,
                n_resamplings=n_resamplings,
                x0=x0,
                full_sample_only=full_sample_only,
                config=config,
                progress=progress,
                error_policy=_validate_error_policy(error_policy),
                full_sample_result=full_sample_result,
            )

    @staticmethod
    def set_thread_counts(
        *,
        num_blas_threads: int | None = None,
        num_openmp_threads: int | None = None,
    ) -> None:
        """Set BLAS/OpenMP thread-count environment variables persistently."""
        set_thread_counts(
            num_blas_threads=num_blas_threads,
            num_openmp_threads=num_openmp_threads,
        )

    def _run_impl(
        self,
        fit_at_resamp: FitAtResamp,
        param_names: Iterable[str],
        n_resamplings: int | None,
        x0: np.ndarray | None,
        full_sample_only: bool,
        config: FitExecutionConfig,
        progress: ProgressKind,
        error_policy: ErrorPolicy,
        full_sample_result: tuple[np.ndarray, float] | None = None,
    ) -> SamplingFitResult:
        param_names = list(param_names)
        n_params = len(param_names)
        if n_resamplings is None:
            n_resamplings = self.sampling_info.num_resamplings
        n_samples = n_resamplings + 1  # full + resamples

        # ---- full sample -------------------------------------------------
        if x0 is None:
            x0 = np.zeros(n_params)
        if full_sample_result is not None:
            full_params, full_chisq = full_sample_result
        else:
            full_params, full_chisq = fit_at_resamp(0, np.asarray(x0, dtype=float))
        full_params = np.asarray(full_params, dtype=float).reshape(-1)
        if full_params.size != n_params:
            raise ValueError(
                f"fit_at_resamp returned {full_params.size} params, expected {n_params}"
            )

        # ---- per-resample loop ------------------------------------------
        param_matrix = np.empty((n_params, n_samples), dtype=float)
        chisq_arr = np.empty(n_samples, dtype=float)
        param_matrix[:, 0] = full_params
        chisq_arr[0] = full_chisq

        n_failed = 0
        failures: dict[int, str] = {}
        if not full_sample_only and n_resamplings > 0:
            with _executor_context(config) as executor:
                pairs = self._loop(
                    fit_at_resamp,
                    indices=range(1, n_samples),
                    x0=full_params,
                    executor=executor,
                    progress=progress,
                    error_policy=error_policy,
                )
                for idx, (params, chisq, error) in pairs:
                    if params is None:
                        n_failed += 1
                        failures[idx] = error or "unknown fit failure"
                        param_matrix[:, idx] = full_params
                        chisq_arr[idx] = float("nan")
                    else:
                        arr = np.asarray(params, dtype=float).reshape(-1)
                        if arr.size != n_params:
                            raise ValueError(
                                f"fit_at_resamp(resamp_idx={idx}) returned "
                                f"{arr.size} params, expected {n_params}"
                            )
                        param_matrix[:, idx] = arr
                        chisq_arr[idx] = float(chisq)
        else:
            # broadcast full-sample values; chi-squared per resample is unknown
            param_matrix[:, 1:] = full_params[:, None]
            chisq_arr[1:] = float("nan")

        return self._build_result(
            param_names=param_names,
            param_matrix=param_matrix,
            chisq_arr=chisq_arr,
            full_params=full_params,
            full_chisq=full_chisq,
            n_failed=n_failed,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _loop(
        self,
        fit_at_resamp: FitAtResamp,
        indices: Iterable[int],
        x0: np.ndarray,
        executor: Executor | None,
        progress: ProgressKind,
        error_policy: ErrorPolicy,
    ):
        """Yield ``(resamp_idx, (params, chisq, error))`` for each index.

        ``params`` is ``None`` when the per-sample fit raised.
        """
        indices = list(indices)
        bar = _maybe_tqdm(progress, total=len(indices), desc="Resample fits")

        if executor is None:
            for i in indices:
                try:
                    params, chisq = fit_at_resamp(i, x0)
                    result = (params, chisq, None)
                except Exception as exc:
                    if error_policy == "raise":
                        raise
                    result = (None, float("nan"), repr(exc))
                if bar is not None:
                    bar.update(1)
                yield i, result
            if bar is not None:
                bar.close()
            return

        futures = {executor.submit(fit_at_resamp, i, x0): i for i in indices}
        try:
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    params, chisq = fut.result()
                    result = (params, chisq, None)
                except Exception as exc:
                    if error_policy == "raise":
                        raise
                    result = (None, float("nan"), repr(exc))
                if bar is not None:
                    bar.update(1)
                yield i, result
        finally:
            if bar is not None:
                bar.close()

    def _build_result(
        self,
        *,
        param_names: list[str],
        param_matrix: np.ndarray,
        chisq_arr: np.ndarray,
        full_params: np.ndarray,
        full_chisq: float,
        n_failed: int,
        failures: dict[int, str],
    ) -> SamplingFitResult:
        params: list[SigmondSampling] = []
        for i, name in enumerate(param_names):
            obs = ObservableInfo(
                name=name,
                index=0,
                op_type="n",
                re_im="re",
                ensemble_info=self.ensemble_info,
            )
            params.append(
                SigmondSampling(
                    data=param_matrix[i],
                    observable_info=obs,
                    sampling_info=self.sampling_info,
                    is_complex=False,
                )
            )

        chi_obs = ObservableInfo(
            name="chi_squared",
            index=0,
            op_type="n",
            re_im="re",
            ensemble_info=self.ensemble_info,
            latex_str=r"\chi^2",
        )
        chi_squared = SigmondSampling(
            data=chisq_arr,
            observable_info=chi_obs,
            sampling_info=self.sampling_info,
            is_complex=False,
        )

        param_collection = ObservableCollection(params)
        plotter = SamplingPlotter(param_collection)

        return SamplingFitResult(
            params=param_collection,
            chi_squared=chi_squared,
            full_sample_params=full_params,
            full_sample_chisq=float(full_chisq),
            n_fits=int(param_matrix.shape[1]),
            n_failed=int(n_failed),
            failures=failures,
            plotter=plotter,
        )


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


_PROGRESS_ALIASES = {
    True: "auto",
    "auto": "auto",
    "notebook": "notebook",
    "nb": "notebook",
    "terminal": "terminal",
    "text": "terminal",
    "std": "terminal",
    "marimo": "marimo",
    "mo": "marimo",
}


def _maybe_tqdm(progress: ProgressKind, *, total: int, desc: str):
    """Build a progress bar exposing ``.update(n)`` / ``.close()``.

    Returns ``None`` when ``progress`` is falsy or the requested backend is
    unavailable.
    """
    if not progress:
        return None
    kind = _PROGRESS_ALIASES.get(progress)
    if kind is None:
        raise ValueError(
            f"Unknown progress kind {progress!r}; expected one of "
            f"{sorted(k for k in _PROGRESS_ALIASES if isinstance(k, str))} or a bool."
        )

    if kind == "auto":
        if _running_in_marimo():
            # Marimo monkey-patches tqdm.notebook.tqdm with its own
            # ProgressBarTqdmPatch (see marimo/_output/formatters/tqdm_formatters.py).
            # Routing through tqdm.notebook benefits from that integration; using
            # tqdm.auto doesn't help because marimo has no IPython kernel so
            # auto-detection would fall through to tqdm.std (terminal-only).
            logger.info("progress='auto': detected marimo notebook; using tqdm.notebook")
            return _build_tqdm_bar("notebook", total=total, desc=desc)
        logger.info("progress='auto': no marimo notebook detected; using tqdm.auto")
        return _build_tqdm_bar("auto", total=total, desc=desc)
    if kind == "marimo":
        if _running_in_marimo():
            return _build_tqdm_bar("notebook", total=total, desc=desc)
        bar = _build_marimo_bar(total=total, desc=desc)
        if bar is not None:
            return bar
        logger.info("progress='marimo' requested but unavailable; falling back to tqdm.auto")
        return _build_tqdm_bar("auto", total=total, desc=desc)
    return _build_tqdm_bar(kind, total=total, desc=desc)


def _build_tqdm_bar(kind: str, *, total: int, desc: str):
    try:
        if kind == "notebook":
            from tqdm.notebook import tqdm
        elif kind == "terminal":
            from tqdm.std import tqdm
        else:
            from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        return None


def _running_in_marimo() -> bool:
    try:
        import marimo as mo
    except ImportError:
        return False
    try:
        return bool(mo.running_in_notebook())
    except Exception:
        return False


class _MarimoBar:
    """Adapter that exposes ``tqdm``-style ``update(n)``/``close()`` over
    :func:`marimo.status.progress_bar`."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._bar = ctx.__enter__()

    def update(self, n: int = 1) -> None:
        try:
            self._bar.update(increment=n)
        except TypeError:
            self._bar.update(n)

    def close(self) -> None:
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass


def _build_marimo_bar(*, total: int, desc: str):
    try:
        import marimo as mo
    except ImportError:
        return None
    try:
        ctx = mo.status.progress_bar(total=total, title=desc)
        return _MarimoBar(ctx)
    except Exception:
        return None


def _validate_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_backend(backend: str) -> FitBackend:
    if backend not in ("serial", "thread", "process"):
        raise ValueError("backend must be one of 'serial', 'thread', or 'process'")
    return backend


def _validate_error_policy(error_policy: str) -> ErrorPolicy:
    if error_policy not in ("record", "raise"):
        raise ValueError("error_policy must be 'record' or 'raise'")
    return error_policy


def _resolve_n_total_params(
    varying_indices: Iterable[int],
    fixed_params: dict[int, float],
    override: int | None,
) -> int:
    required = max([*varying_indices, *fixed_params.keys()], default=-1) + 1
    if override is None:
        return required
    if override < required:
        raise ValueError(
            f"n_total_params={override} is too small for parameter index {required - 1}"
        )
    return override


def _build_chi2_scan_param_stack(
    varying_indices: Iterable[int],
    varying_values: np.ndarray,
    *,
    fixed_params: dict[int, float] | None,
    n_total_params: int | None,
) -> np.ndarray:
    varying_indices = list(varying_indices)
    fixed_params = dict(fixed_params or {})
    n_total_params = _resolve_n_total_params(varying_indices, fixed_params, n_total_params)
    return _build_param_stack(
        n_total_params,
        fixed_params,
        varying_indices,
        np.asarray(varying_values, dtype=float),
    )


def _build_param_stack(
    n_total_params: int,
    fixed_params: dict[int, float],
    varying_indices: list[int],
    varying_values: np.ndarray,
) -> np.ndarray:
    """Build an ``(n_points, n_total_params)`` parameter stack."""
    if varying_values.ndim != 2 or varying_values.shape[1] != len(varying_indices):
        raise ValueError("varying_values must have shape (n_points, len(varying_indices))")
    stack = np.zeros((varying_values.shape[0], n_total_params))
    for idx, val in fixed_params.items():
        stack[:, idx] = val
    for col, idx in enumerate(varying_indices):
        stack[:, idx] = varying_values[:, col]
    return stack


def _evaluate_predictions(
    prediction_func: Callable[[np.ndarray], np.ndarray],
    param_stack: np.ndarray,
) -> np.ndarray:
    """Evaluate ``prediction_func`` over a stack of parameter vectors."""
    n_points = param_stack.shape[0]
    try:
        batched = np.asarray(prediction_func(param_stack))
        if batched.ndim == 2 and batched.shape[0] == n_points:
            return batched
    except Exception:
        pass

    rows = [np.asarray(prediction_func(p)).reshape(-1) for p in param_stack]
    return np.vstack(rows)


def _evaluate_chi2_param_stack(
    stats: SamplingStats,
    prediction_func: Callable[[np.ndarray], np.ndarray],
    param_stack: np.ndarray,
    *,
    use_correlation: bool,
    resamp_idx: int,
    config: FitExecutionConfig,
    progress: ProgressKind,
) -> np.ndarray:
    if config.backend == "serial":
        theory_stack = _evaluate_predictions(prediction_func, param_stack)
        return np.array(
            [
                stats.chi_squared(theory, use_corr=use_correlation, resamp_idx=resamp_idx)
                for theory in theory_stack
            ],
            dtype=float,
        )

    chi2_values = np.empty(param_stack.shape[0], dtype=float)
    bar = _maybe_tqdm(progress, total=param_stack.shape[0], desc="Chi2 scan")
    try:
        with _executor_context(config) as executor:
            futures = {
                executor.submit(
                    _evaluate_chi2_at_params,
                    stats,
                    prediction_func,
                    params,
                    use_correlation,
                    resamp_idx,
                ): i
                for i, params in enumerate(param_stack)
            }
            for fut in as_completed(futures):
                chi2_values[futures[fut]] = fut.result()
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return chi2_values


def _evaluate_chi2_function_stack(
    chi2_func: Callable[[np.ndarray], float],
    param_stack: np.ndarray,
    *,
    config: FitExecutionConfig,
    progress: ProgressKind,
) -> np.ndarray:
    if config.backend == "serial":
        return np.array([chi2_func(params) for params in param_stack], dtype=float)

    chi2_values = np.empty(param_stack.shape[0], dtype=float)
    bar = _maybe_tqdm(progress, total=param_stack.shape[0], desc="Chi2 scan")
    try:
        with _executor_context(config) as executor:
            futures = {
                executor.submit(chi2_func, params): i for i, params in enumerate(param_stack)
            }
            for fut in as_completed(futures):
                chi2_values[futures[fut]] = float(fut.result())
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return chi2_values


def _evaluate_chi2_at_params(
    stats: SamplingStats,
    prediction_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    use_correlation: bool,
    resamp_idx: int,
) -> float:
    theory = np.asarray(prediction_func(params)).reshape(-1)
    return float(stats.chi_squared(theory, use_corr=use_correlation, resamp_idx=resamp_idx))


def _thread_count_from_env(env_names: tuple[str, ...], default: int = 1) -> int:
    for name in env_names:
        value = os.environ.get(name)
        if value is not None:
            return max(1, int(value))
    return default


def _available_cpus() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass

    return os.cpu_count() or 1


def _resolve_num_workers(
    num_workers: int | Literal["auto"] | None,
    *,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
) -> int:
    if num_workers in (None, "auto"):
        return default_num_workers(
            num_blas_threads=num_blas_threads,
            num_openmp_threads=num_openmp_threads,
        )
    return _validate_positive_int(num_workers, "num_workers")


@contextmanager
def _executor_context(config: FitExecutionConfig):
    if config.backend == "serial":
        yield None
        return

    max_workers = _resolve_num_workers(
        config.num_workers,
        num_blas_threads=config.num_blas_threads,
        num_openmp_threads=config.num_openmp_threads,
    )
    executor_cls: type[Executor]
    if config.backend == "thread":
        executor_cls = ThreadPoolExecutor
    else:
        executor_cls = ProcessPoolExecutor

    executor = executor_cls(
        max_workers=max_workers,
        initializer=config.worker_initializer,
        initargs=config.worker_initargs,
    )
    try:
        yield executor
    finally:
        executor.shutdown()


def set_thread_counts(
    *,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
) -> None:
    """Set BLAS/OpenMP thread-count environment variables.

    ``None`` leaves the corresponding environment variables unchanged.
    """
    num_blas_threads = _validate_positive_int(num_blas_threads, "num_blas_threads")
    num_openmp_threads = _validate_positive_int(
        num_openmp_threads,
        "num_openmp_threads",
    )
    if num_blas_threads is not None:
        value = str(num_blas_threads)
        for name in BLAS_THREAD_ENV_VARS:
            os.environ[name] = value
    if num_openmp_threads is not None:
        value = str(num_openmp_threads)
        for name in OPENMP_THREAD_ENV_VARS:
            os.environ[name] = value


@contextmanager
def _thread_count_context(
    *,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
):
    env_names: list[str] = []
    if num_blas_threads is not None:
        env_names.extend(BLAS_THREAD_ENV_VARS)
    if num_openmp_threads is not None:
        env_names.extend(OPENMP_THREAD_ENV_VARS)

    old_values = {name: os.environ.get(name) for name in env_names}
    set_thread_counts(
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
    )
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def default_num_workers(
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
) -> int:
    """Reasonable worker count under SLURM + threaded-library environments.

    Reads ``SLURM_CPUS_PER_TASK`` plus BLAS/OpenMP thread-count environment
    variables and returns ``available_cpus // (blas_threads * openmp_threads)``
    (floor 1).  If SLURM is unset, CPU availability falls back to the process
    affinity mask when available, then ``os.cpu_count()``.

    Explicit ``num_blas_threads`` and ``num_openmp_threads`` override the
    environment when estimating the worker count.
    """
    blas = _validate_positive_int(num_blas_threads, "num_blas_threads")
    openmp = _validate_positive_int(num_openmp_threads, "num_openmp_threads")
    if blas is None:
        blas = _thread_count_from_env(BLAS_THREAD_ENV_VARS)
    if openmp is None:
        openmp = _thread_count_from_env(OPENMP_THREAD_ENV_VARS)
    return max(1, _available_cpus() // (blas * openmp))


def make_process_pool(
    initializer: Callable | None = None,
    initargs: tuple = (),
    num_workers: int | Literal["auto"] | None = "auto",
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
) -> ProcessPoolExecutor:
    """Convenience constructor for a ``ProcessPoolExecutor`` with worker init.

    Mirrors the pattern used by KBfit's chi-square workers: each worker
    builds and holds its own state once, then handles many tasks.  When
    ``num_blas_threads`` or ``num_openmp_threads`` are provided, the matching
    environment variables are set before the pool is constructed so worker
    processes inherit them.
    """
    set_thread_counts(
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
    )
    return ProcessPoolExecutor(
        max_workers=_resolve_num_workers(
            num_workers,
            num_blas_threads=num_blas_threads,
            num_openmp_threads=num_openmp_threads,
        ),
        initializer=initializer,
        initargs=initargs,
    )


__all__ = [
    "FitAtResamp",
    "FitBackend",
    "ErrorPolicy",
    "FitExecutionConfig",
    "Chi2Scan",
    "SamplingFit",
    "SamplingFitResult",
    "evaluate_chi2_function_scan",
    "evaluate_chi2_scan",
    "set_thread_counts",
    "default_num_workers",
    "make_process_pool",
]
