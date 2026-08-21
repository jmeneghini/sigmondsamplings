"""Chi-squared parameter scans for the optimagic-native fitting layer.

Evaluates :math:`\\chi^2` over a grid/stack of parameter vectors without running a
minimizer — useful for likelihood profiling, initial-value surveys, and plotting
:math:`\\chi^2` surfaces. Two entry points:

* :func:`evaluate_chi2_scan` — whitened :math:`\\chi^2` from a flat-vector
  ``prediction_func`` against a :class:`~sigmondsamplings.stats.SamplingStats`.
* :func:`evaluate_chi2_function_scan` — a direct ``chi2_func(params) -> float``.

Both build a parameter stack from varying indices/values plus fixed entries and
evaluate it serially or across a worker pool, reusing the execution helpers in
:mod:`sigmondsamplings.fitting._execution` (threadpoolctl thread limits,
SLURM-aware worker budgeting, tqdm/marimo progress).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._execution import (
    FitBackend,
    ProgressKind,
    executor_context,
    limit_native_threads,
    resolve_num_workers,
    run_jobs,
    validate_backend,
)
from .whitening import WhiteningTransform

if TYPE_CHECKING:
    from ..stats import SamplingStats

__all__ = [
    "Chi2Scan",
    "WhiteningTransform",
    "evaluate_chi2_function_scan",
    "evaluate_chi2_scan",
]


@dataclass(frozen=True)
class Chi2Scan:
    """Parameter stack and the :math:`\\chi^2` evaluated at each row."""

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
    blas_threads: int | None = None,
    progress: ProgressKind = False,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
) -> Chi2Scan:
    """Evaluate whitened :math:`\\chi^2` over rows of varied parameter values.

    ``prediction_func`` maps a parameter vector (length ``n_total_params``) to the
    flat theory vector aligned with ``stats``; the :math:`\\chi^2` uses the data
    covariance (``use_correlation``) at ``resamp_idx``.
    """
    backend = validate_backend(backend)
    param_stack = _build_chi2_scan_param_stack(
        varying_indices, varying_values, fixed_params=fixed_params, n_total_params=n_total_params
    )
    data_values = np.asarray(stats.array[:, resamp_idx], dtype=float)
    whitening = WhiteningTransform.from_stats(stats, use_correlation=use_correlation)
    worker = _Chi2PredictionWorker(data_values, prediction_func, whitening)
    chi2_values = _run_scan(
        worker,
        param_stack,
        backend=backend,
        num_workers=num_workers,
        blas_threads=blas_threads,
        progress=progress,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
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
    blas_threads: int | None = None,
    progress: ProgressKind = False,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
) -> Chi2Scan:
    """Evaluate a direct ``chi2_func(params) -> float`` over a parameter scan."""
    backend = validate_backend(backend)
    param_stack = _build_chi2_scan_param_stack(
        varying_indices, varying_values, fixed_params=fixed_params, n_total_params=n_total_params
    )
    worker = _Chi2FunctionWorker(chi2_func)
    chi2_values = _run_scan(
        worker,
        param_stack,
        backend=backend,
        num_workers=num_workers,
        blas_threads=blas_threads,
        progress=progress,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    return Chi2Scan(param_stack=param_stack, chi2_values=chi2_values)


# ---------------------------------------------------------------------------
# picklable per-row workers
# ---------------------------------------------------------------------------


@dataclass
class _Chi2PredictionWorker:
    data_values: np.ndarray
    prediction_func: Callable[[np.ndarray], np.ndarray]
    whitening: WhiteningTransform

    def __call__(self, params: np.ndarray) -> float:
        theory = np.asarray(self.prediction_func(params), dtype=float).reshape(-1)
        return self.whitening.chi2(self.data_values - theory)


@dataclass
class _Chi2FunctionWorker:
    chi2_func: Callable[[np.ndarray], float]

    def __call__(self, params: np.ndarray) -> float:
        return float(self.chi2_func(params))


# ---------------------------------------------------------------------------
# scan runner
# ---------------------------------------------------------------------------


def _run_scan(
    worker: Callable[[np.ndarray], float],
    param_stack: np.ndarray,
    *,
    backend: FitBackend,
    num_workers: int | Literal["auto"] | None,
    blas_threads: int | None,
    progress: ProgressKind,
    worker_initializer: Callable | None,
    worker_initargs: tuple,
) -> np.ndarray:
    """Evaluate ``worker`` over each row of ``param_stack`` and collect the values."""
    rows = list(param_stack)
    chi2_values = np.empty(len(rows), dtype=float)
    workers = resolve_num_workers(num_workers, inner_cores=1, blas_threads=blas_threads or 1)
    indexed = _IndexedWorker(worker, rows)

    with limit_native_threads(blas_threads if backend != "process" else None):
        with executor_context(
            backend,
            max_workers=workers,
            worker_initializer=worker_initializer,
            worker_initargs=worker_initargs,
            blas_threads=blas_threads,
        ) as executor:
            for i, value, _ in run_jobs(
                indexed,
                range(len(rows)),
                executor=executor,
                progress=progress,
                error_policy="raise",
                desc="Chi2 scan",
            ):
                chi2_values[i] = value
    return chi2_values


@dataclass
class _IndexedWorker:
    """Picklable ``index -> chi2`` adapter for the process backend."""

    worker: Callable[[np.ndarray], float]
    rows: list

    def __call__(self, i: int) -> float:
        return self.worker(self.rows[i])


# ---------------------------------------------------------------------------
# parameter-stack construction
# ---------------------------------------------------------------------------


def _build_chi2_scan_param_stack(
    varying_indices: Iterable[int],
    varying_values: np.ndarray,
    *,
    fixed_params: dict[int, float] | None,
    n_total_params: int | None,
) -> np.ndarray:
    fixed = {int(idx): float(value) for idx, value in dict(fixed_params or {}).items()}
    indices = _validate_param_indices(varying_indices, fixed)
    total = _resolve_n_total_params(indices, fixed, n_total_params)
    return _build_param_stack(total, fixed, indices, np.asarray(varying_values, dtype=float))


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


def _validate_param_indices(
    varying_indices: Iterable[int],
    fixed_params: dict[int, float],
) -> list[int]:
    indices = [int(idx) for idx in varying_indices]
    duplicates = sorted({idx for idx in indices if indices.count(idx) > 1})
    if duplicates:
        raise ValueError(f"varying_indices must be unique; duplicates: {duplicates}")
    negative = sorted(idx for idx in [*indices, *fixed_params.keys()] if idx < 0)
    if negative:
        raise ValueError(f"parameter indices must be non-negative; got {negative}")
    overlap = sorted(set(indices) & set(fixed_params))
    if overlap:
        raise ValueError(f"indices cannot be both fixed and varying; overlap: {overlap}")
    return indices


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
