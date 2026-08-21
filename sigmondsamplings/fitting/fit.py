"""optimagic-native resampling fit driver.

:class:`SamplingFit` ties together the parameter description (:class:`ParamSetResolved`),
the minimizer selection (:class:`MinimizerConfig`), and a least-squares
:class:`Objective` into three entry points:

* :meth:`SamplingFit.fit_one` — a single optimagic fit at one resampling index,
  returning the native :class:`~optimagic.optimization.optimize_result.OptimizeResult`
  with optional full SQLite logging. The low-level building block of the others.
* :meth:`SamplingFit.fit_full_sample` — fit the full sample once, estimate the
  parameter covariance from the residual Jacobian (``inv(JᵀJ)``), and *synthesize*
  pseudo-resamples drawn from ``N(θ̂, cov)`` so the uniform SigmondSampling output
  reproduces ``sqrt(diag(cov))`` under the configured sampling method's error rule.
* :meth:`SamplingFit.fit_resampled` — fit the full sample for the shared initial
  value, then fit every resampling (parallel-capable) and collect the genuine
  spread of best-fit parameters.

The objective is a *whitened-residual vector* marked with
``optimagic.mark.least_squares``; least-squares algorithms consume it directly and
optimagic auto-aggregates it to a scalar :math:`\\chi^2` for scalar algorithms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo
from ..plotting import SamplingPlotter
from ..sampling import SigmondSampling
from ..stats import SamplingStats
from ._execution import (
    ErrorPolicy,
    FitBackend,
    ProgressKind,
    ThreadBudget,
    ThreadSetting,
    executor_context,
    format_observed_threads,
    limit_native_threads,
    plan_thread_budget,
    resolve_mp_context,
    run_jobs,
    validate_backend,
)
from .minimizer import MinimizerConfig
from .model import Model, predict_model
from .objective import LeastSquaresObjective, Objective
from .params import ParamSetResolved
from .result import RANK_RTOL, FitResult

logger = logging.getLogger(__name__)

__all__ = ["SamplingFit"]

KeepPolicy = Literal["summary", "none", "full"]


# ---------------------------------------------------------------------------
# linear algebra helpers
# ---------------------------------------------------------------------------


def _sym_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric positive-semidefinite square root via eigendecomposition."""
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _sym_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root, dropping near-null eigenvalues."""
    vals, vecs = np.linalg.eigh(matrix)
    cutoff = 1e-12 * max(float(vals.max()), 1.0)
    inv = np.zeros_like(vals)
    keep = vals > cutoff
    inv[keep] = 1.0 / np.sqrt(vals[keep])
    return (vecs * inv) @ vecs.T


def _draws_with_sample_cov(target_cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` zero-mean draws whose sample covariance (ddof=1) equals ``target_cov``.

    Standard-normal draws are exactly decorrelated to unit sample covariance and
    then coloured with the symmetric square root of ``target_cov``, so the returned
    rows reproduce the target second moments exactly rather than only in
    expectation — the synthesized SigmondSampling then reports the analytic error.
    """
    p = target_cov.shape[0]
    if n < 2:
        return rng.multivariate_normal(np.zeros(p), target_cov, size=max(n, 0))
    z = rng.standard_normal((n, p))
    z -= z.mean(axis=0, keepdims=True)
    sample_cov = np.atleast_2d(np.cov(z, rowvar=False, ddof=1))
    whitened = z @ _sym_inv_sqrt(sample_cov)
    return whitened @ _sym_sqrt(target_cov)


def _error_factor(method: str, n: int) -> float:
    """SigmondSampling error multiplier on the resample std for ``method``.

    Mirrors :attr:`SigmondSampling.error`: bootstrap reports the plain std,
    jackknife scales it by ``sqrt(n-1)``. The synthesized resamples are drawn so
    that ``std * factor == sqrt(diag(cov))``.
    """
    if method == "jackknife":
        return float(np.sqrt(max(n - 1, 1)))
    return 1.0


# ---------------------------------------------------------------------------
# single-fit plumbing (shared by fit_one and the resample worker)
# ---------------------------------------------------------------------------


def _run_minimize(
    objective: Objective,
    param_set: ParamSetResolved,
    minimizer: MinimizerConfig,
    resamp_idx: int,
    x0: dict[str, float] | None,
    logging_path: str | None = None,
) -> Any:
    """Run one ``optimagic.minimize`` and return the native ``OptimizeResult``."""
    import optimagic as om

    fun = objective.residuals_at(resamp_idx)
    params = dict(x0) if x0 is not None else param_set.start_values()
    kwargs = minimizer.minimize_kwargs()
    bounds = param_set.bounds()
    if bounds is not None:
        kwargs["bounds"] = bounds
    constraints = param_set.constraints_for_optimagic()
    if constraints:
        kwargs["constraints"] = constraints
    if logging_path is not None:
        kwargs["logging"] = om.SQLiteLogOptions(path=logging_path, if_database_exists="replace")
    return om.minimize(fun, params=params, **kwargs)


def _chi2_at(objective: Objective, params: dict[str, float], resamp_idx: int = 0) -> float:
    """:math:`\\chi^2` of ``objective`` at ``params`` for one resampling index."""
    chi2 = getattr(objective, "chi2", None)
    if callable(chi2):
        return float(chi2(params, resamp_idx))
    r = np.asarray(objective.residuals_at(resamp_idx)(params), dtype=float)
    return float(r @ r)


def _fmt_duration(seconds: float) -> str:
    """Human-friendly duration: us / ms / s / m s."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes)}m {secs:04.1f}s"


def _log_thread_budget(label: str, budget: ThreadBudget, *, in_force: bool) -> None:
    """Log the materialized thread budget beside what the runtimes report.

    The budget is what was asked for; ``observed`` is what the loaded native
    libraries actually say afterwards, which is the only way to catch a plan that
    did not take — an extension built without OpenMP, or a runtime
    ``threadpoolctl`` cannot see. ``in_force`` is False for pooled backends,
    where the limits live in the workers and this process's own numbers would be
    misleading; each worker logs its own at DEBUG instead.
    """
    logger.info(
        "%s\n%s\n  observed    : %s",
        label,
        budget.describe(),
        format_observed_threads()
        if in_force
        else "set per worker (see DEBUG); this process unpinned",
    )


def _fmt_fit_counts(result: Any, elapsed: float | None = None) -> str:
    """``iterations   (fun evals: X, jac evals: Y, ~T per fun eval)``.

    Reads the counters off an OptimizeResult; any counter the algorithm did
    not report (None) is skipped and iterations fall back to ``n/a``. When
    ``elapsed`` (seconds) is given and fun evals were counted, appends the
    approximate wall time per fun eval.
    """
    iters = getattr(result, "n_iterations", None)
    out = "n/a" if iters is None else str(iters)
    n_fun = getattr(result, "n_fun_evals", None)
    details = [
        f"{label}: {value}"
        for label, value in (
            ("fun evals", n_fun),
            ("jac evals", getattr(result, "n_jac_evals", None)),
        )
        if value is not None
    ]
    if elapsed is not None and n_fun:
        details.append(f"~{_fmt_duration(elapsed / n_fun)} per fun eval")
    if details:
        out += f"   ({', '.join(details)})"
    return out


def _format_params_table(
    params: dict[str, float], names: list[str], start: dict[str, float] | None = None
) -> str:
    """Aligned ``name | [start] | fitted`` table for logging the fit result."""
    width = max((len(n) for n in names), default=9)
    width = max(width, len("parameter"))
    if start is not None:
        header = f"  {'parameter':<{width}}   {'start':>15}   {'fitted':>15}"
    else:
        header = f"  {'parameter':<{width}}   {'fitted':>15}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for name in names:
        fitted = float(params[name])
        if start is not None:
            lines.append(f"  {name:<{width}}   {start[name]:>15.8g}   {fitted:>15.8g}")
        else:
            lines.append(f"  {name:<{width}}   {fitted:>15.8g}")
    return "\n".join(lines)


def _params_vector(result: Any, names: list[str]) -> np.ndarray:
    return np.array([float(result.params[name]) for name in names], dtype=float)


def _theory_vector(
    objective: Objective, params: dict[str, float], resamp_idx: int = 0
) -> np.ndarray | None:
    """Model prediction at fitted params for objectives that expose a model."""
    model = getattr(objective, "model", None)
    if model is None:
        return None
    return predict_model(model, params, resamp_idx)


@dataclass
class _ResampleWorker:
    """Picklable per-resampling fit callable (process backend safe).

    Returns a compact summary record per resample; with ``keep="full"`` it also
    attaches the native optimagic ``OptimizeResult`` (heavy — full iteration
    history — and picklable so it survives the process backend). Objectives with
    a model also attach one post-fit theory prediction.
    """

    objective: Objective
    param_set: ParamSetResolved
    minimizer: MinimizerConfig
    x0: dict[str, float]
    keep: KeepPolicy = "summary"

    def __call__(self, resamp_idx: int) -> dict[str, Any]:
        result = _run_minimize(self.objective, self.param_set, self.minimizer, resamp_idx, self.x0)
        record = {
            "params": _params_vector(result, self.param_set.names),
            "chi2": float(result.fun),
            "success": bool(result.success),
            "n_iterations": int(getattr(result, "n_iterations", 0) or 0),
            "message": str(result.message),
        }
        theory = _theory_vector(self.objective, result.params, resamp_idx)
        if theory is not None:
            record["theory"] = theory
        if self.keep == "full":
            record["result"] = result
        return record


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


class SamplingFit:
    """Drive optimagic fits over resamplings and package SigmondSampling output.

    Parameters
    ----------
    objective:
        Whitened-residual :class:`Objective` (e.g. :class:`LeastSquaresObjective`).
    param_set:
        Parameter description (start values, bounds, constraints).
    minimizer:
        optimagic algorithm selection; defaults to :class:`MinimizerConfig`
        (``scipy_lbfgsb``).
    sampling_info:
        Sampling metadata for the output samplings. Inferred from the objective's
        ``SamplingStats`` when available; required for custom objectives.
    ensemble_info:
        Ensemble tag attached to the output observables.
    """

    def __init__(
        self,
        objective: Objective,
        param_set: ParamSetResolved,
        *,
        minimizer: MinimizerConfig | None = None,
        sampling_info: SamplingInfo | None = None,
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
    ):
        self.objective = objective
        self.param_set = param_set
        self.minimizer = minimizer or MinimizerConfig()
        self.ensemble_info = ensemble_info
        if sampling_info is None:
            sampling_info = getattr(getattr(objective, "stats", None), "sampling_info", None)
            if sampling_info is None:
                raise ValueError(
                    "sampling_info could not be inferred from the objective; "
                    "pass sampling_info= explicitly for custom objectives"
                )
        self.sampling_info = sampling_info

    @classmethod
    def from_model(
        cls,
        stats: SamplingStats,
        model: Model,
        param_set: ParamSetResolved,
        *,
        minimizer: MinimizerConfig | None = None,
        use_correlation: bool = True,
        sampling_info: SamplingInfo | None = None,
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
    ) -> SamplingFit:
        """Build a fit from ``SamplingStats`` + a flat-vector :class:`Model`."""
        objective = LeastSquaresObjective(stats, model, use_correlation=use_correlation)
        return cls(
            objective,
            param_set,
            minimizer=minimizer,
            sampling_info=sampling_info,
            ensemble_info=ensemble_info,
        )

    # ------------------------------------------------------------------
    # single fit
    # ------------------------------------------------------------------

    def fit_one(
        self,
        resamp_idx: int = 0,
        *,
        x0: dict[str, float] | None = None,
        trace_path: str | None = None,
    ) -> Any:
        """Run one optimagic fit at ``resamp_idx`` and return its ``OptimizeResult``.

        ``trace_path`` is a path to an SQLite database for optimagic's full
        per-iteration trace (``None`` disables it). This is the only entry point
        that supports optimagic's per-iteration SQLite logging — the resample loop
        is deliberately trace-free.
        """
        return _run_minimize(
            self.objective, self.param_set, self.minimizer, resamp_idx, x0, trace_path
        )

    def _run_full_sample_fit(
        self, x0: dict[str, float] | None, *, trace_path: str | None = None
    ) -> Any:
        """Run the full-sample (index 0) fit with pretty, timed INFO logging.

        Evaluates and times the initial :math:`\\chi^2`, announces the fit, runs the
        minimize, then logs the elapsed time, final :math:`\\chi^2`, and an aligned
        start-vs-fitted parameter table. Shared by :meth:`fit_full_sample` and
        :meth:`fit_resampled` so both report the full-sample fit identically.
        """
        names = self.param_set.names
        start = dict(x0) if x0 is not None else self.param_set.start_values()

        t0 = time.perf_counter()
        chi2_0 = _chi2_at(self.objective, start, 0)
        eval_dt = time.perf_counter() - t0
        logger.info(
            "Starting full-sample fit\n"
            "  algorithm    : %s\n"
            "  parameters   : %d\n"
            "  initial chisq: %.6g   (residual eval: %s)",
            self.minimizer.algorithm,
            len(names),
            chi2_0,
            _fmt_duration(eval_dt),
        )

        t1 = time.perf_counter()
        full = _run_minimize(self.objective, self.param_set, self.minimizer, 0, x0, trace_path)
        fit_dt = time.perf_counter() - t1
        logger.info(
            "Full-sample fit complete in %s\n  final chisq  : %.6g\n  iterations   : %s\n%s",
            _fmt_duration(fit_dt),
            float(full.fun),
            _fmt_fit_counts(full, elapsed=fit_dt),
            _format_params_table(full.params, names, start=start),
        )
        return full

    # ------------------------------------------------------------------
    # full-sample + jacobian covariance
    # ------------------------------------------------------------------

    def fit_full_sample(
        self,
        *,
        x0: dict[str, float] | None = None,
        trace_path: str | None = None,
        rng: np.random.Generator | int | None = None,
        blas_threads: ThreadSetting = None,
        omp_threads: ThreadSetting = None,
        omp_width: int | None = None,
    ) -> FitResult:
        """Fit the full sample and synthesize resamples from the Jacobian covariance.

        Estimates the parameter covariance as ``inv(JᵀJ)`` of the whitened residual
        Jacobian (using ``result.jac`` when populated, else
        :func:`optimagic.first_derivative`), then draws pseudo-resamples from
        ``N(θ̂, cov)`` rescaled so the SigmondSampling error reproduces
        ``sqrt(diag(cov))`` for the configured sampling method. The theory band is
        obtained by **linear error propagation** through the model Jacobian (the
        pseudo-resamples were never fit, so the model is not re-evaluated at each
        one). The output is flagged :attr:`FitResult.is_synthetic`.

        Native threads are pinned for the whole central fit (the minimize and the
        Jacobian-covariance evaluations). There is no outer pool here — this is a
        single fit — so ``"auto"`` hands the whole machine to the objective:
        ``omp_threads`` up to ``omp_width`` (the natural width of its inner
        parallel region, e.g. a spectrum fit's block count) and the remaining
        cores to BLAS underneath each. ``None`` leaves that API untouched.
        """
        budget = plan_thread_budget(
            1,
            inner_cores=self.minimizer.n_opt_cores,
            omp_width=omp_width,
            num_workers=1,
            omp_threads=omp_threads,
            blas_threads=blas_threads,
        )
        with limit_native_threads(blas=budget.blas_threads, omp=budget.omp_threads):
            _log_thread_budget("Full-sample fit thread budget", budget, in_force=True)
            result = self._run_full_sample_fit(x0, trace_path=trace_path)
            names = self.param_set.names
            theta = _params_vector(result, names)
            free_names = [name for name, spec in self.param_set.values.items() if not spec.fixed]
            free_idx = [names.index(name) for name in free_names]

            cov, cov_source, jac, cov_singular = self._jacobian_covariance(
                result, names, free_names, free_idx
            )

            generator = np.random.default_rng(rng)
            n_resamp = self.sampling_info.num_resamplings
            factor = _error_factor(self.sampling_info.method, n_resamp)
            param_matrix = self._synthesize_param_matrix(
                theta, cov, free_idx, n_resamp, factor, generator
            )
            theory_matrix = self._linear_theory_matrix(result, theta, param_matrix, free_idx, jac)

            chisq_arr = np.full(n_resamp + 1, np.nan)
            chisq_arr[0] = float(result.fun)

            return self._build_result(
                optimize_result=result,
                param_matrix=param_matrix,
                chisq_arr=chisq_arr,
                cov_source=cov_source,
                residual_jacobian=jac,
                cov_singular=cov_singular,
                is_synthetic=True,
                n_fits=1,
                n_failed=0,
                failures={},
                records=None,
                theory_matrix=theory_matrix,
                trace_path=trace_path,
            )

    def _jacobian_covariance(
        self,
        result: Any,
        names: list[str],
        free_names: list[str],
        free_idx: list[int],
    ) -> tuple[np.ndarray, str, np.ndarray | None, bool]:
        """Parameter covariance ``inv(JᵀJ)`` over the free parameters.

        Returns ``(cov, source, jac, singular)`` — the whitened residual Jacobian
        ``jac`` is handed back so the caller can propagate the theory band without
        recomputing it, and store it on the result for degeneracy diagnostics.
        ``singular`` is ``True`` when ``JᵀJ`` could not be inverted and the
        covariance came from a pseudo-inverse. Fixed parameters get zero variance.
        Falls back to a numerical Jacobian when the optimagic result does not carry
        one (least-squares backends do not).
        """
        n_params = len(names)
        cov = np.zeros((n_params, n_params))
        if not free_idx:
            return cov, "none", None, False

        jac, source = self._free_jacobian(result, names, free_names)
        jtj = jac.T @ jac
        singular = False
        try:
            cov_free = np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov_free = np.linalg.pinv(jtj)
            singular = True
        # np.linalg.inv only raises on exact singularity, so a flat direction sails
        # through and returns a meaningless covariance; check the Jacobian's effective
        # rank as well and flag it on the result.
        sv = np.linalg.svd(jac, compute_uv=False)
        rank = int(np.count_nonzero(sv > RANK_RTOL * sv[0])) if sv[0] > 0 else 0
        if rank < len(free_names):
            singular = True
            logger.warning(
                "Jacobian is rank-deficient (%d of %d free parameters, cond(J)=%.3g) — "
                "the data does not constrain every parameter combination and the "
                "reported parameter covariance is unreliable.",
                rank,
                len(free_names),
                float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            )
        cov[np.ix_(free_idx, free_idx)] = cov_free
        return cov, source, jac, singular

    def _linear_theory_matrix(
        self,
        result: Any,
        theta: np.ndarray,
        param_matrix: np.ndarray,
        free_idx: list[int],
        jac: np.ndarray | None,
    ) -> np.ndarray | None:
        """Theory band by linear propagation through the model Jacobian.

        The synthesized pseudo-resamples were never fit, so rather than evaluating
        the model at each one we propagate linearly (consistent with the Gaussian
        Jacobian approximation): ``theory_j = m(θ̂) + G (θ_j − θ̂)``. The model
        Jacobian ``G = −W⁻¹ J`` is recovered from the whitened residual Jacobian
        ``J`` and the whitening operator, so the only model call is ``m(θ̂)``.
        Returns ``None`` for objectives without a model + whitening (the caller then
        falls back to direct per-column evaluation, or no theory at all).
        """
        model = getattr(self.objective, "model", None)
        whitening = getattr(self.objective, "whitening", None)
        if model is None or whitening is None:
            return None

        theta_dict = {name: float(result.params[name]) for name in self.param_set.names}
        theory_hat = predict_model(model, theta_dict, 0)
        theory_matrix = np.empty((theory_hat.size, param_matrix.shape[1]))
        theory_matrix[:, 0] = theory_hat
        if free_idx and jac is not None:
            g = -whitening.unwhiten(jac)  # (num_obs, n_free)
            dtheta = param_matrix[free_idx, 1:] - theta[free_idx][:, None]  # (n_free, n_resamp)
            theory_matrix[:, 1:] = theory_hat[:, None] + g @ dtheta
        else:
            theory_matrix[:, 1:] = theory_hat[:, None]
        return theory_matrix

    def _free_jacobian(
        self, result: Any, names: list[str], free_names: list[str]
    ) -> tuple[np.ndarray, str]:
        """Jacobian of whitened residuals w.r.t. free params at the optimum."""
        theta = {name: float(result.params[name]) for name in names}
        jac_attr = getattr(result, "jac", None)
        if jac_attr is not None:
            coerced = _coerce_jacobian(jac_attr, free_names)
            if coerced is not None:
                return coerced, "result.jac"

        import optimagic as om

        base = self.objective.residuals_at(0)

        def free_residuals(free_params: dict[str, float]) -> np.ndarray:
            full = dict(theta)
            full.update(free_params)
            return np.asarray(base(full), dtype=float)

        deriv = om.first_derivative(
            free_residuals, {name: theta[name] for name in free_names}
        ).derivative
        jac = np.column_stack([np.asarray(deriv[name], dtype=float) for name in free_names])
        return jac, "first_derivative"

    def _synthesize_param_matrix(
        self,
        theta: np.ndarray,
        cov: np.ndarray,
        free_idx: list[int],
        n_resamp: int,
        factor: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """``(n_params, n_resamp+1)`` matrix: column 0 the fit, rest synthesized."""
        n_params = theta.size
        param_matrix = np.empty((n_params, n_resamp + 1))
        param_matrix[:, 0] = theta
        resamples = np.tile(theta, (n_resamp, 1))  # (n_resamp, n_params)
        if free_idx:
            target = cov[np.ix_(free_idx, free_idx)] / (factor**2)
            draws = _draws_with_sample_cov(target, n_resamp, rng)
            resamples[:, free_idx] += draws
        param_matrix[:, 1:] = resamples.T
        return param_matrix

    # ------------------------------------------------------------------
    # full resample loop
    # ------------------------------------------------------------------

    def fit_resampled(
        self,
        *,
        x0: dict[str, float] | None = None,
        n_resamplings: int | None = None,
        keep: KeepPolicy = "summary",
        backend: FitBackend = "serial",
        num_workers: int | Literal["auto"] | None = "auto",
        blas_threads: ThreadSetting = None,
        omp_threads: ThreadSetting = "auto",
        omp_width: int | None = None,
        full_sample_omp_threads: ThreadSetting = "auto",
        mp_context: str | Any | None = "auto",
        progress: ProgressKind = True,
        error_policy: ErrorPolicy = "record",
        trace_path: str | None = None,
        worker_initializer: Any = None,
        worker_initargs: tuple = (),
    ) -> FitResult:
        """Fit the full sample, then fit every resampling from that initial value.

        The full-sample fit provides both the reported full-sample parameters and
        the shared ``x0`` for every resample fit (and is the only fit that may take
        ``trace_path``, optimagic's per-iteration SQLite trace). The resample loop
        carries no optimagic SQLite logging (process-unsafe and heavy); it records the compact
        per-resample summary instead. ``keep`` controls how much is retained on the
        result: ``"none"`` keeps only the assembled samplings; ``"summary"``
        (default) also keeps the compact per-resample records; ``"full"``
        additionally retains every resample's native optimagic ``OptimizeResult``
        (full iteration history) in :attr:`FitResult.optimize_results` — heavy, so
        opt in only when you need per-resample diagnostics/plots.

        Parallelism is budgeted in two phases, because the two have different
        amounts of it to spend. The full-sample prologue is a *single* fit with no
        outer pool, so ``full_sample_omp_threads="auto"`` gives it the whole
        machine — otherwise it is a one-core serial section in front of a parallel
        loop. The loop itself fills the outer pool first (``num_workers``), then
        spends the per-job remainder on the objective's OpenMP region up to
        ``omp_width`` and on BLAS underneath it. See
        :func:`~._execution.plan_thread_budget`.
        """
        backend = validate_backend(backend)
        names = self.param_set.names

        loop_budget = plan_thread_budget(
            n_resamplings if n_resamplings is not None else self.sampling_info.num_resamplings,
            inner_cores=self.minimizer.n_opt_cores,
            omp_width=omp_width,
            num_workers=num_workers,
            omp_threads=omp_threads,
            blas_threads=blas_threads,
        )
        prologue_budget = plan_thread_budget(
            1,
            inner_cores=self.minimizer.n_opt_cores,
            omp_width=omp_width,
            num_workers=1,
            omp_threads=full_sample_omp_threads,
            blas_threads=blas_threads,
        )

        # Resolve the start method up front: the hazard is forking *after* the
        # prologue has run an OpenMP region, so it is the prologue's width that
        # decides, not the loop's.
        context = resolve_mp_context(
            mp_context,
            uses_openmp=max(prologue_budget.omp_threads or 1, loop_budget.omp_threads or 1) > 1,
        )

        with limit_native_threads(
            blas=prologue_budget.blas_threads, omp=prologue_budget.omp_threads
        ):
            _log_thread_budget(
                "Full-sample (prologue) thread budget", prologue_budget, in_force=True
            )
            full = self._run_full_sample_fit(x0, trace_path=trace_path)
        x0_vec = _params_vector(full, names)
        x0_dict = {name: float(full.params[name]) for name in names}

        n_total = self.sampling_info.num_resamplings
        n_resamp = n_total if n_resamplings is None else int(n_resamplings)
        if not 1 <= n_resamp <= n_total:
            raise ValueError(f"n_resamplings must be in [1, {n_total}], got {n_resamplings!r}")

        param_matrix = np.empty((len(names), n_resamp + 1))
        param_matrix[:, 0] = x0_vec
        chisq_arr = np.full(n_resamp + 1, np.nan)
        chisq_arr[0] = float(full.fun)
        theory_matrix = None
        full_theory = _theory_vector(self.objective, full.params, 0)
        if full_theory is not None:
            theory_matrix = np.empty((full_theory.size, n_resamp + 1))
            theory_matrix[:, 0] = full_theory
        records: list[dict[str, Any]] = []
        results: dict[int, Any] = {0: full}
        failures: dict[int, str] = {}

        worker = _ResampleWorker(self.objective, self.param_set, self.minimizer, x0_dict, keep)
        workers = loop_budget.workers
        indices = range(1, n_resamp + 1)

        logger.info(
            "Resample loop starting\n"
            "  resamples : %d\n"
            "  backend   : %s\n"
            "  budget    : %s\n"
            "  keep      : %s",
            n_resamp,
            backend
            if backend == "serial"
            else f"{backend} ({context.get_start_method() if context is not None else 'default'})",
            loop_budget.summary(),
            keep,
        )
        _log_thread_budget(
            "Resample loop thread budget", loop_budget, in_force=(backend == "serial")
        )

        t0 = time.perf_counter()
        # Pooled backends pin their own workers in the initializer; the serial
        # backend runs the jobs on this thread, so it is pinned here instead.
        with limit_native_threads(
            blas=loop_budget.blas_threads if backend == "serial" else None,
            omp=loop_budget.omp_threads if backend == "serial" else None,
        ):
            with executor_context(
                backend,
                max_workers=workers,
                worker_initializer=worker_initializer,
                worker_initargs=worker_initargs,
                blas_threads=loop_budget.blas_threads,
                omp_threads=loop_budget.omp_threads,
                mp_context=context,
            ) as executor:
                for idx, value, error in run_jobs(
                    worker,
                    indices,
                    executor=executor,
                    progress=progress,
                    error_policy=error_policy,
                    desc="Resample fits",
                ):
                    self._record_resample(
                        idx,
                        value,
                        error,
                        x0_vec=x0_vec,
                        full_theory=full_theory,
                        param_matrix=param_matrix,
                        theory_matrix=theory_matrix,
                        chisq_arr=chisq_arr,
                        records=records,
                        results=results,
                        failures=failures,
                        keep=keep,
                    )
                    logger.debug(
                        "resample %d/%d done%s",
                        idx,
                        n_resamp,
                        "" if error is None else f" FAILED: {error}",
                    )
        loop_dt = time.perf_counter() - t0

        if failures:
            logger.warning("%d/%d resample fits failed: %s", len(failures), n_resamp, failures)
        rate = n_resamp / loop_dt if loop_dt > 0 else float("inf")
        logger.info(
            "Resample loop complete in %s\n"
            "  succeeded : %d\n"
            "  failed    : %d\n"
            "  throughput: %.1f fits/s",
            _fmt_duration(loop_dt),
            n_resamp - len(failures),
            len(failures),
            rate,
        )

        return self._build_result(
            optimize_result=full,
            param_matrix=param_matrix,
            chisq_arr=chisq_arr,
            cov_source=None,
            is_synthetic=False,
            n_fits=n_resamp,
            n_failed=len(failures),
            failures=failures,
            records=records if keep in ("summary", "full") else None,
            optimize_results=results if keep == "full" else None,
            theory_matrix=theory_matrix,
        )

    def _record_resample(
        self,
        idx: int,
        value: dict[str, Any] | None,
        error: str | None,
        *,
        x0_vec: np.ndarray,
        full_theory: np.ndarray | None,
        param_matrix: np.ndarray,
        theory_matrix: np.ndarray | None,
        chisq_arr: np.ndarray,
        records: list[dict[str, Any]],
        results: dict[int, Any],
        failures: dict[int, str],
        keep: KeepPolicy,
    ) -> None:
        """Store one resample outcome into the running matrices/records."""
        if error is not None or value is None:
            param_matrix[:, idx] = x0_vec
            if theory_matrix is not None and full_theory is not None:
                theory_matrix[:, idx] = full_theory
            failures[idx] = error or "fit returned no result"
            return
        param_matrix[:, idx] = value["params"]
        chisq_arr[idx] = value["chi2"]
        theory = value.pop("theory", None)
        if theory_matrix is not None:
            theory_matrix[:, idx] = theory if theory is not None else full_theory
        if not value["success"]:
            failures[idx] = value["message"]
        # keep the heavy OptimizeResult only in `results`, out of the summary record
        result = value.pop("result", None)
        if keep == "full" and result is not None:
            results[idx] = result
        if keep in ("summary", "full"):
            records.append({"resamp_idx": idx, **value})

    # ------------------------------------------------------------------
    # output packaging
    # ------------------------------------------------------------------

    def _output_sampling_info(self, n_resamp: int) -> SamplingInfo:
        """SamplingInfo for the output samplings with the actual resample count."""
        base = self.sampling_info
        return SamplingInfo(
            method=base.method,
            num_resamplings=n_resamp,
            seed=base.seed,
            boot_skip=base.boot_skip,
            **base.extra_params,
        )

    def _build_result(
        self,
        *,
        optimize_result: Any,
        param_matrix: np.ndarray,
        chisq_arr: np.ndarray,
        cov_source: str | None,
        is_synthetic: bool,
        n_fits: int,
        n_failed: int,
        failures: dict[int, str],
        records: list[dict[str, Any]] | None,
        optimize_results: dict[int, Any] | None = None,
        theory_matrix: np.ndarray | None = None,
        trace_path: str | None = None,
        residual_jacobian: np.ndarray | None = None,
        cov_singular: bool = False,
    ) -> FitResult:
        names = self.param_set.names
        free_names = [name for name, spec in self.param_set.values.items() if not spec.fixed]
        n_resamp = param_matrix.shape[1] - 1
        out_info = self._output_sampling_info(n_resamp)

        samplings: list[SigmondSampling] = []
        for i, (name, spec) in enumerate(self.param_set.values.items()):
            obs = spec.observable_info(name)
            samplings.append(
                SigmondSampling(
                    data=param_matrix[i],
                    observable_info=obs,
                    sampling_info=out_info,
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
            sampling_info=out_info,
            is_complex=False,
        )

        param_stats = SamplingStats(samplings)
        plotter = SamplingPlotter(param_stats)
        theory = (
            self._theory_stats_from_matrix(theory_matrix, out_info)
            if theory_matrix is not None
            else self._theory_stats(param_matrix, out_info)
        )

        return FitResult(
            optimize_result=optimize_result,
            params=param_stats,
            chi_squared=chi_squared,
            theory=theory,
            plotter=plotter,
            param_names=names,
            num_data_residuals=int(self.objective.num_data_residuals),
            num_free_params=int(getattr(optimize_result, "n_free", len(names))),
            cov_source=cov_source,
            free_param_names=free_names,
            residual_jacobian=residual_jacobian,
            cov_singular=cov_singular,
            is_synthetic=is_synthetic,
            n_fits=n_fits,
            n_failed=n_failed,
            failures=failures,
            records=records,
            optimize_results=optimize_results,
            trace_path=trace_path,
        )

    def _theory_stats(
        self, param_matrix: np.ndarray, out_info: SamplingInfo
    ) -> SamplingStats | None:
        """Model predictions at every column of ``param_matrix`` as a ``SamplingStats``.

        Evaluates the model at each column's parameters (column 0 the full sample,
        the rest the resamples). Returns ``None`` for objectives without a model
        (e.g. a raw residual function).
        """
        model = getattr(self.objective, "model", None)
        if model is None or getattr(self.objective, "stats", None) is None:
            return None
        names = self.param_set.names
        columns = [
            predict_model(
                model,
                {name: param_matrix[i, j] for i, name in enumerate(names)},
                j,
            )
            for j in range(param_matrix.shape[1])
        ]
        return self._theory_stats_from_matrix(np.column_stack(columns), out_info)

    def _theory_stats_from_matrix(
        self, theory_matrix: np.ndarray, out_info: SamplingInfo
    ) -> SamplingStats:
        """Wrap an ``(num_obs, n_resamp+1)`` theory matrix as a ``SamplingStats``.

        Observable infos are copied from the fitted data so the theory band aligns
        with the data observables; falls back to generic ``theory_i`` names if the
        model's output length differs from the data.
        """
        data_stats = getattr(self.objective, "stats", None)
        data_infos = [s.observable_info for s in data_stats] if data_stats is not None else []
        n_obs = theory_matrix.shape[0]
        if len(data_infos) == n_obs:
            infos = [info.copy() for info in data_infos]
        else:
            infos = [
                ObservableInfo(name=f"theory_{i}", index=i, ensemble_info=self.ensemble_info)
                for i in range(n_obs)
            ]
        return SamplingStats(
            [
                SigmondSampling(theory_matrix[i], infos[i], out_info, is_complex=False)
                for i in range(n_obs)
            ]
        )


def _coerce_jacobian(jac: Any, free_names: list[str]) -> np.ndarray | None:
    """Coerce an optimagic ``result.jac`` to an ``(n_obs, n_free)`` matrix or ``None``.

    optimagic may store the Jacobian as a ``{name: column}`` pytree or as a dense
    array; anything that does not match the free-parameter layout is rejected so the
    caller falls back to a fresh numerical Jacobian.
    """
    if isinstance(jac, dict):
        try:
            return np.column_stack(
                [np.asarray(jac[name], dtype=float).reshape(-1) for name in free_names]
            )
        except (KeyError, ValueError):
            return None
    arr = np.asarray(jac, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == len(free_names):
        return arr
    if arr.ndim == 2 and arr.shape[0] == len(free_names):
        return arr.T
    return None
