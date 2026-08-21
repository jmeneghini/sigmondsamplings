"""Result object for the optimagic-native resampling fit driver.

:class:`FitResult` wraps the native optimagic :class:`~optimagic.optimization.optimize_result.OptimizeResult`
of the full-sample fit together with the SigmondSamplings produced for every fitted
parameter (and :math:`\\chi^2`). It keeps the framework's
:class:`~sigmondsamplings.sampling.SigmondSampling` /
:class:`~sigmondsamplings.observable_collection.ObservableCollection` outputs so the
existing plotting and error-propagation machinery (``SamplingPlotter``,
``SigmondModelFunc``) works unchanged, while exposing optimagic's own diagnostics
(``criterion_plot`` / ``params_plot`` / ``convergence_report``) as thin passthroughs.

Two driver entry points build one of these:

* :meth:`SamplingFit.fit_full_sample <sigmondsamplings.fitting.fit.SamplingFit.fit_full_sample>`
  — the resamples are *synthesized* from the Jacobian covariance, flagged via
  :attr:`is_synthetic`.
* :meth:`SamplingFit.fit_resampled <sigmondsamplings.fitting.fit.SamplingFit.fit_resampled>`
  — the resamples are genuine per-resampling fits.

The Jacobian path also keeps the whitened residual Jacobian itself
(:attr:`FitResult.residual_jacobian`), which is what the degeneracy diagnostics
(:attr:`~FitResult.condition_number`, :attr:`~FitResult.rank`) are computed from.

Either way :attr:`params` is a :class:`~sigmondsamplings.stats.SamplingStats` over
the fitted parameters, so the parameter covariance and per-parameter errors come
straight from that object (:attr:`cov` / :attr:`standard_errors`) rather than being
recomputed here — keeping them consistent with the rest of the framework.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2 as _chi2_dist

if TYPE_CHECKING:
    from ..plotting import SamplingPlotter
    from ..sampling import SigmondSampling
    from ..stats import SamplingStats

__all__ = ["FitResult"]

#: Relative singular-value cut used by :attr:`FitResult.rank`. Chosen well above the
#: ~1e-8 relative noise floor of a finite-difference Jacobian (numpy's default
#: ``matrix_rank`` tolerance sits at machine epsilon and so never fires on one), and
#: well below any direction the data meaningfully constrains: at a ratio of 1e-6 the
#: ``JᵀJ`` inversion behind the covariance has already lost ~12 of 16 digits.
RANK_RTOL = 1e-6


@dataclass
class _LoadedOptimizeResult:
    """Minimal optimize_result stub for a FitResult reconstructed from HDF5."""

    fun: float
    params: dict[str, float]
    success: bool = True
    n_free: int = 0
    convergence_report: Any = None


@dataclass
class FitResult:
    """Output of the optimagic-native :class:`SamplingFit` driver.

    Holds the native optimagic result of the full-sample fit alongside a
    :class:`~sigmondsamplings.stats.SamplingStats` of the fitted parameters and a
    :math:`\\chi^2` sampling.
    """

    #: Native optimagic result of the full-sample (``resamp_idx=0``) fit.
    optimize_result: Any
    #: Fitted-parameter samplings as a ``SamplingStats`` (covariance, errors, …).
    params: SamplingStats
    #: :math:`\\chi^2` sampling (index 0 is the full sample; resamples NaN when
    #: synthesized).
    chi_squared: SigmondSampling
    #: Model predictions at each resampling's best fit, aligned with the fitted
    #: data observables (the fit band). ``None`` for objectives without a model.
    theory: SamplingStats | None
    #: Plotter bound to the parameter samplings.
    plotter: SamplingPlotter
    #: Best-fit parameter names in declared order.
    param_names: list[str]
    #: Number of residuals (observables) the fit was over — drives dof.
    num_data_residuals: int
    #: Number of free parameters after constraints (optimagic ``n_free``).
    num_free_params: int
    #: Where a synthesized covariance came from: ``"result.jac"`` or
    #: ``"first_derivative"`` (jacobian/synthetic results only).
    cov_source: str | None = None
    #: Names of the free (non-fixed) parameters in declared order. This is the
    #: column order of :attr:`residual_jacobian`; :attr:`param_names` differs in
    #: that it also lists fixed parameters.
    free_param_names: list[str] = field(default_factory=list)
    #: Jacobian of the **whitened residuals** w.r.t. the free parameters at the
    #: optimum, shape ``(num_data_residuals, len(free_param_names))``.
    #:
    #: This is ``∂r/∂θ``, not the model Jacobian: the two differ by the
    #: whitening and a sign, ``G = -whitening.unwhiten(J)``. Populated only by the
    #: Jacobian/synthetic path
    #: (:meth:`SamplingFit.fit_full_sample <sigmondsamplings.fitting.fit.SamplingFit.fit_full_sample>`),
    #: since that is where it is computed; ``None`` for genuine resampled fits and
    #: for results reconstructed by :meth:`from_hdf5` (it is not written to HDF5).
    residual_jacobian: np.ndarray | None = None
    #: ``True`` when the Jacobian was rank-deficient at :data:`RANK_RTOL` (or
    #: ``JᵀJ`` outright failed to invert), meaning :attr:`cov` is not trustworthy:
    #: at least one parameter combination is unconstrained by the data.
    cov_singular: bool = False
    #: ``True`` when the resamples are synthesized Gaussian draws, not real fits.
    is_synthetic: bool = False
    #: Number of resample fits attempted (resampled mode).
    n_fits: int = 0
    #: Number of resample fits that failed/did not converge (resampled mode).
    n_failed: int = 0
    #: ``{resamp_idx: message}`` for failed resample fits.
    failures: dict[int, str] = field(default_factory=dict)
    #: Optional per-resample summary records (``keep="summary"`` or ``"full"``).
    records: list[dict[str, Any]] | None = None
    #: Native optimagic ``OptimizeResult`` per resampling index, including the
    #: full sample at ``0`` (only with ``keep="full"``; ``None`` otherwise).
    optimize_results: dict[int, Any] | None = None
    #: Path to the optimagic SQLite log for the full-sample fit (set when
    #: ``fit_one``/``fit_full_sample`` was called with ``trace_path=<path>``).
    trace_path: str | Path | None = None
    log_reader: Any = (
        None  #: optimagic SQLiteLogReader for the full-sample fit (if trace_path is set)
    )

    # ------------------------------------------------------------------
    # scalar fit diagnostics
    # ------------------------------------------------------------------

    @property
    def best_params(self) -> dict[str, float]:
        """Full-sample best-fit parameters as a name->value dict."""
        return {name: float(self.optimize_result.params[name]) for name in self.param_names}

    @property
    def best_theory(self) -> np.ndarray | None:
        """Full-sample best-fit theory vector (``None`` without a model)."""
        if self.theory is None:
            return None
        return np.asarray(self.theory.array[:, 0], dtype=float)

    @property
    def chi2(self) -> float:
        """Full-sample :math:`\\chi^2` (optimagic criterion at the optimum)."""
        return float(self.optimize_result.fun)

    @property
    def dof(self) -> int:
        """Degrees of freedom: residuals minus free parameters."""
        return int(self.num_data_residuals - self.num_free_params)

    @property
    def chi2_per_dof(self) -> float:
        """:math:`\\chi^2/\\mathrm{dof}` (``nan`` when dof <= 0)."""
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    @property
    def cov(self) -> np.ndarray:
        """Parameter covariance from the fitted-parameter ``SamplingStats``.

        For synthetic (jacobian) results this reproduces ``inv(JᵀJ)``; for genuine
        resample fits it is the covariance of the per-resampling best fits. Either
        way the sampling-method scaling matches :attr:`standard_errors`.
        """
        return self.params.cov_matrix

    @property
    def standard_errors(self) -> dict[str, float]:
        """Per-parameter standard errors from the fitted-parameter samplings."""
        errors = np.asarray(self.params.val.error, dtype=float)
        return dict(zip(self.param_names, errors.tolist()))

    # ------------------------------------------------------------------
    # Jacobian diagnostics
    # ------------------------------------------------------------------

    @property
    def singular_values(self) -> np.ndarray | None:
        """Singular values of :attr:`residual_jacobian`, descending.

        ``None`` when no Jacobian was stored (resampled fits, HDF5 round-trips).
        """
        if self.residual_jacobian is None:
            return None
        jac = np.asarray(self.residual_jacobian, dtype=float)
        return np.linalg.svd(jac, compute_uv=False)

    @property
    def condition_number(self) -> float | None:
        """2-norm condition number of :attr:`residual_jacobian` (``None`` when absent).

        ``inf`` for an exactly rank-deficient Jacobian. Note the covariance is formed
        by inverting ``JᵀJ``, whose condition number is the *square* of this, so a
        value approaching ``1/sqrt(eps)`` (~1e8) already means the parameter errors
        carry little precision. A large value indicates a flat direction — a
        parameter combination the data does not constrain.
        """
        sv = self.singular_values
        if sv is None or sv.size == 0:
            return None
        smallest = float(sv[-1])
        if smallest <= 0.0:
            return float("inf")
        return float(sv[0]) / smallest

    @property
    def rank(self) -> int | None:
        """Effective rank of :attr:`residual_jacobian` (``None`` when absent).

        Counts singular values above ``RANK_RTOL * s_max`` rather than using numpy's
        machine-epsilon default, which never fires on a finite-difference Jacobian —
        see :data:`RANK_RTOL`. Use :attr:`singular_values` directly to apply your own
        cut. A rank below ``len(free_param_names)`` means the data does not constrain
        every parameter combination and :attr:`cov` should not be trusted (see
        :attr:`cov_singular`).
        """
        sv = self.singular_values
        if sv is None or sv.size == 0:
            return None
        if sv[0] <= 0.0:
            return 0
        return int(np.count_nonzero(sv > RANK_RTOL * sv[0]))

    def goodness_of_fit(self) -> float:
        """Q-value: survival function of :math:`\\chi^2` at :attr:`dof`."""
        return float(_chi2_dist.sf(self.chi2, self.dof)) if self.dof > 0 else float("nan")

    def aic(self) -> float:
        """Akaike Information Criterion: ``chi2 - 2*dof``."""
        return self.chi2 - 2 * self.dof

    def bic(self) -> float:
        """Bayesian Information Criterion: ``chi2 + nparams*ln(n_obs)``."""
        return self.chi2 + self.num_free_params * np.log(self.num_data_residuals)

    def aicc(self) -> float:
        """Corrected AIC: ``aic + 2k(k+1)/(n-k-1)`` (``nan`` if undefined)."""
        k = self.num_free_params
        n = self.num_data_residuals
        denom = n - k - 1
        if denom <= 0:
            return float("nan")
        return self.aic() + 2 * k * (k + 1) / denom

    # ------------------------------------------------------------------
    # framework integration
    # ------------------------------------------------------------------

    def param(self, name: str) -> SigmondSampling:
        """Return a fitted-parameter sampling by name."""
        return self.params.find(name=name)

    def model_func(
        self,
        func: Callable,
        latex_str: str | None = None,
        independent_var_latex: str | None = None,
    ):
        """Build a :class:`~sigmondsamplings.fitting.model_func.SigmondModelFunc` for plotting/propagation."""
        from ..observable_collection import ObservableCollection
        from .model_func import SigmondModelFunc

        return SigmondModelFunc(
            func,
            ObservableCollection(list(self.params)),
            latex_str=latex_str,
            independent_var_latex=independent_var_latex,
        )

    def summary(self) -> dict[str, Any]:
        """Compact dict of the scalar diagnostics for logging/printing."""
        return {
            "params": self.best_params,
            "standard_errors": self.standard_errors,
            "chi2": self.chi2,
            "dof": self.dof,
            "chi2_per_dof": self.chi2_per_dof,
            "Q": self.goodness_of_fit(),
            "aic": self.aic(),
            "bic": self.bic(),
            "aicc": self.aicc(),
            "success": bool(self.optimize_result.success),
            "n_fits": self.n_fits,
            "n_failed": self.n_failed,
            "cov_source": self.cov_source,
            "cov_singular": self.cov_singular,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "is_synthetic": self.is_synthetic,
        }

    # ------------------------------------------------------------------
    # write outputs
    # ------------------------------------------------------------------

    def write_params(
        self, filename: str | Path, group: str = "params", overwrite: bool = False
    ) -> None:
        """Write fitted parameter samplings to an HDF5 file."""
        from ..observable_collection import ObservableCollection

        ObservableCollection(list(self.params)).to_hdf5(
            str(filename), group=group, overwrite=overwrite, create_backups=False
        )

    def write_chi2(
        self, filename: str | Path, group: str = "chi2", overwrite: bool = False
    ) -> None:
        """Write the chi-squared sampling to an HDF5 file."""
        from ..observable_collection import ObservableCollection

        ObservableCollection([self.chi_squared]).to_hdf5(
            str(filename), group=group, overwrite=overwrite, create_backups=False
        )

    def write_theory(
        self, filename: str | Path, group: str = "theory", overwrite: bool = False
    ) -> None:
        """Write theory-prediction samplings to an HDF5 file.

        Multi-ensemble theory (from multi-ensemble fits) writes each ensemble
        to a separate group via :meth:`~MultiEnsembleCollection.to_hdf5`.
        Raises ``ValueError`` when no theory is available.
        """
        if self.theory is None:
            raise ValueError("No theory predictions available (objective has no model).")
        self.theory.to_hdf5(str(filename), group=group, overwrite=overwrite, create_backups=False)

    def write_jacobian(
        self, filename: str | Path, group: str = "jacobian", overwrite: bool = False
    ) -> None:
        """Write :attr:`residual_jacobian` and its labels to an HDF5 file.

        Stored as a plain ``(num_residuals, n_free)`` dataset in its own group rather
        than as a Sigmond observable group — it is a derivative, not a sampling. Since
        nothing lands under ``Values/``, :class:`~sigmondsamplings.io.loader.SigmondLoader`
        never sees it, so this can share a file with the ``params``/``chi2``/``theory``
        groups written by the other ``write_*`` methods. :attr:`free_param_names`,
        :attr:`cov_source` and :attr:`cov_singular` ride along as attributes so the
        columns stay labelled and the degeneracy diagnostics survive the round trip.

        Raises ``ValueError`` when the result carries no Jacobian — resampled fits, or
        a result already reconstructed from HDF5.
        """
        import h5py

        if self.residual_jacobian is None:
            raise ValueError(
                "No residual Jacobian available; only fit_full_sample() results carry "
                "one (resampled fits and HDF5 round-trips do not)."
            )
        jac = np.asarray(self.residual_jacobian, dtype=float)
        with h5py.File(str(filename), "a") as handle:
            if group in handle:
                if not overwrite:
                    raise ValueError(
                        f"group {group!r} already exists in {str(filename)!r}; "
                        "pass overwrite=True to replace it"
                    )
                del handle[group]
            grp = handle.create_group(group)
            dset = grp.create_dataset("Jacobian", data=jac)
            dset.attrs["FreeParamNames"] = np.array(
                list(self.free_param_names), dtype=h5py.string_dtype()
            )
            dset.attrs["CovSource"] = self.cov_source or ""
            dset.attrs["CovSingular"] = bool(self.cov_singular)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        W = 42
        lines = ["=" * W, "  FitResult", "=" * W]
        for name in self.param_names:
            s = self.param(name)
            lines.append(f"  {name:<12} = {s.pdg_format()}")
        lines.append("")
        lines.append(f"  {'chi2':<12}: {self.chi2:.6g}")
        lines.append(f"  {'dof':<12}: {self.dof:d}")
        lines.append(f"  {'chi2/dof':<12}: {self.chi2_per_dof:.4g}")
        lines.append(f"  {'Q':<12}: {self.goodness_of_fit():.4g}")
        lines.append(f"  {'AIC':<12}: {self.aic():.6g}")
        lines.append(f"  {'BIC':<12}: {self.bic():.6g}")
        lines.append(f"  {'AICc':<12}: {self.aicc():.6g}")
        lines.append(f"  {'n_fits':<12}: {self.n_fits:d}")
        lines.append(f"  {'n_failed':<12}: {self.n_failed:d}")
        lines.append(f"  {'is_synthetic':<12}: {self.is_synthetic}")
        cond = self.condition_number
        if cond is not None:
            n_cols = int(np.asarray(self.residual_jacobian).shape[1])
            lines.append(f"  {'cond(J)':<12}: {cond:.4g}")
            lines.append(f"  {'rank(J)':<12}: {self.rank:d}/{n_cols:d}")
            if self.cov_singular:
                lines.append("  (JᵀJ singular — covariance from pseudo-inverse)")
        lines.append("=" * W)
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        param_lines = ["| Parameters |", "| --- |"]
        for name in self.param_names:
            s = self.param(name)
            obs_latex = s.observable_info.latex_str.strip("$")
            param_lines.append(f"| ${obs_latex}$ = {s.pdg_format()} |")

        diag_lines = [
            "| Statistic | Value |",
            "| --- | --- |",
            rf"| $\chi^2$ | {self.chi2:.6g} |",
            f"| dof | {self.dof:d} |",
            rf"| $\chi^2/\text{{dof}}$ | {self.chi2_per_dof:.4g} |",
            f"| Q | {self.goodness_of_fit():.4g} |",
            f"| AIC | {self.aic():.6g} |",
            f"| BIC | {self.bic():.6g} |",
            f"| AICc | {self.aicc():.6g} |",
            f"| n\\_fits | {self.n_fits:d} |",
            f"| n\\_failed | {self.n_failed:d} |",
            f"| is\\_synthetic | {self.is_synthetic} |",
        ]
        cond = self.condition_number
        if cond is not None:
            n_cols = int(np.asarray(self.residual_jacobian).shape[1])
            diag_lines.append(f"| cond($J$) | {cond:.4g} |")
            diag_lines.append(f"| rank($J$) | {self.rank:d}/{n_cols:d} |")
        return "\n".join(param_lines) + "\n\n" + "\n".join(diag_lines)

    # ------------------------------------------------------------------
    # optimagic diagnostics passthroughs (full-sample fit)
    # ------------------------------------------------------------------

    def _plot_with_res_or_log(self, plot_func: Callable, **kwargs):
        """Call an optimagic plot function with either the log reader or the optimize_result."""
        if isinstance(self.optimize_result, _LoadedOptimizeResult):
            if self.trace_path is not None:
                return plot_func(self.trace_path, **kwargs)
            else:
                raise RuntimeError(
                    "Cannot plot optimagic diagnostics: no log reader available "
                    "(fit was reconstructed from HDF5 without a trace_path)."
                )
        else:
            return plot_func(self.optimize_result, **kwargs)

    def criterion_plot(self, **kwargs):
        """optimagic ``criterion_plot`` of the full-sample fit history."""
        import optimagic as om

        return self._plot_with_res_or_log(om.criterion_plot, **kwargs)

    def params_plot(self, **kwargs):
        """optimagic ``params_plot`` of the full-sample fit history."""
        import optimagic as om

        return self._plot_with_res_or_log(om.params_plot, **kwargs)

    @property
    def convergence_report(self):
        """optimagic convergence report of the full-sample fit."""
        return self.optimize_result.convergence_report

    # ------------------------------------------------------------------
    # round-trip from HDF5
    # ------------------------------------------------------------------

    @classmethod
    def from_hdf5(
        cls,
        params_file: str | Path,
        params_group: str = "params",
        chi2_file: str | Path | None = None,
        chi2_group: str = "chi2",
        theory_file: str | Path | None = None,
        theory_group: str = "theory",
        jacobian_file: str | Path | None = None,
        jacobian_group: str = "jacobian",
        trace_path: str | Path | None = None,
        num_data_residuals: int = 0,
        **kwargs: Any,
    ) -> FitResult:
        """Reconstruct a :class:`FitResult` from HDF5 files written by the write methods.

        Args:
            params_file: HDF5 file written by :meth:`write_params`.
            params_group: Group inside ``params_file`` (default ``"params"``).
            chi2_file: HDF5 file written by :meth:`write_chi2` (optional).
            chi2_group: Group inside ``chi2_file`` (default ``"chi2"``).
            theory_file: HDF5 file written by :meth:`write_theory` (optional).
            theory_group: Group inside ``theory_file`` (default ``"theory"``).
            jacobian_file: HDF5 file written by :meth:`write_jacobian` (optional).
                Without it :attr:`residual_jacobian` — and so the degeneracy
                diagnostics — stay ``None``, since nothing else persists it.
            jacobian_group: Group inside ``jacobian_file`` (default ``"jacobian"``).
            trace_path: Path to the optimagic SQLite log (from a ``trace_path=`` call).
                When provided the final recorded criterion and parameters are used
                to populate the reconstructed ``optimize_result``.
            num_data_residuals: Number of data residuals (observables) used in the
                original fit — required to compute :attr:`dof` correctly.
            **kwargs: Additional keyword arguments forwarded to the :class:`FitResult`
                constructor (e.g. ``is_synthetic``, ``n_fits``, ``n_failed``).
        """
        from ..io.loader import SigmondLoader
        from ..plotting import SamplingPlotter
        from ..stats import SamplingStats

        # --- parameters ---
        param_loader = SigmondLoader(filename=str(params_file), group=params_group)
        param_samplings = list(param_loader.observables)
        param_names = [s.observable_info.name for s in param_samplings]
        params_stats = SamplingStats(param_samplings)

        # --- chi-squared ---
        if chi2_file is not None:
            chi2_loader = SigmondLoader(filename=str(chi2_file), group=chi2_group)
            chi2_samplings = list(chi2_loader.observables)
            if len(chi2_samplings) != 1:
                raise ValueError(
                    f"Expected exactly 1 chi2 sampling in {chi2_file!r}, got {len(chi2_samplings)}"
                )
            chi_squared = chi2_samplings[0]
            chi2_val = float(chi_squared.full_sample_value)
        else:
            # stub: zero-filled chi2 using the first param's sampling info
            from ..info import ObservableInfo
            from ..sampling import SigmondSampling

            ref = param_samplings[0]
            chi_squared = SigmondSampling(
                np.zeros_like(ref.data),
                ObservableInfo("chi_squared", 0, "n", "re", ref.observable_info.ensemble_info),
                ref.sampling_info,
            )
            chi2_val = 0.0

        # --- theory ---
        theory = None
        if theory_file is not None:
            theory_loader = SigmondLoader(filename=str(theory_file), group=theory_group)
            theory = SamplingStats(list(theory_loader.observables))

        # --- residual Jacobian ---
        if jacobian_file is not None:
            import h5py

            with h5py.File(str(jacobian_file), "r") as handle:
                dset = handle[f"{jacobian_group}/Jacobian"]
                kwargs.setdefault("residual_jacobian", np.asarray(dset, dtype=float))
                free_names = dset.attrs.get("FreeParamNames")
                if free_names is not None:
                    kwargs.setdefault(
                        "free_param_names",
                        [n.decode() if isinstance(n, bytes) else str(n) for n in free_names],
                    )
                source = dset.attrs.get("CovSource")
                if source:
                    source = source.decode() if isinstance(source, bytes) else str(source)
                    kwargs.setdefault("cov_source", source)
                if "CovSingular" in dset.attrs:
                    kwargs.setdefault("cov_singular", bool(dset.attrs["CovSingular"]))

        # --- optimize_result stub (optionally populated from SQLite log) ---
        best: dict[str, float] = {
            name: float(s.full_sample_value) for name, s in zip(param_names, param_samplings)
        }
        if trace_path is not None:
            try:
                import optimagic as om

                reader = om.SQLiteLogReader(str(trace_path))
                # use the final recorded func val and params
                last_iter = reader.read_iteration(-1)
                chi2_val = last_iter.scalar_fun
                last_params = last_iter.params
                best.update({k: float(v) for k, v in last_params.items() if k in best})
                log_reader = reader  # store for later access

            except Exception:
                pass  # log read is best-effort
        else:
            log_reader = None

        opt_result = _LoadedOptimizeResult(
            fun=chi2_val,
            params=best,
            n_free=len(param_names),
        )

        # every sampling read back is treated as a free parameter; no Jacobian is
        # persisted, so the degeneracy diagnostics are unavailable on reload
        kwargs.setdefault("free_param_names", list(param_names))

        return cls(
            optimize_result=opt_result,
            params=params_stats,
            chi_squared=chi_squared,
            theory=theory,
            plotter=SamplingPlotter(params_stats),
            param_names=param_names,
            num_data_residuals=num_data_residuals,
            num_free_params=len(param_names),
            trace_path=trace_path,
            log_reader=log_reader,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"FitResult(n_params={len(self.param_names)}, "
            f"chi2={self.chi2:.6g}, dof={self.dof}, "
            f"is_synthetic={self.is_synthetic}, n_failed={self.n_failed})"
        )
