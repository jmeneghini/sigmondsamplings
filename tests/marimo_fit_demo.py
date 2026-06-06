"""Interactive marimo demo for SamplingFit.

Run with:

    marimo edit tests/marimo_fit_demo.py

The demo builds synthetic bootstrap data for a one-state exponential model,
fits the full sample plus resamplings, and shows how the fit driver behaves
under serial and thread backends.  The process backend is intentionally not
used here because functions defined inside notebook cells are often not
picklable.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares

    from sigmondsamplings.fit import SamplingFit, default_num_workers
    from sigmondsamplings.fit_plotter import FitPlotStyle, plot_fit_result
    from sigmondsamplings.info import ObservableInfo, SamplingInfo
    from sigmondsamplings.sampling import SigmondSampling

    return (
        FitPlotStyle,
        ObservableInfo,
        SamplingFit,
        SamplingInfo,
        SigmondSampling,
        default_num_workers,
        least_squares,
        mo,
        np,
        os,
        plot_fit_result,
        plt,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # `SamplingFit` execution demo

    This notebook creates synthetic bootstrap samples for

    \[
    C(t) = A e^{-Et}
    \]

    then fits the full sample and every bootstrap replica with
    `SamplingFit.run`.  Use it to inspect backend selection, BLAS/OpenMP
    thread settings, progress behavior, and recorded resample failures.
    """)
    return


@app.cell
def _(mo):
    controls = mo.ui.dictionary(
        {
            "n_resamplings": mo.ui.slider(
                start=20,
                stop=300,
                step=20,
                value=100,
                label="bootstrap resamplings",
            ),
            "backend": mo.ui.dropdown(
                options=["serial", "thread"],
                value="thread",
                label="backend",
            ),
            "num_workers": mo.ui.slider(
                start=1,
                stop=16,
                step=1,
                value=4,
                label="workers",
            ),
            "num_blas_threads": mo.ui.slider(
                start=1,
                stop=8,
                step=1,
                value=1,
                label="BLAS threads",
            ),
            "num_openmp_threads": mo.ui.slider(
                start=1,
                stop=8,
                step=1,
                value=1,
                label="OpenMP threads",
            ),
            "inject_failure": mo.ui.checkbox(
                value=False,
                label="inject one failed resample",
            ),
        }
    )
    controls
    return (controls,)


@app.cellpip
def _(controls, default_num_workers, mo, os):
    requested_workers = controls["num_workers"].value
    worker_budget = default_num_workers(
        num_blas_threads=controls["num_blas_threads"].value,
        num_openmp_threads=controls["num_openmp_threads"].value,
    )

    mo.hstack(
        [
            mo.stat(
                label="SLURM_CPUS_PER_TASK",
                value=os.environ.get("SLURM_CPUS_PER_TASK", "unset"),
            ),
            mo.stat(label="auto worker budget", value=str(worker_budget)),
            mo.stat(label="requested workers", value=str(requested_workers)),
            mo.stat(
                label="CPU budget requested",
                value=str(
                    requested_workers
                    * controls["num_blas_threads"].value
                    * controls["num_openmp_threads"].value
                ),
            ),
        ],
        justify="start",
    )
    return (requested_workers,)


@app.cell
def _(SamplingInfo, controls, np):
    seed = 20240519
    rng = np.random.default_rng(seed)

    t = np.arange(3, 18, dtype=float)
    true_params = np.array([1.25, 0.31])

    def model(params, x):
        amp, energy = params
        return amp * np.exp(-energy * x)

    def exp_sampling_model(x, amplitude, energy):
        return amplitude * np.exp(-energy * x)

    true_mean = model(true_params, t)
    rel_err = 0.025 + 0.05 * (t - t.min())
    sigma = rel_err * true_mean
    full_mean = true_mean + rng.normal(scale=sigma)

    n_resamplings = controls["n_resamplings"].value
    y_samples = np.empty((n_resamplings + 1, len(t)), dtype=float)
    y_samples[0] = full_mean
    y_samples[1:] = full_mean + rng.normal(scale=sigma, size=(n_resamplings, len(t)))

    sampling_info = SamplingInfo("bootstrap", n_resamplings, seed=seed)
    return exp_sampling_model, model, sampling_info, sigma, t, y_samples


@app.cell
def _(ObservableInfo, SigmondSampling, sampling_info, t, y_samples):
    data_samplings = [
        SigmondSampling(
            data=y_samples[:, i],
            observable_info=ObservableInfo(
                name=f"y_{i}",
                index=i,
                op_type="n",
                re_im="re",
                latex_str=f"y_{{{i}}}",
            ),
            sampling_info=sampling_info,
        )
        for i in range(len(t))
    ]
    return (data_samplings,)


@app.cell
def _(
    SamplingFit,
    controls,
    least_squares,
    model,
    np,
    requested_workers,
    sampling_info,
    sigma,
    t,
    time,
    y_samples,
):
    fit = SamplingFit(sampling_info)

    def fit_at_resamp(resamp_idx, x0):
        if controls["inject_failure"].value and resamp_idx == 7:
            raise RuntimeError("intentional demo failure at resample 7")

        y = y_samples[resamp_idx]

        def residuals(params):
            return (model(params, t) - y) / sigma

        result = least_squares(
            residuals,
            x0=np.asarray(x0, dtype=float),
            bounds=([0.0, 0.0], [np.inf, 2.0]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        chisq = float(np.dot(result.fun, result.fun))
        return result.x, chisq

    tic = time.perf_counter()
    fit_result = fit.run(
        fit_at_resamp,
        param_names=["amplitude", "energy"],
        x0=np.array([1.0, 0.25]),
        backend=controls["backend"].value,
        num_workers=requested_workers,
        num_blas_threads=controls["num_blas_threads"].value,
        num_openmp_threads=controls["num_openmp_threads"].value,
        progress="marimo",
        error_policy="record",
    )
    elapsed = time.perf_counter() - tic
    return elapsed, fit_result


@app.cell
def _(elapsed, fit_result, mo):
    mo.hstack(
        [
            mo.stat(label="elapsed", value=f"{elapsed:.3f} s"),
            mo.stat(label="fits stored", value=str(fit_result.n_fits)),
            mo.stat(label="failed resamples", value=str(fit_result.n_failed)),
            mo.stat(
                label="full-sample chi2",
                value=f"{fit_result.full_sample_chisq:.3f}",
            ),
        ],
        justify="start",
    )
    return


@app.cell
def _(fit_result, mo):
    rows = [
        {
            "parameter": name,
            "full sample": sampling.full_sample_value,
            "resample mean": sampling.mean,
            "error": sampling.error,
        }
        for sampling in fit_result.params
        for name in [sampling.observable_info.name]
    ]
    latex_map = {
        'amplitude': r"A",
        'energy': r"E",
    }
    for obs_name, latex_str in latex_map.items():
        fit_result.params.find(name = obs_name).observable_info.latex_str = latex_str
    # print(fit_result.params
    mo.ui.table(rows, label="Fit parameter samplings")
    return


@app.cell
def _():
    return


@app.cell
def _(fit_result, mo):
    if fit_result.failures:
        failure_rows = [
            {"resamp_idx": idx, "error": message}
            for idx, message in sorted(fit_result.failures.items())
        ]
        mo.vstack(
            [
                mo.md("### Recorded failures"),
                mo.ui.table(failure_rows),
            ]
        )
    else:
        mo.md("No resample failures were recorded.")
    return


@app.cell
def _(exp_sampling_model, fit_result, mo):
    fitted_model = fit_result.model_func(
        exp_sampling_model,
        latex_str=r"A e^{-E {VAR}}",
        independent_var_latex="t",
    )
    mo.md(
        f"""
        ### Collection-backed model

        The fit result stores parameters as an `ObservableCollection`, so the
        fitted model can be built directly from `fit_result.params`.

        ```python
        fitted_model = fit_result.model_func(
            exp_sampling_model,
            latex_str=r"A e^{{-E {{VAR}}}}",
            independent_var_latex="t",
        )
        ```

        `{fitted_model!r}`
        """
    )
    return (fitted_model,)


@app.cell
def _(FitPlotStyle, mo):
    from dataclasses import asdict
    cfg = FitPlotStyle(
        mean_linewidth=1.0,
        show_bootstrap_cloud=True,
        show_confidence_band=True,
        show_confidence_ellipses=False,
        show_data=False,
        data_label=None,
        show_mean=False,
        show_full=False,
        confidence_level=0.95,
        display_params=True,
        metrics=["chi_squared", "dof", "chi2_per_dof", "goodness_of_fit", "aic"],
        annotation_loc="upper right",
        annotation_fontsize=9,
    )
    mo.tree(asdict(cfg))
    return (cfg,)


@app.cell
def _(cfg, data_samplings, fitted_model, np, plot_fit_result, t):
    fig_model, ax_model = plot_fit_result(
        t,
        fitted_model,
        x_fit_values=np.linspace(float(t.min()), float(t.max()), 250),
        figsize=(7.5, 4.5),
        data_samplings=data_samplings,
        style=cfg,
    )
    fig_model
    return


@app.cell
def _(fit_result, np, plt):
    fig_, axes_ = plt.subplots(1, 3, figsize=(12, 3.5))

    for _ax, name in zip(axes_[:2], ["amplitude", "energy"]):
        sampling = fit_result.param(name)
        _ax.hist(sampling.resampled_values, bins=24, color="tab:blue", alpha=0.75)
        _ax.axvline(sampling.full_sample_value, color="black", lw=2, label="full")
        _ax.set_title(f"{name}: {sampling.full_sample_value:.5f} +/- {sampling.error:.5f}")
        _ax.legend()

    chisq_values = fit_result.chi_squared.resampled_values
    chisq_values = chisq_values[np.isfinite(chisq_values)]
    axes_[2].hist(chisq_values, bins=24, color="tab:orange", alpha=0.75)
    axes_[2].axvline(fit_result.full_sample_chisq, color="black", lw=2, label="full")
    axes_[2].set_title("chi-squared")
    axes_[2].legend()

    fig_.tight_layout()
    fig_
    return


@app.cell
def _(mo):
    mo.md("""
    ### Notes

    - `backend="thread"` is useful in this notebook because cell-defined
      functions can run in threads without needing pickle support.
    - For batch scripts, use `backend="process"` when `fit_at_resamp` is a
      top-level picklable function.
    - SLURM still launches Python externally. Inside the allocation,
      `default_num_workers()` uses `SLURM_CPUS_PER_TASK` when present.
    - Keep `num_blas_threads` and `num_openmp_threads` explicit. They are
      separate runtime limits and may oversubscribe if both libraries nest.
    - `fit_result.params` is an `ObservableCollection`; use `fit_result.param(name)`
      for direct lookup or pass the collection into `SigmondModelFunc`.
    - `plot_fit_result` accepts numeric x-values or sampled x-values; it returns `(fig, ax)`.
    """)
    return


if __name__ == "__main__":
    app.run()
