"""Tests for the optimagic-native fit driver (SamplingFit / FitResult / scan)."""

import numpy as np
import pytest

from sigmondsamplings.info import ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats

om = pytest.importorskip("optimagic")

from sigmondsamplings.fitting import (  # noqa: E402
    CallableModel,
    CallableObjective,
    FitResult,
    LeastSquaresObjective,
    MinimizerConfig,
    ParamSetSpec,
    ParamSpec,
    SamplingFit,
    evaluate_chi2_function_scan,
    evaluate_chi2_scan,
)

A_TRUE, E_TRUE = 1.25, 0.31
T = np.arange(1, 9, dtype=float)


def _make_stats(method="bootstrap", n_resamp=400, seed=0, noise=0.02):
    """Synthetic exponential-decay SamplingStats with Gaussian resamples."""
    rng = np.random.default_rng(seed)
    info = SamplingInfo(method, n_resamp, seed=1)
    factor = np.sqrt(n_resamp - 1) if method == "jackknife" else 1.0
    samps = []
    for i, ti in enumerate(T):
        mean = A_TRUE * np.exp(-E_TRUE * ti)
        err = noise * mean + 1e-3
        data = np.empty(n_resamp + 1)
        data[0] = mean
        # draw so that SigmondSampling.error ~= err regardless of method
        data[1:] = rng.normal(mean, err / factor, n_resamp)
        samps.append(SigmondSampling(data, ObservableInfo(name=f"y{i}", index=i), info))
    return SamplingStats(samps)


def _model():
    return CallableModel(lambda th: th[0] * np.exp(-th[1] * T), ["A", "E"])


def _param_set(**over):
    return ParamSetSpec(
        values={
            "A": ParamSpec(initial=over.get("A", 1.0)),
            "E": ParamSpec(initial=over.get("E", 0.5)),
        }
    ).resolve()


def _fit(method="bootstrap", **kw):
    stats = _make_stats(method=method, **kw)
    return SamplingFit.from_model(
        stats, _model(), _param_set(), minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )


class ResampleShiftModel:
    def __init__(self, offsets):
        self.offsets = np.asarray(offsets, dtype=float)

    def predict(self, params):
        return np.array([params["p"]], dtype=float)

    def predict_at(self, params, resamp_idx):
        return np.array([params["p"] + self.offsets[resamp_idx]], dtype=float)


# --- fit_one ------------------------------------------------------------------


def test_fit_one_recovers_truth():
    r = _fit().fit_one(0)
    assert r.params["A"] == pytest.approx(A_TRUE, abs=1e-3)
    assert r.params["E"] == pytest.approx(E_TRUE, abs=1e-3)
    assert r.success


def test_fit_one_writes_sqlite_log(tmp_path):
    db = tmp_path / "fit.db"
    r = _fit().fit_one(0, trace_path=str(db))
    assert db.exists()
    assert r.history is not None


# --- fit_full_sample (jacobian mode) ------------------------------------------


def test_full_sample_synthetic_error_matches_cov_bootstrap():
    result = _fit("bootstrap").fit_full_sample(rng=42)
    assert isinstance(result, FitResult)
    assert result.is_synthetic
    se = result.standard_errors
    for name in result.param_names:
        # SigmondSampling.error must reproduce sqrt(diag(cov)) exactly
        assert result.param(name).error == pytest.approx(se[name], rel=1e-9)


def test_full_sample_synthetic_error_matches_cov_jackknife():
    result = _fit("jackknife", n_resamp=60).fit_full_sample(rng=42)
    se = result.standard_errors
    for name in result.param_names:
        assert result.param(name).error == pytest.approx(se[name], rel=1e-9)


def test_full_sample_cov_source_is_numerical_jacobian():
    # scipy_ls_trf does not populate result.jac -> falls back to first_derivative
    result = _fit().fit_full_sample(rng=0)
    assert result.cov_source == "first_derivative"


def test_full_sample_stores_residual_jacobian():
    result = _fit().fit_full_sample(rng=0)
    jac = result.residual_jacobian
    assert jac is not None
    # rows = whitened residuals (one per fitted observable), cols = free params
    assert jac.shape == (result.num_data_residuals, len(result.free_param_names))
    assert result.free_param_names == ["A", "E"]
    assert np.isfinite(jac).all()


def test_jacobian_reproduces_free_covariance():
    # inv(JᵀJ) is what the driver used for the synthesized covariance, so the
    # stored Jacobian must reproduce the free block of FitResult.cov.
    result = _fit().fit_full_sample(rng=0)
    jac = result.residual_jacobian
    expected = np.linalg.inv(jac.T @ jac)
    free_idx = [result.param_names.index(n) for n in result.free_param_names]
    assert result.cov[np.ix_(free_idx, free_idx)] == pytest.approx(expected, rel=1e-8)


def test_jacobian_diagnostics_on_well_conditioned_fit():
    result = _fit().fit_full_sample(rng=0)
    assert result.rank == len(result.free_param_names)
    assert result.condition_number > 1.0
    assert np.isfinite(result.condition_number)
    assert not result.cov_singular
    sv = result.singular_values
    assert sv.shape == (len(result.free_param_names),)
    assert result.condition_number == pytest.approx(sv[0] / sv[-1])
    assert result.summary()["rank"] == result.rank


def test_jacobian_columns_exclude_fixed_params():
    stats = _make_stats()
    pset = ParamSetSpec(
        values={
            "A": ParamSpec(initial=1.0),
            "E": ParamSpec(initial=E_TRUE, fixed=True),
        }
    ).resolve()
    fit = SamplingFit.from_model(
        stats, _model(), pset, minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_full_sample(rng=0)
    assert result.free_param_names == ["A"]
    assert result.residual_jacobian.shape[1] == 1
    assert result.rank == 1


def test_degenerate_model_is_rank_deficient():
    # A*B is degenerate in the product: one flat direction, so J loses a rank.
    stats = _make_stats()
    model = CallableModel(lambda th: th[0] * th[1] * np.exp(-E_TRUE * T), ["A", "B"])
    pset = ParamSetSpec(values={"A": ParamSpec(initial=1.0), "B": ParamSpec(initial=1.0)}).resolve()
    fit = SamplingFit.from_model(
        stats, model, pset, minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_full_sample(rng=0)
    assert result.rank == 1 < len(result.free_param_names)
    assert result.condition_number > 1e8
    # inv() does not raise on a matrix this ill-conditioned, so the flag has to come
    # from the rank check rather than from a LinAlgError
    assert result.cov_singular


def test_resampled_result_has_no_jacobian():
    result = _fit("bootstrap", n_resamp=40).fit_resampled(progress=False)
    assert result.residual_jacobian is None
    assert result.condition_number is None
    assert result.rank is None
    assert result.singular_values is None
    # free params are still recorded on the resampled path
    assert result.free_param_names == ["A", "E"]


def test_jacobian_round_trips_through_hdf5(tmp_path):
    result = _fit().fit_full_sample(rng=0)
    h5 = tmp_path / "results.h5"
    result.write_params(h5)
    result.write_chi2(h5)
    result.write_jacobian(h5)

    # without jacobian_file the Jacobian is absent (the pre-existing behaviour)
    bare = FitResult.from_hdf5(params_file=h5, chi2_file=h5, num_data_residuals=len(T))
    assert bare.residual_jacobian is None

    loaded = FitResult.from_hdf5(
        params_file=h5, chi2_file=h5, jacobian_file=h5, num_data_residuals=len(T)
    )
    assert loaded.residual_jacobian == pytest.approx(result.residual_jacobian)
    assert loaded.free_param_names == result.free_param_names
    assert loaded.cov_source == result.cov_source
    assert loaded.cov_singular == result.cov_singular
    assert loaded.rank == result.rank
    assert loaded.condition_number == pytest.approx(result.condition_number)


def test_write_jacobian_group_is_invisible_to_the_loader(tmp_path):
    # the jacobian group must not disturb the observable groups in the same file
    result = _fit().fit_full_sample(rng=0)
    h5 = tmp_path / "results.h5"
    result.write_params(h5)
    result.write_jacobian(h5)
    result.write_chi2(h5)  # written after, to catch the group tripping up the writer

    round_trip = FitResult.from_hdf5(params_file=h5, chi2_file=h5, num_data_residuals=len(T))
    assert round_trip.param_names == result.param_names
    assert round_trip.chi2 == pytest.approx(result.chi2)


def test_write_jacobian_requires_overwrite_and_a_jacobian(tmp_path):
    result = _fit().fit_full_sample(rng=0)
    h5 = tmp_path / "results.h5"
    result.write_jacobian(h5)
    with pytest.raises(ValueError, match="already exists"):
        result.write_jacobian(h5)
    result.write_jacobian(h5, overwrite=True)  # no raise

    resampled = _fit("bootstrap", n_resamp=40).fit_resampled(progress=False)
    with pytest.raises(ValueError, match="No residual Jacobian"):
        resampled.write_jacobian(tmp_path / "other.h5")


def test_full_sample_full_value_is_best_fit():
    result = _fit().fit_full_sample(rng=0)
    assert result.param("A").full_sample_value == pytest.approx(A_TRUE, abs=1e-3)
    assert result.param("E").full_sample_value == pytest.approx(E_TRUE, abs=1e-3)
    assert result.chi_squared.full_sample_value == pytest.approx(result.chi2)
    assert np.isnan(result.chi_squared.resampled_values).all()


def test_params_is_stats_and_cov_matches_errors():
    from sigmondsamplings.stats import SamplingStats

    result = _fit().fit_full_sample(rng=0)
    assert isinstance(result.params, SamplingStats)
    # cov is delegated to the params stats object; its diagonal must match the
    # per-parameter standard errors exactly.
    se = result.standard_errors
    diag = np.sqrt(np.diag(result.cov))
    for i, name in enumerate(result.param_names):
        assert diag[i] == pytest.approx(se[name], rel=1e-9)


def test_theory_stats_tracks_best_fit():
    from sigmondsamplings.stats import SamplingStats

    stats = _make_stats()
    fit = SamplingFit.from_model(
        stats, _model(), _param_set(), minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_full_sample(rng=0)
    assert isinstance(result.theory, SamplingStats)
    # one theory observable per fitted data observable, same sample count
    assert result.theory.num_observables == stats.num_observables
    assert result.theory.array.shape == stats.array.shape
    # full-sample theory == model at the best fit == close to the data means
    expected = _model().predict(result.best_params)
    assert np.allclose(result.best_theory, expected)
    assert np.allclose(result.best_theory, stats.array[:, 0], rtol=0.05)


def test_full_sample_theory_band_is_linear_propagation():
    import optimagic as om

    stats = _make_stats()
    calls = {"n": 0}

    def counted(theta):
        calls["n"] += 1
        return theta[0] * np.exp(-theta[1] * T)

    model = CallableModel(counted, ["A", "E"])
    fit = SamplingFit.from_model(
        stats, model, _param_set(), minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_full_sample(rng=42)

    # theory band reproduces sqrt(diag(G cov G^T)) exactly (linear propagation)
    theta = result.best_params
    deriv = om.first_derivative(model.predict, theta).derivative
    g = np.column_stack([deriv["A"], deriv["E"]])
    analytic = np.sqrt(np.diag(g @ result.cov @ g.T))
    assert np.allclose(np.asarray(result.theory.val.error), analytic, rtol=1e-8)

    # the model was NOT evaluated once per synthesized resample (only m(θ̂) + the
    # Jacobian finite differences), so the call count stays tiny vs n_resamp.
    assert calls["n"] < stats.num_samples
    result = _fit("bootstrap", n_resamp=80).fit_resampled(progress=False)
    assert result.theory is not None
    # theory error band is positive everywhere
    assert np.all(np.asarray(result.theory.val.error) > 0)


def test_resampled_theory_uses_post_fit_predictions():
    stats = _make_stats(n_resamp=40)
    calls = {"n": 0}

    def counted(theta):
        calls["n"] += 1
        return theta[0] * np.exp(-theta[1] * T)

    model = CallableModel(counted, ["A", "E"])
    fit = SamplingFit.from_model(
        stats, model, _param_set(), minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_resampled(progress=False)
    calls_after_fit = calls["n"]

    # the stored theory must equal the model at each best fit exactly...
    theory = result.theory.array
    param_arr = result.params.array
    for j in range(param_arr.shape[1]):
        expected = counted(param_arr[:, j])
        assert np.allclose(theory[:, j], expected)
    # ...and the fit made one post-fit prediction per stored sample.
    assert calls["n"] == calls_after_fit + param_arr.shape[1]


def test_resampled_model_predict_at_drives_objective_and_theory():
    n_resamp = 8
    info = SamplingInfo("bootstrap", n_resamp, seed=1)
    offsets = np.linspace(0.0, 0.7, n_resamp + 1)
    p_true = 1.5
    data = p_true + offsets
    stats = SamplingStats([SigmondSampling(data, ObservableInfo(name="y", index=0), info)])
    model = ResampleShiftModel(offsets)
    fit = SamplingFit.from_model(
        stats,
        model,
        ParamSetSpec(values={"p": ParamSpec(initial=0.0)}).resolve(),
        minimizer=MinimizerConfig(algorithm="scipy_ls_trf"),
    )

    obj = fit.objective
    assert obj.whitened_residuals({"p": p_true}, 3) == pytest.approx([0.0])

    result = fit.fit_resampled(progress=False, backend="serial")
    assert np.allclose(result.param("p").data, p_true)
    assert result.theory is not None
    assert np.allclose(result.theory.array[0], data)


def test_theory_none_for_custom_objective():
    stats = _make_stats()
    whitener = np.diag(1.0 / np.asarray(stats.val.error))
    data = stats.array
    model = _model()

    def residual_fn(params, idx):
        return whitener @ (data[:, idx] - model.predict(params))

    fit = SamplingFit(
        CallableObjective(residual_fn, num_data_residuals=len(T)),
        _param_set(),
        sampling_info=stats.sampling_info,
    )
    result = fit.fit_full_sample(rng=0)
    assert result.theory is None
    assert result.best_theory is None


def test_uncorrelated_objective_uses_diagonal_whitening_vector():
    stats = _make_stats()
    obj = LeastSquaresObjective(stats, _model(), use_correlation=False)
    assert obj.whitening.source == "diagonal"
    assert obj.whitening.operator.ndim == 1
    assert obj.whitening.rank == stats.num_observables

    params = {"A": A_TRUE, "E": E_TRUE}
    theory = _model().predict(params)
    expected = stats.whitened_residuals(theory, use_corr=False)
    assert np.allclose(obj.whitened_residuals(params, 0), expected)


def test_model_func_builds_from_param_stats():
    result = _fit().fit_full_sample(rng=0)
    mf = result.model_func(lambda x, A, E: A * np.exp(-E * x))
    assert mf is not None


def test_fixed_param_has_zero_variance():
    stats = _make_stats()
    pset = ParamSetSpec(
        values={
            "A": ParamSpec(initial=1.0),
            "E": ParamSpec(initial=E_TRUE, fixed=True),
        }
    ).resolve()
    fit = SamplingFit.from_model(
        stats, _model(), pset, minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    result = fit.fit_full_sample(rng=0)
    assert result.num_free_params == 1
    assert result.standard_errors["E"] == pytest.approx(0.0, abs=1e-12)
    assert result.param("E").error == pytest.approx(0.0, abs=1e-9)


# --- fit_resampled ------------------------------------------------------------


def test_resampled_spread_matches_jacobian_error():
    fit = _fit("bootstrap", n_resamp=600)
    jac = fit.fit_full_sample(rng=0)
    res = fit.fit_resampled(progress=False)
    assert not res.is_synthetic
    assert res.n_failed == 0
    # empirical spread of real resample fits ~ analytic jacobian error
    for name in res.param_names:
        assert res.param(name).error == pytest.approx(jac.param(name).error, rel=0.1)


def test_resampled_thread_backend_matches_serial():
    fit = _fit("bootstrap", n_resamp=80)
    serial = fit.fit_resampled(progress=False, backend="serial")
    thread = fit.fit_resampled(progress=False, backend="thread", num_workers=2)
    assert np.allclose(serial.params.array, thread.params.array)


def test_resampled_keep_none_drops_records():
    fit = _fit("bootstrap", n_resamp=40)
    assert fit.fit_resampled(progress=False, keep="summary").records is not None
    assert fit.fit_resampled(progress=False, keep="none").records is None


def test_resampled_keep_full_retains_optimize_results():
    fit = _fit("bootstrap", n_resamp=20)
    summary = fit.fit_resampled(progress=False, keep="summary")
    assert summary.optimize_results is None
    # records must not carry the heavy OptimizeResult objects
    assert all("result" not in rec for rec in summary.records)

    full = fit.fit_resampled(progress=False, keep="full")
    assert full.optimize_results is not None
    # one per resample plus the full sample at index 0
    assert set(full.optimize_results) == set(range(0, 21))
    assert full.optimize_results[0] is full.optimize_result
    assert full.optimize_results[5].params["A"] == pytest.approx(
        full.param("A").resampled_values[4], rel=1e-9
    )


def test_resampled_rejects_out_of_range_n():
    fit = _fit(n_resamp=40)
    with pytest.raises(ValueError, match="n_resamplings must be in"):
        fit.fit_resampled(progress=False, n_resamplings=1000)


# --- diagnostics / information criteria ---------------------------------------


def test_information_criteria_and_dof():
    result = _fit().fit_full_sample(rng=0)
    assert result.dof == result.num_data_residuals - result.num_free_params
    assert result.aic() == pytest.approx(result.chi2 - 2 * result.dof)
    assert result.bic() == pytest.approx(
        result.chi2 + result.num_free_params * np.log(result.num_data_residuals)
    )
    assert 0.0 <= result.goodness_of_fit() <= 1.0
    summary = result.summary()
    assert summary["is_synthetic"] is True
    assert set(summary) >= {"chi2", "dof", "aic", "bic", "aicc", "Q"}


# --- custom objective ---------------------------------------------------------


def test_callable_objective_requires_sampling_info():
    stats = _make_stats()
    whitener = np.diag(1.0 / np.asarray(stats.val.error))
    data = stats.array
    model = _model()

    def residual_fn(params, idx):
        np.array([params["A"], params["E"]])
        return whitener @ (data[:, idx] - model.predict(params))

    obj = CallableObjective(residual_fn, num_data_residuals=len(T))
    # no stats on the objective -> sampling_info must be supplied
    with pytest.raises(ValueError, match="sampling_info"):
        SamplingFit(obj, _param_set())

    fit = SamplingFit(obj, _param_set(), sampling_info=stats.sampling_info)
    r = fit.fit_one(0)
    assert r.params["A"] == pytest.approx(A_TRUE, abs=1e-3)


# --- chi2 scan ----------------------------------------------------------------


def test_chi2_scan_minimised_at_truth():
    stats = _make_stats()
    pred = lambda p: p[0] * np.exp(-p[1] * T)  # noqa: E731
    a_vals = np.linspace(1.1, 1.4, 7)
    grid = np.column_stack([a_vals, np.full_like(a_vals, E_TRUE)])
    scan = evaluate_chi2_scan(stats, pred, [0, 1], grid)
    # minimum chi2 nearest A_TRUE
    assert a_vals[int(np.argmin(scan.chi2_values))] == pytest.approx(A_TRUE, abs=0.06)


def test_chi2_function_scan_matches_prediction_scan():
    stats = _make_stats()
    pred = lambda p: p[0] * np.exp(-p[1] * T)  # noqa: E731
    grid = np.column_stack([np.linspace(1.1, 1.4, 5), np.full(5, E_TRUE)])
    a = evaluate_chi2_scan(stats, pred, [0, 1], grid)
    b = evaluate_chi2_function_scan(lambda p: float(stats.chi_squared(pred(p))), [0, 1], grid)
    assert np.allclose(a.chi2_values, b.chi2_values)
