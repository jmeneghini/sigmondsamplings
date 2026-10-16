"""SamplingFit.from_data: fitting a 1-D SigmondData against one of its coords."""

import pickle

import numpy as np
import pytest

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.info import SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats

om = pytest.importorskip("optimagic")

from sigmondsamplings.fitting import (  # noqa: E402
    CallableModel,
    CallableXModel,
    MinimizerConfig,
    ParamSetSpec,
    ParamSpec,
    SamplingFit,
    XModel,
)

A_TRUE, E_TRUE = 1.25, 0.31
LS = MinimizerConfig(algorithm="scipy_ls_trf")


def _exp(t, theta):
    return theta[0] * np.exp(-theta[1] * t)


MODEL = CallableXModel(_exp, ("A", "E"))


def _buffer(times, method="bootstrap", n_resamp=200, seed=0, noise=0.02):
    """``(len(times), n_resamp + 1)`` exponential-decay samples, error ~ noise * mean."""
    rng = np.random.default_rng(seed)
    factor = (n_resamp - 1) / np.sqrt(n_resamp) if method == "jackknife" else 1.0
    mean = A_TRUE * np.exp(-E_TRUE * np.asarray(times, dtype=float))
    err = noise * mean + 1e-4
    out = np.empty((len(times), n_resamp + 1))
    out[:, 0] = mean
    out[:, 1:] = rng.normal(mean[:, None], (err / factor)[:, None], (len(times), n_resamp))
    return out


def _series(times=range(2, 10), method="bootstrap", n_resamp=200, dim="time", **kw):
    info = SamplingInfo(method, n_resamp, seed=1)
    return SigmondSampling.from_data(
        _buffer(list(times), method, n_resamp, **kw),
        "corr",
        info,
        dim_names=[dim],
        coords={dim: list(times)},
    )


def _param_set():
    return ParamSetSpec(values={"A": ParamSpec(initial=1.0), "E": ParamSpec(initial=0.5)}).resolve()


# --- the model ------------------------------------------------------------------


def test_callable_x_model_binds_a_read_only_copy_of_x():
    t = np.array([1.0, 2.0, 3.0])
    bound = MODEL.bind(t)
    t[0] = 99.0
    assert isinstance(MODEL, XModel) and bound is not MODEL and MODEL.x is None
    assert list(bound.x) == [1.0, 2.0, 3.0] and not bound.x.flags.writeable
    assert np.allclose(bound.predict({"A": 2.0, "E": 0.5}), 2.0 * np.exp(-0.5 * bound.x))


def test_unbound_callable_x_model_refuses_to_predict():
    with pytest.raises(ValueError, match="bind"):
        MODEL.predict({"A": 1.0, "E": 0.1})


def test_bound_model_pickles_for_the_process_backend():
    bound = pickle.loads(pickle.dumps(MODEL.bind([3.0, 4.0])))
    assert np.allclose(bound.predict({"A": 1.0, "E": 1.0}), np.exp(-np.array([3.0, 4.0])))


# --- fitting --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bootstrap", "jackknife"])
def test_from_data_recovers_truth_and_keeps_x(method):
    series = _series(method=method)
    fit = SamplingFit.from_data(series, MODEL, _param_set(), minimizer=LS)
    assert fit.x_name == "time"
    assert list(fit.x) == list(range(2, 10)) and not fit.x.flags.writeable
    assert fit.sampling_info == series.sampling_info
    r = fit.fit_one(0)
    assert r.params["A"] == pytest.approx(A_TRUE, abs=1e-6)
    assert r.params["E"] == pytest.approx(E_TRUE, abs=1e-6)


def test_from_data_matches_the_stats_path_with_x_captured_by_hand():
    series = _series()
    t = np.asarray(series.coords["time"], dtype=float)
    by_hand = SamplingFit.from_model(
        SamplingStats(series.to_collection()),
        CallableModel(lambda theta: _exp(t, theta), ("A", "E")),
        _param_set(),
        minimizer=LS,
    )
    from_data = SamplingFit.from_data(series, MODEL, _param_set(), minimizer=LS)
    assert by_hand.x is None and by_hand.x_name is None
    a = by_hand.fit_resampled(n_resamplings=20, progress=False)
    b = from_data.fit_resampled(n_resamplings=20, progress=False)
    assert np.allclose(a.params.to_numpy(), b.params.to_numpy())
    assert np.allclose(a.theory.to_numpy(), b.theory.to_numpy())
    assert a.chi2 == pytest.approx(b.chi2)


def test_x_follows_the_data_order_not_sorted_order():
    """Coords out of order: each theory entry is still evaluated at its own time."""
    times = [7, 2, 5, 3, 9, 4]
    fit = SamplingFit.from_data(_series(times), MODEL, _param_set(), minimizer=LS)
    assert list(fit.x) == times
    r = fit.fit_one(0)
    assert r.params["E"] == pytest.approx(E_TRUE, abs=1e-6)
    theory = fit.objective.model.predict(dict(r.params))
    assert np.allclose(theory, A_TRUE * np.exp(-E_TRUE * np.array(times)), rtol=1e-5)


def test_explicit_x_names_the_dim():
    fit = SamplingFit.from_data(_series(dim="tsep"), MODEL, _param_set(), x="tsep")
    assert fit.x_name == "tsep"


def test_full_sample_fit_from_data_works():
    result = SamplingFit.from_data(_series(), MODEL, _param_set(), minimizer=LS).fit_full_sample(
        rng=1
    )
    assert result.param("E").full_sample_value == pytest.approx(E_TRUE, abs=1e-6)
    assert result.theory.num_observables == 8


# --- what from_data rejects -----------------------------------------------------


def test_complex_series_needs_an_explicit_real_projection():
    series = _series()
    cplx = series + 1j * 0.01
    assert cplx.is_complex
    with pytest.raises(ValueError, match="to_real"):
        SamplingFit.from_data(cplx, MODEL, _param_set())
    fit = SamplingFit.from_data(cplx.to_real(), MODEL, _param_set(), minimizer=LS)
    assert fit.fit_one(0).params["E"] == pytest.approx(E_TRUE, abs=1e-6)


def test_wrong_x_name_is_rejected():
    with pytest.raises(ValueError, match="not a dim"):
        SamplingFit.from_data(_series(), MODEL, _param_set(), x="t")


def test_multi_dim_and_scalar_data_are_rejected():
    series = _series()
    two_d = SigmondSampling.stack([series, series], dim_name="operator")
    with pytest.raises(ValueError, match="1-D"):
        SamplingFit.from_data(two_d, MODEL, _param_set())
    with pytest.raises(ValueError, match="1-D"):
        SamplingFit.from_data(series.isel(time=0), MODEL, _param_set())


def test_stats_and_plain_models_are_pointed_at_from_model():
    series = _series()
    with pytest.raises(TypeError, match="from_model"):
        SamplingFit.from_data(SamplingStats(series.to_collection()), MODEL, _param_set())
    plain = CallableModel(lambda theta: theta[0] * np.ones(8), ("A", "E"))
    with pytest.raises(TypeError, match="XModel"):
        SamplingFit.from_data(series, plain, _param_set())


def test_model_output_length_is_checked():
    short = CallableXModel(lambda t, theta: _exp(t, theta)[:-1], ("A", "E"))
    with pytest.raises(ValueError, match="one per x"):
        SamplingFit.from_data(_series(), short, _param_set())


def test_bins_are_rejected():
    bins = SigmondBins.from_data(
        np.ones((4, 10)), "corr", None, dim_names=["time"], coords={"time": [1, 2, 3, 4]}
    )
    with pytest.raises(TypeError, match="resample"):
        SamplingFit.from_data(bins, MODEL, _param_set())


def test_x_needs_numeric_coordinate_values():
    info = SamplingInfo("bootstrap", 10, seed=1)
    labelled = SigmondSampling.from_data(
        np.ones((2, 11)), "corr", info, dim_names=["op"], coords={"op": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="not numeric"):
        SamplingFit.from_data(labelled, MODEL, _param_set())
    unlabelled = SigmondSampling.from_data(np.ones((2, 11)), "corr", info, dim_names=["time"])
    with pytest.raises(ValueError, match="no coordinate"):
        SamplingFit.from_data(unlabelled, MODEL, _param_set())


def test_parameter_ensemble_inference_override_and_roundtrip(tmp_path):
    from sigmondsamplings.info import EnsembleInfo
    from sigmondsamplings.io.loader import SigmondLoader
    from sigmondsamplings.io.writer import SigmondWriter
    ensemble = EnsembleInfo('fit_pivot', 20, spatial_extent=8, temporal_extent=16, num_streams=1)
    series = SigmondSampling.from_data(_buffer(list(range(2, 10)), n_resamp=20), 'corr',
                                      SamplingInfo.bootstrap(20), dim_names=['time'],
                                      coords={'time': list(range(2, 10))}, ensemble_info=ensemble)
    for override in (None, EnsembleInfo('explicit_fit', 20)):
        expected = ensemble if override is None else override
        fit = SamplingFit.from_data(series, MODEL, _param_set(), minimizer=LS, ensemble_info=override)
        assert fit.ensemble_info == expected
        result = fit.fit_resampled(progress=False)
        assert result.param('A').ensemble_info == expected
        path = tmp_path / (expected.name + '.hdf5')
        SigmondWriter().write_hdf5(str(path), [result.param('A')], overwrite=True)
        assert next(iter(SigmondLoader(path).observables)).ensemble_info == expected
