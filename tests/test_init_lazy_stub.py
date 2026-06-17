import types

import sigmondsamplings as ss


def test_lazy_stub_exposes_public_submodules():
    assert isinstance(ss.sampling, types.ModuleType)
    assert isinstance(ss.fit, types.ModuleType)
    assert isinstance(ss.model_func, types.ModuleType)
    assert isinstance(ss.io, types.ModuleType)


def test_lazy_stub_exposes_current_public_names():
    for name in [
        "SigmondSampling",
        "SamplingFit",
        "FitAtResamp",
        "Chi2Scan",
        "SigmondModelFunc",
        "ParamSpec",
        "Minimizer",
        "SigmondLoader",
        "SigmondWriter",
        "rcparams",
    ]:
        assert name in ss.__all__
        assert getattr(ss, name) is not None
