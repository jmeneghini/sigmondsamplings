import types

import sigmondsamplings as ss


def test_lazy_stub_exposes_public_submodules():
    assert isinstance(ss.sampling, types.ModuleType)
    assert isinstance(ss.fitting, types.ModuleType)
    assert isinstance(ss.io, types.ModuleType)


def test_lazy_stub_exposes_current_public_names():
    for name in [
        "SigmondSampling",
        "SamplingFit",
        "FitResult",
        "Chi2Scan",
        "SigmondModelFunc",
        "ParamSetSpec",
        "ParamSetResolved",
        "ParamSpec",
        "MinimizerConfig",
        "SigmondLoader",
        "SigmondWriter",
        "rcparams",
    ]:
        assert name in ss.__all__
        assert getattr(ss, name) is not None
