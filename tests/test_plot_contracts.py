"""Shared schema/factory contracts, checked without fitting a campaign."""
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import slat.plotting as slp
from sigmondsamplings.plot_contracts import (
    EmptyOptions, PlotInputError, SpectrumStyle, UncertaintyOptions,
    get_plot_definition, plot_definitions,
)
from sigmondsamplings.plotting import SamplingPlotter, draw_spectrum, prepare_spectrum, HMarker
from sigmondsamplings.info import ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from test_spectrum_plotter import levels


def samples(method="bootstrap"):
    info = SamplingInfo(method, 8, **({"seed": 1} if method == "bootstrap" else {}))
    return [SigmondSampling(np.r_[2., np.linspace(1., 3., 8)], ObservableInfo(name="p"), info)]


def test_schema_catalog_import_does_not_load_data_or_backends():
    output = subprocess.check_output([sys.executable, "-c", "import sys,json; import sigmondsamplings.plot_contracts; print(json.dumps(sorted(sys.modules)))"], text=True)
    loaded = json.loads(output)
    assert "sigmondsamplings.plotting" not in loaded
    assert "sigmondsamplings.sampling" not in loaded
    assert "matplotlib.pyplot" not in loaded
    assert "scipy" not in loaded


def test_every_method_has_strict_options_and_only_real_restrictions():
    assert get_plot_definition("correlation").options_model is EmptyOptions
    assert get_plot_definition("params", family="diagnostic").backends == {"plotly"}
    for definition in plot_definitions():
        with pytest.raises(ValueError):
            definition.options_model.model_validate({"irrelevant": True})
        if definition.family == "sampling" and definition.name != "corner":
            assert {"matplotlib", "plotly"} <= definition.backends


@pytest.mark.parametrize("method", ["histogram", "errorbar", "correlation", "eff-sample-size", "summary", "bootstrap-intervals"])
@pytest.mark.parametrize("backend", ["recording", "matplotlib", "plotly"])
def test_catalog_normalizes_and_restores_state(method, backend):
    old_backend, old_style = slp.current(), slp.get_style()
    made = get_plot_definition(method).make(samples(), backend=backend)
    try:
        assert isinstance(made.figure, slp.Figure)
        assert made.figure.backend == backend
    finally:
        made.figure.close()
    assert slp.current() == old_backend and slp.get_style() == old_style


def test_input_checks_precede_figure_creation(monkeypatch):
    monkeypatch.setattr(slp, "figure", lambda *a, **kw: pytest.fail("allocated a figure"))
    with pytest.raises(PlotInputError, match="bootstrap"):
        get_plot_definition("bootstrap-intervals").make(samples("jackknife"), backend="recording")
    with pytest.raises(PlotInputError, match="out of range"):
        get_plot_definition("summary").make(samples(), options={"obs_index": 10}, backend="recording")
    with pytest.raises(PlotInputError, match="nonempty"):
        get_plot_definition("histogram").make([], backend="recording")
    result = SimpleNamespace(optimize_result=SimpleNamespace(history=None), trace_path=None)
    with pytest.raises(PlotInputError, match="history"):
        get_plot_definition("params", family="diagnostic").make(result, backend="plotly")


def test_corner_missing_dependency_precedes_figure(monkeypatch):
    import sigmondsamplings.plotting.catalog as catalog
    original = catalog.find_spec
    monkeypatch.setattr(catalog, "find_spec", lambda name: None if name == "corner" else original(name))
    monkeypatch.setattr(slp, "figure", lambda *a, **kw: pytest.fail("allocated a figure"))
    with pytest.raises(slp.UnsupportedFeature, match="corner package"):
        get_plot_definition("corner").make(samples(), backend="matplotlib")


def test_synthetic_plot_is_labeled():
    data = SimpleNamespace(plotter=SamplingPlotter(samples()), is_synthetic=True)
    with slp.record() as recorder:
        made = get_plot_definition("histogram").make(data, backend="recording")
    assert made.metadata["sampling_origin"] == "synthetic"
    assert made.figure.title == "Synthetic parameter samplings"
    made.figure.close()


def test_prepare_freezes_range_and_statistics_without_allocating(monkeypatch):
    data = levels((0, 0.2, [1., 2., 3., 4.], "ecm", None))
    percentile = np.percentile
    calls = []
    def counting(*args, **kwargs):
        calls.append(True)
        return percentile(*args, **kwargs)
    monkeypatch.setattr(np, "percentile", counting)
    with slp.record() as recorder:
        prepared = prepare_spectrum(data, options={"percentiles": [10, 90]},
                                    overlays=[HMarker(y=5.)])
        assert not recorder.marks
        assert len(calls) == 1
        # The center outside the percentile interval participates in limits.
        assert prepared.energy_range == pytest.approx((-0.04, 5.24))
        axes = draw_spectrum(prepared, overlays=[HMarker(y=100.)])
        assert len(calls) == 1
        assert axes.get_limits("y") == prepared.energy_range
        assert "10–90%: [1.3, 3.7]" in prepared.levels[0][0].hover
    axes.figure.close()


def test_prepared_bounds_payload_is_backend_independent():
    data = levels((0, .2, [1., 2., 3., 4.], "ecm", None))
    prepared = prepare_spectrum(data, options={"percentiles": [10, 90]}, style=SpectrumStyle(ylim=(0., 4.)))
    for backend in ("recording", "matplotlib", "plotly"):
        axes = draw_spectrum(prepared, backend=backend)
        assert axes.get_limits("y") == (0., 4.)
        axes.figure.close()
    assert prepared.levels[0][0].bounds == pytest.approx((1.3, 3.7))
    with pytest.raises(TypeError):
        draw_spectrum(prepared, style=SpectrumStyle())


def test_uncertainty_resolves_rc_at_boundary():
    from sigmondsamplings.rcparams import rc_context
    with rc_context({"plot.error_metric": "confidence_interval", "plot.confidence_level": .9}):
        assert UncertaintyOptions().resolved() == {"error_metric": "confidence_interval", "confidence_level": .9}
    for args in ({"percentiles": (10, 90), "confidence_level": .8}, {"error_metric": "error", "percentiles": (10, 90)}):
        with pytest.raises(ValueError):
            UncertaintyOptions(**args)


def test_export_metadata_does_not_substitute_missing_engine(monkeypatch):
    from slat.plotting import export_capabilities
    import slat.plotting.export as exports
    monkeypatch.setattr(exports, "find_spec", lambda name: None)
    plotly = export_capabilities("plotly")
    assert plotly.defaults == ("html",)
    assert "png" in plotly.accepted
    assert not next(f for f in plotly.formats if f.format == "png").available
    assert next(f for f in plotly.formats if f.format == "html").available
    assert export_capabilities("matplotlib").defaults == ("png", "pdf")


def test_factory_closes_figures_created_before_a_method_failure(monkeypatch):
    closed = []
    before_backend, before_style = slp.current(), slp.get_style()
    def broken(*args, **kwargs):
        figure = slp.figure()
        monkeypatch.setattr(figure, "close", lambda: closed.append(figure))
        raise RuntimeError("drawing failed")
    monkeypatch.setattr(SamplingPlotter, "plot_sampling_errorbar", broken)
    with pytest.raises(RuntimeError, match="drawing failed"):
        get_plot_definition("errorbar").make(samples(), backend="recording")
    assert len(closed) == 1
    assert slp.current() == before_backend and slp.get_style() == before_style


def test_figure_scope_retains_only_returned_figures(monkeypatch):
    closed = []
    with slp.record(), slp.figure_scope() as scope:
        discarded = slp.figure()
        retained = slp.figure()
        monkeypatch.setattr(discarded, "close", lambda: closed.append(discarded))
        monkeypatch.setattr(retained, "close", lambda: closed.append(retained))
        scope.retain(retained)
    assert closed == [discarded]
    retained.close()


def test_sampling_marker_prepares_hover_once(monkeypatch):
    data = levels((0, .2, [1., 2., 3., 4.], "ecm", None))[0]
    percentile = np.percentile
    calls = []
    def counting(*args, **kwargs):
        calls.append(True)
        return percentile(*args, **kwargs)
    monkeypatch.setattr(np, "percentile", counting)
    marker = HMarker.from_sampling(data, percentiles=(10, 90))
    assert len(calls) == 1
    assert marker.hover_text == marker.hover_text
    assert len(calls) == 1
