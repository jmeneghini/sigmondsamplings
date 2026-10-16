"""Correctness contracts for ranges, ownership, metadata and prepared snapshots."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_plot_contracts import samples
from test_plotting_correlators import _stack, fit_result
from test_spectrum_plotter import spectrum

import slat.plotting as slp
from sigmondsamplings.plot_contracts import SpectrumStyle, get_plot_definition
from sigmondsamplings.plotting import (
    FitOverlay,
    HMarker,
    SamplingPlotter,
    SeriesPlotter,
    SeriesStyle,
    draw_spectrum,
    prepare_spectrum,
)
from slat.plotting.lifecycle import close_figure, track_figure, track_native_figure

__all__ = ["fit_result", "spectrum"]


def test_wrap_plain_figure_supports_portable_drawing_and_configuration():
    import plotly.graph_objects as go

    native = go.Figure(layout={"title": "native title", "margin": {"l": 100}})
    original = native.layout.to_plotly_json()
    fig = slp.wrap(native)
    assert fig.shape == (1, 1)
    assert native.layout.to_plotly_json() == original
    ax = fig.axes()
    ax.line([1, 2], [3, 4], label="line")
    ax.errorbar([1], [3], yerr=[0.2])
    ax.band([1, 2], [2, 2], [4, 4])
    ax.bar([1], [2])
    ax.polygon([1, 2, 1], [2, 3, 4])
    ax.hlines(3, 1, 2)
    ax.vlines(1.5, 2, 4)
    ax.axhline(3, hover="reference")
    ax.axvline(1.5)
    ax.heatmap([[1, 2], [3, 4]], colorbar=True)
    ax.text(0.5, 0.5, "annotation", coords="axes")
    ax.set(xlim=(1, 2), ylim=(2, 4), xlabel="x", title="cell title")
    ax.ticks("x", [1, 2], ["a", "b"])
    ax.grid(True)
    ax.legend()
    assert ax.get_limits("x") == (1, 2)
    assert all(t.xaxis == "x" and t.yaxis == "y" for t in native.data if t.legendgroup is None)
    assert native.layout.title.text == "native title"
    fig.close()


def test_wrap_native_grid_retains_holes_and_secondary_axes():
    from plotly.subplots import make_subplots

    native = make_subplots(2, 2, specs=[[{"secondary_y": True}, None], [{}, {"type": "polar"}]])
    original = native.layout.to_plotly_json()
    fig = slp.wrap(native)
    assert fig.shape == (2, 2)
    assert len(fig.flat) == 2
    assert native.layout.to_plotly_json() == original
    with pytest.raises(ValueError, match="no Cartesian"):
        fig.axes(0, 1)
    with pytest.raises(ValueError, match="no Cartesian"):
        fig.axes(1, 1)
    fig.axes(1, 0).line([1], [2])
    assert (native.data[0].xaxis, native.data[0].yaxis) == ("x2", "y3")
    fig.embed_html()
    assert len(native.data) == 2
    assert (native.data[1].xaxis, native.data[1].yaxis) == ("x", "y")
    assert native.layout.yaxis2.overlaying == "y"
    fig.close()


def test_wrap_domain_layout_and_non_cartesian_export():
    import plotly.graph_objects as go

    native = go.Figure(layout={"xaxis": {"domain": [0, 0.4], "anchor": "y"},
                              "yaxis": {"domain": [0, 1]},
                              "xaxis4": {"domain": [0.6, 1], "anchor": "y4"},
                              "yaxis4": {"domain": [0, 1]}})
    fig = slp.wrap(native)
    assert fig.shape == (1, 2)
    fig.axes(0, 1).set(xlim=(2, 8))
    fig.axes(0, 1).axhline(4)
    fig.embed_html()
    assert native.layout.xaxis4.range == (2, 8)
    assert native.layout.shapes[0].xref == "x4 domain"
    assert {(t.xaxis, t.yaxis) for t in native.data} == {("x", "y"), ("x4", "y4")}
    polar = slp.wrap(go.Figure(go.Scatterpolar(r=[1, 2], theta=[0, 90])))
    assert not polar.flat
    assert "scatterpolar" in polar.embed_html()
    fig.close()
    polar.close()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("order", ["scale-first", "limits-first", "together"])
def test_plotly_scale_changes_preserve_data_ranges(reverse, order):
    fig = slp.figure(backend="plotly")
    ax = fig.axes()
    limits = (100, 1) if reverse else (1, 100)
    if order == "scale-first":
        ax.set(yscale="log")
        ax.set(ylim=limits)
    elif order == "limits-first":
        ax.set(ylim=limits)
        ax.set(yscale="log")
    else:
        ax.set(yscale="log", ylim=limits)
    assert ax.get_limits("y", explicit_only=True) == limits
    ax.set(yscale="linear")
    assert ax.get_limits("y", explicit_only=True) == limits
    assert fig.native.layout.yaxis.range == limits
    fig.close()


def test_invalid_log_transition_keeps_previous_axis():
    fig = slp.figure(backend="plotly")
    fig.axes().set(ylim=(-1, 2))
    with pytest.raises(ValueError, match="positive"):
        fig.axes().set(yscale="log")
    assert fig.axes().get_limits("y") == (-1, 2)
    assert fig.native.layout.yaxis.type != "log"
    fig.close()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("setter", [0, 1])
def test_shared_ranges_resolve_both_directions_and_cycles(axis, setter):
    fig = slp.figure(2, 1, sharex=True, sharey=True, backend="plotly")
    # Force a native cycle too: the component walk must be safe.
    first = fig.native.layout[axis + "axis"]
    second = fig.native.layout[axis + "axis2"]
    first.matches, second.matches = axis + "2", axis
    fig.axes(setter).set(**{axis + "lim": (2, 8)})
    for ax in fig.flat:
        assert ax.get_limits(axis, explicit_only=True) == (2, 8)
    fig.axes(1 - setter).set(**{axis + "scale": "log"})
    for ax in fig.flat:
        np.testing.assert_allclose(ax.get_limits(axis), (2, 8))
    fig.close()


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_point_size_is_local_and_obeys_point_sign(backend):
    fig = slp.figure(backend=backend)
    ax = fig.axes()
    ax.set(ylim=(100, 1), yscale="log")
    assert ax.point_size(6) > 0
    assert ax.point_size(-6) == pytest.approx(-ax.point_size(6))
    if backend == "plotly":
        fig.native.update_layout(width=600, height=500, margin={"l": 80, "r": 80, "t": 100, "b": 100})
        assert ax.point_size(6, units="axes") == pytest.approx((6 * 100 / 72) / 300)
        fig.set_size(6, 7)
        before = ax.point_size(6, units="axes")
        ax.set(title="first line\nsecond line\nthird line\nfourth line")
        assert ax.point_size(6, units="axes") > before
    fig.close()


def test_nested_success_transfers_owned_allocation_for_outer_failure():
    with pytest.raises(RuntimeError), slp.figure_scope():
        with slp.figure_scope() as inner:
            fig = slp.figure(backend="recording")
            fig.close = Mock()
            inner.retain(fig)
        raise RuntimeError("outer failure")
    fig.close.assert_called_once()


def test_close_removes_native_alias_from_all_scopes():
    native = object()
    closer = Mock()
    with slp.figure_scope():
        track_native_figure(native, closer)
        with slp.figure_scope():
            proxy = SimpleNamespace(native=native, close=lambda: closer(native))
            close_figure(proxy)
    closer.assert_called_once_with(native)


@pytest.mark.parametrize("caller_owned", [False, True])
def test_direct_frontend_failure_closes_only_owned_figure(monkeypatch, caller_owned):
    fig = slp.figure(backend="recording")
    fig.close = Mock()
    monkeypatch.setattr(slp, "figure", lambda *a, **kw: track_figure(fig))
    monkeypatch.setattr(fig.axes(), "_errorbar", Mock(side_effect=RuntimeError("drawing failed")))
    with pytest.raises(RuntimeError, match="drawing failed"):
        SeriesPlotter(_stack().isel(operator=0)).plot(axes=fig.axes() if caller_owned else None)
    assert fig.close.call_count == (0 if caller_owned else 1)


def test_summary_and_bootstrap_validate_before_allocation(monkeypatch):
    monkeypatch.setattr(slp, "figure", lambda *a, **kw: pytest.fail("allocated before validation"))
    with pytest.raises(ValueError, match="layout"):
        SamplingPlotter(samples()).plot_stats_summary(layout=(1, 1))
    with pytest.raises(ValueError, match="bootstrap"):
        SamplingPlotter(samples("jackknife")).plot_bootstrap_intervals()


def test_corner_failure_cleans_native_allocation(monkeypatch):
    corner = pytest.importorskip("corner")
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    monkeypatch.setattr(corner, "corner", Mock(side_effect=RuntimeError("corner failed")))
    with pytest.raises(RuntimeError, match="corner failed"):
        SamplingPlotter(samples()).plot_corner()
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("index", [None, 1])
def test_summary_metadata_describes_actual_panel_methods(index):
    first, second = samples()[0], samples()[0]
    second.observable_info = type(second.observable_info)(name="second", index=1)
    data = SamplingPlotter([first, second])
    # Mutate a leaf after collection validation to exercise mixed metadata.
    data.stats[1].sampling_info = samples("jackknife")[0].sampling_info
    made = get_plot_definition("summary").make(data, options={"obs_index": index} if index else {}, backend="recording")
    try:
        panels = made.metadata["statistics"]["panels"]
        histograms = panels[:-1]
        assert panels[-1] == {"panel": "correlation"}
        assert [p["observable_index"] for p in histograms] == ([0, 1] if index is None else [1])
        for p in histograms:
            assert p["central"] == "mean"
            if p["method"] == "bootstrap":
                assert p["error_metric"] == "confidence_interval" and p["confidence_level"] == 0.68
            else:
                assert p["error_metric"] == "error" and "confidence_level" not in p
    finally:
        made.figure.close()


def test_summary_non_histogram_metadata_does_not_claim_intervals():
    made = get_plot_definition("summary").make(samples(), options={"panels": ["eff_sample_size", "errorbar"]}, backend="recording")
    assert made.metadata["statistics"] == {"panels": [
        {"panel": "eff_sample_size"}, {"panel": "errorbar", "central": "mean", "error_metric": "error"},
    ]}
    made.figure.close()


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_prepared_spectrum_snapshots_markers_styles_and_metadata(spectrum, backend):
    marker = HMarker(y=0.6, err=(0.3, 0.3), hover="original", color="red")
    style = SpectrumStyle(markersize=8, errorbar_kwargs={"marker": {"opacity": 0.6}} if backend == "plotly" else {})
    prepared = prepare_spectrum(spectrum, style=style, overlays=[marker])
    marker.y, marker.err, marker.hover = 60, (30, 90), "edited"
    style.markersize = 12
    if backend == "plotly":
        style.errorbar_kwargs["marker"]["opacity"] = 0.1
    assert prepared.style.markersize == 8
    assert prepared.style.model_dump(mode="json")["markersize"] == 8
    assert prepared.markers[0][0].y == 0.6
    assert prepared.markers[0][0].hover_text == "original"
    with pytest.raises(TypeError):
        prepared.statistics["central"] = "edited"
    with pytest.raises(ValueError):
        prepared.style.markersize = 12
    for _ in range(2):
        fig = draw_spectrum(prepared, backend=backend).figure
        assert fig.axes().get_limits("y") == prepared.energy_range
        if backend == "plotly":
            assert any(t.hovertext and "original" in t.hovertext for t in fig.native.data)
            assert fig.native.data[0].marker.opacity == 0.6
        fig.close()


def test_sampling_backed_marker_is_immutable_snapshot(spectrum):
    marker = HMarker.from_sampling(spectrum[0], ni_pair="NN", ni_rank=1)
    old = (marker.y, marker.band, marker.hover_text)
    for key, value in (("y", 50), ("ni_rank", 2), ("group", "other"), ("percentiles", (5, 95))):
        with pytest.raises(FrozenInstanceError):
            setattr(marker, key, value)
    spectrum[0].data[:] = 50
    assert (marker.y, marker.band, marker.hover_text) == old


@pytest.mark.parametrize("shared", [False, True])
def test_series_preserves_native_and_shared_caller_limits(shared):
    fig = slp.figure(2, 1, sharex="all" if shared else False, sharey="all" if shared else False, backend="plotly")
    target = fig.axes(1) if shared else fig.axes()
    target.set(xlim=(2, 8), ylim=(0.1, 3))
    SeriesPlotter(_stack().isel(operator=0)).plot(axes=fig.axes())
    assert fig.axes().get_limits("x") == (2, 8)
    assert fig.axes().get_limits("y") == (0.1, 3)
    fig.close()


def test_spectrum_preserves_shared_caller_limits(spectrum):
    from sigmondsamplings.plotting import SectorSpectrumPlotter

    fig = slp.figure(2, 1, sharex=True, backend="plotly")
    fig.axes(1).set(xlim=(-2, 4))
    fig.native.layout.yaxis.matches = "y2"
    fig.axes(1).set(ylim=(0.2, 2))
    SectorSpectrumPlotter(spectrum).plot(axes=fig.axes())
    assert fig.axes().get_limits("x") == (-2, 4)
    assert fig.axes().get_limits("y") == (0.2, 2)
    fig.close()


@pytest.mark.parametrize("log", [False, True])
def test_filtered_reference_only_plot_ranges_include_wide_band(log):
    with slp.record() as rec:
        axes = SeriesPlotter(_stack().isel(operator=0), style=SeriesStyle(max_rel_error=0)).plot(
            yscale="log" if log else "linear", markers=[HMarker(y=5, err=(4, 95))],
        )
    assert not rec.of_kind("errorbar")
    assert rec.count("hlines") == 1
    assert axes.get_limits("x")[1] > axes.get_limits("x")[0]
    assert axes.get_limits("y")[0] < 1 and axes.get_limits("y")[1] > 100


def test_fit_prediction_is_prepared_once_and_expands_filtered_plot(fit_result, monkeypatch):
    original = type(fit_result).theory_series
    calls = Mock(side_effect=original)
    monkeypatch.setattr(type(fit_result), "theory_series", lambda self: calls(self))
    with slp.record() as rec:
        axes = SeriesPlotter(_stack().isel(operator=0), style=SeriesStyle(max_rel_error=0)).plot(
            fits=[FitOverlay(fit_result, label="fit")],
        )
    calls.assert_called_once_with(fit_result)
    line = rec.first("line")
    assert axes.get_limits("x")[0] < min(line.args[0])
    assert axes.get_limits("y")[1] > max(line.args[1])


def test_series_uses_figure_marker_palette_with_explicit_override():
    with slp.record() as rec:
        fig = slp.figure(style=slp.PlotStyle(markers=("s", "D", "^")))
        SeriesPlotter(_stack()).plot(axes=fig.axes())
    assert [m.kwargs["marker"] for m in rec.of_kind("errorbar")] == ["s", "D", "^"]
    with slp.record() as rec:
        SeriesPlotter(_stack(), style=SeriesStyle(marker_cycle=("x",))).plot()
    assert {m.kwargs["marker"] for m in rec.of_kind("errorbar")} == {"x"}


@pytest.mark.parametrize("levels,filled", [([], False), ([1], True), ([1, 1], False), ([2, 1], True), ([1, np.nan], False)])
def test_contour_rejects_invalid_levels(levels, filled):
    with slp.record() as rec:
        fig = slp.figure()
        with pytest.raises(ValueError, match="levels"):
            fig.axes().contour([0, 1], [0, 1], [[0, 1], [1, 2]], levels=levels, filled=filled)
    assert not rec.of_kind("contour")


@pytest.mark.parametrize("kind", ["generic", "spectrum", "correlator"])
@pytest.mark.parametrize("failure", ["save", "show", None])
def test_cli_finish_closes_once_on_failure_and_retains_success(spectrum, monkeypatch, tmp_path, kind, failure):
    from sigmondsamplings.cli.plot import (
        render_correlator_plot,
        render_generic_plot,
        render_spectrum_plot,
    )
    from slat.plotting.recording import RecordingFigure

    closed = []
    monkeypatch.setattr(RecordingFigure, "close", lambda self: closed.append(self))
    for operation in ("save", "show"):
        def finish(self, *args, op=operation, **kwargs):
            if op == failure:
                raise RuntimeError(op + " failed")
        monkeypatch.setattr(RecordingFigure, operation, finish)
    def render():
        options = {"output": tmp_path / "plot.html", "show": True, "backend": "recording"}
        if kind == "generic":
            render_generic_plot(samples(), method="histogram", **options)
        elif kind == "spectrum":
            render_spectrum_plot(spectrum, **options)
        else:
            render_correlator_plot(_stack().isel(operator=0), method="correlator", **options)
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            render()
        assert len(closed) == 1
    else:
        render()
        assert not closed


def test_catalog_failure_before_native_adoption_releases_figure(monkeypatch):
    import plotly.graph_objects as go

    native = go.Figure(go.Scatter(x=[1], y=[2]))
    definition = get_plot_definition("params", family="diagnostic")
    target = SimpleNamespace(**{definition.method: lambda: native})
    monkeypatch.setattr(slp, "as_figure", Mock(side_effect=RuntimeError("adoption failed")))
    with pytest.raises(RuntimeError, match="adoption failed"):
        definition.make(target, backend="plotly")
    assert not native.data


def test_wrap_trace_references_without_explicit_native_layout():
    import plotly.graph_objects as go

    native = go.Figure(go.Scatter(x=[1], y=[2], xaxis="x4", yaxis="y6"))
    before = native.layout.to_plotly_json()
    fig = slp.wrap(native)
    assert native.layout.to_plotly_json() == before
    assert fig.axes().get_limits("x") is None
    fig.axes().line([2], [3])
    fig.axes().set(xlim=(0, 4), ylim=(0, 6))
    assert (native.data[1].xaxis, native.data[1].yaxis) == ("x4", "y6")
    assert native.layout.xaxis4.range == (0, 4)
    assert native.layout.yaxis6.range == (0, 6)
    fig.close()


def test_prepared_range_and_sampling_marker_copy_mutable_inputs(spectrum):
    bounds = [-2, 4]
    prepared = prepare_spectrum(spectrum, xlim=bounds)
    bounds[:] = [-20, 40]
    assert prepared.x_range == (-2, 4)
    percentiles, errors = [10, 90], [0.1, 0.2]
    marker = HMarker(y=0.5, sampling=spectrum[0], err=errors, percentiles=percentiles)
    percentiles[:] = [5, 95]
    errors[:] = [1, 2]
    assert marker.percentiles == (10, 90)
    assert marker.band == (0.4, 0.7)


def test_native_default_point_size_matches_plotly_export_defaults():
    import plotly.graph_objects as go

    fig = slp.wrap(go.Figure())
    before = fig.native.layout.to_plotly_json()
    assert fig.axes().point_size(6, units="axes") == pytest.approx((6 * 100 / 72) / 270)
    assert fig.native.layout.to_plotly_json() == before
    fig.close()
