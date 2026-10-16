"""Legend composition, batched scientific hover and trace-free Plotly cells."""

from __future__ import annotations

import numpy as np
import pytest
from test_plotting_backends import stats
from test_plotting_correlators import _stack, fit_result
from test_plotting_shared import JACK, _series
from test_spectrum_plotter import levels

import slat.plotting as slp
from sigmondsamplings.data import SigmondData
from sigmondsamplings.plot_contracts import UncertaintyOptions
from sigmondsamplings.plotting import FitOverlay, HMarker, SamplingPlotter, SeriesPlotter
from sigmondsamplings.plotting._util import estimate_hover_text, full_sample_and_bounds
from sigmondsamplings.plotting.spectrum import prepare_spectrum
from sigmondsamplings.rcparams import rc_context

# Imported fixtures remain available to pytest in this module.
__all__ = ["stats", "fit_result"]


@pytest.mark.parametrize("central", ["full_sample_value", "mean", "median", "bias_corrected_mean"])
@pytest.mark.parametrize("error_metric", ["error", "confidence_interval"])
def test_batched_hover_matches_scalar_statistics(central, error_metric):
    data = _series()
    # The central values lie outside the CI; clipped offsets cannot recover it.
    data.data[:, 0] = [10, -10, 20]
    with rc_context({"plot.central_metric": central, "plot.error_metric": error_metric,
                     "plot.confidence_level": 0.9}):
        expected = [f"x<br>time={t:g}: {estimate_hover_text(data[i])}"
                    for i, t in enumerate(data.coords["time"])]
        with slp.record() as rec:
            SeriesPlotter(data, labels=["x"]).plot(yscale="log")
    actual = {x: text for mark in rec.of_kind("errorbar")
              for x, text in zip(mark.args[0], mark.kwargs["hover"])}
    assert [actual[t] for t in data.coords["time"]] == expected


@pytest.mark.parametrize("info", [None, JACK])
@pytest.mark.parametrize("central", ["full_sample_value", "mean", "median"])
def test_batched_hover_matches_bins_and_jackknife(info, central):
    data = _series(info)
    with rc_context({"plot.central_metric": central}), slp.record() as rec:
        SeriesPlotter(data, labels=["x"]).plot()
        assert rec.first("errorbar").kwargs["hover"] == [
            f"x<br>time={t:g}: {estimate_hover_text(data[i])}"
            for i, t in enumerate(data.coords["time"])
        ]


@pytest.mark.parametrize("backend", ["recording", "matplotlib", "plotly"])
def test_series_computes_ci_once_without_scalar_slices(monkeypatch, backend):
    data = _series()
    original = SigmondData.confidence_interval
    calls = []

    def confidence_interval(self, *args, **kwargs):
        calls.append(self.shape)
        return original(self, *args, **kwargs)

    def no_scalar_slices(*args, **kwargs):
        raise AssertionError("drawing must use the batched statistics")

    monkeypatch.setattr(SigmondData, "confidence_interval", confidence_interval)
    monkeypatch.setattr(SigmondData, "__getitem__", no_scalar_slices)
    with rc_context({"plot.error_metric": "confidence_interval"}):
        axes = SeriesPlotter(data).plot(backend=backend)
    axes.figure.close()
    assert calls == [(3,)]


def test_matplotlib_does_not_format_hover(monkeypatch):
    import sigmondsamplings.plotting.series as module

    def fail(*args, **kwargs):
        raise AssertionError("static drawing does not need hover")

    monkeypatch.setattr(module, "_hover", fail)
    SeriesPlotter(_series()).plot(backend="matplotlib").figure.close()


def test_spectrum_and_interval_marker_resolve_options_once(monkeypatch):
    collection = levels((0, 0.5, [0.4, 0.45, 0.55, 0.6], "ecm", None),
                        (1, 0.8, [0.7, 0.75, 0.85, 0.9], "ecm", None))
    original = UncertaintyOptions.resolved
    calls = []

    def resolved(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(UncertaintyOptions, "resolved", resolved)
    marker = HMarker.from_sampling(collection[0], percentiles=(10, 90))
    assert len(calls) == 1
    expected = full_sample_and_bounds(collection[0], percentiles=(10, 90))
    assert marker.y == expected[0]
    assert marker.band == expected[1]
    calls.clear()
    prepared = prepare_spectrum(collection, group_by=None, groups={"A1g": collection},
                                options={"percentiles": (10, 90)})
    assert len(calls) == 1
    assert prepared.levels[0][0].bounds == expected[1]


@pytest.mark.parametrize("backend", slp.available())
def test_legend_group_restores_outer_scope_on_failure(backend):
    fig = slp.figure(backend=backend)
    axes = fig.axes()
    try:
        with axes.legend_group():
            axes.line([0, 1], [1, 2], label="outer")
            with pytest.raises(ValueError), axes.legend_group():
                axes.line([0, 1], [2, 3], label="inner")
                raise ValueError("stop")
            axes.line([0, 1], [3, 4])
        axes.line([0, 1], [4, 5], label="separate")
        if backend == "plotly":
            outer, inner, continuation, separate = fig.native.data
            assert outer.legendgroup == continuation.legendgroup
            assert outer.legendgroup != inner.legendgroup
            assert separate.legendgroup is None
    finally:
        fig.close()


def test_interval_boxes_share_one_legend_group_per_level(stats):
    fig = SamplingPlotter(stats).plot_bootstrap_intervals(backend="plotly").figure
    try:
        centre, *bands = fig.native.data
        assert centre.legendgroup is None
        assert len(bands) == 6
        assert bands[0].legendgroup == bands[1].legendgroup == bands[2].legendgroup
        assert bands[3].legendgroup == bands[4].legendgroup == bands[5].legendgroup
        assert bands[0].legendgroup != bands[3].legendgroup
        assert [trace.name for trace in bands if trace.showlegend] == ["68% CI", "95% CI"]
    finally:
        fig.close()


@pytest.mark.parametrize("all_hollow", [False, True])
def test_series_groups_are_local_to_each_cell_even_with_duplicate_labels(all_hollow):
    fig = slp.figure(2, 1, backend="plotly")
    data = _stack().isel(operator=0)
    mask = np.ones(data.shape, dtype=bool) if all_hollow else np.arange(data.shape[0]) % 2 == 0
    plotter = SeriesPlotter(data, labels=["same"], hollow=[mask])
    try:
        for axes in fig.flat:
            plotter.plot(axes=axes)
            axes.legend()
        groups = [{trace.legendgroup for trace in fig.native.data if trace.yaxis == ref}
                  for ref in ("y", "y2")]
        assert len(groups[0]) == len(groups[1]) == 1
        assert groups[0].isdisjoint(groups[1])
        assert sum(bool(trace.showlegend) for trace in fig.native.data) == 2
    finally:
        fig.close()


def test_fit_band_and_central_line_share_a_toggle(fit_result):
    axes = SeriesPlotter(_stack().isel(operator=0)).plot(
        backend="plotly", fits=[FitOverlay(fit_result, label="fit")]
    )
    try:
        _, band, line = axes.figure.native.data
        assert band.legendgroup == line.legendgroup
        assert band.showlegend and not line.showlegend
    finally:
        axes.figure.close()


def test_axis_line_hover_keeps_its_own_item_inside_a_trace_group():
    fig = slp.figure(backend="plotly")
    axes = fig.axes()
    try:
        with axes.legend_group():
            axes.band([0, 1], [0, 0], [1, 1], label="fit")
            axes.axhline(0.5, label="reference", hover="reference hover")
            axes.line([0, 1], [0.5, 0.5])
        band, hover, line = fig.native.data
        shape = fig.native.layout.shapes[-1]
        assert band.legendgroup == line.legendgroup
        assert hover.legendgroup == shape.legendgroup != band.legendgroup
        assert hover.legend == shape.legend == "legend"
        assert shape.showlegend and not hover.showlegend
    finally:
        fig.close()


def test_legend_spacing_changes_only_for_composed_trace_groups():
    fig = slp.figure(backend="plotly")
    axes = fig.axes()
    try:
        axes.axhline(0.5, label="reference", hover="hover")
        axes.legend()
        assert fig.native.layout.legend.tracegroupgap is None
        with axes.legend_group():
            axes.line([0, 1], [1, 2], label="composed")
        axes.legend()
        assert fig.native.layout.legend.tracegroupgap == 0
    finally:
        fig.close()


def test_output_appends_only_missing_visible_cells_and_preserves_trace_indices(tmp_path, monkeypatch):
    fig = slp.figure(2, 2, sharex=True, sharey=True, backend="plotly")
    fig.axes().line([2, 8], [3, 9], label="data")
    fig.axes(0, 1).set(title="title only")
    fig.axes(1, 0).axhline(4)
    fig.hide(1, 1)
    first = fig.native.data[0]
    assert len(fig.native.data) == 1
    fig.embed_html()
    assert fig.native.data[0] is first
    assert [(t.xaxis, t.yaxis) for t in fig.native.data[1:]] == [("x2", "y2"), ("x3", "y3")]
    for trace in fig.native.data[1:]:
        assert trace.x == trace.y == (None,)
        assert trace.showlegend is False and trace.hoverinfo == "skip"
    fig.save(tmp_path / "empty.html")
    monkeypatch.setattr(fig.native, "show", lambda: None)
    fig.show()
    assert len(fig.native.data) == 3
    assert (tmp_path / "empty.html").is_file()
    fig.close()


def test_hidden_native_traces_do_not_count_as_visible_cell_references():
    fig = slp.figure(backend="plotly")
    try:
        fig.axes().line([1, 2], [3, 4], native={"visible": False})
        fig.embed_html()
        assert len(fig.native.data) == 2
        assert fig.native.data[1].meta == {"slat_empty_cell": True}
        fig.hide(0, 0)
        fig.embed_html()
        assert len(fig.native.data) == 2
    finally:
        fig.close()
