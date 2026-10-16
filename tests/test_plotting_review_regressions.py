"""Backend edge cases found in the plotting refactor review."""

import numpy as np
import pytest
from test_plotting_correlators import _stack

import slat.plotting as slp
from sigmondsamplings.plot_contracts import FigureStyle
from sigmondsamplings.plotting import SeriesPlotter, SeriesStyle


@pytest.mark.parametrize(
    ("field", "keyword", "values"),
    [("color_cycle", "color", ("red", "blue")), ("marker_cycle", "marker", ("o", "s"))],
)
def test_series_style_palettes_are_reusable_when_given_iterators(field, keyword, values):
    plotter = SeriesPlotter(_stack(), style=SeriesStyle(**{field: iter(values)}))
    for _ in range(2):
        with slp.record() as rec:
            plotter.plot()
        assert [mark.kwargs[keyword] for mark in rec.of_kind("errorbar")] == [
            values[0], values[1], values[0],
        ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_figure_geometry_rejects_nonfinite_and_nonpositive_values(value):
    for kwargs in ({"figsize": (value, 4)}, {"figsize": (4, value)}, {"dpi": value}):
        with pytest.raises(ValueError):
            FigureStyle(**kwargs)


@pytest.mark.parametrize("backend", slp.available())
def test_resize_updates_the_style_seen_by_axes(backend):
    fig = slp.figure(backend=backend)
    try:
        fig.set_size(8, 5)
        assert fig.style.figsize == (8, 5)
        assert fig.axes().style is fig.style
    finally:
        fig.close()


@pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
def test_suptitle_uses_figure_style_and_explicit_override(backend):
    fig = slp.figure(backend=backend, style=slp.PlotStyle(title_fontsize=19))
    try:
        for override, expected in ((None, 19), (11, 11)):
            fig.suptitle("title", fontsize=override)
            if backend == "matplotlib":
                assert fig.native._suptitle.get_fontsize() == expected
            else:
                assert fig.native.layout.title.font.size == pytest.approx(expected * 100 / 72)
    finally:
        fig.close()


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
def test_plotly_omits_markers_when_none_requested():
    fig = slp.figure(backend="plotly")
    axes = fig.axes()
    axes.line([0, 1], [1, 2], linestyle="none", marker=None)
    axes.errorbar([0, 1], [1, 2], yerr=[0.1, 0.2], marker=None)
    axes.errorbar([0, 1], [1, 2], yerr=[0.1, 0.2], marker=None, linestyle="-")
    assert [trace.mode for trace in fig.native.data] == ["none", "none", "lines"]
    np.testing.assert_allclose(fig.native.data[1].error_y.array, [0.1, 0.2])


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
def test_plotly_ticks_can_reset_rotation_and_accept_array_limits():
    fig = slp.figure(backend="plotly")
    axes = fig.axes()
    axes.ticks("x", [0], ["first"], rotation=45)
    axes.ticks("x", [0], ["second"])
    assert fig.native.layout.xaxis.tickangle == 0
    axes.set(ylim=np.array([1, 10]))
    assert axes.get_limits("y") == (1, 10)


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
@pytest.mark.parametrize("bounds", [{"vmin": 0}, {"vmax": 100}, {"vmin": 0, "vmax": 100}])
def test_plotly_heatmap_preserves_explicit_color_bounds(bounds):
    fig = slp.figure(backend="plotly")
    fig.axes().heatmap([[55, 56]], **bounds)
    trace = fig.native.data[0]
    assert trace.zauto is False
    assert trace.zmin == bounds.get("vmin")
    assert trace.zmax == bounds.get("vmax")


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
def test_plotly_gray_colormap_runs_from_black_to_white():
    fig = slp.figure(backend="plotly")
    fig.axes().heatmap([[0, 1]], cmap="gray")
    scale = fig.native.data[0].colorscale
    assert scale[0] == (0.0, "rgb(0,0,0)")
    assert scale[-1] == (1.0, "rgb(255,255,255)")
