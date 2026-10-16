"""Tests for the backend-neutral plotting API (``slat.plotting``).

Three layers:

* the core -- registry, style, text, mark translation
* structural tests of the sigmondsamplings frontends via the recording backend,
  which assert *what* gets drawn without rendering anything
* smoke tests on whichever real backends are installed

Run with:  python -m pytest tests/test_plotting_backends.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import slat.plotting as slp
from sigmondsamplings.info import ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats
from sigmondsamplings.utils import stacked_positions

REAL_BACKENDS = [b for b in slp.available() if b != "recording"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stats() -> SamplingStats:
    """Three synthetic bootstrap observables."""
    rng = np.random.default_rng(7)
    info = SamplingInfo("bootstrap", 200, seed=1)
    samplings = []
    for i in range(3):
        mean = 1.0 + 0.5 * i
        data = np.empty(201)
        data[0] = mean
        data[1:] = rng.normal(mean, 0.05, 200)
        samplings.append(SigmondSampling(data, ObservableInfo(name=f"obs{i}", index=i), info))
    return SamplingStats(samplings)


@pytest.fixture
def energy_collection():
    """Eight synthetic ecm levels across two PSQ sectors and four irreps."""
    from sigmondsamplings.energy_level_collection import SingleEnsembleEnergyCollection
    from sigmondsamplings.energy_levels import EnergyObsInfo

    rng = np.random.default_rng(3)
    info = SamplingInfo("bootstrap", 50, seed=1)
    samplings = []
    index = 0
    for psq, irreps in ((0, ("A1g", "T1u")), (1, ("A1", "E"))):
        for irrep in irreps:
            for level in range(2):
                mean = 0.5 + 0.1 * level + 0.05 * psq
                data = np.empty(51)
                data[0] = mean
                data[1:] = rng.normal(mean, 0.01, 50)
                samplings.append(
                    SigmondSampling(
                        data,
                        EnergyObsInfo(
                            index=index,
                            irrep=irrep,
                            psq=psq,
                            energy_type="ecm",
                            level_index=level,
                        ),
                        info,
                    )
                )
                index += 1
    return SingleEnsembleEnergyCollection(samplings)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_recording_is_always_available(self):
        assert "recording" in slp.available()

    def test_default_backend_is_matplotlib(self):
        assert slp.current() == "matplotlib"

    def test_backend_context_restores_previous(self):
        before = slp.current()
        with slp.backend("recording"):
            assert slp.current() == "recording"
        assert slp.current() == before

    def test_backend_context_restores_after_exception(self):
        before = slp.current()
        with pytest.raises(RuntimeError), slp.backend("recording"):
            raise RuntimeError("boom")
        assert slp.current() == before

    def test_use_returns_previous_name(self):
        previous = slp.use("recording")
        try:
            assert slp.current() == "recording"
        finally:
            assert slp.use(previous) == "recording"

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            slp.use("ggplot")

    @pytest.mark.parametrize("name", slp.available())
    def test_set_size_resizes_in_inches(self, name):
        fig = slp.figure(2, 1, figsize=(4.0, 3.0), backend=name)
        fig.set_size(8.0, 5.0)
        if name == "matplotlib":
            assert tuple(fig.native.get_size_inches()) == (8.0, 5.0)
        elif name == "plotly":
            from slat.plotting._plotly import PX_PER_INCH

            layout = fig.native.layout
            assert (layout.width, layout.height) == (8 * PX_PER_INCH, 5 * PX_PER_INCH)
            # Row domains are laid out again for the new height.
            assert fig.native.layout.yaxis.domain != slp.figure(
                2, 1, figsize=(4.0, 3.0), backend=name
            ).native.layout.yaxis.domain
        else:
            assert fig.style.figsize == (8.0, 5.0)
        fig.close()

    def test_use_checks_the_dependency(self, monkeypatch):
        import importlib.util

        find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: None if name == "plotly" else find_spec(name)
        )
        with pytest.raises(ImportError, match="needs plotly"):
            slp.use("plotly")
        assert slp.current() == "matplotlib"

    def test_show_blocks_once_for_matplotlib_and_shows_the_rest(self, monkeypatch):
        shown = []
        figures = [slp.figure(backend=b) for b in ("matplotlib", "recording", "matplotlib")]
        for fig in figures:
            monkeypatch.setattr(fig, "show", lambda fig=fig: shown.append(fig))
        slp.show(figures)
        assert shown == [figures[1], figures[0]]
        for fig in figures:
            fig.close()

    def test_figure_backend_argument_does_not_leak(self):
        fig = slp.figure(backend="recording")
        assert fig.backend == "recording"
        assert slp.current() == "matplotlib"


# ---------------------------------------------------------------------------
# Recording backend
# ---------------------------------------------------------------------------


class TestRecording:
    def test_records_marks_in_order(self):
        with slp.record() as rec:
            axes = slp.figure().axes()
            axes.line([0, 1], [0, 1], color="red", label="a")
            axes.band([0, 1], [0, 0], [1, 1], alpha=0.3)
            axes.errorbar([0], [1], yerr=[0.1])
        assert rec.kinds() == ["line", "band", "errorbar"]

    def test_mark_kwargs_are_captured(self):
        with slp.record() as rec:
            slp.figure().axes().band([0, 1], [0, 0], [1, 1], color="blue", alpha=0.22)
        assert rec.first("band").kwargs["alpha"] == 0.22
        assert rec.first("band").kwargs["color"] == "blue"

    def test_labels_helper(self):
        with slp.record() as rec:
            axes = slp.figure().axes()
            axes.line([0], [0], label="one")
            axes.line([0], [0])
            axes.line([0], [0], label="two")
        assert rec.labels() == ["one", "two"]

    def test_first_raises_with_a_useful_message(self):
        with slp.record() as rec:
            slp.figure().axes().line([0], [0])
        with pytest.raises(AssertionError, match="No 'heatmap' mark"):
            rec.first("heatmap")

    def test_cells_are_tracked_for_grids(self):
        with slp.record() as rec:
            fig = slp.figure(2, 2)
            fig.axes(0, 0).line([0], [0])
            fig.axes(1, 1).bar([0], [1])
        assert rec.first("line").cell == (0, 0)
        assert rec.first("bar").cell == (1, 1)

    def test_record_restores_the_previous_backend(self):
        before = slp.current()
        with slp.record():
            assert slp.current() == "recording"
        assert slp.current() == before

    def test_marks_record_resolved_defaults(self):
        with slp.record() as rec:
            slp.figure().axes().hlines(1.0, 0, 1)
        kwargs = rec.first("hlines").kwargs
        assert kwargs["linestyle"] == "-"
        assert kwargs["linewidth"] is None and kwargs["hover"] is None

    def test_figure_size_reaches_the_style(self):
        with slp.record():
            fig = slp.figure(figsize=(3, 2))
        assert fig.style.figsize == (3, 2)


# ---------------------------------------------------------------------------
# Style and text
# ---------------------------------------------------------------------------


class TestStyle:
    def test_color_cycle_is_restartable(self):
        style = slp.PlotStyle(colors=["a", "b"])
        first = style.color_cycle()
        assert [next(first) for _ in range(3)] == ["a", "b", "a"]
        assert next(style.color_cycle()) == "a"

    def test_style_context_restores(self):
        before = slp.get_style()
        with slp.style_context(grid_alpha=0.9) as active:
            assert active.grid_alpha == 0.9
        assert slp.get_style() is before

    def test_replace_leaves_the_original_alone(self):
        base = slp.PlotStyle()
        assert base.replace(dpi=300).dpi == 300
        assert base.dpi == 100.0


class TestText:
    def test_ensure_math_wraps_once(self):
        assert slp.ensure_math("E_0") == "$E_0$"
        assert slp.ensure_math("$E_0$") == "$E_0$"


# ---------------------------------------------------------------------------
# plotly translation helpers
# ---------------------------------------------------------------------------


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
class TestPlotlyTranslation:
    def test_hex_and_named_colors(self):
        from slat.plotting._plotly._convert import _color

        assert _color("#d60000") == "rgb(214,0,0)"
        assert _color("black", 0.5) == "rgba(0,0,0,0.5)"

    def test_matplotlib_greyscale_strings(self):
        from slat.plotting._plotly._convert import _to_rgb

        assert _to_rgb("0.0") == (0, 0, 0)
        assert _to_rgb("1.0") == (255, 255, 255)

    def test_unknown_color_survives_without_alpha(self):
        from slat.plotting._plotly._convert import _color

        assert _color("rebeccapurple", 0.5) == "rebeccapurple"

    def test_marker_and_dash_maps(self):
        from slat.plotting._plotly._convert import _dash, _marker

        assert _marker("o") == "circle"
        assert _dash("--") == "dash"

    def test_hover_text_converts_only_lines_with_math(self):
        from slat.plotting._plotly._convert import _hover_text

        assert _hover_text("Ecm/M_N") == "Ecm/M_N"
        assert _hover_text("$E_{0}$<br>Ecm/M_N") == "E<sub>0</sub><br>Ecm/M_N"

    def test_hover_reaches_every_mark_as_html(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.points([0], [1], hover=["$E_0$"])
        axes.hlines([1.0], 0, 1, hover=["$E_1$"])
        axes.axhline(0.5, hover="$E_2$")
        axes.heatmap(np.eye(2), hover=[["$a_1$", "b"], ["c", "d"]])
        points, hlines, axline, heatmap = fig.native.data
        assert points.hovertext == ("E<sub>0</sub>",)
        assert set(hlines.hovertext) == {"E<sub>1</sub>"}
        assert hlines.meta["slat_hover_segments"][0][-1] == "E<sub>1</sub>"
        assert set(axline.hovertext) == {"E<sub>2</sub>"}
        assert list(heatmap.text[0]) == ["a<sub>1</sub>", "b"]

    def test_asymmetric_errors(self):
        from slat.plotting._plotly._convert import _err

        spec = _err(np.array([[0.1, 0.2], [0.3, 0.4]]), 2)
        assert spec["symmetric"] is False
        np.testing.assert_allclose(spec["arrayminus"], [0.1, 0.2])
        np.testing.assert_allclose(spec["array"], [0.3, 0.4])

    def test_scalar_error_is_broadcast(self):
        from slat.plotting._plotly._convert import _err

        np.testing.assert_allclose(_err(0.5, 3)["array"], [0.5, 0.5, 0.5])

    def test_fill_false_picks_the_open_symbol(self):
        from slat.plotting._plotly._convert import _marker

        assert _marker("o", fill=False) == "circle-open"
        assert _marker("circle-open", fill=False) == "circle-open"
        assert _marker("s") == "square"

    def test_fill_reaches_every_marker_mark(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.points([0], [1], fill=False)
        axes.errorbar([0], [1], yerr=[0.1], fill=False)
        axes.line([0, 1], [1, 2], marker="s", fill=False)
        axes.errorbar([0], [1], yerr=[0.1])
        symbols = [trace.marker.symbol for trace in fig.native.data]
        assert symbols == ["circle-open", "circle-open", "square-open", "circle"]

    def test_log_axis_limits_are_given_to_plotly_in_log10(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.set(yscale="log", ylim=(1e-3, 10.0))
        assert fig.native.layout.yaxis.range == (-3.0, 1.0)
        axes.set(ylim=(1e-2, 1.0))  # the axis is already log
        assert fig.native.layout.yaxis.range == (-2.0, 0.0)
        axes.set(xlim=(0.0, 5.0))  # linear x is untouched
        assert fig.native.layout.xaxis.range == (0.0, 5.0)
        with pytest.raises(ValueError, match="positive"):
            axes.set(ylim=(0.0, 1.0))

    def test_heatmap_hover_becomes_cell_text(self):
        fig = slp.figure(backend="plotly")
        hover = [["a", "b"], ["c", "d"]]
        fig.axes().heatmap(np.eye(2), hover=hover)
        trace = fig.native.data[0]
        assert [list(row) for row in trace.text] == hover
        assert trace.hoverinfo == "text"

    def test_heatmap_without_hover_keeps_the_default(self):
        fig = slp.figure(backend="plotly")
        fig.axes().heatmap(np.eye(2))
        assert fig.native.data[0].text is None

    def test_nan_heatmap_cells_stay_nan(self):
        fig = slp.figure(backend="plotly")
        values = np.array([[1.0, np.nan], [0.5, 0.2]])
        fig.axes().heatmap(values)
        assert np.isnan(np.asarray(fig.native.data[0].z, dtype=float)[0, 1])

    def test_tick_labels_preserve_whole_subscripts(self):
        fig = slp.figure(backend="plotly")
        fig.axes().ticks("x", [0, 1, 2], ["$A_1$", "$T_{1g}$", "$T_{1u}^{-}$"])
        assert fig.native.layout.xaxis.ticktext == (
            "A<sub>1</sub>", "T<sub>1g</sub>", "T<sub>1u</sub><sup>-</sup>"
        )

    def test_escaped_underscore_is_not_a_subscript(self):
        from slat.plotting.text import to_html

        assert to_html(r"$\text{eta(0)\_ref}$") == "eta(0)_ref"
        assert to_html(r"$68\%$") == "68%"
        fig = slp.figure(backend="plotly")
        fig.axes().ticks("x", [0], [r"$\text{K\_star}$"])
        assert fig.native.layout.xaxis.ticktext == ("K_star",)

    def test_axis_line_on_an_empty_subplot_is_kept(self):
        """add_vline skips subplots with no traces unless told otherwise."""
        fig = slp.figure(1, 2, backend="plotly")
        fig.axes(0, 1).axvline(0.5)
        fig.axes(0, 1).axhline(0.5)
        lines = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert len(lines) == 2
        assert lines[0].xref == "x2"

    def test_none_style_draws_segments_and_axis_lines_at_zero_width(self):
        """As on matplotlib, the mark is kept (legend, hover) but not drawn."""
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.hlines([1.0], 0, 1, linestyle="none", label="h", hover=["h"])
        axes.vlines([0.5], 0, 1, linestyle="none")
        axes.axhline(0.5, linestyle="none", label="a", hover="a")
        hlines, vlines = fig.native.data[:2]
        assert (hlines.line.width, hlines.line.dash) == (0, None)
        assert hlines.showlegend and hlines.hovertext
        assert vlines.line.width == 0
        (shape,) = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert (shape.line.width, shape.line.dash) == (0, None)
        assert shape.showlegend

    def test_alpha_without_a_color_becomes_trace_opacity(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.bar([0, 1], [1, 2], alpha=0.7)
        axes.errorbar([0], [1], yerr=[0.1], alpha=0.5)
        axes.line([0, 1], [0, 1], color="#d60000", alpha=0.5)
        axes.axvline(0.5, alpha=0.4)
        bar, errorbar, line = fig.native.data
        assert (bar.opacity, errorbar.opacity) == (0.7, 0.5)
        # A known color carries the alpha itself.
        assert line.opacity is None and line.line.color == "rgba(214,0,0,0.5)"
        (line_shape,) = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert line_shape.opacity == 0.4

    @pytest.mark.parametrize(
        ("style_dpi", "dpi", "scale"), [(100, None, 1.0), (200, None, 2.0), (100, 300, 3.0), (200, 100, 1.0)]
    )
    def test_dpi_sets_only_the_export_resolution(self, monkeypatch, style_dpi, dpi, scale):
        """The layout is 100 px per inch whatever the dpi, as savefig(dpi=) leaves a figure alone."""
        with slp.style_context(dpi=style_dpi):
            fig = slp.figure(figsize=(4, 3), backend="plotly")
        assert (fig.native.layout.width, fig.native.layout.height) == (400, 300)
        seen = {}
        monkeypatch.setattr(fig.native, "write_image", lambda path, scale: seen.update(scale=scale))
        fig.save("out.png", dpi=dpi)
        assert seen["scale"] == scale

    def test_horizontal_segments_have_independent_hover_text(self):
        fig = slp.figure(backend="plotly")
        fig.axes().hlines([2, 3], 0, 1, hover=["N(0) + N(0)", "N(1) + N(1)"])
        trace = fig.native.data[0]
        assert trace.hoverinfo == "text"
        assert trace.hovertext == ("N(0) + N(0)",) * 4 + ("N(1) + N(1)",) * 4
        assert trace.x == (0, 0.5, 1, None) * 2
        assert trace.meta["slat_hover_segments"] == [
            [2, 0, 1, "N(0) + N(0)"], [3, 0, 1, "N(1) + N(1)"]
        ]
        with pytest.raises(ValueError, match="one string per segment"):
            fig.axes().hlines([2, 3], 0, 1, hover=["only one"])

    def test_axis_lines_answer_hover_on_a_hidden_overlay(self):
        fig = slp.figure(1, 2, backend="plotly")
        right = fig.axes(0, 1)
        right.axvline(2.0, color="red", label="a", hover="x = 2")
        right.axvline(3.0, hover="x = 3")
        right.axhline(1.0, hover="y = 1")
        layout = fig.native.layout
        # The shape is still drawn; one invisible trace per line carries the hover.
        assert len(layout.shapes) >= 3
        v, _, h = fig.native.data
        assert set(v.x) == {2.0} and (min(v.y), max(v.y)) == (0.0, 1.0)
        assert v.hovertext[0] == "x = 2" and v.showlegend is False
        assert v.legendgroup == next(s.legendgroup for s in layout.shapes if s.name == "a")
        # Both vertical lines share one overlay y axis on the right subplot.
        assert v.xaxis == "x2" and fig.native.data[1].yaxis == v.yaxis
        overlay = layout[v.yaxis.replace("y", "yaxis")]
        assert overlay.overlaying == "y2" and overlay.range == (0, 1) and overlay.visible is False
        assert h.yaxis == "y2" and layout[h.xaxis.replace("x", "xaxis")].overlaying == "x2"

    @pytest.mark.parametrize(
        ("x", "y", "side"),
        [
            (0.5, -0.12, "b"),
            (0.5, 1.12, "t"),
            (-0.12, 0.5, "l"),
            (1.12, 0.5, "r"),
        ],
    )
    def test_out_of_domain_annotation_reserves_margin(self, x, y, side):
        """plotly clips out-of-domain text; matplotlib makes room for it."""
        fig = slp.figure(backend="plotly")
        before = fig.native.layout.margin[side]
        fig.axes().text(x, y, "d^2 = 0", coords="axes")
        assert fig.native.layout.margin[side] > before

    def test_x_margin_allowance_grows_with_the_text(self):
        margins = []
        for text in ("a", "a much longer label"):
            fig = slp.figure(backend="plotly")
            fig.axes().text(1.05, 0.5, text, coords="axes")
            margins.append(fig.native.layout.margin.r)
        assert margins[1] > margins[0]

    def test_only_the_out_of_domain_axis_is_reserved(self):
        fig = slp.figure(backend="plotly")
        before = fig.native.layout.margin.b
        fig.axes().text(1.12, 0.5, "right", coords="axes")
        assert fig.native.layout.margin.b == before

    def test_in_domain_annotation_leaves_margins_alone(self):
        fig = slp.figure(backend="plotly")
        margin = fig.native.layout.margin
        before = (margin.l, margin.r, margin.t, margin.b)
        fig.axes().text(0.5, 0.5, "inside", coords="axes")
        margin = fig.native.layout.margin
        assert (margin.l, margin.r, margin.t, margin.b) == before

    def test_data_coordinates_never_touch_margins(self):
        fig = slp.figure(backend="plotly")
        margin = fig.native.layout.margin
        before = (margin.l, margin.r, margin.t, margin.b)
        fig.axes().text(-5.0, 99.0, "far away", coords="data")
        margin = fig.native.layout.margin
        assert (margin.l, margin.r, margin.t, margin.b) == before


@pytest.mark.skipif("matplotlib" not in slp.available(), reason="matplotlib not installed")
class TestMatplotlibTranslation:
    def test_fill_false_hollows_errorbar_and_line_markers(self):
        from matplotlib.colors import to_rgba

        fig = slp.figure(backend="matplotlib")
        axes = fig.axes()
        axes.errorbar([0], [1], yerr=[0.1], color="#d60000", fill=False)
        axes.line([0, 1], [1, 2], marker="o", color="#d60000", fill=False)
        errorbar_line, line = axes.native.get_lines()
        for artist in (errorbar_line, line):
            assert artist.get_markerfacecolor() == "none"
            assert to_rgba(artist.get_markeredgecolor()) == to_rgba("#d60000")
        fig.close()

    def test_fill_false_hollows_scatter_points(self):
        from matplotlib.colors import to_rgba

        fig = slp.figure(backend="matplotlib")
        fig.axes().points([0, 1], [1, 2], color="#d60000", fill=False)
        (collection,) = fig.axes().native.collections
        assert collection.get_facecolors().size == 0
        np.testing.assert_allclose(collection.get_edgecolors()[0], to_rgba("#d60000"))
        fig.close()

    def test_filled_is_the_default(self):
        fig = slp.figure(backend="matplotlib")
        fig.axes().errorbar([0], [1], yerr=[0.1], color="#d60000")
        assert fig.axes().native.get_lines()[0].get_markerfacecolor() != "none"
        fig.close()

    def test_nan_heatmap_cells_are_masked(self):
        fig = slp.figure(backend="matplotlib")
        fig.axes().heatmap(np.array([[1.0, np.nan], [0.5, 0.2]]), hover=[["a", "b"], ["c", "d"]])
        (image,) = fig.axes().native.get_images()
        assert np.ma.is_masked(image.get_array()[0, 1])
        fig.close()

    def test_upper_origin_with_coordinates_keeps_each_row_at_its_y(self):
        """values[i] is drawn at y[i], with y[0] at the top, as on plotly."""
        fig = slp.figure(backend="matplotlib")
        axes = fig.axes()
        axes.heatmap(np.array([[1.0, 2.0], [3.0, 4.0]]), x=[0, 1], y=[10, 20], origin="upper")
        (mesh,) = axes.native.collections
        np.testing.assert_allclose(np.asarray(mesh.get_array()).ravel(), [1, 2, 3, 4])
        bottom, top = axes.native.get_ylim()
        assert bottom > top  # inverted: y=10 (row 0) at the top
        fig.close()

    def test_style_grid_reaches_new_axes(self):
        with slp.style_context(grid=True):
            fig = slp.figure(backend="matplotlib")
        assert fig.axes().native.xaxis._major_tick_kw.get("gridOn")
        fig.close()
        assert not slp.figure(backend="matplotlib").axes().native.xaxis._major_tick_kw.get("gridOn")

    def test_suptitle_fontsize(self):
        fig = slp.figure(backend="matplotlib")
        fig.suptitle("title", fontsize=17)
        assert fig.native.get_suptitle() == "title"
        assert fig.native._suptitle.get_fontsize() == 17
        fig.close()


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
class TestPlotlyMatchesMatplotlib:
    """The plotly figure is laid out to look like matplotlib's."""

    @staticmethod
    def _px(points):
        from slat.plotting._plotly import PX_PER_INCH

        return points * PX_PER_INCH / 72.0

    # -- text ---------------------------------------------------------------

    def test_mixed_math_becomes_one_mathjax_span(self):
        from slat.plotting.text import to_plotly

        assert to_plotly(r"$68\%$ CI") == r"$68\%\text{ CI}$"
        assert to_plotly(r"Mixed $x$ label") == r"$\text{Mixed }x\text{ label}$"
        assert to_plotly(r"$\text{eta\_ref} = 1.0(4)$") == r"$\text{eta_ref} = 1.0(4)$"
        assert to_plotly("plain\ntwo lines") == "plain<br>two lines"
        # MathJax cannot break a line, so multi-line math falls back to HTML.
        assert to_plotly("$E_0$\n(Bootstrap)") == "E<sub>0</sub><br>(Bootstrap)"
        assert to_plotly(None) is None

    def test_plain_text_that_would_break_a_mathjax_span_falls_back_to_html(self):
        from slat.plotting.text import to_plotly

        # <br> is a line break like \n, never text inside the span.
        assert to_plotly("$E_0$<br>second") == "E<sub>0</sub><br>second"
        assert to_plotly("plain<br>two") == "plain<br>two"
        # An unbalanced brace in plain text blanks the whole MathJax label; the
        # HTML fallback keeps the text, reading the brace as TeX grouping.
        assert to_plotly("$x$ open { brace") == "x open brace"
        # MathJax prints these literally inside \text{}, so the span is kept.
        assert to_plotly(r"$x$ 50% A & B #1") == r"$x\text{ 50% A & B #1}$"

    def test_labels_are_routed_by_position(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.line([0, 1], [0, 1], label=r"$68\%$ CI")
        axes.set(xlabel=r"time $t$", ylabel="$C(t)$")
        trace = fig.native.data[0]
        assert trace.name == "68% CI"  # legend: HTML, MathJax is unreliable there
        assert fig.native.layout.xaxis.title.text == r"$\text{time }t$"
        assert fig.native.layout.yaxis.title.text == "$C(t)$"

    # -- legends ------------------------------------------------------------

    def test_no_legend_until_asked(self):
        fig = slp.figure(1, 2, backend="plotly")
        fig.axes(0, 0).line([0, 1], [0, 1], label="a")
        fig.axes(0, 1).line([0, 1], [0, 1], label="b")
        layout = fig.native.layout
        assert layout.legend.visible is False and layout.legend2.visible is False
        fig.axes(0, 1).legend()
        assert layout.legend.visible is False and layout.legend2.visible is True

    def test_each_axes_has_its_own_legend_inside_it(self):
        fig = slp.figure(1, 2, backend="plotly")
        right = fig.axes(0, 1)
        right.line([0, 1], [0, 1], label="b")
        right.legend(loc="lower left")
        assert fig.native.data[0].legend == "legend2"
        legend = fig.native.layout.legend2
        x0, x1 = fig.native.layout.xaxis2.domain
        assert x0 < legend.x < x1 and (legend.xanchor, legend.yanchor) == ("left", "bottom")

    def test_labelled_axis_lines_are_legend_entries(self):
        fig = slp.figure(backend="plotly")
        fig.axes().axvline(0.5, label="Mean")
        (shape,) = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert (shape.name, shape.showlegend, shape.legend) == ("Mean", True, "legend")
        assert not fig.native.layout.annotations  # no floating label text

    def test_legend_follows_its_subplot_when_rows_move(self):
        fig = slp.figure(2, 1, backend="plotly")
        top = fig.axes(0, 0)
        top.line([0, 1], [0, 1], label="a")
        top.legend(loc="lower right")
        before = fig.native.layout.legend.y
        fig.axes(1, 0).set(title="a tall\ntwo-line title")
        assert fig.native.layout.legend.y == pytest.approx(fig.native.layout.yaxis.domain[0] + 0.01 * (
            fig.native.layout.yaxis.domain[1] - fig.native.layout.yaxis.domain[0]))
        assert fig.native.layout.legend.y > before

    # -- sizes and look -----------------------------------------------------

    def test_points_become_pixels_with_matplotlibs_defaults(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.line([0, 1], [0, 1])
        axes.errorbar([0], [1], yerr=[0.1], markersize=5, capsize=3, linewidth=1.2)
        line, errorbar = fig.native.data
        assert line.line.width == pytest.approx(self._px(1.5))
        assert errorbar.marker.size == pytest.approx(self._px(5))
        assert errorbar.error_y.width == pytest.approx(self._px(3))
        assert errorbar.error_y.thickness == pytest.approx(self._px(1.2))
        assert fig.native.layout.font.size == pytest.approx(self._px(10))

    def test_errorbars_have_no_caps_by_default(self):
        fig = slp.figure(backend="plotly")
        fig.axes().errorbar([0], [1], yerr=[0.1])
        assert fig.native.data[0].error_y.width == 0

    def test_no_grid_or_zero_line_by_default_and_a_frame(self):
        fig = slp.figure(1, 2, backend="plotly")
        layout = fig.native.layout
        assert layout.xaxis.showgrid is False and layout.yaxis2.zeroline is False
        frames = [s.name for s in layout.shapes if s.type == "rect"]
        assert frames == ["slat-frame", "slat-frame2"]
        fig.hide(0, 1)
        assert [s.name for s in fig.native.layout.shapes] == ["slat-frame"]

    def test_grid_on_one_axis_only(self):
        fig = slp.figure(backend="plotly")
        fig.axes().grid(True, axis="y")
        assert fig.native.layout.yaxis.showgrid is True
        assert fig.native.layout.xaxis.showgrid is False

    def test_bars_have_no_outline(self):
        fig = slp.figure(backend="plotly")
        fig.axes().bar([0], [1], color="#1f77b4")
        assert fig.native.data[0].marker.line.width == 0

    def test_axis_lines_are_opaque_despite_the_template(self):
        fig = slp.figure(backend="plotly")
        fig.axes().axhline(0.5, color="red")
        (shape,) = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert shape.opacity == 1.0

    def test_axis_labels_default_to_15_points_on_both_backends(self):
        assert slp.get_style().label_fontsize == 15.0
        fig = slp.figure(backend="plotly")
        fig.axes().set(xlabel="x", ylabel="y")
        assert fig.native.layout.xaxis.title.font.size == pytest.approx(self._px(15))
        assert fig.native.layout.yaxis.title.font.size == pytest.approx(self._px(15))
        if "matplotlib" in slp.available():
            mpl = slp.figure(backend="matplotlib")
            mpl.axes().set(xlabel="x", ylabel="y")
            assert mpl.axes().native.xaxis.label.get_fontsize() == 15.0
            assert mpl.axes().native.yaxis.label.get_fontsize() == 15.0
            mpl.close()

    def test_font_draws_a_readable_pi(self):
        """plotly's default Open Sans/Verdana draws Greek pi like Cyrillic "п"."""
        family = slp.figure(backend="plotly").native.layout.font.family
        assert family.split(",")[0].strip() == "Arial"

    def test_powers_of_ten_not_si_prefixes(self):
        fig = slp.figure(backend="plotly")
        assert fig.native.layout.yaxis.exponentformat == "power"

    # -- stacking -----------------------------------------------------------

    def test_zorder_and_matplotlibs_default_order(self):
        fig = slp.figure(backend="plotly")
        axes = fig.axes()
        axes.errorbar([0], [1], yerr=[0.1])
        axes.band([0, 1], [0, 0], [1, 1], color="red")
        axes.band([0, 1], [0, 0], [1, 1], color="red", zorder=0)
        axes.line([0, 1], [0, 1], zorder=2.5)
        errorbar, band, back, line = fig.native.data
        assert band.zorder < errorbar.zorder  # patches under lines, as on matplotlib
        assert back.zorder < band.zorder < line.zorder

    def test_low_zorder_axis_lines_go_under_the_traces(self):
        fig = slp.figure(backend="plotly")
        fig.axes().axvline(0.5, zorder=0)
        fig.axes().axvline(0.6)
        low, default = [s for s in fig.native.layout.shapes if s.type == "line"]
        assert (low.layer, default.layer) == ("below", "above")

    # -- titles and layout --------------------------------------------------

    def test_axes_title_is_its_own_and_survives_suptitle(self):
        fig = slp.figure(backend="plotly")
        fig.axes().set(title="axes title")
        fig.suptitle("figure title")
        titles = [a.text for a in fig.native.layout.annotations]
        assert titles == ["axes title"]
        assert fig.native.layout.title.text == "figure title"
        fig.axes().set(title="replaced")
        assert [a.text for a in fig.native.layout.annotations] == ["replaced"]

    def test_titles_make_room_above_the_axes(self):
        fig = slp.figure(backend="plotly")
        bare = fig.native.layout.margin.t
        fig.axes().set(title="axes")
        with_axes_title = fig.native.layout.margin.t
        fig.suptitle("figure", fontsize=16)
        assert bare < with_axes_title < fig.native.layout.margin.t
        assert fig.native.layout.title.font.size == pytest.approx(self._px(16))

    def test_row_gap_grows_for_a_lower_title(self):
        fig = slp.figure(2, 1, backend="plotly")
        layout = fig.native.layout
        gap = layout.yaxis.domain[0] - layout.yaxis2.domain[1]
        fig.axes(1, 0).set(title="first\nsecond")
        assert layout.yaxis.domain[0] - layout.yaxis2.domain[1] > gap
        assert layout.yaxis.domain[1] == 1.0 and layout.yaxis2.domain[0] == 0.0

    def test_shared_x_rows_sit_close(self):
        shared = slp.figure(2, 1, sharex=True, backend="plotly").native.layout
        separate = slp.figure(2, 1, backend="plotly").native.layout
        assert (shared.yaxis.domain[0] - shared.yaxis2.domain[1]) < (
            separate.yaxis.domain[0] - separate.yaxis2.domain[1]
        )

    def test_height_ratios_are_kept(self):
        layout = slp.figure(2, 1, height_ratios=[1, 2], backend="plotly").native.layout
        top = layout.yaxis.domain[1] - layout.yaxis.domain[0]
        bottom = layout.yaxis2.domain[1] - layout.yaxis2.domain[0]
        assert bottom == pytest.approx(2 * top)

    def test_colorbar_sits_beside_its_own_subplot(self):
        fig = slp.figure(2, 2, backend="plotly")
        before = fig.native.layout.xaxis4.domain
        fig.axes(1, 1).heatmap(np.eye(2), colorbar_label="corr")
        layout = fig.native.layout
        x0, x1 = layout.xaxis4.domain
        y0, y1 = layout.yaxis4.domain
        assert x0 == before[0] and x1 < before[1]  # room taken from the subplot
        colorbar = fig.native.data[0].colorbar
        assert x1 < colorbar.x < before[1]
        assert colorbar.y == pytest.approx((y0 + y1) / 2)
        assert colorbar.len == pytest.approx(y1 - y0)
        assert colorbar.title.side == "right"


# ---------------------------------------------------------------------------
# Every mark works on every installed backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
class TestBackendSmoke:
    def test_every_portable_mark_renders(self, backend):
        x = np.linspace(0, 1, 10)
        y = np.sin(6 * x)
        fig = slp.figure(backend=backend)
        axes = fig.axes()
        axes.set(ylim=(-2, 2), xlim=(0, 1))
        axes.line(x, y, color="#d60000", linewidth=2, label="line")
        axes.points(x, y, color="0.35", markersize=5)
        axes.errorbar(x[::3], y[::3], yerr=0.1, color="black", capsize=4)
        axes.band(x, y - 0.1, y + 0.1, color="#d60000", alpha=0.3)
        axes.bar([0.2, 0.5], [1.0, 0.5], width=0.1, color="#018700")
        axes.polygon([0, 0.1, 0.1], [0, 0, 0.2], facecolor="blue", edgecolor="black")
        axes.hlines([0.5, -0.5], 0, 1, color="purple", linestyle="--")
        axes.vlines(0.5, -1, 1, color="orange")
        axes.axhline(0, color="black", linestyle=":")
        axes.axvline(0.8, color="green")
        axes.text(0.02, 0.95, "note", coords="axes", ha="left", va="top")
        axes.text(0.5, 0.0, "below", coords=("data", "axes"), ha="center", va="top")
        axes.set(xlabel="x", ylabel="y", title="t")
        axes.ticks("x", [0, 0.5, 1], ["a", "b", "c"], rotation=45)
        axes.grid(True, alpha=0.3)
        axes.legend(loc="upper right")
        fig.close()

    def test_heatmap_and_contour(self, backend):
        grid = np.random.default_rng(0).random((5, 5))
        fig = slp.figure(backend=backend)
        axes = fig.axes()
        axes.heatmap(grid, cmap="RdBu", vmin=0, vmax=1, origin="upper", colorbar_label="c")
        axes.contour(np.arange(5), np.arange(5), grid, levels=[0.3, 0.7], labels=True)
        fig.close()

    def test_hollow_markers_render(self, backend):
        fig = slp.figure(backend=backend)
        axes = fig.axes()
        axes.points([0, 1], [1, 2], color="#d60000", fill=False)
        axes.errorbar([0, 1], [1, 2], yerr=0.1, color="black", fill=False)
        axes.line([0, 1], [1, 2], marker="o", fill=False)
        fig.close()

    def test_nan_heatmap_with_hover_renders(self, backend):
        """NaN cells render empty on every backend: a contract frontends rely on."""
        values = np.array([[1.0, np.nan], [np.nan, 0.2]])
        hover = [["a", "b"], ["c", "d"]]
        fig = slp.figure(backend=backend)
        fig.axes().heatmap(values, origin="upper", hover=hover)
        fig.embed_html()
        fig.close()

    def test_grid_layout_with_height_ratios(self, backend):
        fig = slp.figure(2, 2, height_ratios=[3, 1], sharex=True, backend=backend)
        assert fig.shape == (2, 2)
        assert len(fig.flat) == 4
        fig.axes(0, 0).line([0, 1], [0, 1])
        fig.hide(1, 1)
        fig.suptitle("grid")
        fig.close()

    def test_embed_html_is_self_describing(self, backend):
        fig = slp.figure(backend=backend)
        fig.axes().line([0, 1], [0, 1])
        html = fig.embed_html()
        assert "<img" in html or "<div" in html
        fig.close()

    def test_save_roundtrips(self, backend, tmp_path):
        fig = slp.figure(backend=backend)
        fig.axes().line([0, 1], [0, 1])
        out = tmp_path / ("plot.html" if backend == "plotly" else "plot.png")
        fig.save(out)
        assert out.exists() and out.stat().st_size > 0
        fig.close()

    def test_plotly_html_loads_mathjax(self, backend, tmp_path):
        if backend != "plotly":
            pytest.skip("MathJax is used by the Plotly HTML backend")
        fig = slp.figure(backend="plotly")
        fig.axes().set(ylabel=r"$E_{\mathrm{cm}}/M_N$")
        out = tmp_path / "math.html"
        fig.save(out)
        for html in (out.read_text(), fig.embed_html()):
            assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/' in html
            assert "graph.on('plotly_relayout'" in html
            assert "Plotly.restyle(graph, {'marker.size': sizes}" in html
            assert "{plot_id}" not in html
        fig.close()

    def test_wrap_is_idempotent_for_wrapped_objects(self, backend):
        axes = slp.figure(backend=backend).axes()
        assert slp.wrap(axes) is axes


class TestPointSize:
    @pytest.mark.skipif("matplotlib" not in slp.available(), reason="matplotlib not installed")
    def test_marker_extent_is_positive_once_limits_are_set(self):
        axes = slp.figure(backend="matplotlib").axes()
        axes.set(ylim=(0.0, 1.0))
        assert axes.marker_extent(6.0) > 0.0

    @pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
    def test_plotly_marker_extent_needs_limits(self):
        axes = slp.figure(backend="plotly").axes()
        assert axes.marker_extent(6.0) == 0.0
        axes.set(ylim=(0.0, 1.0))
        assert axes.marker_extent(6.0) > 0.0

    @pytest.mark.skipif("matplotlib" not in slp.available(), reason="matplotlib not installed")
    def test_point_size_is_signed(self):
        axes = slp.figure(backend="matplotlib").axes()
        axes.set(ylim=(0.0, 1.0))
        assert axes.point_size(-50, "y", units="axes") < 0
        assert axes.point_size(50, "y", units="axes") > 0


# ---------------------------------------------------------------------------
# stacked_positions no longer needs a matplotlib Axes
# ---------------------------------------------------------------------------


class TestStackedPositions:
    def test_marker_extent_separates_points_whose_errors_do_not_overlap(self):
        y = np.array([1.0, 1.02])
        yerr = np.full(2, 0.005)
        np.testing.assert_allclose(stacked_positions(y, yerr, x=0.0, width=0.3), [0.0, 0.0])
        np.testing.assert_allclose(
            stacked_positions(y, yerr, x=0.0, width=0.3, marker_extent=0.05), [-0.15, 0.15]
        )

    def test_marker_extent_zero_is_the_no_marker_case(self):
        y = np.array([1.0, 1.5, 2.0])
        yerr = np.full(3, 0.01)
        np.testing.assert_allclose(
            stacked_positions(y, yerr, x=0.0, marker_extent=0.0),
            stacked_positions(y, yerr, x=0.0),
        )


# ---------------------------------------------------------------------------
# Frontends: structure, asserted through the recording backend
# ---------------------------------------------------------------------------


class TestSamplingPlotterStructure:
    def test_histogram_emits_bars_and_reference_lines(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_sampling_histogram()

        assert rec.count("bar") == 1
        # lower CI, upper CI, mean, full sample
        assert rec.count("axvline") == 4
        assert any("Full Sample" in label for label in rec.labels())

    @pytest.mark.parametrize(
        "method",
        [
            "plot_sampling_histogram",
            "plot_sampling_errorbar",
            "plot_effective_sample_size",
            "plot_bootstrap_intervals",
        ],
    )
    def test_every_mark_has_an_explicit_color(self, stats, method):
        """Neither backend's automatic color cycle is used, so they agree."""
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            getattr(SamplingPlotter(stats), method)()

        marks = [m for m in rec.marks if m.kind in ("bar", "errorbar", "axvline", "band")]
        assert marks
        assert all(m.kwargs.get("color") is not None for m in marks)

    def test_histogram_bins_come_from_numpy(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_sampling_histogram(bins=12)

        centers, heights = rec.first("bar").args
        assert len(centers) == 12 and len(heights) == 12

    def test_errorbar_carries_hover_text(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_sampling_errorbar()

        mark = rec.first("errorbar")
        assert len(mark.kwargs["hover"]) == 3
        assert rec.count("ticks") == 1

    @pytest.mark.parametrize(
        "method, kind, expected",
        [
            ("plot_sampling_histogram", "bar", " of 200 ("),
            ("plot_effective_sample_size", "bar", "ESS = "),
            ("plot_bootstrap_intervals", "errorbar", "95% CI: ["),
        ],
    )
    def test_marks_carry_hover_text(self, stats, method, kind, expected):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            getattr(SamplingPlotter(stats), method)()

        hover = rec.first(kind).kwargs["hover"]
        assert len(hover) == len(rec.first(kind).args[0])
        assert all(expected in text for text in hover)

    def test_estimates_are_pdg_formatted(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_sampling_errorbar()
        hover = rec.first("errorbar").kwargs["hover"]
        assert all("±" not in text for text in hover)
        assert stats[0].pdg_format(central="mean") in hover[0]

        with slp.record() as rec:
            SamplingPlotter(stats).plot_sampling_histogram()
        assert f"Full Sample: {stats[0].pdg_format()}" in rec.labels()

    def test_correlation_matrix_is_upper_origin(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_correlation_matrix()

        mark = rec.first("heatmap")
        assert mark.kwargs["origin"] == "upper"
        assert mark.kwargs["vmin"] == -1 and mark.kwargs["vmax"] == 1

    def test_effective_sample_size_is_a_bar_chart(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_effective_sample_size()

        assert rec.count("bar") == 1
        assert len(rec.first("bar").args[0]) == 3

    def test_bootstrap_intervals_label_each_level_once(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_bootstrap_intervals(confidence_levels=[0.68, 0.95])

        # one box per (observable, level), but only one legend entry per level
        assert rec.count("band") == 6
        assert rec.labels().count("68% CI") == 1
        assert rec.labels().count("95% CI") == 1

    def test_summary_grid_hides_unused_cells(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            fig = SamplingPlotter(stats).plot_stats_summary(title="summary")

        # 3 histograms + 1 correlation = 4 slots in a 2x2 grid
        assert fig.shape == (2, 2)
        assert fig.hidden == []
        assert fig.title == "summary"
        assert rec.count("heatmap") == 1
        assert rec.count("bar") == 3

    def test_summary_with_one_panel_hides_the_rest(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record():
            fig = SamplingPlotter(stats).plot_stats_summary(
                panels=["histogram", "correlation"], obs_index=0
            )
        assert fig.shape == (1, 2)
        assert fig.hidden == []

    def test_corner_is_matplotlib_only(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record():
            with pytest.raises(slp.UnsupportedFeature, match="matplotlib-only"):
                SamplingPlotter(stats).plot_corner()


class TestSpectrumPlotterStructure:
    def test_one_errorbar_per_level_with_hover(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter

        with slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot()

        assert rec.count("errorbar") == len(energy_collection)
        assert all(len(m.kwargs["hover"]) == 1 for m in rec.of_kind("errorbar"))

    def test_excluded_levels_use_the_excluded_color(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter, SpectrumStyle

        style = SpectrumStyle(excluded_color="lightgrey")
        with slp.record() as rec:
            SectorSpectrumPlotter(
                energy_collection, style=style, excluded_levels=lambda s: True
            ).plot()

        colors = {m.kwargs["color"] for m in rec.of_kind("errorbar")}
        assert colors == {"lightgrey"}

    def test_markers_draw_lines_and_labels(self, energy_collection):
        from sigmondsamplings.plotting import HMarker, SectorSpectrumPlotter

        with slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot(
                markers=[HMarker(y=0.5, label="threshold")]
            )

        assert rec.count("hlines") == 1
        assert any(m.args[2] == "threshold" for m in rec.of_kind("text"))

    def test_outer_labels_mix_data_and_axes_coordinates(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter

        with slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot()

        outer = [m for m in rec.of_kind("text") if m.kwargs.get("coords") == ("data", "axes")]
        assert outer, "expected PSQ labels below the columns"

    def test_limits_are_seeded_before_marker_extent_is_read(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter

        with slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot()

        kinds = rec.kinds()
        assert kinds.index("set") < kinds.index("point_size")

    def test_rejects_both_group_by_and_groups(self, energy_collection):
        from sigmondsamplings.plotting import SpectrumPlotter

        with slp.record():
            with pytest.raises(ValueError, match="exactly one"):
                SpectrumPlotter(energy_collection).plot(group_by="psq", groups={})


# ---------------------------------------------------------------------------
# Frontends render on the real backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
class TestFrontendRendering:
    def test_sampling_plots(self, backend, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        plotter = SamplingPlotter(stats)
        with slp.backend(backend):
            for axes in (
                plotter.plot_sampling_histogram(),
                plotter.plot_sampling_errorbar(),
                plotter.plot_correlation_matrix(),
                plotter.plot_effective_sample_size(),
                plotter.plot_bootstrap_intervals(),
            ):
                assert axes.backend == backend
                axes.figure.close()

    def test_spectrum_plot(self, backend, energy_collection):
        from sigmondsamplings.plotting import HMarker, SectorSpectrumPlotter

        with slp.backend(backend):
            axes = SectorSpectrumPlotter(energy_collection).plot(
                markers=[HMarker(y=0.5, label="thr")]
            )
            assert axes.backend == backend
            axes.figure.close()

    def test_summary_grid(self, backend, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.backend(backend):
            fig = SamplingPlotter(stats).plot_stats_summary(panels=["errorbar", "correlation"])
            assert fig.backend == backend
            fig.close()


# ---------------------------------------------------------------------------
# Compatibility with the pre-backend API
# ---------------------------------------------------------------------------


@pytest.mark.skipif("matplotlib" not in slp.available(), reason="matplotlib not installed")
class TestCompatibility:
    def test_raw_matplotlib_axes_is_wrapped(self, stats):
        import matplotlib.pyplot as plt

        from sigmondsamplings.plotting import SamplingPlotter

        fig, ax = plt.subplots()
        axes = SamplingPlotter(stats).plot_sampling_errorbar(axes=ax)
        assert axes.native is ax
        plt.close(fig)

    def test_package_reexports_the_plotters(self):
        import sigmondsamplings as ss
        from sigmondsamplings import plotting

        assert ss.SamplingPlotter is plotting.SamplingPlotter
        assert all(
            getattr(ss, name) is getattr(plotting, name)
            for name in (
                "HMarker",
                "SectorSpectrumPlotter",
                "SpectrumPlotter",
                "SpectrumStyle",
            )
        )

    def test_wrap_adopts_a_raw_matplotlib_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        axes = slp.wrap(ax)
        assert isinstance(axes, slp.Axes)
        assert axes.backend == "matplotlib"
        assert axes.native is ax
        plt.close(fig)

    def test_as_figure_accepts_axes_figures_and_native_objects(self):
        import matplotlib.pyplot as plt

        with slp.record():
            fig = slp.figure()
            assert slp.as_figure(fig) is fig
            assert slp.as_figure(fig.axes()) is fig
        native, ax = plt.subplots()
        assert slp.as_figure(native).native is native
        assert slp.as_figure(ax).native is native
        plt.close(native)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


@pytest.mark.skipif("matplotlib" not in slp.available(), reason="matplotlib not installed")
class TestCLIPlotting:
    """End-to-end invocation, so a missed keyword in the call chain fails here.

    ``ss query`` reaches the plotters through ``_query_view`` -> ``run_query_view``
    -> ``render_*_plot``; checking only that the typer option exists misses a
    parameter that was never threaded through the middle of that chain.
    """

    @pytest.fixture
    def cli(self, monkeypatch, energy_collection):
        import matplotlib

        matplotlib.use("Agg")
        import sigmondsamplings.cli.main as main

        monkeypatch.setattr(main, "load_collection", lambda *a, **k: energy_collection)
        return main

    @staticmethod
    def _run(cli, args):
        from typer.testing import CliRunner

        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        return result

    @pytest.mark.parametrize("backend", REAL_BACKENDS)
    def test_plot_spectrum(self, cli, tmp_path, backend):
        out = tmp_path / ("s.html" if backend == "plotly" else "s.png")
        self._run(
            cli,
            [
                "query",
                "energy",
                str(tmp_path / "data.h5"),
                "--plot-spectrum",
                "--no-gui",
                "--plot-output",
                str(out),
                "--plot-backend",
                backend,
            ],
        )
        assert out.stat().st_size > 0

    @pytest.mark.parametrize("backend", REAL_BACKENDS)
    def test_generic_plot(self, cli, tmp_path, backend):
        out = tmp_path / ("e.html" if backend == "plotly" else "e.png")
        self._run(
            cli,
            [
                "query",
                "energy",
                str(tmp_path / "data.h5"),
                "--plot",
                "errorbar",
                "--no-gui",
                "--plot-output",
                str(out),
                "--plot-backend",
                backend,
            ],
        )
        assert out.stat().st_size > 0

    def test_default_backend_needs_no_flag(self, cli, tmp_path):
        out = tmp_path / "s.png"
        self._run(
            cli,
            [
                "query",
                "energy",
                str(tmp_path / "data.h5"),
                "--plot-spectrum",
                "--no-gui",
                "--plot-output",
                str(out),
            ],
        )
        assert out.stat().st_size > 0

    def test_obs_subcommand_also_plots(self, cli, tmp_path):
        out = tmp_path / "h.png"
        self._run(
            cli,
            [
                "query",
                "obs",
                str(tmp_path / "data.h5"),
                "--plot",
                "histogram",
                "--no-gui",
                "--plot-output",
                str(out),
            ],
        )
        assert out.stat().st_size > 0

    def test_unknown_backend_is_reported(self, cli, tmp_path):
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            cli.app,
            [
                "query",
                "energy",
                str(tmp_path / "data.h5"),
                "--plot-spectrum",
                "--no-gui",
                "--plot-backend",
                "ggplot",
            ],
        )
        assert result.exit_code != 0
