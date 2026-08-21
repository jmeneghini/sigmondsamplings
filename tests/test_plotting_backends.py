"""Tests for the backend-neutral plotting API (``slat.plotting``).

Three layers:

* the core -- registry, style, text, mark translation
* structural tests of the sigmondsamplings frontends via the recording backend,
  which assert *what* gets drawn without rendering anything
* smoke tests on whichever real backends are installed

Run with:  python -m pytest tests/test_plotting_backends.py -v
"""

from __future__ import annotations

import warnings

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


# ---------------------------------------------------------------------------
# Style and text
# ---------------------------------------------------------------------------


class TestStyle:
    def test_color_cycle_is_restartable(self):
        style = slp.PlotStyle(colors=["a", "b"])
        first = style.color_cycle()
        assert [next(first) for _ in range(3)] == ["a", "b", "a"]
        assert next(style.color_cycle()) == "a"

    def test_pixel_size_uses_dpi(self):
        assert slp.PlotStyle(figsize=(6.0, 4.0), dpi=100).pixel_size == (600, 400)

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

    def test_strip_math(self):
        assert slp.strip_math("$E_0$") == "E_0"

    def test_to_display_passes_through_when_math_is_supported(self):
        assert slp.to_display("$E_0$", mathtext=True) == "$E_0$"

    def test_to_display_falls_back_to_unicode(self):
        assert slp.to_display("$E_0$", mathtext=False) == "E₀"


# ---------------------------------------------------------------------------
# plotly translation helpers
# ---------------------------------------------------------------------------


@pytest.mark.skipif("plotly" not in slp.available(), reason="plotly not installed")
class TestPlotlyTranslation:
    def test_hex_and_named_colors(self):
        from slat.plotting._plotly import _color

        assert _color("#d60000") == "rgb(214,0,0)"
        assert _color("black", 0.5) == "rgba(0,0,0,0.5)"

    def test_matplotlib_greyscale_strings(self):
        from slat.plotting._plotly import _to_rgb

        assert _to_rgb("0.0") == (0, 0, 0)
        assert _to_rgb("1.0") == (255, 255, 255)

    def test_unknown_color_survives_without_alpha(self):
        from slat.plotting._plotly import _color

        assert _color("rebeccapurple", 0.5) == "rebeccapurple"

    def test_marker_and_dash_maps(self):
        from slat.plotting._plotly import _dash, _marker

        assert _marker("o") == "circle"
        assert _dash("--") == "dash"

    def test_asymmetric_errors(self):
        from slat.plotting._plotly import _err

        spec = _err(np.array([[0.1, 0.2], [0.3, 0.4]]), 2)
        assert spec["symmetric"] is False
        np.testing.assert_allclose(spec["arrayminus"], [0.1, 0.2])
        np.testing.assert_allclose(spec["array"], [0.3, 0.4])

    def test_scalar_error_is_broadcast(self):
        from slat.plotting._plotly import _err

        np.testing.assert_allclose(_err(0.5, 3)["array"], [0.5, 0.5, 0.5])

    def test_tick_labels_fall_back_to_unicode(self):
        fig = slp.figure(backend="plotly")
        fig.axes().ticks("x", [0, 1], ["$A_1$", "$T_{1g}$"])
        assert fig.native.layout.xaxis.ticktext == ("A₁", "T₁g")

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
        axes.line(x, y, color="#d60000", width=2, label="line")
        axes.points(x, y, color="0.35", size=5)
        axes.errorbar(x[::3], y[::3], yerr=0.1, color="black", capsize=4)
        axes.band(x, y - 0.1, y + 0.1, color="#d60000", alpha=0.3)
        axes.bar([0.2, 0.5], [1.0, 0.5], width=0.1, color="#018700")
        axes.polygon([0, 0.1, 0.1], [0, 0, 0.2], facecolor="blue", edgecolor="black")
        axes.hlines([0.5, -0.5], 0, 1, color="purple", style="--")
        axes.vlines(0.5, -1, 1, color="orange")
        axes.axhline(0, color="black", style=":")
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
        assert axes.supports("contour")
        axes.contour(np.arange(5), np.arange(5), grid, levels=[0.3, 0.7], labels=True)
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

    def test_wrap_is_idempotent_for_wrapped_objects(self, backend):
        axes = slp.figure(backend=backend).axes()
        assert slp.wrap(axes) is axes


class TestCapabilities:
    def test_pixel_metrics_only_exact_on_matplotlib(self):
        with slp.record():
            assert slp.figure().axes().supports("pixel_metrics")
        if "plotly" in slp.available():
            assert not slp.figure(backend="plotly").axes().supports("pixel_metrics")

    def test_unknown_feature_rejected(self):
        with slp.record():
            with pytest.raises(ValueError, match="Unknown feature"):
                slp.figure().axes().supports("teleportation")

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
    def test_marker_extent_matches_the_legacy_markersize_path(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        y = np.array([1.0, 1.01, 1.02, 2.0])
        yerr = np.full(4, 0.005)

        fig, ax = plt.subplots()
        ax.set_ylim(0.9, 2.1)
        legacy = stacked_positions(y, yerr, x=0.0, width=0.3, markersize=6, ax=ax)

        axes = slp.wrap(ax)
        ported = stacked_positions(y, yerr, x=0.0, width=0.3, marker_extent=axes.marker_extent(6))
        plt.close(fig)

        np.testing.assert_allclose(ported, legacy)

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
    def test_legacy_ax_keyword_still_works_and_warns(self, stats):
        import matplotlib.pyplot as plt

        from sigmondsamplings.plotting import SamplingPlotter

        fig, ax = plt.subplots()
        with pytest.warns(DeprecationWarning, match="`ax` argument is deprecated"):
            axes = SamplingPlotter(stats).plot_sampling_errorbar(ax=ax)
        assert axes.native is ax
        plt.close(fig)

    def test_axes_and_ax_together_is_an_error(self, stats):
        import matplotlib.pyplot as plt

        from sigmondsamplings.plotting import SamplingPlotter

        fig, ax = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TypeError, match="not both"):
                SamplingPlotter(stats).plot_sampling_errorbar(axes=slp.wrap(ax), ax=ax)
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
