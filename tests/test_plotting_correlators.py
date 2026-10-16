"""SeriesPlotter, CorrelatorPlotter and FitOverlay.

Structural tests through ``slp.record()``, in the style of
``test_plotting_backends.py``, plus one smoke test per real backend.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import slat.plotting as slp
from sigmondsamplings.correlators.effective import effective_energy
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.plotting import (
    CorrelatorPlotter,
    FitOverlay,
    HMarker,
    SeriesPlotter,
    SeriesStyle,
)

REAL_BACKENDS = [b for b in slp.available() if b != "recording"]
T = 32
BOOT = SamplingInfo("bootstrap", 40, seed=1)
ENS = EnsembleInfo("ens", 100, temporal_extent=T)
TIMES = np.arange(16)
MASSES = (0.3, 0.5, 0.7)


def _buffer(masses=MASSES, times=TIMES, noise=0.01, seed=0):
    """``(time, operator, N)`` exponentials with relative noise."""
    rng = np.random.default_rng(seed)
    mean = np.exp(-np.outer(times, masses))[..., None]
    samples = mean * (1 + noise * rng.normal(size=mean.shape[:2] + (BOOT.num_resamplings + 1,)))
    samples[..., 0] = mean[..., 0]
    return samples


def _stack(buf=None, **kw) -> SigmondData:
    buf = _buffer(**kw) if buf is None else buf
    return SigmondData.from_data(
        buf,
        "C",
        BOOT,
        dim_names=("time", "operator"),
        coords={
            "time": list(TIMES[: buf.shape[0]]),
            "operator": [f"op{i}" for i in range(buf.shape[1])],
        },
        ensemble_info=ENS,
    )


def _errorbars(rec, cell=None):
    return [m for m in rec.of_kind("errorbar") if cell is None or m.cell == cell]


# ---------------------------------------------------------------------------
# SeriesPlotter
# ---------------------------------------------------------------------------


class TestSeriesPlotter:
    def test_one_errorbar_per_series_with_hue_labels(self):
        with slp.record() as rec:
            SeriesPlotter(_stack()).plot()
        marks = _errorbars(rec)
        assert len(marks) == 3
        assert [m.kwargs["label"] for m in marks] == ["op0", "op1", "op2"]
        assert rec.count("legend") == 1

    def test_series_are_dodged_symmetrically(self):
        with slp.record() as rec:
            SeriesPlotter(_stack(), style=SeriesStyle(dodge=0.2)).plot()
        first = [m.args[0][0] for m in _errorbars(rec)]
        np.testing.assert_allclose(first, [-0.1, 0.0, 0.1])

    def test_hover_shows_the_true_x(self):
        with slp.record() as rec:
            SeriesPlotter(_stack()).plot()
        assert _errorbars(rec)[0].kwargs["hover"][2].startswith("op0<br>time=2: ")

    def test_hover_is_the_pdg_estimate(self):
        corr = _stack()
        with slp.record() as rec:
            SeriesPlotter(corr).plot()
        hover = _errorbars(rec)[1].kwargs["hover"][3]
        assert hover == f"op1<br>time=3: {corr.isel(operator=1, time=3).pdg_format()}"

    def test_colors_follow_the_active_style_cycle(self):
        with slp.style_context(colors=["#111111", "#222222", "#333333"]), slp.record() as rec:
            SeriesPlotter(_stack()).plot()
        assert [m.kwargs["color"] for m in _errorbars(rec)] == ["#111111", "#222222", "#333333"]

    def test_pinned_colors_win(self):
        with slp.record() as rec:
            SeriesPlotter(_stack(), colors=["r", "g", "b"]).plot()
        assert [m.kwargs["color"] for m in _errorbars(rec)] == ["r", "g", "b"]

    def test_negative_points_are_hollow_exactly_on_a_log_axis(self):
        buf = _buffer()
        buf[5, 1] *= -1  # the whole of C(5) for op1 is negative
        corr = _stack(buf)
        with slp.record() as rec:
            SeriesPlotter(corr).plot(yscale="log")
        op1 = [
            m for m in _errorbars(rec) if m.kwargs["color"] == _errorbars(rec)[1].kwargs["color"]
        ]
        hollow = [m for m in _errorbars(rec) if m.kwargs["fill"] is False]
        assert len(hollow) == 1
        np.testing.assert_allclose(hollow[0].args[0], [5.0])
        assert hollow[0].args[1][0] > 0  # |y| is drawn
        assert hollow[0].kwargs["label"] is None and len(op1) == 2

        with slp.record() as rec:
            SeriesPlotter(corr).plot()  # linear: nothing hollow
        assert all(m.kwargs["fill"] for m in _errorbars(rec))

    def test_negative_drop_omits_the_point(self):
        buf = _buffer()
        buf[5, 1] *= -1
        with slp.record() as rec:
            SeriesPlotter(_stack(buf), style=SeriesStyle(negative="drop")).plot(yscale="log")
        assert len(_errorbars(rec)) == 3
        assert all(m.kwargs["fill"] for m in _errorbars(rec))
        assert sum(len(m.args[0]) for m in _errorbars(rec)) == 3 * len(TIMES) - 1

    def test_max_rel_error_hides_points_and_the_range_ignores_them(self):
        buf = _buffer(masses=(0.3,))
        buf[-1, 0, 1:] = buf[-1, 0, 0] + 100.0 * np.linspace(-1, 1, BOOT.num_resamplings)
        corr = _stack(buf)
        with slp.record() as rec:
            SeriesPlotter(corr, style=SeriesStyle(max_rel_error=0.5)).plot()
        (mark,) = _errorbars(rec)
        assert len(mark.args[0]) == len(TIMES) - 1
        ylim = next(m.kwargs["ylim"] for m in rec.of_kind("set") if m.kwargs.get("ylim"))
        assert ylim[1] < 2.0

    def test_limits_are_seeded_before_drawing(self):
        with slp.record() as rec:
            SeriesPlotter(_stack()).plot(yscale="log")
        kinds = rec.kinds()
        seed = next(i for i, m in enumerate(rec.marks) if m.kind == "set" and m.kwargs.get("ylim") is not None)
        assert seed < kinds.index("errorbar")
        assert rec.marks[seed].kwargs["yscale"] == "log"
        assert rec.marks[seed].kwargs["ylim"][0] > 0

    def test_sequence_of_1d_series(self):
        corr = _stack()
        with slp.record() as rec:
            SeriesPlotter([corr.isel(operator=0), corr.isel(operator=2)], labels=["a", "b"]).plot()
        assert [m.kwargs["label"] for m in _errorbars(rec)] == ["a", "b"]

    def test_asymmetric_errors_under_the_interval_metric(self):
        from sigmondsamplings.rcparams import rc_context

        with rc_context({"plot.error_metric": "confidence_interval"}), slp.record() as rec:
            SeriesPlotter(_stack().isel(operator=0)).plot()
        (mark,) = _errorbars(rec)
        assert np.shape(mark.kwargs["yerr"]) == (2, len(TIMES))
        hover = mark.kwargs["hover"][0]
        assert _stack().isel(operator=0, time=0).pdg_format() in hover
        assert "<br>68% CI: [" in hover

    @pytest.mark.parametrize(
        ("make", "match"),
        [
            (lambda c: c.to_complex(), "complex"),
            (lambda c: SigmondData.stack([c, c], dim_name="x"), "at most one more dim"),
            (lambda c: c.rename_dims(time="t"), "no 'time' dim"),
        ],
    )
    def test_rejected_inputs(self, make, match):
        with pytest.raises(ValueError, match=match):
            SeriesPlotter(make(_stack()))

    def test_label_and_color_counts_are_checked(self):
        with pytest.raises(ValueError, match="2 labels for 3 series"):
            SeriesPlotter(_stack(), labels=["a", "b"])
        with pytest.raises(ValueError, match="1 colors for 3 series"):
            SeriesPlotter(_stack(), colors=["r"])

    def test_effective_energy_is_drawn_at_its_offset(self):
        eff = effective_energy(_stack().isel(operator=0), step=2)
        with slp.record() as rec:
            SeriesPlotter(eff).plot()
        (mark,) = _errorbars(rec)
        assert mark.args[0][0] == pytest.approx(1.0)  # t=0 at 0 + step/2
        assert "time=0: " in mark.kwargs["hover"][0]  # hover names the source time
        assert any(m.kwargs.get("ylabel") == r"$a_t E_{\mathrm{eff}}$" for m in rec.of_kind("set"))

    def test_explicit_offset_overrides(self):
        eff = effective_energy(_stack().isel(operator=0), step=2)
        with slp.record() as rec:
            SeriesPlotter(eff, x_offset=0.0).plot()
        assert _errorbars(rec)[0].args[0][0] == pytest.approx(0.0)


class TestMarkersOnSeries:
    def test_marker_band_then_line_then_label(self):
        with slp.record() as rec:
            SeriesPlotter(_stack()).plot(markers=[HMarker(0.5, label="thr", err=0.02, color="red")])
        kinds = [k for k in rec.kinds() if k in ("band", "hlines", "text")]
        assert kinds == ["band", "hlines", "text"]
        band = rec.first("band")
        np.testing.assert_allclose(band.args[1], [0.48, 0.48])
        assert band.kwargs["color"] == "red"

    def test_marker_level_is_inside_the_seeded_range(self):
        with slp.record() as rec:
            SeriesPlotter(_stack()).plot(markers=[HMarker(5.0)])
        ylim = next(m.kwargs["ylim"] for m in rec.of_kind("set") if m.kwargs.get("ylim"))
        assert ylim[1] > 5.0


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------




from sigmondsamplings.fitting import (  # noqa: E402
    CallableXModel,
    MinimizerConfig,
    ParamSetSpec,
    ParamSpec,
    SamplingFit,
)

EXP = CallableXModel(lambda t, th: th[0] * np.exp(-th[1] * t), ("A", "E"))


@pytest.fixture(scope="module")
def fit_result():
    pytest.importorskip("optimagic")
    series = _stack().isel(operator=0).sel(time=list(range(3, 12)))
    params = ParamSetSpec(
        values={"A": ParamSpec(initial=1.0), "E": ParamSpec(initial=0.5)}
    ).resolve()
    fit = SamplingFit.from_data(
        series, EXP, params, minimizer=MinimizerConfig(algorithm="scipy_ls_trf")
    )
    return fit.fit_full_sample()


class TestFitResultX:
    def test_x_is_copied_from_the_fit(self, fit_result):
        assert fit_result.x_name == "time"
        np.testing.assert_array_equal(fit_result.x, np.arange(3, 12))

    def test_theory_series_is_labelled_like_the_data(self, fit_result):
        theory = fit_result.theory_series()
        assert theory.dims == ("time",)
        np.testing.assert_array_equal(theory.coords["time"], np.arange(3, 12))
        np.testing.assert_allclose(
            np.asarray(theory.full_sample_value),
            [s.full_sample_value for s in fit_result.theory],
        )

    def test_x_survives_the_hdf5_round_trip(self, fit_result, tmp_path):
        from sigmondsamplings.fitting.result import FitResult

        path = tmp_path / "fit.hdf5"
        fit_result.write_params(path)
        fit_result.write_theory(path)
        loaded = FitResult.from_hdf5(path, theory_file=path)

        assert loaded.x_name == "time"
        np.testing.assert_array_equal(loaded.x, fit_result.x)
        FitOverlay(loaded, energy="E")  # usable from disk alone

    def test_results_without_x_are_refused(self, fit_result):
        from dataclasses import replace

        with pytest.raises(ValueError, match="from_data"):
            FitOverlay(replace(fit_result, x=None, x_name=None))
        with pytest.raises(ValueError, match="not a fit parameter"):
            FitOverlay(fit_result, energy="M")
        with pytest.raises(ValueError, match="x values"):
            replace(fit_result, x=None).theory_series()


class TestFigureStyle:
    def test_series_colors_come_from_the_figure_style(self):
        with slp.record():
            axes = slp.figure(style=slp.PlotStyle(colors=("#111111",))).axes()
            with slp.style_context(colors=("#999999",)):
                CorrelatorPlotter(_stack().isel(operator=0)).plot(axes=axes)
            colors = {m.kwargs["color"] for m in axes.native if m.kind == "errorbar"}
        assert colors == {"#111111"}


class TestFitOverlay:
    def test_on_a_correlator_plot_draws_the_model_band_and_line(self, fit_result):
        with slp.record() as rec:
            CorrelatorPlotter(_stack().isel(operator=0)).plot(
                fits=[FitOverlay(fit_result, label="fit")]
            )
        assert [k for k in rec.kinds() if k in ("band", "line")] == ["band", "line"]
        band = rec.first("band")
        np.testing.assert_array_equal(band.args[0], np.arange(3, 12))
        assert band.kwargs["label"] == "fit"

    def test_on_an_effective_energy_plot_draws_the_energy_and_the_model(self, fit_result):
        with slp.record() as rec:
            CorrelatorPlotter(_stack().isel(operator=0)).plot_effective_energy(
                fits=[FitOverlay(fit_result, energy="E", label="fit")]
            )
        assert [k for k in rec.kinds() if k in ("band", "line")] == ["band", "line", "line"]
        band, (flat, model) = rec.first("band"), rec.of_kind("line")
        np.testing.assert_array_equal(band.args[0], [3.0, 11.0])
        e = fit_result.param("E").full_sample_value
        np.testing.assert_allclose(flat.args[1], [e, e])
        # the model's own effective energy, at the data's plot offset: flat for one exponential
        np.testing.assert_allclose(model.args[0], np.arange(3, 11) + 0.5)
        np.testing.assert_allclose(model.args[1], e, rtol=1e-10)
        assert model.kwargs["linestyle"] == "--"

    def test_color_follows_the_named_series(self, fit_result):
        with slp.record() as rec:
            CorrelatorPlotter(_stack(), colors=["r", "g", "b"]).plot(
                fits=[FitOverlay(fit_result, series="op1")]
            )
        assert rec.first("band").kwargs["color"] == "g"


# ---------------------------------------------------------------------------
# CorrelatorPlotter
# ---------------------------------------------------------------------------


class TestCorrelatorPlotter:
    def test_correlator_view_is_log_scale(self):
        with slp.record() as rec:
            CorrelatorPlotter(_stack()).plot()
        assert any(m.kwargs.get("yscale") == "log" for m in rec.of_kind("set"))
        assert any(m.kwargs.get("ylabel") == "$C(t)$" for m in rec.of_kind("set"))

    def test_views_share_labels_and_colors(self):
        plotter = CorrelatorPlotter(_stack(), labels=["a", "b", "c"])
        with slp.record() as rec:
            plotter.plot()
            plotter.plot_effective_energy()
        marks = _errorbars(rec)
        corr, eff = marks[:3], marks[3:]
        assert [m.kwargs["color"] for m in corr] == [m.kwargs["color"] for m in eff]
        assert [m.kwargs["label"] for m in eff] == ["a", "b", "c"]

    def test_effective_energy_keeps_failed_samples_as_nan(self):
        buf = _buffer()
        buf[4, 0, 3] *= -1
        eff = CorrelatorPlotter(_stack(buf)).effective_energy()
        assert 3 in eff.coords["time"]  # kept; the plot drops only op0's point there

    def test_summary_is_two_rows_with_markers_only_on_the_energy_panel(self):
        with slp.record() as rec:
            fig = CorrelatorPlotter(_stack()).plot_summary(markers=[HMarker(0.5)], title="summary")
        assert fig.shape == (2, 1) and fig.title == "summary"
        assert [m.cell for m in rec.of_kind("hlines")] == [(1, 0)]
        assert len(_errorbars(rec, (0, 0))) == 3 and len(_errorbars(rec, (1, 0))) == 3

    def test_summary_panels(self):
        with slp.record():
            fig = CorrelatorPlotter(_stack()).plot_summary(panels=["effective_energy"])
        assert fig.shape == (1, 1)
        with pytest.raises(ValueError, match="panels"):
            CorrelatorPlotter(_stack()).plot_summary(panels=["heatmap"])

    def test_complex_input_takes_the_requested_component(self):
        corr = _stack()
        complex_ = corr + 1j * corr
        with slp.record() as rec:
            CorrelatorPlotter(complex_, component="imag").plot()
        np.testing.assert_allclose(
            _errorbars(rec)[0].args[1], np.asarray(corr.isel(operator=0).full_sample_value)
        )
        with pytest.raises(ValueError, match="no imaginary part"):
            CorrelatorPlotter(corr, component="imag")

    def test_rejects_other_shapes(self):
        with pytest.raises(ValueError, match="dims"):
            CorrelatorPlotter(SigmondData.stack([_stack(), _stack()], dim_name="x"))


# ---------------------------------------------------------------------------
# Real backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", REAL_BACKENDS)
class TestRendering:
    def test_every_view(self, backend, fit_result):
        buf = _buffer()
        buf[5, 1] *= -1
        plotter = CorrelatorPlotter(_stack(buf))
        fits = [FitOverlay(fit_result, energy="E", label="fit")]
        markers = [HMarker(0.5, label="thr", err=0.01)]
        with slp.backend(backend), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for axes in (
                plotter.plot(fits=fits),
                plotter.plot_effective_energy(kind="symmetric", fits=fits, markers=markers),
            ):
                assert axes.backend == backend
                axes.figure.close()
            fig = plotter.plot_summary(fits=fits, markers=markers)
            fig.embed_html()
            fig.close()


def test_fit_overlay_uses_selected_series_settings_and_explicit_display_offset(fit_result):
    corr = _stack().isel(operator=0)
    one, two = effective_energy(corr, step=1), effective_energy(corr, step=2)
    with slp.record() as rec:
        SeriesPlotter([one, two], labels=["one", "two"], x_offset=.125).plot(
            fits=[FitOverlay(fit_result, series="two")])
    model = rec.first("line")
    np.testing.assert_array_equal(model.args[0], np.arange(3, 10) + .125)
    np.testing.assert_array_equal(fit_result.x, np.arange(3, 12))


def test_raw_fit_overlay_display_offset_does_not_change_fit_coordinates(fit_result):
    with slp.record() as rec:
        SeriesPlotter(_stack().isel(operator=0), x_offset=2).plot(fits=[FitOverlay(fit_result)])
    np.testing.assert_array_equal(rec.first("band").args[0], np.arange(3, 12) + 2)
    np.testing.assert_array_equal(fit_result.x, np.arange(3, 12))
