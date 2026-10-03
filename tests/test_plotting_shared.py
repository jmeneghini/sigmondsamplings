"""Shared plotting pieces: ``_util`` helpers, ``HMarker`` and the frontends moved onto them.

Structural tests through ``slp.record()``, in the style of
``test_plotting_backends.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import slat.plotting as slp
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.plotting._util import (
    central_and_error,
    labeled_heatmap,
    summary_grid,
)
from sigmondsamplings.plotting.markers import HMarker
from sigmondsamplings.rcparams import rc_context
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats

RNG = np.random.default_rng(11)
ENS = EnsembleInfo("ens", 30)
BOOT = SamplingInfo("bootstrap", 200, seed=1)
JACK = SamplingInfo("jackknife", 30)


def _series(info=BOOT, *, complex_=False) -> SigmondData:
    n = 30 if info is None else info.num_resamplings + 1
    buf = RNG.normal(loc=[[1.0], [2.0], [3.0]], scale=0.1, size=(3, n))
    if complex_:
        buf = buf + 1j * RNG.normal(size=buf.shape)
    return SigmondData.from_data(
        buf,
        "C",
        info,
        dim_names=("time",),
        coords={"time": [2, 3, 4]},
        ensemble_info=ENS,
        is_complex=complex_,
    )


@pytest.fixture
def stats() -> SamplingStats:
    samplings = []
    for i in range(3):
        data = np.empty(201)
        data[0] = 1.0 + i
        data[1:] = RNG.normal(1.0 + i, 0.05, 200)
        samplings.append(SigmondSampling(data, ObservableInfo(name=f"obs{i}", index=i), BOOT))
    return SamplingStats(samplings)


# ---------------------------------------------------------------------------
# central_and_error
# ---------------------------------------------------------------------------


class TestCentralAndError:
    def test_defaults_are_full_sample_and_error(self):
        data = _series()
        y, yerr = central_and_error(data)
        np.testing.assert_allclose(y, data.data[:, 0])
        np.testing.assert_allclose(yerr, np.asarray(data.error))
        assert y.shape == yerr.shape == (3,)

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("mean", lambda raw: raw[:, 1:].mean(axis=-1)),
            ("median", lambda raw: np.median(raw[:, 1:], axis=-1)),
            ("bias_corrected_mean", lambda raw: 2 * raw[:, 0] - raw[:, 1:].mean(axis=-1)),
        ],
    )
    def test_central_metric_is_honoured(self, metric, expected):
        data = _series()
        with rc_context({"plot.central_metric": metric}):
            y, _ = central_and_error(data)
        np.testing.assert_allclose(y, expected(data.data))

    def test_confidence_interval_is_asymmetric_offsets(self):
        data = _series()
        with rc_context({"plot.error_metric": "confidence_interval"}):
            y, yerr = central_and_error(data, confidence_level=0.9)
        lower, upper = np.percentile(data.data[:, 1:], [5, 95], axis=-1)
        assert yerr.shape == (2, 3)
        np.testing.assert_allclose(yerr[0], np.maximum(y - lower, 0))
        np.testing.assert_allclose(yerr[1], np.maximum(upper - y, 0))

    def test_interval_offsets_are_never_negative(self):
        buf = np.concatenate([[10.0], RNG.normal(0.0, 0.1, 200)])  # full far above
        data = SigmondSampling(buf, ObservableInfo(name="x", index=0), BOOT)
        with rc_context({"plot.error_metric": "confidence_interval"}):
            _, yerr = central_and_error(data)
        assert yerr[0] > 0.0 and yerr[1] == 0.0

    def test_bootstrap_only_metrics_name_the_rc_key_on_jackknife(self):
        data = _series(JACK)
        with rc_context({"plot.error_metric": "confidence_interval"}):
            with pytest.raises(ValueError, match="plot.error_metric"):
                central_and_error(data)
        with rc_context({"plot.central_metric": "bias_corrected_mean"}):
            with pytest.raises(ValueError, match="plot.central_metric"):
                central_and_error(data)

    def test_bins_use_the_bin_mean_and_standard_error(self):
        data = _series(None)
        y, yerr = central_and_error(data)
        np.testing.assert_allclose(y, data.data.mean(axis=-1))
        np.testing.assert_allclose(yerr, data.data.std(axis=-1, ddof=1) / np.sqrt(30))

    def test_complex_data_asks_for_a_component(self):
        with pytest.raises(ValueError, match="complex"):
            central_and_error(_series(complex_=True))

    def test_scalar_sampling(self):
        data = SigmondSampling(np.linspace(0, 1, 201), ObservableInfo(name="x", index=0), BOOT)
        y, yerr = central_and_error(data)
        assert y.shape == () and yerr.shape == ()


# ---------------------------------------------------------------------------
# labeled_heatmap / summary_grid
# ---------------------------------------------------------------------------


class TestLabeledHeatmap:
    def test_marks_and_conventions(self):
        with slp.record() as rec:
            labeled_heatmap(
                slp.figure().axes(),
                np.eye(2),
                labels=["a", "b"],
                cmap="RdBu",
                vmin=-1,
                vmax=1,
                hover=[["1", "2"], ["3", "4"]],
            )
        assert rec.kinds() == ["heatmap", "ticks", "ticks"]
        heatmap = rec.first("heatmap")
        assert heatmap.kwargs["origin"] == "upper"
        assert heatmap.kwargs["hover"] == [["1", "2"], ["3", "4"]]
        xticks, yticks = rec.of_kind("ticks")
        assert xticks.args == ("x", range(2), ["a", "b"]) and xticks.kwargs["rotation"] == 45
        assert yticks.args == ("y", range(2), ["a", "b"])

    def test_annotations_skip_empty_cells(self):
        with slp.record() as rec:
            labeled_heatmap(
                slp.figure().axes(),
                np.array([[1.0, np.nan], [0.0, 1.0]]),
                labels=["a", "b"],
                annotate=[[None, "×"], ["", None]],
            )
        (text,) = rec.of_kind("text")
        # column 1, row 0: x is the column, y the row
        assert text.args == (1, 0, "×")
        assert text.kwargs["ha"] == "center" and text.kwargs["va"] == "center"


@pytest.mark.parametrize(
    ("n", "shape"), [(1, (1, 1)), (3, (1, 3)), (4, (2, 2)), (5, (2, 3)), (10, (3, 4))]
)
def test_summary_grid(n, shape):
    assert summary_grid(n) == shape


# ---------------------------------------------------------------------------
# HMarker
# ---------------------------------------------------------------------------


class TestHMarker:
    def test_band_is_symmetric_or_asymmetric(self):
        assert HMarker(y=1.0).band is None
        assert HMarker(y=1.0, err=0.1).band == pytest.approx((0.9, 1.1))
        assert HMarker(y=1.0, err=(0.1, 0.3)).band == pytest.approx((0.9, 1.3))

    def test_from_sampling_uses_full_sample_and_rc_errors(self):
        data = _series()[0]
        marker = HMarker.from_sampling(data, label="thr", color="red")
        assert marker.y == pytest.approx(data.full_sample_value)
        assert marker.err == pytest.approx(data.error)
        assert (marker.label, marker.color) == ("thr", "red")
        with rc_context({"plot.error_metric": "confidence_interval"}):
            asym = HMarker.from_sampling(data)
        assert isinstance(asym.err, tuple) and len(asym.err) == 2

    def test_from_sampling_rejects_arrays(self):
        with pytest.raises(ValueError, match="single observable"):
            HMarker.from_sampling(_series())

    def test_is_the_same_class_everywhere(self):
        import sigmondsamplings as ss
        from sigmondsamplings import plotting
        from sigmondsamplings.plotting import spectrum

        assert spectrum.HMarker is HMarker is plotting.HMarker is ss.HMarker


# ---------------------------------------------------------------------------
# Frontends on the shared pieces
# ---------------------------------------------------------------------------


@pytest.fixture
def energy_collection():
    from sigmondsamplings.energy_level_collection import SingleEnsembleEnergyCollection
    from sigmondsamplings.energy_levels import EnergyObsInfo

    info = SamplingInfo("bootstrap", 50, seed=1)
    samplings = []
    for index, (irrep, level) in enumerate([("A1g", 0), ("A1g", 1), ("T1u", 0)]):
        data = np.empty(51)
        data[0] = 0.5 + 0.1 * level
        data[1:] = RNG.normal(data[0], 0.01, 50)
        samplings.append(
            SigmondSampling(
                data,
                EnergyObsInfo(
                    index=index, irrep=irrep, psq=0, energy_type="ecm", level_index=level
                ),
                info,
            )
        )
    return SingleEnsembleEnergyCollection(samplings)


class TestSpectrumOnSharedPieces:
    def test_marker_band_is_drawn_behind_the_line(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter

        with slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot(
                markers=[HMarker(y=0.55, err=0.01, color="red"), HMarker(y=0.7)]
            )
        kinds = [k for k in rec.kinds() if k in ("band", "hlines")]
        assert kinds == ["band", "hlines", "hlines"]
        band = rec.first("band")
        np.testing.assert_allclose(band.args[1], [0.54, 0.54])
        np.testing.assert_allclose(band.args[2], [0.56, 0.56])
        assert band.kwargs["color"] == "red"
        assert band.args[0] == [rec.first("hlines").args[1], rec.first("hlines").args[2]]

    def test_default_colors_come_from_the_active_style(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter

        with slp.style_context(colors=["#111111", "#222222"]), slp.record() as rec:
            SectorSpectrumPlotter(energy_collection).plot()
        colors = [m.kwargs["color"] for m in rec.of_kind("errorbar")]
        # Colors follow level indices: A1g has levels 0/1, T1u level 0.
        assert colors == ["#111111", "#222222", "#111111"]

    def test_style_color_cycle_still_wins(self, energy_collection):
        from sigmondsamplings.plotting import SectorSpectrumPlotter, SpectrumStyle

        style = SpectrumStyle(color_cycle=["#abcdef"])
        with slp.style_context(colors=["#111111"]), slp.record() as rec:
            SectorSpectrumPlotter(energy_collection, style=style).plot()
        assert {m.kwargs["color"] for m in rec.of_kind("errorbar")} == {"#abcdef"}


class TestCorrelationMatrixOnLabeledHeatmap:
    def test_records_the_same_marks_as_before_the_move(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_correlation_matrix()
        assert rec.kinds() == ["heatmap", "ticks", "ticks", "set"]
        heatmap = rec.first("heatmap")
        assert heatmap.kwargs["cmap"] == "RdBu"
        assert (heatmap.kwargs["vmin"], heatmap.kwargs["vmax"]) == (-1, 1)
        assert heatmap.kwargs["origin"] == "upper"
        assert heatmap.kwargs["colorbar_label"] == "Correlation"

    def test_hover_names_the_observable_pair(self, stats):
        from sigmondsamplings.plotting import SamplingPlotter

        with slp.record() as rec:
            SamplingPlotter(stats).plot_correlation_matrix()
        hover = rec.first("heatmap").kwargs["hover"]
        assert len(hover) == 3 and all(len(row) == 3 for row in hover)
        assert hover[0][2].startswith("obs0 0<br>obs2 2<br>")
