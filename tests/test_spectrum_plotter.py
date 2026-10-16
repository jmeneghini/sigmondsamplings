"""Scientific semantics of spectrum points, bounds, groups, colors and limits."""

import numpy as np
import pytest

import slat.plotting as slp
from sigmondsamplings.energy_level_collection import SingleEnsembleEnergyCollection
from sigmondsamplings.energy_levels import EnergyObsInfo
from sigmondsamplings.info import SamplingInfo
from sigmondsamplings.plotting import (
    HMarker,
    SectorSpectrumPlotter,
    SpectrumPlotter,
    energy_level_hover_text,
)
from sigmondsamplings.rcparams import rc_context
from sigmondsamplings.sampling import SigmondSampling


def levels(*rows, method="bootstrap"):
    """Rows are (level, full value, resamples, energy type, reference particle)."""
    info = SamplingInfo(method, len(rows[0][2]), **({"seed": 1} if method == "bootstrap" else {}))
    return SingleEnsembleEnergyCollection(
        [
            SigmondSampling(
                np.array([full, *resamples]),
                EnergyObsInfo(
                    index=i,
                    psq=0,
                    irrep="A1g",
                    level_index=level,
                    energy_type=kind,
                    ref_particle=ref,
                ),
                info,
            )
            for i, (level, full, resamples, kind, ref) in enumerate(rows)
        ]
    )


@pytest.fixture
def spectrum():
    return levels(
        (0, 0.5, [0.45, 0.48, 0.51, 0.6], "ecm", None),
        (1, 0.8, [0.7, 0.75, 0.9, 1.0], "ecm", None),
    )


def test_full_sample_is_used_even_with_mean_rc(spectrum):
    with rc_context({"plot.central_metric": "mean"}), slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(markers=[HMarker.from_sampling(spectrum[0])])
    assert rec.first("errorbar").args[1] == spectrum[0].full_sample_value
    assert rec.first("hlines").args[0] == spectrum[0].full_sample_value
    assert f"Ecm = {spectrum[0].pdg_format()}" in rec.first("errorbar").kwargs["hover"][0]


def test_unset_label_fontsizes_defer_to_backend_default(spectrum):
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(
            markers=[HMarker.from_sampling(spectrum[0], label="ref")]
        )
    assert rec.first("ticks").kwargs["fontsize"] is None
    assert [m.kwargs["fontsize"] for m in rec.of_kind("text")] == [None, None]


def test_grid_follows_the_style_without_vertical_lines(spectrum):
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot()
    assert rec.count("grid") == 0
    with slp.style_context(grid=True):
        layout = SectorSpectrumPlotter(spectrum).plot(backend="plotly").figure.native.layout
    assert layout.yaxis.showgrid is True
    assert layout.xaxis.showgrid is False
    with slp.style_context(grid=True):
        ax = SectorSpectrumPlotter(spectrum).plot(backend="matplotlib").native
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert all(line.get_visible() for line in ax.get_ygridlines())
    ax.figure.clf()



def test_drawing_follows_the_figure_style_not_the_global_one(spectrum):
    figure_style = slp.PlotStyle(grid=True, colors=("#111111", "#222222"))
    with slp.record():
        axes = slp.figure(style=figure_style).axes()
        with slp.style_context(grid=False, colors=("#999999",)):
            SectorSpectrumPlotter(spectrum).plot(axes=axes)
        marks = axes.native
    assert [m.args[0] for m in marks if m.kind == "grid"] == [False]
    assert {m.kwargs["color"] for m in marks if m.kind == "errorbar"} <= {"#111111", "#222222"}

def test_level_hover_keeps_gaussian_error_separate_from_percentiles(spectrum):
    sample = spectrum[0]
    with rc_context({"plot.central_metric": "mean"}), slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(percentiles=(10, 90))
    text = rec.first("errorbar").kwargs["hover"][0]
    low, high = np.percentile(sample.data[1:], [10, 90])
    assert f"{sample.pdg_format()}" in text
    assert f"10–90%: [{low:.6g}, {high:.6g}]" in text
    assert "NI:" not in text
    ni = energy_level_hover_text(sample, ni_pair="N(1)N(1)", percentiles=(10, 90))
    assert ni == "NI: N(1)N(1)<br>" + text


def test_ni_level_without_sampling_shares_the_hover_layout(spectrum):
    from sigmondsamplings.plotting import ni_level_hover_text

    text = ni_level_hover_text("N(1)N(1)", "T1g", 0, 2.1234567, ni_rank=1)
    assert text == (
        "NI: N(1)N(1)<br>T1g, d²=0, rank 1<br>Ecm = 2.12346"
        "<br>Gaussian error and percentiles unavailable"
    )
    unclassified = ni_level_hover_text("N(1)N(1)", "T1g", 0, 2.1, ni_rank="unclassified")
    assert "<br>T1g, d²=0, coupling unclassified<br>" in unclassified
    sampled = energy_level_hover_text(spectrum[0], ni_pair="N(1)N(1)", irrep="T1g", psq=0,
                                      include_level=False, ni_rank=1)
    assert sampled.split("<br>")[:2] == text.split("<br>")[:2]


def test_sampling_marker_automatically_formats_hover(spectrum):
    sample = spectrum[0]
    marker = HMarker.from_sampling(sample, percentiles=(10, 90), ni_pair="N(1)N(1)")
    assert marker.sampling is sample
    assert marker.hover is None
    assert f"{sample.pdg_format()}" in marker.hover_text
    assert "10–90%:" in marker.hover_text
    assert "NI: N(1)N(1)" in marker.hover_text
    assert "level 0" not in marker.hover_text
    assert "rank" not in marker.hover_text
    ranked = HMarker.from_sampling(sample, ni_pair="N(1)N(1)", ni_rank=2, group=0, column="T1g")
    assert ranked.hover is None
    assert "T1g, d²=0, rank 2<br>" in ranked.hover_text
    assert "rank" not in HMarker.from_sampling(sample, ni_rank=2).hover_text
    custom = HMarker.from_sampling(sample, hover="custom")
    assert custom.hover_text == "custom"


def test_jackknife_hover_does_not_invent_percentiles():
    collection = levels((0, 0.5, [0.45, 0.48, 0.51, 0.6], "ecm", None), method="jackknife")
    text = energy_level_hover_text(collection[0])
    assert "(Gaussian)" not in text
    assert "Percentiles unavailable for jackknife" in text


def test_marker_hover_is_independent_of_annotation(spectrum):
    marker = HMarker(y=0.6, hover="N(0) + N(0)")
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(markers=[marker])
    assert rec.first("hlines").kwargs["hover"] == "N(0) + N(0)"
    assert marker.label is None
    fig = slp.figure(backend="plotly")
    SectorSpectrumPlotter(spectrum).plot(axes=fig.axes(), markers=[marker])
    assert any(t.hoverinfo == "text" and t.hovertext[0] == marker.hover for t in fig.native.data)


@pytest.mark.parametrize(
    "options",
    [
        {"percentiles": (10, 90)},
        {"confidence_level": 0.8},
        {"error_metric": "confidence_interval", "confidence_level": 0.8},
    ],
)
def test_percentile_levels_and_marker_share_exact_bounds(spectrum, options):
    expected = np.percentile(spectrum[0].data[1:], [10, 90])
    marker = HMarker.from_sampling(spectrum[0], **options)
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(markers=[marker], **options)
    point = rec.first("errorbar")
    errors = point.kwargs["yerr"][:, 0]
    np.testing.assert_allclose([point.args[1] - errors[0], point.args[1] + errors[1]], expected)
    np.testing.assert_allclose(marker.band, expected)
    assert point.args[1] == marker.y == 0.5


@pytest.mark.parametrize("rc_metric", ["error", "confidence_interval"])
def test_marker_error_follows_resolved_metric(spectrum, rc_metric):
    # "error" is the resolved metric only without an interval choice, since the
    # validator rejects error_metric="error" alongside percentiles/confidence_level.
    sample = spectrum[0]
    with rc_context({"plot.error_metric": rc_metric}):
        default = HMarker.from_sampling(sample)
        explicit = HMarker.from_sampling(sample, percentiles=(10, 90))
    if rc_metric == "error":
        assert default.err == float(sample.error)
    else:
        assert isinstance(default.err, tuple)
    np.testing.assert_allclose(explicit.band, np.percentile(sample.data[1:], [10, 90]))


def test_global_percentile_error_setting(spectrum):
    with rc_context({"plot.error_metric": "confidence_interval"}), slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot()
    error = rec.first("errorbar").kwargs["yerr"][:, 0]
    lo, hi = spectrum[0].confidence_interval(0.68)
    np.testing.assert_allclose(error, [0.5 - lo, hi - 0.5])


def test_interval_outside_point_is_not_clipped():
    data = levels((0, 0.5, [1.0, 1.1, 1.2, 1.3], "ecm", None))
    marker = HMarker.from_sampling(data[0], percentiles=(10, 90))
    expected = np.percentile(data[0].data[1:], [10, 90])
    with slp.record() as rec:
        SectorSpectrumPlotter(data).plot(percentiles=(10, 90), markers=[marker])
    assert rec.count("errorbar") == 0
    assert rec.first("points").args[1] == [0.5]
    np.testing.assert_allclose(rec.first("vlines").args[1:], expected)
    np.testing.assert_allclose(marker.band, expected)
    ylim = rec.first("set").kwargs["ylim"]
    assert ylim[0] < 0.5 and ylim[1] > expected[1]


@pytest.mark.parametrize(
    "options",
    [
        {"percentiles": (90, 10)},
        {"percentiles": (-1, 90)},
        {"percentiles": (10, 101)},
        {"confidence_level": 1},
        {"percentiles": (10, 90), "confidence_level": 0.8},
        {"error_metric": "error", "percentiles": (10, 90)},
    ],
)
def test_invalid_interval_options_raise(spectrum, options):
    with pytest.raises(ValueError):
        SectorSpectrumPlotter(spectrum).plot(backend="recording", **options)
    with pytest.raises(ValueError):
        HMarker.from_sampling(spectrum[0], **options)


def test_jackknife_percentiles_are_rejected():
    data = levels((0, 0.5, [0.4, 0.5, 0.6, 0.7], "ecm", None), method="jackknife")
    with pytest.raises(ValueError, match="bootstrap"):
        SectorSpectrumPlotter(data).plot(backend="recording", percentiles=(16, 84))


@pytest.mark.parametrize("kind,ref", [("elab", None), ("ecm", "pi")])
def test_incompatible_quantities_are_rejected(kind, ref):
    data = levels(
        (0, 0.5, [0.4, 0.5, 0.6, 0.7], "ecm", None),
        (1, 0.8, [0.7, 0.8, 0.9, 1.0], kind, ref),
    )
    with slp.record() as rec, pytest.raises(ValueError, match="reference normalization"):
        SectorSpectrumPlotter(data).plot()
    assert not rec.figures  # reject before drawing anything


def test_energy_type_selects_data_and_label():
    data = levels(
        (0, 0.5, [0.4, 0.5, 0.6, 0.7], "ecm", None),
        (1, 0.8, [0.7, 0.8, 0.9, 1.0], "elab", None),
    )
    with slp.record() as rec:
        SectorSpectrumPlotter(data).plot(energy_type="elab")
    assert rec.count("errorbar") == 1
    assert rec.first("errorbar").args[1] == 0.8
    assert any(r"E_{\mathrm{lab}}" in (m.kwargs["ylabel"] or "") for m in rec.of_kind("set"))
    with pytest.raises(ValueError, match="No energy levels"):
        SectorSpectrumPlotter(data).plot(energy_type="decm", backend="recording")


def test_explicit_groups_determine_the_axis_label(spectrum):
    other = levels((0, 0.9, [0.8, 0.9, 1.0, 1.1], "elab", "pi"))
    with slp.record() as rec:
        SpectrumPlotter(spectrum).plot(groups={"other": other})
    label = next(m.kwargs["ylabel"] for m in rec.of_kind("set") if m.kwargs.get("ylabel") is not None)
    assert r"E_{\mathrm{lab}}" in label and r"M_{\pi}" in label
    with pytest.raises(ValueError, match="reference normalization"):
        SpectrumPlotter(spectrum).plot(groups={"a": spectrum, "b": other}, backend="recording")


def test_limits_include_marker_bands_and_ignore_absent_groups(spectrum):
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot(
            markers=[
                HMarker(y=1.5, err=(0.1, 0.3)),
                HMarker(y=-1.0, err=0.2, group=0),
                HMarker(y=100.0, group=999),
            ]
        )
    ylim = rec.first("set").kwargs["ylim"]
    assert ylim[0] < -1.2 and 1.8 < ylim[1] < 3
    assert rec.count("hlines") == 2


def test_column_markers_span_only_their_column():
    def column(psq, irrep, full):
        info = SamplingInfo("bootstrap", 3, seed=1)
        obs = EnergyObsInfo(index=0, psq=psq, irrep=irrep, level_index=0, energy_type="ecm")
        return SigmondSampling(np.array([full, full, full, full]), obs, info)

    data = SingleEnsembleEnergyCollection(
        [column(0, "A1g", 1.0), column(1, "A2", 1.1), column(1, "E", 1.2)]
    )
    with slp.record() as rec:
        SectorSpectrumPlotter(data).plot(
            markers=[
                HMarker(y=1.15, group=1),
                HMarker(y=1.05, group=1, column="E"),
                HMarker(y=1.05, group=0, column="E"),  # no such column: dropped
                HMarker(y=50.0, group=1, column="B1"),  # dropped, and not in ylim
            ]
        )
    lines = {
        call.args[0]: call.args[1:3]
        for call in rec.of_kind("hlines")
        if call.args[0] in (1.05, 1.15)
    }
    assert set(lines) == {1.05, 1.15}
    pad = 0.3
    assert lines[1.15] == pytest.approx((1.0 - pad, 2.0 + pad))
    assert lines[1.05] == pytest.approx((2.0 - pad, 2.0 + pad))
    assert rec.first("set").kwargs["ylim"][1] < 2


def test_existing_limits_and_explicit_overrides(spectrum):
    with slp.record():
        ax = slp.figure().axes()
        ax.set(xlim=(-5, 5), ylim=(0, 2))
        SectorSpectrumPlotter(spectrum).plot(axes=ax, markers=[HMarker(y=10)])
        assert ax.get_limits("x") == (-5, 5)
        assert ax.get_limits("y") == (0, 2)
        SectorSpectrumPlotter(spectrum).plot(axes=ax, xlim=(-2, 2), ylim=(-1, 3))
        assert ax.get_limits("x") == (-2, 2)
        assert ax.get_limits("y") == (-1, 3)


def test_colors_follow_level_index_when_reordered_or_filtered(spectrum):
    with slp.style_context(colors=["red", "blue"]):
        for selected in (spectrum, spectrum[::-1], spectrum[1:]):
            with slp.record() as rec:
                SectorSpectrumPlotter(selected).plot()
            colors = {m.args[1]: m.kwargs["color"] for m in rec.of_kind("errorbar")}
            assert all(
                color == ("red" if full == 0.5 else "blue") for full, color in colors.items()
            )


def test_outer_groups_have_uniform_spacing_and_midpoint_separators(spectrum):
    groups = {"first": {"a": spectrum, "b": spectrum}, "second": {"c": spectrum}}
    with slp.record() as rec:
        SpectrumPlotter(spectrum).plot(groups=groups)
    assert rec.first("ticks").args[1] == [0.0, 1.0, 2.0]
    assert rec.first("axvline").args == (1.5,)
    outer = [m for m in rec.of_kind("text") if m.kwargs.get("coords") == ("data", "axes")]
    assert [(m.args[0], m.args[2]) for m in outer] == [(0.5, "first"), (2.0, "second")]
    with slp.record() as rec:
        SectorSpectrumPlotter(spectrum).plot()
    assert rec.count("axvline") == 0


@pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
def test_group_separator_spans_the_real_axes(backend, spectrum):
    axes = SpectrumPlotter(spectrum).plot(
        groups={"first": {"a": spectrum}, "second": {"b": spectrum}}, backend=backend
    )
    if backend == "plotly":
        (divider,) = [s for s in axes.figure.native.layout.shapes if s.type == "line"]
        assert divider.x0 == divider.x1 == 0.5
        assert divider.y0 < 0 and divider.y1 == 1
    else:
        divider = axes.native.lines[-1]
        np.testing.assert_allclose(divider.get_xdata(), [0.5, 0.5])
        assert divider.get_ydata()[0] < 0 and divider.get_ydata()[1] == 1
        assert not divider.get_clip_on()
    axes.figure.close()


@pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
def test_real_backends_preserve_ranges_and_exact_percentiles(backend, spectrum):
    ax = slp.figure(backend=backend).axes()
    assert ax.get_limits("y", explicit_only=True) is None
    ax.set(xlim=(-2, 3), ylim=(0, 3))
    SectorSpectrumPlotter(spectrum).plot(axes=ax, percentiles=(10, 90), markers=[HMarker(y=2)])
    assert ax.get_limits("x") == (-2, 3)
    assert ax.get_limits("y") == (0, 3)
    expected = np.percentile(spectrum[0].data[1:], [10, 90])
    if backend == "plotly":
        point = ax.figure.native.data[0]
        assert point.y[0] == 0.5
        assert not point.error_y.symmetric
        np.testing.assert_allclose(
            [point.y[0] - point.error_y.arrayminus[0], point.y[0] + point.error_y.array[0]],
            expected,
        )
    else:
        segment = ax.native.containers[0].lines[2][0].get_segments()[0]
        np.testing.assert_allclose(segment[:, 1], expected)
    ax.figure.close()
    outside = levels((0, 0.5, [1.0, 1.1, 1.2, 1.3], "ecm", None))
    ax = SectorSpectrumPlotter(outside).plot(backend=backend, percentiles=(10, 90))
    assert ax.get_limits("y")[0] < 0.5 and ax.get_limits("y")[1] > 1.27
    ax.figure.close()


@pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
def test_real_backend_automatic_limits_include_marker_bands(backend, spectrum):
    ax = SectorSpectrumPlotter(spectrum).plot(
        backend=backend, markers=[HMarker(y=1.5, err=(0.1, 0.3))]
    )
    assert ax.get_limits("y")[1] > 1.8
    ax.figure.close()
