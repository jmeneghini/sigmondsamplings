"""Matrix-level views: ``CorrelatorMatrix.normalized`` / ``significance``,
``compact_str`` labels, and ``CorrelatorMatrixPlotter``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from correlator_data import BL, BOOT, OPS, ckey, observable

import slat.plotting as slp
from sigmondsamplings.correlators import BasicLapHOperator, CorrelatorIndex, parse_operator
from sigmondsamplings.info import EnsembleInfo
from sigmondsamplings.plotting import CorrelatorMatrixPlotter
from sigmondsamplings.plotting.correlators import MAX_GRID_OPERATORS

ENS = EnsembleInfo("ens_corr", 40, temporal_extent=32)
OPS4 = tuple(BL(f"pion P=(0,0,0) A1um_1 SD_{k}") for k in range(4))
TIMES = tuple(range(1, 9))


def _matrix(n=3, *, upper_only=True, hermitian=True, complex_=False, seed=2, ops=OPS4):
    """A physical-looking Hermitian matrix: (Z e^{-E t}) Z^T plus relative noise."""
    rng = np.random.default_rng(seed)
    energies = np.linspace(0.3, 1.0, n)
    z = rng.normal(size=(n, n)) + 2 * np.eye(n)
    samples = BOOT.num_resamplings + 1
    items = []
    for t in TIMES:
        c = (z * np.exp(-energies * t)) @ z.T
        for i in range(n):
            for j in range(i if upper_only else 0, n):
                data = c[i, j] * (1 + 0.02 * rng.normal(size=samples))
                if complex_:
                    data = data + 0.01j * c[i, j] * rng.normal(size=samples)
                data[0] = c[i, j]
                items.append(observable(ckey(i, j, t, hermitian, ops=ops), data, BOOT, ENS))
    return CorrelatorIndex.from_collection(items).matrix(ops[:n], hermitian=hermitian)


def _errorbars(rec, cell=None):
    return [m for m in rec.of_kind("errorbar") if cell is None or m.cell == cell]


# ---------------------------------------------------------------------------
# normalized / significance
# ---------------------------------------------------------------------------


class TestNormalized:
    def test_divides_every_sample_by_the_full_sample_factors(self):
        m = _matrix()
        out = m.normalized(3)
        data = np.asarray(m.values.data)
        f = data[TIMES.index(3), np.arange(3), np.arange(3), 0].real
        np.testing.assert_allclose(out.data, data / np.sqrt(np.outer(f, f))[None, :, :, None])
        assert out.dims == ("time", "sink", "source")
        np.testing.assert_allclose(np.diagonal(np.asarray(out.sel(time=3).full_sample_value)), 1.0)
        assert type(out).__name__ != "CorrelatorMatrix"

    def test_non_hermitian_complex_uses_the_modulus(self):
        m = _matrix(2, upper_only=False, hermitian=False, complex_=True)
        data = np.asarray(m.values.data)
        f = np.abs(data[0, np.arange(2), np.arange(2), 0])
        np.testing.assert_allclose(
            m.normalized(1).data, data / np.sqrt(np.outer(f, f))[None, :, :, None]
        )
        assert m.normalized(1).is_complex

    def test_missing_time_and_negative_factor_raise(self):
        m = _matrix()
        with pytest.raises(ValueError, match="not a time"):
            m.normalized(99)
        flipped = m._replace(values=m.values * np.array([-1.0]))
        with pytest.raises(ValueError, match="not positive"):
            flipped.normalized(3)


def test_significance_is_modulus_over_error():
    m = _matrix()
    sig = m.significance(4)
    at = m.values.sel(time=4)
    expected = np.abs(np.asarray(at.full_sample_value)) / np.asarray(at.error)
    np.testing.assert_allclose(sig.values, expected)
    assert sig.dims == ("sink", "source")
    with pytest.raises(ValueError, match="not a time"):
        m.significance(0)


# ---------------------------------------------------------------------------
# compact labels
# ---------------------------------------------------------------------------


def test_compact_str():
    assert OPS[0].compact_str == "π(0,0,0) SD5"
    assert OPS[2].compact_str == "ππ[(0,0,1),(0,0,-1)]"
    two = BL("isotriplet_pion_pion A1um_1 CG_1 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_0]")
    assert two.compact_str == "ππ[(0,0,1),(0,0,-1)] SS1,SS0 CG1"
    assert parse_operator("GI{isotriplet S=0 PSQ=1 A2m ROT 3}").compact_str == "ROT 3"
    assert BasicLapHOperator.vacuum().compact_str == "vacuum"


# ---------------------------------------------------------------------------
# CorrelatorMatrixPlotter
# ---------------------------------------------------------------------------


class TestGrid:
    def test_n_squared_cells_with_hollow_exactly_where_not_stored(self):
        m = _matrix()
        with slp.record() as rec:
            fig = CorrelatorMatrixPlotter(m).plot_grid()
        assert fig.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                marks = _errorbars(rec, (i, j))
                hollow = [mk for mk in marks if mk.kwargs["fill"] is False]
                if i > j:  # read from the flipped direction
                    assert len(marks) == 1 and hollow == marks
                else:
                    assert hollow == []

    def test_diagonal_and_off_diagonal_ranges_differ(self):
        with slp.record() as rec:
            CorrelatorMatrixPlotter(_matrix()).plot_grid(what="normalized")
        ylims = {m.cell: m.kwargs["ylim"] for m in rec.of_kind("set") if m.kwargs.get("ylim")}
        diag = {ylims[(i, i)] for i in range(3)}
        off = {ylims[(i, j)] for i in range(3) for j in range(3) if i != j}
        assert len(diag) == 1 and len(off) == 1 and diag != off

    def test_fixed_ranges_follow_sigmond(self):
        with slp.record() as rec:
            CorrelatorMatrixPlotter(_matrix()).plot_grid(
                what="normalized", diag_range=1.2, offdiag_range=0.5
            )
        ylims = {m.cell: m.kwargs["ylim"] for m in rec.of_kind("set") if m.kwargs.get("ylim")}
        assert ylims[(1, 1)] == (-0.1, 1.2) and ylims[(0, 2)] == (-0.5, 0.5)

    def test_labels_on_the_edges_only(self):
        with slp.record() as rec:
            CorrelatorMatrixPlotter(_matrix(), labels=["a", "b", "c"]).plot_grid()
        text = {}
        for m in rec.of_kind("set"):
            for key in ("title", "ylabel"):
                if m.kwargs.get(key):
                    text.setdefault(key, {})[m.cell] = m.kwargs[key]
        assert text["title"] == {(0, 0): "a", (0, 1): "b", (0, 2): "c"}
        assert text["ylabel"] == {(0, 0): "a", (1, 0): "b", (2, 0): "c"}

    def test_effective_energy_grid_hides_off_diagonals(self):
        with slp.record() as rec:
            fig = CorrelatorMatrixPlotter(_matrix()).plot_grid(what="effective_energy")
        assert len(fig.hidden) == 6
        assert {m.cell for m in _errorbars(rec)} == {(0, 0), (1, 1), (2, 2)}

    def test_size_guard(self):
        ops = tuple(BL(f"pion P=(0,0,0) A1um_1 SD_{k}") for k in range(MAX_GRID_OPERATORS + 1))
        m = _matrix(MAX_GRID_OPERATORS + 1, ops=ops)
        with pytest.raises(ValueError, match="plot_heatmap"):
            CorrelatorMatrixPlotter(m).plot_grid()

    def test_invalid_what(self):
        with pytest.raises(ValueError, match="what must be"):
            CorrelatorMatrixPlotter(_matrix()).plot_grid(what="eigenvalues")


class TestConsistency:
    def test_an_operator_has_one_color_everywhere(self):
        plotter = CorrelatorMatrixPlotter(_matrix())
        with slp.record() as rec:
            plotter.plot_diagonal()
        diagonal = [m.kwargs["color"] for m in _errorbars(rec)]
        with slp.record() as rec:
            plotter.plot_grid()
        grid = [_errorbars(rec, (i, i))[0].kwargs["color"] for i in range(3)]
        with slp.record() as rec:
            plotter.plot_diagonal(effective=True)
        effective = [m.kwargs["color"] for m in _errorbars(rec)]
        assert diagonal == grid == effective == list(plotter.operator_colors)

    def test_colors_are_fixed_at_construction(self):
        with slp.style_context(colors=["#111111", "#222222", "#333333"]):
            plotter = CorrelatorMatrixPlotter(_matrix())
        with slp.record() as rec:
            plotter.plot_diagonal()
        assert [m.kwargs["color"] for m in _errorbars(rec)] == ["#111111", "#222222", "#333333"]

    def test_one_tick_label_per_operator_everywhere(self):
        plotter = CorrelatorMatrixPlotter(_matrix(), labels="short")
        assert plotter.operator_labels == tuple(op.compact_str for op in OPS4[:3])
        with slp.record() as rec:
            plotter.plot_heatmap(3)
            plotter.plot_diagonal()
        assert rec.of_kind("ticks")[0].args[2] == list(plotter.operator_labels)
        assert [m.kwargs["label"] for m in _errorbars(rec)] == list(plotter.operator_labels)

    def test_label_and_color_counts(self):
        with pytest.raises(ValueError, match="2 colors for 3"):
            CorrelatorMatrixPlotter(_matrix(), colors=["r", "g"])
        with pytest.raises(TypeError, match="CorrelatorMatrix"):
            CorrelatorMatrixPlotter(_matrix().values)


class TestHeatmap:
    def test_normalized_defaults_to_normalizing_at_the_shown_time(self):
        with slp.record() as rec:
            CorrelatorMatrixPlotter(_matrix()).plot_heatmap(4)
        heatmap = rec.first("heatmap")
        np.testing.assert_allclose(np.diagonal(heatmap.args[0]), 1.0)
        assert (heatmap.kwargs["cmap"], heatmap.kwargs["vmin"], heatmap.kwargs["vmax"]) == (
            "RdBu",
            -1.0,
            1.0,
        )
        assert heatmap.kwargs["hover"][0][1].startswith(f"{OPS4[0]}<br>{OPS4[1]}<br>")

    def test_significance_and_hermiticity(self):
        m = _matrix()
        plotter = CorrelatorMatrixPlotter(m)
        with slp.record() as rec:
            plotter.plot_heatmap(4, value="significance")
            plotter.plot_heatmap(4, value="hermiticity", tol=2.0)
        sig, herm = rec.of_kind("heatmap")
        np.testing.assert_allclose(sig.args[0], m.significance(4).values)
        assert herm.kwargs["cmap"] == "Reds" and herm.kwargs["vmax"] == 4.0
        check = m.check_hermitian(2.0)
        np.testing.assert_allclose(
            np.diagonal(herm.args[0]), check.diagonal_imag.sel(time=4).values
        )

    def test_invalid_value_and_time(self):
        plotter = CorrelatorMatrixPlotter(_matrix())
        with pytest.raises(ValueError, match="value must be"):
            plotter.plot_heatmap(4, value="phase")
        with pytest.raises(ValueError, match="not a time"):
            plotter.plot_heatmap(0)


class TestSummary:
    def test_default_panels(self):
        with slp.record() as rec:
            fig = CorrelatorMatrixPlotter(_matrix()).plot_summary(title="m")
        assert fig.shape == (1, 3) and fig.title == "m"
        assert rec.count("heatmap") == 1
        assert {m.cell for m in _errorbars(rec)} == {(0, 0), (0, 1)}

    def test_spare_cells_are_hidden(self):
        with slp.record():
            fig = CorrelatorMatrixPlotter(_matrix()).plot_summary(
                panels=["diagonal", "normalized", "significance", "hermiticity", "effective_energy"]
            )
        assert fig.shape == (2, 3) and fig.hidden == [(1, 2)]

    def test_invalid_panels(self):
        with pytest.raises(ValueError, match="panels"):
            CorrelatorMatrixPlotter(_matrix()).plot_summary(panels=["grid"])


@pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
def test_every_view_renders(backend):
    plotter = CorrelatorMatrixPlotter(_matrix(complex_=True), labels="latex")
    with slp.backend(backend), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for fig in (
            plotter.plot_grid(),
            plotter.plot_grid(what="normalized", norm_time=2),
            plotter.plot_grid(what="effective_energy"),
            plotter.plot_summary(
                panels=["diagonal", "effective_energy", "normalized", "significance", "hermiticity"]
            ),
        ):
            fig.embed_html()
            fig.close()
        axes = plotter.plot_heatmap(3)
        axes.figure.close()
