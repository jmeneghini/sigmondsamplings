"""Matrix coverage and available sub-bases (``CorrelatorIndex.coverage`` / ``available_bases``),
and the coverage plot. None of these read samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from correlator_data import BL, OPS, ckey, observable, series, vev

import slat.plotting as slp
from sigmondsamplings.correlators import CorrelatorIndex, CorrelatorMatrixInfo
from sigmondsamplings.correlators import matrix as matrix_module
from sigmondsamplings.plotting import plot_coverage

RNG = np.random.default_rng(9)


@pytest.fixture(autouse=True)
def _reset_rng():
    global RNG
    RNG = np.random.default_rng(9)
TIMES = (3, 4, 5)

#: Eleven operators of one channel, standing in for the rho E250 pSq3 Ep basis.
E250 = tuple(BL(f"pion P=(0,0,0) A1um_1 SD_{k}") for k in range(11))
#: Its unmeasured upper-triangle cells: VI_0 and VI_1 (here 9 and 10) with each
#: other, and each with operator 8. Three of 66.
E250_HOLES = {(8, 9), (8, 10), (9, 10)}


def _index(cells, ops=OPS, times=TIMES, herm=True, vevs=()):
    items = [
        observable(ckey(i, j, t, herm, ops=ops), series(RNG, loc=1.0))
        for i, j in cells
        for t in times
    ]
    items += [vev(ops[i], series(RNG)) for i in vevs]
    return CorrelatorIndex.from_collection(items)


def _upper(n, holes=()):
    return [(i, j) for i in range(n) for j in range(i, n) if (i, j) not in holes]


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_hermitian_completion_and_holes(self):
        # upper triangle only, with (2, 2) unmeasured: the herm_file pattern of test_cli_corr
        index = _index([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)])
        cov = index.coverage(OPS)
        assert cov.dims == ("time", "sink", "source")
        assert list(cov.coords["time"]) == list(TIMES)
        expected = np.ones((3, 3), dtype=bool)
        expected[2, 2] = False
        assert (cov.values == expected).all()
        assert cov.attrs["operators"] == OPS and cov.attrs["key_form"] == "HermMat"

    def test_non_hermitian_needs_each_direction(self):
        index = _index([(0, 0), (0, 1), (1, 1)], herm=False)
        cov = index.coverage(OPS[:2], hermitian=False)
        assert cov.values[:, 0, 1].all() and not cov.values[:, 1, 0].any()
        herm = index.coverage(OPS[:2])  # a Hermitian request may use the flipped correlator
        assert herm.values.all()

    def test_times_are_every_time_any_cell_has(self):
        items = [observable(ckey(0, 0, t, True), series(RNG)) for t in (1, 2, 3)]
        items += [observable(ckey(i, j, 2, True), series(RNG)) for i, j in [(0, 1), (1, 1)]]
        cov = CorrelatorIndex.from_collection(items).coverage(OPS[:2])
        assert list(cov.coords["time"]) == [1, 2, 3]
        assert cov.sel(time=2).values.all()
        assert cov.sel(time=1).values.tolist() == [[True, False], [False, False]]

    def test_derived_connected_cells_need_both_vevs(self):
        index = _index(_upper(2), vevs=[0])
        cov = index.coverage(OPS[:2], subtract_vev=True)
        assert cov.values[:, 0, 0].all()
        assert not cov.values[:, 0, 1].any() and not cov.values[:, 1, 1].any()
        assert cov.attrs["subtract_vev"] is True

    def test_key_form_ambiguity_raises_like_matrix(self):
        items = [
            observable(ckey(i, j, 3, herm), series(RNG))
            for herm in (True, False)
            for i, j in _upper(2)
        ]
        index = CorrelatorIndex.from_collection(items)
        with pytest.raises(ValueError, match="several key forms"):
            index.coverage(OPS[:2])
        assert index.coverage(OPS[:2], key_form="plain").attrs["key_form"] == "plain"

    def test_basis_as_channel_spec_or_strings(self):
        index = _index(_upper(3))
        (channel,) = index.channels
        by_channel = index.coverage(channel)
        by_info = index.coverage(CorrelatorMatrixInfo(OPS, hermitian=True))
        by_strings = index.coverage([str(op) for op in OPS])
        assert by_channel.equals(by_info) and by_info.equals(by_strings)

    def test_agrees_with_matrix(self):
        index = _index(_upper(3, holes={(1, 2)}))
        cov = index.coverage(OPS)
        assert not cov.values[:, 1, 2].any()
        with pytest.raises(ValueError, match="not every requested cell"):
            index.matrix(OPS, TIMES)
        for basis in index.available_bases(OPS):
            assert index.matrix(basis, TIMES).times == TIMES


# ---------------------------------------------------------------------------
# available_bases
# ---------------------------------------------------------------------------


class TestAvailableBases:
    def test_drops_the_operator_without_a_diagonal(self):
        index = _index([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)])
        assert index.available_bases(OPS) == [OPS[:2]]

    def test_e250_hole_pattern(self):
        index = _index(_upper(11, E250_HOLES), ops=E250, times=(4,))
        bases = index.available_bases(E250)
        # 9..10 and 8 are pairwise incompatible: three maximal bases of nine
        assert [len(b) for b in bases] == [9, 9, 9]
        assert bases[0] == E250[:9]
        assert bases[1] == E250[:8] + (E250[9],)
        assert bases[2] == E250[:8] + (E250[10],)
        assert len(index.coverage(E250).values[0].nonzero()[0]) == 121 - 2 * len(E250_HOLES)

    def test_times_restrict_the_requirement(self):
        items = [observable(ckey(i, j, t, True), series(RNG)) for i, j in _upper(2) for t in (3, 4)]
        items += [observable(ckey(0, 0, 5, True), series(RNG))]
        index = CorrelatorIndex.from_collection(items)
        assert index.available_bases(OPS[:2]) == [(OPS[0],)]  # 5 has only (0, 0)
        assert index.available_bases(OPS[:2], [3, 4]) == [OPS[:2]]
        assert index.available_bases(OPS[:2], [7]) == []  # nothing stored at 7

    def test_greedy_above_the_exact_limit(self, monkeypatch):
        monkeypatch.setattr(matrix_module, "EXACT_BASIS_LIMIT", 4)
        index = _index(_upper(11, E250_HOLES), ops=E250, times=(4,))
        (basis,) = index.available_bases(E250)
        assert len(basis) == 9 and set(E250[:8]) <= set(basis)

    def test_bases_keep_basis_order(self):
        index = _index(_upper(3))
        reordered = (OPS[2], OPS[0], OPS[1])
        assert index.available_bases(reordered) == [reordered]


# ---------------------------------------------------------------------------
# plot_coverage
# ---------------------------------------------------------------------------


class TestPlotCoverage:
    def test_missing_cells_are_empty_and_crossed(self):
        cov = _index([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]).coverage(OPS)
        with slp.record() as rec:
            plot_coverage(cov)
        values = rec.first("heatmap").args[0]
        assert np.isnan(values[2, 2]) and np.nansum(values) == 8
        (cross,) = [m for m in rec.of_kind("text") if m.args[2] == "×"]
        assert cross.args[:2] == (2, 2)
        xticks = rec.of_kind("ticks")[0]
        assert xticks.args[2] == ["0", "1", "2"]

    def test_fraction_and_hover(self):
        items = [observable(ckey(0, 0, t, True), series(RNG)) for t in (1, 2, 3, 5)]
        items += [observable(ckey(i, j, 2, True), series(RNG)) for i, j in [(0, 1), (1, 1)]]
        cov = CorrelatorIndex.from_collection(items).coverage(OPS[:2])
        with slp.record() as rec:
            plot_coverage(cov, labels="short")
        heatmap = rec.first("heatmap")
        np.testing.assert_allclose(heatmap.args[0], [[1.0, 0.25], [0.25, 0.25]])
        assert heatmap.kwargs["hover"][0][0] == f"{OPS[0]}<br>{OPS[0]}<br>4/4 times: 1..3,5"
        assert rec.of_kind("ticks")[0].args[2] == [OPS[0].compact_str, OPS[1].compact_str]

    def test_label_options(self):
        from sigmondsamplings.plotting import operator_labels

        assert operator_labels(OPS, "latex")[0].startswith("$")
        assert operator_labels(OPS, lambda op: "x") == ["x"] * 3
        assert operator_labels(OPS, ["a", "b", "c"]) == ["a", "b", "c"]
        with pytest.raises(ValueError, match="2 labels for 3"):
            operator_labels(OPS, ["a", "b"])
        with pytest.raises(ValueError, match="labels must be"):
            operator_labels(OPS, "long")

    @pytest.mark.parametrize("backend", [b for b in slp.available() if b != "recording"])
    def test_renders(self, backend):
        cov = _index(_upper(11, E250_HOLES), ops=E250, times=(4,)).coverage(E250)
        axes = plot_coverage(cov, backend=backend)
        axes.figure.embed_html()
        axes.figure.close()
