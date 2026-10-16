"""Correlator plots, drawn through :class:`~sigmondsamplings.plotting.series.SeriesPlotter`.

A :class:`CorrelatorPlotter` is bound to one correlator or a stack of them
(``(time,)`` or ``(time, operator)``). Its correlator and effective-energy
views share one ``SeriesPlotter`` configuration, so a series has the same label
and color in both. A :class:`CorrelatorMatrixPlotter` is bound to a
``CorrelatorMatrix`` and fixes each operator's label and color once, so the
diagonal, the grid and the heatmaps of one matrix agree. :func:`plot_coverage`
draws which cells of a matrix are stored, from ``CorrelatorIndex.coverage``
alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

import slat.plotting as slp
from slat.plotting.lifecycle import figure_factory

from ..data import SigmondData
from ._util import (
    central_lo_hi,
    cycle_colors,
    kept_points,
    labeled_heatmap,
    pad_linear,
    resolve_axes,
)
from .markers import FitOverlay, HMarker
from .series import SeriesPlotter, SeriesStyle

if TYPE_CHECKING:
    import xarray as xr

    from ..correlators.matrix import CorrelatorMatrix
    from ..correlators.operators import Operator

__all__ = [
    "MAX_GRID_OPERATORS",
    "CorrelatorMatrixPlotter",
    "CorrelatorPlotter",
    "operator_labels",
    "plot_coverage",
]

OperatorLabels = str | Callable[["Operator"], str] | Sequence[str]

TIME = "time"
_COMPONENTS = ("real", "imag")
_PANELS = ("correlator", "effective_energy")
_PANEL_RATIO = {"correlator": 1, "effective_energy": 2}


def _matrix_figsize(n: int) -> tuple[float, float]:
    """Default size for an n-by-n matrix image, with room for its colorbar."""
    side = min(12.0, 3.0 + 0.45 * n)
    return (side + 1.5, side)


def _component(corr: SigmondData, component: str) -> SigmondData:
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of {_COMPONENTS}, got {component!r}")
    return corr.component(component)


class CorrelatorPlotter:
    """Plot one correlator, or a stack of them, and its effective energy.

    Args:
        corr: ``SigmondData`` with dims ``(time,)`` or ``(time, X)``, where each
            entry of ``X`` (e.g. ``operator``) is a separate correlator.
        component: ``"real"`` or ``"imag"``, taken from complex data. Real data
            is used as it is.
        labels: One label per correlator; default the ``X`` coord values.
        colors: One color per correlator; default the style's color cycle.
        style: Visual settings shared by every view.
    """

    def __init__(
        self,
        corr: SigmondData,
        *,
        component: str = "real",
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        style: SeriesStyle | None = None,
    ):
        if not isinstance(corr, SigmondData):
            raise TypeError(f"CorrelatorPlotter expects SigmondData, got {type(corr).__name__}")
        if TIME not in corr.dims or corr.ndim > 2:
            raise ValueError(f"CorrelatorPlotter takes dims (time,) or (time, X), got {corr.dims}")
        self.corr = _component(corr, component)
        self.labels = None if labels is None else list(labels)
        self.colors = None if colors is None else list(colors)
        self.style = style or SeriesStyle()

    def _series(self, data: SigmondData) -> SeriesPlotter:
        return SeriesPlotter(data, labels=self.labels, colors=self.colors, style=self.style)

    def effective_energy(
        self,
        *,
        kind: str = "forward",
        step: int = 1,
        time_extent: int | None = None,
        subtract_const: float = 0.0,
    ) -> SigmondData:
        """The effective energies plotted by :meth:`plot_effective_energy`.

        Failed samples are NaN rather than dropped (``on_fail="nan"``), so a
        time that fails for one correlator stays for the others.
        """
        return self.corr.ts.effective_energy(
            kind=kind,
            step=step,
            time_extent=time_extent,
            subtract_const=subtract_const,
            on_fail="nan",
        )

    @figure_factory
    def plot(
        self,
        *,
        yscale: str = "log",
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Axes:
        """``C(t)``, on a log axis by default (negative values drawn hollow)."""
        return self._series(self.corr).plot(
            yscale=yscale,
            markers=markers,
            fits=fits,
            xlabel=xlabel,
            ylabel=r"$C(t)$" if ylabel is None else ylabel,
            title=title, axes=axes, figsize=figsize, backend=backend,
        )

    @figure_factory
    def plot_effective_energy(
        self,
        *,
        kind: str = "forward",
        step: int = 1,
        time_extent: int | None = None,
        subtract_const: float = 0.0,
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Axes:
        """``E_eff(t)``, drawn at Sigmond's position (``t + step/2`` for the plain kinds).

        See :func:`~sigmondsamplings.correlators.effective.effective_energy`
        for ``kind``, ``step``, ``time_extent`` and ``subtract_const``.
        """
        eff = self.effective_energy(
            kind=kind, step=step, time_extent=time_extent, subtract_const=subtract_const
        )
        return self._series(eff).plot(
            markers=markers,
            fits=fits,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title, axes=axes, figsize=figsize, backend=backend,
        )

    @figure_factory
    def plot_summary(
        self,
        *,
        panels: Sequence[str] = _PANELS,
        kind: str = "forward",
        step: int = 1,
        time_extent: int | None = None,
        subtract_const: float = 0.0,
        yscale: str = "log",
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Figure:
        """``C(t)`` above ``E_eff(t)`` on a shared time axis.

        ``markers`` are reference energies, so they go on the effective-energy
        panel only; ``fits`` go on both. ``yscale`` is the correlator panel's.
        """
        panels = list(panels)
        unknown = [p for p in panels if p not in _PANELS]
        if unknown or not panels or len(set(panels)) != len(panels):
            raise ValueError(f"panels must be distinct names from {_PANELS}, got {panels}")
        ratios = [_PANEL_RATIO[p] for p in panels]
        fig = slp.figure(
            len(panels),
            1,
            height_ratios=ratios,
            sharex=True,
            figsize=figsize or (8.0, 2.5 * sum(ratios)),
            backend=backend,
        )
        for row, panel in enumerate(panels):
            target = fig.axes(row, 0)
            xlabel = None if row == len(panels) - 1 else ""
            if panel == "correlator":
                self.plot(yscale=yscale, fits=fits, xlabel=xlabel, axes=target)
            else:
                self.plot_effective_energy(
                    kind=kind,
                    step=step,
                    time_extent=time_extent,
                    subtract_const=subtract_const,
                    markers=markers,
                    fits=fits,
                    xlabel=xlabel,
                    axes=target,
                )
        if title:
            fig.suptitle(title)
        return fig


# ---------------------------------------------------------------------------
# Operator labels and coverage
# ---------------------------------------------------------------------------


def operator_labels(operators: Sequence[Operator], labels: OperatorLabels = "index") -> list[str]:
    """Tick and legend labels for a basis.

    ``"index"`` numbers the operators by basis position (short, so grids and
    heatmaps stay legible), ``"short"`` uses the operator's ``compact_str``
    (e.g. ``ππ[(0,0,1),(0,0,-1)]``), ``"latex"`` its ``latex_str``; a callable
    maps each operator, and a sequence gives the labels outright.
    """
    ops = list(operators)
    if callable(labels):
        return [labels(op) for op in ops]
    if isinstance(labels, str):
        if labels == "index":
            return [str(i) for i in range(len(ops))]
        if labels == "short":
            return [op.compact_str for op in ops]
        if labels == "latex":
            return [slp.ensure_math(op.latex_str) for op in ops]
        raise ValueError(
            f"labels must be 'index', 'short', 'latex', a callable or a sequence, got {labels!r}"
        )
    out = [str(label) for label in labels]
    if len(out) != len(ops):
        raise ValueError(f"{len(out)} labels for {len(ops)} operators")
    return out


def _span(times: Sequence[int]) -> str:
    times = [int(t) for t in times]
    if not times:
        return "none"
    runs, start = [], times[0]
    for prev, cur in zip(times, times[1:] + [None]):
        if cur is None or cur != prev + 1:  # a run ends at prev
            runs.append(str(start) if start == prev else f"{start}..{prev}")
            start = cur
    return ",".join(runs)


@figure_factory
def plot_coverage(
    coverage: xr.DataArray,
    *,
    labels: OperatorLabels = "index",
    title: str | None = None,
    axes: slp.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
) -> slp.Axes:
    """Which cells of a correlation matrix are stored, as a heatmap.

    ``coverage`` is the output of ``CorrelatorIndex.coverage``; no samples are
    read and no matrix is built, so this works exactly when ``matrix()`` fails.
    Each cell shows the fraction of times it is stored; cells never stored are
    empty and marked ``×``. The hover gives both operators in full and the
    times the cell has.

    Args:
        coverage: Boolean ``(time, sink, source)`` array with
            ``attrs["operators"]``.
        labels: Tick labels; see :func:`operator_labels`.
        title: Default names the key form and the number of times.
        axes, figsize, backend: As for every plot.
    """
    ops = tuple(coverage.attrs["operators"])
    ticks = operator_labels(ops, labels)
    available = np.asarray(coverage.values, dtype=bool)
    times = [int(t) for t in coverage.coords["time"].values]
    n = len(ops)
    fraction = available.mean(axis=0) if times else np.zeros((n, n))
    fraction = np.where(fraction > 0, fraction, np.nan)
    names = [str(op) or "vacuum" for op in ops]
    hover = [
        [
            f"{names[i]}<br>{names[j]}<br>{available[:, i, j].sum()}/{len(times)} times: "
            f"{_span([t for k, t in enumerate(times) if available[k, i, j]])}"
            for j in range(n)
        ]
        for i in range(n)
    ]
    annotate = [["×" if np.isnan(fraction[i, j]) else None for j in range(n)] for i in range(n)]

    axes = resolve_axes(axes, figsize=figsize or _matrix_figsize(n), backend=backend)
    labeled_heatmap(
        axes,
        fraction,
        labels=ticks,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="fraction of times stored",
        hover=hover,
        annotate=annotate,
    )
    form = coverage.attrs.get("key_form", "")
    axes.set(
        xlabel="source",
        ylabel="sink",
        title=title if title is not None else f"Coverage ({form}, {len(times)} times)",
    )
    return axes


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------

#: Above this many operators :meth:`CorrelatorMatrixPlotter.plot_grid` refuses:
#: an N x N grid of that size is too slow to draw and too small to read.
MAX_GRID_OPERATORS = 12

_GRID_WHATS = ("correlator", "normalized", "effective_energy")
_HEATMAP_VALUES = ("normalized", "significance", "hermiticity")
_OFFDIAG_COLOR = "#555555"


class CorrelatorMatrixPlotter:
    """Every view of one correlation matrix, with one label and color per operator.

    Args:
        matrix: The ``CorrelatorMatrix``.
        labels: Operator labels for ticks, legends and grid edges; see
            :func:`operator_labels` (default ``"index"``, the basis position).
        colors: One color per operator; default the style's cycle, resolved
            once here so every view uses the same ones.
        component: ``"real"`` or ``"imag"`` of complex data, for every view.
        style: Visual settings for the series views.

    Attributes:
        operator_labels: The resolved label of each operator.
        operator_colors: The color of each operator in every view.
    """

    def __init__(
        self,
        matrix: CorrelatorMatrix,
        *,
        labels: OperatorLabels = "index",
        colors: Sequence[str] | None = None,
        component: str = "real",
        style: SeriesStyle | None = None,
    ):
        from ..correlators.matrix import CorrelatorMatrix

        if not isinstance(matrix, CorrelatorMatrix):
            raise TypeError(
                f"CorrelatorMatrixPlotter expects a CorrelatorMatrix, got {type(matrix).__name__}"
            )
        if component not in _COMPONENTS:
            raise ValueError(f"component must be one of {_COMPONENTS}, got {component!r}")
        self.matrix = matrix
        self.component = component
        self.style = style or SeriesStyle()
        self.operator_labels = tuple(operator_labels(matrix.operators, labels))
        n = len(matrix.operators)
        if colors is None:
            # Fixed here, before any figure exists, so every view of the matrix
            # agrees; the active style supplies the palette.
            colors = cycle_colors(self.style.color_cycle, n, slp.get_style())
        colors = tuple(colors)
        if len(colors) != n:
            raise ValueError(f"{len(colors)} colors for {n} operators")
        self.operator_colors = colors
        self._names = tuple(str(op) or "vacuum" for op in matrix.operators)

    # -- series views ---------------------------------------------------------

    def diagonal(self) -> CorrelatorPlotter:
        """A :class:`CorrelatorPlotter` of the diagonal, with these labels and colors."""
        return CorrelatorPlotter(
            self.matrix.diagonal(),
            component=self.component,
            labels=self.operator_labels,
            colors=self.operator_colors,
            style=self.style,
        )

    @figure_factory
    def plot_diagonal(
        self,
        *,
        effective: bool = False,
        kind: str = "forward",
        step: int = 1,
        time_extent: int | None = None,
        subtract_const: float = 0.0,
        yscale: str = "log",
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        title: str | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Axes:
        """Every ``C_ii(t)`` overlaid, or their effective energies (``effective=True``)."""
        plotter = self.diagonal()
        if effective:
            return plotter.plot_effective_energy(
                kind=kind,
                step=step,
                time_extent=time_extent,
                subtract_const=subtract_const,
                markers=markers,
                fits=fits,
                title=title, axes=axes, figsize=figsize, backend=backend,
            )
        return plotter.plot(
            yscale=yscale, markers=markers, fits=fits,
            title=title, axes=axes, figsize=figsize, backend=backend,
        )

    # -- grid -----------------------------------------------------------------

    @figure_factory
    def plot_grid(
        self,
        *,
        what: str = "correlator",
        norm_time: int | None = None,
        kind: str = "forward",
        step: int = 1,
        time_extent: int | None = None,
        subtract_const: float = 0.0,
        diag_range: float | None = None,
        offdiag_range: float | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Figure:
        """N x N small multiples: cell ``(i, j)`` is ``C_ij(t)`` against ``t``.

        Args:
            what: ``"correlator"``, ``"normalized"`` (``C~_ij``, see
                ``CorrelatorMatrix.normalized``) or ``"effective_energy"`` (of
                the diagonals; off-diagonal cells hidden).
            norm_time: For ``"normalized"``; default the first time.
            kind, step, time_extent, subtract_const: For ``"effective_energy"``.
            diag_range: Fix the diagonal cells' y range to ``[-0.1, diag_range]``
                (Sigmond's ``DiagonalVerticalRange``); automatic when None.
            offdiag_range: Fix the off-diagonal cells' y range to
                ``[-offdiag_range, offdiag_range]``; automatic when None.
            title: Figure title; default names ``what``.
            figsize, backend: As for every plot.

        Diagonal cells use each operator's color, off-diagonal ones a neutral
        grey. A point is hollow where ``matrix.stored`` is false, i.e. the value
        came from the flipped direction. Row labels are the first column's y
        labels and column labels the first row's titles.

        Raises:
            ValueError: More than :data:`MAX_GRID_OPERATORS` operators.
        """
        n = len(self.matrix.operators)
        if n > MAX_GRID_OPERATORS:
            raise ValueError(
                f"a {n}x{n} grid is too many panels (limit {MAX_GRID_OPERATORS}); use "
                "plot_heatmap, or plot_grid on matrix.select(...) of a sub-basis"
            )
        if what not in _GRID_WHATS:
            raise ValueError(f"what must be one of {_GRID_WHATS}, got {what!r}")
        tN = self.matrix.times[0] if norm_time is None else norm_time
        fig = slp.figure(n, n, figsize=figsize or (1.9 * n + 1.2, 1.6 * n + 1.0), backend=backend)

        if what == "effective_energy":
            eff = self.diagonal().effective_energy(
                kind=kind, step=step, time_extent=time_extent, subtract_const=subtract_const
            )
            ylim = _fixed(diag_range, "diag") or self._auto_range(eff, np.ones(eff.shape, bool))
            for i in range(n):
                self._draw_cell(fig, i, i, eff.isel(operator=i), None, ylim)
                for j in range(n):
                    if j != i:
                        fig.hide(i, j)
        else:
            if what == "correlator":
                values = self.matrix.values
            else:
                values = self.matrix.normalized(tN)
            values = _component(values, self.component)
            diag = np.zeros(values.shape, dtype=bool)
            diag[:, np.arange(n), np.arange(n)] = True
            ranges = {
                True: _fixed(diag_range, "diag") or self._auto_range(values, diag),
                False: _fixed(offdiag_range, "offdiag") or self._auto_range(values, ~diag),
            }
            stored = None if self.matrix.stored is None else np.asarray(self.matrix.stored.values)
            for i in range(n):
                for j in range(n):
                    hollow = None if stored is None else ~stored[:, i, j]
                    cell = values.isel(sink=i, source=j)
                    self._draw_cell(fig, i, j, cell, hollow, ranges[i == j])

        if title is None:
            title = {
                "correlator": "Correlation matrix",
                "normalized": f"Normalized correlation matrix (tN = {tN})",
                "effective_energy": "Effective energies of the diagonal",
            }[what]
        fig.suptitle(title)
        return fig

    def _auto_range(self, values: SigmondData, mask: np.ndarray) -> tuple[float, float] | None:
        """The y range of the kept points under ``mask``, padded like a series plot."""
        y, lo, hi = central_lo_hi(values)
        keep = mask & kept_points(y, lo, hi, self.style.max_rel_error)
        if not keep.any():
            return None
        bottom, top = float((y - lo)[keep].min()), float((y + hi)[keep].max())
        return pad_linear(bottom, top, self.style.ypad_frac)

    def _draw_cell(self, fig, i: int, j: int, cell: SigmondData, hollow, ylim) -> None:
        n = len(self.matrix.operators)
        color = self.operator_colors[i] if i == j else _OFFDIAG_COLOR
        SeriesPlotter(
            cell,
            labels=[f"{self._names[i]}<br>{self._names[j]}"],
            colors=[color],
            hollow=None if hollow is None else [hollow],
            style=self.style,
        ).plot(
            axes=fig.axes(i, j),
            xlabel="$t/a_t$" if i == n - 1 else "",
            ylabel=self.operator_labels[i] if j == 0 else "",
            title=self.operator_labels[j] if i == 0 else None,
            ylim=ylim,
        )

    # -- heatmaps -------------------------------------------------------------

    @figure_factory
    def plot_heatmap(
        self,
        time: int,
        *,
        value: str = "normalized",
        norm_time: int | None = None,
        tol: float = 3.0,
        title: str | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Axes:
        """The matrix at one ``time`` as a heatmap.

        Args:
            time: The time shown.
            value: ``"normalized"``: ``C~_ij`` (the chosen component) on
                RdBu from -1 to 1, normalized at ``norm_time`` (default
                ``time``, so the diagonal is 1). ``"significance"``:
                ``|C_ij| / error`` on viridis from 0. ``"hermiticity"``: the
                ``check_hermitian`` residual in units of the error, with the
                diagonal's imaginary part on the diagonal, on Reds up to
                ``2 * tol``.
            tol: The tolerance for ``"hermiticity"``.
            title, axes, figsize, backend: As for every plot.
        """
        if value not in _HEATMAP_VALUES:
            raise ValueError(f"value must be one of {_HEATMAP_VALUES}, got {value!r}")
        if time not in self.matrix.times:
            raise ValueError(f"time {time} is not a time of the matrix {list(self.matrix.times)}")
        n = len(self.matrix.operators)
        if value == "normalized":
            tN = time if norm_time is None else norm_time
            normalized = self.matrix.normalized(tN).sel(time=time)
            full = np.asarray(normalized.full_sample_value)
            shown = full.imag if self.component == "imag" else full.real
            text = [
                [normalized.isel(sink=i, source=j).pdg_format() for j in range(n)]
                for i in range(n)
            ]
            options = dict(cmap="RdBu", vmin=-1.0, vmax=1.0, colorbar_label=f"normalized (tN={tN})")
            default_title = f"Normalized matrix at t={time} (tN={tN})"
        elif value == "significance":
            shown = np.asarray(self.matrix.significance(time).values)
            text = [[f"{v:.3g} σ" for v in row] for row in shown]
            options = dict(cmap="viridis", vmin=0.0, vmax=None, colorbar_label="|C| / error")
            default_title = f"Significance at t={time}"
        else:
            check = self.matrix.sel_times([time]).check_hermitian(tol)
            shown = np.array(check.residual.sel(time=time).values, dtype=float)
            shown[np.arange(n), np.arange(n)] = check.diagonal_imag.sel(time=time).values
            text = [[f"{v:.3g} σ" for v in row] for row in shown]
            options = dict(
                cmap="Reds", vmin=0.0, vmax=2.0 * tol, colorbar_label="deviation / error"
            )
            default_title = f"Hermiticity at t={time} (tol {tol:g}σ)"
        hover = [
            [f"{self._names[i]}<br>{self._names[j]}<br>{text[i][j]}" for j in range(n)]
            for i in range(n)
        ]
        axes = resolve_axes(axes, figsize=figsize or _matrix_figsize(n), backend=backend)
        labeled_heatmap(axes, shown, labels=self.operator_labels, hover=hover, **options)
        axes.set(title=default_title if title is None else title, xlabel="source", ylabel="sink")
        return axes


def _fixed(extent: float | None, which: str) -> tuple[float, float] | None:
    """Sigmond's fixed vertical ranges: diagonal ``[-0.1, r]``, off-diagonal ``[-r, r]``."""
    if extent is None:
        return None
    extent = float(extent)
    if extent <= 0:
        raise ValueError(f"{which}_range must be positive, got {extent}")
    return (-0.1, extent) if which == "diag" else (-extent, extent)
