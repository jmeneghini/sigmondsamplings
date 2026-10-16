"""Series plots: any ``SigmondData`` over one dim, drawn as error bars.

Correlators, effective energies and (later) principal correlators are all the
same kind of object: a ``SigmondData`` over one dim, usually ``time``, with at
most one more dim whose entries are separate series. :class:`SeriesPlotter`
draws that once; the correlator plotters are thin layers over it.

It draws numbers computed elsewhere and does no math of its own: an overlaid
fit's effective energy comes from the same
:func:`~sigmondsamplings.correlators.effective.effective_energy` as the data's.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from uncertainties import ufloat

import slat.plotting as slp
from slat.constants import IndexedCycle
from slat.plotting.lifecycle import figure_factory

from ..data import SigmondData
from ..rcparams import rc
from ._util import (
    _central_and_error_with_bounds,
    central_and_error,
    central_lo_hi,
    cycle_colors,
    kept_points,
    pad_linear,
    resolve_axes,
)
from .markers import FitOverlay, HMarker, _snapshot_marker, draw_hmarker

__all__ = ["SeriesPlotter", "SeriesStyle"]

_NEGATIVE = ("hollow", "drop")
_YSCALES = ("linear", "log")


@dataclass
class SeriesStyle:
    """Every visual setting of a series plot.

    Attributes:
        markersize / capsize / linewidth: Error-bar marks, in points.
        dodge: Total x spread shared by the series, in x units, so their error
            bars do not overlap.
        color_cycle: Series colors; None takes the figure style's palette.
        marker_cycle: Series symbols; None takes the figure style's marker palette.
        negative: On a log axis, ``"hollow"`` draws ``|y|`` with open markers
            where the central value is negative; ``"drop"`` omits those points.
        max_rel_error: Hide points whose error exceeds this fraction of
            ``|y|`` (Sigmond's cut for effective energies at large ``t``). The
            automatic y range is taken from the kept points only.
        ypad_frac: Padding added to the automatic y range, as a fraction of it
            (of the decades spanned, on a log axis).
        errorbar_kwargs: Backend kwargs forwarded to every error-bar mark.
    """

    markersize: float = 5.0
    capsize: float = 3.0
    linewidth: float = 1.2
    dodge: float = 0.2
    color_cycle: Iterable[str] | None = None
    marker_cycle: Iterable[str] | None = None
    negative: str = "hollow"
    max_rel_error: float | None = None
    ypad_frac: float = 0.05
    errorbar_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Palettes are reused across views; snapshot one-shot iterables once.
        if self.color_cycle is not None:
            self.color_cycle = tuple(self.color_cycle)
        if self.marker_cycle is not None:
            self.marker_cycle = tuple(self.marker_cycle)


@dataclass
class _Series:
    data: SigmondData  # 1-D over the x dim
    label: str | None
    x_offset: float
    hollow: np.ndarray | None = None  # caller-flagged open markers, aligned with data


@dataclass
class _Points:
    """One series prepared for drawing."""

    x: np.ndarray  # drawn positions: coord + offset + dodge
    x_true: np.ndarray  # coord values, for hover
    y: np.ndarray  # signed central values
    lo: np.ndarray  # lower error offsets
    hi: np.ndarray  # upper error offsets
    keep: np.ndarray  # drawn at all
    hollow: np.ndarray  # drawn with open markers
    absolute: np.ndarray  # drawn as |y| (negative values on a log axis)
    symmetric: bool
    interval_bounds: tuple[np.ndarray, np.ndarray] | None

    def plotted_errors(self):
        return (
            np.where(self.absolute, self.hi, self.lo),
            np.where(self.absolute, self.lo, self.hi),
        )

    def plotted_y(self) -> np.ndarray:
        return np.where(self.absolute, np.abs(self.y), self.y)


@dataclass
class _FitCurve:
    x: np.ndarray
    y: np.ndarray
    label: str | None
    linestyle: str = "-"
    bounds: tuple[np.ndarray, np.ndarray] | None = None


def _effective_info(data: SigmondData):
    from ..metadata import EffectiveEnergyInfo

    if "time" not in data.dims:
        return None
    elements = data.ts.elements
    if elements.ndim:
        raise ValueError("select a single series before resolving effective-energy settings")
    element = elements.item()
    return element if isinstance(element, EffectiveEnergyInfo) else None


def _default_offset(data: SigmondData) -> float:
    return data.ts.axis.offset


def _series_label(data, x):
    from ..labels import coordinate_label, correlator_label
    from ..correlators.observables import CorrelatorObsInfo
    from ..metadata import AtObsInfo

    if x == "time":
        data.ts.elements  # Validate explicit context even with an offset/label override.
    if data.observable_info._latex_str is not None:
        return slp.ensure_math(data.observable_info.latex_str)
    if x == "time" and data.shape[0]:
        infos = list(data.infos.flat)
        overrides = [info._latex_str for info in infos]
        if overrides[0] is not None and all(label == overrides[0] for label in overrides):
            label = overrides[0]
            if "{VAR}" in label or "{var}" in label:
                label = coordinate_label(label, "t")
            return slp.ensure_math(label)
        element = data.ts.elements.item()
        if isinstance(infos[0], AtObsInfo):
            return slp.ensure_math(coordinate_label(element.latex_str, "t"))
        if isinstance(infos[0], CorrelatorObsInfo):
            return slp.ensure_math(correlator_label(element, rot_idx=infos[0].rot_idx))
    return slp.ensure_math(data.observable_info.latex_str)


class SeriesPlotter:
    """Plot ``SigmondData`` series over one dim as error bars.

    Args:
        data: One ``SigmondData`` with the ``x`` dim and at most one other dim
            (each of whose entries is a series), or a sequence of 1-D ones.
        x: The dim drawn along the x axis.
        x_offset: Added to x when drawing. None takes it from the data: the
            plot offset of an effective energy, else 0.
        hue: The dim whose entries are separate series; default the one
            non-``x`` dim of a 2-D input.
        labels: One label per series. Default: the ``hue`` coord values, or each
            1-D input's ``latex_str``.
        colors: One color per series, pinning them instead of the cycle.
        hollow: One boolean mask per series, aligned with its ``x`` values,
            marking points to draw with open markers (e.g. matrix cells read
            from the flipped direction). None draws every point filled, apart
            from negative values on a log axis.
        style: Visual settings.
    """

    def __init__(
        self,
        data: SigmondData | Sequence[SigmondData],
        *,
        x: str = "time",
        x_offset: float | None = None,
        hue: str | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        hollow: Sequence[Sequence[bool]] | None = None,
        style: SeriesStyle | None = None,
    ):
        self.x = x
        self.style = style or SeriesStyle()
        if self.style.negative not in _NEGATIVE:
            raise ValueError(f"style.negative must be one of {_NEGATIVE}")
        pieces = self._split(data, hue)
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(pieces):
                raise ValueError(f"{len(labels)} labels for {len(pieces)} series")
        if colors is not None:
            colors = tuple(colors)
            if len(colors) != len(pieces):
                raise ValueError(f"{len(colors)} colors for {len(pieces)} series")
        if hollow is not None:
            hollow = [np.asarray(mask, dtype=bool) for mask in hollow]
            if len(hollow) != len(pieces):
                raise ValueError(f"{len(hollow)} hollow masks for {len(pieces)} series")
            for mask, (piece, _) in zip(hollow, pieces):
                if mask.shape != piece.shape:
                    raise ValueError(
                        f"a hollow mask of shape {mask.shape} for a series of {piece.shape}"
                    )
        self.colors = colors
        self.series = [
            _Series(
                data=piece,
                label=labels[k] if labels is not None else default,
                x_offset=(_default_offset(piece) if x == "time" else 0.0) if x_offset is None else float(x_offset),
                hollow=None if hollow is None else hollow[k],
            )
            for k, (piece, default) in enumerate(pieces)
        ]

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _split(self, data, hue: str | None) -> list[tuple[SigmondData, str]]:
        """The 1-D series and their default labels."""
        if isinstance(data, SigmondData):
            self._check(data)
            if data.ndim == 1:
                if hue is not None:
                    raise ValueError(f"hue={hue!r} given, but the data has only dim {self.x!r}")
                return [(data, _series_label(data, self.x))]
            if data.ndim > 2:
                raise ValueError(
                    f"a series plot takes {self.x!r} plus at most one more dim, got dims "
                    f"{data.dims}; select down with .sel()/.isel() first"
                )
            (other,) = [d for d in data.dims if d != self.x]
            hue = other if hue is None else hue
            if hue != other:
                raise ValueError(f"hue={hue!r} is not a dim of the data (dims {data.dims})")
            values = data.coords.get(hue)
            return [
                (
                    data.isel(**{hue: k}),
                    str(values[k]) if values is not None else f"{hue}={k}",
                )
                for k in range(data.shape[data.dims.index(hue)])
            ]

        pieces = list(data)
        if not pieces:
            raise ValueError("no series to plot")
        if hue is not None:
            raise ValueError("hue= applies to a single 2-D input, not a sequence")
        for piece in pieces:
            if not isinstance(piece, SigmondData):
                raise TypeError(f"expected SigmondData series, got {type(piece).__name__}")
            self._check(piece)
            if piece.ndim != 1:
                raise ValueError(
                    f"each series in a sequence must be 1-D over {self.x!r}, got dims {piece.dims}"
                )
        return [(piece, _series_label(piece, self.x)) for piece in pieces]

    def _check(self, data: SigmondData) -> None:
        if self.x not in data.dims:
            raise ValueError(f"the data has no {self.x!r} dim (dims {data.dims})")
        if data.is_complex:
            raise ValueError(
                "cannot plot complex data; project explicitly first, e.g. .to_real() "
                "or CorrelatorMatrix.diagonal(component='real')"
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @figure_factory
    def plot(
        self,
        *,
        yscale: str = "linear",
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        ylim: tuple[float, float] | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
    ) -> slp.Axes:
        """Draw every series, then the reference ``markers`` and the ``fits``.

        Args:
            yscale: ``"linear"`` or ``"log"``.
            markers: Horizontal reference levels, drawn across the plot.
            fits: Finished fits to overlay (see :class:`FitOverlay`).
            xlabel, ylabel, title: Axis text; the defaults name the x dim and,
                for an effective energy, the y quantity.
            ylim: Fixed y limits. Otherwise preserve explicit caller limits;
                automatic ranges include kept data, marker bands and fit curves.
                With only markers, x spans the available series coordinates.
            axes: Axes to draw on (creates a figure if None).
            figsize: Figure size.
            backend: Backend for this call (uses the active one if None).

        Returns:
            The Axes drawn on.
        """
        if yscale not in _YSCALES:
            raise ValueError(f"yscale must be one of {_YSCALES}, got {yscale!r}")
        markers = tuple(_snapshot_marker(marker) for marker in markers or ())
        fits = tuple(fits or ())
        prepared = [self._prepare(k, s, yscale) for k, s in enumerate(self.series)]
        fit_curves = [self._prepare_fit(fit) for fit in fits]
        axes = resolve_axes(axes, figsize=figsize, backend=backend)
        colors = self._resolve_colors(axes.style)
        symbols = self._resolve_markers(axes.style)

        xlim = self._seed_limits(axes, prepared, yscale, markers, ylim, fit_curves)
        for points, series, color, symbol in zip(prepared, self.series, colors, symbols):
            with axes.legend_group():
                self._draw_series(axes, points, series, color, symbol)
        for marker in markers or ():
            self._draw_marker(axes, marker, xlim)
        for fit, curves in zip(fits, fit_curves):
            with axes.legend_group():
                self._draw_fit(axes, fit, curves, self._fit_color(fit, colors))

        effective_infos = [_effective_info(series.data) for series in self.series]
        info = effective_infos[0]
        if xlabel is None:
            xlabel = "$t/a_t$" if self.x == "time" else self.x
        if ylabel is None and info is not None:
            from ..labels import effective_energy_label
            references = [None if item is None or item.target is None else item.target.ref_particle
                          for item in effective_infos]
            if all(item is not None for item in effective_infos) and len(set(references)) == 1:
                rotations = {item.rot_idx for item in effective_infos}
                ylabel = slp.ensure_math(effective_energy_label(
                    rot_idx=info.rot_idx if len(rotations) == 1 else None,
                    ref_particle=references[0], lattice=True))
        axes.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if len(self.series) > 1 or any(f.label for f in fits or ()):
            axes.legend()
        return axes

    def _resolve_colors(self, style: slp.PlotStyle) -> list[str]:
        if self.colors is not None:
            return list(self.colors)
        return cycle_colors(self.style.color_cycle, len(self.series), style)

    def _resolve_markers(self, style: slp.PlotStyle) -> list[str]:
        cycle = IndexedCycle(style.markers if self.style.marker_cycle is None
                             else self.style.marker_cycle)
        return [next(cycle) for _ in self.series]

    def _prepare(self, k: int, series: _Series, yscale: str) -> _Points:
        data = series.data
        coord = data.coords.get(self.x)
        x_true = (
            np.asarray(coord, dtype=float)
            if coord is not None
            else np.arange(data.shape[0], dtype=float)
        )
        y, yerr, bounds = _central_and_error_with_bounds(data)
        # A symmetric error is one array; an interval is (lower, upper) offsets.
        symmetric = yerr.ndim == y.ndim
        lo, hi = (yerr, yerr) if symmetric else yerr

        keep = np.isfinite(x_true) & kept_points(y, lo, hi, self.style.max_rel_error)
        absolute = np.zeros_like(keep)
        if yscale == "log":
            keep &= y != 0.0
            negative = y < 0.0
            if self.style.negative == "drop":
                keep &= ~negative
            else:
                absolute = keep & negative
        hollow = absolute | (keep & series.hollow if series.hollow is not None else False)

        n = len(self.series)
        dodge = self.style.dodge * (k / (n - 1) - 0.5) if n > 1 else 0.0
        return _Points(
            x=x_true + series.x_offset + dodge,
            x_true=x_true,
            y=y,
            lo=lo,
            hi=hi,
            keep=keep,
            hollow=hollow,
            absolute=absolute,
            symmetric=symmetric,
            interval_bounds=bounds,
        )

    def _seed_limits(self, axes, prepared, yscale, markers, ylim=None, fit_curves=()):
        """Respect explicit axes limits; automatic ranges include every drawn overlay."""
        xlim = axes.get_limits("x", explicit_only=True)
        if ylim is None:
            ylim = axes.get_limits("y", explicit_only=True)
        xs = [p.x[p.keep] for p in prepared]
        ys = []
        for points in prepared:
            y = points.plotted_y()[points.keep]
            lo, hi = points.plotted_errors()
            bottom = y - lo[points.keep]
            if yscale == "log":
                bottom = np.where(bottom > 0, bottom, y)
            ys.extend((bottom, y + hi[points.keep]))
        for marker in markers:
            ys.append(np.asarray([marker.y, *(marker.band or ())]))
        for curves in fit_curves:
            for curve in curves:
                keep = np.isfinite(curve.x) & np.isfinite(curve.y)
                if yscale == "log":
                    keep &= curve.y > 0
                xs.append(curve.x[keep])
                ys.append(curve.y[keep])
                if curve.bounds is not None:
                    ys.extend(bound[keep] for bound in curve.bounds)
        x = np.concatenate(xs)
        if not x.size and markers:
            # A reference-only plot still spans the series' available x coordinates.
            x = np.concatenate([p.x[np.isfinite(p.x)] for p in prepared])
        if xlim is None and x.size:
            xlim = (float(x.min()) - 0.5, float(x.max()) + 0.5)
        if ylim is None and ys:
            y = np.concatenate(ys)
            y = y[np.isfinite(y)]
            if yscale == "log":
                y = y[y > 0]
            if y.size:
                bottom, top = float(y.min()), float(y.max())
                if yscale == "log":
                    factor = (top / bottom) ** self.style.ypad_frac if top > bottom else 2.0
                    ylim = (bottom / factor, top * factor)
                else:
                    ylim = pad_linear(bottom, top, self.style.ypad_frac)
        axes.set(yscale=yscale, xlim=xlim, ylim=ylim)
        return xlim

    def _draw_series(self, axes, points: _Points, series: _Series, color: str, symbol: str):
        y_drawn = points.plotted_y()
        lo_drawn, hi_drawn = points.plotted_errors()
        # Matplotlib ignores hover. Recording retains it as an observable output.
        hover = None
        if axes.backend != "matplotlib":
            gaussian = points.lo if points.symmetric else np.asarray(series.data.error)
            hover = {
                i: _hover(
                    series.label, self.x, points.x_true[i], points.y[i], gaussian[i],
                    None if points.interval_bounds is None else (
                        points.interval_bounds[0][i], points.interval_bounds[1][i]
                    ),
                )
                for i in np.flatnonzero(points.keep)
            }
        first = True
        for fill, mask in ((True, points.keep & ~points.hollow), (False, points.hollow)):
            if not mask.any():
                continue
            yerr = (
                lo_drawn[mask] if points.symmetric else np.stack([lo_drawn[mask], hi_drawn[mask]])
            )
            axes.errorbar(
                points.x[mask],
                y_drawn[mask],
                yerr=yerr,
                color=color,
                marker=symbol,
                markersize=self.style.markersize,
                capsize=self.style.capsize,
                linewidth=self.style.linewidth,
                label=series.label if first else None,
                hover=None if hover is None else [hover[i] for i in np.flatnonzero(mask)],
                fill=fill,
                native=self.style.errorbar_kwargs or None,
            )
            first = False

    def _draw_marker(self, axes, marker: HMarker, xlim) -> None:
        if xlim is None:
            return
        x0, x1 = xlim
        draw_hmarker(axes, marker, x0, x1, x1, ha="right", va="bottom")

    def _fit_series_index(self, fit: FitOverlay) -> int:
        if fit.series is None:
            return 0
        if isinstance(fit.series, int):
            return fit.series
        labels = [s.label for s in self.series]
        if fit.series not in labels:
            raise ValueError(f"FitOverlay series={fit.series!r} is not a series label: {labels}")
        return labels.index(fit.series)

    def _fit_color(self, fit: FitOverlay, colors: list[str]) -> str:
        return fit.color if fit.color is not None else colors[self._fit_series_index(fit)]

    def _prepare_fit(self, fit: FitOverlay):
        from ..correlators.effective import effective_energy

        result = fit.result
        theory = result.theory_series()
        series = self.series[self._fit_series_index(fit)]
        info = _effective_info(series.data)
        if info is None:
            if result.x_name != self.x:
                raise ValueError(f"the fit's x is {result.x_name!r} but this plot's x is {self.x!r}")
            x = np.asarray(theory.coords[self.x], dtype=float) + series.x_offset
            y, lo, hi = central_lo_hi(theory)
            return (_FitCurve(x, y, fit.label, bounds=(y - lo, y + hi)),)

        curves = []
        label = fit.label
        if fit.energy is not None:
            x = np.asarray(result.x, dtype=float)
            y, lo, hi = central_lo_hi(result.param(fit.energy))
            span = np.array([x.min(), x.max()])
            curves.append(_FitCurve(span, np.full(2, y), label,
                                    bounds=(np.full(2, y - lo), np.full(2, y + hi))))
            label = None
        model = effective_energy(
            theory, kind=info.kind, step=info.step, time_extent=info.time_extent,
            subtract_const=info.subtract_const, on_fail="nan",
        )
        central, _ = central_and_error(model)
        times = np.asarray(model.coords["time"], dtype=float) + series.x_offset
        curves.append(_FitCurve(times, central, label, linestyle="--"))
        return tuple(curves)

    def _draw_fit(self, axes, fit: FitOverlay, curves, color):
        for curve in curves:
            label = curve.label
            if curve.bounds is not None:
                axes.band(curve.x, *curve.bounds, color=color, alpha=fit.band_alpha,
                          label=label, zorder=0)
                label = None
            axes.line(curve.x, curve.y, color=color, linewidth=self.style.linewidth,
                      linestyle=curve.linestyle, label=label, zorder=1)


def _hover(label, x_name, x, central, gaussian_error, bounds) -> str:
    """Format batched estimates with the scalar PDG convention and exact CI bounds."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.",
            category=UserWarning,
        )
        text = f"{ufloat(central, gaussian_error):.2uS}"
    if bounds is not None:
        text += f"<br>{rc['plot.confidence_level']:.0%} CI: [{bounds[0]:.6g}, {bounds[1]:.6g}]"
    head = f"{label}<br>" if label else ""
    return f"{head}{x_name}={x:g}: {text}"
