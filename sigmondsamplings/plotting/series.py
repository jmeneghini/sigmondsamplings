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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import slat.plotting as slp
from slat.constants import IndexedCycle
from slat.plotting.style import resolve_colors

from ..data import SigmondData
from ._util import central_and_error, latex_label, resolve_axes
from .markers import FitOverlay, HMarker

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
        color_cycle: Series colors; None takes ``slp.get_style()``'s.
        marker_cycle: Series marker symbols; None uses ``"o"`` for all.
        negative: On a log axis, ``"hollow"`` draws ``|y|`` with open markers
            where the central value is negative; ``"drop"`` omits those points.
        max_rel_error: Hide points whose error exceeds this fraction of
            ``|y|`` (Sigmond's cut for effective energies at large ``t``). The
            automatic y range is taken from the kept points only.
        grid_alpha: Grid-line transparency.
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
    grid_alpha: float = 0.3
    ypad_frac: float = 0.05
    errorbar_kwargs: dict[str, Any] = field(default_factory=dict)


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

    def plotted_errors(self):
        return (
            np.where(self.absolute, self.hi, self.lo),
            np.where(self.absolute, self.lo, self.hi),
        )

    def plotted_y(self) -> np.ndarray:
        return np.where(self.absolute, np.abs(self.y), self.y)


def _default_offset(data: SigmondData) -> float:
    from ..correlators.effective import EffectiveEnergyObsInfo

    info = data.observable_info
    return info.time_offset if isinstance(info, EffectiveEnergyObsInfo) else 0.0


def _effective_info(data: SigmondData):
    from ..correlators.effective import EffectiveEnergyObsInfo

    info = data.observable_info
    return info if isinstance(info, EffectiveEnergyObsInfo) else None


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
                x_offset=_default_offset(piece) if x_offset is None else float(x_offset),
                hollow=None if hollow is None else hollow[k],
            )
            for k, (piece, default) in enumerate(pieces)
        ]

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _split(self, data, hue: str | None) -> list[tuple[SigmondData, str | None]]:
        """The 1-D series and their default labels."""
        if isinstance(data, SigmondData):
            self._check(data)
            if data.ndim == 1:
                if hue is not None:
                    raise ValueError(f"hue={hue!r} given, but the data has only dim {self.x!r}")
                return [(data, _latex(data))]
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
        return [(piece, _latex(piece)) for piece in pieces]

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
        ax: Any = None,
    ) -> slp.Axes:
        """Draw every series, then the reference ``markers`` and the ``fits``.

        Args:
            yscale: ``"linear"`` or ``"log"``.
            markers: Horizontal reference levels, drawn across the plot.
            fits: Finished fits to overlay (see :class:`FitOverlay`).
            xlabel, ylabel, title: Axis text; the defaults name the x dim and,
                for an effective energy, the y quantity.
            ylim: Fixed y limits instead of the automatic range.
            axes: Axes to draw on (creates a figure if None).
            figsize: Figure size.
            backend: Backend for this call (uses the active one if None).
            ax: Deprecated alias for ``axes``.

        Returns:
            The Axes drawn on.
        """
        if yscale not in _YSCALES:
            raise ValueError(f"yscale must be one of {_YSCALES}, got {yscale!r}")
        axes = resolve_axes(axes, ax, figsize=figsize, backend=backend)
        colors = self._resolve_colors()
        symbols = self._resolve_markers()
        prepared = [self._prepare(k, s, yscale) for k, s in enumerate(self.series)]

        xlim = self._seed_limits(axes, prepared, yscale, markers or (), ylim)
        for points, series, color, symbol in zip(prepared, self.series, colors, symbols):
            self._draw_series(axes, points, series, color, symbol)
        for marker in markers or ():
            self._draw_marker(axes, marker, xlim)
        for fit in fits or ():
            self._draw_fit(axes, fit, self._fit_color(fit, colors))

        info = _effective_info(self.series[0].data)
        if xlabel is None:
            xlabel = "$t$" if self.x == "time" else self.x
        if ylabel is None and info is not None:
            ylabel = latex_label(info.latex_str)
        axes.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axes.grid(True, alpha=self.style.grid_alpha)
        if len(self.series) > 1 or any(f.label for f in fits or ()):
            axes.legend()
        return axes

    def _resolve_colors(self) -> list[str]:
        if self.colors is not None:
            return list(self.colors)
        cycle = resolve_colors(self.style.color_cycle, slp.get_style())
        return [next(cycle) for _ in self.series]

    def _resolve_markers(self) -> list[str]:
        if self.style.marker_cycle is None:
            return ["o"] * len(self.series)
        cycle = IndexedCycle(self.style.marker_cycle)
        return [next(cycle) for _ in self.series]

    def _prepare(self, k: int, series: _Series, yscale: str) -> _Points:
        data = series.data
        coord = data.coords.get(self.x)
        x_true = (
            np.asarray(coord, dtype=float)
            if coord is not None
            else np.arange(data.shape[0], dtype=float)
        )
        y, yerr = central_and_error(data)
        symmetric = yerr.ndim == y.ndim
        lo, hi = (yerr, yerr) if symmetric else (yerr[0], yerr[1])

        keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if self.style.max_rel_error is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.maximum(lo, hi) / np.abs(y)
            keep &= rel <= self.style.max_rel_error
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
        )

    def _seed_limits(
        self, axes, prepared, yscale, markers, ylim=None
    ) -> tuple[float, float] | None:
        """Set limits from the kept points before drawing, so one noisy time cannot flatten the rest."""
        xs = np.concatenate([p.x[p.keep] for p in prepared])
        if xs.size == 0:
            axes.set(yscale=yscale, ylim=ylim)
            return None
        xlim = (float(xs.min()) - 0.5, float(xs.max()) + 0.5)
        if ylim is not None:
            axes.set(yscale=yscale, xlim=xlim, ylim=tuple(ylim))
            return xlim
        y = np.concatenate([p.plotted_y()[p.keep] for p in prepared])
        lo = np.concatenate([p.plotted_errors()[0][p.keep] for p in prepared])
        hi = np.concatenate([p.plotted_errors()[1][p.keep] for p in prepared])
        levels = np.array([m.y for m in markers], dtype=float)
        pad = self.style.ypad_frac
        if yscale == "log":
            bottom = np.where(y - lo > 0.0, y - lo, y)
            levels = levels[levels > 0.0]
            y_min = float(min(bottom.min(), levels.min(initial=np.inf)))
            y_max = float(max((y + hi).max(), levels.max(initial=-np.inf)))
            factor = (y_max / y_min) ** pad if y_max > y_min else 2.0
            ylim = (y_min / factor, y_max * factor)
        else:
            y_min = float(min((y - lo).min(), levels.min(initial=np.inf)))
            y_max = float(max((y + hi).max(), levels.max(initial=-np.inf)))
            span = y_max - y_min
            margin = pad * span if span > 0 else (abs(y_max) * pad or 1.0)
            ylim = (y_min - margin, y_max + margin)
        axes.set(yscale=yscale, xlim=xlim, ylim=ylim)
        return xlim

    def _draw_series(self, axes, points: _Points, series: _Series, color: str, symbol: str):
        plain = slp.to_display(series.label, mathtext=False)
        y_drawn = points.plotted_y()
        lo_drawn, hi_drawn = points.plotted_errors()
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
                width=self.style.linewidth,
                label=series.label if first else None,
                hover=[
                    _hover(plain, self.x, xt, yv, lo, hi, points.symmetric)
                    for xt, yv, lo, hi in zip(
                        points.x_true[mask], points.y[mask], points.lo[mask], points.hi[mask]
                    )
                ],
                fill=fill,
                native=self.style.errorbar_kwargs or None,
            )
            first = False

    def _draw_marker(self, axes, marker: HMarker, xlim) -> None:
        if xlim is None:
            return
        x0, x1 = xlim
        band = marker.band
        if band is not None:
            axes.band(
                [x0, x1],
                [band[0], band[0]],
                [band[1], band[1]],
                color=marker.color,
                alpha=marker.band_alpha,
                zorder=0,
            )
        axes.hlines(
            marker.y,
            x0,
            x1,
            color=marker.color,
            style=marker.linestyle,
            width=marker.linewidth,
            alpha=marker.alpha,
            zorder=1,
            hover=marker.hover_text,
        )
        if marker.label is not None:
            axes.text(
                x1,
                marker.y,
                marker.label,
                ha="right",
                va="bottom",
                color=marker.color,
                alpha=marker.alpha,
            )

    def _fit_color(self, fit: FitOverlay, colors: list[str]) -> str:
        if fit.color is not None:
            return fit.color
        if fit.series is None:
            return colors[0]
        if isinstance(fit.series, int):
            return colors[fit.series]
        labels = [s.label for s in self.series]
        if fit.series not in labels:
            raise ValueError(f"FitOverlay series={fit.series!r} is not a series label: {labels}")
        return colors[labels.index(fit.series)]

    def _draw_fit(self, axes, fit: FitOverlay, color: str) -> None:
        from ..correlators.effective import effective_energy

        result = fit.result
        theory = result.theory_series()
        info = _effective_info(self.series[0].data)
        width = self.style.linewidth
        alpha = fit.band_alpha

        if info is None:
            if result.x_name != self.x:
                raise ValueError(
                    f"the fit's x is {result.x_name!r} but this plot's x is {self.x!r}"
                )
            x = np.asarray(theory.coords[self.x], dtype=float)
            y, yerr = central_and_error(theory)
            lo, hi = (yerr, yerr) if yerr.ndim == y.ndim else (yerr[0], yerr[1])
            axes.band(x, y - lo, y + hi, color=color, alpha=alpha, label=fit.label, zorder=0)
            axes.line(x, y, color=color, width=width, zorder=1)
            return

        # Effective-energy plot: the energy parameter over the fit range, and the
        # model's own effective energy from the same calculator as the data.
        x = np.asarray(result.x, dtype=float)
        label = fit.label
        if fit.energy is not None:
            y, yerr = central_and_error(result.param(fit.energy))
            lo, hi = (yerr, yerr) if np.ndim(yerr) == 0 else (yerr[0], yerr[1])
            x0, x1 = float(x.min()), float(x.max())
            axes.band(
                [x0, x1],
                [y - lo] * 2,
                [y + hi] * 2,
                color=color,
                alpha=alpha,
                label=label,
                zorder=0,
            )
            axes.line([x0, x1], [y, y], color=color, width=width, zorder=1)
            label = None
        model = effective_energy(
            theory,
            kind=info.kind,
            step=info.step,
            time_extent=info.time_extent,
            subtract_const=info.subtract_const,
            on_fail="nan",
        )
        central, _ = central_and_error(model)
        times = np.asarray(model.coords["time"], dtype=float) + info.time_offset
        axes.line(times, central, color=color, width=width, style="--", label=label, zorder=1)


def _latex(data: SigmondData) -> str | None:
    text = data.observable_info.latex_str
    return latex_label(text) if text else None


def _hover(label, x_name, x, y, lo, hi, symmetric) -> str:
    err = f"± {lo:.2g}" if symmetric else f"+{hi:.2g}/−{lo:.2g}"
    head = f"{label}<br>" if label else ""
    return f"{head}{x_name}={x:g}: {y:.6g} {err}"
