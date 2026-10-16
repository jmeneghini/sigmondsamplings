"""Shared helpers for the sigmondsamplings plot frontends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import slat.plotting as slp
from slat.plotting.style import resolve_colors

from ..rcparams import rc

if TYPE_CHECKING:
    from ..data import SigmondData

__all__ = [
    "central_and_error",
    "central_lo_hi",
    "cycle_colors",
    "kept_points",
    "pad_linear",
    "full_sample_and_bounds",
    "energy_level_hover_text",
    "ni_level_hover_text",
    "estimate_hover_text",
    "labeled_heatmap",
    "resolve_axes",
    "summary_grid",
]

#: An NI level's residue rank in the irrep basis (0: it does not couple), or
#: ``"unclassified"`` when the rank could not be determined.
NIRank = int | Literal["unclassified"]


def resolve_axes(
    axes: Any = None,
    *,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    style: slp.PlotStyle | None = None,
) -> slp.Axes:
    """Return the Axes to draw on, creating a figure when none was given.

    ``axes`` may be an :class:`slat.plotting.Axes` or a native axes that
    :func:`slat.plotting.wrap` adopts, such as a raw ``matplotlib.axes.Axes``.
    """
    if axes is not None:
        return slp.wrap(axes, style=style)
    return slp.figure(figsize=figsize, style=style, backend=backend).axes()


def full_sample_and_bounds(
    data: SigmondData,
    *,
    error_metric: str | None = None,
    confidence_level: float | None = None,
    percentiles: tuple[float, float] | None = None,
) -> tuple[float, tuple[float, float]]:
    """Full-sample scalar and absolute uncertainty bounds.

    Explicit percentiles use the resamples (bins for raw-bin data), never the
    full-sample entry. Bounds remain exact when they do not contain the point.
    Percentiles and confidence intervals require bootstrap samplings or bins;
    they are unavailable for jackknife data.
    """
    from ..plot_contracts import UncertaintyOptions

    if data.ndim != 0 or data.is_complex:
        raise ValueError("reference levels require a real, scalar observable")
    resolved = UncertaintyOptions(error_metric=error_metric, confidence_level=confidence_level,
                                  percentiles=percentiles).resolved()
    return _full_sample_and_bounds(data, resolved)


def _full_sample_and_bounds(
    data: SigmondData, resolved: dict[str, Any]
) -> tuple[float, tuple[float, float]]:
    """Scalar bounds using uncertainty choices already validated by the caller."""
    if data.ndim != 0 or data.is_complex:
        raise ValueError("reference levels require a real, scalar observable")
    metric = resolved["error_metric"]
    central = float(data.full_sample_value)
    if metric == "error":
        error = float(data.error)
        bounds = (central - error, central + error)
    else:
        if data._analyzer.method == "jackknife":
            raise ValueError(
                "confidence intervals require bootstrap samplings or bins; "
                "unavailable for jackknife data"
            )
        bounds = data.confidence_interval(
            resolved.get("confidence_level"), percentiles=resolved.get("percentiles")
        )
    if not np.all(np.isfinite((central, *bounds))) or bounds[0] > bounds[1]:
        raise ValueError("reference levels require finite central values and ordered bounds")
    return central, bounds


def _hover_context(
    irrep, psq, level, ni_pair, ni_rank: NIRank | None, include_level: bool
) -> list[str]:
    """The ``NI: pair`` row and the ``irrep, d²=…, level/coupling`` row, when present."""
    rows = [f"NI: {ni_pair}"] if ni_pair else []
    context = [str(irrep)] if irrep is not None else []
    if psq is not None:
        context.append(f"d²={psq}")
    if include_level and level is not None:
        context.append(f"level {level}")
    if ni_pair and ni_rank is not None:
        context.append("coupling unclassified" if ni_rank == "unclassified" else f"rank {ni_rank}")
    if context:
        rows.append(", ".join(context))
    return rows


def ni_level_hover_text(
    ni_pair: str, irrep, psq, ecm: float, *, ni_rank: NIRank | None = None
) -> str:
    """Tooltip for a non-interacting level known only as a number, without a sampling.

    Laid out as :func:`energy_level_hover_text` lays out a sampling-backed NI
    level, with the coupling (``ni_rank``) in the context row.
    """
    rows = _hover_context(irrep, psq, None, ni_pair, ni_rank, include_level=False)
    rows.append(f"Ecm = {ecm:.6g}")
    rows.append("Gaussian error and percentiles unavailable")
    return "<br>".join(rows)


def energy_level_hover_text(
    sampling: SigmondData, *, percentiles=None, confidence_level=None,
    ni_pair: str | None = None, irrep=None, psq=None, energy_type=None,
    ref_particle=None, include_level: bool = True, interval_bounds=None,
    ni_rank: NIRank | None = None,
) -> str:
    """Shared energy tooltip: full value with Gaussian error and percentile bounds.

    The PDG value always uses the sampling's statistical error, independently
    of the plotted interval. NI callers may supply a compact pair string and
    override metadata on a sampling derived through kinematic arithmetic.
    ``ni_rank`` is the NI level's residue rank in the irrep basis (0 means it
    does not couple) or ``"unclassified"``; it is shown only for NI levels,
    alongside the irrep.
    """
    obs = sampling.observable_info
    irrep = irrep if irrep is not None else getattr(obs, "irrep", None)
    psq = psq if psq is not None else getattr(obs, "psq", None)
    kind = energy_type or getattr(obs, "energy_type", "ecm")
    ref = ref_particle or getattr(obs, "ref_particle", None)
    label = {"ecm": "Ecm", "elab": "Elab"}.get(kind, kind)
    if ref:
        label += f"/M_{ref}"
    rows = _hover_context(
        irrep, psq, getattr(obs, "level_index", None), ni_pair, ni_rank, include_level
    )
    rows.append(f"{label} = {sampling.pdg_format()}")
    if sampling._analyzer.method == "jackknife":
        rows.append("Percentiles unavailable for jackknife")
    else:
        q = percentiles
        if q is None:
            confidence = confidence_level if confidence_level is not None else rc["plot.confidence_level"]
            q = (50 * (1 - confidence), 50 * (1 + confidence))
        bounds = interval_bounds
        if bounds is None:
            _, bounds = full_sample_and_bounds(sampling, percentiles=q)
        rows.append(f"{q[0]:g}–{q[1]:g}%: [{bounds[0]:.6g}, {bounds[1]:.6g}]")
    return "<br>".join(rows)


def estimate_hover_text(
    data: SigmondData, *, central: str | None = None, confidence_level: float | None = None
) -> str:
    """A scalar estimate as hover text, in the data's own PDG form, e.g. ``1.235(12)``.

    The central value is ``central`` (a :meth:`SigmondData.central` metric),
    default ``rc["plot.central_metric"]``, so the hover names the value that
    is drawn. When ``rc["plot.error_metric"]`` is ``"confidence_interval"``
    the drawn interval follows on its own row.
    """
    metric = rc["plot.central_metric"] if central is None else central
    text = data.pdg_format(central=None if metric == "full_sample_value" else metric)
    if rc["plot.error_metric"] == "confidence_interval" and not data.is_complex:
        level = rc["plot.confidence_level"] if confidence_level is None else confidence_level
        try:
            lower, upper = data.confidence_interval(level)
        except ValueError:
            return text
        text += f"<br>{level:.0%} CI: [{lower:.6g}, {upper:.6g}]"
    return text


def central_and_error(
    data: SigmondData, *, confidence_level: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Central values and error bars for plotting, as numpy arrays.

    Honours ``rc["plot.central_metric"]`` and ``rc["plot.error_metric"]``. The
    central value has ``data.shape``. The error is either a symmetric array of
    the same shape or, for ``"confidence_interval"``, a ``(2, *data.shape)``
    array of ``(lower, upper)`` offsets from the central value, which is what
    :meth:`slat.plotting.Axes.errorbar` takes as an asymmetric ``yerr``.
    Offsets are clipped at zero, since the central value can fall outside the
    percentile interval.

    Args:
        data: Real samplings or bins, any shape.
        confidence_level: For the interval; defaults to
            ``rc["plot.confidence_level"]``.

    Raises:
        ValueError: Complex data (project to a component first), or a metric
            the sampling mode cannot provide (the bootstrap-only ones on
            jackknife data).
    """
    central, error, _ = _central_and_error_with_bounds(data, confidence_level=confidence_level)
    return central, error


def _central_and_error_with_bounds(data: SigmondData, *, confidence_level=None):
    """Keep absolute CI bounds alongside the clipped offsets used by error bars."""
    if data.is_complex:
        raise ValueError(
            "cannot plot complex data; project explicitly first, e.g. .to_real() "
            "or CorrelatorMatrix.diagonal(component='real')"
        )
    central_metric = rc["plot.central_metric"]
    error_metric = rc["plot.error_metric"]
    try:
        central = np.asarray(data.central(central_metric))
    except ValueError as exc:
        raise ValueError(f"rc['plot.central_metric'] = {central_metric!r}: {exc}") from None

    if error_metric == "error":
        return central, np.asarray(data.error), None

    level = rc["plot.confidence_level"] if confidence_level is None else confidence_level
    try:
        lower, upper = (np.asarray(bound) for bound in data.confidence_interval(level))
    except ValueError as exc:
        raise ValueError(f"rc['plot.error_metric'] = {error_metric!r}: {exc}") from None
    offsets = np.stack([np.maximum(central - lower, 0.0), np.maximum(upper - central, 0.0)])
    return central, offsets, (lower, upper)


def central_lo_hi(
    data: SigmondData, *, confidence_level: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`central_and_error` with the error split into lower and upper offsets.

    A symmetric error gives equal ``lo`` and ``hi``; callers that must tell the
    two cases apart use :func:`central_and_error`.
    """
    central, yerr = central_and_error(data, confidence_level=confidence_level)
    if yerr.ndim == central.ndim:
        return central, yerr, yerr
    return central, yerr[0], yerr[1]


def kept_points(y, lo, hi, max_rel_error: float | None = None) -> np.ndarray:
    """Mask of the points a series plot draws: finite, and within ``max_rel_error``.

    ``max_rel_error`` bounds the larger error offset as a fraction of ``|y|``.
    """
    keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if max_rel_error is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            keep &= np.maximum(lo, hi) / np.abs(y) <= max_rel_error
    return keep


def pad_linear(bottom: float, top: float, frac: float) -> tuple[float, float]:
    """``(bottom, top)`` widened by ``frac`` of the span on each side.

    A zero span is widened by ``frac`` of ``|top|``, or by 1 when that is zero.
    """
    span = top - bottom
    margin = frac * span if span > 0 else (abs(top) * frac or 1.0)
    return (bottom - margin, top + margin)


def labeled_heatmap(
    axes: slp.Axes,
    values,
    *,
    labels: Sequence[str],
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    hover: Sequence[Sequence[str]] | None = None,
    annotate: Sequence[Sequence[str | None]] | None = None,
    native: dict[str, Any] | None = None,
) -> None:
    """A square matrix as a heatmap with row 0 at the top and labelled ticks.

    ``labels`` name both the rows and the columns. ``hover`` is per-cell hover
    text (plotly). ``annotate`` is per-cell text drawn at the cell centre;
    ``None`` or ``""`` entries draw nothing. NaN cells render empty.
    """
    values = np.asarray(values)
    axes.heatmap(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # Row 0 at the top, the usual convention for a matrix.
        origin="upper",
        colorbar_label=colorbar_label,
        hover=hover,
        native=native,
    )
    positions = range(len(labels))
    axes.ticks("x", positions, list(labels), rotation=45)
    axes.ticks("y", positions, list(labels))
    if annotate is not None:
        for i, row in enumerate(annotate):
            for j, text in enumerate(row):
                if text:
                    axes.text(j, i, text, ha="center", va="center")


def cycle_colors(palette: Sequence[str] | None, n: int, style: slp.PlotStyle) -> list[str]:
    """``n`` colors cycling through ``palette``, or through ``style``'s when None."""
    cycle = resolve_colors(palette, style)
    return [next(cycle) for _ in range(n)]


def summary_grid(n: int) -> tuple[int, int]:
    """``(nrows, ncols)`` for ``n`` summary panels: one row up to 3, then near-square."""
    if n <= 3:
        return (1, n)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)
