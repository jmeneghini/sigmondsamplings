"""Shared helpers for the sigmondsamplings plot frontends."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import slat.plotting as slp

from ..rcparams import rc

if TYPE_CHECKING:
    from ..data import SigmondData

__all__ = [
    "central_and_error",
    "full_sample_and_bounds",
    "energy_level_hover_text",
    "labeled_heatmap",
    "latex_label",
    "require_backend",
    "resolve_axes",
    "summary_grid",
]


def resolve_axes(
    axes: slp.Axes | None = None,
    ax: Any = None,
    *,
    figsize: tuple[float, float] | None = None,
    backend: str | None = None,
    style: slp.PlotStyle | None = None,
) -> slp.Axes:
    """Return the Axes to draw on, creating a figure when none was given.

    ``ax`` is the pre-backend spelling and is accepted for compatibility: a raw
    ``matplotlib.axes.Axes`` handed in that way is wrapped transparently.
    """
    if ax is not None:
        if axes is not None:
            raise TypeError("Pass either `axes` or the legacy `ax`, not both")
        warnings.warn(
            "The `ax` argument is deprecated; pass `axes` instead. "
            "A raw matplotlib Axes is still accepted and will be wrapped.",
            DeprecationWarning,
            stacklevel=3,
        )
        axes = ax
    if axes is not None:
        return slp.wrap(axes, style=style)
    return slp.figure(figsize=figsize, style=style, backend=backend).axes()


def require_backend(axes: slp.Axes, name: str, what: str) -> None:
    """Raise unless ``axes`` uses backend ``name``.

    For plots that are genuinely single-backend rather than merely unported.
    """
    if axes.backend != name:
        raise slp.UnsupportedFeature(
            f"{what} is {name}-only; the active backend is {axes.backend!r}"
        )


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
    Percentiles and confidence intervals require bootstrap samplings or bins; unavailable for jackknife data.
    """
    from ..plot_contracts import UncertaintyOptions

    if data.ndim != 0 or data.is_complex:
        raise ValueError("reference levels require a real, scalar observable")
    resolved = UncertaintyOptions(error_metric=error_metric, confidence_level=confidence_level,
                                  percentiles=percentiles).resolved()
    metric = resolved["error_metric"]
    central = float(data.full_sample_value)
    if metric == "error":
        error = float(data.error)
        bounds = (central - error, central + error)
    else:
        if data._analyzer.method == "jackknife":
            raise ValueError("confidence intervals require bootstrap samplings or bins; unavailable for jackknife data")
        if percentiles is not None:
            lower, upper = np.percentile(data._analyzer.resamples(data._as_numpy()), percentiles)
        else:
            lower, upper = data.confidence_interval(resolved["confidence_level"])
        bounds = (float(lower), float(upper))
    if not np.all(np.isfinite((central, *bounds))) or bounds[0] > bounds[1]:
        raise ValueError("reference levels require finite central values and ordered bounds")
    return central, bounds


def energy_level_hover_text(
    sampling: SigmondData, *, percentiles=None, confidence_level=None,
    ni_pair: str | None = None, irrep=None, psq=None, energy_type=None,
    ref_particle=None, include_level: bool = True, interval_bounds=None,
) -> str:
    """Shared energy tooltip: full value with Gaussian error and percentile bounds.

    The PDG value always uses the sampling's statistical error, independently
    of the plotted interval. NI callers may supply a compact pair string and
    override metadata on a sampling derived through kinematic arithmetic.
    """
    obs = sampling.observable_info
    irrep = irrep if irrep is not None else getattr(obs, "irrep", None)
    psq = psq if psq is not None else getattr(obs, "psq", None)
    kind = energy_type or getattr(obs, "energy_type", "ecm")
    ref = ref_particle or getattr(obs, "ref_particle", None)
    label = {"ecm": "Ecm", "elab": "Elab"}.get(kind, kind)
    if ref:
        label += f"/M_{ref}"
    rows = [f"NI: {ni_pair}"] if ni_pair else []
    context = [str(irrep)] if irrep is not None else []
    if psq is not None:
        context.append(f"d²={psq}")
    level = getattr(obs, "level_index", None)
    if include_level and level is not None:
        context.append(f"level {level}")
    if context:
        rows.append(", ".join(context))
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


def latex_label(text: str | None) -> str | None:
    """Wrap a bare LaTeX fragment in ``$...$``."""
    return slp.ensure_math(text)


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
    if data.is_complex:
        raise ValueError(
            "cannot plot complex data; project explicitly first, e.g. .to_real() "
            "or CorrelatorMatrix.diagonal(component='real')"
        )
    raw = data._as_numpy()
    analyzer = data._analyzer

    central_metric = rc["plot.central_metric"]
    error_metric = rc["plot.error_metric"]
    try:
        if central_metric == "full_sample_value":
            central = analyzer.full(raw)
        elif central_metric == "mean":
            central = analyzer.mean(raw)
        elif central_metric == "median":
            central = np.median(analyzer.resamples(raw), axis=-1)
        elif central_metric == "bias_corrected_mean":
            central = analyzer.full(raw) - analyzer.bias(raw)
        else:  # pragma: no cover - rc validates the key
            raise ValueError(f"unknown central metric {central_metric!r}")
    except ValueError as exc:
        raise ValueError(f"rc['plot.central_metric'] = {central_metric!r}: {exc}") from None

    if error_metric == "error":
        return np.asarray(central), np.asarray(analyzer.error(raw))

    level = rc["plot.confidence_level"] if confidence_level is None else confidence_level
    try:
        lower, upper = analyzer.confidence_interval(raw, level)
    except ValueError as exc:
        raise ValueError(f"rc['plot.error_metric'] = {error_metric!r}: {exc}") from None
    offsets = np.stack([np.maximum(central - lower, 0.0), np.maximum(upper - central, 0.0)])
    return np.asarray(central), offsets


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


def summary_grid(n: int) -> tuple[int, int]:
    """``(nrows, ncols)`` for ``n`` summary panels: one row up to 3, then near-square."""
    if n <= 3:
        return (1, n)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)
