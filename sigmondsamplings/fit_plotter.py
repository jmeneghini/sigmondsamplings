"""Plot fitted sampling models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from .colors import COLORS
from .fit import (
    Chi2Scan,
    FitBackend,
    ProgressKind,
    evaluate_chi2_function_scan,
    evaluate_chi2_scan,
)
from .obervable_collection import ObservableCollection
from .rcparams import rc
from .sampling import SigmondSampling
from .stats import SamplingStats

if TYPE_CHECKING:
    from .fit import SamplingFitResult
    from .model_func import SigmondModelFunc


_ANNOTATION_ANCHORS: dict[str, tuple[float, float, str, str]] = {
    "upper right": (0.98, 0.98, "right", "top"),
    "upper left": (0.02, 0.98, "left", "top"),
    "lower right": (0.98, 0.02, "right", "bottom"),
    "lower left": (0.02, 0.02, "left", "bottom"),
    "upper center": (0.5, 0.98, "center", "top"),
    "lower center": (0.5, 0.02, "center", "bottom"),
    "center right": (0.98, 0.5, "right", "center"),
    "center left": (0.02, 0.5, "left", "center"),
    "center": (0.5, 0.5, "center", "center"),
}


@dataclass
class FitPlotStyle:
    """Per-element styling for :func:`plot_fit_result`.

    Any color field left as ``None`` resolves to the rc-derived primary fit
    color. ``confidence_level=None`` resolves to ``rc["plot.confidence_level"]``.
    Label fields with ``None`` either auto-format from the confidence level
    (``band_label``, ``interval_label``) or are suppressed entirely.

    Every element also has a ``*_kwargs`` dict that is merged into the
    matplotlib call last, so it overrides any typed field on conflict.
    """

    confidence_level: float | None = None
    fit_color: str | None = None

    # Exact-x data points (errorbar)
    show_data: bool = True
    data_color: str = "black"
    data_marker: str = "o"
    data_markersize: float = 4.0
    data_capsize: float = 3.0
    data_label: str | None = "model at data"
    data_kwargs: dict[str, Any] = field(default_factory=dict)

    # Noisy-x confidence ellipses
    show_confidence_ellipses: bool = True
    ellipse_facecolor: str | None = None  # None -> fit_color
    ellipse_edgecolor: str = "black"
    ellipse_alpha: float = 0.35
    ellipse_linewidth: float = 1.0
    ellipse_center_color: str | None = None  # None -> data_color
    ellipse_center_markersize: float = 3.0
    interval_label: str | None = None  # None -> auto from confidence_level
    ellipse_kwargs: dict[str, Any] = field(default_factory=dict)

    # Bootstrap cloud
    show_bootstrap_cloud: bool = True
    cloud_color: str = "0.35"
    cloud_alpha: float = 0.12
    cloud_size: float = 5.0
    cloud_kwargs: dict[str, Any] = field(default_factory=dict)

    # Confidence band
    show_confidence_band: bool = True
    band_color: str | None = None  # None -> fit_color
    band_alpha: float = 0.22
    band_label: str | None = None  # None -> auto from confidence_level
    band_kwargs: dict[str, Any] = field(default_factory=dict)

    # Resample-mean fit line
    show_mean: bool = True
    mean_color: str | None = None  # None -> fit_color
    mean_linewidth: float = 2.0
    mean_label: str | None = "resample mean"
    mean_kwargs: dict[str, Any] = field(default_factory=dict)

    # Full-sample fit line
    show_full: bool = True
    full_color: str = "black"
    full_linewidth: float = 1.5
    full_linestyle: str = "--"
    full_label: str | None = "full sample"
    full_kwargs: dict[str, Any] = field(default_factory=dict)

    # Axes
    grid: bool = True
    grid_alpha: float = 0.25
    show_legend: bool = True
    legend_frameon: bool = False
    xlabel: str | None = None  # None -> auto from model
    ylabel: str | None = None  # None -> auto from model

    # Fit-summary annotation box (params + metrics)
    display_params: bool = False
    metrics: Iterable[str] | None = None  # e.g. ["chi_squared", "chi2_per_dof", "aic"]
    annotation_loc: str = "upper right"
    annotation_fontsize: float | None = None
    annotation_bbox: dict[str, Any] | None = None


@dataclass
class _ResolvedStyle:
    """A FitPlotStyle with rc/None-derived fallbacks already resolved."""

    confidence_level: float
    fit_color: str
    ellipse_facecolor: str
    ellipse_center_color: str
    band_color: str
    mean_color: str
    interval_label: str | None
    band_label: str | None
    style: FitPlotStyle


def _resolve_style(style: FitPlotStyle | None) -> _ResolvedStyle:
    style = style or FitPlotStyle()

    cl = style.confidence_level
    if cl is None:
        cl = rc["plot.confidence_level"]

    primary = style.fit_color
    if primary is None:
        palette = rc.get("plot.colors") or COLORS
        primary = palette[0]

    return _ResolvedStyle(
        confidence_level=cl,
        fit_color=primary,
        ellipse_facecolor=style.ellipse_facecolor or primary,
        ellipse_center_color=style.ellipse_center_color or style.data_color,
        band_color=style.band_color or primary,
        mean_color=style.mean_color or primary,
        interval_label=(
            style.interval_label
            if style.interval_label is not None
            else f"{cl * 100:.0f}% interval"
        ),
        band_label=(style.band_label if style.band_label is not None else f"{cl * 100:.0f}% band"),
        style=style,
    )


def plot_fit_result(
    x_values: Iterable[float] | Iterable[SigmondSampling] | np.ndarray,
    model_func: SigmondModelFunc | Callable,
    fit_result: SamplingFitResult | None = None,
    *,
    x_fit_values: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    x_fit_range: tuple[float, float] | None = None,
    n_fit_points: int = 100,
    show_fit: bool = True,
    figsize: tuple[float, float] = (10, 6),
    model_latex_str: str | None = None,
    independent_var_latex: str | None = None,
    data_samplings: Iterable[SigmondSampling] | None = None,
    style: FitPlotStyle | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a fitted model against exact or sampled x-values.

    Exact x-values are plotted with vertical model intervals.  Sampled x-values
    are plotted with confidence ellipses and optional bootstrap clouds.

    All per-element styling (colors, labels, alphas, linewidths, raw matplotlib
    kwargs) lives on :class:`FitPlotStyle`. Defaults for confidence level and
    primary color are read from ``rc["plot.confidence_level"]`` and
    ``rc["plot.colors"][0]`` (falling back to :data:`sigmondsamplings.COLORS`).
    """
    resolved = _resolve_style(style)

    model = _resolve_model(
        model_func,
        fit_result,
        model_latex_str=model_latex_str,
        independent_var_latex=independent_var_latex,
    )
    x_samplings = _as_x_samplings(x_values, model, independent_var_latex)
    if len(x_samplings) == 0:
        raise ValueError("x_values must contain at least one value")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_has_noise = [_sampling_has_variation(x) for x in x_samplings]
    y_samplings = [
        _evaluate_model_at_x(model, x, noisy) for x, noisy in zip(x_samplings, x_has_noise)
    ]

    if resolved.style.show_data:
        if any(x_has_noise):
            _plot_noisy_x_points(ax, x_samplings, y_samplings, x_has_noise, resolved)
        else:
            _plot_exact_x_points(ax, x_samplings, y_samplings, resolved)

    if resolved.style.show_bootstrap_cloud:
        _plot_bootstrap_cloud(ax, x_samplings, y_samplings, x_has_noise, resolved)

    if show_fit:
        x_fit = _x_fit_grid(
            x_samplings,
            x_has_noise,
            x_fit_values,
            x_fit_range,
            n_fit_points,
        )
        _plot_model_band(ax, model, x_fit, resolved.style.show_confidence_band, resolved)

    _apply_fit_labels(ax, model, x_samplings, resolved)
    _render_fit_annotation(ax, model, x_samplings, data_samplings, resolved)
    fig.tight_layout()
    return fig, ax


def _resolve_model(
    model_func: SigmondModelFunc | Callable,
    fit_result: SamplingFitResult | None,
    *,
    model_latex_str: str | None,
    independent_var_latex: str | None,
):
    from .model_func import SigmondModelFunc

    if isinstance(model_func, SigmondModelFunc):
        return model_func
    if fit_result is None:
        raise TypeError("fit_result is required when model_func is a raw callable")
    return fit_result.model_func(
        model_func,
        latex_str=model_latex_str,
        independent_var_latex=independent_var_latex,
    )


def _as_x_samplings(
    x_values: Iterable[float] | Iterable[SigmondSampling] | np.ndarray,
    model,
    independent_var_latex: str | None,
) -> ObservableCollection:
    if isinstance(x_values, SigmondSampling):
        return ObservableCollection([x_values])

    values = _as_flat_values(x_values)
    if not values:
        return ObservableCollection([])
    if all(isinstance(x, SigmondSampling) for x in values):
        return ObservableCollection(values)
    if any(isinstance(x, SigmondSampling) for x in values):
        raise TypeError("x_values must be all numeric values or all SigmondSampling objects")

    from .info import ObservableInfo

    x_latex = independent_var_latex or model.independent_var_latex or "x"
    return ObservableCollection(
        [
            SigmondSampling.from_single_value(
                float(x),
                ObservableInfo(
                    name=x_latex.strip("$"),
                    index=i,
                    op_type="n",
                    re_im="re",
                    latex_str=x_latex,
                ),
                model.sampling_info,
            )
            for i, x in enumerate(values)
        ]
    )


def _as_flat_values(values) -> list:
    if isinstance(values, np.ndarray):
        return list(np.asarray(values, dtype=object).reshape(-1))
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]


def _as_numeric_array(values, name: str) -> np.ndarray:
    if isinstance(values, SigmondSampling):
        raise TypeError(f"{name} must be numeric, not SigmondSampling objects")
    values = _as_flat_values(values)
    if any(isinstance(x, SigmondSampling) for x in values):
        raise TypeError(f"{name} must be numeric, not SigmondSampling objects")
    return np.asarray(values, dtype=float)


def _sampling_has_variation(sampling: SigmondSampling) -> bool:
    values = np.asarray(sampling.resampled_values)
    return values.size > 0 and not np.allclose(values, values[0])


def _sampling_yerr(sampling: SigmondSampling, confidence_level: float):
    center = sampling.full_sample_value
    if sampling.sampling_info.method == "bootstrap":
        lower, upper = sampling.confidence_interval(confidence_level)
        return np.array([[max(0.0, center - lower)], [max(0.0, upper - center)]])
    return sampling.error


def _evaluate_model_at_x(model, x_sampling: SigmondSampling, x_has_noise: bool):
    x_value = x_sampling if x_has_noise else x_sampling.full_sample_value
    return model(x_value)


def _data_errorbar_kwargs(resolved: _ResolvedStyle, *, include_label: bool) -> dict[str, Any]:
    s = resolved.style
    kwargs = {
        "fmt": s.data_marker,
        "color": s.data_color,
        "markersize": s.data_markersize,
        "capsize": s.data_capsize,
        "zorder": 10,
    }
    if include_label:
        kwargs["label"] = s.data_label
    kwargs.update(s.data_kwargs)
    if not include_label:
        kwargs.pop("label", None)
    return kwargs


def _plot_exact_x_points(
    ax: plt.Axes,
    x_samplings: list[SigmondSampling],
    y_samplings: list[SigmondSampling],
    resolved: _ResolvedStyle,
) -> None:
    x_points = np.array([x.full_sample_value for x in x_samplings])
    y_points = np.array([y.full_sample_value for y in y_samplings])
    y_errors = np.hstack([_sampling_yerr(y, resolved.confidence_level) for y in y_samplings])
    ax.errorbar(
        x_points,
        y_points,
        yerr=y_errors,
        **_data_errorbar_kwargs(resolved, include_label=True),
    )


def _plot_noisy_x_points(
    ax: plt.Axes,
    x_samplings: list[SigmondSampling],
    y_samplings: list[SigmondSampling],
    x_has_noise: list[bool],
    resolved: _ResolvedStyle,
) -> None:
    s = resolved.style
    ellipse_kwargs = {
        "alpha": s.ellipse_alpha,
        "fill": True,
        "edgecolor": s.ellipse_edgecolor,
        "facecolor": resolved.ellipse_facecolor,
        "linewidth": s.ellipse_linewidth,
        "zorder": 8,
    }
    ellipse_kwargs.update(s.ellipse_kwargs)

    exact_kwargs = _data_errorbar_kwargs(resolved, include_label=False)

    for x, y, noisy in zip(x_samplings, y_samplings, x_has_noise):
        if noisy:
            if s.show_confidence_ellipses:
                center_x, center_y, width, height, angle = SamplingStats.confidence_ellipse_params(
                    x, y, resolved.confidence_level
                )
                ax.add_patch(
                    Ellipse(
                        (center_x, center_y),
                        width,
                        height,
                        angle=angle,
                        **ellipse_kwargs,
                    )
                )
            else:
                center_x = x.full_sample_value
                center_y = y.full_sample_value
            ax.plot(
                center_x,
                center_y,
                s.data_marker,
                color=resolved.ellipse_center_color,
                markersize=s.ellipse_center_markersize,
                zorder=10,
            )
        else:
            ax.errorbar(
                x.full_sample_value,
                y.full_sample_value,
                yerr=_sampling_yerr(y, resolved.confidence_level),
                **exact_kwargs,
            )

    if s.show_confidence_ellipses and resolved.interval_label:
        ax.plot(
            [],
            [],
            s.data_marker,
            color=resolved.ellipse_center_color,
            label=resolved.interval_label,
        )


def _plot_bootstrap_cloud(
    ax: plt.Axes,
    x_samplings: list[SigmondSampling],
    y_samplings: list[SigmondSampling],
    x_has_noise: list[bool],
    resolved: _ResolvedStyle,
) -> None:
    s = resolved.style
    cloud_kwargs = {
        "alpha": s.cloud_alpha,
        "s": s.cloud_size,
        "color": s.cloud_color,
        "zorder": 2,
    }
    cloud_kwargs.update(s.cloud_kwargs)
    for x, y, noisy in zip(x_samplings, y_samplings, x_has_noise):
        y_samples = y.resampled_values
        if noisy:
            x_samples = x.resampled_values
        else:
            x_samples = np.full_like(y_samples, x.full_sample_value, dtype=float)
        ax.scatter(x_samples, y_samples, **cloud_kwargs)


def _x_fit_grid(
    x_samplings: list[SigmondSampling],
    x_has_noise: list[bool],
    x_fit_values: np.ndarray | None,
    x_fit_range: tuple[float, float] | None,
    n_fit_points: int,
) -> np.ndarray:
    if x_fit_values is not None:
        return _as_numeric_array(x_fit_values, "x_fit_values")

    if x_fit_range is None:
        x_centers = [
            x.mean if noisy else x.full_sample_value for x, noisy in zip(x_samplings, x_has_noise)
        ]
        x_min, x_max = min(x_centers), max(x_centers)
        span = x_max - x_min
        if span == 0:
            span = abs(x_min) if x_min != 0 else 1.0
        x_fit_range = (x_min - 0.1 * span, x_max + 0.1 * span)

    return np.linspace(x_fit_range[0], x_fit_range[1], n_fit_points)


def _plot_model_band(
    ax: plt.Axes,
    model,
    x_fit: np.ndarray,
    show_confidence_band: bool,
    resolved: _ResolvedStyle,
) -> None:
    s = resolved.style
    means, lowers, uppers, fulls = model.evaluate_summary(x_fit, resolved.confidence_level)

    band_kwargs = {
        "alpha": s.band_alpha,
        "color": resolved.band_color,
        "linewidth": 0,
        "label": resolved.band_label,
        "zorder": 3,
    }
    band_kwargs.update(s.band_kwargs)

    mean_kwargs = {
        "color": resolved.mean_color,
        "linewidth": s.mean_linewidth,
        "label": s.mean_label,
        "zorder": 6,
    }
    mean_kwargs.update(s.mean_kwargs)

    full_kwargs = {
        "color": s.full_color,
        "linewidth": s.full_linewidth,
        "linestyle": s.full_linestyle,
        "label": s.full_label,
        "zorder": 7,
    }
    full_kwargs.update(s.full_kwargs)

    if show_confidence_band:
        ax.fill_between(x_fit, lowers, uppers, **band_kwargs)
    if s.show_mean:
        ax.plot(x_fit, means, **mean_kwargs)
    if s.show_full:
        ax.plot(x_fit, fulls, **full_kwargs)


def _apply_fit_labels(
    ax: plt.Axes,
    model,
    x_samplings: list[SigmondSampling],
    resolved: _ResolvedStyle,
) -> None:
    s = resolved.style

    if s.ylabel is not None:
        ax.set_ylabel(s.ylabel)
    else:
        y_label = model.get_latex_str_with_var()
        if y_label:
            ax.set_ylabel(_latex_label(y_label))
        elif getattr(model.func, "__name__", None):
            ax.set_ylabel(model.func.__name__)

    if s.xlabel is not None:
        ax.set_xlabel(s.xlabel)
    else:
        x_label = model.independent_var_latex or _x_latex_from_samplings(x_samplings)
        if x_label:
            ax.set_xlabel(_latex_label(x_label))

    if s.grid:
        ax.grid(True, alpha=s.grid_alpha)
    if s.show_legend:
        ax.legend(frameon=s.legend_frameon)


def _render_fit_annotation(
    ax: plt.Axes,
    model,
    x_samplings: ObservableCollection,
    data_samplings: Iterable[SigmondSampling] | None,
    resolved: _ResolvedStyle,
) -> None:
    s = resolved.style
    if not s.display_params and not s.metrics:
        return

    entries: list[str] = []
    if s.display_params:
        entries.extend(model.format_params())
    if s.metrics:
        if data_samplings is None:
            raise ValueError(
                "FitPlotStyle.metrics requested but data_samplings=None; pass "
                "the fitted data samplings to plot_fit_result()."
            )
        data_samplings = tuple(data_samplings)
        if len(data_samplings) != len(x_samplings):
            raise ValueError(
                "data_samplings length must match x_values length when metrics are requested"
            )
        data_stats = SamplingStats(data_samplings)
        x_arr = np.array([x.full_sample_value for x in x_samplings])
        entries.extend(
            model.format_metrics(
                data_stats,
                x_arr,
                s.metrics,
            )
        )

    if not entries:
        return

    anchor = _ANNOTATION_ANCHORS.get(s.annotation_loc, _ANNOTATION_ANCHORS["upper right"])
    x, y, ha, va = anchor
    bbox = s.annotation_bbox
    if bbox is None:
        bbox = {
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.85,
        }
    ax.text(
        x,
        y,
        "\n".join(entries),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=s.annotation_fontsize,
        bbox=bbox,
        zorder=20,
    )


def _x_latex_from_samplings(x_samplings: list[SigmondSampling]) -> str | None:
    if not x_samplings:
        return None
    return x_samplings[0].observable_info.latex_str


def _latex_label(value: str | None) -> str | None:
    if not value:
        return None
    return f"${value.strip('$')}$"


# ----------------------------------------------------------------------
# Chi-squared landscape plotting
# ----------------------------------------------------------------------


@dataclass
class Chi2PlotStyle:
    """Styling for :func:`plot_chi2_1d` and :func:`plot_chi2_2d`.

    Confidence-level deltas are passed as a list of ``(delta_chi2, label, linestyle)``
    triples; the default matches 1σ/2σ/3σ for a single-parameter scan.
    """

    # Curve / surface
    line_color: str | None = None  # None -> rc primary color
    line_width: float = 2.0
    cmap: str = "viridis"

    # Minimum marker
    show_min: bool = True
    min_color: str = "crimson"
    min_marker: str = "*"
    min_markersize: float = 14.0
    min_marker_edgecolor: str = "white"
    min_marker_edgewidth: float = 1.0
    annotate_min: bool = True
    annotate_min_fontsize: float = 9.0

    # Confidence levels (delta_chi2, label, linestyle)
    confidence_deltas: tuple[tuple[float, str, str], ...] = (
        (1.0, r"1$\sigma$", "-"),
        (4.0, r"2$\sigma$", "--"),
        (9.0, r"3$\sigma$", ":"),
    )
    # 1D: translucent horizontal bands between the curve minimum and each level
    confidence_color_1d: str = "tab:red"
    confidence_band_alphas_1d: tuple[float, ...] = (0.22, 0.13, 0.07)
    confidence_line_alpha_1d: float = 0.8
    # 2D: white iso-chi2 contours
    confidence_color_2d: str = "white"
    confidence_alpha_2d: float = 0.9
    confidence_linewidth_2d: float = 1.2
    contour_label_levels: bool = True
    contour_label_fontsize: float = 8.0

    # 2D heatmap
    use_pcolormesh: bool = True
    pcolormesh_shading: str = "auto"  # "auto" / "gouraud" / "nearest"
    n_contour_levels: int = 20  # used when use_pcolormesh=False
    show_crosshair_at_min: bool = True
    crosshair_color: str = "white"
    crosshair_alpha: float = 0.6
    crosshair_linewidth: float = 0.8
    crosshair_linestyle: str = "--"

    # Axes
    grid: bool = True
    grid_alpha: float = 0.25
    show_legend: bool = True
    legend_frameon: bool = False
    title: str | None = None  # None -> auto
    xlabel: str | None = None  # None -> auto from param_names
    ylabel: str | None = None  # None -> "χ²" (1D) or auto (2D)


def _resolve_chi2_color(style: Chi2PlotStyle) -> str:
    if style.line_color is not None:
        return style.line_color
    palette = rc.get("plot.colors") or COLORS
    return palette[0]


def _resolve_figsize(figsize: tuple[float, float] | None) -> tuple[float, float]:
    return figsize or (10, 6)


def _get_or_create_axes(
    ax: plt.Axes | None, figsize: tuple[float, float] | None
) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=_resolve_figsize(figsize))
    return fig, ax


def _param_label(param_names: list[str] | None, idx: int) -> str:
    if param_names and idx < len(param_names):
        return param_names[idx]
    return f"Parameter {idx}"


def _min_marker_kwargs(style: Chi2PlotStyle) -> dict[str, Any]:
    return {
        "marker": style.min_marker,
        "color": style.min_color,
        "markersize": style.min_markersize,
        "markeredgecolor": style.min_marker_edgecolor,
        "markeredgewidth": style.min_marker_edgewidth,
        "linestyle": "none",
    }


def _visible_sigma_levels(
    chi2_min: float, chi2_max: float, style: Chi2PlotStyle
) -> list[tuple[int, float, str, str, float]]:
    """Return ``(rank, delta, label, linestyle, level)`` for sigma deltas
    whose level falls inside ``[chi2_min, chi2_max]``."""
    out: list[tuple[int, float, str, str, float]] = []
    for rank, (delta, label, linestyle) in enumerate(style.confidence_deltas):
        level = chi2_min + delta
        if level <= chi2_max:
            out.append((rank, delta, label, linestyle, level))
    return out


def _apply_chi2_axes(
    ax: plt.Axes,
    style: Chi2PlotStyle,
    *,
    default_xlabel: str,
    default_ylabel: str,
    default_title: str,
    apply_grid: bool = True,
) -> None:
    ax.set_xlabel(style.xlabel if style.xlabel is not None else default_xlabel)
    ax.set_ylabel(style.ylabel if style.ylabel is not None else default_ylabel)
    ax.set_title(style.title if style.title is not None else default_title)
    if apply_grid and style.grid:
        ax.grid(True, alpha=style.grid_alpha)
    if style.show_legend:
        ax.legend(frameon=style.legend_frameon)


def _draw_confidence_levels_1d(
    ax: plt.Axes,
    scan_values: np.ndarray,
    chi2_values: np.ndarray,
    style: Chi2PlotStyle,
) -> None:
    chi2_min = float(np.min(chi2_values))
    chi2_max = float(np.max(chi2_values))
    x_min, x_max = scan_values[0], scan_values[-1]
    band_alphas = style.confidence_band_alphas_1d

    for rank, delta, label, linestyle, level in _visible_sigma_levels(chi2_min, chi2_max, style):
        alpha_band = band_alphas[rank] if rank < len(band_alphas) else band_alphas[-1]
        ax.fill_between(
            [x_min, x_max],
            chi2_min,
            level,
            color=style.confidence_color_1d,
            alpha=alpha_band,
            zorder=1,
        )
        ax.plot(
            [x_min, x_max],
            [level, level],
            color=style.confidence_color_1d,
            linestyle=linestyle,
            linewidth=1.0,
            alpha=style.confidence_line_alpha_1d,
            label=f"{label} (Δχ²={delta:g})",
            zorder=2,
        )


def _line_collection_kwargs(line_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(line_kwargs)
    if "linewidth" in kwargs:
        linewidth = kwargs.pop("linewidth")
        kwargs.setdefault("linewidths", linewidth)
    if "linestyle" in kwargs:
        linestyle = kwargs.pop("linestyle")
        kwargs.setdefault("linestyles", linestyle)
    allowed = {"linewidths", "linestyles", "label", "zorder", "alpha"}
    unsupported = sorted(set(kwargs) - allowed)
    if unsupported:
        raise TypeError(
            "color_function renders with LineCollection, which does not support "
            f"plot_kwargs: {', '.join(unsupported)}"
        )
    return kwargs


def _plot_chi2_1d_scan(
    scan_values: np.ndarray,
    scan: Chi2Scan,
    param_index: int,
    param_names: list[str] | None,
    title_suffix: str,
    ax: plt.Axes | None,
    figsize: tuple[float, float] | None,
    style: Chi2PlotStyle,
    color_function: Callable[[np.ndarray, float], Any] | None,
    plot_kwargs: dict[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_or_create_axes(ax, figsize)
    chi2_values = scan.chi2_values

    _draw_confidence_levels_1d(ax, scan_values, chi2_values, style)

    line_color = _resolve_chi2_color(style)
    line_kwargs = {
        "linewidth": style.line_width,
        "label": f"$\\chi^2$ {title_suffix}".strip(),
        "zorder": 5,
    }
    line_kwargs.update(plot_kwargs)

    if color_function is None:
        line_kwargs.setdefault("color", line_color)
        ax.plot(scan_values, chi2_values, **line_kwargs)
    else:
        colors = [color_function(p, c) for p, c in zip(scan.param_stack, chi2_values)]
        points = np.column_stack([scan_values, chi2_values])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        line_kwargs.pop("color", None)
        lc = LineCollection(
            segments,
            colors=colors[:-1],
            **_line_collection_kwargs(line_kwargs),
        )
        ax.add_collection(lc)
        ax.set_xlim(scan_values.min(), scan_values.max())
        ax.set_ylim(chi2_values.min(), chi2_values.max())

    if style.show_min:
        min_idx = int(np.argmin(chi2_values))
        min_x = float(scan_values[min_idx])
        min_chi2 = float(chi2_values[min_idx])
        ax.axvline(
            min_x,
            color=style.min_color,
            alpha=0.4,
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )
        ax.plot(
            min_x,
            min_chi2,
            **_min_marker_kwargs(style),
            label=rf"Min: $\chi^2$={min_chi2:.3f}",
            zorder=6,
        )
        if style.annotate_min:
            ax.annotate(
                rf"$\hat{{\theta}}={min_x:.4g}$",
                xy=(min_x, min_chi2),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=style.annotate_min_fontsize,
                color=style.min_color,
                zorder=7,
            )

    param_name = _param_label(param_names, param_index)
    _apply_chi2_axes(
        ax,
        style,
        default_xlabel=param_name,
        default_ylabel=r"$\chi^2$",
        default_title=rf"$\chi^2$ landscape: {param_name} {title_suffix}".strip(),
    )

    return fig, ax


def plot_chi2_1d(
    stats: SamplingStats,
    prediction_func: Callable[[np.ndarray], np.ndarray],
    param_index: int,
    param_range: tuple[float, float],
    *,
    n_points: int = 100,
    fixed_params: dict[int, float] | None = None,
    param_names: list[str] | None = None,
    n_total_params: int | None = None,
    color_function: Callable[[np.ndarray, float], Any] | None = None,
    ax: plt.Axes | None = None,
    resamp_idx: int = 0,
    use_correlation: bool = True,
    backend: FitBackend = "serial",
    num_workers: int | str | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    figsize: tuple[float, float] | None = None,
    style: Chi2PlotStyle | None = None,
    **plot_kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot χ² as a function of a single parameter.

    Parameters
    ----------
    stats:
        :class:`SamplingStats` providing the chi-squared evaluation.
    prediction_func:
        ``f(params) -> theory_values``. May accept a batch of shape
        ``(n_points, n_total_params)`` for a speed-up; falls back to per-row
        calls if not.
    param_index:
        Index of the parameter to vary.
    param_range:
        ``(min, max)`` for the scan.
    n_total_params:
        Total parameter count for ``prediction_func``. Defaults to
        ``max(param_index, fixed_params keys) + 1`` — set this explicitly when
        your model has extra parameters that aren't being varied or fixed.
    color_function:
        Optional ``f(params, chi2) -> color``; segments are colored with a
        :class:`LineCollection`.
    """
    style = style or Chi2PlotStyle()
    fixed_params = dict(fixed_params or {})

    scan_values = np.linspace(param_range[0], param_range[1], n_points)
    scan = evaluate_chi2_scan(
        stats,
        prediction_func,
        [param_index],
        scan_values.reshape(-1, 1),
        fixed_params=fixed_params,
        n_total_params=n_total_params,
        use_correlation=use_correlation,
        resamp_idx=resamp_idx,
        backend=backend,
        num_workers=num_workers,
        progress=progress,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    return _plot_chi2_1d_scan(
        scan_values,
        scan,
        param_index,
        param_names,
        f"(resample {resamp_idx})",
        ax,
        figsize,
        style,
        color_function,
        plot_kwargs,
    )


def plot_chi2_function_1d(
    chi2_func: Callable[[np.ndarray], float],
    param_index: int,
    param_range: tuple[float, float],
    *,
    n_points: int = 100,
    fixed_params: dict[int, float] | None = None,
    param_names: list[str] | None = None,
    n_total_params: int | None = None,
    color_function: Callable[[np.ndarray, float], Any] | None = None,
    ax: plt.Axes | None = None,
    backend: FitBackend = "serial",
    num_workers: int | str | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    figsize: tuple[float, float] | None = None,
    style: Chi2PlotStyle | None = None,
    **plot_kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a direct ``chi2_func(params)`` scan over one parameter."""
    style = style or Chi2PlotStyle()
    scan_values = np.linspace(param_range[0], param_range[1], n_points)
    scan = evaluate_chi2_function_scan(
        chi2_func,
        [param_index],
        scan_values.reshape(-1, 1),
        fixed_params=fixed_params,
        n_total_params=n_total_params,
        backend=backend,
        num_workers=num_workers,
        progress=progress,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    return _plot_chi2_1d_scan(
        scan_values,
        scan,
        param_index,
        param_names,
        "",
        ax,
        figsize,
        style,
        color_function,
        plot_kwargs,
    )


def _chi2_2d_scan_grid(
    param_ranges: tuple[tuple[float, float], tuple[float, float]],
    n_points: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1_values = np.linspace(param_ranges[0][0], param_ranges[0][1], n_points[0])
    p2_values = np.linspace(param_ranges[1][0], param_ranges[1][1], n_points[1])
    p1_grid, p2_grid = np.meshgrid(p1_values, p2_values, indexing="ij")
    flat_varying = np.column_stack([p1_grid.ravel(), p2_grid.ravel()])
    return p1_grid, p2_grid, flat_varying


def _plot_chi2_2d_scan(
    scan: Chi2Scan,
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    param_indices: tuple[int, int],
    param_names: list[str] | None,
    title_suffix: str,
    use_plotly: bool,
    ax: plt.Axes | None,
    figsize: tuple[float, float] | None,
    style: Chi2PlotStyle,
    plot_kwargs: dict[str, Any],
):
    chi2_grid = scan.chi2_values.reshape(p1_grid.shape)
    if use_plotly:
        return _plot_chi2_2d_plotly(
            p1_grid,
            p2_grid,
            chi2_grid,
            param_indices,
            param_names,
            title_suffix,
            style,
            **plot_kwargs,
        )
    return _plot_chi2_2d_matplotlib(
        p1_grid,
        p2_grid,
        chi2_grid,
        param_indices,
        param_names,
        title_suffix,
        ax,
        figsize,
        style,
        **plot_kwargs,
    )


def plot_chi2_2d(
    stats: SamplingStats,
    prediction_func: Callable[[np.ndarray], np.ndarray],
    param_indices: tuple[int, int],
    param_ranges: tuple[tuple[float, float], tuple[float, float]],
    *,
    n_points: tuple[int, int] = (50, 50),
    fixed_params: dict[int, float] | None = None,
    param_names: list[str] | None = None,
    n_total_params: int | None = None,
    use_plotly: bool = False,
    resamp_idx: int = 0,
    use_correlation: bool = True,
    backend: FitBackend = "serial",
    num_workers: int | str | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    style: Chi2PlotStyle | None = None,
    **plot_kwargs: Any,
):
    """Plot χ² as a 2D surface (plotly) or contour map (matplotlib).

    Returns ``(fig, ax)`` for matplotlib or a ``plotly.graph_objects.Figure``.
    """
    style = style or Chi2PlotStyle()
    fixed_params = dict(fixed_params or {})

    p1_grid, p2_grid, flat_varying = _chi2_2d_scan_grid(param_ranges, n_points)

    scan = evaluate_chi2_scan(
        stats,
        prediction_func,
        list(param_indices),
        flat_varying,
        fixed_params=fixed_params,
        n_total_params=n_total_params,
        use_correlation=use_correlation,
        resamp_idx=resamp_idx,
        backend=backend,
        num_workers=num_workers,
        progress=progress,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    return _plot_chi2_2d_scan(
        scan,
        p1_grid,
        p2_grid,
        param_indices,
        param_names,
        f"(resample {resamp_idx})",
        use_plotly,
        ax,
        figsize,
        style,
        plot_kwargs,
    )


def plot_chi2_function_2d(
    chi2_func: Callable[[np.ndarray], float],
    param_indices: tuple[int, int],
    param_ranges: tuple[tuple[float, float], tuple[float, float]],
    *,
    n_points: tuple[int, int] = (50, 50),
    fixed_params: dict[int, float] | None = None,
    param_names: list[str] | None = None,
    n_total_params: int | None = None,
    use_plotly: bool = False,
    backend: FitBackend = "serial",
    num_workers: int | str | None = "auto",
    progress: ProgressKind = False,
    num_blas_threads: int | None = None,
    num_openmp_threads: int | None = None,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    style: Chi2PlotStyle | None = None,
    **plot_kwargs: Any,
):
    """Plot a direct ``chi2_func(params)`` scan over two parameters."""
    style = style or Chi2PlotStyle()
    p1_grid, p2_grid, flat_varying = _chi2_2d_scan_grid(param_ranges, n_points)
    scan = evaluate_chi2_function_scan(
        chi2_func,
        list(param_indices),
        flat_varying,
        fixed_params=fixed_params,
        n_total_params=n_total_params,
        backend=backend,
        num_workers=num_workers,
        progress=progress,
        num_blas_threads=num_blas_threads,
        num_openmp_threads=num_openmp_threads,
        worker_initializer=worker_initializer,
        worker_initargs=worker_initargs,
    )
    return _plot_chi2_2d_scan(
        scan,
        p1_grid,
        p2_grid,
        param_indices,
        param_names,
        "",
        use_plotly,
        ax,
        figsize,
        style,
        plot_kwargs,
    )


def _plot_chi2_2d_plotly(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    chi2_grid: np.ndarray,
    param_indices: tuple[int, int],
    param_names: list[str] | None,
    title_suffix: str,
    style: Chi2PlotStyle,
    **kwargs: Any,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for 3D surface plots. Install with: pip install plotly"
        ) from exc

    p1_name = _param_label(param_names, param_indices[0])
    p2_name = _param_label(param_names, param_indices[1])

    surface_kwargs = {"colorscale": style.cmap if style.cmap != "viridis" else "Viridis"}
    surface_kwargs.update(kwargs)

    fig = go.Figure(data=[go.Surface(x=p1_grid, y=p2_grid, z=chi2_grid, **surface_kwargs)])
    title = style.title or f"χ² landscape {title_suffix}".strip()
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=style.xlabel or p1_name,
            yaxis_title=style.ylabel or p2_name,
            zaxis_title="χ²",
        ),
    )
    return fig


def _plot_chi2_2d_matplotlib(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    chi2_grid: np.ndarray,
    param_indices: tuple[int, int],
    param_names: list[str] | None,
    title_suffix: str,
    ax: plt.Axes | None,
    figsize: tuple[float, float] | None,
    style: Chi2PlotStyle,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_or_create_axes(ax, figsize)
    p1_name = _param_label(param_names, param_indices[0])
    p2_name = _param_label(param_names, param_indices[1])

    if style.use_pcolormesh:
        mesh_kwargs = {"shading": style.pcolormesh_shading, "cmap": style.cmap}
        mesh_kwargs.update(kwargs)
        mappable = ax.pcolormesh(p1_grid, p2_grid, chi2_grid, **mesh_kwargs)
    else:
        contour_kwargs = {"levels": style.n_contour_levels, "cmap": style.cmap}
        contour_kwargs.update(kwargs)
        mappable = ax.contourf(p1_grid, p2_grid, chi2_grid, **contour_kwargs)

    fig.colorbar(mappable, ax=ax, label=r"$\chi^2$")

    visible = _visible_sigma_levels(float(np.min(chi2_grid)), float(np.max(chi2_grid)), style)
    sigma_levels = [lvl for *_, lvl in visible]
    sigma_styles = [ls for _, _, _, ls, _ in visible]
    sigma_labels = {lvl: label for _, _, label, _, lvl in visible}

    if sigma_levels:
        cs = ax.contour(
            p1_grid,
            p2_grid,
            chi2_grid,
            levels=sigma_levels,
            colors=style.confidence_color_2d,
            linestyles=sigma_styles,
            linewidths=style.confidence_linewidth_2d,
            alpha=style.confidence_alpha_2d,
        )
        if style.contour_label_levels:
            ax.clabel(
                cs,
                inline=True,
                fontsize=style.contour_label_fontsize,
                fmt=lambda lvl: sigma_labels.get(lvl, f"{lvl:.2f}"),
            )

    if style.show_min:
        min_idx = np.unravel_index(int(np.argmin(chi2_grid)), chi2_grid.shape)
        min_p1 = float(p1_grid[min_idx])
        min_p2 = float(p2_grid[min_idx])
        if style.show_crosshair_at_min:
            ax.axvline(
                min_p1,
                color=style.crosshair_color,
                alpha=style.crosshair_alpha,
                linewidth=style.crosshair_linewidth,
                linestyle=style.crosshair_linestyle,
            )
            ax.axhline(
                min_p2,
                color=style.crosshair_color,
                alpha=style.crosshair_alpha,
                linewidth=style.crosshair_linewidth,
                linestyle=style.crosshair_linestyle,
            )
        ax.plot(
            min_p1,
            min_p2,
            **_min_marker_kwargs(style),
            label=rf"Min: $\chi^2={chi2_grid[min_idx]:.3f}$",
        )

    _apply_chi2_axes(
        ax,
        style,
        default_xlabel=p1_name,
        default_ylabel=p2_name,
        default_title=(rf"$\chi^2$ landscape: {p1_name} vs {p2_name} {title_suffix}").strip(),
        apply_grid=False,
    )

    return fig, ax


__all__ = [
    "plot_fit_result",
    "FitPlotStyle",
    "plot_chi2_1d",
    "plot_chi2_2d",
    "plot_chi2_function_1d",
    "plot_chi2_function_2d",
    "Chi2PlotStyle",
]
