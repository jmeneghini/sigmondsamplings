"""Statistics plots for SigmondSampling collections.

Ported onto :mod:`slat.plotting`, so every method here works on both the
matplotlib and plotly backends. The one exception is :meth:`SamplingPlotter.plot_corner`,
which wraps the third-party ``corner`` package and is matplotlib-only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

import slat.plotting as slp

from ..observable_collection import ObservableCollection
from ..sampling import SigmondSampling
from ..stats import SamplingStats
from ._util import resolve_axes

__all__ = ["SamplingPlotter"]

# Summary plot configuration
_VALID_PANELS = frozenset({"errorbar", "correlation", "histogram", "eff_sample_size"})
_DEFAULT_PANELS = ["histogram", "correlation"]
_PANEL_W = 6.5  # inches per column
_PANEL_H = 5.0  # inches per row


def _summary_grid(n: int) -> tuple[int, int]:
    """Compute (nrows, ncols) grid shape for n panels."""
    if n <= 3:
        return (1, n)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return (nrows, ncols)


def _summary_figsize(
    nrows: int, ncols: int, panels: list[str], num_observables: int
) -> tuple[float, float]:
    w = ncols * _PANEL_W
    h = nrows * _PANEL_H
    if "correlation" in panels:
        h += min(3.0, num_observables * 0.15)
    return (w, h)


def _as_observable_collection(
    samplings: SigmondSampling | Iterable[SigmondSampling],
) -> ObservableCollection:
    if isinstance(samplings, ObservableCollection):
        return samplings
    if isinstance(samplings, SigmondSampling):
        return ObservableCollection([samplings])
    return ObservableCollection(list(samplings))


def _sampling_latex_label(sampling: SigmondSampling) -> str:
    return f"${sampling.observable_info.latex_str.strip('$')}$"


class SamplingPlotter:
    """
    A comprehensive plotting class for SigmondSampling and SamplingStats objects.

    This class provides methods for visualizing statistical data, correlations,
    and fitting results from lattice QCD analysis workflows.

    Every method accepts ``axes`` (an existing :class:`slat.plotting.Axes`) and
    ``backend`` (a backend name for this call only). The legacy ``ax`` keyword
    still works and accepts a raw matplotlib Axes.
    """

    def __init__(
        self,
        stats: Iterable[SigmondSampling] | None = None,
        default_figsize: tuple[float, float] = (10, 6),
        default_style: dict[str, Any] = None,
    ):
        """
        Initialize the plotter.

        Args:
            stats: Optional SamplingStats object for default data source
            default_figsize: Default figure size for plots
            default_style: Default styling parameters
        """
        if stats is not None and not isinstance(stats, SamplingStats):
            self.stats = SamplingStats(stats)
        else:
            self.stats = stats
        self.default_figsize = default_figsize
        self.default_style = default_style or {}

    def plot_sampling_histogram(
        self,
        sampling: SigmondSampling | int | None = None,
        bins: int | str = "auto",
        axes: slp.Axes | None = None,
        confidence_level: float = 0.68,
        show_bias: bool = False,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
        **kwargs,
    ) -> slp.Axes:
        """
        Plot histogram of resampled values for a SigmondSampling object.

        Args:
            sampling: SigmondSampling object or index to plot (uses first from stats if None)
            bins: Number of bins or binning strategy, passed to ``np.histogram``
            axes: Axes to draw on (creates a figure if None)
            confidence_level: Confidence level for bootstrap CI (0.68 = 1σ)
            show_bias: Mark the bias-corrected mean for bootstrap samplings
            figsize: Figure size (uses default if None)
            backend: Backend for this call (uses the active one if None)
            ax: Deprecated alias for ``axes``
            **kwargs: Extra backend kwargs forwarded to the bar mark

        Returns:
            The Axes drawn on
        """
        # Use provided sampling or first from stats
        if sampling is None:
            if self.stats is None:
                raise ValueError("Must provide sampling or initialize with SamplingStats")
            sampling = self.stats[0]
        elif isinstance(sampling, int):
            if self.stats is None:
                raise ValueError("Must provide sampling or initialize with SamplingStats")
            sampling = self.stats[sampling]

        axes = resolve_axes(axes, ax, figsize=figsize or self.default_figsize, backend=backend)

        resampled = sampling.resampled_values

        # Add vertical lines for mean and error bounds
        mean_val = sampling.mean
        error_val = sampling.error

        # Use confidence interval for bootstrap, error bounds for jackknife
        if sampling.sampling_info.method == "bootstrap":
            lower, upper = sampling.confidence_interval(confidence_level)
            # also use CI to auto adjust bounds (keeping 99.9% CI)
            lower99, upper99 = sampling.confidence_interval(0.999)
            axes.set(xlim=(lower99, upper99))
            axes.axvline(
                lower,
                color="red",
                style="--",
                alpha=0.7,
                label=rf"${confidence_level * 100:.1f}\%$ CI",
            )
            axes.axvline(upper, color="red", style="--", alpha=0.7)
        else:
            # For jackknife or other methods, use error bounds
            axes.axvline(
                mean_val - error_val,
                color="red",
                style="--",
                alpha=0.7,
                label="Mean ± Error",
            )
            axes.axvline(mean_val + error_val, color="red", style="--", alpha=0.7)

        axes.axvline(
            mean_val,
            color="red",
            style="-",
            width=2,
            label=f"Mean: {mean_val:.6f}",
        )

        # Bin in numpy and emit bars, so both backends agree on the binning.
        heights, edges = np.histogram(resampled, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        axes.bar(centers, heights, width=np.diff(edges), alpha=0.7, native=kwargs or None)

        # Add full sample value
        axes.axvline(
            sampling.full_sample_value,
            color="orange",
            style="-",
            width=2,
            label=f"Full Sample: {sampling.full_sample_value:.6f}",
        )

        # Add bias information for bootstrap
        if sampling.sampling_info.method == "bootstrap" and show_bias:
            bias = sampling.bootstrap_bias
            bias_corrected = sampling.bias_corrected_mean
            if abs(bias) > 1e-10:  # Only show if bias is significant
                axes.axvline(
                    bias_corrected,
                    color="green",
                    style=":",
                    width=2,
                    alpha=0.8,
                    label=f"Bias Corrected: {bias_corrected:.6f}",
                )

        # Labels and formatting
        title_header = (
            f"${sampling.latex_str}$"
            if sampling.latex_str
            else sampling.observable_name.replace("_", " ")
        )
        axes.set(
            xlabel="Value",
            ylabel="Density",
            title=(
                f"{title_header}\n"
                f"({sampling.sampling_info.method.title()}, "
                f"N={sampling.sampling_info.num_resamplings})"
            ),
        )
        axes.legend()
        axes.grid(True, alpha=0.3)

        return axes

    def plot_sampling_errorbar(
        self,
        samplings: SigmondSampling | list[SigmondSampling] | None = None,
        x_values: np.ndarray | list | None = None,
        axes: slp.Axes | None = None,
        labels: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
        **kwargs,
    ) -> slp.Axes:
        """
        Plot error bar representation of SigmondSampling objects.

        Args:
            samplings: Single SigmondSampling or list (uses all from stats if None)
            x_values: X-axis values for each sampling (uses indices if None)
            axes: Axes to draw on (creates a figure if None)
            labels: Labels for each sampling (uses latex_str if None)
            figsize: Figure size (uses default if None)
            backend: Backend for this call (uses the active one if None)
            ax: Deprecated alias for ``axes``
            **kwargs: Extra backend kwargs forwarded to the errorbar mark

        Returns:
            The Axes drawn on
        """
        # Use provided samplings or all from stats
        if samplings is None:
            if self.stats is None:
                raise ValueError("Must provide samplings or initialize with SamplingStats")
            samplings = self.stats

        axes = resolve_axes(axes, ax, figsize=figsize or self.default_figsize, backend=backend)
        samplings = _as_observable_collection(samplings)

        n_samplings = len(samplings)

        # Set default x values
        if x_values is None:
            x_values = np.arange(n_samplings)
        elif len(x_values) != n_samplings:
            raise ValueError("Length of x_values must match number of samplings")

        # Set default labels - use str() method which handles latex_str automatically
        if labels is None:
            labels = [_sampling_latex_label(s) for s in samplings]
        elif len(labels) != n_samplings:
            raise ValueError("Length of labels must match number of samplings")

        # Extract means and errors
        means = np.array([s.mean for s in samplings])
        errors = np.array([s.error for s in samplings])

        axes.errorbar(
            x_values,
            means,
            yerr=errors,
            marker="o",
            capsize=5,
            width=2,
            markersize=6,
            # On plotly this puts the observable name under the cursor.
            hover=[f"{lab}: {m:.6g} ± {e:.2g}" for lab, m, e in zip(labels, means, errors)],
            native=kwargs or None,
        )

        # Labels and formatting
        is_index_axis = np.array_equal(x_values, np.arange(n_samplings))
        if n_samplings == 1:
            title = (
                f"${samplings[0].observable_info.latex_str}$ "
                f"({samplings[0].sampling_info.method.title()})"
            )
        else:
            ensemble_names = list({s.ensemble_info.name for s in samplings})
            title = "Multiple Observables"
            if len(ensemble_names) == 1:
                title += f" ({ensemble_names[0]})"

        axes.set(
            xlabel="Observable" if is_index_axis else "X Value",
            ylabel="Value",
            title=title,
        )

        # Set x-tick labels if custom labels provided
        if labels and n_samplings <= 20:  # Only show labels if not too many points
            axes.ticks("x", x_values, labels, rotation=45)

        axes.grid(True, alpha=0.3)

        return axes

    def plot_corner(
        self,
        labels: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """
        Create corner plot for multi-observable correlation visualization.

        matplotlib-only: this wraps the third-party ``corner`` package, which
        has no plotly equivalent.

        Args:
            labels: Labels for each observable (uses observable names if None)
            **kwargs: Additional arguments passed to corner.corner()

        Returns:
            corner plot figure
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for corner plots")

        if slp.current() != "matplotlib":
            raise slp.UnsupportedFeature(
                "plot_corner is matplotlib-only (it wraps the `corner` package); "
                f"the active backend is {slp.current()!r}"
            )

        try:
            import corner
        except ImportError:
            raise ImportError(
                "corner package is required for corner plots. Install with: pip install corner"
            )

        # Set default labels - use str() method which handles latex_str automatically
        if labels is None:
            labels = ["$" + s + "$" for s in self.stats.obs.latex_str]
        elif len(labels) != len(self.stats):
            raise ValueError("Length of labels must match number of selected observables")

        # Default corner plot settings
        corner_kwargs = {
            "labels": labels,
            "show_titles": False,
            "title_kwargs": {"fontsize": 10},
            "label_kwargs": {"fontsize": 14},
            # add tick number size
            "tick_kwargs": {"fontsize": 10},
            "hist_kwargs": {"density": True, "alpha": 0.7},
            "scatter_kwargs": {"alpha": 0.6, "s": 1},
            "contour_kwargs": {"colors": "blue"},
            "bins": 30,
            "truths": self.stats.val.full_sample_value,
        }
        corner_kwargs.update(kwargs)

        # Create corner plot
        fig = corner.corner(self.stats.array.T, **corner_kwargs)

        # Add ensemble info to the figure title
        ensemble_names = list({s.ensemble_info.name for s in self.stats})
        if len(ensemble_names) == 1:
            fig.suptitle(
                f"Ensemble: {ensemble_names[0]} ({self.stats[0].sampling_info.method.title()})",
                y=0.98,
                fontsize=16,
            )
        else:
            fig.suptitle(
                f"Multiple Ensembles ({self.stats[0].sampling_info.method.title()})",
                y=0.98,
                fontsize=16,
            )

        return slp.wrap(fig)

    def plot_correlation_matrix(
        self,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
        **kwargs,
    ) -> slp.Axes:
        """
        Plot correlation matrix heatmap.

        Args:
            axes: Axes to draw on (creates a figure if None)
            figsize: Figure size (uses default if None)
            backend: Backend for this call (uses the active one if None)
            ax: Deprecated alias for ``axes``
            **kwargs: Extra backend kwargs forwarded to the heatmap mark

        Returns:
            The Axes drawn on
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for correlation matrix")

        axes = resolve_axes(axes, ax, figsize=figsize or self.default_figsize, backend=backend)

        axes.heatmap(
            self.stats.corr_matrix,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            # Row 0 at the top, the usual convention for a correlation matrix.
            origin="upper",
            colorbar_label="Correlation",
            native=kwargs or None,
        )

        # Labels - use str() method which handles latex_str automatically
        labels = ["$" + s + "$" for s in self.stats.obs.latex_str]
        axes.ticks("x", range(len(labels)), labels, rotation=45)
        axes.ticks("y", range(len(labels)), labels)
        axes.set(title="Observable Correlation Matrix")

        return axes

    def plot_effective_sample_size(
        self,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        max_labels: int = 20,
        backend: str | None = None,
        ax: Any = None,
        **kwargs,
    ) -> slp.Axes:
        """
        Plot effective sample sizes for all observables as a bar chart.

        Args:
            axes: Axes to draw on (creates a figure if None)
            figsize: Figure size (uses default if None)
            max_labels: Max number of observables before suppressing x-tick labels
            backend: Backend for this call (uses the active one if None)
            ax: Deprecated alias for ``axes``
            **kwargs: Extra backend kwargs forwarded to the bar mark

        Returns:
            The Axes drawn on
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for effective sample size plot")

        axes = resolve_axes(axes, ax, figsize=figsize or self.default_figsize, backend=backend)

        eff_sizes = self.stats.effective_sample_size
        obs_names = ["$" + s + "$" for s in self.stats.obs.latex_str]

        axes.bar(list(range(len(eff_sizes))), eff_sizes, native=kwargs or None)
        axes.set(
            xlabel="Observable Index",
            ylabel="Effective Sample Size",
            title="Effective Sample Sizes",
        )
        if len(obs_names) <= max_labels:
            axes.ticks("x", range(len(obs_names)), obs_names, rotation=45)
        axes.grid(True, alpha=0.3)

        return axes

    def plot_stats_summary(
        self,
        figsize: tuple[float, float] | None = None,
        panels: list[str] | None = None,
        obs_index: int | None = None,
        title: str | None = None,
        layout: tuple[int, int] | None = None,
        backend: str | None = None,
    ) -> slp.Figure:
        """
        Create a summary plot composed of configurable panels.

        Args:
            figsize: Figure size. Computed adaptively from panel count and observable
                count if None.
            panels: Ordered list of panel names to include. Valid names:
                ``"errorbar"``, ``"correlation"``, ``"histogram"``,
                ``"eff_sample_size"``. Defaults to ``["histogram", "correlation"]``.
            obs_index: Index of the observable shown in the histogram panel.
                ``None`` (default) plots one histogram per observable.
            title: Optional overall figure title (suptitle).
            layout: Override grid shape ``(nrows, ncols)``. Computed from panel
                count if None.
            backend: Backend for this call (uses the active one if None)

        Returns:
            The Figure
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for summary plots")

        panels = list(panels) if panels is not None else list(_DEFAULT_PANELS)

        unknown = [p for p in panels if p not in _VALID_PANELS]
        if unknown:
            raise ValueError(f"Unknown panel(s): {unknown}. Valid panels: {sorted(_VALID_PANELS)}")

        # Expand panels into (panel_name, histogram_idx) slots.
        # "histogram" with obs_index=None becomes one slot per observable;
        # combined panels (correlation, errorbar, eff_sample_size) are always one slot.
        slots: list[tuple[str, int | None]] = []
        for p in panels:
            if p == "histogram" and obs_index is None:
                for i in range(self.stats.num_observables):
                    slots.append(("histogram", i))
            else:
                slots.append((p, obs_index if p == "histogram" else None))

        n = len(slots)
        nrows, ncols = layout if layout is not None else _summary_grid(n)
        figsize = figsize or _summary_figsize(nrows, ncols, panels, self.stats.num_observables)

        fig = slp.figure(nrows, ncols, figsize=figsize, backend=backend)

        def _draw(target: slp.Axes, panel: str, idx: int | None) -> None:
            if panel == "histogram":
                self.plot_sampling_histogram(sampling=idx, axes=target)
            elif panel == "correlation":
                self.plot_correlation_matrix(axes=target)
            elif panel == "errorbar":
                self.plot_sampling_errorbar(axes=target)
            elif panel == "eff_sample_size":
                self.plot_effective_sample_size(axes=target)

        for i, (panel, idx) in enumerate(slots):
            _draw(fig.axes(i // ncols, i % ncols), panel, idx)

        # Hide unused cells
        for i in range(n, nrows * ncols):
            fig.hide(i // ncols, i % ncols)

        if title:
            fig.suptitle(title)

        return fig

    def plot_bootstrap_intervals(
        self,
        samplings: SigmondSampling | list[SigmondSampling] | None = None,
        x_values: np.ndarray | list | None = None,
        confidence_levels: list[float] = [0.68, 0.95],
        axes: slp.Axes | None = None,
        labels: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
        **kwargs,
    ) -> slp.Axes:
        """
        Plot bootstrap confidence intervals for multiple confidence levels.

        Args:
            samplings: Single SigmondSampling or list (uses all from stats if None)
            x_values: X-axis values for each sampling (uses indices if None)
            confidence_levels: List of confidence levels to show
            axes: Axes to draw on (creates a figure if None)
            labels: Labels for each sampling (uses observable names if None)
            figsize: Figure size (uses default if None)
            backend: Backend for this call (uses the active one if None)
            ax: Deprecated alias for ``axes``
            **kwargs: ``central_kwargs`` / ``band_kwargs`` dicts of backend kwargs

        Returns:
            The Axes drawn on
        """
        # Use provided samplings or all from stats
        if samplings is None:
            if self.stats is None:
                raise ValueError("Must provide samplings or initialize with SamplingStats")
            samplings = self.stats

        axes = resolve_axes(axes, ax, figsize=figsize or self.default_figsize, backend=backend)

        samplings = _as_observable_collection(samplings)

        # Filter only bootstrap samplings
        bootstrap_samplings = [s for s in samplings if s.sampling_info.method == "bootstrap"]
        if not bootstrap_samplings:
            raise ValueError("No bootstrap samplings found")

        n_samplings = len(bootstrap_samplings)

        # Set default x values
        if x_values is None:
            x_values = np.arange(n_samplings)
        elif len(x_values) != n_samplings:
            raise ValueError("Length of x_values must match number of bootstrap samplings")

        # Set default labels - use str() method which handles latex_str automatically
        if labels is None:
            labels = [_sampling_latex_label(s) for s in bootstrap_samplings]
        elif len(labels) != n_samplings:
            raise ValueError("Length of labels must match number of bootstrap samplings")

        # Extract means
        means = np.array([s.mean for s in bootstrap_samplings])

        # Plot central values
        axes.errorbar(
            x_values,
            means,
            marker="o",
            markersize=8,
            color="black",
            label="Mean",
            native=kwargs.get("central_kwargs"),
        )

        # Plot confidence intervals for different levels
        colors = ["red", "blue", "green", "purple", "orange"]
        for i, conf_level in enumerate(confidence_levels):
            color = colors[i % len(colors)]

            # Calculate confidence intervals
            lower_bounds = []
            upper_bounds = []

            for sampling in bootstrap_samplings:
                try:
                    lower, upper = sampling.confidence_interval(conf_level)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                except ValueError:
                    # Skip if confidence interval calculation fails
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)

            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            # Plot confidence bands, labelling only the first box of each level
            label: str | None = f"{conf_level:.0%} CI"
            for j, x in enumerate(x_values):
                if np.isnan(lower_bounds[j]) or np.isnan(upper_bounds[j]):
                    continue
                axes.band(
                    [x - 0.3, x + 0.3],
                    [lower_bounds[j], lower_bounds[j]],
                    [upper_bounds[j], upper_bounds[j]],
                    color=color,
                    alpha=0.3,
                    label=label,
                    native=kwargs.get("band_kwargs"),
                )
                label = None

        # Labels and formatting
        is_index_axis = np.array_equal(x_values, np.arange(n_samplings))
        axes.set(
            xlabel="Observable Index" if is_index_axis else "X Value",
            ylabel="Value",
            title="Bootstrap Confidence Intervals",
        )

        # Set x-tick labels
        if labels and n_samplings <= 20:
            axes.ticks("x", x_values, labels, rotation=45)

        axes.legend()
        axes.grid(True, alpha=0.3)

        return axes
