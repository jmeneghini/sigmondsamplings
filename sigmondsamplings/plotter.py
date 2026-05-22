"""
Plotting utilities for SigmondSamplings and SigmondStats.
"""

import math
from collections.abc import Iterable
from typing import (
    Any,
)

import matplotlib.pyplot as plt
import numpy as np

from .obervable_collection import ObservableCollection
from .sampling import SigmondSampling
from .stats import SamplingStats

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


def _get_axes(
    ax: plt.Axes | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


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
        ax: plt.Axes | None = None,
        confidence_level: float = 0.68,
        show_bias: bool = False,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot histogram of resampled values for a SigmondSampling object.

        Args:
            sampling: SigmondSampling object or index to plot (uses first from stats if None)
            bins: Number of bins or binning strategy for histogram
            ax: Matplotlib axes to plot on (creates new if None)
            confidence_level: Confidence level for bootstrap CI (0.68 = 1σ)
            figsize: Figure size (uses default if None)
            **kwargs: Additional arguments passed to matplotlib hist()

        Returns:
            matplotlib Axes object
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

        figsize = figsize or self.default_figsize
        _, ax = _get_axes(ax, figsize)

        resampled = sampling.resampled_values

        # Add vertical lines for mean and error bounds
        mean_val = sampling.mean
        error_val = sampling.error

        # Use confidence interval for bootstrap, error bounds for jackknife
        if sampling.sampling_info.method == "bootstrap":
            lower, upper = sampling.confidence_interval(confidence_level)
            # also use CI to auto adjust bounds (keeping 99.9% CI)
            lower99, upper99 = sampling.confidence_interval(0.999)
            ax.set_xlim((lower99, upper99))
            ax.axvline(
                lower,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=rf"${confidence_level * 100:.1f}\%$ CI",
            )
            ax.axvline(upper, color="red", linestyle="--", alpha=0.7)
        else:
            # For jackknife or other methods, use error bounds
            ax.axvline(
                mean_val - error_val,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Mean ± Error",
            )
            ax.axvline(mean_val + error_val, color="red", linestyle="--", alpha=0.7)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_val:.6f}",
        )

        # Plot histogram
        ax.hist(resampled, bins=bins, alpha=0.7, density=True, **kwargs)

        # Add full sample value
        ax.axvline(
            sampling.full_sample_value,
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Full Sample: {sampling.full_sample_value:.6f}",
        )

        # Add bias information for bootstrap
        if sampling.sampling_info.method == "bootstrap" and show_bias:
            bias = sampling.bootstrap_bias
            bias_corrected = sampling.bias_corrected_mean
            if abs(bias) > 1e-10:  # Only show if bias is significant
                ax.axvline(
                    bias_corrected,
                    color="green",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Bias Corrected: {bias_corrected:.6f}",
                )

        # Labels and formatting
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        title_header = (
            f"${sampling.latex_str}$"
            if sampling.latex_str
            else sampling.observable_name.replace("_", " ")
        )
        ax.set_title(
            f"{title_header}\n"
            f"({sampling.sampling_info.method.title()}, "
            f"N={sampling.sampling_info.num_resamplings})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_sampling_errorbar(
        self,
        samplings: SigmondSampling | list[SigmondSampling] | None = None,
        x_values: np.ndarray | list | None = None,
        ax: plt.Axes | None = None,
        labels: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot error bar representation of SigmondSampling objects.

        Args:
            samplings: Single SigmondSampling or list (uses all from stats if None)
            x_values: X-axis values for each sampling (uses indices if None)
            ax: Matplotlib axes to plot on (creates new if None)
            labels: Labels for each sampling (uses latex_str if None)
            figsize: Figure size (uses default if None)
            **kwargs: Additional arguments passed to matplotlib errorbar()

        Returns:
            matplotlib Axes object
        """
        # Use provided samplings or all from stats
        if samplings is None:
            if self.stats is None:
                raise ValueError("Must provide samplings or initialize with SamplingStats")
            samplings = self.stats

        figsize = figsize or self.default_figsize
        _, ax = _get_axes(ax, figsize)
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

        # Plot error bars
        errorbar_kwargs = {"fmt": "o", "capsize": 5, "capthick": 2, "markersize": 6}
        errorbar_kwargs.update(kwargs)

        ax.errorbar(x_values, means, yerr=errors, **errorbar_kwargs)

        # Labels and formatting
        ax.set_xlabel(
            "Observable" if np.array_equal(x_values, np.arange(n_samplings)) else "X Value"
        )
        ax.set_ylabel("Value")

        if n_samplings == 1:
            ax.set_title(
                f"${samplings[0].observable_info.latex_str}$ "
                f"({samplings[0].sampling_info.method.title()})"
            )
        else:
            ensemble_names = list(set(s.ensemble_info.name for s in samplings))
            title = "Multiple Observables"
            if len(ensemble_names) == 1:
                title += f" ({ensemble_names[0]})"
            ax.set_title(title)

        # Set x-tick labels if custom labels provided
        if labels and n_samplings <= 20:  # Only show labels if not too many points
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.grid(True, alpha=0.3)

        return ax

    def plot_corner(
        self,
        labels: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """
        Create corner plot for multi-observable correlation visualization using corner package.

        Args:
            labels: Labels for each observable (uses observable names if None)
            **kwargs: Additional arguments passed to corner.corner()

        Returns:
            corner plot figure
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for corner plots")

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
        ensemble_names = list(set(s.ensemble_info.name for s in self.stats))
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

        return fig

    def plot_correlation_matrix(
        self,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot correlation matrix heatmap.

        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            figsize: Figure size (uses default if None)
            **kwargs: Additional arguments passed to matplotlib imshow()

        Returns:
            matplotlib Axes object
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for correlation matrix")

        figsize = figsize or self.default_figsize
        _, ax = _get_axes(ax, figsize)

        corr_matrix = self.stats.corr_matrix

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap="RdBu", vmin=-1, vmax=1, **kwargs)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Correlation")

        # Labels - use str() method which handles latex_str automatically
        labels = ["$" + s + "$" for s in self.stats.obs.latex_str]

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        ax.set_title("Observable Correlation Matrix")

        return ax

    def plot_effective_sample_size(
        self,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        max_labels: int = 20,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot effective sample sizes for all observables as a bar chart.

        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            figsize: Figure size (uses default if None)
            max_labels: Max number of observables before suppressing x-tick labels
            **kwargs: Additional arguments passed to matplotlib bar()

        Returns:
            matplotlib Axes object
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for effective sample size plot")

        figsize = figsize or self.default_figsize
        _, ax = _get_axes(ax, figsize)

        eff_sizes = self.stats.effective_sample_size
        obs_names = ["$" + s + "$" for s in self.stats.obs.latex_str]

        ax.bar(range(len(eff_sizes)), eff_sizes, **kwargs)
        ax.set_xlabel("Observable Index")
        ax.set_ylabel("Effective Sample Size")
        ax.set_title("Effective Sample Sizes")
        if len(obs_names) <= max_labels:
            ax.set_xticks(range(len(obs_names)))
            ax.set_xticklabels(obs_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_stats_summary(
        self,
        figsize: tuple[float, float] | None = None,
        panels: list[str] | None = None,
        obs_index: int | None = None,
        title: str | None = None,
        layout: tuple[int, int] | None = None,
    ) -> plt.Figure:
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

        Returns:
            matplotlib Figure object
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

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        def _draw(ax: plt.Axes, panel: str, idx: int | None) -> None:
            if panel == "histogram":
                self.plot_sampling_histogram(sampling=idx, ax=ax)
            elif panel == "correlation":
                self.plot_correlation_matrix(ax=ax)
            elif panel == "errorbar":
                self.plot_sampling_errorbar(ax=ax)
            elif panel == "eff_sample_size":
                self.plot_effective_sample_size(ax=ax)

        for i, (panel, idx) in enumerate(slots):
            _draw(axes[i // ncols, i % ncols], panel, idx)

        # Hide unused cells
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        if title:
            fig.suptitle(title, y=1.02)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()

        return fig

    def plot_bootstrap_intervals(
        self,
        samplings: SigmondSampling | list[SigmondSampling] | None = None,
        x_values: np.ndarray | list | None = None,
        confidence_levels: list[float] = [0.68, 0.95],
        ax: plt.Axes | None = None,
        labels: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot bootstrap confidence intervals for multiple confidence levels.

        Args:
            samplings: Single SigmondSampling or list (uses all from stats if None)
            x_values: X-axis values for each sampling (uses indices if None)
            confidence_levels: List of confidence levels to show
            ax: Matplotlib axes to plot on (creates new if None)
            labels: Labels for each sampling (uses observable names if None)
            figsize: Figure size (uses default if None)
            **kwargs: Additional plotting arguments

        Returns:
            matplotlib Axes object
        """
        # Use provided samplings or all from stats
        if samplings is None:
            if self.stats is None:
                raise ValueError("Must provide samplings or initialize with SamplingStats")
            samplings = self.stats

        figsize = figsize or self.default_figsize
        _, ax = _get_axes(ax, figsize)

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
        central_kwargs = {
            "fmt": "o",
            "markersize": 8,
            "color": "black",
            "label": "Mean",
        }
        central_kwargs.update(kwargs.get("central_kwargs", {}))
        ax.errorbar(x_values, means, **central_kwargs)

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

            # Plot confidence bands
            band_kwargs = {
                "alpha": 0.3,
                "color": color,
                "label": f"{conf_level:.0%} CI",
            }
            band_kwargs.update(kwargs.get("band_kwargs", {}))

            for j, x in enumerate(x_values):
                if not (np.isnan(lower_bounds[j]) or np.isnan(upper_bounds[j])):
                    ax.fill_between(
                        [x - 0.3, x + 0.3],
                        [lower_bounds[j], lower_bounds[j]],
                        [upper_bounds[j], upper_bounds[j]],
                        **band_kwargs,
                    )
                    # Only add label once
                    band_kwargs.pop("label", None)

        # Labels and formatting
        ax.set_xlabel(
            "Observable Index" if np.array_equal(x_values, np.arange(n_samplings)) else "X Value"
        )
        ax.set_ylabel("Value")
        ax.set_title("Bootstrap Confidence Intervals")

        # Set x-tick labels
        if labels and n_samplings <= 20:
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

