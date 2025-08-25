"""
Plotting utilities for SigmondSamplings and SigmondStats.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Optional, Tuple, Callable, Any
from .sampling import SigmondSampling
from .stats import SamplingStats


class SigmondPlotter:
    """
    A comprehensive plotting class for SigmondSampling and SamplingStats objects.
    
    This class provides methods for visualizing statistical data, correlations,
    and fitting results from lattice QCD analysis workflows.
    """
    
    def __init__(self, stats: Optional[SamplingStats] = None, 
                 default_figsize: Tuple[float, float] = (10, 6),
                 default_style: Dict[str, Any] = None):
        """
        Initialize the plotter.
        
        Args:
            stats: Optional SamplingStats object for default data source
            default_figsize: Default figure size for plots
            default_style: Default styling parameters
        """
        self.stats = stats
        self.default_figsize = default_figsize
        self.default_style = default_style or {}
        
    def plot_sampling_histogram(self, sampling: Optional[SigmondSampling | int] = None, 
                               bins: Union[int, str] = 'auto',
                               ax: Optional[plt.Axes] = None, show_stats: bool = True,
                               confidence_level: float = 0.68,
                               show_bias: bool = False,
                               figsize: Optional[Tuple[float, float]] = None,
                               **kwargs) -> plt.Axes:
        """
        Plot histogram of resampled values for a SigmondSampling object.
        
        Args:
            sampling: SigmondSampling object or index to plot (uses first from stats if None)
            bins: Number of bins or binning strategy for histogram
            ax: Matplotlib axes to plot on (creates new if None)
            show_stats: Whether to show mean and error as vertical lines
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
            sampling = self.stats.samplings[0]
        elif isinstance(sampling, int):
            if self.stats is None:
                raise ValueError("Must provide sampling or initialize with SamplingStats")
            sampling = self.stats.samplings[sampling]

        if ax is None:
            figsize = figsize or self.default_figsize
            _, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram of resampled values
        resampled = sampling.resampled_values
        ax.hist(resampled, bins=bins, alpha=0.7, density=True, **kwargs)
        
        if show_stats:
            # Add vertical lines for mean and error bounds
            mean_val = sampling.mean
            error_val = sampling.error
            
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, 
                      label=f'Mean: {mean_val:.6f}')
            
            # Use confidence interval for bootstrap, error bounds for jackknife
            if sampling.sampling_info.method == 'bootstrap':
                lower, upper = sampling.confidence_interval(confidence_level)
                # also use CI to auto adjust bounds (keeping 99.9% CI)
                lower99, upper99 = sampling.confidence_interval(0.99)
                ax.set_xlim((lower99, upper99))
                ax.axvline(lower, color='red', linestyle='--', alpha=0.7,
                            label=rf'${confidence_level*100:.1f}\%$ CI')
                ax.axvline(upper, color='red', linestyle='--', alpha=0.7)
            else:
                # For jackknife or other methods, use error bounds
                ax.axvline(mean_val - error_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean ± Error')
                ax.axvline(mean_val + error_val, color='red', linestyle='--', alpha=0.7)
            
            # Add full sample value
            ax.axvline(sampling.full_sample_value, color='orange', linestyle='-', 
                      linewidth=2, label=f'Full Sample: {sampling.full_sample_value:.6f}')
            
            # Add bias information for bootstrap
            if sampling.sampling_info.method == 'bootstrap' and show_bias:
                bias = sampling.bootstrap_bias()
                bias_corrected = sampling.bias_corrected_mean()
                if abs(bias) > 1e-10:  # Only show if bias is significant
                    ax.axvline(bias_corrected, color='green', linestyle=':', linewidth=2,
                              alpha=0.8, label=f'Bias Corrected: {bias_corrected:.6f}')
        
        # Labels and formatting
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{str(sampling.observable_info)} Distribution\n'
                    f'({sampling.sampling_info.method.title()}, '
                    f'N={sampling.sampling_info.num_resamplings})')
        
        if show_stats:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_sampling_errorbar(self, samplings: Optional[Union[SigmondSampling, List[SigmondSampling]]] = None,
                              x_values: Optional[Union[np.ndarray, List]] = None,
                              ax: Optional[plt.Axes] = None, 
                              labels: Optional[List[str]] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              **kwargs) -> plt.Axes:
        """
        Plot error bar representation of SigmondSampling objects.
        
        Args:
            samplings: Single SigmondSampling or list (uses all from stats if None)
            x_values: X-axis values for each sampling (uses indices if None)
            ax: Matplotlib axes to plot on (creates new if None)
            labels: Labels for each sampling (uses latex_str or observable names if None)
            figsize: Figure size (uses default if None)
            **kwargs: Additional arguments passed to matplotlib errorbar()
            
        Returns:
            matplotlib Axes object
        """
        # Use provided samplings or all from stats
        if samplings is None:
            if self.stats is None:
                raise ValueError("Must provide samplings or initialize with SamplingStats")
            samplings = self.stats.samplings
        
        if ax is None:
            figsize = figsize or self.default_figsize
            _, ax = plt.subplots(figsize=figsize)
        
        # Ensure samplings is a list
        if isinstance(samplings, SigmondSampling):
            samplings = [samplings]
        
        n_samplings = len(samplings)
        
        # Set default x values
        if x_values is None:
            x_values = np.arange(n_samplings)
        elif len(x_values) != n_samplings:
            raise ValueError("Length of x_values must match number of samplings")
        
        # Set default labels - use str() method which handles latex_str automatically
        if labels is None:
            labels = [str(s.observable_info) for s in samplings]
        elif len(labels) != n_samplings:
            raise ValueError("Length of labels must match number of samplings")
        
        # Extract means and errors
        means = np.array([s.mean for s in samplings])
        errors = np.array([s.error for s in samplings])
        
        # Plot error bars
        errorbar_kwargs = {'fmt': 'o', 'capsize': 5, 'capthick': 2, 'markersize': 6}
        errorbar_kwargs.update(kwargs)
        
        ax.errorbar(x_values, means, yerr=errors, **errorbar_kwargs)
        
        # Labels and formatting
        ax.set_xlabel('Observable Index' if np.array_equal(x_values, np.arange(n_samplings)) else 'X Value')
        ax.set_ylabel('Value')
        
        if n_samplings == 1:
            ax.set_title(f'{samplings[0].observable_info.name} '
                        f'({samplings[0].sampling_info.method.title()})')
        else:
            ensemble_names = list(set(s.ensemble_info.ensemble_name for s in samplings))
            title = f'Multiple Observables'
            if len(ensemble_names) == 1:
                title += f' ({ensemble_names[0]})'
            ax.set_title(title)
        
        # Set x-tick labels if custom labels provided
        if labels and n_samplings <= 20:  # Only show labels if not too many points
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_corner(self, observables: Optional[List[int]] = None,
                   labels: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Create corner plot for multi-observable correlation visualization using corner package.
        
        Args:
            observables: Indices of observables to include (uses all if None)
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
            raise ImportError("corner package is required for corner plots. "
                             "Install with: pip install corner")
        
        # Select observables to plot
        if observables is None:
            selected_samplings = self.stats.samplings
        else:
            if max(observables) >= len(self.stats.samplings):
                raise IndexError("Observable index out of range")
            selected_samplings = [self.stats.samplings[i] for i in observables]
        
        # Create data matrix (samples x observables)
        data_matrix = np.column_stack([s.resampled_values for s in selected_samplings])
        
        # Set default labels - use str() method which handles latex_str automatically
        if labels is None:
            labels = [str(s.observable_info) for s in selected_samplings]
        elif len(labels) != len(selected_samplings):
            raise ValueError("Length of labels must match number of selected observables")
        
        # Default corner plot settings
        corner_kwargs = {
            'labels': labels,
            'show_titles': True,
            'title_kwargs': {'fontsize': 12},
            'label_kwargs': {'fontsize': 14},
            'hist_kwargs': {'density': True, 'alpha': 0.7},
            'scatter_kwargs': {'alpha': 0.6, 's': 1},
            'contour_kwargs': {'colors': 'blue'},
            'bins': 30
        }
        corner_kwargs.update(kwargs)
        
        # Create corner plot
        fig = corner.corner(data_matrix, **corner_kwargs)
        
        # Add ensemble info to the figure title
        ensemble_names = list(set(s.ensemble_info.ensemble_name for s in selected_samplings))
        if len(ensemble_names) == 1:
            fig.suptitle(f'Ensemble: {ensemble_names[0]} '
                        f'({selected_samplings[0].sampling_info.method.title()})',
                        y=0.98, fontsize=16)
        else:
            fig.suptitle(f'Multiple Ensembles '
                        f'({selected_samplings[0].sampling_info.method.title()})',
                        y=0.98, fontsize=16)
        
        return fig
    
    def plot_fit_result(self, x_values: np.ndarray, 
                       fitted_params: Dict[str, SigmondSampling], model_func: Callable,
                       ax: Optional[plt.Axes] = None, x_fit_range: Optional[Tuple[float, float]] = None,
                       n_fit_points: int = 100, confidence_level: float = 0.68,
                       show_data: bool = True, show_fit: bool = True, 
                       show_uncertainty: bool = True, 
                       figsize: Optional[Tuple[float, float]] = None,
                       **kwargs) -> plt.Axes:
        """
        Plot data points with fitted function and uncertainty bands.
        
        Args:
            x_values: X-values corresponding to each observable  
            fitted_params: Dictionary of fitted parameters from SamplingStats.fit_function()
            model_func: The model function used in fitting
            ax: Matplotlib axes to plot on (creates new if None)
            x_fit_range: Range for plotting fit function (uses data range if None)
            n_fit_points: Number of points for smooth fit curve
            confidence_level: Confidence level for uncertainty bands (0.68 = 1σ)
            show_data: Whether to show data points with error bars
            show_fit: Whether to show fitted function
            show_uncertainty: Whether to show uncertainty bands
            figsize: Figure size (uses default if None)
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Axes object
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for fit results")
            
        if ax is None:
            figsize = figsize or self.default_figsize
            _, ax = plt.subplots(figsize=figsize)
        
        if len(x_values) != self.stats.num_observables:
            raise ValueError("Length of x_values must match number of observables")
        
        # Plot data points with error bars
        if show_data:
            data_means = self.stats.means()
            data_errors = self.stats.errors()
            
            data_kwargs = {'fmt': 'o', 'capsize': 5, 'capthick': 2, 
                          'markersize': 6, 'label': 'Data', 'color': 'black'}
            data_kwargs.update(kwargs.get('data_kwargs', {}))
            
            ax.errorbar(x_values, data_means, yerr=data_errors, **data_kwargs)
        
        # Plot fitted function and uncertainty bands
        if show_fit or show_uncertainty:
            # Set up x range for fit plotting
            if x_fit_range is None:
                x_min, x_max = np.min(x_values), np.max(x_values)
                x_range = x_max - x_min
                x_fit_range = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            
            x_fit = np.linspace(x_fit_range[0], x_fit_range[1], n_fit_points)
            
            # Extract parameter values for all samples
            param_names = sorted(fitted_params.keys())
            param_arrays = [fitted_params[name].data for name in param_names]
            
            # Calculate fit function values for all parameter samples
            fit_samples = []
            for sample_idx in range(len(param_arrays[0])):
                params = np.array([p[sample_idx] for p in param_arrays])
                y_fit_sample = model_func(x_fit, params)
                fit_samples.append(y_fit_sample)
            
            fit_samples = np.array(fit_samples)
            
            # Calculate central value and uncertainty bands
            fit_mean = np.mean(fit_samples, axis=0)
            fit_std = np.std(fit_samples, axis=0, ddof=1)
            
            # Apply bootstrap/jackknife correction if needed
            if fitted_params[param_names[0]].sampling_info.method == 'jackknife':
                n_samples = len(fit_samples)
                fit_std *= np.sqrt(n_samples - 1)
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            percentiles = [50 * alpha, 100 - 50 * alpha]
            fit_lower = np.percentile(fit_samples, percentiles[0], axis=0)
            fit_upper = np.percentile(fit_samples, percentiles[1], axis=0)
            
            # Plot fitted function
            if show_fit:
                fit_kwargs = {'color': 'red', 'linewidth': 2, 'label': 'Fit'}
                fit_kwargs.update(kwargs.get('fit_kwargs', {}))
                ax.plot(x_fit, fit_mean, **fit_kwargs)
            
            # Plot uncertainty bands
            if show_uncertainty:
                band_kwargs = {'alpha': 0.3, 'color': 'red', 
                              'label': f'{confidence_level:.0%} Confidence'}
                band_kwargs.update(kwargs.get('band_kwargs', {}))
                ax.fill_between(x_fit, fit_lower, fit_upper, **band_kwargs)
        
        # Labels and formatting
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add chi-squared information if available
        param_means = {name: param.mean for name, param in fitted_params.items()}
        param_values = np.array([param_means[name] for name in param_names])
        theory_values = model_func(x_values, param_values)
        chi_sq, dof = self.stats.chi_squared(theory_values, use_correlation=True)
        
        ax.set_title(f'Fit Result (χ²/dof = {chi_sq:.2f}/{dof} = {chi_sq/dof:.3f})')
        
        if show_data and (show_fit or show_uncertainty):
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_correlation_matrix(self, ax: Optional[plt.Axes] = None, 
                               figsize: Optional[Tuple[float, float]] = None,
                               **kwargs) -> plt.Axes:
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
            
        if ax is None:
            figsize = figsize or self.default_figsize
            _, ax = plt.subplots(figsize=figsize)
        
        corr_matrix = self.stats.correlation_matrix()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, **kwargs)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Correlation')
        
        # Labels - use str() method which handles latex_str automatically
        labels = [str(s.observable_info) for s in self.stats.samplings]
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        ax.set_title('Observable Correlation Matrix')
        
        return ax
    
    def plot_stats_summary(self, figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create a comprehensive summary plot with multiple panels.
        
        Args:
            figsize: Figure size (uses larger default if None)
            
        Returns:
            matplotlib Figure object
        """
        if self.stats is None:
            raise ValueError("Must initialize with SamplingStats for summary plots")
        
        figsize = figsize or (15, 10)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Error bar plot
        self.plot_sampling_errorbar(ax=axes[0, 0])
        axes[0, 0].set_title('Observable Values')
        
        # Correlation matrix
        self.plot_correlation_matrix(ax=axes[0, 1])
        
        # First observable histogram
        self.plot_sampling_histogram(ax=axes[1, 0])
        
        # Effective sample sizes
        eff_sizes = self.stats.effective_sample_size()
        obs_names = [str(s.observable_info) for s in self.stats.samplings]
        
        axes[1, 1].bar(range(len(eff_sizes)), eff_sizes)
        axes[1, 1].set_xlabel('Observable Index')
        axes[1, 1].set_ylabel('Effective Sample Size')
        axes[1, 1].set_title('Effective Sample Sizes')
        axes[1, 1].set_xticks(range(len(obs_names)))
        axes[1, 1].set_xticklabels(obs_names, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_bootstrap_intervals(self, samplings: Optional[Union[SigmondSampling, List[SigmondSampling]]] = None,
                                x_values: Optional[Union[np.ndarray, List]] = None,
                                confidence_levels: List[float] = [0.68, 0.95],
                                ax: Optional[plt.Axes] = None,
                                labels: Optional[List[str]] = None,
                                figsize: Optional[Tuple[float, float]] = None,
                                **kwargs) -> plt.Axes:
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
            samplings = self.stats.samplings
        
        if ax is None:
            figsize = figsize or self.default_figsize
            _, ax = plt.subplots(figsize=figsize)
        
        # Ensure samplings is a list
        if isinstance(samplings, SigmondSampling):
            samplings = [samplings]
        
        # Filter only bootstrap samplings
        bootstrap_samplings = [s for s in samplings if s.sampling_info.method == 'bootstrap']
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
            labels = [str(s.observable_info) for s in bootstrap_samplings]
        elif len(labels) != n_samplings:
            raise ValueError("Length of labels must match number of bootstrap samplings")
        
        # Extract means
        means = np.array([s.mean for s in bootstrap_samplings])
        
        # Plot central values
        central_kwargs = {'fmt': 'o', 'markersize': 8, 'color': 'black', 'label': 'Mean'}
        central_kwargs.update(kwargs.get('central_kwargs', {}))
        ax.errorbar(x_values, means, **central_kwargs)
        
        # Plot confidence intervals for different levels
        colors = ['red', 'blue', 'green', 'purple', 'orange']
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
            band_kwargs = {'alpha': 0.3, 'color': color, 
                          'label': f'{conf_level:.0%} CI'}
            band_kwargs.update(kwargs.get('band_kwargs', {}))
            
            for j, x in enumerate(x_values):
                if not (np.isnan(lower_bounds[j]) or np.isnan(upper_bounds[j])):
                    ax.fill_between([x-0.3, x+0.3], 
                                   [lower_bounds[j], lower_bounds[j]], 
                                   [upper_bounds[j], upper_bounds[j]], 
                                   **band_kwargs)
                    # Only add label once
                    band_kwargs.pop('label', None)
        
        # Labels and formatting
        ax.set_xlabel('Observable Index' if np.array_equal(x_values, np.arange(n_samplings)) else 'X Value')
        ax.set_ylabel('Value')
        ax.set_title('Bootstrap Confidence Intervals')
        
        # Set x-tick labels
        if labels and n_samplings <= 20:
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
