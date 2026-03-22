"""
Color function examples for Chi2Plotter.

This module provides pre-built color functions that can be used with the Chi2Plotter
to customize the coloring of chi-squared landscape plots.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model_func import SigmondModelFunc


def delta_chi2_color_function(chi2_min_global: float = None):
    """
    Create a color function based on Δχ² from the global minimum.

    Args:
        chi2_min_global: Global minimum χ² value for reference

    Returns:
        Color function that maps Δχ² to colors
    """

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        if chi2_min_global is None:
            # Use current minimum as reference
            delta = 0
        else:
            delta = chi2_val - chi2_min_global

        # Map Δχ² to color using matplotlib colormap
        import matplotlib.cm as cm

        if delta <= 1.0:
            return cm.Greens(0.8)  # 1σ - green
        elif delta <= 4.0:
            return cm.Blues(0.8)  # 2σ - blue
        elif delta <= 9.0:
            return cm.Oranges(0.8)  # 3σ - orange
        else:
            return cm.Reds(0.8)  # >3σ - red

    return color_func


def confidence_level_color_function(chi2_min_global: float = None):
    """
    Create a color function based on confidence levels.

    Args:
        chi2_min_global: Global minimum χ² value for reference

    Returns:
        Color function that assigns discrete colors for confidence levels
    """

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        if chi2_min_global is None:
            delta = 0
        else:
            delta = chi2_val - chi2_min_global

        if delta <= 1.0:
            return "green"  # 1σ
        elif delta <= 4.0:
            return "yellow"  # 2σ
        elif delta <= 9.0:
            return "orange"  # 3σ
        else:
            return "red"  # >3σ

    return color_func


def parameter_based_color_function(param_index: int, threshold: float):
    """
    Create a color function based on parameter values.

    Args:
        param_index: Index of parameter to use for coloring
        threshold: Threshold value for color change

    Returns:
        Color function that colors based on parameter value
    """

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        params = model_func.get_parameter_means()
        if param_index < len(params):
            param_val = params[param_index]
            return "red" if param_val > threshold else "blue"
        else:
            return "gray"

    return color_func


def gradient_color_function(param_index: int, param_range: tuple[float, float]):
    """
    Create a color function that uses a gradient based on parameter values.

    Args:
        param_index: Index of parameter to use for coloring
        param_range: (min, max) range for parameter normalization

    Returns:
        Color function that creates gradient coloring
    """

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        import matplotlib.cm as cm

        params = model_func.get_parameter_means()
        if param_index < len(params):
            param_val = params[param_index]
            # Normalize parameter to [0, 1] range
            normalized = (param_val - param_range[0]) / (param_range[1] - param_range[0])
            normalized = np.clip(normalized, 0, 1)
            return cm.viridis(normalized)
        else:
            return "gray"

    return color_func


def custom_chi2_threshold_function(thresholds: list = None, colors: list = None):
    """
    Create a color function with custom χ² thresholds and colors.

    Args:
        thresholds: List of χ² threshold values (default: [1, 4, 9])
        colors: List of colors corresponding to thresholds (default: ['green', 'yellow', 'orange', 'red'])

    Returns:
        Color function with custom thresholds
    """
    if thresholds is None:
        thresholds = [1.0, 4.0, 9.0]
    if colors is None:
        colors = ["green", "yellow", "orange", "red"]

    if len(colors) != len(thresholds) + 1:
        raise ValueError("colors list must have one more element than thresholds list")

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        # Find the minimum chi2 from the model (this would need to be tracked globally)
        # For now, use 0 as reference
        delta = chi2_val  # Assuming chi2_val is already relative to minimum

        for i, threshold in enumerate(thresholds):
            if delta <= threshold:
                return colors[i]
        return colors[-1]  # Return last color if above all thresholds

    return color_func


def heat_map_color_function(colormap: str = "viridis"):
    """
    Create a color function that uses a continuous colormap.

    Args:
        colormap: Matplotlib colormap name

    Returns:
        Color function that uses continuous coloring
    """

    def color_func(model_func: "SigmondModelFunc", chi2_val: float):
        import matplotlib.cm as cm

        # Normalize chi2 value (this would need global min/max tracking)
        # For now, just use the value directly
        normalized = min(max(chi2_val / 10.0, 0), 1)  # Simple normalization

        colormap_func = getattr(cm, colormap, cm.viridis)
        return colormap_func(normalized)

    return color_func
