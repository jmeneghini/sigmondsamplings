"""Plots of labeled single-pivot overlaps, retaining their normalization."""
from __future__ import annotations

import numpy as np

from slat.plotting.lifecycle import figure_factory

from ..correlators.observables import OverlapObsInfo
from ..correlators.operators import parse_operator
from ..data import SigmondData
from ._util import central_and_error, estimate_hover_text, resolve_axes

__all__ = ["OverlapPlotter"]


class OverlapPlotter:
    """Bars for one operator or a rectangular operator/level heatmap.

    Input is the result of ``SinglePivot.overlaps``. Plots carry the original
    rotated labels, normalization convention and normalization time.
    """

    def __init__(self, overlaps: SigmondData):
        if overlaps.dims != ("operator", "level") or overlaps.is_complex:
            raise ValueError("overlap plots require real (operator, level) data")
        infos = list(overlaps.infos.flat)
        if not infos or not all(isinstance(info, OverlapObsInfo) for info in infos):
            raise ValueError("overlap plots require pivot overlap metadata")
        conventions = {(i.pivot_id, i.normalize, i.norm_time) for i in infos}
        if len(conventions) != 1:
            raise ValueError("overlaps come from different pivots or normalizations")
        self.overlaps = overlaps
        _, self.normalized, self.norm_time = conventions.pop()
        self.value_label = "Full-row normalized overlap" if self.normalized else "Overlap |Z|² |A|"

    @figure_factory
    def plot_bars(self, operator, *, axes=None, figsize=None, backend=None, title=None):
        """One source operator's overlaps, with sampling-aware error bars."""
        coord = parse_operator(operator).coordinate
        row = self.overlaps.sel(operator=coord)
        central, error = central_and_error(row)
        x = np.arange(row.shape[0])
        levels = [str(parse_operator(label)) for label in row.coords["level"]]
        hover = [f"{coord}<br>{label}<br>{estimate_hover_text(row.isel(level=i))}"
                 for i, label in enumerate(levels)]
        axes = resolve_axes(axes, figsize=figsize, backend=backend)
        axes.bar(x, central, hover=hover)
        axes.errorbar(x, central, yerr=error, color="black", marker=None)
        axes.ticks("x", x, levels, rotation=45)
        axes.set(xlabel="Rotated level", ylabel=self.value_label,
                 title=title if title is not None else f"{coord} (tN={self.norm_time})")
        return axes

    @figure_factory
    def plot_heatmap(self, *, axes=None, figsize=None, backend=None, title=None):
        """Operator rows and original rotated-level columns; hover includes errors."""
        central, _ = central_and_error(self.overlaps)
        ops = [str(parse_operator(c)) for c in self.overlaps.coords["operator"]]
        levels = [str(parse_operator(c)) for c in self.overlaps.coords["level"]]
        hover = [[f"{op}<br>{level}<br>{estimate_hover_text(self.overlaps.isel(operator=i, level=j))}"
                  for j, level in enumerate(levels)] for i, op in enumerate(ops)]
        axes = resolve_axes(axes, figsize=figsize, backend=backend)
        axes.heatmap(central, origin="upper", colorbar_label=self.value_label, hover=hover,
                     vmin=0, vmax=1 if self.normalized else None)
        axes.ticks("x", range(len(levels)), levels, rotation=45)
        axes.ticks("y", range(len(ops)), ops)
        axes.set(xlabel="Rotated level", ylabel="Operator",
                 title=title if title is not None else f"{self.value_label} (tN={self.norm_time})")
        return axes
