"""
Spectrum plotting for energy-level collections.

Provides a base `SpectrumPlotter` that renders columns of energy levels grouped
either via a `group_by` key (attribute name or callable) or via explicit
user-supplied collections. A preset `SectorSpectrumPlotter` reproduces the
PSQ -> irrep layout.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .energy_level_collection import SingleEnsembleEnergyCollection
from .sampling import SigmondSampling
from .utils import stacked_positions

__all__ = ["SpectrumPlotter", "SectorSpectrumPlotter", "SpectrumStyle", "HMarker"]


GroupKey = str | Callable[[SigmondSampling], Hashable]
FlatGroups = Mapping[Hashable, SingleEnsembleEnergyCollection]
NestedGroups = Mapping[Hashable, FlatGroups]
LabelFn = Callable[[Hashable], str]
ExcludedPredicate = Callable[[SigmondSampling], bool]
ExcludedSpec = Iterable[tuple[int, str, int]]
ExcludedLevels = ExcludedPredicate | ExcludedSpec


@dataclass
class HMarker:
    """A horizontal marker line on the spectrum plot.

    Args:
        y: Vertical position (energy value).
        label: Text drawn at the right edge of the line. LaTeX allowed.
        group: Outer group key this marker belongs to. When set, the line
            spans only the columns of that outer group. When None, the
            line spans the full plot width.
        color: Line color.
        linestyle: Matplotlib linestyle string.
        linewidth: Line width in points.
        alpha: Line and label transparency.
    """

    y: float
    label: str | None = None
    group: Hashable | None = None
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.0
    alpha: float = 0.7


@dataclass
class SpectrumStyle:
    """Styling configuration for SpectrumPlotter.

    Any value left as None falls back to the corresponding matplotlib rcParam
    where applicable.
    """

    col_spacing: float = 1.0
    outer_gap: float = 1.5
    stack_width: float = 0.3
    capsize: float = 5.0
    markersize: float = 6.0
    capthick: float = 1.5
    linewidth: float = 1.5
    fontsize: float | None = None
    excluded_color: str = "lightgrey"
    outer_label_offset: float = -50.0
    grid_alpha: float = 0.3
    ypad_frac: float = 0.05
    color_cycle: Iterable[str] | None = None
    errorbar_kwargs: dict[str, Any] = field(default_factory=dict)
    marker_label_fontsize: float | None = None
    marker_label_pad: float = 4.0
    marker_edge_pad: float = 0.3


class SpectrumPlotter:
    """
    Plot an energy spectrum as columns of error bars.

    Groupings may be supplied either via a `group_by` key (single key for flat
    columns, or a pair of keys for nested outer/inner grouping) or explicitly
    as a mapping of collections.

    Args:
        collection: SingleEnsembleEnergyCollection to plot.
        style: Optional SpectrumStyle override.
        excluded_levels: Levels rendered in `style.excluded_color` without
            disturbing the color cycle. Accepts either a predicate
            `f(sampling) -> bool`, or an iterable of
            `(psq, irrep, level_index)` tuples matched against each
            sampling's observable_info. Spec entries that don't match any
            sampling in the collection are silently ignored.
    """

    def __init__(
        self,
        collection: SingleEnsembleEnergyCollection,
        *,
        style: SpectrumStyle | None = None,
        excluded_levels: ExcludedLevels | None = None,
    ):
        if not isinstance(collection, SingleEnsembleEnergyCollection):
            raise TypeError(
                "SpectrumPlotter requires a SingleEnsembleEnergyCollection, "
                f"got {type(collection).__name__}"
            )
        self.collection = collection
        self.style = style or SpectrumStyle()
        self._is_excluded = self._build_excluded_predicate(excluded_levels)

    @staticmethod
    def _build_excluded_predicate(
        excluded_levels: ExcludedLevels | None,
    ) -> ExcludedPredicate:
        if excluded_levels is None:
            return lambda _s: False
        if callable(excluded_levels):
            return excluded_levels
        spec_set = set()
        for entry in excluded_levels:
            psq, irrep, level = entry
            spec_set.add((psq, irrep, level))

        def _match(sampling: SigmondSampling) -> bool:
            obs = sampling.observable_info
            return (obs.psq, obs.irrep, obs.level_index) in spec_set

        return _match

    def plot(
        self,
        *,
        group_by: GroupKey | tuple[GroupKey, GroupKey] | None = None,
        groups: FlatGroups | NestedGroups | None = None,
        outer_sort_key: Callable[[Hashable], Any] | None = None,
        inner_sort_key: Callable[[Hashable], Any] | None = None,
        column_label_fn: LabelFn | None = None,
        outer_label_fn: LabelFn | None = None,
        markers: list[HMarker] | None = None,
        energy_type: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> plt.Axes:
        """
        Plot the spectrum.

        Exactly one of `group_by` or `groups` must be provided.

        Args:
            group_by: Attribute name / callable for a flat grouping, or a
                two-tuple of keys for nested (outer, inner) grouping.
            groups: Explicit mapping. Flat: {column_label -> collection}.
                Nested: {outer_label -> {column_label -> collection}}.
            outer_sort_key: Sort function for outer group keys.
            inner_sort_key: Sort function for column keys within an outer group.
            column_label_fn: Map column key to xtick label (LaTeX allowed).
            outer_label_fn: Map outer key to annotation label below columns.
            markers: Horizontal marker lines. Each HMarker with
                ``group=None`` spans the full plot; those with a group key
                span only the columns of that outer group.
            energy_type: Restrict y-axis label to one energy type when the
                collection mixes types.
            ax: Axes to draw on (creates figure if None).
            figsize: Figure size (adaptive default if None).

        Returns:
            Matplotlib Axes object.
        """
        if (group_by is None) == (groups is None):
            raise ValueError("Provide exactly one of `group_by` or `groups`")

        nested = self._resolve_groups(
            group_by=group_by,
            groups=groups,
            outer_sort_key=outer_sort_key,
            inner_sort_key=inner_sort_key,
        )

        columns, outer_midpoints, outer_ranges, x_end = self._layout(nested)

        if ax is None:
            fig_w = max(6.0, len(columns) * 1.2)
            figsize = figsize or (fig_w, 6.0)
            _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        self._seed_axes_limits(ax, columns, x_end)
        self._draw_columns(ax, columns)
        self._draw_xticks(ax, columns, column_label_fn)
        self._draw_outer_labels(ax, outer_midpoints, outer_label_fn)
        if markers:
            self._draw_markers(ax, markers, outer_ranges, x_end)
        self._set_ylabel(ax, energy_type)

        ax.set_xlim(-self.style.col_spacing, x_end)
        ax.grid(True, alpha=self.style.grid_alpha, axis="y")

        return ax

    def _resolve_groups(
        self,
        *,
        group_by: GroupKey | tuple[GroupKey, GroupKey] | None,
        groups: FlatGroups | NestedGroups | None,
        outer_sort_key: Callable[[Hashable], Any] | None,
        inner_sort_key: Callable[[Hashable], Any] | None,
    ) -> list[tuple[Hashable, list[tuple[Hashable, SingleEnsembleEnergyCollection]]]]:
        """Normalize inputs to an ordered [(outer_key, [(col_key, coll), ...])]."""
        if groups is not None:
            first_val = next(iter(groups.values()))
            if isinstance(first_val, Mapping):
                nested_raw = groups
            else:
                nested_raw = {None: groups}
        else:
            if isinstance(group_by, tuple) and len(group_by) == 2:
                outer_key, inner_key = group_by
                outer_groups = self.collection.group_by(key=outer_key)
                nested_raw = {ok: sub.group_by(key=inner_key) for ok, sub in outer_groups.items()}
            else:
                nested_raw = {None: self.collection.group_by(key=group_by)}

        out: list[tuple[Hashable, list[tuple[Hashable, SingleEnsembleEnergyCollection]]]] = []
        outer_keys = list(nested_raw.keys())
        try:
            outer_keys = (
                sorted(outer_keys, key=outer_sort_key) if outer_sort_key else sorted(outer_keys)
            )
        except TypeError:
            pass
        for ok in outer_keys:
            inner_map = nested_raw[ok]
            inner_keys = list(inner_map.keys())
            try:
                inner_keys = (
                    sorted(inner_keys, key=inner_sort_key) if inner_sort_key else sorted(inner_keys)
                )
            except TypeError:
                pass
            out.append((ok, [(ik, inner_map[ik]) for ik in inner_keys]))
        return out

    def _layout(
        self,
        nested: list[tuple[Hashable, list[tuple[Hashable, SingleEnsembleEnergyCollection]]]],
    ) -> tuple[
        list[tuple[Hashable, Hashable, float, SingleEnsembleEnergyCollection]],
        dict[Hashable, float],
        dict[Hashable, tuple[float, float]],
        float,
    ]:
        columns: list[tuple[Hashable, Hashable, float, SingleEnsembleEnergyCollection]] = []
        outer_midpoints: dict[Hashable, float] = {}
        outer_ranges: dict[Hashable, tuple[float, float]] = {}
        x = 0.0
        for i, (outer_key, inner) in enumerate(nested):
            if i > 0:
                x += self.style.outer_gap
            start = x
            for col_key, sub in inner:
                columns.append((outer_key, col_key, x, sub))
                x += self.style.col_spacing
            end = x - self.style.col_spacing
            outer_midpoints[outer_key] = (start + end) / 2
            outer_ranges[outer_key] = (start, end)
        return columns, outer_midpoints, outer_ranges, x

    def _seed_axes_limits(self, ax, columns, x_end) -> None:
        all_ys = np.array([s.mean for _, _, _, sub in columns for s in sub])
        all_yerrs = np.array([s.error for _, _, _, sub in columns for s in sub])
        if len(all_ys) == 0:
            return
        y_lo = float(np.min(all_ys - all_yerrs))
        y_hi = float(np.max(all_ys + all_yerrs))
        pad = self.style.ypad_frac * (y_hi - y_lo) if y_hi > y_lo else 1.0
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlim(-self.style.col_spacing, x_end)

    def _color_cycle(self):
        from itertools import cycle

        if self.style.color_cycle is not None:
            return cycle(self.style.color_cycle)
        try:
            from kbfit import COLORS, IndexedCycle

            return IndexedCycle(COLORS)
        except ImportError:
            return cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    def _draw_columns(self, ax, columns) -> None:
        eb_kwargs = {
            "fmt": "o",
            "capsize": self.style.capsize,
            "capthick": self.style.capthick,
            "linewidth": self.style.linewidth,
            "markersize": self.style.markersize,
        }
        eb_kwargs.update(self.style.errorbar_kwargs)

        for _, _, x_center, sub in columns:
            ys = np.array([s.mean for s in sub])
            yerrs = np.array([s.error for s in sub])
            xs = stacked_positions(
                ys,
                yerrs,
                x=x_center,
                width=self.style.stack_width,
                markersize=self.style.markersize,
                ax=ax,
            )
            color_cycle = self._color_cycle()
            for xi, yi, yerri, samp in zip(xs, ys, yerrs, sub):
                cycled = next(color_cycle)
                color = self.style.excluded_color if self._is_excluded(samp) else cycled
                ax.errorbar(xi, yi, yerr=yerri, color=color, **eb_kwargs)

    def _draw_xticks(self, ax, columns, column_label_fn) -> None:
        label_fn = column_label_fn or self._default_column_label
        positions = [x for _, _, x, _ in columns]
        labels = [label_fn(col_key) for _, col_key, _, _ in columns]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=self.style.fontsize)

    def _draw_outer_labels(self, ax, outer_midpoints, outer_label_fn) -> None:
        if all(k is None for k in outer_midpoints):
            return
        label_fn = outer_label_fn or self._default_outer_label
        fs = self.style.fontsize if self.style.fontsize is not None else 10
        for outer_key, mid_x in outer_midpoints.items():
            ax.annotate(
                label_fn(outer_key),
                xy=(mid_x, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, self.style.outer_label_offset),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=fs,
            )

    def _draw_markers(
        self,
        ax: plt.Axes,
        markers: list[HMarker],
        outer_ranges: dict[Hashable, tuple[float, float]],
        x_end: float,
    ) -> None:
        pad = self.style.marker_edge_pad
        label_fs = self.style.marker_label_fontsize or self.style.fontsize
        label_pad = self.style.marker_label_pad

        for m in markers:
            if m.group is not None:
                if m.group not in outer_ranges:
                    continue
                x_start, x_stop = outer_ranges[m.group]
                x_start -= pad
                x_stop += pad
            else:
                x_start = -self.style.col_spacing + pad
                x_stop = x_end - self.style.col_spacing + pad

            ax.hlines(
                m.y,
                x_start,
                x_stop,
                colors=m.color,
                linestyles=m.linestyle,
                linewidths=m.linewidth,
                alpha=m.alpha,
                zorder=1,
            )

            if m.label is not None:
                ax.annotate(
                    m.label,
                    xy=(x_stop, m.y),
                    xytext=(label_pad, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=label_fs,
                    color=m.color,
                    alpha=m.alpha,
                )

    def _set_ylabel(self, ax, energy_type: str | None) -> None:
        energy_types = self.collection.energy_types
        if energy_type is not None:
            if energy_type not in energy_types:
                raise ValueError(f"energy_type '{energy_type}' not in collection")
            energy_types = [energy_type]

        if len(energy_types) == 1:
            latex = self.collection[0].observable_info.specify_latex_str(
                include_irrep=False,
                include_psq=False,
                include_particles=False,
                include_level_index=False,
            )
            ax.set_ylabel(f"${latex}$", fontsize=self.style.fontsize)
        else:
            import logging

            logging.warning(
                f"Multiple energy types found ({energy_types}), using generic 'E' label"
            )
            ax.set_ylabel("E", fontsize=self.style.fontsize)

    @staticmethod
    def _default_column_label(key: Hashable) -> str:
        return str(key)

    @staticmethod
    def _default_outer_label(key: Hashable) -> str:
        return str(key)


class SectorSpectrumPlotter(SpectrumPlotter):
    """
    Preset plotter grouping by PSQ (outer) then irrep (inner).

    Reproduces the original `plot_spectrum` layout with PSQ annotations below
    the irrep columns.
    """

    def plot(
        self,
        *,
        markers: list[HMarker] | None = None,
        energy_type: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> plt.Axes:
        return super().plot(
            group_by=("psq", "irrep"),
            column_label_fn=self._irrep_label,
            outer_label_fn=self._psq_label,
            markers=markers,
            energy_type=energy_type,
            ax=ax,
            figsize=figsize,
        )

    @staticmethod
    def _psq_label(psq: Hashable) -> str:
        return f"$d^2 = {psq}$"

    @staticmethod
    def _irrep_label(irrep: Hashable) -> str:
        try:
            from kbfit import get_irrep_latex_str

            return f"${get_irrep_latex_str(irrep)}$"
        except ImportError:
            return str(irrep)
