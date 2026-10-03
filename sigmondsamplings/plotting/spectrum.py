"""
Spectrum plotting for energy-level collections.

Provides a base `SpectrumPlotter` that renders columns of energy levels grouped
either via a `group_by` key (attribute name or callable) or via explicit
user-supplied collections. A preset `SectorSpectrumPlotter` reproduces the
PSQ -> irrep layout.

Ported onto :mod:`slat.plotting`, so the same plot renders on matplotlib or
plotly. On plotly each level carries hover text naming its irrep, momentum and
level index.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

import slat.plotting as slp
from slat import get_irrep_latex_str
from slat.plotting.style import resolve_colors

from ..energy_level_collection import SingleEnsembleEnergyCollection
from ..plot_contracts import SpectrumOptions, SpectrumStyle
from ..sampling import SigmondSampling
from ..utils import stacked_positions
from ._util import energy_level_hover_text, full_sample_and_bounds, resolve_axes
from .markers import HMarker

__all__ = ["HMarker", "SectorSpectrumPlotter", "SpectrumPlotter", "SpectrumStyle", "PreparedSpectrum", "prepare_spectrum", "draw_spectrum"]


GroupKey = str | Callable[[SigmondSampling], Hashable]
FlatGroups = Mapping[Hashable, SingleEnsembleEnergyCollection]
NestedGroups = Mapping[Hashable, FlatGroups]
LabelFn = Callable[[Hashable], str]
ExcludedPredicate = Callable[[SigmondSampling], bool]
ExcludedSpec = Iterable[tuple[int, str, int]]
ExcludedLevels = ExcludedPredicate | ExcludedSpec


@dataclass(frozen=True)
class PreparedLevel:
    central: float
    bounds: tuple[float, float]
    color_key: int
    excluded: bool
    hover: str


@dataclass(frozen=True)
class PreparedSpectrum:
    """Backend-independent numerical payload and resolved drawing decisions."""
    style: SpectrumStyle
    columns: tuple
    levels: tuple[tuple[PreparedLevel, ...], ...]
    outer_midpoints: dict
    outer_ranges: dict
    x_end: float
    x_range: tuple[float, float]
    energy_range: tuple[float, float]
    ylabel: str
    statistics: dict
    markers: tuple[HMarker, ...]
    column_label_fn: Any
    outer_label_fn: Any



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
        self.style = SpectrumStyle.model_validate(style or {})
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
            psq, irrep, level_or_levels = entry

            # Check if the third element is a collection of levels (like a list)
            if isinstance(level_or_levels, list | tuple | set):
                for level in level_or_levels:
                    spec_set.add((psq, irrep, level))
            else:
                # Fallback for a single integer level index
                spec_set.add((psq, irrep, level_or_levels))

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
        error_metric: str | None = None,
        confidence_level: float | None = None,
        percentiles: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
    ) -> slp.Axes:
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
                span only the columns of that outer group, and those with a
                ``column`` key as well span only that one column. Markers whose
                group or column is not drawn are dropped.
            energy_type: Select this energy type before plotting. All plotted
                levels must share an energy type and reference normalization.
            error_metric: 'error' or 'confidence_interval'; defaults to rc.
                Central points always use the full-sample value.
            confidence_level: Equal-tail percentile interval, e.g. 0.68.
            percentiles: Explicit percentile bounds, e.g. (16, 84), computed
                from bootstrap resamples. Exclusive with confidence_level.
            xlim, ylim: Explicit limits. Otherwise existing explicit axes limits
                are preserved; automatic y limits include marker lines/bands.
            axes: Axes to draw on (creates a figure if None).
            figsize: Figure size (adaptive default if None).
            backend: Backend for this call (uses the active one if None).
            ax: Deprecated alias for ``axes``.

        Returns:
            The Axes drawn on.
        """
        # Preserve explicit limits of caller-supplied axes before preparation.
        existing = axes if axes is not None else ax
        if existing is not None:
            existing = slp.wrap(existing)
            xlim = xlim if xlim is not None else existing.get_limits("x", explicit_only=True)
            ylim = ylim if ylim is not None else existing.get_limits("y", explicit_only=True)
        prepared = self.prepare(
            group_by=group_by, groups=groups, outer_sort_key=outer_sort_key,
            inner_sort_key=inner_sort_key, column_label_fn=column_label_fn,
            outer_label_fn=outer_label_fn, markers=markers,
            options=SpectrumOptions(energy_type=energy_type, error_metric=error_metric,
                                    confidence_level=confidence_level, percentiles=percentiles),
            xlim=xlim, ylim=ylim,
        )
        return draw_spectrum(prepared, axes=axes, ax=ax, figsize=figsize, backend=backend)

    def prepare(self, *, options=None, group_by=None, groups=None,
                outer_sort_key=None, inner_sort_key=None, column_label_fn=None,
                outer_label_fn=None, markers=None, xlim=None, ylim=None) -> PreparedSpectrum:
        """Compute statistics, hover data and final range once, without a figure."""
        options = SpectrumOptions.model_validate(options or {})
        if (group_by is None) == (groups is None):
            raise ValueError("Provide exactly one of `group_by` or `groups`")
        nested = self._resolve_groups(group_by=group_by, groups=groups,
                                      outer_sort_key=outer_sort_key, inner_sort_key=inner_sort_key)
        nested = self._select_energy_type(nested, options.energy_type)
        columns, midpoints, ranges, x_end = self._layout(nested)
        column_ranges = {(outer, col): (x, x) for outer, col, x, _ in columns}
        markers = tuple(m for m in markers or () if self._marker_range(m, ranges, column_ranges, x_end) is not None)
        statistics = options.resolved()
        statistics.pop("energy_type", None)
        statistics.pop("hover_confidence_level", None)
        if options.hover_confidence_level is not None and statistics["error_metric"] != "error":
            raise ValueError("hover_confidence_level applies only to Gaussian errors; interval plots reuse their bounds")
        hover_confidence = None
        if statistics["error_metric"] == "error":
            from ..rcparams import rc
            hover_confidence = options.hover_confidence_level or rc["plot.confidence_level"]
        values = []
        for _, _, _, sub in columns:
            levels = []
            for sampling in sub:
                central, bounds = full_sample_and_bounds(sampling, **statistics)
                # CI bounds double as hover percentiles; Gaussian bounds do not.
                hover = energy_level_hover_text(sampling,
                    percentiles=statistics.get("percentiles"),
                    confidence_level=statistics.get("confidence_level", hover_confidence),
                    interval_bounds=bounds if statistics["error_metric"] == "confidence_interval" else None)
                levels.append(PreparedLevel(central, bounds,
                    sampling.observable_info.level_index or 0, self._is_excluded(sampling), hover))
            values.append(tuple(levels))
        y_range = ylim if ylim is not None else self.style.ylim
        if y_range is None:
            ys = [v for column in values for level in column for v in (level.central, *level.bounds)]
            for marker in markers:
                ys.append(marker.y)
                if marker.band is not None:
                    ys.extend(marker.band)
            lo, hi = min(ys), max(ys)
            pad = self.style.ypad_frac * (hi - lo) if hi > lo else 1.0
            y_range = (lo-pad, hi+pad)
        return PreparedSpectrum(self.style.model_copy(deep=True), tuple(columns), tuple(values),
            midpoints, ranges, x_end, xlim or self.style.xlim or (-self.style.col_spacing, x_end),
            tuple(y_range), self._ylabel([s for _, _, _, sub in columns for s in sub]),
            {"central": "full_sample", **statistics, **({"hover_confidence_level": hover_confidence} if hover_confidence is not None else {})}, markers,
            column_label_fn, outer_label_fn)


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
            if not groups:
                raise ValueError("No energy levels to plot")
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

    @staticmethod
    def _select_energy_type(nested, energy_type):
        selected = []
        for outer_key, inner in nested:
            columns = []
            for key, sub in inner:
                if not isinstance(sub, SingleEnsembleEnergyCollection):
                    raise TypeError("Every spectrum group must be a SingleEnsembleEnergyCollection")
                if energy_type is not None:
                    sub = sub.filter(energy_type=energy_type)
                if len(sub):
                    columns.append((key, sub))
            if columns:
                selected.append((outer_key, columns))
        if not selected:
            raise ValueError(f"No energy levels to plot for energy_type={energy_type!r}")
        return selected

    @staticmethod
    def _ylabel(samplings):
        quantities = {
            (s.observable_info.energy_type, s.observable_info.ref_particle) for s in samplings
        }
        if len(quantities) != 1:
            raise ValueError(
                "Spectrum levels must share one energy type and reference normalization; "
                "select/filter compatible data before plotting"
            )
        obs = samplings[0].observable_info
        if obs.energy_type is None:
            raise ValueError("Spectrum levels must specify their energy type")
        latex = obs.specify_latex_str(
            include_irrep=False,
            include_psq=False,
            include_particles=False,
            include_level_index=False,
            is_single_hadron=False,
        )
        return f"${latex}$"

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
        for outer_key, inner in nested:
            start = x
            for col_key, sub in inner:
                columns.append((outer_key, col_key, x, sub))
                x += self.style.col_spacing
            end = x - self.style.col_spacing
            outer_midpoints[outer_key] = (start + end) / 2
            outer_ranges[outer_key] = (start, end)
        return columns, outer_midpoints, outer_ranges, x

    def _color_cycle(self):
        return resolve_colors(self.style.color_cycle, slp.get_style())

    def _draw_columns(self, axes: slp.Axes, columns, prepared) -> None:
        marker_extent = axes.marker_extent(self.style.markersize)
        colors = self._color_cycle().items

        for (_, _, x_center, sub), values in zip(columns, prepared):
            ys = np.array([level.central for level in values])
            bounds = np.array([level.bounds for level in values])
            # Adapt exact asymmetric intervals to the existing symmetric stacking
            # helper without changing its lane-assignment algorithm.
            lo = np.minimum(bounds[:, 0], ys)
            hi = np.maximum(bounds[:, 1], ys)
            xs = stacked_positions(
                (lo + hi) / 2,
                (hi - lo) / 2,
                x=x_center,
                width=self.style.stack_width,
                marker_extent=marker_extent,
            )
            for xi, level in zip(xs, values):
                yi, (lower, upper) = level.central, level.bounds
                cycled = colors[level.color_key % len(colors)]
                color = self.style.excluded_color if level.excluded else cycled
                hover = [level.hover]
                if not lower <= yi <= upper:
                    # A percentile interval need not contain the full-sample value.
                    # Draw its exact endpoints separately instead of clipping it
                    # or passing negative errors to a backend.
                    axes.vlines(xi, lower, upper, color=color, width=self.style.linewidth)
                    cap = abs(axes.point_size(self.style.capsize, "x"))
                    axes.hlines(
                        [lower, upper], xi - cap, xi + cap, color=color, width=self.style.linewidth
                    )
                    axes.points(
                        [xi], [yi], color=color, size=self.style.markersize, marker="o", hover=hover
                    )
                    continue
                axes.errorbar(
                    xi,
                    yi,
                    yerr=np.array([[yi - lower], [upper - yi]]),
                    color=color,
                    marker="o",
                    capsize=self.style.capsize,
                    width=self.style.linewidth,
                    markersize=self.style.markersize,
                    hover=hover,
                    native=self.style.errorbar_kwargs or None,
                )

    def _draw_xticks(self, axes: slp.Axes, columns, column_label_fn) -> None:
        label_fn = column_label_fn or self._default_column_label
        positions = [x for _, _, x, _ in columns]
        labels = [label_fn(col_key) for _, col_key, _, _ in columns]
        fs = self.style.column_label_fontsize
        if fs is None:
            fs = self.style.fontsize
        axes.ticks("x", positions, labels, rotation=45, fontsize=fs)

    def _draw_outer_labels(self, axes: slp.Axes, outer_midpoints, outer_label_fn) -> None:
        if all(k is None for k in outer_midpoints):
            return
        label_fn = outer_label_fn or self._default_outer_label
        fs = self.style.outer_label_fontsize
        if fs is None:
            fs = self.style.fontsize if self.style.fontsize is not None else 10
        # The offset is in points below the axes; convert once to axes fraction
        # so the label lands in the same place on either backend.
        y = axes.point_size(self.style.outer_label_offset, "y", units="axes")
        for outer_key, mid_x in outer_midpoints.items():
            axes.text(
                mid_x,
                y,
                label_fn(outer_key),
                # x in data units so the label tracks its columns, y below the axes.
                coords=("data", "axes"),
                ha="center",
                va="top",
                fontsize=fs,
            )

    def _draw_group_separators(self, axes: slp.Axes, outer_ranges) -> None:
        """Divide adjacent outer groups halfway between their columns."""
        ranges = list(outer_ranges.values())
        ymin = -abs(axes.point_size(self.style.group_separator_tick_length, "y", units="axes"))
        for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
            axes.axvline(
                (previous_end + next_start) / 2,
                ymin=ymin,
                color=self.style.group_separator_color,
                width=self.style.group_separator_linewidth,
                style=self.style.group_separator_linestyle,
                zorder=0,
            )

    def _marker_range(
        self,
        m: HMarker,
        outer_ranges: dict[Hashable, tuple[float, float]],
        column_ranges: dict[tuple[Hashable, Hashable], tuple[float, float]],
        x_end: float,
    ) -> tuple[float, float] | None:
        """The x span a marker covers, or None when its group/column is not drawn."""
        pad = self.style.marker_edge_pad
        if m.column is not None:
            span = column_ranges.get((m.group, m.column))
        elif m.group is not None:
            span = outer_ranges.get(m.group)
        else:
            return (-self.style.col_spacing + pad, x_end - self.style.col_spacing + pad)
        if span is None:
            return None
        return (span[0] - pad, span[1] + pad)

    def _draw_markers(
        self,
        axes: slp.Axes,
        markers: list[HMarker],
        outer_ranges: dict[Hashable, tuple[float, float]],
        column_ranges: dict[tuple[Hashable, Hashable], tuple[float, float]],
        x_end: float,
    ) -> None:
        label_fs = self.style.marker_label_fontsize or self.style.fontsize
        label_offset = axes.point_size(self.style.marker_label_pad, "x", units="data")

        for m in markers:
            span = self._marker_range(m, outer_ranges, column_ranges, x_end)
            if span is None:
                continue
            x_start, x_stop = span

            band = m.band
            if band is not None:
                axes.band(
                    [x_start, x_stop],
                    [band[0], band[0]],
                    [band[1], band[1]],
                    color=m.color,
                    alpha=m.band_alpha,
                    zorder=0,
                )
            axes.hlines(
                m.y,
                x_start,
                x_stop,
                color=m.color,
                style=m.linestyle,
                width=m.linewidth,
                alpha=m.alpha,
                zorder=1,
                hover=m.hover_text,
            )

            if m.label is not None:
                axes.text(
                    x_stop + label_offset,
                    m.y,
                    m.label,
                    ha="left",
                    va="center",
                    fontsize=label_fs,
                    color=m.color,
                    alpha=m.alpha,
                )

    @staticmethod
    def _default_column_label(key: Hashable) -> str:
        return str(key)

    @staticmethod
    def _default_outer_label(key: Hashable) -> str:
        return str(key)


class SectorSpectrumPlotter(SpectrumPlotter):
    """
    Preset plotter grouping by PSQ (outer) then irrep (inner).

    Irrep columns are evenly spaced, with vertical separators between PSQ
    groups and PSQ annotations below the columns.
    """

    def plot(
        self,
        *,
        markers: list[HMarker] | None = None,
        energy_type: str | None = None,
        error_metric: str | None = None,
        confidence_level: float | None = None,
        percentiles: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        axes: slp.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        ax: Any = None,
    ) -> slp.Axes:
        return super().plot(
            group_by=("psq", "irrep"),
            column_label_fn=self._irrep_label,
            outer_label_fn=self._psq_label,
            markers=markers,
            energy_type=energy_type,
            error_metric=error_metric,
            confidence_level=confidence_level,
            percentiles=percentiles,
            xlim=xlim,
            ylim=ylim,
            axes=axes,
            figsize=figsize,
            backend=backend,
            ax=ax,
        )

    @staticmethod
    def _psq_label(psq: Hashable) -> str:
        return f"$d^2 = {psq}$"

    @staticmethod
    def _irrep_label(irrep: Hashable) -> str:
        return f"${get_irrep_latex_str(irrep)}$"


def prepare_spectrum(collection, *, options=None, style=None, excluded_levels=None,
                     overlays=None, group_by=("psq", "irrep"), **kwargs) -> PreparedSpectrum:
    """Prepare a sector spectrum; ordinary overlays participate in auto limits."""
    plotter = SectorSpectrumPlotter(collection, style=style, excluded_levels=excluded_levels)
    return plotter.prepare(options=options, group_by=group_by, markers=overlays,
                           column_label_fn=plotter._irrep_label,
                           outer_label_fn=plotter._psq_label, **kwargs)


def draw_spectrum(prepared: PreparedSpectrum, *, overlays=None, axes=None, ax=None,
                  figsize=None, backend=None) -> slp.Axes:
    """Draw prepared values. Additional overlays never expand the fixed range.

    Pass ordinary range-expanding overlays to preparation; pass range-derived
    markers (such as NI levels) here. No replacement style is accepted.
    """
    # Drawing uses only prepared payloads, never accesses the level statistics.
    plotter = object.__new__(SpectrumPlotter)
    plotter.style = prepared.style
    created = axes is None and ax is None
    axes = resolve_axes(axes, ax,
                        figsize=figsize or (max(6., len(prepared.columns)*1.2), 6.), backend=backend)
    try:
        axes.set(xlim=prepared.x_range, ylim=prepared.energy_range)
        plotter._draw_columns(axes, prepared.columns, prepared.levels)
        plotter._draw_xticks(axes, prepared.columns, prepared.column_label_fn)
        plotter._draw_group_separators(axes, prepared.outer_ranges)
        plotter._draw_outer_labels(axes, prepared.outer_midpoints, prepared.outer_label_fn)
        columns = {(outer, col): (x,x) for outer, col, x, _ in prepared.columns}
        plotter._draw_markers(axes, [*prepared.markers, *(overlays or ())], prepared.outer_ranges, columns, prepared.x_end)
        axes.set(ylabel=prepared.ylabel, xlim=prepared.x_range, ylim=prepared.energy_range)
        axes.grid(True, alpha=prepared.style.grid_alpha, axis="y")
        return axes
    except BaseException:
        if created:
            axes.figure.close()
        raise
