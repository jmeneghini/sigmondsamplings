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
from types import MappingProxyType
from typing import Any

import numpy as np
from pydantic import field_serializer

import slat.plotting as slp
from slat import get_irrep_latex_str
from slat.plotting.lifecycle import figure_factory
from slat.plotting.style import resolve_colors

from ..energy_level_collection import SingleEnsembleEnergyCollection
from ..plot_contracts import SpectrumOptions, SpectrumStyle
from ..sampling import SigmondSampling
from ..utils import stacked_positions
from ._util import _full_sample_and_bounds, energy_level_hover_text, resolve_axes
from .markers import HMarker, _PreparedMarker, _snapshot_marker, draw_hmarker

__all__ = [
    "HMarker",
    "PreparedSpectrum",
    "SectorSpectrumPlotter",
    "SpectrumPlotter",
    "SpectrumStyle",
    "draw_spectrum",
    "prepare_spectrum",
]


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


class _PreparedStyle(SpectrumStyle):
    model_config = {**SpectrumStyle.model_config, "frozen": True}

    @field_serializer("errorbar_kwargs")
    def _serialize_native(self, value):
        return _native_kwargs(value)


def _freeze(value):
    """Freeze drawing metadata only; sampling buffers never enter the payload."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = value.copy()
        copied.flags.writeable = False
        return copied
    return value


def _native_kwargs(value):
    if isinstance(value, Mapping):
        return {key: _native_kwargs(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_native_kwargs(item) for item in value)
    return value


@dataclass(frozen=True)
class PreparedSpectrum:
    """Immutable drawing values, styles and metadata, independent of source edits."""
    style: SpectrumStyle
    #: ``(outer_key, column_key, x)`` per drawn column, in drawing order.
    columns: tuple[tuple[Hashable, Hashable, float], ...]
    levels: tuple[tuple[PreparedLevel, ...], ...]
    outer_midpoints: Mapping
    outer_ranges: Mapping
    column_ranges: Mapping
    x_end: float
    x_range: tuple[float, float]
    energy_range: tuple[float, float]
    ylabel: str
    statistics: Mapping
    #: Range-expanding markers with the x span each covers.
    markers: tuple[tuple[_PreparedMarker, tuple[float, float]], ...]
    column_label_fn: LabelFn | None
    outer_label_fn: LabelFn | None


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

        # The third entry is one level index or a collection of them.
        spec_set = {
            (psq, irrep, level)
            for psq, irrep, levels in excluded_levels
            for level in (levels if isinstance(levels, list | tuple | set) else (levels,))
        }

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

        Returns:
            The Axes drawn on.
        """
        # Preserve explicit limits of caller-supplied axes before preparation.
        if axes is not None:
            axes = slp.wrap(axes)
            xlim = xlim if xlim is not None else axes.get_limits("x", explicit_only=True)
            ylim = ylim if ylim is not None else axes.get_limits("y", explicit_only=True)
        prepared = self.prepare(
            group_by=group_by, groups=groups, outer_sort_key=outer_sort_key,
            inner_sort_key=inner_sort_key, column_label_fn=column_label_fn,
            outer_label_fn=outer_label_fn, markers=markers,
            options=SpectrumOptions(energy_type=energy_type, error_metric=error_metric,
                                    confidence_level=confidence_level, percentiles=percentiles),
            xlim=xlim, ylim=ylim,
        )
        return draw_spectrum(prepared, axes=axes, figsize=figsize, backend=backend)

    def prepare(self, *, options=None, group_by=None, groups=None,
                outer_sort_key=None, inner_sort_key=None, column_label_fn=None,
                outer_label_fn=None, markers=None, xlim=None, ylim=None) -> PreparedSpectrum:
        """Compute statistics, hover data and final range once, without a figure."""
        options = SpectrumOptions.model_validate(options or {})
        statistics = options.resolved()
        statistics.pop("energy_type", None)
        statistics.pop("hover_confidence_level", None)
        interval = statistics["error_metric"] == "confidence_interval"
        if options.hover_confidence_level is not None and interval:
            raise ValueError("hover_confidence_level applies only to Gaussian errors; interval plots reuse their bounds")
        if (group_by is None) == (groups is None):
            raise ValueError("Provide exactly one of `group_by` or `groups`")
        nested = self._resolve_groups(group_by=group_by, groups=groups,
                                      outer_sort_key=outer_sort_key, inner_sort_key=inner_sort_key)
        nested = self._select_energy_type(nested, options.energy_type)
        columns, midpoints, ranges, x_end = self._layout(nested)
        column_ranges = {(outer, col): (x, x) for outer, col, x, _ in columns}
        placed = tuple(
            (_snapshot_marker(m), span)
            for m in markers or ()
            if (span := _marker_range(m, self.style, ranges, column_ranges, x_end)) is not None
        )
        hover_confidence = None
        if not interval:
            from ..rcparams import rc
            hover_confidence = options.hover_confidence_level or rc["plot.confidence_level"]
        values = []
        for _, _, _, sub in columns:
            levels = []
            for sampling in sub:
                central, bounds = _full_sample_and_bounds(sampling, statistics)
                # CI bounds double as hover percentiles; Gaussian bounds do not.
                hover = energy_level_hover_text(sampling,
                    percentiles=statistics.get("percentiles"),
                    confidence_level=statistics.get("confidence_level", hover_confidence),
                    interval_bounds=bounds if interval else None)
                levels.append(PreparedLevel(central, bounds,
                    sampling.observable_info.level_index or 0, self._is_excluded(sampling), hover))
            values.append(tuple(levels))
        y_range = ylim if ylim is not None else self.style.ylim
        if y_range is None:
            ys = [v for column in values for level in column for v in (level.central, *level.bounds)]
            for marker, _ in placed:
                ys.append(marker.y)
                if marker.band is not None:
                    ys.extend(marker.band)
            lo, hi = min(ys), max(ys)
            pad = self.style.ypad_frac * (hi - lo) if hi > lo else 1.0
            y_range = (lo-pad, hi+pad)
        recorded = {"central": "full_sample", **statistics}
        if hover_confidence is not None:
            recorded["hover_confidence_level"] = hover_confidence
        style = _PreparedStyle.model_validate(self.style.model_dump())
        object.__setattr__(style, "errorbar_kwargs", _freeze(style.errorbar_kwargs))
        return PreparedSpectrum(
            style=style,
            columns=tuple((outer, col, x) for outer, col, x, _ in columns),
            levels=tuple(values),
            outer_midpoints=_freeze(midpoints),
            outer_ranges=_freeze(ranges),
            column_ranges=_freeze(column_ranges),
            x_end=x_end,
            x_range=tuple(xlim if xlim is not None else
                          self.style.xlim or (-self.style.col_spacing, x_end)),
            energy_range=tuple(y_range),
            ylabel=self._ylabel([s for _, _, _, sub in columns for s in sub]),
            statistics=_freeze(recorded),
            markers=placed,
            column_label_fn=column_label_fn,
            outer_label_fn=outer_label_fn,
        )

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


class SectorSpectrumPlotter(SpectrumPlotter):
    """
    Preset plotter grouping by PSQ (outer) then irrep (inner).

    Irrep columns are evenly spaced, with vertical separators between PSQ
    groups and PSQ annotations below the columns.
    """

    def plot(self, **kwargs: Any) -> slp.Axes:
        """:meth:`SpectrumPlotter.plot` with the PSQ/irrep grouping and labels fixed."""
        return super().plot(
            group_by=("psq", "irrep"),
            column_label_fn=self._irrep_label,
            outer_label_fn=self._psq_label,
            **kwargs,
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


@figure_factory
def draw_spectrum(prepared: PreparedSpectrum, *, overlays=None, axes=None,
                  figsize=None, backend=None) -> slp.Axes:
    """Draw prepared values. Additional overlays never expand the fixed range.

    Pass ordinary range-expanding overlays to preparation; pass range-derived
    markers (such as NI levels) here. No replacement style is accepted.
    """
    # Overlays whose group or column is not drawn are skipped, as in preparation.
    placed = [
        (_snapshot_marker(m), span)
        for m in overlays or ()
        if (span := _marker_range(m, prepared.style, prepared.outer_ranges,
                                  prepared.column_ranges, prepared.x_end)) is not None
    ]
    axes = resolve_axes(axes,
                        figsize=figsize or (max(6., len(prepared.columns)*1.2), 6.), backend=backend)
    # Limits first: point_size and marker_extent depend on them.
    axes.set(xlim=prepared.x_range, ylim=prepared.energy_range)
    _draw_columns(axes, prepared)
    _draw_xticks(axes, prepared)
    _draw_group_separators(axes, prepared)
    _draw_outer_labels(axes, prepared)
    _draw_markers(axes, prepared.style, [*prepared.markers, *placed])
    axes.set(ylabel=prepared.ylabel)
    if axes.style.grid:
        # Vertical lines between irrep columns mean nothing; keep only the
        # horizontal ones.
        axes.grid(False, axis="x")
    return axes


def _fontsize(style: SpectrumStyle, specific: float | None) -> float | None:
    """A label-specific font size, else the style-wide one (None: backend default)."""
    return specific if specific is not None else style.fontsize


def _marker_range(
    m: HMarker,
    style: SpectrumStyle,
    outer_ranges: dict[Hashable, tuple[float, float]],
    column_ranges: dict[tuple[Hashable, Hashable], tuple[float, float]],
    x_end: float,
) -> tuple[float, float] | None:
    """The x span a marker covers, or None when its group/column is not drawn."""
    pad = style.marker_edge_pad
    if m.column is not None:
        span = column_ranges.get((m.group, m.column))
    elif m.group is not None:
        span = outer_ranges.get(m.group)
    else:
        return (-style.col_spacing + pad, x_end - style.col_spacing + pad)
    if span is None:
        return None
    return (span[0] - pad, span[1] + pad)


def _draw_columns(axes: slp.Axes, prepared: PreparedSpectrum) -> None:
    style = prepared.style
    marker_extent = axes.marker_extent(style.markersize)
    colors = resolve_colors(style.color_cycle, axes.style).items

    for (_, _, x_center), values in zip(prepared.columns, prepared.levels):
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
            width=style.stack_width,
            marker_extent=marker_extent,
        )
        for xi, level in zip(xs, values):
            yi, (lower, upper) = level.central, level.bounds
            cycled = colors[level.color_key % len(colors)]
            color = style.excluded_color if level.excluded else cycled
            hover = [level.hover]
            if not lower <= yi <= upper:
                # A percentile interval need not contain the full-sample value.
                # Draw its exact endpoints separately instead of clipping it
                # or passing negative errors to a backend.
                axes.vlines(xi, lower, upper, color=color, linewidth=style.linewidth)
                cap = abs(axes.point_size(style.capsize, "x"))
                axes.hlines(
                    [lower, upper], xi - cap, xi + cap, color=color, linewidth=style.linewidth
                )
                axes.points(
                    [xi], [yi], color=color, markersize=style.markersize, marker="o", hover=hover
                )
                continue
            axes.errorbar(
                xi,
                yi,
                yerr=np.array([[yi - lower], [upper - yi]]),
                color=color,
                marker="o",
                capsize=style.capsize,
                linewidth=style.linewidth,
                markersize=style.markersize,
                hover=hover,
                native=_native_kwargs(style.errorbar_kwargs) or None,
            )


def _draw_xticks(axes: slp.Axes, prepared: PreparedSpectrum) -> None:
    label_fn = prepared.column_label_fn or str
    positions = [x for _, _, x in prepared.columns]
    labels = [label_fn(col_key) for _, col_key, _ in prepared.columns]
    fs = _fontsize(prepared.style, prepared.style.column_label_fontsize)
    axes.ticks("x", positions, labels, rotation=45, fontsize=fs)


def _draw_outer_labels(axes: slp.Axes, prepared: PreparedSpectrum) -> None:
    if all(k is None for k in prepared.outer_midpoints):
        return
    style = prepared.style
    label_fn = prepared.outer_label_fn or str
    fs = _fontsize(style, style.outer_label_fontsize)
    # The offset is in points below the axes; convert once to axes fraction
    # so the label lands in the same place on either backend.
    y = axes.point_size(style.outer_label_offset, "y", units="axes")
    for outer_key, mid_x in prepared.outer_midpoints.items():
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


def _draw_group_separators(axes: slp.Axes, prepared: PreparedSpectrum) -> None:
    """Divide adjacent outer groups halfway between their columns."""
    style = prepared.style
    ranges = list(prepared.outer_ranges.values())
    ymin = -abs(axes.point_size(style.group_separator_tick_length, "y", units="axes"))
    for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
        axes.axvline(
            (previous_end + next_start) / 2,
            ymin=ymin,
            color=style.group_separator_color,
            linewidth=style.group_separator_linewidth,
            linestyle=style.group_separator_linestyle,
            zorder=0,
        )


def _draw_markers(
    axes: slp.Axes, style: SpectrumStyle, markers: list[tuple[HMarker, tuple[float, float]]]
) -> None:
    label_fs = _fontsize(style, style.marker_label_fontsize)
    label_offset = axes.point_size(style.marker_label_pad, "x", units="data")
    for m, (x_start, x_stop) in markers:
        draw_hmarker(
            axes,
            m,
            x_start,
            x_stop,
            x_stop + label_offset,
            ha="left",
            va="center",
            fontsize=label_fs,
        )
