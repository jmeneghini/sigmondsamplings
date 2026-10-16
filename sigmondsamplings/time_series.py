"""Computed time-series views; cells and coordinates remain the source of truth."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import xarray as xr

from .metadata import source_elements


def _numeric(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


@dataclass(frozen=True)
class TimeAxis:
    _view: TimeSeriesView

    def _common(self, field):
        elements = self._view.elements
        data = self._view._data
        axis = data.dims.index("time")
        infos = np.moveaxis(data.infos, axis, -1)
        values = []
        for index in np.ndindex(elements.shape):
            element = elements.values[index]
            if field == "offset":
                value = getattr(element, "time_offset", 0.0)
                if value is None:
                    raise ValueError("conflicting source time offsets in derived series")
            else:
                value = getattr(element, "time_extent", infos[index + (0,)].ensemble_info.temporal_extent)
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError(f"no series element establishes an axis {field}")
        if len(values) != 1:
            raise ValueError(f"conflicting time axis {field} values: {values!r}")
        return values[0]

    @property
    def offset(self) -> float:
        return self._common("offset")

    @property
    def extent(self) -> int | None:
        return self._common("extent")


class TimeSeriesView:
    """A live, computed view of a named numeric time dimension.

    No axis state or element cache is attached to the data. Source times stay
    unchanged by display offsets; numeric floating coordinates are preserved.
    """
    __slots__ = ("_data",)

    def __init__(self, data):
        if "time" not in data.dims or "time" not in data.coords:
            raise ValueError(".ts requires a named time dimension with numeric coordinates")
        times = data.coords["time"]
        if times.ndim != 1 or len(times) != data.shape[data.dims.index("time")]:
            raise ValueError("time coordinates must be one-dimensional and match the time axis")
        if not all(_numeric(value) for value in times):
            raise ValueError("time coordinates must be finite real numbers")
        if len(set(times.tolist())) != len(times):
            raise ValueError("time coordinates must be unique")
        self._data = data

    @property
    def times(self):
        return self._data.coords["time"].copy()

    @property
    def elements(self) -> xr.DataArray:
        data = self._data
        # Establish full element/source context along time before exposing the
        # actual at-time element (a frozen native correlator or metadata record).
        validated = source_elements(data)
        axis = data.dims.index("time")
        infos = np.moveaxis(data.infos, axis, -1)
        values = np.empty(validated.shape, dtype=object)
        for index in np.ndindex(values.shape):
            info = infos[index + (0,)]
            values[index] = getattr(info, "element", validated[index])
        dims = tuple(dim for dim in data.dims if dim != "time")
        coords = {dim: data.coords[dim] for dim in dims if dim in data.coords}
        return xr.DataArray(values, dims=dims, coords=coords, name="element")

    @property
    def axis(self) -> TimeAxis:
        return TimeAxis(self)

    def at(self, time):
        return self._data.sel(time=time)

    def window(self, start, stop):
        if not _numeric(start) or not _numeric(stop):
            raise ValueError("window bounds must be finite real numbers")
        if start > stop:
            raise ValueError("window bounds are reversed")
        times = self.times
        return self._data.isel(time=np.flatnonzero((times >= start) & (times <= stop)))

    def effective_energy(self, **kwargs):
        from .correlators.effective import effective_energy

        return effective_energy(self._data, **kwargs)

    def plot(self, *, x_offset=None, hue=None, labels=None, colors=None, hollow=None, style=None, **kwargs):
        from .plotting.series import SeriesPlotter

        return SeriesPlotter(self._data, x="time", x_offset=x_offset, hue=hue, labels=labels,
                             colors=colors, hollow=hollow, style=style).plot(**kwargs)
