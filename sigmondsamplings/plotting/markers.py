"""Overlay descriptors shared by the plot frontends.

An :class:`HMarker` is a horizontal reference level (a threshold, a known
energy, a fitted level) that draws the same way on a spectrum plot and on a
series plot such as an effective energy. A :class:`FitOverlay` puts a finished
fit on a series plot. Both are plain data; the plotters draw them.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..rcparams import rc
from ._util import energy_level_hover_text, full_sample_and_bounds

if TYPE_CHECKING:
    from ..data import SigmondData
    from ..fitting.result import FitResult

__all__ = ["FitOverlay", "HMarker"]


@dataclass
class HMarker:
    """A horizontal marker line, optionally with an error band.

    Args:
        y: Vertical position (energy value).
        label: Text drawn at the right edge of the line. LaTeX allowed.
        group: Outer group key this marker belongs to, on a spectrum plot.
            When set, the line spans only the columns of that outer group.
            When None, the line spans the full plot width. Series plots
            ignore it.
        column: Column key within ``group`` (e.g. an irrep under a ``psq``
            group). When set, the line spans only that one column; ``group``
            is then the column's outer key (None for a flat grouping). Series
            plots ignore it.
        color: Line and band color.
        linestyle: Line style string (``"-"``, ``"--"``, ``":"``, ``"-."``).
        linewidth: Line width in points.
        alpha: Line and label transparency.
        err: Draw a band behind the line: ``y ± err``, or ``(lower, upper)``
            offsets for an asymmetric band. Offsets may be signed when the
            line lies outside a percentile interval. None draws the line only.
        band_alpha: Band transparency.
        hover: Hover text for interactive backends, independent of the visible label.
            Overrides the automatic hover text of sampling-backed markers.
        ni_pair: Optional compact NI pair name for automatic sampling hover text.
    """

    y: float
    label: str | None = None
    group: Hashable | None = None
    column: Hashable | None = None
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.0
    alpha: float = 0.7
    err: float | tuple[float, float] | None = None
    band_alpha: float = 0.2
    hover: str | None = None
    sampling: SigmondData | None = field(default=None, repr=False, compare=False)
    ni_pair: str | None = None
    percentiles: tuple[float, float] | None = None
    confidence_level: float | None = None
    _automatic_hover: str | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_sampling(
        cls,
        sampling: SigmondData,
        *,
        error_metric: str | None = None,
        confidence_level: float | None = None,
        percentiles: tuple[float, float] | None = None,
        hover_confidence_level: float | None = None,
        **kwargs: Any,
    ) -> HMarker:
        """A marker at the full-sample value, with an uncertainty band.

        Errors default to ``rc["plot.error_metric"]``. ``percentiles=(16, 84)``
        selects explicit bootstrap/bin percentile bounds; ``confidence_level``
        selects an equal-tail interval. Neither changes the central value.
        Asymmetric offsets can be signed when the point is outside the interval,
        keeping the band at the exact percentile bounds. The sampling is retained
        for automatic PDG/percentile hover text; ``hover`` overrides it and
        ``ni_pair`` adds a compact NI pair name. ``kwargs`` set style.
        """
        if sampling.ndim != 0:
            raise ValueError(
                f"HMarker.from_sampling needs a single observable, got dims {sampling.dims}"
            )
        central, (lower, upper) = full_sample_and_bounds(
            sampling,
            error_metric=error_metric,
            confidence_level=confidence_level,
            percentiles=percentiles,
        )
        err = (central - lower, upper - central)
        if percentiles is None and confidence_level is None:
            if (error_metric or rc["plot.error_metric"]) == "error":
                err = float(sampling.error)
        marker = cls(
            y=float(central), err=err, sampling=sampling, percentiles=percentiles,
            confidence_level=confidence_level, **kwargs,
        )
        interval = percentiles is not None or confidence_level is not None or (error_metric or rc["plot.error_metric"]) == "confidence_interval"
        marker._automatic_hover = energy_level_hover_text(
            sampling, ni_pair=marker.ni_pair, percentiles=percentiles,
            confidence_level=confidence_level if interval else hover_confidence_level,
            interval_bounds=(lower, upper) if interval else None,
            irrep=marker.column, psq=marker.group, include_level=marker.ni_pair is None,
        )
        return marker

    @property
    def hover_text(self) -> str | None:
        """Explicit hover text or a summary derived from the retained sampling."""
        if self.hover is not None or self.sampling is None:
            return self.hover
        if self._automatic_hover is not None:
            return self._automatic_hover
        return energy_level_hover_text(
            self.sampling, ni_pair=self.ni_pair, percentiles=self.percentiles,
            confidence_level=self.confidence_level, irrep=self.column, psq=self.group,
            include_level=self.ni_pair is None,
        )

    @property
    def band(self) -> tuple[float, float] | None:
        """``(lo, hi)`` of the error band, or None without ``err``."""
        if self.err is None:
            return None
        if np.ndim(self.err) == 1:
            lower, upper = self.err
        else:
            lower = upper = self.err
        return (self.y - lower, self.y + upper)


@dataclass
class FitOverlay:
    """A finished fit drawn over a series plot.

    On a correlator plot it draws the fitted model (:attr:`FitResult.theory`)
    over the fitted ``x`` as a band plus a central line. On an effective-energy
    plot it draws the ``energy`` parameter as a flat band over the fit range,
    and the model's own effective energy (the same calculator applied to the
    theory) as a line, which shows the excited-state curvature the fit claims.

    Args:
        result: The fit. It must carry ``x`` (a fit built by
            ``SamplingFit.from_data``, or read back with its theory).
        energy: Name of the parameter drawn as a flat band on effective-energy
            plots. None draws only the model's effective energy there.
        label: Legend label for the model band.
        color: Band and line color. None takes the color of ``series``.
        series: Which series of the plot the fit belongs to, by label or
            position, for its color. None means the first.
        band_alpha: Band transparency.
    """

    result: FitResult
    energy: str | None = None
    label: str | None = None
    color: str | None = None
    series: int | str | None = None
    band_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.result.x is None or self.result.x_name is None:
            raise ValueError(
                "a FitOverlay needs a fit result with x values; build the fit with "
                "SamplingFit.from_data (from_model results have none)"
            )
        if self.result.theory is None:
            raise ValueError("a FitOverlay needs a fit result with theory predictions")
        if self.energy is not None and self.energy not in self.result.param_names:
            raise ValueError(
                f"energy={self.energy!r} is not a fit parameter; "
                f"parameters are {self.result.param_names}"
            )
