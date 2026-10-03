"""Lightweight plot schemas and catalog shared by Python, SS CLI and consumers.

Importing this module does not load data, fitting, or plotting backends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

from pydantic import Field, model_validator

from slat.toml_config import StrictModel


class FigureStyle(StrictModel):
    figsize: tuple[float, float] | None = None
    dpi: float = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _size(self):
        if self.figsize is not None and min(self.figsize) <= 0:
            raise ValueError("figure.figsize must be positive")
        return self


class DiagnosticStyle(StrictModel):
    figure: FigureStyle = Field(default_factory=FigureStyle)


class EmptyOptions(StrictModel):
    pass


class HistogramOptions(StrictModel):
    sampling: int | None = Field(default=None, ge=0)
    bins: int | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"] = Field(
        default="auto", union_mode="left_to_right"
    )
    confidence_level: float = Field(default=0.68, gt=0, lt=1)
    show_bias: bool = False

    @model_validator(mode="after")
    def _bins(self):
        if isinstance(self.bins, int) and self.bins < 1:
            raise ValueError("bins must be positive")
        return self


class SummaryOptions(StrictModel):
    panels: tuple[Literal["errorbar", "correlation", "histogram", "eff_sample_size"], ...] = Field(
        default=("histogram", "correlation"), min_length=1
    )
    obs_index: int | None = Field(default=None, ge=0)


class BootstrapOptions(StrictModel):
    confidence_levels: tuple[float, ...] = (0.68, 0.95)

    @model_validator(mode="after")
    def _levels(self):
        if not self.confidence_levels or any(not 0 < level < 1 for level in self.confidence_levels):
            raise ValueError("confidence_levels must be nonempty and within (0, 1)")
        return self


class UncertaintyOptions(StrictModel):
    error_metric: Literal["error", "confidence_interval"] | None = None
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    percentiles: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _interval(self):
        if self.percentiles is not None and self.confidence_level is not None:
            raise ValueError("pass percentiles or confidence_level, not both")
        if (
            self.percentiles is not None
            and not 0 <= self.percentiles[0] < self.percentiles[1] <= 100
        ):
            raise ValueError("percentiles must be increasing within [0, 100]")
        if self.error_metric == "error" and (
            self.percentiles is not None or self.confidence_level is not None
        ):
            raise ValueError("interval choices require confidence_interval errors")
        return self

    def resolved(self) -> dict[str, Any]:
        from .rcparams import rc

        values = self.model_dump(mode="json", exclude_none=True)
        interval = self.percentiles is not None or self.confidence_level is not None
        values["error_metric"] = self.error_metric or (
            "confidence_interval" if interval else rc["plot.error_metric"]
        )
        if values["error_metric"] == "confidence_interval" and self.percentiles is None:
            values.setdefault("confidence_level", rc["plot.confidence_level"])
        return values


class SpectrumOptions(UncertaintyOptions):
    hover_confidence_level: float | None = Field(default=None, gt=0, lt=1)
    energy_type: Literal["ecm", "elab", "delab", "decm", "qcmsq"] | None = None


class SpectrumStyle(StrictModel):
    col_spacing: float = Field(default=1, gt=0)
    group_separator_color: str = "black"
    group_separator_linewidth: float = Field(default=1, ge=0)
    group_separator_linestyle: str = "-"
    group_separator_tick_length: float = Field(default=10, ge=0)
    stack_width: float = Field(default=0.3, ge=0)
    capsize: float = Field(default=5, ge=0)
    capthick: float = Field(default=1.5, ge=0)
    markersize: float = Field(default=6, ge=0)
    linewidth: float = Field(default=1.5, ge=0)
    fontsize: float | None = None
    column_label_fontsize: float | None = None
    outer_label_fontsize: float | None = None
    outer_label_offset: float = -50
    marker_label_fontsize: float | None = None
    marker_label_pad: float = Field(default=4, ge=0)
    marker_edge_pad: float = Field(default=0.3, ge=0)
    grid_alpha: float = Field(default=0.3, ge=0, le=1)
    ypad_frac: float = Field(default=0.05, ge=0)
    color_cycle: tuple[str, ...] | None = Field(default=None, min_length=1)
    excluded_color: str = "lightgrey"
    errorbar_kwargs: dict[str, Any] | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _limits(self):
        import math

        for name in ("xlim", "ylim"):
            limits = getattr(self, name)
            if limits is not None and (
                not all(map(math.isfinite, limits)) or limits[0] >= limits[1]
            ):
                raise ValueError(f"{name} must be finite and increasing")
        return self


class PlotInputError(ValueError):
    """Inputs cannot supply the requested diagnostic."""


@dataclass(frozen=True)
class PlotResult:
    figure: Any
    options: dict[str, Any]
    style: dict[str, Any] | None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PlotDefinition:
    name: str
    family: str
    options_model: type[StrictModel]
    style_model: type[StrictModel] | None
    backends: frozenset[str]
    validate_input: Callable
    make: Callable


def _validate(name, family, data, options):
    from .plotting.catalog import validate_input

    return validate_input(name, family, data, options)


def _make(name, family, data, *, options=None, style=None, backend=None):
    from .plotting.catalog import make_plot

    return make_plot(
        get_plot_definition(name, family=family),
        data,
        options=options,
        style=style,
        backend=backend,
    )


_METHODS = {
    "histogram": HistogramOptions,
    "errorbar": EmptyOptions,
    "correlation": EmptyOptions,
    "eff-sample-size": EmptyOptions,
    "summary": SummaryOptions,
    "bootstrap-intervals": BootstrapOptions,
    "corner": EmptyOptions,
}
_DEFINITIONS = tuple(
    PlotDefinition(
        name,
        family,
        model,
        DiagnosticStyle,
        frozenset(
            {"plotly"}
            if family == "diagnostic"
            else {"matplotlib"}
            if name == "corner"
            else {"matplotlib", "plotly", "recording"}
        ),
        partial(_validate, name, family),
        partial(_make, name, family),
    )
    for family, methods in (
        ("sampling", _METHODS),
        ("diagnostic", {"criterion": EmptyOptions, "params": EmptyOptions}),
    )
    for name, model in methods.items()
)


def plot_definitions(*, family=None) -> tuple[PlotDefinition, ...]:
    return tuple(d for d in _DEFINITIONS if family is None or d.family == family)


def get_plot_definition(name, *, family="sampling") -> PlotDefinition:
    for definition in plot_definitions(family=family):
        if definition.name == name:
            return definition
    raise ValueError(
        f"unknown {family} plot {name!r}; accepted: {[d.name for d in plot_definitions(family=family)]}"
    )
