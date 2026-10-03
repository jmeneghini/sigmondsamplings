"""Numeric model layer for fitting.

The objective differentiates and evaluates the model thousands of times per fit,
so the contract is deliberately thin and allocation-light: a model maps an
optimagic params PyTree (a ``{name: value}`` dict) to a **flat theory vector**
aligned with the ``SamplingStats`` observable order. Flattening any structure —
multiple channels, time slices, ensembles — is the caller's responsibility; the
fit layer never loops per observable.

A model of an independent variable (:class:`XModel`, e.g. :class:`CallableXModel`)
is the unbound form: :meth:`SamplingFit.from_data
<sigmondsamplings.fitting.fit.SamplingFit.from_data>` binds it to the coordinate
values of a 1-D ``SigmondData`` so ``x`` comes from the data it labels.

This is distinct from :class:`sigmondsamplings.model_func.SigmondModelFunc`,
which is the error-propagation / plotting engine for an already-fitted model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CallableModel",
    "CallableXModel",
    "Model",
    "ResampledModel",
    "XModel",
    "predict_model",
]


@runtime_checkable
class Model(Protocol):
    """A flat-vector theory model keyed by parameter name.

    ``predict`` receives the optimagic params dict and returns a 1-D array of
    length ``num_observables`` in the same order as the fitted ``SamplingStats``.
    """

    def predict(self, params: dict[str, float]) -> np.ndarray: ...


@runtime_checkable
class ResampledModel(Model, Protocol):
    """A model whose theory vector can depend on the resampling index.

    ``resamp_idx=0`` is the full-sample column. Positive indices match the
    corresponding resampling column in the fitted ``SamplingStats``.
    """

    def predict_at(self, params: dict[str, float], resamp_idx: int) -> np.ndarray: ...


@runtime_checkable
class XModel(Protocol):
    """A model of an independent variable, bound to the data's ``x`` before fitting.

    ``bind`` receives the coordinate values (a read-only float array, one entry per
    fitted observable, in data order) and returns the :class:`Model` the objective
    evaluates.
    """

    def bind(self, x: np.ndarray) -> Model: ...


def predict_model(
    model: Model,
    params: dict[str, float],
    resamp_idx: int | None = None,
) -> np.ndarray:
    """Evaluate a model, using ``predict_at`` when the model provides it."""
    predict_at = getattr(model, "predict_at", None)
    if resamp_idx is not None and predict_at is not None:
        return np.asarray(predict_at(params, resamp_idx), dtype=float).reshape(-1)
    return np.asarray(model.predict(params), dtype=float).reshape(-1)


@dataclass(frozen=True)
class CallableModel:
    """Adapt a positional ``func(theta) -> flat vector`` to the :class:`Model` API.

    ``func`` takes the parameter values as a single ``np.ndarray`` (in
    ``param_names`` order) and must return the flat theory vector. The dict→array
    conversion uses the cached ``param_names`` tuple so the per-call overhead is a
    single ``np.fromiter`` ahead of the user's (vectorized) evaluation.
    """

    func: Callable[[np.ndarray], np.ndarray]
    param_names: tuple[str, ...]

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], param_names: Sequence[str]):
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "param_names", tuple(param_names))

    def predict(self, params: dict[str, float]) -> np.ndarray:
        theta = np.fromiter(
            (params[name] for name in self.param_names),
            dtype=float,
            count=len(self.param_names),
        )
        return np.asarray(self.func(theta), dtype=float).reshape(-1)


@dataclass(frozen=True, eq=False)
class CallableXModel:
    """Adapt ``func(x, theta) -> flat vector`` to the :class:`XModel` API.

    ``theta`` is as for :class:`CallableModel`. Unbound (``x is None``) it only
    records the function; :meth:`bind` returns a copy holding ``x``, which is a
    :class:`Model`. Keeping ``x`` on the model (rather than in a closure) keeps
    it inspectable and picklable for the process backend when ``func`` is.
    """

    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    param_names: tuple[str, ...]
    x: np.ndarray | None = None

    def __init__(
        self,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        param_names: Sequence[str],
        x: np.ndarray | None = None,
    ):
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "param_names", tuple(param_names))
        if x is not None:
            x = np.array(x, dtype=float)
            x.flags.writeable = False
        object.__setattr__(self, "x", x)

    def bind(self, x: np.ndarray) -> CallableXModel:
        return CallableXModel(self.func, self.param_names, x)

    def predict(self, params: dict[str, float]) -> np.ndarray:
        if self.x is None:
            raise ValueError("CallableXModel has no x values; bind(x) it first")
        theta = np.fromiter(
            (params[name] for name in self.param_names),
            dtype=float,
            count=len(self.param_names),
        )
        return np.asarray(self.func(self.x, theta), dtype=float).reshape(-1)
