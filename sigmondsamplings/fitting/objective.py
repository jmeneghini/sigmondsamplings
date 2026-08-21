"""Least-squares objective for resampling fits.

The fit objective is a **whitened-residual vector** marked with
``optimagic.mark.least_squares``: least-squares algorithms consume it directly,
and optimagic auto-aggregates it to a scalar :math:`\\chi^2` for scalar
algorithms, so a single objective drives every minimizer choice.

Two ways to build one:

* :class:`LeastSquaresObjective` — the default path. Whitens
  ``data - predict_model(model, params, resamp_idx)`` with the data covariance from
  :class:`~sigmondsamplings.stats.SamplingStats`. The whitening operator is built
  **once** (no per-call Cholesky) so the inner loop is one matrix-vector product
  plus the model evaluation.
* :class:`CallableObjective` — wrap a user ``residual_fn(params, resamp_idx)``
  that already returns whitened residuals, bypassing our covariance handling
  entirely (custom/partial whitening, external residual engines, …).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .model import Model, predict_model
from .whitening import WhiteningTransform

if TYPE_CHECKING:
    from ..stats import SamplingStats

__all__ = [
    "CallableObjective",
    "LeastSquaresObjective",
    "Objective",
    "WhiteningTransform",
]

#: A residual function bound to a resampling index: ``f(params) -> residuals``.
ResidualFn = Callable[[dict[str, float]], np.ndarray]


def _least_squares(fn: ResidualFn) -> ResidualFn:
    """Tag ``fn`` as returning least-squares root contributions for optimagic."""
    import optimagic as om

    return om.mark.least_squares(fn)


@runtime_checkable
class Objective(Protocol):
    """What the fit driver needs from an objective.

    ``residuals_at`` returns the optimagic-marked least-squares callable (whitened
    residuals) for a resampling index; ``num_data_residuals`` is the residual
    count for dof bookkeeping.
    """

    num_data_residuals: int

    def residuals_at(self, resamp_idx: int) -> ResidualFn: ...


class LeastSquaresObjective:
    """Whitened-residual objective from ``SamplingStats`` + a flat-vector model."""

    def __init__(
        self,
        stats: SamplingStats,
        model: Model,
        *,
        use_correlation: bool = True,
    ):
        self.stats = stats
        self.model = model
        self.use_correlation = use_correlation
        # Cache the whitening operator and the data matrix once.
        self.whitening = WhiteningTransform.from_stats(stats, use_correlation=use_correlation)
        self._data = np.asarray(stats.array, dtype=float)  # (num_obs, num_samples)

    @property
    def num_observables(self) -> int:
        return self._data.shape[0]

    @property
    def num_data_residuals(self) -> int:
        return self._data.shape[0]

    def whitened_residuals(self, params: dict[str, float], resamp_idx: int) -> np.ndarray:
        """Whitened ``data[:, resamp_idx] - model_prediction``."""
        diff = self._data[:, resamp_idx] - predict_model(self.model, params, resamp_idx)
        return self.whitening.apply(diff)

    def residuals_at(self, resamp_idx: int) -> ResidualFn:
        def residuals(params: dict[str, float]) -> np.ndarray:
            return self.whitened_residuals(params, resamp_idx)

        return _least_squares(residuals)

    def chi2(self, params: dict[str, float], resamp_idx: int = 0) -> float:
        """:math:`\\chi^2` at ``params`` for one resampling index."""
        w = self.whitened_residuals(params, resamp_idx)
        return float(w @ w)


class CallableObjective:
    """Objective wrapping a user residual function over (params, resamp_idx).

    ``residual_fn`` must return the **whitened** residual vector; the fit layer
    does no covariance handling. ``num_data_residuals`` is required for dof.
    """

    def __init__(
        self,
        residual_fn: Callable[[dict[str, float], int], np.ndarray],
        *,
        num_data_residuals: int,
    ):
        self._residual_fn = residual_fn
        self.num_data_residuals = int(num_data_residuals)

    def whitened_residuals(self, params: dict[str, float], resamp_idx: int) -> np.ndarray:
        return np.asarray(self._residual_fn(params, resamp_idx), dtype=float).reshape(-1)

    def residuals_at(self, resamp_idx: int) -> ResidualFn:
        def residuals(params: dict[str, float]) -> np.ndarray:
            return self.whitened_residuals(params, resamp_idx)

        return _least_squares(residuals)

    def chi2(self, params: dict[str, float], resamp_idx: int = 0) -> float:
        w = self.whitened_residuals(params, resamp_idx)
        return float(w @ w)
