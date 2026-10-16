"""
Statistical analysis of the sample axis, shared by every Sigmond data container.

Mirrors sigmond's ``MCObsHandler::jack_analyze`` / ``boot_analyze`` and adds the
raw-bins case sigmond never needs. Every container stores its Monte Carlo data
with the *sample axis last*; how that axis is read depends only on the
``SamplingInfo`` (``None`` meaning raw bins):

=========  ===================  ===================  =================================
mode       central value        resamples            error
=========  ===================  ===================  =================================
bins       mean(data)           data                 sqrt(var / N)
jackknife  data[..., 0]         data[..., 1:]        sqrt((1 - 1/n) * sum((x - avg)^2))
bootstrap  data[..., 0]         data[..., 1:]        sqrt(sum((x - avg)^2) / (n - 1))
=========  ===================  ===================  =================================

The three modes differ only in :meth:`_Analyzer.variance_scale`, the factor
applied to the ``ddof=1`` variance (or covariance) of the resamples. Errors,
covariances and the synthetic-resample factor in fitting all read it, so each
formula exists exactly once.

All methods take plain numpy arrays (``SigmondDataArray`` routes xarray data
through ``xr.apply_ufunc``) and reduce over the last axis.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .info import SamplingInfo

__all__ = ["NonlinearBinsWarning", "analyzer_for"]


class NonlinearBinsWarning(UserWarning):
    """A non-linear operation was applied bin-by-bin to raw bins.

    ``f(bins)`` estimates ``<f(x)>``, not ``f(<x>)``; the latter needs
    ``SigmondBins.resample(...)`` first. Control it with the standard filters::

        warnings.simplefilter("error", NonlinearBinsWarning)   # strict
        warnings.simplefilter("ignore", NonlinearBinsWarning)  # intentional
    """


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _external_stacklevel() -> int:
    """Stacklevel pointing at the first frame outside this package."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level


class _Analyzer:
    """How to read the sample axis for one sampling mode."""

    method: str | None = None
    noun: str = "samplings"
    resample_slice = slice(1, None)

    def full(self, data: np.ndarray) -> np.ndarray:
        return data[..., 0]

    def resamples(self, data: np.ndarray) -> np.ndarray:
        return data[..., self.resample_slice]

    def mean(self, data: np.ndarray) -> np.ndarray:
        return self.resamples(data).mean(axis=-1)

    def central(self, data: np.ndarray, metric: str) -> np.ndarray:
        """The central estimate named by ``metric``."""
        if metric == "full_sample_value":
            return self.full(data)
        if metric == "mean":
            return self.mean(data)
        if metric == "median":
            return np.median(self.resamples(data), axis=-1)
        if metric == "bias_corrected_mean":
            return self.full(data) - self.bias(data)
        raise ValueError(
            f"unknown central metric {metric!r}; use one of "
            "'full_sample_value', 'mean', 'median', 'bias_corrected_mean'"
        )

    def variance_scale(self, n: int) -> float:
        """Factor on the ``ddof=1`` (co)variance of ``n`` resamples."""
        raise NotImplementedError

    def error(self, data: np.ndarray) -> np.ndarray:
        res = self.resamples(data)
        n = res.shape[-1]
        return res.std(axis=-1, ddof=1) * np.sqrt(self.variance_scale(n))

    def percentile_interval(self, data: np.ndarray, percentiles: tuple[float, float]):
        """``(lower, upper)`` percentiles of the resamples (of the bins, for raw bins)."""
        lower, upper = np.percentile(self.resamples(data), percentiles, axis=-1)
        return lower, upper

    def confidence_interval(self, data: np.ndarray, confidence_level: float = 0.68):
        alpha = 1 - confidence_level
        return self.percentile_interval(data, (100 * alpha / 2, 100 * (1 - alpha / 2)))

    def bias(self, data: np.ndarray) -> np.ndarray:
        raise ValueError("Bootstrap bias is only available for bootstrap resampling")

    def incompatibilities(self, a, b) -> list[str]:
        """Reasons containers ``a`` and ``b`` cannot be combined elementwise."""
        errors = []
        if a.sampling_info != b.sampling_info:
            errors.append("different sampling methods")
        if a._sample_length != b._sample_length:
            errors.append("different data lengths")
        return errors

    def check_ufunc(self, ufunc: np.ufunc, is_data: Sequence[bool]) -> None:
        """Hook run before a ufunc is applied; resamples accept any ufunc."""


class _JackknifeAnalyzer(_Analyzer):
    method = "jackknife"

    def variance_scale(self, n: int) -> float:
        # sigmond jack_analyze: (1 - 1/n) * sum((x - avg)^2) == (n-1)^2/n * var_ddof1
        return (n - 1) ** 2 / n

    def percentile_interval(self, data: np.ndarray, percentiles: tuple[float, float]):
        raise ValueError("Confidence intervals are only supported for bootstrap resampling")


class _BootstrapAnalyzer(_Analyzer):
    method = "bootstrap"

    def variance_scale(self, n: int) -> float:
        # sigmond boot_analyze: sum((x - avg)^2) / (nb - 1) == var_ddof1
        return 1.0

    def bias(self, data: np.ndarray) -> np.ndarray:
        return self.mean(data) - self.full(data)


# Ufuncs that are linear in a bins operand regardless of the other operands.
_LINEAR_UFUNCS = frozenset({np.add, np.subtract, np.negative, np.positive, np.conjugate})


class _BinsAnalyzer(_Analyzer):
    method = None
    noun = "bins"
    resample_slice = slice(None)

    def full(self, data: np.ndarray) -> np.ndarray:
        return data.mean(axis=-1)

    def variance_scale(self, n: int) -> float:
        # standard error of the mean
        return 1.0 / n

    def error(self, data: np.ndarray) -> np.ndarray:
        if data.shape[-1] < 2:
            return np.full(data.shape[:-1], np.nan)
        return super().error(data)

    def incompatibilities(self, a, b) -> list[str]:
        errors = []
        if a._sample_length != b._sample_length:
            errors.append("different number of bins")
        if a.ensemble_info != b.ensemble_info:
            errors.append("different ensembles")
        return errors

    def check_ufunc(self, ufunc: np.ufunc, is_data: Sequence[bool]) -> None:
        if ufunc in _LINEAR_UFUNCS:
            return
        if ufunc is np.multiply and sum(is_data) <= 1:
            return
        if ufunc is np.true_divide and len(is_data) == 2 and not is_data[1]:
            return
        warnings.warn(
            f"np.{ufunc.__name__} applied bin-by-bin to raw bins estimates <f(x)>, "
            f"not f(<x>). Call .resample(...) first for f(<x>), or use "
            f"SigmondBins.per_bin(...) to mark this as intentional.",
            NonlinearBinsWarning,
            stacklevel=_external_stacklevel(),
        )


_ANALYZERS: dict[str | None, _Analyzer] = {
    None: _BinsAnalyzer(),
    "jackknife": _JackknifeAnalyzer(),
    "bootstrap": _BootstrapAnalyzer(),
}


def analyzer_for(sampling_info: SamplingInfo | None) -> _Analyzer:
    """Return the analyzer for ``sampling_info`` (``None`` means raw bins)."""
    method = None if sampling_info is None else sampling_info.method
    try:
        return _ANALYZERS[method]
    except KeyError:
        raise ValueError(f"Unknown sampling method '{method}'") from None
