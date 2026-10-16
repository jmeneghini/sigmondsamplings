"""
Container class for raw Monte Carlo bins data (SigmondBins).

SigmondBins shares the public API of SigmondSampling (both derive from
:class:`~sigmondsamplings.data.SigmondData`) so that raw-bins observables can be
used interchangeably in plotting, statistics, and ObservableCollection queries
without being forced through a resampling step.

Key differences from SigmondSampling:
    - No SamplingInfo (``sampling_info`` is always ``None``).
    - ``data`` is the raw bin sequence; there is no "[0] = full sample" slot.
    - ``full_sample_value`` / ``mean`` are the unweighted mean of the bins.
    - ``error`` is the standard error of the mean: ``std(bins, ddof=1) / sqrt(N)``.
    - Non-linear arithmetic emits :class:`~sigmondsamplings.analysis.NonlinearBinsWarning`.

The class also provides ``resample()`` (bootstrap/jackknife), which produces one
or more SigmondSampling objects of the same shape.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np

from .analysis import NonlinearBinsWarning
from .data import SigmondData
from .info import EnsembleInfo, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

# ---------------------------------------------------------------------------
# Block resampling
# ---------------------------------------------------------------------------


def _block_layout(n_total: int, ensemble_info: EnsembleInfo) -> tuple[int, int]:
    """Return ``(n_blocks, block_size)`` for ``n_total`` bins of ``ensemble_info``."""
    if "rebin" in ensemble_info.tweak_info:
        block_size = int(ensemble_info.tweak_info["rebin"])
        n_blocks = n_total // block_size
    elif ensemble_info.num_bins:
        n_blocks = int(ensemble_info.num_bins)
        block_size = n_total // n_blocks
    else:
        block_size = 1
        n_blocks = n_total

    if block_size < 1:
        block_size = 1
        n_blocks = n_total
    return n_blocks, block_size


def _stat_func(stat_name: str) -> Callable:
    """Statistic reducing over ``axis``."""
    stat_funcs = {
        "mean": np.mean,
        "variance": lambda x, **kwargs: np.var(x, ddof=1, **kwargs),
        "std": lambda x, **kwargs: np.std(x, ddof=1, **kwargs),
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }
    if stat_name not in stat_funcs and "raw_moment" not in stat_name:
        raise ValueError(f"Unknown statistic '{stat_name}'")
    if "moment" in stat_name:
        power = int(stat_name.split("_")[2])
        return lambda x, **kwargs: np.mean(x**power, **kwargs)
    return stat_funcs[stat_name]


def _mean_power(stat_name: str) -> int | None:
    """``p`` if the statistic is the mean of ``x**p`` (``mean`` is ``p = 1``), else None."""
    if stat_name == "mean":
        return 1
    if "raw_moment" in stat_name:
        return int(stat_name.split("_")[2])
    return None


def _block_resample(
    data,
    ensemble_info: EnsembleInfo,
    sampling_info: SamplingInfo,
    statistics: Sequence[str],
) -> list[np.ndarray]:
    """Block bootstrap/jackknife over the last axis of ``data`` (shape ``(..., N)``).

    Returns one array of shape ``(..., R + 1)`` per statistic, laid out as
    sigmond samplings: the full-sample statistic first, then the resamples.
    Trailing bins that do not fill a whole block are dropped.

    Blocks are equal-sized, so a resample's mean is the mean of its block means:
    means (and raw moments) are computed from the block means directly, without
    building each resampled trace. Other statistics build the traces.
    """
    data = np.asarray(data)
    lead = data.shape[:-1]
    n_blocks, block_size = _block_layout(int(data.shape[-1]), ensemble_info)
    n_keep = n_blocks * block_size
    truncated = data[..., :n_keep]
    blocks = truncated.reshape(*lead, n_blocks, block_size)

    method = sampling_info.method.lower()
    if method == "bootstrap":
        n_resamples = sampling_info.num_resamplings
        rng = np.random.RandomState(sampling_info.seed)
        block_indices = rng.randint(0, n_blocks, size=(n_resamples, n_blocks))
        # weights[r, b]: fraction of resample r drawn from block b
        weights = np.zeros((n_resamples, n_blocks))
        np.add.at(weights, (np.arange(n_resamples)[:, None], block_indices), 1.0 / n_blocks)

        def resample_means(block_means):
            return block_means @ weights.T

        def traces():
            for idx in block_indices:
                yield blocks[..., idx, :].reshape(*lead, n_keep)

    elif method == "jackknife":
        if sampling_info.num_resamplings != n_blocks:
            raise ValueError(
                f"Jackknife over {n_blocks} blocks gives {n_blocks} resamplings, but "
                f"sampling_info.num_resamplings is {sampling_info.num_resamplings}"
            )
        all_indices = np.arange(n_blocks)

        def resample_means(block_means):
            total = block_means.sum(axis=-1, keepdims=True)
            return (total - block_means) / (n_blocks - 1)

        def traces():
            for i in range(n_blocks):
                keep = np.delete(all_indices, i)
                yield blocks[..., keep, :].reshape(*lead, n_keep - block_size)

    else:
        raise ValueError(f"Unknown sampling method '{method}'")

    results = []
    for stat_name in statistics:
        power = _mean_power(stat_name)
        if power is not None:
            x = truncated if power == 1 else truncated**power
            full = x.mean(axis=-1)
            resampled = resample_means(x.reshape(*lead, n_blocks, block_size).mean(axis=-1))
        else:
            stat = _stat_func(stat_name)
            full = stat(truncated, axis=-1)
            resampled = np.stack([stat(trace, axis=-1) for trace in traces()], axis=-1)
        results.append(np.concatenate([full[..., None], resampled], axis=-1))
    return results


class SigmondBins(SigmondData):
    """
    Raw Monte Carlo bins, one observable (0-d) or an N-D array of them.

    Behaves like a ``SigmondSampling`` for the purposes of statistics, arithmetic,
    and collection membership, but carries no ``SamplingInfo``.
    """

    def __init__(
        self,
        data: np.ndarray | list,
        observable_info: ObservableInfo,
        is_complex: bool = False,
    ):
        """
        Initialize a single-observable SigmondBins.

        Args:
            data: Raw bins (1D array or list).
            observable_info: Metadata for the observable.
            is_complex: Whether the data is complex-valued.
        """
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("Bins data must be 1-dimensional array")

        self.data = data.astype(complex if is_complex else float)
        self.observable_info = observable_info
        self.sampling_info = None
        self.is_complex = is_complex

    # ------------------------------------------------------------------
    # SigmondData hooks
    # ------------------------------------------------------------------

    def _kind_key(self) -> tuple:
        return (self.observable_info, self.num_bins, self.is_complex)

    @property
    def _sample_length(self) -> int:
        return self.num_bins

    @property
    def num_bins(self) -> int:
        return int(self.data.shape[-1])

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def per_bin(self, func: Callable, *args, **kwargs) -> "SigmondBins":
        """Apply ``func`` bin-by-bin, marking a non-linear ``<f(x)>`` as intended.

        ``bins.per_bin(np.log)`` is ``np.log(bins)`` without the
        :class:`NonlinearBinsWarning`.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NonlinearBinsWarning)
            return func(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(
        self, sampling_info: SamplingInfo, statistic: str | list[str] = "mean"
    ) -> SigmondSampling | list[SigmondSampling]:
        """
        Perform block Bootstrap or Jackknife on the raw bins, returning
        SigmondSampling object(s) of the same shape.

        Blocking follows ``ensemble_info``. A statistic other than ``"mean"`` (or
        several statistics) appends ``_<statistic>`` to the names.

        Args:
            sampling_info: Parameters for resampling (method, seed, count).
            statistic: Statistic to compute ("mean", "variance", "median", "raw_moment_N", ...).
        """
        statistics = [statistic] if isinstance(statistic, str) else list(statistic)
        all_data = _block_resample(self.data, self.ensemble_info, sampling_info, statistics)

        results = []
        for stat_name, buf in zip(statistics, all_data):
            suffix = f"_{stat_name}" if len(statistics) > 1 or stat_name != "mean" else ""
            results.append(self._resampled(buf, sampling_info, suffix))
        return results[0] if isinstance(statistic, str) else results

    def _resampled(self, buf: np.ndarray, sampling_info: SamplingInfo, suffix: str):
        from .arithmetic import DerivedObsInfo, arithmetic_info
        from .metadata import AtObsInfo

        def result_info(info):
            if isinstance(info, (DerivedObsInfo, AtObsInfo)):
                return arithmetic_info("resample" + suffix, [info]) if suffix else info.copy()
            result = info.copy()
            if suffix:
                result.name = f"{result.name}{suffix}"
            return result

        if self.ndim == 0:
            info = self.observable_info
            if isinstance(info, (DerivedObsInfo, AtObsInfo)):
                return SigmondSampling(buf, result_info(info), sampling_info, is_complex=self.is_complex)
            obs_info = ObservableInfo(
                name=f"{info.name}{suffix}",
                index=info.index,
                op_type=info.op_type,
                re_im=info.re_im,
                ensemble_info=info.ensemble_info,
                latex_str=getattr(info, "_latex_str", None),
            )
            return SigmondSampling(buf, obs_info, sampling_info, is_complex=self.is_complex)

        obs_info = result_info(self.observable_info)
        infos = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            infos[idx] = result_info(self._infos[idx])
        return SigmondSampling._from_parts(
            buf, obs_info, sampling_info, self.is_complex, self._dims, self._coords, infos
        )

    # ------------------------------------------------------------------
    # Reprs
    # ------------------------------------------------------------------

    def __repr__(self):
        if self.ndim:
            return self._array_repr()
        try:
            return (
                f"SigmondBins(name='{self.observable_info}', "
                f"n_bins={self.num_bins}, value={self.pdg_format()})"
            )
        except Exception:
            return f"SigmondBins(name='{self.observable_info}', n_bins={self.num_bins})"
