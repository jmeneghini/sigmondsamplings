"""
Container class for raw Monte Carlo bins data (SigmondBins).

SigmondBins mirrors the public API of SigmondSampling so that raw-bins
observables can be used interchangeably in plotting, statistics, and
ObservableCollection queries without being forced through a resampling step.

Key differences from SigmondSampling:
    - No SamplingInfo (``sampling_info`` is always ``None``).
    - ``data`` is the raw bin sequence; there is no "[0] = full sample" slot.
    - ``full_sample_value`` / ``mean`` are the unweighted mean of the bins.
    - ``error`` is the standard error of the mean: ``std(bins, ddof=1) / sqrt(N)``.

The class also retains the existing ``resample()`` method (bootstrap/jackknife)
which produces one or more SigmondSampling objects, optionally using Dask for
out-of-core processing of very large bin sequences.
"""

import importlib
import importlib.util
from typing import Any, Union

import numpy as np

from .info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling


class _LazyModule:
    """Import a (possibly dotted) module on first attribute access.

    ``dask.array`` is expensive to import (~0.5s; it pulls in pandas/toolz/yaml).
    ``lazy_loader.load`` only defers *top-level* modules -- for a dotted name it
    eagerly imports the parent package -- so we use this proxy to defer the
    import until a dask array is actually touched, not merely because this
    module was imported.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


da = _LazyModule("dask.array")
DASK_AVAILABLE = importlib.util.find_spec("dask") is not None

try:
    from uncertainties import ufloat

    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    UNCERTAINTIES_AVAILABLE = False


class SigmondBins:
    """
    Wrapper around a 1D array of raw Monte Carlo bins for a single observable.

    Behaves like a ``SigmondSampling`` for the purposes of statistics, arithmetic,
    and collection membership, but carries no ``SamplingInfo``.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, "da.Array"],
        observable_info: ObservableInfo,
        is_complex: bool = False,
        use_dask: bool | None = None,
    ):
        """
        Initialize SigmondBins.

        Args:
            data: Raw bins (1D array or list). Dask arrays are accepted when
                ``use_dask`` is True.
            observable_info: Metadata for the observable.
            is_complex: Whether the data is complex-valued.
            use_dask: If True, keep data as a Dask array and build resampling
                graphs lazily. If None (default), Dask is used only when the
                caller passes an existing ``dask.array``. Stats accessors
                (``mean``, ``error``, ...) always materialize to numpy.
        """
        # Determine backend
        if use_dask is None:
            use_dask = DASK_AVAILABLE and isinstance(data, da.Array) if DASK_AVAILABLE else False
        elif use_dask and not DASK_AVAILABLE:
            raise ImportError("Dask requested but not installed.")

        self.use_dask = use_dask
        self.xp = da if use_dask else np

        # Ingest data into the requested backend
        if use_dask:
            if not isinstance(data, da.Array):
                data = da.from_array(np.asarray(data), chunks="auto")
        else:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)

        if data.ndim != 1:
            raise ValueError("Bins data must be 1-dimensional array")

        target_dtype = complex if is_complex else float
        self.data = data.astype(target_dtype)
        self.observable_info = observable_info
        self.is_complex = is_complex

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------

    @property
    def sampling_info(self) -> None:
        """SigmondBins carries no SamplingInfo."""
        return None

    @property
    def ensemble_info(self) -> EnsembleInfo:
        """Get ensemble info from the observable."""
        return self.observable_info.ensemble_info

    @ensemble_info.setter
    def ensemble_info(self, value: EnsembleInfo):
        self.observable_info.ensemble_info = value

    @property
    def num_bins(self) -> int:
        return int(self.data.shape[0])

    # ------------------------------------------------------------------
    # Array materialization helpers
    # ------------------------------------------------------------------

    def _as_numpy(self) -> np.ndarray:
        """Return bins as a concrete numpy array (triggering .compute() if dask)."""
        if self.use_dask:
            return np.asarray(self.data.compute())
        return np.asarray(self.data)

    # ------------------------------------------------------------------
    # Sampling-like statistics
    # ------------------------------------------------------------------

    @property
    def full_sample_value(self):
        """Unweighted mean over bins (matches sigmond_query's reported value)."""
        arr = self._as_numpy()
        return np.mean(arr).item()

    @property
    def mean(self):
        """Mean of the bins (identical to ``full_sample_value`` for bins)."""
        return self.full_sample_value

    @property
    def _std(self) -> float:
        """Sample standard deviation of the bins (ddof=1)."""
        arr = self._as_numpy()
        return float(np.std(arr, ddof=1))

    @property
    def error(self) -> float:
        """Standard error of the mean: std(bins, ddof=1) / sqrt(N)."""
        n = self.num_bins
        if n < 2:
            return float("nan")
        return self._std / float(np.sqrt(n))

    def confidence_interval(self, confidence_level: float = 0.68) -> tuple:
        """Percentile-based confidence interval over the raw bins."""
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        arr = self._as_numpy()
        lower = np.percentile(arr, lower_percentile).item()
        upper = np.percentile(arr, upper_percentile).item()
        return (lower, upper)

    def to_ufloat(self) -> "ufloat":
        """Return an uncertainties.ufloat built from (mean, error)."""
        if not UNCERTAINTIES_AVAILABLE:
            raise ImportError(
                "uncertainties package is required for ufloat conversion. "
                "Install with: pip install uncertainties"
            )
        if self.is_complex:
            raise ValueError("Complex bins cannot be converted to ufloat. Use .to_real() first.")
        return ufloat(self.full_sample_value, self.error)

    def pdg_format(self, format_spec: str = ".2uS") -> str:
        """Format (mean, error) using PDG conventions via the uncertainties package."""
        return f"{self.to_ufloat():{format_spec}}"

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def to_real(self) -> "SigmondBins":
        """Return a new SigmondBins holding only the real part of the data."""
        if not self.is_complex:
            return self
        real_data = self.data.real
        return SigmondBins(
            real_data, self.observable_info, is_complex=False, use_dask=self.use_dask
        )

    def to_complex(self) -> "SigmondBins":
        """Promote real bins to complex dtype."""
        if self.is_complex:
            return self
        complex_data = self.data.astype(complex)
        return SigmondBins(
            complex_data, self.observable_info, is_complex=True, use_dask=self.use_dask
        )

    def with_observable_info(self, observable_info: ObservableInfo) -> "SigmondBins":
        """
        Return a metadata-only view with replaced observable info.

        The bin array/backend is shared with the original object. Lazy subclasses
        can override this to preserve deferred reads.
        """
        new = self.__class__.__new__(self.__class__)
        new.data = self.data
        new.observable_info = observable_info
        new.is_complex = self.is_complex
        new.use_dask = self.use_dask
        new.xp = self.xp
        return new

    def copy(self) -> "SigmondBins":
        """Return a copy with independent bin data and observable metadata."""
        data = self.data.copy()
        return SigmondBins(
            data,
            self.observable_info.copy(),
            is_complex=self.is_complex,
            use_dask=self.use_dask,
        )

    def bounded(self, lower: float, upper: float) -> "SigmondBins":
        """Clip bins into [lower, upper]."""
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        arr = self._as_numpy().copy()
        np.clip(arr, lower, upper, out=arr)
        return SigmondBins(arr, self.observable_info, is_complex=self.is_complex, use_dask=False)

    def unwrap(self, discont=np.pi, axis=-1) -> "SigmondBins":
        """Unwrap phase jumps along the bins axis."""
        arr = self._as_numpy()
        unwrapped = np.unwrap(arr, discont=discont, axis=axis)
        return SigmondBins(
            unwrapped, self.observable_info, is_complex=self.is_complex, use_dask=False
        )

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(
        self, sampling_info: SamplingInfo, statistic: str | list[str] = "mean"
    ) -> SigmondSampling | list[SigmondSampling]:
        """
        Perform block Bootstrap or Jackknife on the raw bins, returning
        SigmondSampling object(s).

        Args:
            sampling_info: Parameters for resampling (method, seed, count).
            statistic: Statistic to compute ("mean", "variance", "median", "raw_moment_N", ...).
        """
        if isinstance(statistic, str):
            statistics = [statistic]
            return_single = True
        else:
            statistics = list(statistic)
            return_single = False

        # --- BLOCKING LOGIC ---
        N_total = int(self.data.shape[0])
        ensemble_info = self.ensemble_info

        if "rebin" in ensemble_info.tweak_info:
            block_size = int(ensemble_info.tweak_info["rebin"])
            n_blocks = N_total // block_size
        elif ensemble_info.num_bins:
            n_blocks = int(ensemble_info.num_bins)
            block_size = N_total // n_blocks
        else:
            block_size = 1
            n_blocks = N_total

        if block_size < 1:
            block_size = 1
            n_blocks = N_total

        n_keep = n_blocks * block_size
        data_truncated = self.data[:n_keep]
        blocks_view = data_truncated.reshape(n_blocks, block_size)

        stat_funcs = {
            "mean": self.xp.mean,
            "variance": lambda x, **kwargs: self.xp.var(x, ddof=1, **kwargs),
            "std": lambda x, **kwargs: self.xp.std(x, ddof=1, **kwargs),
            "median": self.xp.median,
            "min": self.xp.min,
            "max": self.xp.max,
        }

        def get_stat_func(stat_name: str):
            if stat_name not in stat_funcs and "raw_moment" not in stat_name:
                raise ValueError(f"Unknown statistic '{stat_name}'")
            if "moment" in stat_name:
                power = int(stat_name.split("_")[2])
                return lambda x, **kwargs: self.xp.mean(x**power, **kwargs)
            return stat_funcs[stat_name]

        method = sampling_info.method.lower()
        resampled_traces = None

        if method == "bootstrap":
            n_resamples = sampling_info.num_resamplings
            if self.use_dask:
                rng = da.random.RandomState(sampling_info.seed)
            else:
                rng = np.random.RandomState(sampling_info.seed)
            block_indices = rng.randint(0, n_blocks, size=(n_resamples, n_blocks))
            resampled_blocks = blocks_view[block_indices]
            resampled_traces = resampled_blocks.reshape(n_resamples, n_keep)
        elif method == "jackknife":
            pass
        else:
            raise ValueError(f"Unknown sampling method '{method}'")

        results = []
        for stat_name in statistics:
            stat_func = get_stat_func(stat_name)

            full_sample_value = stat_func(data_truncated)

            if method == "bootstrap":
                resampled_values = stat_func(resampled_traces, axis=1)
            elif method == "jackknife":
                if self.use_dask:
                    jk_values = []
                    all_indices = da.arange(n_blocks)
                    for i in range(n_blocks):
                        mask = all_indices != i
                        jk_blocks = blocks_view[mask]
                        jk_trace = jk_blocks.flatten()
                        jk_values.append(stat_func(jk_trace))
                    resampled_values = da.stack(jk_values)
                else:
                    resampled_values = np.empty(n_blocks, dtype=self.data.dtype)
                    all_indices = np.arange(n_blocks)
                    for i in range(n_blocks):
                        keep_indices = np.delete(all_indices, i)
                        jk_blocks = blocks_view[keep_indices]
                        jk_trace = jk_blocks.flatten()
                        resampled_values[i] = stat_func(jk_trace)

            if self.use_dask:
                if isinstance(full_sample_value, (float, complex, np.number)):
                    full_val_arr = da.from_array([full_sample_value])
                elif hasattr(full_sample_value, "reshape"):
                    full_val_arr = full_sample_value.reshape(1)
                else:
                    full_val_arr = da.from_array(np.array([full_sample_value]))
                combined_data = da.concatenate([full_val_arr, resampled_values])
                final_data = combined_data.compute()
            else:
                final_data = np.concatenate([[full_sample_value], resampled_values])

            new_name = self.observable_info.name
            if len(statistics) > 1 or stat_name != "mean":
                new_name = f"{new_name}_{stat_name}"

            obs_info = ObservableInfo(
                name=new_name,
                index=self.observable_info.index,
                op_type=self.observable_info.op_type,
                re_im=self.observable_info.re_im,
                ensemble_info=self.observable_info.ensemble_info,
                latex_str=getattr(self.observable_info, "_latex_str", None),
            )

            results.append(
                SigmondSampling(final_data, obs_info, sampling_info, is_complex=self.is_complex)
            )

        return results[0] if return_single else results

    # ------------------------------------------------------------------
    # Compatibility / arithmetic
    # ------------------------------------------------------------------

    def _check_compatible(self, others: set["SigmondBins"]):
        """Validate that other SigmondBins can be combined with self."""
        incompatible = {}
        for other in others:
            errors = []
            if len(self.data) != len(other.data):
                errors.append("different number of bins")
            if self.ensemble_info != other.ensemble_info:
                errors.append("different ensembles")
            if errors:
                incompatible[other] = errors
        if incompatible:
            msgs = [
                f" - {other!r}: {', '.join(reasons)}" for other, reasons in incompatible.items()
            ]
            raise ValueError("Incompatible bins found:\n" + "\n".join(msgs))
        return True

    # ------------------------------------------------------------------
    # Comparisons / hashing / reprs
    # ------------------------------------------------------------------

    @property
    def latex_str(self):
        return f"{self.observable_info.latex_str} = {self.pdg_format()}"

    def __lt__(self, other):
        return self.full_sample_value < (
            other.full_sample_value if isinstance(other, SigmondBins) else other
        )

    def __le__(self, other):
        return self.full_sample_value <= (
            other.full_sample_value if isinstance(other, SigmondBins) else other
        )

    def __gt__(self, other):
        return self.full_sample_value > (
            other.full_sample_value if isinstance(other, SigmondBins) else other
        )

    def __ge__(self, other):
        return self.full_sample_value >= (
            other.full_sample_value if isinstance(other, SigmondBins) else other
        )

    def __eq__(self, other):
        if not isinstance(other, SigmondBins):
            return False
        return (
            self.observable_info == other.observable_info
            and self.num_bins == other.num_bins
            and self.is_complex == other.is_complex
        )

    def __hash__(self):
        return hash((self.observable_info, self.num_bins, self.is_complex))

    def __repr__(self):
        try:
            return (
                f"SigmondBins(name='{self.observable_info}', "
                f"n_bins={self.num_bins}, "
                f"mean={self.full_sample_value:.6f}, error={self.error:.6f})"
            )
        except Exception:
            return f"SigmondBins(name='{self.observable_info}', n_bins={self.num_bins})"

    def __str__(self):
        return f"{self.full_sample_value:.6f} ± {self.error:.6f}"

    def __len__(self):
        return self.num_bins

    # ------------------------------------------------------------------
    # NumPy ufunc protocol
    # ------------------------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        arr = self._as_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Union["SigmondBins", Any]:
        """NumPy ufunc protocol for SigmondBins. Operations are applied per-bin."""
        if method != "__call__" or kwargs:
            return NotImplemented

        bins_operands = {arg for arg in inputs if isinstance(arg, SigmondBins)}

        # Disallow mixing SigmondBins with SigmondSampling to avoid silent errors.
        if any(isinstance(arg, SigmondSampling) for arg in inputs):
            raise TypeError(
                "Cannot mix SigmondBins and SigmondSampling in a single operation. "
                "Call SigmondBins.resample(...) first to convert."
            )

        other_bins = bins_operands - {self}
        self._check_compatible(other_bins)

        new_inputs = [(arg._as_numpy() if isinstance(arg, SigmondBins) else arg) for arg in inputs]

        result_data = ufunc(*new_inputs)
        if result_data is NotImplemented:
            return NotImplemented

        is_complex = (
            any(b.is_complex for b in bins_operands)
            or any(np.iscomplexobj(arg) for arg in new_inputs)
            or np.iscomplexobj(result_data)
        )

        # Determine observable_info for result, mirroring SigmondSampling's logic.
        if len(bins_operands) == 1:
            result_observable_info = self.observable_info
        else:
            first_info = next(iter(bins_operands)).observable_info
            if all(b.observable_info == first_info for b in bins_operands):
                result_observable_info = first_info
            elif all(
                b.observable_info.ensemble_info == first_info.ensemble_info for b in bins_operands
            ):
                result_observable_info = ObservableInfo(
                    "mixed_operation", 0, "n", "re", first_info.ensemble_info
                )
            else:
                result_observable_info = ObservableInfo(
                    "mixed_operation", 0, "n", "re", INDEP_ENSEMBLE
                )

        return SigmondBins(
            result_data,
            result_observable_info,
            is_complex=is_complex,
            use_dask=False,
        )

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __neg__(self):
        return np.negative(self)

    def __pos__(self):
        return np.positive(self)

    def __abs__(self):
        return np.absolute(self)
