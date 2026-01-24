"""
Container class for raw sampling data (Bins), handling blocking and resampling logic.
Supports Dask for out-of-core processing of large datasets.
"""

import numpy as np
from typing import Union, List, Any, Optional

from .info import ObservableInfo, SamplingInfo, DEFAULT_ENSEMBLE
from .sampling import SigmondSampling

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class SigmondBins:
    """
    Container for raw time-series/MC-chain data.

    Responsible for:
    1. Storing raw bins (optionally as Dask arrays).
    2. Handling blocking logic (autocorrelation mitigation).
    3. Resampling (Bootstrap/Jackknife) to produce SigmondSampling objects.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, "da.Array"],
        observable_info: ObservableInfo,
        is_complex: bool = False,
        use_dask: Optional[bool] = None,
    ):
        """
        Initialize SigmondBins.

        Args:
            data: Raw data array.
            observable_info: Metadata for the observable.
            is_complex: Whether data is complex-valued.
            use_dask: If True, uses Dask for resampling operations.
                      If None, defaults to True if Dask is installed.
        """
        # Determine backend
        if use_dask is None:
            self.use_dask = DASK_AVAILABLE
        else:
            if use_dask and not DASK_AVAILABLE:
                raise ImportError("Dask requested but not installed.")
            self.use_dask = use_dask

        # Abstracted array module (numpy or dask.array)
        self.xp = da if self.use_dask else np

        # Ingest Data
        if self.use_dask:
            if not isinstance(data, da.Array):
                # Convert to dask array with auto chunks if not already one
                data = da.from_array(np.array(data), chunks="auto")
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)

        if data.ndim != 1:
            raise ValueError("Bins data must be 1-dimensional array")

        target_dtype = complex if is_complex else float
        self.data = data.astype(target_dtype)
        self.observable_info = observable_info
        self.is_complex = is_complex

    @property
    def ensemble_info(self):
        return self.observable_info.ensemble_info

    def resample(
        self, sampling_info: SamplingInfo, statistic: Union[str, List[str]] = "mean"
    ) -> Union[SigmondSampling, List[SigmondSampling]]:
        """
        Perform Block Bootstrap or Jackknife on the raw bins.

        If using Dask, computations are built lazily and executed (.compute())
        immediately before creating the SigmondSampling object, as SigmondSampling
        expects in-memory numpy arrays.

        Args:
            sampling_info: Parameters for resampling (method, seed, count).
            statistic: Statistic to compute ("mean", "variance", "median", etc).

        Returns:
            SigmondSampling object(s).
        """
        # Normalize statistic input
        if isinstance(statistic, str):
            statistics = [statistic]
            return_single = True
        else:
            statistics = statistic
            return_single = False

        # --- BLOCKING LOGIC ---
        # Note: We must compute length eagerly even if dask
        N_total = len(self.data) if not self.use_dask else self.data.shape[0]
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

        # Truncate data to fit perfectly into blocks
        n_keep = n_blocks * block_size
        data_truncated = self.data[:n_keep]

        # Reshape into (n_blocks, block_size)
        # This view allows us to manipulate whole blocks at once
        blocks_view = data_truncated.reshape(n_blocks, block_size)

        # --- STATISTIC FUNCTIONS ---
        # Define functions compatible with both numpy and dask
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

        # --- RESAMPLING GENERATION ---
        method = sampling_info.method.lower()
        resampled_traces = None

        # We prepare the "indices" or "views" here
        if method == "bootstrap":
            n_resamples = sampling_info.num_resamplings

            if self.use_dask:
                # Dask Random State
                rng = da.random.RandomState(sampling_info.seed)
                # Generate indices: (n_resamples, n_blocks)
                block_indices = rng.randint(0, n_blocks, size=(n_resamples, n_blocks))
                # Advanced indexing in Dask to pick blocks
                resampled_blocks = blocks_view[block_indices]
            else:
                rng = np.random.RandomState(sampling_info.seed)
                block_indices = rng.randint(0, n_blocks, size=(n_resamples, n_blocks))
                resampled_blocks = blocks_view[block_indices]

            # Flatten back to time-series structure: (n_resamples, n_total_truncated)
            resampled_traces = resampled_blocks.reshape(n_resamples, n_keep)

        elif method == "jackknife":
            # Jackknife is handled specifically in the loop below due to
            # the "delete one" nature being expensive to vectorize fully without enormous memory.
            pass
        else:
            raise ValueError(f"Unknown sampling method '{method}'")

        # --- COMPUTE STATISTICS ---
        results = []
        for stat_name in statistics:
            stat_func = get_stat_func(stat_name)

            # 1. Full sample statistic
            full_sample_value = stat_func(data_truncated)

            # 2. Resampled statistics
            if method == "bootstrap":
                # Compute across the time axis (axis 1)
                resampled_values = stat_func(resampled_traces, axis=1)

            elif method == "jackknife":
                if self.use_dask:
                    # Dask Jackknife: Construct graph of N delayed computations
                    jk_values = []
                    all_indices = da.arange(n_blocks)

                    # Note: For massive n_blocks, this loop builds a large graph.
                    # Ideally, map_blocks or generalized ufuncs are used, but
                    # "remove one block and stitch" is topologically complex.
                    for i in range(n_blocks):
                        # Boolean mask is cleaner for Dask graph than delete
                        mask = all_indices != i
                        jk_blocks = blocks_view[mask]
                        jk_trace = jk_blocks.flatten()
                        jk_values.append(stat_func(jk_trace))

                    resampled_values = da.stack(jk_values)
                else:
                    # Numpy Jackknife
                    resampled_values = np.empty(n_blocks, dtype=self.data.dtype)
                    all_indices = np.arange(n_blocks)

                    for i in range(n_blocks):
                        keep_indices = np.delete(all_indices, i)
                        jk_blocks = blocks_view[keep_indices]
                        jk_trace = jk_blocks.flatten()
                        resampled_values[i] = stat_func(jk_trace)

            # 3. Concatenate (Full + Resamples)
            if self.use_dask:
                # Ensure full_sample is a 1D dask array for concatenation
                full_val_arr = da.from_array(np.array([full_sample_value]))
                # If full_sample_value was a dask scalar, compute it or wrap it
                if isinstance(full_sample_value, (float, complex, np.number)):
                    full_val_arr = da.from_array([full_sample_value])
                elif hasattr(full_sample_value, "reshape"):
                    full_val_arr = full_sample_value.reshape(1)

                combined_data = da.concatenate([full_val_arr, resampled_values])

                # TRIGGER COMPUTATION
                # We return a concrete SigmondSampling object
                final_data = combined_data.compute()
            else:
                final_data = np.concatenate([[full_sample_value], resampled_values])

            # 4. Create ObservableInfo for the result
            new_name = self.observable_info.name
            if len(statistics) > 1 or stat_name != "mean":
                new_name = f"{new_name}_{stat_name}"

            obs_info = ObservableInfo(
                name=new_name,
                index=self.observable_info.index,
                op_type=self.observable_info.op_type,
                re_im=self.observable_info.re_im,
                ensemble_info=self.observable_info.ensemble_info,
                latex_str=self.observable_info.latex_str,
            )

            results.append(
                SigmondSampling(
                    final_data, obs_info, sampling_info, is_complex=self.is_complex
                )
            )

        return results[0] if return_single else results
