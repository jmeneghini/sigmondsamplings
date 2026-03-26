"""
Utility functions for SigmondSamplings package.
"""

import numpy as np

from .sampling import (
    DEFAULT_ENSEMBLE,
    EnsembleInfo,
    ObservableInfo,
    SamplingInfo,
    SigmondSampling,
)


def create_gaussian_sampling(
    mean: float,
    std: float,
    sampling_info: SamplingInfo,
    observable_info: ObservableInfo,
) -> SigmondSampling:
    """
    Create a SigmondSampling object with Gaussian-distributed data.

    Args:
        mean: Mean value for the distribution
        std: Standard deviation for the distribution
        sampling_info: SamplingInfo object
        observable_name: Name for the observable

    Returns:
        SigmondSampling object with synthetic data
    """
    num_samples = sampling_info.num_resamplings
    np.random.seed(sampling_info.seed)

    # Generate resampled values
    resampled_values = np.random.normal(mean, std, num_samples)

    # Create data array with full sample value at index 0
    data = np.zeros(num_samples + 1)
    data[0] = mean  # Full sample value
    data[1:] = resampled_values

    return SigmondSampling(data, observable_info, sampling_info)


def create_uniform_sampling(
    low: float,
    high: float,
    num_samples: int,
    observable_name: str = "synthetic",
    ensemble_info: EnsembleInfo | None = None,
    sampling_method: str = "bootstrap",
    seed: int | None = None,
) -> SigmondSampling:
    """
    Create a SigmondSampling object with uniformly-distributed data.

    Args:
        low: Lower bound for uniform distribution
        high: Upper bound for uniform distribution
        num_samples: Number of resampled values to generate
        observable_name: Name for the observable
        ensemble_info: EnsembleInfo object (uses DEFAULT_ENSEMBLE if None)
        sampling_method: Sampling method ('bootstrap' or 'jackknife')
        seed: Random seed for reproducibility

    Returns:
        SigmondSampling object with synthetic data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate resampled values
    resampled_values = np.random.uniform(low, high, num_samples)
    mean = (low + high) / 2

    # Create data array with full sample value at index 0
    data = np.zeros(num_samples + 1)
    data[0] = mean  # Full sample value
    data[1:] = resampled_values

    # Use provided ensemble_info or default
    if ensemble_info is None:
        ensemble_info = DEFAULT_ENSEMBLE

    sampling_info = SamplingInfo(sampling_method, num_samples, seed or 0)
    observable_info = ObservableInfo(observable_name, 0, "n", "re", ensemble_info)

    return SigmondSampling(data, observable_info, sampling_info)


def create_complex_gaussian_sampling(
    mean_real: float,
    std_real: float,
    mean_imag: float,
    std_imag: float,
    sampling_info: SamplingInfo,
    observable_name: str = "synthetic_complex",
    ensemble_info: EnsembleInfo | None = None,
) -> SigmondSampling:
    """
    Create a SigmondSampling object with complex Gaussian-distributed data.

    Args:
        mean_real: Mean value for the real part
        std_real: Standard deviation for the real part
        mean_imag: Mean value for the imaginary part
        std_imag: Standard deviation for the imaginary part
        sampling_info: SamplingInfo object
        observable_name: Name for the observable
        ensemble_info: EnsembleInfo object (uses DEFAULT_ENSEMBLE if None)

    Returns:
        SigmondSampling object with synthetic complex data
    """
    num_samples = sampling_info.num_resamplings
    np.random.seed(sampling_info.seed)

    # Generate resampled values
    real_part = np.random.normal(mean_real, std_real, num_samples)
    imag_part = np.random.normal(mean_imag, std_imag, num_samples)
    resampled_values = real_part + 1j * imag_part

    mean_complex = mean_real + 1j * mean_imag

    # Create data array with full sample value at index 0
    data = np.zeros(num_samples + 1, dtype=complex)
    data[0] = mean_complex  # Full sample value
    data[1:] = resampled_values

    # Use provided ensemble_info or default
    if ensemble_info is None:
        ensemble_info = DEFAULT_ENSEMBLE

    observable_info = ObservableInfo(observable_name, 0, "n", "cx", ensemble_info)

    return SigmondSampling(data, observable_info, sampling_info, is_complex=True)


def bootstrap_resample(data: np.ndarray, num_samples: int, seed: int | None = None) -> np.ndarray:
    """
    Perform bootstrap resampling of data.

    Args:
        data: Original data array
        num_samples: Number of bootstrap samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of bootstrap samples
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    samples = np.zeros(num_samples)

    for i in range(num_samples):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        samples[i] = np.mean(data[indices])

    return samples


def jackknife_resample(data: np.ndarray) -> np.ndarray:
    """
    Perform jackknife resampling of data.

    Args:
        data: Original data array

    Returns:
        Array of jackknife samples
    """
    n = len(data)
    samples = np.zeros(n)
    total_sum = np.sum(data)

    for i in range(n):
        # Leave-one-out mean
        samples[i] = (total_sum - data[i]) / (n - 1)

    return samples


def combine_real_imaginary(
    real_sampling: SigmondSampling, imag_sampling: SigmondSampling
) -> SigmondSampling:
    """
    Combine real and imaginary parts into a complex sampling.

    Args:
        real_sampling: SigmondSampling for real part
        imag_sampling: SigmondSampling for imaginary part

    Returns:
        Combined complex SigmondSampling
    """
    # Check compatibility
    if real_sampling.ensemble_info != imag_sampling.ensemble_info:
        raise ValueError("Real and imaginary parts must have same ensemble info")

    if real_sampling.sampling_info != imag_sampling.sampling_info:
        raise ValueError("Real and imaginary parts must have same sampling info")

    if len(real_sampling.data) != len(imag_sampling.data):
        raise ValueError("Real and imaginary parts must have same data length")

    # Combine data
    complex_data = real_sampling.data + 1j * imag_sampling.data

    # Create complex observable info
    complex_obs_info = ObservableInfo(
        real_sampling.observable_info.name + "_complex",
        real_sampling.observable_info.index,
        real_sampling.observable_info.op_type,
        "cx",
        real_sampling.observable_info.ensemble_info,
    )

    return SigmondSampling(
        complex_data, complex_obs_info, real_sampling.sampling_info, is_complex=True
    )


def split_complex_sampling(
    complex_sampling: SigmondSampling,
) -> tuple[SigmondSampling, SigmondSampling]:
    """
    Split a complex sampling into real and imaginary parts.

    Args:
        complex_sampling: Complex SigmondSampling to split

    Returns:
        Tuple of (real_sampling, imaginary_sampling)
    """
    if not complex_sampling.is_complex:
        raise ValueError("Sampling is not complex")

    # Extract real and imaginary parts
    real_data = np.real(complex_sampling.data)
    imag_data = np.imag(complex_sampling.data)

    # Create observable infos
    real_obs_info = ObservableInfo(
        complex_sampling.observable_info.name + "_real",
        complex_sampling.observable_info.index,
        complex_sampling.observable_info.op_type,
        "re",
        complex_sampling.observable_info.ensemble_info,
    )

    imag_obs_info = ObservableInfo(
        complex_sampling.observable_info.name + "_imag",
        complex_sampling.observable_info.index,
        complex_sampling.observable_info.op_type,
        "im",
        complex_sampling.observable_info.ensemble_info,
    )

    # Create samplings
    real_sampling = SigmondSampling(
        real_data, real_obs_info, complex_sampling.sampling_info, is_complex=False
    )

    imag_sampling = SigmondSampling(
        imag_data, imag_obs_info, complex_sampling.sampling_info, is_complex=False
    )

    return real_sampling, imag_sampling


def compute_autocorrelation(data: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Compute normalized autocorrelation function of time-series data.

    The autocorrelation function is defined as:
        C(t) = <x(t) x(0)> - <x>^2
               -------------------
                  <x^2> - <x>^2

    Args:
        data: Time-series data array (e.g., raw Monte Carlo bins)
        max_lag: Maximum lag to compute (default: len(data)//2)

    Returns:
        Autocorrelation array of length (max_lag + 1), with C(0) = 1.0

    Example:
        >>> bins = np.random.randn(1000)
        >>> acf = compute_autocorrelation(bins, max_lag=50)
        >>> acf[0]  # Should be 1.0
        1.0
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 2

    max_lag = min(max_lag, n - 1)

    # Center the data
    data_centered = data - np.mean(data)

    # Compute autocorrelation using FFT for efficiency
    autocorr = np.correlate(data_centered, data_centered, mode="full")
    autocorr = autocorr[n - 1 : n + max_lag]  # Keep only positive lags

    # Normalize by variance and number of samples at each lag
    variance = autocorr[0]
    if variance > 0:
        autocorr = autocorr / variance
    else:
        autocorr = np.zeros_like(autocorr)

    return autocorr


def integrated_autocorrelation_time(
    data: np.ndarray, max_lag: int | None = None, window_method: str = "auto"
) -> float:
    """
    Compute integrated autocorrelation time using automatic windowing.

    The integrated autocorrelation time is:
        τ_int = 1/2 + Σ_{t=1}^{W} C(t)

    where W is determined by the windowing method to reduce noise.

    Args:
        data: Time-series data array (e.g., raw Monte Carlo bins)
        max_lag: Maximum lag to consider (default: len(data)//4)
        window_method: Windowing method ('auto' or 'madras-sokal')
            - 'auto': Stop when lag >= 2*tau_int (adaptive windowing)
            - 'madras-sokal': Use Madras-Sokal criterion

    Returns:
        Integrated autocorrelation time τ_int (always >= 0.5)

    Example:
        >>> bins = np.random.randn(1000)
        >>> tau = integrated_autocorrelation_time(bins)
        >>> print(f"τ_int = {tau:.2f}")
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 4

    # Compute autocorrelation
    autocorr = compute_autocorrelation(data, max_lag)

    # Integrated autocorrelation time using automatic windowing
    tau_int = 0.5
    if window_method == "auto" or window_method == "madras-sokal":
        # Madras-Sokal automatic windowing
        for i in range(1, len(autocorr)):
            tau_int += autocorr[i]
            # Stop when window is large enough relative to tau_int
            if i >= 2 * tau_int:
                break
    else:
        # Sum all available lags
        tau_int += np.sum(autocorr[1:])

    # Ensure tau_int >= 0.5
    return max(0.5, tau_int)


def effective_sample_size(data: np.ndarray, max_lag: int | None = None) -> float:
    """
    Calculate effective number of independent samples accounting for autocorrelation.

    The effective sample size is:
        N_eff = N / (2 * τ_int)

    where N is the number of samples and τ_int is the integrated autocorrelation time.

    Args:
        data: Time-series data array (e.g., raw Monte Carlo bins)
        max_lag: Maximum lag to consider (default: len(data)//4)

    Returns:
        Effective sample size (1 <= N_eff <= N)

    Example:
        >>> bins = np.random.randn(1000)
        >>> n_eff = effective_sample_size(bins)
        >>> print(f"Effective samples: {n_eff:.1f} / {len(bins)}")
    """
    n = len(data)
    tau_int = integrated_autocorrelation_time(data, max_lag)
    return n / (2 * tau_int)


def rebin_data(bins: np.ndarray, rebin_size: int) -> np.ndarray:
    """
    Rebin data by averaging consecutive bins to reduce autocorrelation.

    Args:
        bins: Original bins array
        rebin_size: Number of consecutive bins to combine into one

    Returns:
        Rebinned array with length len(bins) // rebin_size

    Example:
        >>> bins = np.array([1, 2, 3, 4, 5, 6])
        >>> rebin_data(bins, 2)
        array([1.5, 3.5, 5.5])  # Averages (1,2), (3,4), (5,6)
    """
    if rebin_size <= 0:
        raise ValueError("rebin_size must be positive")

    if rebin_size == 1:
        return bins

    n_bins = len(bins)
    n_rebinned = n_bins // rebin_size

    if n_rebinned == 0:
        raise ValueError(f"rebin_size {rebin_size} too large for {n_bins} bins")

    # Truncate to multiple of rebin_size and reshape
    truncated = bins[: n_rebinned * rebin_size]
    reshaped = truncated.reshape(n_rebinned, rebin_size)

    # Average over the rebin_size axis
    return np.mean(reshaped, axis=1)


def stacked_positions(y, yerr, x=None, width=0.16, pad=0.0):
    """Compute jittered x-positions that stack overlapping data points into columns.

    Only points whose y-intervals (y ± yerr) overlap with at least one other
    point in their group are displaced. Non-overlapping points are returned at
    exactly their group center x. Displaced points are assigned to the minimum
    number of non-overlapping columns via greedy interval scheduling, then placed
    at evenly-spaced positions centered about x. Column spacing is uniform across
    all groups.

    Parameters
    ----------
    y : array-like
        Y values of the data points.
    yerr : array-like or scalar
        Y errors. A scalar is broadcast to all points.
    x : array-like, scalar, or None
        Group centers. None defaults to 0. A scalar is broadcast to all points.
        An array assigns each point to a group; points sharing the same x value
        are stacked together independently.
    width : float
        Total horizontal span available for stacking within each group. The
        outermost columns are placed at most ±width/2 from the group center.
    pad : float
        Extra padding added to each side of a y-interval before overlap testing,
        increasing the separation required before two points share a column.

    Returns
    -------
    xj : np.ndarray
        Jittered x-positions, one per input point. Points with no overlapping
        neighbours are returned at exactly x.
    """
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    if yerr.ndim == 0:
        yerr = np.full_like(y, yerr)

    # 1. Standardize x to an array of group centers
    if x is None:
        x = np.zeros_like(y, dtype=float)
    elif isinstance(x, (int, float)):
        x = np.full_like(y, x, dtype=float)
    else:
        x = np.asarray(x)

    offsets = np.zeros_like(y, dtype=float)

    # 2. Process each x-group independently
    for group_val in np.unique(x):
        # Sorted indices for this group (stable greedy sweep)
        idx = np.where(x == group_val)[0]
        idx = idx[np.argsort(y[idx])]

        ym = y[idx] - yerr[idx] - pad  # interval lower bounds
        yM = y[idx] + yerr[idx] + pad  # interval upper bounds
        n = len(idx)

        # 3. Find which points overlap with at least one other in the group.
        #    Non-overlapping points keep offset = 0 and are excluded from stacking.
        has_overlap = np.zeros(n, dtype=bool)
        for a in range(n):
            for b in range(n):
                if a != b and ym[a] <= yM[b] and yM[a] >= ym[b]:
                    has_overlap[a] = True
                    break

        if not np.any(has_overlap):
            continue  # all points are isolated — nothing to stack

        # Local indices (within idx) and original indices of overlapping points
        over_local = np.where(has_overlap)[0]
        over_idx = idx[over_local]
        ym_o = ym[over_local]
        yM_o = yM[over_local]

        # 4. Greedy interval scheduling: assign each overlapping point to the
        #    first column it fits in, opening a new column if needed.
        columns: list[list[tuple[float, float]]] = []
        col_assign = np.full(len(over_idx), -1, dtype=int)

        for j in range(len(over_idx)):
            placed = False
            for col_idx, intervals in enumerate(columns):
                int_arr = np.array(intervals)
                if np.any((ym_o[j] <= int_arr[:, 1]) & (yM_o[j] >= int_arr[:, 0])):
                    continue
                columns[col_idx].append((ym_o[j], yM_o[j]))
                col_assign[j] = col_idx
                placed = True
                break

            if not placed:
                columns.append([(ym_o[j], yM_o[j])])
                col_assign[j] = len(columns) - 1

        # 5. Map k columns to k evenly-spaced positions centered about 0
        k = len(columns)
        positions = np.linspace(-width / 2, width / 2, k) if k > 1 else np.array([0.0])

        for j, i in enumerate(over_idx):
            offsets[i] = positions[col_assign[j]]

    # 6. Scale all offsets globally so the largest reaches exactly ±width/2,
    #    keeping column spacing consistent across groups.
    max_abs = np.max(np.abs(offsets))
    if max_abs > 0:
        offsets *= (width / 2) / max_abs

    return x + offsets
