"""
Utility functions for SigmondSamplings package.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from .sampling import SigmondSampling, ObservableInfo, EnsembleInfo, SamplingInfo, DEFAULT_ENSEMBLE


def create_gaussian_sampling(mean: float, std: float,
                            sampling_info: SamplingInfo,
                            observable_name: str = "synthetic",
                            ensemble_info: Optional[EnsembleInfo] = None) -> SigmondSampling:
    """
    Create a SigmondSampling object with Gaussian-distributed data.
    
    Args:
        mean: Mean value for the distribution
        std: Standard deviation for the distribution
        sampling_info: SamplingInfo object
        observable_name: Name for the observable
        ensemble_info: EnsembleInfo object (uses DEFAULT_ENSEMBLE if None)
        
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
    
    # Use provided ensemble_info or default
    if ensemble_info is None:
        ensemble_info = DEFAULT_ENSEMBLE
    
    observable_info = ObservableInfo(observable_name, 0, 'n', 're', ensemble_info)
    
    return SigmondSampling(data, observable_info, sampling_info)


def create_uniform_sampling(low: float, high: float, num_samples: int,
                          observable_name: str = "synthetic",
                          ensemble_info: Optional[EnsembleInfo] = None,
                          sampling_method: str = "bootstrap",
                          seed: Optional[int] = None) -> SigmondSampling:
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
    observable_info = ObservableInfo(observable_name, 0, 'n', 're', ensemble_info)
    
    return SigmondSampling(data, observable_info, sampling_info)


def create_complex_gaussian_sampling(mean_real: float, std_real: float,
                                   mean_imag: float, std_imag: float,
                                   sampling_info: SamplingInfo,
                                   observable_name: str = "synthetic_complex",
                                   ensemble_info: Optional[EnsembleInfo] = None) -> SigmondSampling:
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
    
    observable_info = ObservableInfo(observable_name, 0, 'n', 'cx', ensemble_info)
    
    return SigmondSampling(data, observable_info, sampling_info, is_complex=True)


def bootstrap_resample(data: np.ndarray, num_samples: int, 
                      seed: Optional[int] = None) -> np.ndarray:
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


def combine_real_imaginary(real_sampling: SigmondSampling, 
                          imag_sampling: SigmondSampling) -> SigmondSampling:
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
        'cx',
        real_sampling.observable_info.ensemble_info
    )
    
    return SigmondSampling(complex_data, complex_obs_info,
                          real_sampling.sampling_info, is_complex=True)


def split_complex_sampling(complex_sampling: SigmondSampling) -> Tuple[SigmondSampling, SigmondSampling]:
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
        're',
        complex_sampling.observable_info.ensemble_info
    )
    
    imag_obs_info = ObservableInfo(
        complex_sampling.observable_info.name + "_imag",
        complex_sampling.observable_info.index,
        complex_sampling.observable_info.op_type,
        'im',
        complex_sampling.observable_info.ensemble_info
    )
    
    # Create samplings
    real_sampling = SigmondSampling(real_data, real_obs_info,
                                   complex_sampling.sampling_info, is_complex=False)
    
    imag_sampling = SigmondSampling(imag_data, imag_obs_info,
                                   complex_sampling.sampling_info, is_complex=False)
    
    return real_sampling, imag_sampling


def effective_sample_size(data: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Calculate effective sample size using autocorrelation.
    
    Args:
        data: Data array
        max_lag: Maximum lag to consider (default: len(data)//4)
        
    Returns:
        Effective sample size
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 4
    
    # Calculate autocorrelation
    data_centered = data - np.mean(data)
    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[n-1:] / autocorr[n-1]
    
    # Calculate integrated autocorrelation time
    tau_int = 0.5
    for i in range(1, min(max_lag, len(autocorr))):
        tau_int += autocorr[i]
        if i >= 2 * tau_int:
            break
    
    return n / (2 * tau_int)


def block_average(data: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform block averaging on data.
    
    Args:
        data: Data array
        block_size: Size of each block
        
    Returns:
        Tuple of (block_means, block_errors)
    """
    n = len(data)
    num_blocks = n // block_size
    
    if num_blocks == 0:
        raise ValueError("Block size too large for data")
    
    # Reshape data into blocks
    truncated_data = data[:num_blocks * block_size]
    blocks = truncated_data.reshape(num_blocks, block_size)
    
    # Calculate block means
    block_means = np.mean(blocks, axis=1)
    
    # Calculate error of block means
    block_error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
    
    return block_means, block_error 

def get_psq_from_string(name: str) -> Optional[int]:
        """
        Get the psq value from name.
        Args:
            name: Name of the observable
            
        Returns:
            psq: psq value if valid, None otherwise
        """
        # in name, we either have PSQ=int or P=(int,int,int)
        if 'PSQ=' in name:
            return int(name.split('PSQ=')[1][0])
        elif 'P=' in name:
            P_str= name.split('P=')[1].split(')')[0].split('(')[1]
            P_tuple = tuple(int(x) for x in P_str.split(','))
            psq = sum([P**2 for P in P_tuple])
            return psq
        else:
            return None