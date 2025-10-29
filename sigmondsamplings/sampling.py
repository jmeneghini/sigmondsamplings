"""
Core sampling classes for handling Sigmond samplings data.
"""

import numpy as np
import xml.etree.ElementTree as ET
from typing import Union, Optional, Dict, Any
import re

try:
    from uncertainties import ufloat
    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    UNCERTAINTIES_AVAILABLE = False


# Default ensemble for general use - accessible to users

class EnsembleInfo:
    """Information about the Monte Carlo ensemble."""
    
    def __init__(self, ensemble_name: str, num_measurements: int, num_bins: int, 
                 tweak_info: Optional[Dict[str, Any]] = None):
        self.ensemble_name = ensemble_name
        self.num_measurements = num_measurements
        self.num_bins = num_bins
        self.tweak_info = tweak_info or {}
    
    def __eq__(self, other):
        if not isinstance(other, EnsembleInfo):
            return False
        return (self.ensemble_name == other.ensemble_name and
                self.num_measurements == other.num_measurements and
                self.num_bins == other.num_bins and
                self.tweak_info == other.tweak_info)
    
    def __repr__(self):
        return f"EnsembleInfo('{self.ensemble_name}', {self.num_measurements}, {self.num_bins})"

DEFAULT_ENSEMBLE = EnsembleInfo("indep", 1, 1)

class SamplingInfo:
    """Information about the sampling method (Bootstrap/Jackknife)."""
    
    def __init__(self, method: str, num_resamplings: int, seed: int = 0, 
                 boot_skip: int = 0, **kwargs):
        self.method = method.lower()
        self.num_resamplings = num_resamplings
        self.seed = seed
        self.boot_skip = boot_skip
        self.extra_params = kwargs
    
    def __eq__(self, other):
        if not isinstance(other, SamplingInfo):
            return False
        return (self.method == other.method and
                self.num_resamplings == other.num_resamplings and
                self.seed == other.seed and
                self.boot_skip == other.boot_skip and
                self.extra_params == other.extra_params)
    
    def __repr__(self):
        return f"SamplingInfo('{self.method}', {self.num_resamplings}, seed={self.seed})"


class ObservableInfo:
    """Information about a specific observable."""
    
    def __init__(self, name: str, index: int = 0, op_type: str = "n",
                 re_im: str = "re", ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE, latex_str: str = None):
        self.name = name
        self.index = index
        self.op_type = op_type
        self.re_im = re_im
        self.ensemble_info = ensemble_info
        self.latex_str = latex_str # used for plotting
    
    @classmethod
    def from_string(cls, obs_string: str, ensemble_info: EnsembleInfo = DEFAULT_ENSEMBLE) -> 'ObservableInfo':
        """
        Parse observable info from string format.
        
        Expected format: "name index op_type re_im"
        """
        parts = obs_string.strip().split()
        if len(parts) != 4:
            raise ValueError(f"Invalid observable string format: {obs_string}")
        
        name, index_str, op_type, re_im = parts
        try:
            index = int(index_str)
        except ValueError:
            raise ValueError(f"Invalid index in observable string: {index_str}")
        
        return cls(name, index, op_type, re_im, ensemble_info)
    
    def __eq__(self, other):
        if not isinstance(other, ObservableInfo):
            return False
        return (self.name == other.name and 
                self.index == other.index and 
                self.op_type == other.op_type and 
                self.re_im == other.re_im and
                self.ensemble_info == other.ensemble_info)
        
    def _repr_latex__(self):
        """LaTeX representation for Jupyter notebooks."""
        if self.latex_str:
            return f"${self.latex_str}$"
        else:
            return self.__str__()
    
    def __repr__(self):
        return f"ObservableInfo(name='{self.name}', index={self.index}, ensemble='{self.ensemble_info.ensemble_name}')"
    
    def __str__(self):
        return f"{self.name} {self.index}" # Simple MCObs string format


class SigmondSampling:
    """
    Wrapper around numpy array for Sigmond samplings data.
    
    The first element [0] contains the full sample value (mean).
    Elements [1:] contain the resampled values (bootstrap/jackknife).
    """
    
    def __init__(self, data: Union[np.ndarray, list], observable_info: ObservableInfo,
                 sampling_info: SamplingInfo, is_complex: bool = False):
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim != 1:
            raise ValueError("Data must be 1-dimensional array")
        
        if len(data) < 2:
            raise ValueError("Data must have at least 2 elements (full sample + resamples)")
        
        self.data = data.astype(complex) if is_complex else data.astype(float)
        self.observable_info = observable_info
        self.sampling_info = sampling_info
        self.is_complex = is_complex
    
    @property
    def ensemble_info(self) -> EnsembleInfo:
        """Get ensemble info from the observable."""
        return self.observable_info.ensemble_info
    
    @ensemble_info.setter
    def ensemble_info(self, value: EnsembleInfo):
        """Set ensemble info for the observable."""
        self.observable_info.ensemble_info = value
    
    @property
    def full_sample_value(self):
        """The full sample value (mean of all measurements)."""
        return self.data[0]
    
    @property
    def resampled_values(self):
        """The resampled values (bootstrap/jackknife samples)."""
        return self.data[1:]
    
    @property
    def mean(self):
        """Mean of the resampled values (excluding full sample)."""
        return np.mean(self.resampled_values)
    
    @property
    def std(self):
        """Standard deviation of the resampled values."""
        return np.std(self.resampled_values, ddof=1)
    
    @property
    def error(self):
        """Statistical error estimate."""
        if self.sampling_info.method == 'bootstrap':
            return self.std
        elif self.sampling_info.method == 'jackknife':
            # Jackknife error correction
            n = len(self.resampled_values)
            return self.std * np.sqrt(n - 1)
        else:
            return self.std

    def to_ufloat(self):
        """
        Convert to uncertainties.ufloat object for PDG formatting.

        Returns:
            uncertainties.ufloat: Object with value and uncertainty

        Raises:
            ImportError: If uncertainties package is not available
        """
        if not UNCERTAINTIES_AVAILABLE:
            raise ImportError("uncertainties package is required for ufloat conversion. Install with: pip install uncertainties")

        if self.is_complex:
            raise ValueError("Complex samplings cannot be converted to ufloat. Use .to_real() first.")

        return ufloat(self.full_sample_value, self.error)

    def pdg_format(self, format_spec: str = '.2uS') -> str:
        """
        Format value and error using PDG conventions via uncertainties package.

        Args:
            format_spec: Format specification for uncertainties package
                       Common options:
                       - '.1uS': Shorthand notation like 1.23(4)
                       - '.1uP': Pretty-print with ± symbol
                       - '.1uL': LaTeX notation
                       - '.1ue': Scientific notation

        Returns:
            str: Formatted string using PDG conventions

        Raises:
            ImportError: If uncertainties package is not available
            ValueError: If sampling is complex
        """
        ufloat_obj = self.to_ufloat()
        return f"{ufloat_obj:{format_spec}}"
    
    def confidence_interval(self, confidence_level: float = 0.68) -> tuple:
        """
        Calculate confidence interval for bootstrap resampling.
        
        Args:
            confidence_level: Confidence level (0.68 = 1σ, 0.95 = 2σ, etc.)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
            
        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != 'bootstrap':
            raise ValueError("Confidence intervals are only supported for bootstrap resampling")
        
        # Calculate percentiles for the confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        # Calculate bounds from resampled values
        lower_bound = np.percentile(self.resampled_values, lower_percentile)
        upper_bound = np.percentile(self.resampled_values, upper_percentile)
        print(f"Confidence interval: ({lower_bound}, {upper_bound})")
        return (lower_bound, upper_bound)
    
    def bootstrap_bias(self) -> float:
        """
        Calculate bootstrap bias estimate.
        
        Returns:
            Bootstrap bias (mean of resamples - full sample value)
            
        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != 'bootstrap':
            raise ValueError("Bootstrap bias is only available for bootstrap resampling")
        
        return self.mean - self.full_sample_value
    
    def bias_corrected_mean(self) -> float:
        """
        Calculate bias-corrected mean estimate.
        
        Returns:
            Bias-corrected mean (full sample - bootstrap bias)
            
        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != 'bootstrap':
            raise ValueError("Bias correction is only available for bootstrap resampling")
        
        bias = self.bootstrap_bias()
        return self.full_sample_value - bias
    
    def bounded(self, lower: float, upper: float) -> 'SigmondSampling':
        """
        Return a new SigmondSampling with resampled values clipped into [lower, upper].
        
        Any resampling below the lower bound is set to the lower bound.
        Any resampling above the upper bound is set to the upper bound.
        The full-sample value [0] is unchanged.
        
        Args:
            lower: Lower bound
            upper: Upper bound
            
        Returns:
            SigmondSampling: new instance with bounded resamples
        """
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        
        # Copy data so we don't mutate the original
        bounded_data = self.data.copy()
        
        # Only apply clipping to resamples, not full sample
        bounded_data[1:] = np.clip(bounded_data[1:], lower, upper)
        
        return SigmondSampling(
            bounded_data,
            self.observable_info,
            self.sampling_info,
            is_complex=self.is_complex
        )
        
    def to_real(self):
        """
        Convert complex data to real part if the sampling is complex.
        
        Returns:
            SigmondSampling: A new instance with real data.
        """
        if not self.is_complex:
            return self
        
        real_data = self.data.real
        return SigmondSampling(
            real_data,
            self.observable_info,
            self.sampling_info,
            is_complex=False
        )
        
    def to_complex(self):
        """
        Convert real data to complex if the sampling is real.
        
        Returns:
            SigmondSampling: A new instance with complex data.
        """
        if self.is_complex:
            return self
        
        complex_data = self.data.astype(complex)
        return SigmondSampling(
            complex_data,
            self.observable_info,
            self.sampling_info,
            is_complex=True
        )
    
    def _check_compatible(self, others: set['SigmondSampling']):
        """
        Check if a set of samplings are compatible with self for arithmetic operations.

        This method validates that all provided `SigmondSampling` instances are
        compatible with the current instance by checking for matching sampling 
        methods and data sizes. Different observables are allowed - compatibility
        is based on sampling methodology, not observables.

        Args:
            others: A set of `SigmondSampling` instances to check for compatibility.

        Raises:
            ValueError: If any of the samplings are not compatible, specifying the
                        reasons for each incompatible instance.
        """
        incompatible = {}
        for other in others:
            errors = []

            if self.sampling_info != other.sampling_info:
                errors.append("different sampling methods")

            if len(self.data) != len(other.data):
                errors.append("different data lengths")
            
            if errors:
                incompatible[other] = errors
        
        if incompatible:
            error_messages = []
            for other, reasons in incompatible.items():
                error_messages.append(f" - {other!r}: {', '.join(reasons)}")
            raise ValueError("Incompatible samplings found:\n" + "\n".join(error_messages))

        return True
    
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Union['SigmondSampling', Any]:
        """Implements NumPy ufunc protocol for SigmondSampling."""
        if method != '__call__' or kwargs:
            return NotImplemented

        samplings = {arg for arg in inputs if isinstance(arg, SigmondSampling)}
        other_samplings = samplings - {self}

        self._check_compatible(other_samplings)

        new_inputs = [arg.data if isinstance(arg, SigmondSampling) else arg for arg in inputs]
        
        result_data = ufunc(*new_inputs)

        if result_data is NotImplemented:
            return NotImplemented

        is_complex = any(s.is_complex for s in samplings) or \
                     any(np.iscomplexobj(arg) for arg in new_inputs) or \
                     np.iscomplexobj(result_data)

        # Determine observable_info for result
        if len(samplings) == 1:
            # Only one sampling involved, use its observable_info
            result_observable_info = self.observable_info
        else:
            # Multiple samplings - check if all have same observable_info
            first_info = next(iter(samplings)).observable_info
            if all(s.observable_info == first_info for s in samplings):
                # All samplings have same observable_info
                result_observable_info = first_info
            else:
                # Mixed observable_infos - create a mixed one
                result_observable_info = ObservableInfo(
                    "mixed_operation", 0, "mixed", "re", DEFAULT_ENSEMBLE
                )

        return SigmondSampling(
            result_data,
            result_observable_info,
            self.sampling_info,
            is_complex=is_complex
        )
    
    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)
    
    def __neg__(self):
        return np.negative(self)
    
    def __pos__(self):
        return np.positive(self)
    
    def __abs__(self):
        return np.absolute(self)
    
    def unwrap(self, discont=np.pi, axis=-1):
        """
        Unwrap phase angles by changing jumps greater than discont to their 2*pi complement.
        
        Args:
            discont: Maximum discontinuity between values (default: pi)
            axis: Axis along which unwrap will operate (default: -1)
            
        Returns:
            SigmondSampling with unwrapped phase data
        """
        unwrapped_data = np.unwrap(self.data, discont=discont, axis=axis)
        return SigmondSampling(
            unwrapped_data,
            self.observable_info,
            self.sampling_info,
            self.is_complex
        )

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
    
    def __lt__(self, other):
        return self.full_sample_value < (other.full_sample_value if isinstance(other, SigmondSampling) else other)
    
    def __le__(self, other):
        return self.full_sample_value <= (other.full_sample_value if isinstance(other, SigmondSampling) else other)
    
    def __gt__(self, other):
        return self.full_sample_value > (other.full_sample_value if isinstance(other, SigmondSampling) else other)
    
    def __ge__(self, other):
        return self.full_sample_value >= (other.full_sample_value if isinstance(other, SigmondSampling) else other)

    def __repr__(self):
        return f"SigmondSampling(full={self.full_sample_value:.6f}, mean={self.mean:.6f}, error={self.error:.6f})"
    
    def __str__(self):
        return f"{self.full_sample_value:.6f} ± {self.error:.6f}"

