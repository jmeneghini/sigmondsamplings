"""
Core sampling classes for handling Sigmond samplings data.
"""

import numpy as np
from typing import Union, List, Any

from .info import EnsembleInfo, SamplingInfo, ObservableInfo, DEFAULT_ENSEMBLE

try:
    from uncertainties import ufloat

    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    UNCERTAINTIES_AVAILABLE = False


class SigmondSampling:
    """
    Wrapper around numpy array for Sigmond samplings data.

    The first element [0] contains the full sample value (mean).
    Elements [1:] contain the resampled values (bootstrap/jackknife).
    """

    def __init__(
        self,
        data: Union[np.ndarray, list],
        observable_info: ObservableInfo,
        sampling_info: SamplingInfo,
        is_complex: bool = False,
    ):

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 1:
            raise ValueError("Data must be 1-dimensional array")

        if len(data) < 2:
            raise ValueError(
                "Data must have at least 2 elements (full sample + resamples)"
            )

        self.data = data.astype(complex) if is_complex else data.astype(float)
        self.observable_info = observable_info
        self.sampling_info = sampling_info
        self.is_complex = is_complex

    @classmethod
    def from_single_value(
        cls,
        value: Union[float, complex],
        observable_info: ObservableInfo,
        sampling_info: SamplingInfo,
    ) -> "SigmondSampling":
        """
        Create a SigmondSampling from a single value (no resampling).

        Args:
            value: The full sample value
            observable_info: ObservableInfo for the sampling
            sampling_info: SamplingInfo for the sampling
            is_complex: Whether the data is complex

        Returns:
            SigmondSampling instance with single value ('fixed' sampling)
        """
        data = np.array([value] * (sampling_info.num_resamplings + 1))
        is_complex = np.iscomplexobj(value)
        return cls(data, observable_info, sampling_info, is_complex=is_complex)

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
        return self.data[0].item()

    @property
    def resampled_values(self):
        """The resampled values (bootstrap/jackknife samples)."""
        return self.data[1:]

    @property
    def mean(self):
        """Mean of the resampled values (excluding full sample)."""
        return np.mean(self.resampled_values).item()

    @property
    def _std(self):
        """Standard deviation of the resampled values (excluding full sample)."""
        return np.std(self.resampled_values, ddof=1)

    @property
    def error(self):
        """Statistical error estimate."""
        if self.sampling_info.method == "bootstrap":
            res = self._std
        elif self.sampling_info.method == "jackknife":
            # Jackknife error correction
            n = len(self.resampled_values)
            res = self._std * np.sqrt(n - 1)
        else:
            res = self._std
        return res.item()

    @property
    def bootstrap_bias(self) -> float:
        """
        Calculate bootstrap bias estimate.

        Returns:
            Bootstrap bias (mean of resamples - full sample value)

        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != "bootstrap":
            raise ValueError(
                "Bootstrap bias is only available for bootstrap resampling"
            )

        return self.mean - self.full_sample_value

    @property
    def bias_corrected_mean(self) -> float:
        """
        Calculate bias-corrected mean estimate.

        Returns:
            Bias-corrected mean (full sample - bootstrap bias)

        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != "bootstrap":
            raise ValueError(
                "Bias correction is only available for bootstrap resampling"
            )

        bias = self.bootstrap_bias
        return self.full_sample_value - bias

    def to_ufloat(self):
        """
        Convert to uncertainties.ufloat object for PDG formatting.

        Returns:
            uncertainties.ufloat: Object with value and uncertainty

        Raises:
            ImportError: If uncertainties package is not available
        """
        if not UNCERTAINTIES_AVAILABLE:
            raise ImportError(
                "uncertainties package is required for ufloat conversion. Install with: pip install uncertainties"
            )

        if self.is_complex:
            raise ValueError(
                "Complex samplings cannot be converted to ufloat. Use .to_real() first."
            )

        return ufloat(self.full_sample_value, self.error)

    def pdg_format(self, format_spec: str = ".2uS") -> str:
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
        if self.sampling_info.method != "bootstrap":
            raise ValueError(
                "Confidence intervals are only supported for bootstrap resampling"
            )

        # Calculate percentiles for the confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        # Calculate bounds from resampled values
        lower_bound = np.percentile(self.resampled_values, lower_percentile).item()
        upper_bound = np.percentile(self.resampled_values, upper_percentile).item()
        return (lower_bound, upper_bound)

    def bounded(self, lower: float, upper: float) -> "SigmondSampling":
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
            is_complex=self.is_complex,
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
            real_data, self.observable_info, self.sampling_info, is_complex=False
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
            complex_data, self.observable_info, self.sampling_info, is_complex=True
        )

    def as_energy_level(self, force_type: str = "auto", **manual_overrides):
        """
        Convert this sampling to an energy level with enhanced ObservableInfo.

        Args:
            force_type: Force specific type ('single_hadron', 'multi_hadron', 'auto')
            **manual_overrides: Manual attribute overrides

        Returns:
            SigmondSampling with energy level ObservableInfo
        """
        # Import here to avoid circular imports
        from .energy_levels import create_energy_obs_info

        try:
            new_obs_info = create_energy_obs_info(
                self.observable_info, force_type, **manual_overrides
            )
        except Exception as e:
            raise ValueError(
                f"Error converting to energy level: {e}. "
                f"ObservableInfo: {self.observable_info}, "
                f"force_type: {force_type}, overrides: {manual_overrides}"
            ) from e

        # Return new sampling with energy level observable info
        return SigmondSampling(
            self.data, new_obs_info, self.sampling_info, self.is_complex
        )

    def _check_compatible(self, others: set["SigmondSampling"]):
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
            raise ValueError(
                "Incompatible samplings found:\n" + "\n".join(error_messages)
            )

        return True

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
            unwrapped_data, self.observable_info, self.sampling_info, self.is_complex
        )

    @property
    def latex_str(self):
        return f"{self.observable_info.latex_str} = {self.pdg_format()}"

    def __lt__(self, other):
        return self.full_sample_value < (
            other.full_sample_value if isinstance(other, SigmondSampling) else other
        )

    def __le__(self, other):
        return self.full_sample_value <= (
            other.full_sample_value if isinstance(other, SigmondSampling) else other
        )

    def __gt__(self, other):
        return self.full_sample_value > (
            other.full_sample_value if isinstance(other, SigmondSampling) else other
        )

    def __ge__(self, other):
        return self.full_sample_value >= (
            other.full_sample_value if isinstance(other, SigmondSampling) else other
        )

    def __eq__(self, other):
        """Check equality based on observable_info, sampling_info, and is_complex."""
        if not isinstance(other, SigmondSampling):
            return False
        return (
            self.observable_info == other.observable_info
            and self.sampling_info == other.sampling_info
            and self.is_complex == other.is_complex
        )

    def __hash__(self):
        """Make SigmondSampling hashable for use as dictionary keys."""
        return hash(
            (
                self.observable_info,
                self.sampling_info.method,
                self.sampling_info.num_resamplings,
                self.is_complex,
            )
        )

    def __repr__(self):
        return f"SigmondSampling(name='{str(self.observable_info)}', full={self.full_sample_value:.6f}, mean={self.mean:.6f}, error={self.error:.6f})"

    def __str__(self):
        return f"{self.full_sample_value:.6f} ± {self.error:.6f}"

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Union["SigmondSampling", Any]:
        """Implements NumPy ufunc protocol for SigmondSampling."""
        if method != "__call__" or kwargs:
            return NotImplemented

        samplings = {arg for arg in inputs if isinstance(arg, SigmondSampling)}
        other_samplings = samplings - {self}

        self._check_compatible(other_samplings)

        new_inputs = [
            arg.data if isinstance(arg, SigmondSampling) else arg for arg in inputs
        ]

        result_data = ufunc(*new_inputs)

        if result_data is NotImplemented:
            return NotImplemented

        is_complex = (
            any(s.is_complex for s in samplings)
            or any(np.iscomplexobj(arg) for arg in new_inputs)
            or np.iscomplexobj(result_data)
        )

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
            elif all(
                s.observable_info.ensemble_info == first_info.ensemble_info
                for s in samplings
            ):
                # Same ensemble_info - create a new observable_info with mixed name
                result_observable_info = ObservableInfo(
                    "mixed_operation", 0, "n", "re", first_info.ensemble_info
                )
            else:
                # Mixed ensembles - use default ensemble
                result_observable_info = ObservableInfo(
                    "mixed_operation", 0, "n", "re", DEFAULT_ENSEMBLE
                )

        return SigmondSampling(
            result_data,
            result_observable_info,
            self.sampling_info,
            is_complex=is_complex,
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
