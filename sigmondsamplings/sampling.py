"""
Core sampling classes for handling Sigmond samplings data.
"""

import numpy as np

from .data import SigmondData
from .info import INDEP_ENSEMBLE, EnsembleInfo, ObservableInfo, SamplingInfo  # noqa: F401


class SigmondSampling(SigmondData):
    """
    Jackknife/bootstrap samplings: one observable (0-d) or an N-D array of them.

    Along the sample axis (always last), element 0 is the full-sample value and
    elements 1: are the resamples. Statistics, arithmetic, labels and formatting
    live on :class:`SigmondData`.
    """

    def __init__(
        self,
        data: np.ndarray | list,
        observable_info: ObservableInfo,
        sampling_info: SamplingInfo,
        is_complex: bool = False,
    ):

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

    @classmethod
    def from_single_value(
        cls,
        value: float | complex,
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


    def _kind_key(self) -> tuple:
        return (self.observable_info, self.sampling_info, self.is_complex)

    @property
    def num_resamplings(self) -> int:
        return int(self.data.shape[-1]) - 1

    @property
    def resampled_values(self):
        """The resampled values (bootstrap/jackknife samples), sample axis last."""
        return self.data[..., 1:]

    @property
    def bootstrap_bias(self):
        """
        Bootstrap bias estimate: mean of the resamples minus the full-sample value.

        Raises:
            ValueError: If sampling method is not bootstrap
        """
        return self._stat(self._analyzer.bias)

    @property
    def bias_corrected_mean(self):
        """
        Bias-corrected mean estimate: full sample minus the bootstrap bias.

        Raises:
            ValueError: If sampling method is not bootstrap
        """
        if self.sampling_info.method != "bootstrap":
            raise ValueError("Bias correction is only available for bootstrap resampling")
        return self.full_sample_value - self.bootstrap_bias

    def to_ufloat(self, resamp_idx: int = 0) -> "ufloat":  # noqa: F821
        """
        Convert to uncertainties.ufloat object for PDG formatting.

        Args:
            resamp_idx: Index of the data to use (default 0 = full sample)

        Raises:
            ImportError: If uncertainties package is not available
            ValueError: If sampling is complex
        """
        return self._ufloat(self._require_scalar("to_ufloat").data[resamp_idx])

    def pdg_format(
        self, format_spec: str = ".2uS", resamp_idx=0, *, central: str | None = None
    ) -> str:
        """
        Format value and error using PDG conventions via uncertainties package.

        Args:
            format_spec: Format specification for uncertainties package
                       Common options:
                       - '.1uS': Shorthand notation like 1.23(4)
                       - '.1uP': Pretty-print with ± symbol
                       - '.1uL': LaTeX notation
                       - '.1ue': Scientific notation
            resamp_idx: Index of the data to use (default 0 = full sample)
            central: A :meth:`central` metric shown instead of ``resamp_idx``'s
                value, e.g. ``"mean"``

        Returns:
            str: Formatted string using PDG conventions

        Complex samplings are formatted as ``re ± im i``, each part with its own error.

        Raises:
            ImportError: If uncertainties package is not available
        """
        if central is not None:
            return super().pdg_format(format_spec, central=central)
        if self.is_complex:
            return self._complex_pdg_format(format_spec, resamp_idx=resamp_idx)
        ufloat_obj = self.to_ufloat(resamp_idx=resamp_idx)
        return f"{ufloat_obj:{format_spec}}"

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

        return self.with_observable_info(new_obs_info)

    def ref_observable_info(self, samp: "SigmondSampling") -> ObservableInfo:
        """Observable metadata this sampling would carry as a ratio to *samp*.

        Split out of :meth:`create_ref_sampling` so callers can predict the result
        without paying for the division - notably the idempotency check in
        ``EnergyLevelMixin.create_ref``.
        """
        from .energy_levels import EnergyObsInfo, SHEnergyObsInfo
        from .arithmetic import DerivedObsInfo, arithmetic_info
        from .metadata import AtObsInfo

        if isinstance(self.observable_info, (DerivedObsInfo, AtObsInfo)):
            return arithmetic_info("reference", [self.observable_info, samp.observable_info])

        new_obs_info = self.observable_info.copy()
        if isinstance(new_obs_info, EnergyObsInfo):
            new_obs_info.ref_particle = (
                samp.observable_info.particle or "ref"
                if isinstance(samp.observable_info, SHEnergyObsInfo)
                else "ref"
            )
            new_obs_info.update_name(strict=False)
        else:
            new_obs_info.name = f"{self.observable_info.name}_ref"
        return new_obs_info

    def create_ref_sampling(self, samp: "SigmondSampling") -> "SigmondSampling":
        """Create a reference sampling for the given particle."""
        new_ref = self / samp
        # Build from self.observable_info (not new_ref's) to preserve the subclass type.
        # Generic arithmetic describes a derived ratio; this domain method
        # explicitly establishes its energy/reference role.
        if new_ref.ndim:
            from .arithmetic import arithmetic_info
            new_ref.observable_info = arithmetic_info("reference", [self.observable_info, samp.observable_info])
            infos = np.empty(new_ref.shape, dtype=object)
            for index in np.ndindex(new_ref.shape):
                positions = dict(zip(new_ref.dims, index))
                numerator = self.isel(**{dim: positions[dim] for dim in self.dims}) if self.ndim else self
                denominator = samp.isel(**{dim: positions[dim] for dim in samp.dims}) if samp.ndim else samp
                infos[index] = numerator.ref_observable_info(denominator)
            new_ref = new_ref._with_replaced_infos(infos)
        else:
            new_ref.observable_info = self.ref_observable_info(samp)
        return new_ref

    def __repr__(self):
        if self.ndim:
            return self._array_repr()
        return (
            f"SigmondSampling(name='{self.observable_info}', value={self.pdg_format()}, "
            f"method='{self.sampling_info.method}', N={self.sampling_info.num_resamplings})"
        )
