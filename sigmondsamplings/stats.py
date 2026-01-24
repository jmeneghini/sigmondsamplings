"""
Statistical analysis tools for Sigmond samplings.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Type, TypeVar, Iterable
from .sampling import SigmondSampling, EnsembleInfo, ObservableInfo, SamplingInfo
from .obervable_collection import ObservableCollection
from .ensemble_collection import MultiEnsembleCollection
from functools import cached_property

T = TypeVar("T", bound="SamplingStats")


class SamplingStats(MultiEnsembleCollection):
    """
    Statistical analysis tools for energy-level samplings from multiple ensembles.

    Extends MultiEnsembleEnergyCollection to provide statistical operations like
    covariance matrices, chi-squared calculations, and fitting.

    Inherits from MultiEnsembleCollection with return_type fixed to "numpy".

    Note: This class is IMMUTABLE - filtering/sorting operations return new instances.
    The underlying data is stored as a tuple to prevent modification.
    """

    def __init__(
        self,
        data: Optional[Iterable[SigmondSampling]] = None,
    ):
        """
        Initialize SamplingStats with sampling_info and energy-level validation.

        Args:
            data: Input data - can be:
                - List of SigmondSampling objects with energy-level observables
                - ObservableCollection / MultiEnsembleCollection
                - None for empty collection

        Raises:
            ValueError: If samplings have inconsistent sampling_info or non-energy observables
        """
        super().__init__(data, "numpy")
        # Convert to tuple for immutability
        self._sampling_info = self.sampling_info
        self._data = tuple(self._data)
        self._numpy_data = self.to_numpy()  # Each sampling is column vector

    @classmethod
    def _fast_load(
        cls: Type[T],
        data: List[SigmondSampling],
        return_type: str,
    ) -> T:
        """
        Internal constructor for efficient creation from trusted data.

        Used by filter/sort methods. Preserves SamplingStats type and recaches numpy data.
        """
        instance = cls.__new__(cls)
        instance._data = tuple(data)  # Immutable tuple
        instance._return_type = "numpy"  # Always numpy for SamplingStats
        instance._shared_attr_cache = {}
        instance._numpy_data = instance.to_numpy()
        return instance

    @property
    def num_observables(self) -> int:
        """Number of observables in the collection."""
        return len(self._data)

    @property
    def num_samples(self) -> int:
        """Number of resampling iterations (including full sample at index 0)."""
        return self._numpy_data.shape[1]

    @property
    def array(self) -> np.ndarray:
        """Get data as a 2D numpy array (num_observables x num_samples)."""
        return self._numpy_data

    def __setitem__(self, key, value):
        """Block item assignment to maintain immutability."""
        raise TypeError(
            "SamplingStats is immutable. Use filter() or other methods to create new instances."
        )

    def __repr__(self) -> str:
        """String representation."""
        if not self._data:
            return "SamplingStats(empty)"
        return (
            f"SamplingStats(n_obs={len(self)}, "
            f"n_ensembles={len(self.ensembles)}, "
            f"sampling='{self._sampling_info}')"
        )

    def values_at_index(self, index: int) -> np.ndarray:
        """
        Get values of all observables at a specific resampling index.

        Args:
            index: Resampling index (0 for full sample)

        Returns:
            Array of shape (num_observables,) with values at the specified index
        """
        return self._numpy_data[:, index]

    # -------------------------------------------------------------------------
    # Covariance and Correlation Methods
    # -------------------------------------------------------------------------

    @cached_property
    def inv_cholesky_cov_matrix(self) -> np.ndarray:
        """
        Calculate the inverse Cholesky decomposition of the covariance matrix.

        Returns:
            Inverse Cholesky factor of the covariance matrix
        """
        try:
            L = np.linalg.cholesky(self.cov_matrix)
            inv_L = np.linalg.inv(L)
            return inv_L
        except np.linalg.LinAlgError:
            raise ValueError(
                "Covariance matrix is not positive definite for Cholesky decomposition"
            )

    @cached_property
    def cov_matrix(self) -> np.ndarray:
        """
        Calculate the covariance matrix between all observables.

        For samplings from different ensembles, covariance is zero.

        Returns:
            Covariance matrix (num_observables x num_observables)
        """
        cov_arr = np.zeros((self.num_observables, self.num_observables))
        for i in range(self.num_observables):
            for j in range(i, self.num_observables):
                cov_ij = self.cov(i, j)
                cov_arr[i, j] = cov_ij
                cov_arr[j, i] = cov_ij

        return cov_arr
    
    @cached_property
    def cov_matrix_cond_num(self) -> float:
        """Calculate the condition number of the covariance matrix."""
        cov_matrix = self.cov_matrix
        return np.linalg.cond(cov_matrix)

    @cached_property
    def corr_matrix(self) -> np.ndarray:
        """
        Calculate the correlation matrix between all observables.

        Returns:
            Correlation matrix (num_observables x num_observables)
        """
        cov_matrix = self.cov_matrix
        stds = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stds, stds)
        return corr_matrix

    @cached_property
    def corr_matrix_cond_num(self) -> float:
        """Calculate the condition number of the correlation matrix."""
        return np.linalg.cond(self.corr_matrix)

    def cov(self, obs1_idx: int, obs2_idx: int) -> float:
        """
        Calculate covariance between two specific observables.

        For samplings from different ensembles, returns zero.

        Args:
            obs1_idx: Index of first observable
            obs2_idx: Index of second observable

        Returns:
            Covariance value
        """

        n = len(self._data)
        if obs1_idx >= n or obs2_idx >= n:
            raise IndexError("Observable index out of range")

        sampling1 = self._data[obs1_idx]
        sampling2 = self._data[obs2_idx]

        # Return zero covariance for different ensembles
        if (
            sampling1.observable_info.ensemble_info
            != sampling2.observable_info.ensemble_info
        ):
            return 0.0

        # Same diagonal element - return variance
        if obs1_idx == obs2_idx:
            return sampling1.error**2

        data1 = sampling1.resampled_values
        data2 = sampling2.resampled_values

        cov = np.cov(data1, data2, ddof=1)[0, 1]

        # Apply jackknife correction if needed
        if self._sampling_info and self._sampling_info.method == "jackknife":
            n_samples = len(data1)
            cov *= n_samples - 1

        return cov
    
    def corr(self, obs1_idx: int, obs2_idx: int) -> float:
        """
        Calculate correlation between two specific observables.

        Args:
            obs1_idx: Index of first observable
            obs2_idx: Index of second observable

        Returns:
            Correlation coefficient
        """
        cov = self.cov(obs1_idx, obs2_idx)
        err1 = self._data[obs1_idx].error
        err2 = self._data[obs2_idx].error
        return cov / (err1 * err2)

    def min_and_max_val_with_buffer(
        self, buffer: float = 0.3
    ) -> Tuple[SigmondSampling, SigmondSampling]:
        """
        Get min and max values for plotting, with buffer.

        Args:
            buffer: Fractional buffer to add to min/max range
        Returns:
            Tuple of (min_values, max_values) with buffer applied
        """
        min_val = self.find_data(mode="min")
        max_val = self.find_data(mode="max")
        delta = max_val - min_val
        diff = delta * buffer
        min_buff = min_val - diff
        max_buff = max_val + diff
        return min_buff, max_buff

    # -------------------------------------------------------------------------
    # Chi-Squared and Residual Methods
    # -------------------------------------------------------------------------

    def whitened_residuals(
        self,
        theory_values: np.ndarray,
        use_corr: bool = True,
        resamp_idx: int = 0,
    ) -> np.ndarray:
        """
        Calculate whitened residuals with respect to theory values.

        Args:
            theory_values: Array of theoretical values to compare against
            use_corr: Whether to use full covariance matrix
            resamp_idx: Resampling index to use (0 for full sample)

        Returns:
            Array of whitened residuals
        """
        if len(theory_values) != self.num_observables:
            raise ValueError("Theory values length must match number of observables")
        if resamp_idx < 0 or resamp_idx > self.num_samples:
            raise IndexError("Resampling index out of range")

        obs = self._numpy_data[:, resamp_idx]
        diff = obs - theory_values

        if use_corr:
            cov_matrix = self.cov_matrix
            try:
                L = np.linalg.cholesky(cov_matrix)
                whitened = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    valid_mask = eigenvals > 1e-12 * np.max(eigenvals)
                    if np.sum(valid_mask) == 0:
                        raise np.linalg.LinAlgError("All eigenvalues too small")

                    sqrt_inv_eigenvals = np.zeros_like(eigenvals)
                    sqrt_inv_eigenvals[valid_mask] = 1.0 / np.sqrt(
                        eigenvals[valid_mask]
                    )

                    whitened = eigenvecs @ (
                        sqrt_inv_eigenvals[:, np.newaxis] * (eigenvecs.T @ diff)
                    )
                except np.linalg.LinAlgError:
                    errors = self.val.error
                    whitened = diff / errors
        else:
            errors = self.val.error
            whitened = diff / errors

        return whitened

    def chi_squared(
        self,
        theory_values: np.ndarray,
        use_corr: bool = True,
        resamp_idx: int = 0,
    ) -> float:
        """
        Calculate chi-squared with respect to theory values.

        Args:
            theory_values: Array of theoretical values to compare against
            use_corr: Whether to use full covariance matrix
            resamp_idx: Resampling index to use (0 for full sample)

        Returns:
            Chi-squared value
        """
        whitened = self.whitened_residuals(theory_values, use_corr, resamp_idx)
        return np.sum(whitened**2)

    @cached_property
    def effective_sample_size(self) -> np.ndarray:
        """
        Estimate effective sample size for each observable using autocorrelation.

        Returns:
            Array of effective sample sizes
        """
        eff_sizes = []

        for sampling in self._data:
            data = sampling.resampled_values
            n = len(data)
            autocorr = np.correlate(
                data - np.mean(data), data - np.mean(data), mode="full"
            )
            autocorr = autocorr[n - 1 :] / autocorr[n - 1]

            tau_int = 0.5
            for i in range(1, min(n // 4, len(autocorr))):
                tau_int += autocorr[i]
                if i >= 2 * tau_int:
                    break

            eff_size = n / (2 * tau_int)
            eff_sizes.append(max(1, eff_size))

        return np.array(eff_sizes)

    @staticmethod
    def confidence_ellipse_params(
        x_sampling: SigmondSampling,
        y_sampling: SigmondSampling,
        confidence_level: float = 0.68,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate parameters for a confidence ellipse of correlated 2D data.

        Args:
            x_sampling: SigmondSampling for x-coordinate
            y_sampling: SigmondSampling for y-coordinate
            confidence_level: Confidence level for the ellipse (default: 0.68 = 1σ)

        Returns:
            Tuple of (center_x, center_y, width, height, angle_degrees)
            - center_x, center_y: Center of ellipse (means)
            - width, height: Ellipse dimensions (scaled by chi-squared value)
            - angle_degrees: Rotation angle in degrees
        """
        # Extract resampled values
        x_data = x_sampling.resampled_values
        y_data = y_sampling.resampled_values

        if len(x_data) != len(y_data):
            raise ValueError("x and y samplings must have same number of resamples")

        # Calculate centers (means)
        center_x = x_sampling.mean
        center_y = y_sampling.mean

        # Calculate covariance matrix
        cov_matrix = np.cov(x_data, y_data)

        # Get eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (largest first)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Calculate angle (in degrees)
        angle_rad = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        angle_degrees = np.degrees(angle_rad)

        # Scale ellipse by chi-squared value for confidence level
        # For 2D data, chi-squared distribution with 2 DOF
        from scipy.stats import chi2

        chi2_val = chi2.ppf(confidence_level, df=2)

        # Width and height are 2 * sqrt(eigenvalue * chi2_val)
        width = 2 * np.sqrt(eigenvals[0] * chi2_val)
        height = 2 * np.sqrt(eigenvals[1] * chi2_val)

        return center_x, center_y, width, height, angle_degrees

    def chi_squared_by_samplings(
        self,
        theory_values: Union[np.ndarray, List[SigmondSampling], ObservableCollection],
        use_corr: bool = True,
    ) -> SigmondSampling:
        """
        Calculate chi-squared for each resampling.

        Args:
            theory_values: Theory values - array, list of SigmondSampling, or collection
            use_corr: Whether to use full covariance matrix

        Returns:
            SigmondSampling object containing chi-squared values for each resampling
        """
        if not self._data:
            raise ValueError("Cannot compute chi-squared on empty SamplingStats")

        data_matrix = self._numpy_data
        n_samples = data_matrix.shape[1]

        # Prepare theory values
        if isinstance(theory_values, np.ndarray):
            if len(theory_values) != self.num_observables:
                raise ValueError(
                    "Theory values length must match number of observables"
                )
            theory_data = np.tile(theory_values, (n_samples, 1)).T
        elif isinstance(theory_values, ObservableCollection):
            theory_data = theory_values.to_numpy()
            if theory_data.shape[0] != self.num_observables:
                raise ValueError(
                    "Number of theory samplings must match number of observables"
                )
            if theory_data.shape[1] != n_samples:
                raise ValueError("Theory samplings must have same length as data")
        elif isinstance(theory_values, list):
            if len(theory_values) != self.num_observables:
                raise ValueError(
                    "Number of theory samplings must match number of observables"
                )
            if not all(isinstance(t, SigmondSampling) for t in theory_values):
                raise ValueError("All theory values must be SigmondSampling objects")
            if not all(len(t.data) == n_samples for t in theory_values):
                raise ValueError("Theory samplings must have same length as data")
            theory_data = np.array([t.data for t in theory_values])
        else:
            raise ValueError(
                "Theory values must be array, list, or ObservableCollection"
            )

        # Compute covariance
        if use_corr:
            try:
                cov_matrix = self.cov_matrix
                inv_cov = np.linalg.inv(cov_matrix)
                use_covariance = True
            except np.linalg.LinAlgError:
                use_covariance = False
        else:
            use_covariance = False

        errors = np.array(self.val.error)
        diff_matrix = data_matrix - theory_data

        if use_covariance:
            chi_squared_values = np.einsum(
                "ij,ji->i", diff_matrix.T, inv_cov @ diff_matrix
            )
        else:
            chi_squared_values = np.sum(
                (diff_matrix / errors[:, np.newaxis]) ** 2, axis=0
            )
    
        # Use independent ensemble
        observable_info = ObservableInfo(
            name="chi_squared",
            index=0,
            op_type="n",
            re_im="re",
        )

        return SigmondSampling(
            data=chi_squared_values,
            observable_info=observable_info,
            sampling_info=self._sampling_info,
            is_complex=False,
        )

    # -------------------------------------------------------------------------
    # Fitting Methods
    # -------------------------------------------------------------------------

    def fit_function(
        self,
        x_values: Union[np.ndarray, List[SigmondSampling], ObservableCollection],
        model_func: Callable,
        initial_params: np.ndarray,
        param_bounds: Optional[List[Tuple[float, float]]] = None,
        use_corr: bool = True,
        method: str = "minimize",
    ) -> Dict[str, SigmondSampling]:
        """
        Fit a function to the observables with proper error propagation.

        Args:
            x_values: X values - array, list of SigmondSampling, or collection
            model_func: Function f(x, params)
            initial_params: Initial guess for parameters
            param_bounds: Optional bounds for parameters
            use_corr: Whether to use correlation matrix
            method: 'minimize' or 'curve_fit'

        Returns:
            Dictionary with fitted parameters as SigmondSampling objects
        """
        if not self._data:
            raise ValueError("Cannot fit on empty SamplingStats")

        data_matrix = self._numpy_data
        n_samples = data_matrix.shape[1]

        # Handle x-values
        if isinstance(x_values, ObservableCollection):
            x_matrix = x_values.to_numpy()
            if x_matrix.shape[0] != self.num_observables:
                raise ValueError("Number of x samplings must match observables")
            x_has_uncertainty = True
            x_array = np.array(x_values.val.mean)
        elif isinstance(x_values, list) and all(
            isinstance(x, SigmondSampling) for x in x_values
        ):
            if len(x_values) != self.num_observables:
                raise ValueError("Number of x samplings must match observables")
            x_has_uncertainty = True
            x_matrix = np.array([x.data for x in x_values])
            x_array = np.array([x.mean for x in x_values])
        else:
            x_array = np.array(x_values)
            if len(x_array) != self.num_observables:
                raise ValueError("Number of x values must match observables")
            x_has_uncertainty = False
            x_matrix = None

        try:
            from scipy.optimize import minimize, curve_fit
        except ImportError:
            raise ImportError("scipy is required for fitting")

        # Precompute covariance
        if use_corr:
            try:
                cov_matrix = self.cov_matrix
                inv_cov = np.linalg.inv(cov_matrix)
                use_covariance = True
            except np.linalg.LinAlgError:
                use_covariance = False
        else:
            use_covariance = False

        errors = np.array(self.val.error)

        def chi_squared(params):
            theory_vals = model_func(x_array, params)
            return self.chi_squared(theory_vals, use_corr)

        # Fit full sample
        if method == "minimize":
            result = minimize(chi_squared, initial_params, bounds=param_bounds)
            if not result.success:
                raise RuntimeError(f"Fitting failed: {result.message}")
            best_params = result.x
        elif method == "curve_fit":
            y_data = np.array(self.val.mean)
            sigma = np.sqrt(np.diag(cov_matrix)) if use_covariance else errors

            best_params, _ = curve_fit(
                lambda x, *p: model_func(x, np.array(p)),
                x_array,
                y_data,
                p0=initial_params,
                sigma=sigma,
                absolute_sigma=True,
                bounds=param_bounds if param_bounds else (-np.inf, np.inf),
            )
        else:
            raise ValueError("Method must be 'minimize' or 'curve_fit'")

        # Fit each resampling
        num_params = len(best_params)
        param_samples = [[best_params[p_idx]] for p_idx in range(num_params)]

        for sample_idx in range(1, n_samples):
            x_sample = x_matrix[:, sample_idx] if x_has_uncertainty else x_array
            y_sample = data_matrix[:, sample_idx]

            def chi_squared_sample(params, x=x_sample, y=y_sample):
                theory_vals = model_func(x, params)
                diff = y - theory_vals
                if use_covariance:
                    return diff @ inv_cov @ diff
                else:
                    return np.sum((diff / errors) ** 2)

            try:
                if method == "minimize":
                    result_sample = minimize(
                        chi_squared_sample, best_params, bounds=param_bounds
                    )
                    sample_params = (
                        result_sample.x if result_sample.success else best_params
                    )
                elif method == "curve_fit":
                    sample_params, _ = curve_fit(
                        lambda x, *p: model_func(x, np.array(p)),
                        x_sample,
                        y_sample,
                        p0=best_params,
                        sigma=errors,
                        absolute_sigma=True,
                        bounds=param_bounds if param_bounds else (-np.inf, np.inf),
                    )

                for p_idx, param_val in enumerate(sample_params):
                    param_samples[p_idx].append(param_val)
            except Exception:
                for p_idx in range(num_params):
                    param_samples[p_idx].append(best_params[p_idx])

        # Create output
        ens_info = self.ensemble_info or self._data[0].observable_info.ensemble_info
        fitted_params = {}
        for p_idx in range(num_params):
            param_data = np.array(param_samples[p_idx])
            observable_info = ObservableInfo(
                name=f"param_{p_idx}",
                index=0,
                op_type="n",
                re_im="re",
                ensemble_info=ens_info,
            )
            fitted_params[f"param_{p_idx}"] = SigmondSampling(
                data=param_data,
                observable_info=observable_info,
                sampling_info=self._sampling_info,
                is_complex=False,
            )

        return fitted_params

    def goodness_of_fit(
        self,
        x_values: Union[np.ndarray, List[SigmondSampling], ObservableCollection],
        model_func: Callable,
        fitted_params: Dict[str, SigmondSampling],
        use_corr: bool = True,
    ) -> SigmondSampling:
        """
        Calculate goodness of fit using fitted parameters.

        Args:
            x_values: X values
            model_func: Model function
            fitted_params: Fitted parameters from fit_function
            use_corr: Whether to use correlation matrix

        Returns:
            SigmondSampling with chi-squared values
        """
        if not self._data:
            raise ValueError("Cannot compute goodness of fit on empty SamplingStats")

        n_samples = self._numpy_data.shape[1]

        # Handle x-values
        if isinstance(x_values, ObservableCollection):
            x_matrix = x_values.to_numpy()
            x_has_uncertainty = True
        elif isinstance(x_values, list) and all(
            isinstance(x, SigmondSampling) for x in x_values
        ):
            x_has_uncertainty = True
            x_matrix = np.array([x.data for x in x_values])
        else:
            x_array = np.array(x_values)
            x_has_uncertainty = False
            x_matrix = None

        param_names = sorted(fitted_params.keys())
        param_data = np.array([fitted_params[name].data for name in param_names])

        theory_data = np.zeros((self.num_observables, n_samples))

        for sample_idx in range(n_samples):
            params = param_data[:, sample_idx]
            x_vals = x_matrix[:, sample_idx] if x_has_uncertainty else x_array
            theory_data[:, sample_idx] = model_func(x_vals, params)

        theory_samplings = []
        for obs_idx in range(self.num_observables):
            observable_info = ObservableInfo(
                name=f"theory_{obs_idx}",
                index=0,
                op_type="n",
                re_im="re",
            )
            theory_samplings.append(
                SigmondSampling(
                    data=theory_data[obs_idx],
                    observable_info=observable_info,
                    sampling_info=self._sampling_info,
                    is_complex=False,
                )
            )

        return self.chi_squared_by_samplings(theory_samplings, use_corr)

    def fit_polynomial(
        self, x_values: np.ndarray, degree: int, use_corr: bool = True
    ) -> Dict[str, SigmondSampling]:
        """Convenience method for polynomial fitting."""

        def poly_func(x, params):
            return np.polyval(params, x)

        initial_params = np.ones(degree + 1)
        return self.fit_function(
            x_values, poly_func, initial_params, use_corr=use_corr
        )

    def fit_exponential(
        self, x_values: np.ndarray, use_corr: bool = True
    ) -> Dict[str, SigmondSampling]:
        """Convenience method for exponential fitting: A * exp(-m * x)."""

        def exp_func(x, params):
            A, m = params
            return A * np.exp(-m * x)

        y_data = np.array(self.val.mean)
        A_guess = y_data[0] if len(y_data) > 0 else 1.0
        m_guess = 0.1

        initial_params = np.array([A_guess, m_guess])
        return self.fit_function(
            x_values, exp_func, initial_params, use_corr=use_corr
        )

    def summary(self) -> Dict:
        """Generate a summary of statistical information."""
        return {
            "num_observables": self.num_observables,
            "num_samples": self.num_samples,
            "ensembles": [e.name for e in self.ensembles],
            "sampling_method": (
                self._sampling_info.method if self._sampling_info else None
            ),
            "means": np.array(self.val.mean),
            "errors": np.array(self.val.error),
            "effective_sample_sizes": self.effective_sample_size,
            "correlation_matrix": self.corr_matrix
        }
