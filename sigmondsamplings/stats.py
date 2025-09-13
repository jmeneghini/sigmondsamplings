"""
Statistical analysis tools for Sigmond samplings.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from .sampling import SigmondSampling, EnsembleInfo, ObservableInfo, SamplingInfo


class SamplingStats:
    """Statistical analysis tools for multiple SigmondSampling objects."""
    
    def __init__(self, samplings: List[SigmondSampling] = None):
        """
        Initialize with a list of SigmondSampling objects.
        
        Args:
            samplings: List of SigmondSampling objects to analyze
        """
        self.samplings = samplings
        self._check_consistency()
    
    def _check_consistency(self):
        """Check that all samplings have consistent sampling info."""
        if not self.samplings:
            return
        reference = self.samplings[0]
        for i, sampling in enumerate(self.samplings[1:], 1):
            if sampling.sampling_info != reference.sampling_info:
                raise ValueError(f"Sampling {i} has different sampling info than sampling 0")
            
            if len(sampling.data) != len(reference.data):
                raise ValueError(f"Sampling {i} has different data length than sampling 0")

    @property
    def observable_infos(self) -> List[ObservableInfo]:
        """Get the observable info from all samplings."""
        return [s.observable_info for s in self.samplings]
    
    @property
    def sampling_info(self) -> SamplingInfo:
        """Get the sampling info from all samplings."""
        return self.samplings[0].sampling_info

    @property
    def unique_ensembles(self) -> List[EnsembleInfo]:
        """Get list of unique ensemble infos from all observables."""
        unique = []
        for sampling in self.samplings:
            ensemble_info = sampling.observable_info.ensemble_info
            if ensemble_info not in unique:
                unique.append(ensemble_info)
        return unique
    
    @property
    def num_observables(self) -> int:
        """Number of observables."""
        return len(self.samplings)
    
    @property
    def num_samples(self) -> int:
        """Number of samples (excluding full sample value)."""
        return len(self.samplings[0].resampled_values)

    def full_samples(self) -> np.ndarray:
        """Get the full sample values of all observables."""
        return np.array([s.full_sample_value for s in self.samplings])

    def means(self) -> np.ndarray:
        """Get the means of all observables."""
        return np.array([s.mean for s in self.samplings])
    
    def sample_means(self) -> np.ndarray:
        """Get the means of all observables (alias for means)."""
        return self.means()
    
    def errors(self) -> np.ndarray:
        """Get the errors of all observables."""
        return np.array([s.error for s in self.samplings])
    
    def sample_errors(self) -> np.ndarray:
        """Get the errors of all observables (alias for errors)."""
        return self.errors()

    def add_samplings(self, new_samplings: List[SigmondSampling] | SigmondSampling):
        """
        Add new SigmondSampling objects to the analysis.

        Args:
            new_samplings: List of SigmondSampling objects to add
        """
        if isinstance(new_samplings, SigmondSampling):
            new_samplings = [new_samplings]
        self.samplings.extend(new_samplings)
        self._check_consistency()

    def update_sampling_index(self, index: int, new_data: np.ndarray):
        """
        Update the data for a specific sampling index. The size of the
        new data array must match the number of SigmondSampling objects
        present. Indexing starts at 0 with the full sample.

        Args:
            index: Resampling index to update
            new_data: New data for the resampling
        """
        if index < 0 or index > self.num_samples:
            raise IndexError("Resampling index out of range")
        if len(new_data) != self.num_observables:
            raise ValueError("New data length must match number of observables")

        for i, sampling in enumerate(self.samplings):
            sampling.resampled_values[index] = new_data[i]

    def covariance_matrix(self) -> np.ndarray:
        """
        Calculate the covariance matrix between all observables.
        
        For samplings from different ensembles, covariance is zero.
        
        Returns:
            Covariance matrix (num_observables x num_observables)
        """
        n_obs = self.num_observables
        cov_matrix = np.zeros((n_obs, n_obs))
        
        # Fill diagonal and off-diagonal elements
        for i in range(n_obs):
            for j in range(i, n_obs):
                cov_val = self.covariance(i, j)
                cov_matrix[i, j] = cov_val
                if i != j:
                    cov_matrix[j, i] = cov_val
        
        return cov_matrix
    
    def covariance_matrix_cond_num(self) -> float:
        """
        Calculate the condition number of the covariance matrix.
        
        Returns:
            Condition number of the covariance matrix
        """
        cov_matrix = self.covariance_matrix()
        return np.linalg.cond(cov_matrix)
    
    def correlation_matrix(self) -> np.ndarray:
        """
        Calculate the correlation matrix between all observables.
        
        Returns:
            Correlation matrix (num_observables x num_observables)
        """
        cov_matrix = self.covariance_matrix()
        
        # Extract standard deviations
        stds = np.sqrt(np.diag(cov_matrix))
        
        # Calculate correlation matrix
        corr_matrix = cov_matrix / np.outer(stds, stds)
        
        return corr_matrix
    
    def correlation_matrix_cond_num(self) -> float:
        """
        Calculate the condition number of the correlation matrix.
        
        Returns:
            Condition number of the correlation matrix
        """
        corr_matrix = self.correlation_matrix()
        return np.linalg.cond(corr_matrix)
    
    def covariance(self, obs1_idx: int, obs2_idx: int) -> float:
        """
        Calculate covariance between two specific observables.
        
        For samplings from different ensembles, returns zero.
        
        Args:
            obs1_idx: Index of first observable
            obs2_idx: Index of second observable
            
        Returns:
            Covariance value
        """
        if obs1_idx >= len(self.samplings) or obs2_idx >= len(self.samplings):
            raise IndexError("Observable index out of range")
        
        sampling1 = self.samplings[obs1_idx]
        sampling2 = self.samplings[obs2_idx]
        
        # Return zero covariance for different ensembles
        if sampling1.observable_info.ensemble_info != sampling2.observable_info.ensemble_info:
            return 0.0
        
        # Same diagonal element - return variance
        if obs1_idx == obs2_idx:
            return sampling1.error ** 2
        
        data1 = sampling1.resampled_values
        data2 = sampling2.resampled_values
        
        cov = np.cov(data1, data2, ddof=1)[0, 1]
        
        # Apply jackknife correction if needed
        if sampling1.sampling_info.method == 'jackknife':
            n = len(data1)
            cov *= (n - 1)
        
        return cov
    
    def correlation(self, obs1_idx: int, obs2_idx: int) -> float:
        """
        Calculate correlation between two specific observables.
        
        Args:
            obs1_idx: Index of first observable
            obs2_idx: Index of second observable
            
        Returns:
            Correlation coefficient
        """
        cov = self.covariance(obs1_idx, obs2_idx)
        err1 = self.samplings[obs1_idx].error
        err2 = self.samplings[obs2_idx].error
        
        return cov / (err1 * err2)
    
    def chi_squared(self, theory_values: np.ndarray, 
                   use_correlation: bool = True) -> Tuple[float, int]:
        """
        Calculate chi-squared with respect to theory values.
        
        Args:
            theory_values: Array of theoretical values to compare against
            use_correlation: Whether to use full covariance matrix (True) or 
                           just diagonal errors (False)
            
        Returns:
            Tuple of (chi_squared_value, degrees_of_freedom)
        """
        if len(theory_values) != self.num_observables:
            raise ValueError("Theory values length must match number of observables")
        
        means = self.means()
        diff = means - theory_values
        
        if use_correlation:
            cov_matrix = self.covariance_matrix()
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                chi_sq = diff @ inv_cov @ diff
            except np.linalg.LinAlgError:
                # Fall back to diagonal if matrix is singular
                errors = self.errors()
                chi_sq = np.sum((diff / errors) ** 2)
        else:
            errors = self.errors()
            chi_sq = np.sum((diff / errors) ** 2)
        
        dof = self.num_observables
        return chi_sq, dof
    
    def effective_sample_size(self) -> np.ndarray:
        """
        Estimate effective sample size for each observable using autocorrelation.
        
        Returns:
            Array of effective sample sizes
        """
        eff_sizes = []
        
        for sampling in self.samplings:
            data = sampling.resampled_values
            
            # Simple autocorrelation estimate
            n = len(data)
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[n-1:] / autocorr[n-1]
            
            # Find integrated autocorrelation time
            tau_int = 0.5
            for i in range(1, min(n//4, len(autocorr))):
                tau_int += autocorr[i]
                if i >= 2 * tau_int:
                    break
            
            eff_size = n / (2 * tau_int)
            eff_sizes.append(max(1, eff_size))
        
        return np.array(eff_sizes)
    
    def chi_squared_by_samplings(self, theory_values: Union[np.ndarray, List[SigmondSampling]], 
                                use_correlation: bool = True) -> SigmondSampling:
        """
        Calculate chi-squared for each resampling, treating theory values as samples.
        
        Args:
            theory_values: Either array of theoretical values (same for all resamplings)
                         or list of SigmondSampling objects for varying theory values
            use_correlation: Whether to use full covariance matrix (True) or 
                           just diagonal errors (False)
            
        Returns:
            SigmondSampling object containing chi-squared values for each resampling
        """
        # Validate theory values input
        if isinstance(theory_values, np.ndarray):
            if len(theory_values) != self.num_observables:
                raise ValueError("Theory values length must match number of observables")
            # Convert to constant theory for all resamplings
            theory_data = np.tile(theory_values, (self.num_samples + 1, 1)).T
        elif isinstance(theory_values, list):
            if len(theory_values) != self.num_observables:
                raise ValueError("Number of theory samplings must match number of observables")
            # Check that all theory samplings are compatible
            if not all(isinstance(t, SigmondSampling) for t in theory_values):
                raise ValueError("All theory values must be SigmondSampling objects")
            if not all(len(t.data) == len(self.samplings[0].data) for t in theory_values):
                raise ValueError("All theory samplings must have same length as data samplings")
            theory_data = np.array([t.data for t in theory_values])
        else:
            raise ValueError("Theory values must be numpy array or list of SigmondSampling objects")
        
        # Get the covariance matrix once (uses existing calculation with proper corrections)
        if use_correlation:
            try:
                cov_matrix = self.covariance_matrix()
                inv_cov = np.linalg.inv(cov_matrix)
                use_covariance = True
            except np.linalg.LinAlgError:
                # Fall back to diagonal if matrix is singular
                use_covariance = False
        else:
            use_covariance = False
        
        # Use the existing error calculation from SigmondSampling objects
        errors = self.errors()
        
        chi_squared_values = []
        
        # Calculate chi-squared for full sample (index 0) and each resampling
        for sample_idx in range(self.num_samples + 1):
            # Get data values for this sample
            data_values = np.array([s.data[sample_idx] for s in self.samplings])
            theory_vals = theory_data[:, sample_idx]
            
            diff = data_values - theory_vals
            
            if use_covariance:
                chi_sq = diff @ inv_cov @ diff
            else:
                # Use diagonal errors (already properly calculated by SigmondSampling.error)
                chi_sq = np.sum((diff / errors) ** 2)
            
            chi_squared_values.append(chi_sq)
        
        # Create chi-squared sampling object
        chi_sq_data = np.array(chi_squared_values)
        
        observable_info = ObservableInfo(
            name='chi_squared',
            index=0,
            op_type='n',
            re_im='re',
            ensemble_info=self.ensemble_info,
        )
        
        # Use the same ensemble and sampling info as the input data
        chi_sq_sampling = SigmondSampling(
            data=chi_sq_data,
            observable_info=observable_info,
            sampling_info=self.samplings[0].sampling_info,
            is_complex=False
        )
        
        return chi_sq_sampling

    def fit_function(self, x_values: Union[np.ndarray, List[SigmondSampling]], model_func: Callable, 
                    initial_params: np.ndarray,
                    param_bounds: Optional[List[Tuple[float, float]]] = None,
                    use_correlation: bool = True,
                    method: str = 'minimize') -> Dict[str, SigmondSampling]:
        """
        Fit a function to the observables with proper error propagation.
        
        Args:
            x_values: Either array of fixed x values or list of SigmondSampling objects
                     for x values with uncertainties
            model_func: Function f(x, params) that takes x values and parameter array
            initial_params: Initial guess for parameters
            param_bounds: Optional bounds for parameters [(min, max), ...]
            use_correlation: Whether to use correlation matrix in fitting
            method: Optimization method ('minimize' or 'curve_fit')
            
        Returns:
            Dictionary with fitted parameters as SigmondSampling objects
        """
        # Handle both fixed x-values and x-values with uncertainties
        if isinstance(x_values, list) and all(isinstance(x, SigmondSampling) for x in x_values):
            if len(x_values) != self.num_observables:
                raise ValueError("Number of x samplings must match number of observables")
            # Check compatibility with y-data samplings
            for x_samp in x_values:
                if (x_samp.sampling_info != self.samplings[0].sampling_info or
                    len(x_samp.data) != len(self.samplings[0].data)):
                    raise ValueError("X samplings must be compatible with Y samplings")
            x_has_uncertainty = True
            x_samplings = x_values
            # Use mean x values for initial fit
            x_array = np.array([x.mean for x in x_values])
        else:
            x_array = np.array(x_values)
            if len(x_array) != self.num_observables:
                raise ValueError("Number of x values must match number of observables")
            x_has_uncertainty = False
            x_samplings = None
            
        try:
            from scipy.optimize import minimize, curve_fit
        except ImportError:
            raise ImportError("scipy is required for fitting. Install with: pip install scipy")
        
        # Define chi-squared function using existing infrastructure
        def chi_squared(params):
            theory_vals = model_func(x_array, params)
            chi_sq, _ = self.chi_squared(theory_vals, use_correlation)
            return chi_sq
        
        # Fit using full sample (mean values)
        if method == 'minimize':
            result = minimize(chi_squared, initial_params, bounds=param_bounds)
            print(initial_params, param_bounds)
            if not result.success:
                raise RuntimeError(f"Fitting failed: {result.message}")
            best_params = result.x
        elif method == 'curve_fit':
            y_data = self.means()
            y_errors = self.errors()
            if use_correlation:
                try:
                    cov_matrix = self.covariance_matrix()
                    sigma = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    sigma = y_errors
            else:
                sigma = y_errors
                
            best_params, _ = curve_fit(
                lambda x, *p: model_func(x, np.array(p)),
                x_array, y_data,
                p0=initial_params, sigma=sigma,
                absolute_sigma=True,
                bounds=param_bounds if param_bounds else (-np.inf, np.inf)
            )
        else:
            raise ValueError("Method must be 'minimize' or 'curve_fit'")
        
        # Now fit each resampling to get parameter distributions
        num_params = len(best_params)
        param_samples = []
        
        # Initialize parameter arrays with full sample values
        for p_idx in range(num_params):
            param_samples.append([best_params[p_idx]])
        
        # Fit each resampling
        for sample_idx in range(self.num_samples):
            # Get x values for this sample
            if x_has_uncertainty:
                x_sample = np.array([x.resampled_values[sample_idx] for x in x_samplings])
            else:
                x_sample = x_array
            
            # Get resampled data for this sample
            y_sample = np.array([s.resampled_values[sample_idx] for s in self.samplings])
            
            # Define chi-squared for this sample using the existing infrastructure approach
            def chi_squared_sample(params):
                theory_vals = model_func(x_sample, params)
                diff = y_sample - theory_vals
                
                if use_correlation:
                    try:
                        cov_matrix = self.covariance_matrix()
                        inv_cov = np.linalg.inv(cov_matrix)
                        return diff @ inv_cov @ diff
                    except np.linalg.LinAlgError:
                        errors = self.errors()
                        return np.sum((diff / errors) ** 2)
                else:
                    errors = self.errors()
                    return np.sum((diff / errors) ** 2)
            
            # Fit this sample
            try:
                if method == 'minimize':
                    result_sample = minimize(chi_squared_sample, best_params, bounds=param_bounds)
                    sample_params = result_sample.x if result_sample.success else best_params
                elif method == 'curve_fit':
                    y_errors = self.errors()
                    sample_params, _ = curve_fit(
                        lambda x, *p: model_func(x, np.array(p)),
                        x_sample, y_sample,
                        p0=best_params, sigma=y_errors,
                        absolute_sigma=True,
                        bounds=param_bounds if param_bounds else (-np.inf, np.inf)
                    )
                
                # Store parameter values for this sample
                for p_idx, param_val in enumerate(sample_params):
                    param_samples[p_idx].append(param_val)
                    
            except Exception:
                # If any fit fails, use the best fit parameters
                for p_idx in range(num_params):
                    param_samples[p_idx].append(best_params[p_idx])
        
        # Create SigmondSampling objects for each parameter
        fitted_params = {}
        for p_idx in range(num_params):
            param_data = np.array(param_samples[p_idx])
            name = f'param_{p_idx}'
            observable_info = ObservableInfo(
                name = name,
                index = 0,
                op_type='n',
                re_im='re',
                ensemble_info=self.ensemble_info,
                )
            fitted_params[f'param_{p_idx}'] = SigmondSampling(
                data=param_data,
                observable_info=observable_info,
                sampling_info=self.samplings[0].sampling_info,
                is_complex=False
            )
        
        return fitted_params
    
    def goodness_of_fit(self, x_values: Union[np.ndarray, List[SigmondSampling]], model_func: Callable, 
                       fitted_params: Dict[str, SigmondSampling],
                       use_correlation: bool = True) -> SigmondSampling:
        """
        Calculate goodness of fit using fitted parameters and existing chi_squared_by_samplings.
        
        Args:
            x_values: Either array of fixed x values or list of SigmondSampling objects
                     for x values with uncertainties
            model_func: The model function used in fitting
            fitted_params: Dictionary of fitted parameters from fit_function
            use_correlation: Whether to use correlation matrix
            
        Returns:
            SigmondSampling object containing chi-squared values
        """
        # Handle both fixed x-values and x-values with uncertainties
        if isinstance(x_values, list) and all(isinstance(x, SigmondSampling) for x in x_values):
            if len(x_values) != self.num_observables:
                raise ValueError("Number of x samplings must match number of observables")
            x_has_uncertainty = True
            x_samplings = x_values
        else:
            x_array = np.array(x_values)
            if len(x_array) != self.num_observables:
                raise ValueError("Number of x values must match number of observables")
            x_has_uncertainty = False
        
        # Extract parameter arrays in order
        param_names = sorted(fitted_params.keys())
        param_samplings = [fitted_params[name] for name in param_names]
        
        # Calculate theory values for all observables and all samples
        theory_samplings = []
        
        for obs_idx in range(self.num_observables):
            theory_values = []
            
            # Calculate theory values for this observable across all samples
            for sample_idx in range(self.num_samples + 1):
                params = np.array([ps.data[sample_idx] for ps in param_samplings])
                
                if x_has_uncertainty:
                    x_val = x_samplings[obs_idx].data[sample_idx]
                else:
                    x_val = x_array[obs_idx]
                
                theory_val = model_func(np.array([x_val]), params)[0]
                theory_values.append(theory_val)
            
            # Create SigmondSampling for this observable's theory values
            theory_sampling = SigmondSampling(
                data=np.array(theory_values),
                ensemble_info=self.ensemble_info,
                sampling_info=self.samplings[0].sampling_info,
                is_complex=False
            )
            theory_samplings.append(theory_sampling)
        
        # Use the existing chi_squared_by_samplings method - it handles everything!
        return self.chi_squared_by_samplings(theory_samplings, use_correlation)
    
    def fit_polynomial(self, x_values: np.ndarray, degree: int, 
                      use_correlation: bool = True) -> Dict[str, SigmondSampling]:
        """
        Convenience method for polynomial fitting.
        
        Args:
            x_values: Array of x values corresponding to each observable
            degree: Degree of polynomial
            use_correlation: Whether to use correlation matrix
            
        Returns:
            Dictionary with fitted coefficients as SigmondSampling objects
        """
        def poly_func(x, params):
            return np.polyval(params, x)
        
        initial_params = np.ones(degree + 1)
        return self.fit_function(x_values, poly_func, initial_params, use_correlation=use_correlation)
    
    def fit_exponential(self, x_values: np.ndarray, use_correlation: bool = True) -> Dict[str, SigmondSampling]:
        """
        Convenience method for exponential fitting: A * exp(-m * x).
        
        Args:
            x_values: Array of x values corresponding to each observable
            use_correlation: Whether to use correlation matrix
            
        Returns:
            Dictionary with 'param_0' (A) and 'param_1' (m) as SigmondSampling objects
        """
        def exp_func(x, params):
            A, m = params
            return A * np.exp(-m * x)
        
        # Initial guess based on data
        y_data = self.means()
        A_guess = y_data[0] if len(y_data) > 0 else 1.0
        m_guess = 0.1
        
        initial_params = np.array([A_guess, m_guess])
        return self.fit_function(x_values, exp_func, initial_params, use_correlation=use_correlation)

    def summary(self) -> Dict:
        """
        Generate a summary of statistical information.
        
        Returns:
            Dictionary containing summary statistics
        """
        return {
            'num_observables': self.num_observables,
            'num_samples': self.num_samples,
            'ensemble': self.ensemble_info.ensemble_name,
            'sampling_method': self.samplings[0].sampling_info.method,
            'means': self.means(),
            'errors': self.errors(),
            'effective_sample_sizes': self.effective_sample_size(),
            'correlation_matrix': self.correlation_matrix()
        } 
    