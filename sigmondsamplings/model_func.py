"""
Model function wrapper for SigmondSamplings with automatic parameter handling.
"""

import numpy as np
import warnings
from typing import Callable, List, Optional, Union
from .sampling import SigmondSampling, ObservableInfo, SamplingInfo, DEFAULT_ENSEMBLE
from .stats import SamplingStats
import inspect


class SigmondModelFunc:
    """
    A wrapper class for model functions that automatically handles parameter
    sampling and uncertainty propagation for lattice QCD analysis.
    
    This class takes a regular model function f(x, param1, param2, ...) and
    manages the statistical sampling of parameters internally, providing
    automatic error propagation and integration with the SigmondSamplings
    ecosystem.
    """
    
    def __init__(self, func: Callable, 
                 parameter_infos: List[ObservableInfo],
                 sampling_info: SamplingInfo,
                 latex_str: Optional[str] = None,
                 independent_var_latex: Optional[str] = None):
        """
        Initialize the model function wrapper.
        
        Args:
            func: Model function f(x, param1, param2, ...) 
            parameter_infos: List of ObservableInfo for each parameter
            sampling_info: SamplingInfo describing the resampling method
            latex_str: Optional LaTeX string for the function (e.g., r"A e^{-m{VAR}}" where {VAR} gets replaced)
            independent_var_latex: Optional LaTeX string for independent variable (e.g., r"t" for time)
        """
        self.func = func
        self.parameter_infos = parameter_infos
        self.sampling_info = sampling_info
        self.latex_str = latex_str
        self.independent_var_latex = independent_var_latex
        
        # Validate function signature matches parameter count
        sig = inspect.signature(func)
        n_params = len(sig.parameters) - 1  # Subtract x parameter
        if n_params != len(parameter_infos):
            raise ValueError(f"Function expects {n_params} parameters but got {len(parameter_infos)} parameter infos")
    
    @classmethod
    def from_sampling_stats(cls, func: Callable, parameter_stats: SamplingStats, 
                           latex_str: Optional[str] = None,
                           independent_var_latex: Optional[str] = None) -> 'SigmondModelFunc':
        """
        Alternative constructor using a SamplingStats object for parameters.
        
        Args:
            func: Model function f(x, param1, param2, ...)
            parameter_stats: SamplingStats object containing the parameter data
            latex_str: Optional LaTeX string for the function
            independent_var_latex: Optional LaTeX string for independent variable
            
        Returns:
            SigmondModelFunc instance with parameters already set
        """
        # Extract parameter infos and sampling info from the stats object
        parameter_infos = parameter_stats.observable_infos
        sampling_info = parameter_stats.sampling_info
        
        # Create the model instance
        model = cls(func, parameter_infos, sampling_info, latex_str, independent_var_latex)
        
        # Set the parameters directly using the stats object
        model.params = parameter_stats
        model.sampling_info = sampling_info
        model.observable_infos = parameter_stats.observable_infos
        
        return model

    def set_parameters(self, param_data: Union[List[np.ndarray], List[SigmondSampling]]):
        """
        Set the model parameters from fitted data.
        
        Args:
            param_data: Either list of parameter arrays or list of SigmondSampling objects
        """
        if len(param_data) != len(self.parameter_infos):
            raise ValueError("Number of parameter data entries must match number of parameter infos")
        
        samplings = []
        if isinstance(param_data[0], np.ndarray):
            # Convert arrays to SigmondSampling
            for data, info in zip(param_data, self.parameter_infos):
                samplings.append(SigmondSampling(data, info, self.sampling_info))
        elif isinstance(param_data[0], SigmondSampling):
            samplings = param_data
        else:
            raise ValueError("Invalid parameter data format")

        # now we form a stats object and update the initial info if different
        self.params = SamplingStats(samplings)
        if self.sampling_info != self.params.samplings[0].sampling_info:
            # now warn the user
            warnings.warn("Parameter sampling_info updated to match provided parameter data: "
                          f"{self.sampling_info} -> {self.params.samplings[0].sampling_info}")
            self.sampling_info = self.params.samplings[0].sampling_info
            
        # now update observable infos
        self.observable_infos = self.params.observable_infos

    def get_latex_str_with_var(self, var_latex: Optional[str] = None, index: Optional[int] = None) -> str:
        """
        Get the complete LaTeX string with independent variable substituted.
        
        Args:
            var_latex: Optional override for the independent variable LaTeX string
            index: Optional index for the variable substitution

        Returns:
            Complete LaTeX string with variable substituted
        """
        if not self.latex_str:
            return None
        
        # Use provided var_latex, or stored independent_var_latex, or default 'x'
        var_to_use = var_latex or self.independent_var_latex or 'x'
        var_to_use = var_to_use.strip('$')  # Remove $ if present
        if index is not None:
            var_to_use = var_to_use + f"_{{{index}}}"

        # Replace {VAR} placeholder with the actual variable
        if '{VAR}' in self.latex_str:
            return self.latex_str.replace('{VAR}', var_to_use)
        elif '{var}' in self.latex_str:
            return self.latex_str.replace('{var}', var_to_use)
        else:
            # If no placeholder, return as-is (assumes it's already complete)
            return self.latex_str

    @property
    def parameters(self) -> List[SigmondSampling]:
        """Get the parameter SigmondSampling objects."""
        if not hasattr(self, 'params'):
            raise ValueError("Parameters not set. Call set_parameters() first.")
        return self.params.samplings
    
    def get_parameter_dict(self) -> dict:
        """
        Export parameters as dictionary for compatibility with existing infrastructure.
        
        Returns:
            Dictionary mapping parameter names to SigmondSampling objects
        """
        return {info.name: param for info, param in zip(self.parameter_infos, self.parameters)}
    
    def get_parameter_means(self) -> np.ndarray:
        """Get mean values of all parameters."""
        return self.params.means()
    
    def get_parameter_errors(self) -> np.ndarray:
        """Get error estimates of all parameters."""
        return self.params.errors()

    def __call__(self, x_values: Union[np.ndarray, List[SigmondSampling], SigmondSampling], 
                 output_info: Optional[ObservableInfo] = None) -> Union[SigmondSampling, List[SigmondSampling]]:
        """
        Evaluate model at x_values using internal parameters.
        Uses the ufunc nature of SigmondSampling for automatic error propagation.
        
        Args:
            x_values: Input values where to evaluate the model - can be:
                     - np.ndarray of fixed x values  
                     - Single SigmondSampling for x with uncertainties
                     - List of SigmondSampling objects for multiple x with uncertainties
            output_info: Optional ObservableInfo for the result
            
        Returns:
            SigmondSampling or list of SigmondSampling containing model evaluations with uncertainties
        """
        if not hasattr(self, 'params'):
            raise ValueError("Parameters not set. Call set_parameters() first.")
        
        # Handle different input types for x_values
        if isinstance(x_values, SigmondSampling):
            # Single x with uncertainty - use ufunc operations directly
            result = self.func(x_values, *self.params.samplings)
            
            # Update observable info if provided
            if output_info is not None:
                result.observable_info = output_info
            else:
                # Use x_values' observable info to create meaningful result info
                x_info = x_values.observable_info
                result.observable_info = ObservableInfo(
                    f"{self.func.__name__}({x_info.name})", x_info.index, "n", "re",
                    x_info.ensemble_info,
                    latex_str=self.get_latex_str_with_var(x_info.latex_str)
                )
            
            return result
            
        elif isinstance(x_values, list) and all(isinstance(x, SigmondSampling) for x in x_values):
            # Multiple x values with uncertainties
            results = []
            for i, x_val in enumerate(x_values):
                result = self.func(x_val, *self.params.samplings)
                
                # Update observable info - use x_val's observable info when appropriate
                if output_info is not None:
                    info = ObservableInfo(
                        output_info.name, i, output_info.op_type, output_info.re_im,
                        output_info.ensemble_info, output_info.latex_str
                    )
                else:
                    # Use x_val's observable info to create meaningful result info
                    x_info = x_val.observable_info
                    info = ObservableInfo(
                        f"{self.func.__name__}({x_info.name})", x_info.index, "n", "re",
                        x_info.ensemble_info,
                        latex_str=self.get_latex_str_with_var(x_info.latex_str)
                    )
                    print(x_info.latex_str)
                result.observable_info = info
                results.append(result)
            
            return results
            
        else:
            # Fixed x values - convert to numpy array and evaluate point by point
            x_values = np.asarray(x_values)
            if x_values.ndim == 0:
                x_values = np.array([x_values])
                return_single = True
            else:
                return_single = False
            
            results = []
            for i, x_val in enumerate(x_values):
                # Evaluate at fixed x value using parameter uncertainties
                result = self.func(x_val, *self.params.samplings)
                
                # Update observable info
                if output_info is not None:
                    info = ObservableInfo(
                        output_info.name, i, output_info.op_type, output_info.re_im,
                        output_info.ensemble_info, output_info.latex_str
                    )
                else:
                    info = ObservableInfo(
                        f"{self.func.__name__}_result", i, "n", "re",
                        self.parameter_infos[0].ensemble_info,
                        latex_str=self.get_latex_str_with_var(index=i)
                    )
                result.observable_info = info
                results.append(result)
            
            return results[0] if return_single else results
    
    def evaluate_with_uncertainty(self, x_values: Union[np.ndarray, List[SigmondSampling], SigmondSampling], 
                                 confidence_level: float = 0.68) -> tuple:
        """
        Evaluate model with uncertainty bands.
        
        Args:
            x_values: Input values where to evaluate the model (can have uncertainties)
            confidence_level: Confidence level for uncertainty bands
            
        Returns:
            Tuple of (mean_values, lower_bounds, upper_bounds)
        """        
        # Get all evaluations using the main __call__ method
        results = self(x_values)
        
        # Handle single result vs list of results
        if isinstance(results, SigmondSampling):
            mean_val = results.mean
            
            if results.sampling_info.method == 'bootstrap':
                lower, upper = results.confidence_interval(confidence_level)
            else:
                error = results.error
                lower = mean_val - error
                upper = mean_val + error
            
            return mean_val, lower, upper
        
        else:
            # List of results
            means = np.array([r.mean for r in results])
            
            if results[0].sampling_info.method == 'bootstrap':
                intervals = [r.confidence_interval(confidence_level) for r in results]
                lowers = np.array([interval[0] for interval in intervals])
                uppers = np.array([interval[1] for interval in intervals])
            else:
                errors = np.array([r.error for r in results])
                lowers = means - errors
                uppers = means + errors
            
            return means, lowers, uppers

    def evaluate_full_sample(self, x_values: Union[np.ndarray, List[SigmondSampling], SigmondSampling]) -> np.ndarray:
        """
        Evaluate model at full sample points.
        
        Args:
            x_values: Input values where to evaluate the model (can have uncertainties)
            
        Returns:
            Array of shape (n_samples, len(x_values)) containing all evaluations
        """
        # Get all evaluations using the main __call__ method
        if isinstance(x_values, SigmondSampling):
            x_fulls = x_values.full_sample_value
        elif isinstance(x_values, list):
            if all(isinstance(x, SigmondSampling) for x in x_values):
                x_fulls = np.array([x.full_sample_value for x in x_values])
            elif all(isinstance(x, float) for x in x_values):
                x_fulls = np.array(x_values)
        elif isinstance(x_values, np.ndarray):
            x_fulls = x_values
        else:
            raise ValueError("Invalid x_values type")

        y_res = self(x_fulls)
        if isinstance(y_res, SigmondSampling):
            return y_res.full_sample_value
        elif isinstance(y_res, list):
            return np.array([r.full_sample_value for r in y_res])
        else:
            raise ValueError("Invalid y_res type")

    def evaluate_samples(self, x_values: Union[np.ndarray, List[SigmondSampling], SigmondSampling]) -> np.ndarray:
        """
        Return all bootstrap/jackknife sample evaluations.
        
        Args:
            x_values: Input values where to evaluate the model (can have uncertainties)
            
        Returns:
            Array of shape (n_samples, len(x_values)) containing all evaluations
        """
        # Get all evaluations using the main __call__ method
        results = self(x_values)
        
        # Handle single result vs list of results
        if isinstance(results, SigmondSampling):
            # Single x value - return column vector
            return results.data.reshape(-1, 1)  # Shape: (n_samples, 1)
        else:
            # Multiple x values - stack the data arrays
            return np.column_stack([r.data for r in results])  # Shape: (n_samples, len(x_values))
    
    def __repr__(self):
        param_names = [info.name for info in self.parameter_infos]
        return f"SigmondModelFunc({self.func.__name__}, parameters={param_names})"
    
    def __str__(self):
        param_labels = [str(info) for info in self.parameter_infos]
        param_str = ", ".join(param_labels)
        return f"{self.func.__name__}(x; {param_str})"
    


# Convenience functions for common models
def exponential_decay_model(sampling_info: SamplingInfo, 
                           ensemble_info = DEFAULT_ENSEMBLE,
                           latex_str: str = r"A e^{-m{VAR}}",
                           independent_var_latex: str = "t") -> SigmondModelFunc:
    """Create exponential decay model: A * exp(-m * x)"""
    def exp_func(x, A, m):
        return A * np.exp(-m * x)
    
    amp_info = ObservableInfo("amplitude", 0, "n", "re", ensemble_info, r"$A$")
    mass_info = ObservableInfo("mass", 0, "n", "re", ensemble_info, r"$m$")
    
    return SigmondModelFunc(exp_func, [amp_info, mass_info], sampling_info, latex_str, independent_var_latex)


def polynomial_model(degree: int, sampling_info: SamplingInfo,
                    ensemble_info = DEFAULT_ENSEMBLE, 
                    latex_str: Optional[str] = None,
                    independent_var_latex: str = "x") -> SigmondModelFunc:
    """Create polynomial model of specified degree"""
    def poly_func(x, *coeffs):
        return np.polyval(coeffs, x)
    
    param_infos = [
        ObservableInfo(f"coeff_{i}", 0, "n", "re", ensemble_info, f"$c_{i}$")
        for i in range(degree + 1)
    ]
    
    if latex_str is None:
        latex_str = "P({VAR})" if degree > 1 else r"c_0 + c_1 {VAR}"
    
    return SigmondModelFunc(poly_func, param_infos, sampling_info, latex_str, independent_var_latex)


def gaussian_model(sampling_info: SamplingInfo,
                  ensemble_info = DEFAULT_ENSEMBLE,
                  latex_str: str = r"A e^{-\frac{({VAR}-\mu)^2}{2\sigma^2}}",
                  independent_var_latex: str = "x") -> SigmondModelFunc:
    """Create Gaussian model: A * exp(-(x-mu)^2 / (2*sigma^2))"""
    def gauss_func(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    amp_info = ObservableInfo("amplitude", 0, "n", "re", ensemble_info, r"$A$")
    mean_info = ObservableInfo("mean", 0, "n", "re", ensemble_info, r"$\mu$")
    sigma_info = ObservableInfo("sigma", 0, "n", "re", ensemble_info, r"$\sigma$")
    
    return SigmondModelFunc(gauss_func, [amp_info, mean_info, sigma_info], sampling_info, latex_str, independent_var_latex)