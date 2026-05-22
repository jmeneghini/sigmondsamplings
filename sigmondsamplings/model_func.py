"""
Model function wrapper for SigmondSamplings with automatic parameter handling.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from .obervable_collection import ObservableCollection
from .sampling import DEFAULT_ENSEMBLE, ObservableInfo, SamplingInfo, SigmondSampling

if TYPE_CHECKING:
    from .stats import SamplingStats


METRIC_LATEX: dict[str, str] = {
    "chi_squared": r"\chi^2",
    "chi2_per_dof": r"\chi^2/\mathrm{dof}",
    "dof": r"\mathrm{dof}",
    "aic": r"\mathrm{AIC}",
    "bic": r"\mathrm{BIC}",
    "aicc": r"\mathrm{AICc}",
    "goodness_of_fit": "Q",
}


def _format_metric(label: str, value: float | int, *, integer: bool = False) -> str:
    label = label.strip("$")
    if integer:
        return rf"${label} = {int(value):d}$"
    return rf"${label} = {float(value):.4g}$"


class SigmondModelFunc:
    """
    A wrapper class for model functions that automatically handles parameter
    sampling and uncertainty propagation for lattice QCD analysis.

    This class takes a regular model function f(x, param1, param2, ...) and
    manages the statistical sampling of parameters internally, providing
    automatic error propagation.

    The parameters are stored in an ObservableCollection, providing fast
    filtering and querying capabilities. Access parameters via:
        - model.params.val.mean  # Parameter means
        - model.params.val.error  # Parameter errors
        - model.params.to_dict()  # Dict mapping ObsInfo -> SigmondSampling
    """

    def __init__(
        self,
        func: Callable,
        parameter_infos: list[ObservableInfo] | ObservableCollection,
        sampling_info: SamplingInfo | None = None,
        latex_str: str | None = None,
        independent_var_latex: str | None = None,
    ):
        """
        Initialize the model function wrapper.

        Args:
            func: Model function f(x, param1, param2, ...)
            parameter_infos: List of ObservableInfo for each parameter, or an
                ObservableCollection containing fitted parameter samplings.
            sampling_info: SamplingInfo describing the resampling method. Required
                when parameter_infos is a list; inferred when an ObservableCollection
                is provided.
            latex_str: Optional LaTeX string for the function (e.g., r"A e^{-m{VAR}}" where {VAR} gets replaced)
            independent_var_latex: Optional LaTeX string for independent variable (e.g., r"t" for time)
        """
        parameter_collection = None
        if isinstance(parameter_infos, ObservableCollection):
            parameter_collection = parameter_infos
            sampling_info = parameter_collection.shared_attr("sampling_info", strict=True)
            parameter_infos = [s.observable_info for s in parameter_collection]
        elif sampling_info is None:
            raise ValueError("sampling_info is required when parameter_infos is not a collection")

        self.func = func
        self._initial_parameter_infos = list(parameter_infos)
        self.sampling_info = sampling_info
        self.latex_str = latex_str
        self.independent_var_latex = independent_var_latex

        self._validate_function_signature(len(self._initial_parameter_infos))
        if parameter_collection is not None:
            self.params = parameter_collection

    def _validate_function_signature(self, n_params: int) -> None:
        """Validate fixed-arity model functions while allowing ``*params`` models."""
        sig = inspect.signature(self.func)
        parameters = list(sig.parameters.values())
        if not parameters:
            raise ValueError("Model function must accept an independent variable argument")

        model_params = parameters[1:]
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in model_params):
            return

        positional = [
            p
            for p in model_params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required_keyword_only = [
            p
            for p in model_params
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is inspect.Signature.empty
        ]
        if required_keyword_only:
            names = ", ".join(p.name for p in required_keyword_only)
            raise ValueError(f"Model function has unsupported required keyword-only parameters: {names}")

        if len(positional) != n_params:
            raise ValueError(
                f"Function expects {len(positional)} parameters but got {n_params} parameter infos"
            )

    @property
    def parameter_infos(self) -> list[ObservableInfo]:
        """Get parameter ObservableInfo objects."""
        if hasattr(self, "params"):
            return [s.observable_info for s in self.params]
        return self._initial_parameter_infos

    @classmethod
    def from_observable_collection(
        cls,
        func: Callable,
        parameter_collection: ObservableCollection,
        latex_str: str | None = None,
        independent_var_latex: str | None = None,
    ) -> SigmondModelFunc:
        """
        Alternative constructor using an ObservableCollection for parameters.

        Args:
            func: Model function f(x, param1, param2, ...)
            parameter_collection: ObservableCollection containing the parameter data
            latex_str: Optional LaTeX string for the function
            independent_var_latex: Optional LaTeX string for independent variable

        Returns:
            SigmondModelFunc instance with parameters already set
        """
        return cls(func, parameter_collection, None, latex_str, independent_var_latex)

    def set_parameters(
        self,
        param_data: list[np.ndarray] | list[SigmondSampling] | ObservableCollection,
    ):
        """
        Set the model parameters from fitted data.

        Args:
            param_data: Either list of parameter arrays, list of SigmondSampling objects,
                       or an ObservableCollection
        """
        # Handle ObservableCollection input
        if isinstance(param_data, ObservableCollection):
            self.params = param_data
            sampling_info = param_data.shared_attr("sampling_info", strict=True)
            if self.sampling_info != sampling_info:
                warnings.warn(
                    "Parameter sampling_info updated to match provided parameter data: "
                    f"{self.sampling_info} -> {sampling_info}"
                )
                self.sampling_info = sampling_info
            return

        # Handle list input
        if len(param_data) != len(self.parameter_infos):
            raise ValueError(
                "Number of parameter data entries must match number of parameter infos"
            )
        if not param_data:
            raise ValueError("Parameter data cannot be empty")

        samplings = []
        if isinstance(param_data[0], np.ndarray):
            # Convert arrays to SigmondSampling
            for data, info in zip(param_data, self.parameter_infos):
                samplings.append(SigmondSampling(data, info, self.sampling_info))
        elif isinstance(param_data[0], SigmondSampling):
            samplings = param_data
        else:
            raise ValueError("Invalid parameter data format")

        # Create ObservableCollection and update the initial info if different
        self.params = ObservableCollection(samplings)
        first_sampling = next(iter(self.params), None)
        if first_sampling is not None and self.sampling_info != first_sampling.sampling_info:
            warnings.warn(
                "Parameter sampling_info updated to match provided parameter data: "
                f"{self.sampling_info} -> {first_sampling.sampling_info}"
            )
            self.sampling_info = first_sampling.sampling_info

    def format_params(self) -> list[str]:
        """Format fitted parameters using each sampling's canonical LaTeX string."""
        if not hasattr(self, "params"):
            raise AttributeError(
                "Model has no fitted params; build it via fit_result.model_func() first"
            )
        return [s.latex_str for s in self.params]

    def format_metrics(
        self,
        stats: SamplingStats,
        x_data: Iterable[float] | np.ndarray,
        names: Iterable[str],
    ) -> list[str]:
        """Compute and format goodness-of-fit metrics from the stats module.

        Dispatches each ``name`` to the matching :class:`SamplingStats` method
        (``chi_squared``, ``aic``, ``bic``, ``aicc``, ``goodness_of_fit``);
        also accepts the derived names ``dof`` and ``chi2_per_dof``. nparams
        is taken from ``len(self.params)`` and theory values are computed from
        full-sample model values at ``x_data``. chi-squared is cached so
        multiple chi2-derived metrics share one evaluation.

        Display labels come from :data:`METRIC_LATEX`; unknown names fall
        through to ``getattr(stats, name)(theory_values)`` with the raw name
        as the label. Formatting is intentionally fixed for plot annotations.
        """
        if not hasattr(self, "params"):
            raise AttributeError(
                "Model has no fitted params; build it via fit_result.model_func() first"
            )
        x_arr = np.asarray(
            [
                x.full_sample_value if isinstance(x, SigmondSampling) else float(x)
                for x in x_data
            ]
        )
        theory_full = self._evaluate_full_sample_values(x_arr)
        nparams = len(self.params)
        n_obs = stats.num_observables
        dof = n_obs - nparams

        chi2_cache: list[float] = []

        def chi2() -> float:
            if not chi2_cache:
                chi2_cache.append(float(stats.chi_squared(theory_full)))
            return chi2_cache[0]

        entries: list[str] = []
        for name in names:
            if name == "dof":
                entries.append(_format_metric(METRIC_LATEX["dof"], dof, integer=True))
                continue
            if name == "chi2_per_dof":
                value = chi2() / dof if dof > 0 else float("nan")
                label = METRIC_LATEX["chi2_per_dof"]
            elif name == "chi_squared":
                value = chi2()
                label = METRIC_LATEX["chi_squared"]
            elif name in ("aic", "bic", "aicc"):
                value = float(getattr(stats, name)(nparams=nparams, chi2_val=chi2()))
                label = METRIC_LATEX[name]
            elif name == "goodness_of_fit":
                value = float(stats.goodness_of_fit(nparams=nparams, chi2_val=chi2()))
                label = METRIC_LATEX[name]
            else:
                method = getattr(stats, name, None)
                if not callable(method):
                    raise ValueError(
                        f"Unknown metric {name!r}; expected a SamplingStats method "
                        f"or one of {sorted(METRIC_LATEX)}"
                    )
                value = float(method(theory_full))
                label = name
            entries.append(_format_metric(label, value))
        return entries

    def get_latex_str_with_var(self, var_latex: str | None = None, index: int | None = None) -> str:
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
        var_to_use = var_latex or self.independent_var_latex or "x"
        var_to_use = var_to_use.strip("$")  # Remove $ if present
        if index is not None:
            var_to_use = var_to_use + f"_{{{index}}}"

        # Replace {VAR} placeholder with the actual variable
        if "{VAR}" in self.latex_str:
            return self.latex_str.replace("{VAR}", var_to_use)
        elif "{var}" in self.latex_str:
            return self.latex_str.replace("{var}", var_to_use)
        else:
            # If no placeholder, return as-is (assumes it's already complete)
            return self.latex_str

    def __call__(
        self,
        x_values: np.ndarray | list[SigmondSampling] | SigmondSampling,
        output_info: ObservableInfo | None = None,
    ) -> SigmondSampling | list[SigmondSampling]:
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
        if not hasattr(self, "params"):
            raise ValueError("Parameters not set. Call set_parameters() first.")

        # Handle different input types for x_values
        params = list(self.params)

        if isinstance(x_values, SigmondSampling):
            result = self.func(x_values, *params)
            result.observable_info = self._result_info_for_x(x_values, None, output_info)
            return result

        sequence_values = self._coerce_sequence(x_values)
        if sequence_values is not None and all(
            isinstance(x, SigmondSampling) for x in sequence_values
        ):
            results: list[SigmondSampling] = []
            for i, x_val in enumerate(sequence_values):
                result = self.func(x_val, *params)
                result.observable_info = self._result_info_for_x(x_val, i, output_info)
                results.append(result)
            return results
        if sequence_values is not None:
            if any(isinstance(x, SigmondSampling) for x in sequence_values):
                raise TypeError("x_values must be all numeric values or all SigmondSampling objects")
            x_values = sequence_values

        # Fixed x values - convert to numpy array and evaluate point by point
        x_values = np.asarray(x_values)
        if x_values.ndim == 0:
            x_values = np.array([x_values])
            return_single = True
        else:
            return_single = False

        results = []
        for i, x_val in enumerate(x_values):
            result = self.func(x_val, *params)
            result.observable_info = self._result_info_for_fixed_x(i, output_info)
            results.append(result)

        return results[0] if return_single else results

    def _coerce_sequence(self, x_values) -> list | None:
        if isinstance(x_values, np.ndarray):
            return list(np.asarray(x_values, dtype=object).reshape(-1))
        if isinstance(x_values, Iterable) and not isinstance(x_values, (str, bytes)):
            return list(x_values)
        return None

    def _result_info_for_x(
        self,
        x_value: SigmondSampling,
        index: int | None,
        output_info: ObservableInfo | None,
    ) -> ObservableInfo:
        if output_info is not None:
            if index is None:
                return output_info
            return ObservableInfo(
                output_info.name,
                index,
                output_info.op_type,
                output_info.re_im,
                output_info.ensemble_info,
                output_info.latex_str,
            )

        x_info = x_value.observable_info
        return ObservableInfo(
            f"{self.func.__name__}({x_info.name})",
            x_info.index,
            "n",
            "re",
            x_info.ensemble_info,
            latex_str=self.get_latex_str_with_var(x_info.latex_str),
        )

    def _result_info_for_fixed_x(
        self,
        index: int,
        output_info: ObservableInfo | None,
    ) -> ObservableInfo:
        if output_info is not None:
            return ObservableInfo(
                output_info.name,
                index,
                output_info.op_type,
                output_info.re_im,
                output_info.ensemble_info,
                output_info.latex_str,
            )

        return ObservableInfo(
            f"{self.func.__name__}_result",
            index,
            "n",
            "re",
            next(iter(self.params)).observable_info.ensemble_info,
            latex_str=self.get_latex_str_with_var(index=index),
        )

    @staticmethod
    def _as_result_list(results: SigmondSampling | list[SigmondSampling]):
        if isinstance(results, SigmondSampling):
            return [results], True
        return results, False

    def evaluate_with_uncertainty(
        self,
        x_values: np.ndarray | list[SigmondSampling] | SigmondSampling,
        confidence_level: float = 0.68,
    ) -> tuple:
        """
        Evaluate model with uncertainty bands.

        Args:
            x_values: Input values where to evaluate the model (can have uncertainties)
            confidence_level: Confidence level for uncertainty bands

        Returns:
            Tuple of (mean_values, lower_bounds, upper_bounds)
        """
        means, lowers, uppers, _ = self.evaluate_summary(x_values, confidence_level)
        return means, lowers, uppers

    def evaluate_summary(
        self,
        x_values: np.ndarray | list[SigmondSampling] | SigmondSampling,
        confidence_level: float = 0.68,
    ) -> tuple:
        """Evaluate mean, interval bounds, and full-sample values in one pass."""
        result_list, return_single = self._as_result_list(self(x_values))
        if not result_list:
            raise ValueError("x_values must contain at least one value")
        means = np.array([r.mean for r in result_list])
        fulls = np.array([r.full_sample_value for r in result_list])

        if result_list and result_list[0].sampling_info.method == "bootstrap":
            intervals = [r.confidence_interval(confidence_level) for r in result_list]
            lowers = np.array([interval[0] for interval in intervals])
            uppers = np.array([interval[1] for interval in intervals])
        else:
            errors = np.array([r.error for r in result_list])
            lowers = means - errors
            uppers = means + errors

        if return_single:
            return means.item(), lowers.item(), uppers.item(), fulls.item()
        return means, lowers, uppers, fulls

    def _evaluate_full_sample_values(
        self,
        x_values: np.ndarray | list[SigmondSampling] | SigmondSampling,
    ) -> float | np.ndarray:
        result_list, return_single = self._as_result_list(self(x_values))
        if not result_list:
            raise ValueError("x_values must contain at least one value")
        fulls = np.array([r.full_sample_value for r in result_list])
        return fulls.item() if return_single else fulls

    def evaluate_full_sample(
        self, x_values: np.ndarray | list[SigmondSampling] | SigmondSampling
    ) -> float | np.ndarray:
        """Evaluate model using full-sample values."""
        return self._evaluate_full_sample_values(x_values)

    def evaluate_samples(
        self, x_values: np.ndarray | list[SigmondSampling] | SigmondSampling
    ) -> np.ndarray:
        """
        Return all bootstrap/jackknife sample evaluations.

        Args:
            x_values: Input values where to evaluate the model (can have uncertainties)

        Returns:
            Array of shape (n_samples, len(x_values)) containing all evaluations
        """
        # Get all evaluations using the main __call__ method
        result_list, return_single = self._as_result_list(self(x_values))
        if return_single:
            # Single x value - return column vector
            return result_list[0].data.reshape(-1, 1)  # Shape: (n_samples, 1)
        # Multiple x values - stack the data arrays
        return np.column_stack([r.data for r in result_list])  # Shape: (n_samples, len(x_values))

    def __repr__(self):
        if hasattr(self, "params"):
            param_names = list(self.params.obs.name)
            return f"SigmondModelFunc({self.func.__name__}, parameters={param_names})"
        return f"SigmondModelFunc({self.func.__name__}, parameters not set)"

    def __str__(self):
        if hasattr(self, "params"):
            param_labels = [str(info) for info in self.params.obs]
            param_str = ", ".join(param_labels)
            return f"{self.func.__name__}(x; {param_str})"
        return f"{self.func.__name__}(x; parameters not set)"


# Convenience functions for common models
def exponential_decay_model(
    sampling_info: SamplingInfo,
    ensemble_info=DEFAULT_ENSEMBLE,
    latex_str: str = r"A e^{-m{VAR}}",
    independent_var_latex: str = "t",
) -> SigmondModelFunc:
    """Create exponential decay model: A * exp(-m * x)"""

    def exp_func(x, A, m):
        return A * np.exp(-m * x)

    amp_info = ObservableInfo("amplitude", 0, "n", "re", ensemble_info, r"$A$")
    mass_info = ObservableInfo("mass", 0, "n", "re", ensemble_info, r"$m$")

    return SigmondModelFunc(
        exp_func, [amp_info, mass_info], sampling_info, latex_str, independent_var_latex
    )


def polynomial_model(
    degree: int,
    sampling_info: SamplingInfo,
    ensemble_info=DEFAULT_ENSEMBLE,
    latex_str: str | None = None,
    independent_var_latex: str = "x",
) -> SigmondModelFunc:
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


def gaussian_model(
    sampling_info: SamplingInfo,
    ensemble_info=DEFAULT_ENSEMBLE,
    latex_str: str = r"A e^{-\frac{({VAR}-\mu)^2}{2\sigma^2}}",
    independent_var_latex: str = "x",
) -> SigmondModelFunc:
    """Create Gaussian model: A * exp(-(x-mu)^2 / (2*sigma^2))"""

    def gauss_func(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    amp_info = ObservableInfo("amplitude", 0, "n", "re", ensemble_info, r"$A$")
    mean_info = ObservableInfo("mean", 0, "n", "re", ensemble_info, r"$\mu$")
    sigma_info = ObservableInfo("sigma", 0, "n", "re", ensemble_info, r"$\sigma$")

    return SigmondModelFunc(
        gauss_func,
        [amp_info, mean_info, sigma_info],
        sampling_info,
        latex_str,
        independent_var_latex,
    )
