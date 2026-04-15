"""
Statistical analysis tools for Sigmond samplings.
"""

from collections.abc import Callable, Iterable
from functools import cached_property
from typing import TypeVar

import numpy as np
import scipy.linalg
from scipy.stats import chi2

from .ensemble_collection import MultiEnsembleCollection
from .obervable_collection import ObservableCollection
from .sampling import ObservableInfo, SigmondSampling

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
        data: Iterable[SigmondSampling] | None = None,
        use_gvar: bool = False,
    ):
        """
        Initialize SamplingStats with sampling_info and energy-level validation.

        Args:
            data: Input data - can be:
                - List of SigmondSampling objects with energy-level observables
                - ObservableCollection / MultiEnsembleCollection
                - None for empty collection
            use_gvar: Whether to convert values to gvar objects (only supporting covariance currently)

        Raises:
            ValueError: If samplings have inconsistent sampling_info or non-energy observables
        """
        super().__init__(data, "numpy")
        self.use_gvar = use_gvar
        # Convert to tuple for immutability
        self._sampling_info = self.sampling_info
        self._data = tuple(self._data)
        self._numpy_data = self.to_numpy()  # Each sampling is column vector

    @classmethod
    def _fast_load(
        cls: type[T],
        data: list[SigmondSampling],
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
        instance._sampling_info = instance.sampling_info
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
    def inv_cov_matrix(self) -> np.ndarray:
        """
        Cached inverse of the covariance matrix.

        Returns:
            Inverse covariance matrix (num_observables x num_observables)

        Raises:
            np.linalg.LinAlgError: If the covariance matrix is singular
        """
        return np.linalg.inv(self.cov_matrix)

    @cached_property
    def inv_cholesky_cov_matrix(self) -> np.ndarray:
        """
        Calculate the inverse Cholesky decomposition of the covariance matrix.

        Returns:
            Inverse Cholesky factor of the covariance matrix
        """
        try:
            L = scipy.linalg.cholesky(self.cov_matrix, lower=True)
            return scipy.linalg.solve_triangular(L, np.eye(len(L)), lower=True)
        except scipy.linalg.LinAlgError:
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
        resampled = self._numpy_data[:, 1:]  # (N, n_resamples)
        cov_arr = np.atleast_2d(np.cov(resampled, ddof=1))

        if self.use_gvar:
            import gvar
            cov_arr = gvar.evalcov(gvar.dataset.avg_data(self.array[:, 1:].T, bstrap=True))

        if self._sampling_info and self._sampling_info.method == "jackknife":
            cov_arr = cov_arr * (resampled.shape[1] - 1)

        # Zero out entries between observables from different ensembles
        ensembles = [s.observable_info.ensemble_info for s in self._data]
        for i in range(self.num_observables):
            for j in range(i + 1, self.num_observables):
                if ensembles[i] != ensembles[j]:
                    cov_arr[i, j] = 0.0
                    cov_arr[j, i] = 0.0

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

    # should add static versions of these.

    def cov(self, obs1_idx: int, obs2_idx: int, bias=False) -> float:
        """
        Calculate covariance between two specific observables.

        For samplings from different ensembles, returns zero.

        Args:
            obs1_idx: Index of first observable
            obs2_idx: Index of second observable
            bias:     Whether to use biased estimator (divide by n) or unbiased (divide by n-1).
                        default is False (unbiased)

        Returns:
            Covariance value
        """

        n = len(self._data)
        if obs1_idx >= n or obs2_idx >= n:
            raise IndexError("Observable index out of range")

        sampling1 = self._data[obs1_idx]
        sampling2 = self._data[obs2_idx]

        # Return zero covariance for different ensembles
        if sampling1.observable_info.ensemble_info != sampling2.observable_info.ensemble_info:
            return 0.0

        # Same diagonal element - return variance
        if obs1_idx == obs2_idx:
            return sampling1.error**2

        data1 = sampling1.resampled_values
        data2 = sampling2.resampled_values

        cov = np.cov(data1, data2, bias=bias)[0, 1]

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
    ) -> tuple[SigmondSampling, SigmondSampling]:
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
        # silly work around for delta being zero
        if np.isclose(delta.full_sample_value, 0.0):
            delta.data = np.full_like(delta.data, 10 * np.max(self.val.error))
        diff = delta * buffer
        min_buff = min_val - diff
        max_buff = max_val + diff
        return min_buff, max_buff

    # -------------------------------------------------------------------------
    # Chi-Squared and Residual Methods
    # -------------------------------------------------------------------------

    # TODO: speed up this algorithm

    def _transformed_cov_matrix(self, A: np.ndarray) -> np.ndarray:
        """
        Compute the M×M covariance of A @ x directly from resampled samples.

        More accurate than A @ C @ A.T: estimates the M×M matrix we actually need
        rather than the full N×N C first. Cross-ensemble entries are zeroed out
        since different ensembles are statistically independent.

        Args:
            A: Transformation matrix of shape (M, N)

        Returns:
            Covariance matrix of shape (M, M)
        """
        resampled = self._numpy_data[:, 1:]  # N x n_resamples
        transformed = A @ resampled  # M x n_resamples

        cov = np.atleast_2d(np.cov(transformed, ddof=1))

        if self._sampling_info and self._sampling_info.method == "jackknife":
            cov *= resampled.shape[1] - 1

        # Zero out entries where the two rows draw exclusively from different ensembles
        ensembles = [s.observable_info.ensemble_info for s in self._data]
        M = A.shape[0]
        for i in range(M):
            for j in range(i + 1, M):
                src_i = {k for k in np.nonzero(A[i])[0]}
                src_j = {k for k in np.nonzero(A[j])[0]}
                if {ensembles[k] for k in src_i}.isdisjoint({ensembles[k] for k in src_j}):
                    cov[i, j] = 0.0
                    cov[j, i] = 0.0

        return cov

    def get_transformation_matrix(
        self,
        linear_superposition: list[list[tuple[int, float]]],
    ) -> np.ndarray:
        """
        Build the linear transformation matrix A for a superposition of observables.

        Each entry in linear_superposition defines one combined observable as a weighted
        sum of existing observables. Observable indices not mentioned in any entry are
        retained as identity rows (in sorted order, before the explicit reductions).
        The result is a matrix A of shape (M, N) where N = num_observables and
        M = n_untouched + n_reductions.

        The transformed residuals are A @ r. The transformed covariance is computed
        directly from the resampled samples via _transformed_cov_matrix(A).

        Args:
            linear_superposition: List of reductions, each a list of (index, coefficient)
                pairs. E.g. [[(1, 0.5), (2, -0.5)]] creates E = 0.5*E1 - 0.5*E2 while
                all other observable indices are retained as identity rows.

        Returns:
            A: ndarray of shape (M, N)
        """
        N = self.num_observables
        touched: set = set()
        for reduction in linear_superposition:
            for idx, _ in reduction:
                if idx < 0 or idx >= N:
                    raise IndexError(f"Observable index {idx} out of range [0, {N})")
                touched.add(idx)

        untouched = sorted(set(range(N)) - touched)
        M = len(untouched) + len(linear_superposition)
        A = np.zeros((M, N))

        for row, col in enumerate(untouched):
            A[row, col] = 1.0

        offset = len(untouched)
        for i, reduction in enumerate(linear_superposition):
            for idx, coeff in reduction:
                A[offset + i, idx] = float(coeff)

        return A

    @staticmethod
    def _whiten(
        diff: np.ndarray,
        cov_matrix: np.ndarray | None,
        errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Whiten a residual vector.

        If cov_matrix is provided, uses Cholesky decomposition with fallbacks to
        eigendecomposition and then diagonal scaling.
        If cov_matrix is None, divides element-wise by errors.
        """
        if cov_matrix is not None:
            try:
                L = scipy.linalg.cholesky(cov_matrix, lower=True)
                return scipy.linalg.solve_triangular(L, diff, lower=True)
            except scipy.linalg.LinAlgError:
                pass
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                valid_mask = eigenvals > 1e-12 * np.max(eigenvals)
                if np.sum(valid_mask) == 0:
                    raise np.linalg.LinAlgError("All eigenvalues too small")
                sqrt_inv = np.zeros_like(eigenvals)
                sqrt_inv[valid_mask] = 1.0 / np.sqrt(eigenvals[valid_mask])
                return eigenvecs @ (sqrt_inv * (eigenvecs.T @ diff))
            except np.linalg.LinAlgError:
                return diff / np.sqrt(np.diag(cov_matrix))
        return diff / errors

    def residuals(
        self,
        theory_values: np.ndarray,
        resamp_idx: int = 0,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
    ) -> np.ndarray:
        """
        Calculate residuals with respect to theory values.

        Args:
            theory_values: Array of theoretical values to compare against
            resamp_idx: Resampling index to use (0 for full sample)
            linear_superposition: Optional linear combinations of observables. Each entry
                is a list of (index, coefficient) pairs defining one combined observable.
                Untouched indices are retained as-is. The returned residuals are A @ r
                where A is the transformation matrix from get_transformation_matrix.

        Returns:
            Array of residuals
        """
        if len(theory_values) != self.num_observables:
            raise ValueError("Theory values length must match number of observables")
        if resamp_idx < 0 or resamp_idx > self.num_samples:
            raise IndexError("Resampling index out of range")

        obs = self._numpy_data[:, resamp_idx]
        diff = obs - theory_values
        if linear_superposition is not None:
            A = self.get_transformation_matrix(linear_superposition)
            diff = A @ diff
        return diff

    def whitened_residuals(
        self,
        theory_values: np.ndarray | None = None,
        use_corr: bool = True,
        resamp_idx: int = 0,
        cov_matrix=None,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        residuals: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate whitened residuals with respect to theory values.

        Args:
            theory_values: Theory values to compare against. Not required if residuals
                is provided.
            use_corr: Whether to use full covariance matrix
            resamp_idx: Resampling index to use (0 for full sample)
            cov_matrix: Optional covariance matrix to use instead of self.cov_matrix. Only used if use_corr is True. This will not be transformed if linear_superposition is provided - the user must provide the appropriately transformed covariance matrix in that case.
            linear_superposition: Optional linear combinations of observables. Each entry
                is a list of (index, coefficient) pairs defining one combined observable.
                Residuals become A @ r and covariance becomes A @ C @ A.T before whitening.
                Still used to derive the transformed covariance even when residuals are
                pre-computed.
            residuals: Pre-computed (possibly transformed) residuals from self.residuals().
                If provided, theory_values, resamp_idx are ignored.

        Returns:
            Array of whitened residuals
        """
        if residuals is None:
            if theory_values is None:
                raise ValueError("Either theory_values or residuals must be provided")
            residuals = self.residuals(theory_values, resamp_idx, linear_superposition)

        if use_corr:
            if cov_matrix is not None:
                if cov_matrix.shape != (residuals.size, residuals.size):
                    raise ValueError(
                        f"Provided covariance matrix shape {cov_matrix.shape} does not match expected shape {(residuals.size, residuals.size)}"
                    )
                cov_matrix = np.asarray(cov_matrix)
            elif linear_superposition is not None:
                A = self.get_transformation_matrix(linear_superposition)
                cov_matrix = self._transformed_cov_matrix(A)
            else:
                cov_matrix = self.cov_matrix
            return self._whiten(residuals, cov_matrix)
        else:
            if linear_superposition is not None:
                A = self.get_transformation_matrix(linear_superposition)
                errors_sq = np.array([s.error for s in self._data]) ** 2
                errors = np.sqrt(np.diag(A * errors_sq @ A.T))
            else:
                errors = np.array(self.val.error)
            return self._whiten(residuals, None, errors)

    def chi_squared(
        self,
        theory_values: np.ndarray | None = None,
        use_corr: bool = True,
        resamp_idx: int = 0,
        cov_matrix=None,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        residuals: np.ndarray | None = None,
        whitened: np.ndarray | None = None,
    ) -> float:
        """
        Calculate chi-squared with respect to theory values.

        Args:
            theory_values: Theory values to compare against. Not required if residuals
                or whitened is provided.
            use_corr: Whether to use full covariance matrix.
            resamp_idx: Resampling index to use (0 for full sample).
            cov_matrix: Optional covariance matrix to use instead of self.cov_matrix.
                Only used if use_corr is True and linear_superposition is None.
            linear_superposition: Optional linear combinations of observables; see
                get_transformation_matrix for format.
            residuals: Pre-computed residuals from self.residuals(); ignored if
                whitened is provided.
            whitened: Pre-computed whitened residuals; if provided all other args
                are ignored.

        Returns:
            Chi-squared value
        """
        if whitened is None:
            whitened = self.whitened_residuals(
                theory_values, use_corr, resamp_idx, cov_matrix,
                linear_superposition, residuals=residuals,
            )
        return float(np.sum(whitened**2))

    def goodness_of_fit(
        self,
        theory_values: np.ndarray | None = None,
        nparams: int = 0,
        use_corr: bool = True,
        resamp_idx: int = 0,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        chi2_val: float | None = None,
        whitened: np.ndarray | None = None,
    ) -> float:
        """
        Calculate Q (goodness-of-fit) with respect to theory values.

        Pass ``whitened`` to avoid recomputing chi-squared and to supply n_obs
        without an extra get_transformation_matrix call.  Alternatively pass
        both ``chi2_val`` and ``whitened`` (whitened is used only for n_obs).

        Args:
            theory_values: Theory values to compare against. Not required if
                chi2_val or whitened is provided.
            nparams: Number of fitted parameters (for degrees of freedom).
            use_corr: Whether to use full covariance matrix.
            resamp_idx: Resampling index to use (0 for full sample).
            linear_superposition: Optional linear combinations of observables; see
                get_transformation_matrix. When provided without whitened, an extra
                get_transformation_matrix call is made to determine n_obs.
            chi2_val: Pre-computed chi-squared value; if provided skips chi_squared.
            whitened: Pre-computed whitened residuals. Provides chi2 (if chi2_val is
                None) and n_obs = len(whitened), avoiding redundant computation.

        Returns:
            Q value
        """
        if chi2_val is None:
            chi2_val = self.chi_squared(
                theory_values, use_corr, resamp_idx,
                linear_superposition=linear_superposition, whitened=whitened,
            )

        if whitened is not None:
            n_obs = len(whitened)
        elif linear_superposition is not None:
            n_obs = self.get_transformation_matrix(linear_superposition).shape[0]
        else:
            n_obs = self.num_observables

        dof = n_obs - nparams
        return float(chi2.sf(chi2_val, dof))

    def aic(
        self,
        nparams: int = 0,
        theory_values: np.ndarray | None = None,
        use_corr: bool = True,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        whitened: np.ndarray | None = None,
        chi2_val: float | None = None,
    ) -> float:
        """
        Akaike Information Criterion: AIC = chi2 + 2 * nparams.

        Args:
            nparams: Number of fitted parameters.
            theory_values: Theory values to compare against. Not required if whitened
                or chi2_val is provided.
            use_corr: Whether to use full covariance matrix.
            linear_superposition: Optional linear combinations; see get_transformation_matrix.
            whitened: Pre-computed whitened residuals (skips whitened_residuals call).
            chi2_val: Pre-computed chi-squared value (skips chi_squared call).

        Returns:
            AIC value.
        """
        if chi2_val is None:
            chi2_val = self.chi_squared(
                theory_values, use_corr=use_corr,
                linear_superposition=linear_superposition, whitened=whitened,
            )
        return chi2_val + 2 * nparams

    def bic(
        self,
        nparams: int = 0,
        theory_values: np.ndarray | None = None,
        use_corr: bool = True,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        whitened: np.ndarray | None = None,
        chi2_val: float | None = None,
    ) -> float:
        """
        Bayesian Information Criterion: BIC = chi2 + nparams * ln(n_obs).

        Args:
            nparams: Number of fitted parameters.
            theory_values: Theory values to compare against. Not required if whitened
                or chi2_val is provided.
            use_corr: Whether to use full covariance matrix.
            linear_superposition: Optional linear combinations; see get_transformation_matrix.
            whitened: Pre-computed whitened residuals (skips whitened_residuals call).
            chi2_val: Pre-computed chi-squared value (skips chi_squared call).

        Returns:
            BIC value.
        """
        if chi2_val is None:
            chi2_val = self.chi_squared(
                theory_values, use_corr=use_corr,
                linear_superposition=linear_superposition, whitened=whitened,
            )
        if whitened is not None:
            n_obs = len(whitened)
        elif linear_superposition is not None:
            n_obs = self.get_transformation_matrix(linear_superposition).shape[0]
        else:
            n_obs = self.num_observables
        return chi2_val + nparams * np.log(n_obs)

    def aicc(
        self,
        nparams: int = 0,
        theory_values: np.ndarray | None = None,
        use_corr: bool = True,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        whitened: np.ndarray | None = None,
        chi2_val: float | None = None,
    ) -> float:
        """
        Corrected Akaike Information Criterion: AICc = AIC + 2k(k+1)/(n-k-1).

        Preferred over AIC when n_obs is small relative to nparams.

        Args:
            nparams: Number of fitted parameters.
            theory_values: Theory values to compare against. Not required if whitened
                or chi2_val is provided.
            use_corr: Whether to use full covariance matrix.
            linear_superposition: Optional linear combinations; see get_transformation_matrix.
            whitened: Pre-computed whitened residuals (skips whitened_residuals call).
            chi2_val: Pre-computed chi-squared value (skips chi_squared call).

        Returns:
            AICc value.
        """
        if chi2_val is None:
            chi2_val = self.chi_squared(
                theory_values, use_corr=use_corr,
                linear_superposition=linear_superposition, whitened=whitened,
            )
        if whitened is not None:
            n_obs = len(whitened)
        elif linear_superposition is not None:
            n_obs = self.get_transformation_matrix(linear_superposition).shape[0]
        else:
            n_obs = self.num_observables
        aic_val = chi2_val + 2 * nparams
        denom = n_obs - nparams - 1
        if denom <= 0:
            return float("inf")
        return aic_val + 2 * nparams * (nparams + 1) / denom

    def fit_summary(
        self,
        theory_values: np.ndarray,
        nparams: int,
        use_corr: bool = True,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
        print_results: bool = False,
    ) -> dict:
        """
        Summary of goodness-of-fit statistics for given theory values.

        Computes whitened residuals once and reuses them for chi2, Q, and AIC.

        Args:
            theory_values: Theory values to compare against.
            nparams: Number of fitted parameters (for dof, Q, AIC).
            use_corr: Whether to use full covariance matrix.
            linear_superposition: Optional linear combinations; see get_transformation_matrix.
            print_results: If True, print a formatted summary to stdout.

        Returns:
            Dictionary with keys: residuals, whitened_residuals, chi2, dof,
            chi2_per_dof, Q, AIC.
        """
        r = self.residuals(theory_values, linear_superposition=linear_superposition)
        w = self.whitened_residuals(use_corr=use_corr, linear_superposition=linear_superposition, residuals=r)
        chi2_val = self.chi_squared(whitened=w)
        dof = len(w) - nparams

        result: dict = {
            "residuals": r,
            "whitened_residuals": w,
            "chi2": chi2_val,
            "dof": dof,
            "chi2_per_dof": chi2_val / dof if dof > 0 else float("nan"),
            "Q": self.goodness_of_fit(nparams=nparams, chi2_val=chi2_val, whitened=w),
            "AIC": self.aic(nparams, chi2_val=chi2_val),
        }

        lines = [
            "**Fit Summary**",
            "",
            "| Statistic | Value |",
            "| --- | --- |",
            f"| $\\chi^2$ | {result['chi2']:.6g} |",
            f"| dof | {result['dof']} |",
            f"| $\\chi^2/\\text{dof}$ | {result['chi2_per_dof']:.4g} |",
            f"| Q | {result['Q']:.4g} |",
            f"| AIC | {result['AIC']:.6g} |",
            "",
            "| Obs | Residual | Whitened |",
            "| --- | --- | --- |",
        ]
        for i, (res, wh) in enumerate(zip(result["residuals"], result["whitened_residuals"])):
            lines.append(f"| {i} | {res:.6g} | {wh:.6g} |")
        result["markdown"] = "\n".join(lines)

        if print_results:
            W = 42
            print("=" * W)
            print("  Fit Summary")
            print("=" * W)
            print(f"  chi2        : {result['chi2']:.6g}")
            print(f"  dof         : {result['dof']}")
            print(f"  chi2/dof    : {result['chi2_per_dof']:.4g}")
            print(f"  Q           : {result['Q']:.4g}")
            print(f"  AIC         : {result['AIC']:.6g}")
            print(f"\n  {'Obs':<5}  {'Residual':>14}  {'Whitened':>14}")
            print(f"  {'-'*(W-2)}")
            for i, (r, wh) in enumerate(
                zip(result["residuals"], result["whitened_residuals"])
            ):
                print(f"  {i:<5}  {r:>14.6g}  {wh:>14.6g}")
            print("=" * W)

        return result
    
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
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
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
    ) -> tuple[float, float, float, float, float]:
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
        theory_values: np.ndarray | list[SigmondSampling] | ObservableCollection,
        use_corr: bool = True,
        linear_superposition: list[list[tuple[int, float]]] | None = None,
    ) -> SigmondSampling:
        """
        Calculate chi-squared for each resampling.

        Args:
            theory_values: Theory values - array, list of SigmondSampling, or collection
            use_corr: Whether to use full covariance matrix
            linear_superposition: Optional linear combinations of observables; see
                get_transformation_matrix for format. Applied as A @ diff and A @ C @ A.T
                before computing chi-squared at each resampling index.

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
                raise ValueError("Theory values length must match number of observables")
            theory_data = np.tile(theory_values, (n_samples, 1)).T
        elif isinstance(theory_values, ObservableCollection):
            theory_data = theory_values.to_numpy()
            if theory_data.shape[0] != self.num_observables:
                raise ValueError("Number of theory samplings must match number of observables")
            if theory_data.shape[1] != n_samples:
                raise ValueError("Theory samplings must have same length as data")
        elif isinstance(theory_values, list):
            if len(theory_values) != self.num_observables:
                raise ValueError("Number of theory samplings must match number of observables")
            if not all(isinstance(t, SigmondSampling) for t in theory_values):
                raise ValueError("All theory values must be SigmondSampling objects")
            if not all(len(t.data) == n_samples for t in theory_values):
                raise ValueError("Theory samplings must have same length as data")
            theory_data = np.array([t.data for t in theory_values])
        else:
            raise ValueError("Theory values must be array, list, or ObservableCollection")

        diff_matrix = data_matrix - theory_data  # shape: (N, n_samples)

        # Apply linear superposition: A @ diff, A @ C @ A.T
        A = None
        if linear_superposition is not None:
            A = self.get_transformation_matrix(linear_superposition)
            diff_matrix = A @ diff_matrix  # shape: (M, n_samples)

        # Compute covariance
        if use_corr:
            try:
                if A is not None:
                    cov_matrix = self._transformed_cov_matrix(A)
                    inv_cov = np.linalg.inv(cov_matrix)
                else:
                    inv_cov = self.inv_cov_matrix
                use_covariance = True
            except np.linalg.LinAlgError:
                use_covariance = False
        else:
            use_covariance = False

        if use_covariance:
            chi_squared_values = np.einsum("ij,ji->i", diff_matrix.T, inv_cov @ diff_matrix)
        else:
            if A is not None:
                errors_sq = np.array([s.error for s in self._data]) ** 2
                errors = np.sqrt(np.diag(A * errors_sq @ A.T))
            else:
                errors = np.array(self.val.error)
            chi_squared_values = np.sum((diff_matrix / errors[:, np.newaxis]) ** 2, axis=0)

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
        x_values: np.ndarray | list[SigmondSampling] | ObservableCollection,
        model_func: Callable,
        initial_params: np.ndarray,
        param_bounds: list[tuple[float, float]] | None = None,
        use_corr: bool = True,
        method: str = "minimize",
    ) -> dict[str, SigmondSampling]:
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
        elif isinstance(x_values, list) and all(isinstance(x, SigmondSampling) for x in x_values):
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
            from scipy.optimize import curve_fit, minimize
        except ImportError:
            raise ImportError("scipy is required for fitting")

        # Precompute covariance
        if use_corr:
            try:
                inv_cov = self.inv_cov_matrix
                use_covariance = True
            except np.linalg.LinAlgError:
                use_covariance = False
        else:
            use_covariance = False

        errors = np.array(self.val.error)

        def chi_sq(params):
            theory_vals = model_func(x_array, params)
            return self.chi_squared(theory_vals, use_corr)

        # Fit full sample
        if method == "minimize":
            result = minimize(chi_sq, initial_params, bounds=param_bounds)
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
                    result_sample = minimize(chi_squared_sample, best_params, bounds=param_bounds)
                    sample_params = result_sample.x if result_sample.success else best_params
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

    # def goodness_of_fit(
    #     self,
    #     x_values: Union[np.ndarray, List[SigmondSampling], ObservableCollection],
    #     model_func: Callable,
    #     fitted_params: Dict[str, SigmondSampling],
    #     use_corr: bool = True,
    # ) -> SigmondSampling:
    #     """
    #     Calculate goodness of fit using fitted parameters.

    #     Args:
    #         x_values: X values
    #         model_func: Model function
    #         fitted_params: Fitted parameters from fit_function
    #         use_corr: Whether to use correlation matrix

    #     Returns:
    #         SigmondSampling with chi-squared values
    #     """
    #     if not self._data:
    #         raise ValueError("Cannot compute goodness of fit on empty SamplingStats")

    #     n_samples = self._numpy_data.shape[1]

    #     # Handle x-values
    #     if isinstance(x_values, ObservableCollection):
    #         x_matrix = x_values.to_numpy()
    #         x_has_uncertainty = True
    #     elif isinstance(x_values, list) and all(
    #         isinstance(x, SigmondSampling) for x in x_values
    #     ):
    #         x_has_uncertainty = True
    #         x_matrix = np.array([x.data for x in x_values])
    #     else:
    #         x_array = np.array(x_values)
    #         x_has_uncertainty = False
    #         x_matrix = None

    #     param_names = sorted(fitted_params.keys())
    #     param_data = np.array([fitted_params[name].data for name in param_names])

    #     theory_data = np.zeros((self.num_observables, n_samples))

    #     for sample_idx in range(n_samples):
    #         params = param_data[:, sample_idx]
    #         x_vals = x_matrix[:, sample_idx] if x_has_uncertainty else x_array
    #         theory_data[:, sample_idx] = model_func(x_vals, params)

    #     theory_samplings = []
    #     for obs_idx in range(self.num_observables):
    #         observable_info = ObservableInfo(
    #             name=f"theory_{obs_idx}",
    #             index=0,
    #             op_type="n",
    #             re_im="re",
    #         )
    #         theory_samplings.append(
    #             SigmondSampling(
    #                 data=theory_data[obs_idx],
    #                 observable_info=observable_info,
    #                 sampling_info=self._sampling_info,
    #                 is_complex=False,
    #             )
    #         )

    #     return self.chi_squared_by_samplings(theory_samplings, use_corr)

    def fit_polynomial(
        self, x_values: np.ndarray, degree: int, use_corr: bool = True
    ) -> dict[str, SigmondSampling]:
        """Convenience method for polynomial fitting."""

        def poly_func(x, params):
            return np.polyval(params, x)

        initial_params = np.ones(degree + 1)
        return self.fit_function(x_values, poly_func, initial_params, use_corr=use_corr)

    def fit_exponential(
        self, x_values: np.ndarray, use_corr: bool = True
    ) -> dict[str, SigmondSampling]:
        """Convenience method for exponential fitting: A * exp(-m * x)."""

        def exp_func(x, params):
            A, m = params
            return A * np.exp(-m * x)

        y_data = np.array(self.val.mean)
        A_guess = y_data[0] if len(y_data) > 0 else 1.0
        m_guess = 0.1

        initial_params = np.array([A_guess, m_guess])
        return self.fit_function(x_values, exp_func, initial_params, use_corr=use_corr)

    def summary(self, print_results: bool = False) -> dict:
        """
        Summary of dataset statistics (observables, covariance, correlations).

        Args:
            print_results: If True, print a formatted table to stdout.

        Returns:
            Dictionary with keys: num_observables, num_samples, ensembles,
            sampling_method, means, errors, cov_matrix, corr_matrix,
            cov_cond_num, corr_cond_num, effective_sample_sizes.
        """
        result: dict = {
            "num_observables": self.num_observables,
            "num_samples": self.num_samples,
            "ensembles": [e.name for e in self.ensembles],
            "sampling_method": (self._sampling_info.method if self._sampling_info else None),
            "means": np.array(self.val.mean),
            "errors": np.array(self.val.error),
            "cov_matrix": self.cov_matrix,
            "corr_matrix": self.corr_matrix,
            "cov_cond_num": self.cov_matrix_cond_num,
            "corr_cond_num": self.corr_matrix_cond_num,
            "effective_sample_sizes": self.effective_sample_size,
        }

        if print_results:
            W = 52
            print("=" * W)
            print("  SamplingStats Summary")
            print("=" * W)
            print(f"  Observables : {result['num_observables']}")
            print(f"  Samples     : {result['num_samples']}")
            print(f"  Ensembles   : {', '.join(result['ensembles'])}")
            print(f"  Method      : {result['sampling_method']}")
            print(f"\n  {'Obs':<5}  {'Mean':>18}  {'Error':>14}  {'Eff. N':>7}")
            print(f"  {'-'*(W-2)}")
            for i, (m, e, n) in enumerate(
                zip(result["means"], result["errors"], result["effective_sample_sizes"])
            ):
                print(f"  {i:<5}  {m:>18.6g}  {e:>14.6g}  {n:>7.1f}")
            print(f"\n  Cov cond. number  : {result['cov_cond_num']:.4g}")
            print(f"  Corr cond. number : {result['corr_cond_num']:.4g}")
            print("=" * W)

        return result
