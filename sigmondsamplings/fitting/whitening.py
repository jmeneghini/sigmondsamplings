"""Cached whitening transforms for fitting residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..stats import SamplingStats

__all__ = ["WhiteningTransform"]


@dataclass(frozen=True)
class WhiteningTransform:
    """Apply a cached covariance whitening operation to residual vectors.

    Uncorrelated fits store a 1-D ``1 / error`` scale vector. Correlated fits
    store a 2-D linear operator, either the inverse Cholesky factor from
    :class:`SamplingStats` or a symmetric eigendecomposition fallback.
    """

    operator: np.ndarray
    use_correlation: bool
    source: str
    rank: int

    @classmethod
    def from_stats(
        cls,
        stats: SamplingStats,
        *,
        use_correlation: bool = True,
    ) -> WhiteningTransform:
        """Build a whitening transform from a :class:`SamplingStats` covariance."""
        if not use_correlation:
            errors = np.asarray(stats.val.error, dtype=float)
            _validate_positive_finite(errors, "sampling errors")
            return cls(
                operator=1.0 / errors,
                use_correlation=False,
                source="diagonal",
                rank=int(errors.size),
            )

        try:
            operator = np.asarray(stats.inv_cholesky_cov_matrix, dtype=float)
            return cls(
                operator=operator,
                use_correlation=True,
                source="cholesky",
                rank=int(operator.shape[0]),
            )
        except (ValueError, np.linalg.LinAlgError):
            cov = np.asarray(stats.cov_matrix, dtype=float)
            return cls._from_eigendecomposition(cov)

    @classmethod
    def _from_eigendecomposition(cls, cov: np.ndarray) -> WhiteningTransform:
        vals, vecs = np.linalg.eigh(cov)
        scale = max(float(np.max(vals)), 1.0)
        mask = vals > 1e-12 * scale
        rank = int(np.sum(mask))
        if rank == 0:
            raise ValueError("covariance matrix has no positive eigenvalues")
        inv_sqrt = np.zeros_like(vals)
        inv_sqrt[mask] = 1.0 / np.sqrt(vals[mask])
        operator = (vecs * inv_sqrt) @ vecs.T
        return cls(
            operator=operator,
            use_correlation=True,
            source="eigendecomposition",
            rank=rank,
        )

    def apply(self, residuals: np.ndarray) -> np.ndarray:
        """Return whitened residuals for a 1-D residual vector."""
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        if self.operator.ndim == 1:
            if residuals.size != self.operator.size:
                raise ValueError(
                    f"residual vector has length {residuals.size}, expected {self.operator.size}"
                )
            return self.operator * residuals
        if residuals.size != self.operator.shape[1]:
            raise ValueError(
                f"residual vector has length {residuals.size}, expected {self.operator.shape[1]}"
            )
        return self.operator @ residuals

    def unwhiten(self, values: np.ndarray) -> np.ndarray:
        """Apply the inverse whitening operator ``W⁻¹`` to ``values``.

        Inverse of :meth:`apply`: maps whitened space back to data space. Used to
        recover the model Jacobian ``G = -W⁻¹ J`` from the whitened residual
        Jacobian ``J`` for linear theory propagation. Accepts a 1-D vector or a
        2-D array whose columns are each transformed. The rank-deficient
        eigendecomposition fallback uses the pseudo-inverse (the null directions
        the covariance cannot resolve are dropped).
        """
        values = np.asarray(values, dtype=float)
        if self.operator.ndim == 1:
            inv = 1.0 / self.operator
            return inv[:, None] * values if values.ndim == 2 else inv * values
        if self.source == "eigendecomposition" and self.rank < self.operator.shape[0]:
            return np.linalg.pinv(self.operator) @ values
        return np.linalg.solve(self.operator, values)

    def chi2(self, residuals: np.ndarray) -> float:
        """Return squared norm of the whitened residuals."""
        whitened = self.apply(residuals)
        return float(whitened @ whitened)


def _validate_positive_finite(values: np.ndarray, name: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
