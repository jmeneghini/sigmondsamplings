"""Fixed full-sample single pivots, with explicit CN/RN numerical policies.

The solve is performed once. Every sampling (including sampling zero) uses
that same transformation. ``solve_gevp`` accepts Hermitian off-diagonals within
``rtol=1e-10, atol=1e-12``; diagonal imaginary parts are discarded. At the
labeled pivot boundary Sigmond's upper-triangle convention is used instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from ..analysis import analyzer_for
from ..data import SigmondData
from ..info import EnsembleInfo, SamplingInfo
from ..sampling import SigmondSampling
from ..sampling_array import ArrayObsInfo
from . import _kernels
from .keys import CorrelatorMatrixInfo
from .matrix import CorrelatorMatrix, Transformation, _joint_resample, _ratio
from .observables import OverlapObsInfo
from .operators import GenIrrepOperator, Operator, parse_operator

if TYPE_CHECKING:
    from ..fitting.result import FitResult

__all__ = ["GEVPSolution", "NullSpaceCheck", "OffDiagonalCheck", "PivotDiagnostics", "PivotError",
           "PivotSettings", "PivotWarning", "SinglePivot", "solve_gevp"]
logger = logging.getLogger(__name__)


class PivotError(ValueError):
    """Invalid pivot input or a failed numerical solve."""


class PivotWarning(UserWarning):
    """Recoverable pivot diagnostic or convention issue."""


def _freeze(value):
    """A read-only copy, so a frozen record cannot be changed through its arrays."""
    a = np.array(value)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class NullSpaceCheck:
    matrix_eigenvalues: np.ndarray
    cutoff: float
    matrix_null_dimension: int
    metric_null_dimension: int
    projections: np.ndarray
    matrix_elements: np.ndarray
    ok: bool

    def __post_init__(self):
        for key in ("matrix_eigenvalues", "projections", "matrix_elements"):
            object.__setattr__(self, key, _freeze(getattr(self, key)))


@dataclass(frozen=True, eq=False)
class GEVPSolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    zmatrix: np.ndarray
    metric_eigenvalues: np.ndarray
    matrix_eigenvalues: np.ndarray
    metric_whitening: np.ndarray
    metric_rank: int
    rank: int
    metric_cutoff: float
    matrix_cutoff: float
    inverse_condition: float
    null_space: NullSpaceCheck | None
    reconstruction_residual: float

    def __post_init__(self):
        for key in ("eigenvalues", "eigenvectors", "zmatrix", "metric_eigenvalues",
                    "matrix_eigenvalues", "metric_whitening"):
            object.__setattr__(self, key, _freeze(getattr(self, key)))
        n = self.metric_eigenvalues.size
        if not 1 <= self.rank <= self.metric_rank <= n:
            raise PivotError("invalid GEVP ranks")
        shapes = {"eigenvalues": (self.rank,), "eigenvectors": (n, self.rank),
                  "zmatrix": (n, self.rank), "metric_eigenvalues": (n,),
                  "matrix_eigenvalues": (self.metric_rank,),
                  "metric_whitening": (n, self.metric_rank)}
        for key, shape in shapes.items():
            a = getattr(self, key)
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise PivotError(f"invalid GEVP {key} shape or values")


def _policy(epsilon, alarm, numeric_type):
    if numeric_type not in ("complex", "real"):
        raise PivotError("numeric_type must be 'complex' or 'real'")
    if not isinstance(epsilon, Real) or not np.isfinite(epsilon) or epsilon < 0:
        raise PivotError("min_inv_condition must be finite and nonnegative")
    alarm = -5 * float(epsilon) if alarm is None else alarm
    if not isinstance(alarm, Real) or not np.isfinite(alarm) or alarm > 0:
        raise PivotError("negative_eigenvalue_alarm must be finite and nonpositive")
    return float(epsilon), float(alarm)


def _real_input(a, numeric_type, name):
    if numeric_type == "real" and np.iscomplexobj(a):
        if np.any(a.imag != 0):
            raise PivotError(f"{name}: real numeric_type rejects nonzero imaginary values")
        return a.real
    return a


def solve_gevp(A, B, *, min_inv_condition: float,
               negative_eigenvalue_alarm: float | None = None,
               numeric_type: Literal["complex", "real"] = "complex") -> GEVPSolution:
    """Solve ``A Y = B Y Λ`` after independent metric and reduced-matrix cuts.

    No normalization-time rescaling is implicit. Arrays are owned, immutable
    copies in the raw LAPACK gauge. Cuts retain equality, and no tolerance
    overrides the specified negative-eigenvalue alarm.
    """
    epsilon, alarm = _policy(min_inv_condition, negative_eigenvalue_alarm, numeric_type)
    return _solve(A, B, epsilon, alarm, numeric_type, stacklevel=3)


def _solve(A, B, epsilon, alarm, numeric_type, *, stacklevel):
    """:func:`solve_gevp` after policy validation; warnings point ``stacklevel`` frames up."""
    matrices = []
    for name, value in (("A", A), ("B", B)):
        a = np.asarray(value)
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
            raise PivotError(f"{name}: expected a nonempty square matrix")
        if a.dtype.kind not in "biufc":
            raise PivotError(f"{name}: expected numeric values")
        if not np.all(np.isfinite(a)):
            raise PivotError(f"{name}: values must be finite")
        a = _real_input(a, numeric_type, name)
        canonical = _kernels.hermitian_upper(a)
        offdiag = ~np.eye(len(a), dtype=bool)
        if not np.allclose(a[offdiag], canonical[offdiag], rtol=1e-10, atol=1e-12):
            raise PivotError(f"{name}: off-diagonals must be Hermitian")
        matrices.append(canonical)
    if matrices[0].shape != matrices[1].shape:
        raise PivotError("A and B must have equal shapes")
    try:
        result = _kernels.gevp_solve(*matrices, epsilon, alarm, numeric_type)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise PivotError(f"GEVP failed: {exc}") from exc
    if result["null_space"] is not None:
        result["null_space"] = NullSpaceCheck(**result["null_space"])
        if not result["null_space"].ok:
            warnings.warn("metric null space is not contained in the matrix null space",
                          PivotWarning, stacklevel=stacklevel)
    if not np.isfinite(result["inverse_condition"]):
        warnings.warn("retained matrix maximum is zero; inverse condition is unavailable",
                      PivotWarning, stacklevel=stacklevel)
    return GEVPSolution(**result)


def _ensemble_state(ensemble):
    # Include extents/streams and tweaks, which EnsembleInfo.__eq__ omits.
    return dict(name=ensemble.name, num_measurements=ensemble.num_measurements,
                spatial_extent=ensemble.spatial_extent, temporal_extent=ensemble.temporal_extent,
                num_streams=ensemble.num_streams, num_bins=ensemble.num_bins,
                tweak_info=deepcopy(ensemble.tweak_info))


def _sampling_state(plan):
    if plan is None:
        return None
    return dict(method=plan.method, num_resamplings=plan.num_resamplings,
                seed=plan.seed, boot_skip=plan.boot_skip, **deepcopy(plan.extra_params))


def _data(buffer, name, plan, ensemble, dims, coords):
    return SigmondData.from_data(buffer, ArrayObsInfo(name, ensemble_info=ensemble), plan,
                                dim_names=dims, coords=coords, is_complex=np.iscomplexobj(buffer))


def _validate_matrix(matrix, numeric_type, *, finite=True):
    if not isinstance(matrix, CorrelatorMatrix) or not matrix.info.hermitian:
        raise PivotError("pivot requires a Hermitian CorrelatorMatrix")
    if matrix.info.subtract_vev and matrix.is_bins:
        raise PivotError("connected bins are not a connected sampling estimator; resample first")
    if finite and not np.all(np.isfinite(matrix.values.data)):
        raise PivotError("correlator values must be finite")
    _real_input(matrix.values.data, numeric_type, "correlators")
    if matrix.vevs is not None:
        if not np.all(np.isfinite(matrix.vevs.data)):
            raise PivotError("VEVs must be finite")
        _real_input(matrix.vevs.data, numeric_type, "VEVs")
    if matrix.transformation is not None:
        _real_input(matrix.transformation.coefficients, numeric_type, "source coefficients")


@dataclass(frozen=True)
class PivotSettings:
    """The construction choices recorded with a pivot (Sigmond's ``SinglePivotInitiate``).

    ``negative_eigenvalue_alarm=None`` resolves to ``-5 * min_inv_condition``.
    ``zero_elements`` holds ``(sink, source)`` operator coordinates.
    """

    norm_time: int
    metric_time: int
    diagonalize_time: int
    min_inv_condition: float
    negative_eigenvalue_alarm: float | None = None
    numeric_type: Literal["complex", "real"] = "complex"
    set_imaginary_parts_zero: bool = False
    zero_elements: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        epsilon, alarm = _policy(self.min_inv_condition, self.negative_eigenvalue_alarm,
                                 self.numeric_type)
        times = (self.norm_time, self.metric_time, self.diagonalize_time)
        for time in times:
            if isinstance(time, (bool, np.bool_)) or not isinstance(time, (int, np.integer)) or time < 0:
                raise PivotError("pivot times must be nonnegative integers")
        norm, metric, diagonalize = map(int, times)
        if norm > min(metric, diagonalize) or metric == diagonalize:
            raise PivotError("require norm_time <= metric_time, diagonalize_time and distinct solve times")
        for key, value in dict(norm_time=norm, metric_time=metric, diagonalize_time=diagonalize,
                               min_inv_condition=epsilon, negative_eigenvalue_alarm=alarm,
                               set_imaginary_parts_zero=bool(self.set_imaginary_parts_zero),
                               zero_elements=tuple(tuple(p) for p in self.zero_elements)).items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True, eq=False)
class PivotDiagnostics:
    """Build-time diagnostics; none of them affects the transformation.

    ``fractional_errors`` is ``(2, N)``: diagonal ``error / |value|`` at the
    metric and diagonalization times. ``metric_samples`` is ``(N, samplings)``,
    the per-sampling rescaled metric eigenvalues (``None`` for bins).
    """

    fractional_errors: np.ndarray
    metric_samples: np.ndarray | None
    invalid_metric_samples: tuple[int, ...]
    discarded_imaginary_diagonal_max: float
    zero_vev_phase_count: int

    def __post_init__(self):
        object.__setattr__(self, "fractional_errors", _freeze(self.fractional_errors))
        if self.metric_samples is not None:
            object.__setattr__(self, "metric_samples", _freeze(self.metric_samples))
        object.__setattr__(self, "invalid_metric_samples",
                           tuple(int(i) for i in self.invalid_metric_samples))


def _split(record):
    """A dataclass's scalar fields and ndarray fields, for persistence."""
    scalars, arrays = {}, {}
    for f in fields(record):
        value = getattr(record, f.name)
        (arrays if isinstance(value, np.ndarray) else scalars)[f.name] = value
    return scalars, arrays


_SCHEMA = 2


@dataclass(frozen=True, eq=False, repr=False)
class SinglePivot:
    """A fixed full-sample pivot: build once with :meth:`from_matrix`, then
    :meth:`rotate` any sampling or bin matrix of the same ensemble.

    The stored state is the solve (``gevp``), the normalization diagonal and
    the column phases; :attr:`transformation` and :attr:`zmatrix` derive from
    them. ``source_transformation`` records an improved basis's coefficients.
    """

    settings: PivotSettings
    info: CorrelatorMatrixInfo
    rotated_operators: tuple[GenIrrepOperator, ...]
    gevp: GEVPSolution
    norm_diagonal: np.ndarray
    column_phases: np.ndarray
    vev_rephased: bool
    source_transformation: Transformation | None
    diagnostics: PivotDiagnostics
    _ensemble: EnsembleInfo
    _sampling: SamplingInfo | None

    def __post_init__(self):
        object.__setattr__(self, "norm_diagonal", _freeze(self.norm_diagonal))
        object.__setattr__(self, "column_phases", _freeze(self.column_phases))
        object.__setattr__(self, "_ensemble", deepcopy(self._ensemble))
        object.__setattr__(self, "_sampling", deepcopy(self._sampling))
        trans = self.source_transformation
        if trans is not None:
            object.__setattr__(self, "source_transformation", Transformation(
                tuple(trans.source_operators), _freeze(trans.coefficients)))

    def _validate(self):
        """Cross-field consistency, for state that did not come from :meth:`from_matrix`."""
        n, rank = len(self.info), self.gevp.rank
        trans, plan, samples = self.source_transformation, self._sampling, self.diagnostics.metric_samples
        problems = {
            "gevp": self.gevp.eigenvectors.shape[0] != n,
            "norm_diagonal": self.norm_diagonal.shape != (n,)
            or not np.all(np.isfinite(self.norm_diagonal)) or np.any(self.norm_diagonal == 0),
            "column_phases": self.column_phases.shape != (rank,)
            or not np.allclose(abs(self.column_phases), 1),
            "rotated_operators": len(set(self.rotated_operators)) != rank
            or len(self.rotated_operators) != rank
            or any(not isinstance(op, GenIrrepOperator) or op.channel != self.info.channel
                   for op in self.rotated_operators),
            "source_transformation": trans is not None and (
                trans.coefficients.shape != (len(trans.source_operators), n)
                or not np.all(np.isfinite(trans.coefficients))),
            "fractional_errors": self.diagnostics.fractional_errors.shape != (2, n),
            "metric_samples": samples is not None and (
                plan is None or samples.shape != (n, plan.num_resamplings + 1)),
        }
        bad = [name for name, problem in problems.items() if problem]
        if bad:
            raise PivotError(f"inconsistent pivot fields: {', '.join(bad)}")

    @property
    def operators(self):
        return self.info.operators

    @property
    def ensemble_info(self) -> EnsembleInfo:
        return deepcopy(self._ensemble)

    @property
    def sampling_info(self) -> SamplingInfo | None:
        return deepcopy(self._sampling)

    @property
    def identifier(self) -> str:
        """A short hash of what defines this pivot: ensemble, basis (and any source
        coefficients), settings and rotated labels. Names derived observables."""
        trans = self.source_transformation
        state = json.dumps(dict(
            ensemble=_ensemble_state(self._ensemble), operators=list(self.info.coordinates),
            subtract_vev=self.info.subtract_vev, settings=asdict(self.settings),
            rotated_operators=[op.coordinate for op in self.rotated_operators],
            source_operators=None if trans is None else [op.coordinate for op in trans.source_operators],
        ), sort_keys=True)
        digest = hashlib.sha256(state.encode())
        if trans is not None:
            digest.update(np.ascontiguousarray(trans.coefficients).tobytes())
        return digest.hexdigest()[:8]

    @property
    def transformation(self) -> np.ndarray:
        """``T`` (``N x rank``) with ``C_rot = T† C T``: normalization, eigenvectors, phases."""
        scale = 1 / np.sqrt(abs(self.norm_diagonal))
        return scale[:, None] * self.gevp.eigenvectors * self.column_phases

    @property
    def zmatrix(self) -> np.ndarray:
        """The phase-fixed overlap matrix ``B Y`` in the normalized basis."""
        return self.gevp.zmatrix * self.column_phases

    @property
    def metric_eigenvalue_samples(self):
        samples = self.diagnostics.metric_samples
        if samples is None:
            return None
        return _data(samples.copy(), "pivot_metric_eigenvalues", self.sampling_info,
                     self.ensemble_info, ("eigenvalue",),
                     {"eigenvalue": [str(i) for i in range(len(self.operators))]})

    @property
    def diagonal_fractional_errors(self):
        return xr.DataArray(self.diagnostics.fractional_errors.copy(), dims=("stage", "operator"),
                            coords={"stage": ["metric", "matrix"],
                                    "operator": list(self.info.coordinates)})

    @classmethod
    def from_matrix(cls, matrix: CorrelatorMatrix, *, norm_time: int, metric_time: int,
                    diagonalize_time: int, min_inv_condition: float,
                    negative_eigenvalue_alarm: float | None = None,
                    numeric_type: Literal["complex", "real"] = "complex", name: str = "ROT",
                    set_imaginary_parts_zero: bool = False,
                    zero_elements: Sequence[tuple[Operator | str, Operator | str]] = ()):
        """Build a fixed pivot on the supplied full estimator, preserving its basis."""
        _validate_matrix(matrix, numeric_type)
        positions = []
        for pair in zero_elements:
            if len(pair) != 2:
                raise PivotError("zero_elements entries need two operator labels")
            try:
                positions.append(tuple(matrix.info.position(op) for op in pair))
            except (ValueError, KeyError) as exc:
                raise PivotError(f"invalid construction mask: {exc}") from exc
        settings = PivotSettings(
            norm_time=norm_time, metric_time=metric_time, diagonalize_time=diagonalize_time,
            min_inv_condition=min_inv_condition, negative_eigenvalue_alarm=negative_eigenvalue_alarm,
            numeric_type=numeric_type, set_imaginary_parts_zero=set_imaginary_parts_zero,
            zero_elements=tuple((matrix.operators[i].coordinate, matrix.operators[j].coordinate)
                                for i, j in positions))
        s = settings
        for time in (s.norm_time, s.metric_time, s.diagonalize_time):
            if time not in matrix.times:
                raise PivotError(f"pivot time {time} is unavailable")

        def mask(a):
            a = a.real.copy() if s.set_imaginary_parts_zero else a.copy()
            for i, j in positions:
                a[..., i, j] = a[..., j, i] = 0
            return a

        full = _kernels.hermitian_upper(np.asarray(matrix.values.full_sample_value))
        N, B, A = [mask(full[matrix.times.index(t)])
                   for t in (s.norm_time, s.metric_time, s.diagonalize_time)]
        diagonal = N.diagonal().real.copy()
        if np.any(diagonal == 0) or not np.all(np.isfinite(diagonal)):
            raise PivotError("normalization diagonal must be finite and nonzero")
        if np.any(diagonal < 0):
            warnings.warn("negative normalization diagonals: using absolute values",
                          PivotWarning, stacklevel=2)
        scale = 1 / np.sqrt(abs(diagonal))
        if not np.all(np.isfinite(scale)):
            raise PivotError("normalization scale must be finite")
        rescale = scale[:, None] * scale[None, :]
        gevp = _solve(A * rescale, B * rescale, s.min_inv_condition, s.negative_eigenvalue_alarm,
                      s.numeric_type, stacklevel=3)
        R, Z = scale[:, None] * gevp.eigenvectors, gevp.zmatrix
        rephased = matrix.info.subtract_vev and matrix.vevs is not None
        phase = np.ones(gevp.rank, dtype=R.dtype)
        zero_phase = 0
        if rephased:
            w = R.conj().T @ np.asarray(matrix.vevs.full_sample_value)
            if s.numeric_type == "real":
                phase = np.where(w.real < 0, -1.0, 1.0)
            else:
                zero_phase = int(np.count_nonzero(w == 0))
                phase = np.divide(w, abs(w), out=np.ones_like(w), where=w != 0)
                if zero_phase:
                    warnings.warn(f"{zero_phase} zero rotated VEVs: using phase 1",
                                  PivotWarning, stacklevel=2)
        else:
            for col in range(gevp.rank):
                largest = np.max(abs(Z[:, col]))
                candidates = np.flatnonzero(abs(Z[:, col]) == largest)
                row = min(candidates, key=lambda i: matrix.operators[i].coordinate)
                if largest != 0:
                    phase[col] = Z[row, col].conjugate() / largest
        rotated = tuple(GenIrrepOperator.from_channel(matrix.info.channel, name=name, index=i)
                        for i in range(gevp.rank))
        # Diagnostics use their own normalization on each sampling, never bins.
        metric_samples = None
        invalid_samples = []
        if not matrix.is_bins:
            norm_samples, b_samples = [mask(_kernels.hermitian_upper(np.moveaxis(matrix.at(t).data, -1, 0)))
                                      for t in (s.norm_time, s.metric_time)]
            d = abs(np.diagonal(norm_samples, axis1=-2, axis2=-1).real)
            valid = np.all(np.isfinite(d) & (d > 0), axis=-1)
            metric_samples = np.full((len(matrix), norm_samples.shape[0]), np.nan)
            indices = np.flatnonzero(valid)
            if len(indices):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    scales = 1 / np.sqrt(d[indices])
                    scaled = b_samples[indices] * scales[:, :, None] * scales[:, None, :]
                finite = np.all(np.isfinite(scaled), axis=(-2, -1))
                valid[indices[~finite]] = False
                indices, scaled = indices[finite], scaled[finite]
                metric_samples[:, indices] = np.linalg.eigvalsh(scaled).T
            valid &= np.all(np.isfinite(metric_samples), axis=0)
            invalid_samples = np.flatnonzero(~valid).tolist()
            if invalid_samples:
                warnings.warn(f"metric diagnostics: {len(invalid_samples)} invalid samplings preserved as NaN",
                              PivotWarning, stacklevel=2)
        diag = matrix.diagonal(component="real")
        errors = []
        for t in (s.metric_time, s.diagonalize_time):
            datum = diag.sel(time=t)
            with np.errstate(divide="ignore", invalid="ignore"):
                errors.append(np.asarray(datum.error) / abs(np.asarray(datum.full_sample_value)))
        diagnostics = PivotDiagnostics(
            fractional_errors=np.asarray(errors), metric_samples=metric_samples,
            invalid_metric_samples=invalid_samples,
            discarded_imaginary_diagonal_max=float(np.max(abs(
                np.diagonal(matrix.values.data, axis1=1, axis2=2).imag))),
            zero_vev_phase_count=zero_phase)
        pivot = cls(settings=settings, info=matrix.info, rotated_operators=rotated, gevp=gevp,
                    norm_diagonal=diagonal, column_phases=phase, vev_rephased=bool(rephased),
                    source_transformation=matrix.transformation, diagnostics=diagnostics,
                    _ensemble=matrix.values.ensemble_info, _sampling=matrix.sampling_info)
        logger.info("single pivot: %s", pivot)
        logger.debug("build diagnostics: %s", pivot.summary())
        return pivot

    def __repr__(self):
        s = self.settings
        return (f"SinglePivot(N={len(self.operators)}, rank={self.gevp.rank}, "
                f"times=({s.norm_time},{s.metric_time},{s.diagonalize_time}), "
                f"cut={s.min_inv_condition:g}, channel={self.info.channel})")

    def _check_ensemble(self, ensemble, what):
        if _ensemble_state(ensemble) != _ensemble_state(self._ensemble):
            raise PivotError(f"{what}: ensemble differs from the pivot build ensemble")

    def rotate(self, matrix: CorrelatorMatrix) -> CorrelatorMatrix:
        """Apply the fixed transformation in the pivot or recorded source basis."""
        numeric_type = self.settings.numeric_type
        _validate_matrix(matrix, numeric_type)
        self._check_ensemble(matrix.values.ensemble_info, "rotation")
        if matrix.info.channel != self.info.channel:
            raise PivotError("rotation channel differs from the pivot channel")
        labels, source = set(matrix.operators), self.source_transformation
        if labels == set(self.operators):
            basis, coefficients = self.operators, self.transformation
        elif source is not None and labels == set(source.source_operators):
            basis, coefficients = source.source_operators, source.coefficients @ self.transformation
        else:
            raise PivotError("rotation requires the complete pivot or recorded source basis")
        ordered = matrix if matrix.operators == basis else matrix.select(basis)
        # Canonicalize input and output inside bounded time chunks, including
        # removing contraction roundoff as the native HermMat writer does.
        rotated = ordered._transform(coefficients, self.rotated_operators,
                                     canonical_hermitian=True)._replace(numeric_type=numeric_type)
        from ..metadata import RotationBinding

        bindings = [RotationBinding(self.identifier, index) for index in range(len(rotated))]
        infos = rotated.values.infos.copy()
        for t, i, j in np.ndindex(infos.shape):
            infos[t, i, j] = infos[t, i, j].with_rotation(bindings[i], bindings[j])
        rotated = rotated._replace(values=rotated.values._with_replaced_infos(infos))
        if matrix.info.subtract_vev or not self.vev_rephased:
            return rotated._replace(vevs=None)
        if rotated.vevs is not None:
            rotated = rotated._replace(vevs=rotated.vevs.component("real"))
        return rotated

    def summary(self) -> dict[str, Any]:
        g, s, d = self.gevp, self.settings, self.diagnostics
        plan = self._sampling
        report = dict(resampling_mode="bins" if plan is None else plan.method,
                      operators=list(self.info.coordinates),
                      rotated_operators=[op.coordinate for op in self.rotated_operators],
                      norm_time=s.norm_time, metric_time=s.metric_time,
                      diagonalize_time=s.diagonalize_time, metric_rank=g.metric_rank,
                      matrix_rank=g.rank, metric_all_eigenvalues=g.metric_eigenvalues.tolist(),
                      matrix_all_eigenvalues=g.matrix_eigenvalues.tolist(),
                      matrix_retained_eigenvalues=g.eigenvalues.tolist(),
                      metric_cutoff=g.metric_cutoff, matrix_cutoff=g.matrix_cutoff,
                      numeric_type=s.numeric_type, reconstruction_residual=g.reconstruction_residual,
                      inverse_condition=g.inverse_condition,
                      set_imaginary_parts_zero=s.set_imaginary_parts_zero,
                      zero_elements=[list(p) for p in s.zero_elements],
                      diagonal_correlator_fractional_errors=d.fractional_errors.tolist(),
                      metric_diagnostic_unavailable_reason=None if d.metric_samples is not None else "bins",
                      invalid_metric_samples=list(d.invalid_metric_samples),
                      discarded_imaginary_diagonal_max=d.discarded_imaginary_diagonal_max,
                      phase_rule="vev" if self.vev_rephased else "largest_z_entry",
                      vev_rephasing_unavailable_reason="missing VEVs"
                      if self.info.subtract_vev and not self.vev_rephased else None,
                      zero_vev_phase_count=d.zero_vev_phase_count)
        report["null_space"] = None if g.null_space is None else {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in vars(g.null_space).items()}
        return report

    def overlaps(self, amplitudes: Mapping[Operator | str, SigmondSampling | FitResult], *,
                 energies: Mapping[Operator | str, SigmondSampling] | None = None,
                 normalize: bool = False, amplitude: str = "A",
                 energy: str | None = None) -> SigmondData:
        """``abs(Z)^2 abs(A)`` per operator (rows) and rotated level (columns).

        ``amplitudes`` maps every rotated level to its amplitude sampling, or
        to its :class:`~sigmondsamplings.fitting.result.FitResult`, from which
        parameter ``amplitude`` is taken. With fit results, ``energy`` names the
        energy parameter that orders the levels; with samplings, ``energies``
        gives them directly. Ordering is a stable full-sample sort that keeps
        the rotated labels; without energies the rotated order is kept.
        ``normalize`` divides each operator row by its full-sample sum.

            pivot.overlaps(fits, energy="E", normalize=True)
        """
        from ..fitting.result import FitResult

        labels = self.rotated_operators
        fits = {key: value for key, value in amplitudes.items() if isinstance(value, FitResult)}
        if fits and len(fits) != len(amplitudes):
            raise PivotError("amplitudes: give fit results or samplings for every level, not a mix")
        if energy is not None:
            if not fits:
                raise PivotError("energy= names a fit parameter: pass fit results, or energies= samplings")
            if energies is not None:
                raise PivotError("give energy= or energies=, not both")

        def param(fit, name):
            value = fit.param(name)
            if value is None:
                raise PivotError(f"fit result has no parameter {name!r}")
            return value

        if fits:
            amplitudes = {key: param(fit, amplitude) for key, fit in fits.items()}
            if energy is not None:
                energies = {key: param(fit, energy) for key, fit in fits.items()}

        def validate(mapping, kind):
            canonical = {}
            for key, value in mapping.items():
                op = parse_operator(key)
                if op in canonical:
                    raise PivotError(f"{kind}: duplicate canonical level {op.coordinate}")
                canonical[op] = value
            if set(canonical) != set(labels):
                raise PivotError(f"{kind}: missing or extra fit levels")
            values = [canonical[op] for op in labels]
            for op, value in zip(labels, values):
                if not isinstance(value, SigmondSampling) or value.dims or value.is_complex or np.iscomplexobj(value.data):
                    raise PivotError(f"{op.coordinate}: {kind} must be real scalar samplings")
                if not np.all(np.isfinite(value.data)):
                    raise PivotError(f"{op.coordinate}: {kind} contains invalid fit samplings")
                self._check_ensemble(value.ensemble_info, f"{op.coordinate} {kind}")
                if value.sampling_info != values[0].sampling_info or value.data.shape != values[0].data.shape:
                    raise PivotError(f"{op.coordinate}: {kind} sampling plan or length mismatch")
                if len(value.data) != value.sampling_info.num_resamplings + 1:
                    raise PivotError(f"{op.coordinate}: {kind} incomplete fit samplings")
            return values

        values = validate(amplitudes, "amplitude")
        order = np.arange(len(labels))
        if energies is not None:
            energy_values = validate(energies, "energy")
            if energy_values[0].sampling_info != values[0].sampling_info or energy_values[0].data.shape != values[0].data.shape:
                raise PivotError("energy and amplitude sampling plans or lengths differ")
            order = np.argsort([v.full_sample_value for v in energy_values], kind="stable")
            logger.debug("reorder energies: permutation=%s", order.tolist())
        ordered = [labels[i].coordinate for i in order]
        amp = np.stack([values[i].data for i in order])
        logger.debug("overlap amplitude sign-change counts: %s", {
            op.coordinate: int(np.count_nonzero(np.sign(value.data[1:]) != np.sign(value.data[0])))
            for op, value in zip(labels, values)})
        with np.errstate(over="ignore", invalid="ignore"):
            buffer = abs(self.zmatrix[:, order, None]) ** 2 * abs(amp)[None, ...]
        if not np.all(np.isfinite(buffer)):
            raise PivotError("overlap multiplication produced nonfinite values")
        if normalize:
            sums = np.sum(buffer[..., 0], axis=1)
            invalid = ~np.isfinite(sums) | (sums == 0)
            if invalid.any():
                raise PivotError(f"zero/nonfinite overlap row: {self.operators[np.flatnonzero(invalid)[0]].coordinate}")
            buffer /= sums[:, None, None]
        coords = {"operator": list(self.info.coordinates), "level": ordered}
        parent = ArrayObsInfo("pivot_z2_normalized" if normalize else "pivot_z2", ensemble_info=self.ensemble_info)
        s, pivot_id = self.settings, self.identifier
        infos = np.empty((len(self.operators), len(labels)), dtype=object)
        for i, op in enumerate(self.operators):
            for j, level in enumerate(ordered):
                infos[i, j] = OverlapObsInfo(op, level, pivot_id=pivot_id, normalize=normalize,
                                             norm_time=s.norm_time, metric_time=s.metric_time,
                                             diagonalize_time=s.diagonalize_time,
                                             ensemble_info=self.ensemble_info, index=i)
        return SigmondData.from_data(buffer, parent, values[0].sampling_info,
                                    dim_names=("operator", "level"), coords=coords, infos=infos)

    def off_diagonal_check(self, rotated: CorrelatorMatrix, *, times=None):
        """Separate real/imaginary sigma counts with fixed component denominators."""
        _validate_matrix(rotated, self.settings.numeric_type, finite=False)
        if set(rotated.operators) != set(self.rotated_operators):
            raise PivotError("off-diagonal check requires the rotated labels")
        if rotated.info.channel != self.info.channel:
            raise PivotError("off-diagonal check: channel differs from the pivot channel")
        self._check_ensemble(rotated.values.ensemble_info, "off-diagonal check")
        matrix = rotated.select(self.rotated_operators)
        if self.info.subtract_vev and not matrix.info.subtract_vev:
            if matrix.vevs is None or not self.vev_rephased:
                raise PivotError("connected off-diagonal check needs the rotated real VEVs")
            if matrix.is_bins:
                plan = self.sampling_info
                if plan is None:
                    plan = SamplingInfo.jackknife_for(self.ensemble_info)
                # Joint resampling is essential for bootstrap as well as jackknife.
                c, v = _joint_resample([matrix.values.data, matrix.vevs.data],
                                       self.ensemble_info, plan)
            else:
                plan, c, v = matrix.sampling_info, matrix.values.data, matrix.vevs.data
            values = _data(c - v[None, :, None, :] * v[None, None, :, :].conj(),
                           "connected_rotation", plan, self.ensemble_info,
                           matrix.values.dims, matrix.values.coords)
            matrix = matrix._replace(info=CorrelatorMatrixInfo(matrix.operators, True, True),
                                     values=values, vevs=None)
        selected = [t for t in matrix.times if t >= self.settings.diagonalize_time] if times is None else list(times)
        if len(set(selected)) != len(selected) or any(t not in matrix.times for t in selected):
            raise PivotError("check times must be unique available times")
        # (time, pair, sample) upper-triangle entries; CN also counts the Im parts.
        rows, cols = np.triu_indices(len(matrix), 1)
        data = matrix.values.data[[matrix.times.index(t) for t in selected]][:, rows, cols]
        if self.settings.numeric_type == "complex":
            data = np.concatenate([data.real, data.imag], axis=1)
        else:
            data = data.real
        analyzer = analyzer_for(matrix.sampling_info)
        full, error = analyzer.full(data), analyzer.error(data)
        invalid = ~np.isfinite(full) | ~np.isfinite(error)
        ratios = _ratio(abs(full), error)
        n_invalid = invalid.sum(axis=1)
        with np.errstate(invalid="ignore"):
            largest = np.max(ratios, axis=1, initial=-np.inf)
        check = OffDiagonalCheck(
            times=tuple(selected),
            max_ratios=np.where((n_invalid == 0) & (data.shape[1] > 0), largest, np.nan),
            n_entries=np.full(len(selected), data.shape[1]), n_invalid=n_invalid,
            n_beyond=np.stack([(ratios > sigma).sum(axis=1) for sigma in range(1, 5)], axis=-1)
            .reshape(len(selected), 4),
            numeric_type=self.settings.numeric_type,
            phase_rule="vev" if self.vev_rephased else "largest_z_entry")
        for i, time in enumerate(selected):
            logger.info("off-diagonal check time=%s entries=%s invalid=%s max=%s", time,
                        check.n_entries[i], check.n_invalid[i], check.max_ratios[i])
        return check

    def write(self, path, *, group="pivot", mode="a", overwrite=False):
        """Write to an SS-native HDF5 group (not Sigmond's pivot-file format)."""
        import h5py

        gevp, gevp_arrays = _split(self.gevp)
        null = gevp.pop("null_space")
        null, null_arrays = (None, {}) if null is None else _split(null)
        diagnostics, diagnostic_arrays = _split(self.diagnostics)
        diagnostics["invalid_metric_samples"] = list(diagnostics["invalid_metric_samples"])
        trans = self.source_transformation
        metadata = dict(
            settings=asdict(self.settings), operators=list(self.info.coordinates),
            subtract_vev=self.info.subtract_vev,
            rotated_operators=[op.coordinate for op in self.rotated_operators],
            vev_rephased=self.vev_rephased, ensemble=_ensemble_state(self._ensemble),
            sampling=_sampling_state(self._sampling),
            source_operators=None if trans is None else [op.coordinate for op in trans.source_operators],
            gevp=gevp, null_space=null, diagnostics=diagnostics)
        arrays = {"norm_diagonal": self.norm_diagonal, "column_phases": self.column_phases,
                  **{f"gevp/{k}": v for k, v in gevp_arrays.items()},
                  **{f"null_space/{k}": v for k, v in null_arrays.items()},
                  **{f"diagnostics/{k}": v for k, v in diagnostic_arrays.items()}}
        if trans is not None:
            arrays["source_coefficients"] = trans.coefficients
        with h5py.File(Path(path), mode) as file:
            if group in file:
                if not overwrite:
                    raise FileExistsError(f"pivot group {group!r} already exists")
                del file[group]
            node = file.create_group(group)
            node.attrs["ss_pivot_schema"] = _SCHEMA
            node.attrs["metadata"] = json.dumps(metadata)
            for key, array in arrays.items():
                node.create_dataset(key, data=array)

    @classmethod
    def read(cls, path, *, group="pivot"):
        """Restore a pivot written by :meth:`write`, without rebuilding it."""
        import h5py

        try:
            with h5py.File(path, "r") as file:
                node = file[group]
                if node.attrs.get("ss_pivot_schema") != _SCHEMA:
                    raise PivotError("unsupported SS pivot schema")
                meta = json.loads(node.attrs["metadata"])
                arrays = {}

                def collect(name, item):
                    if isinstance(item, h5py.Dataset):
                        arrays[name] = item[...]

                node.visititems(collect)

            def section(prefix, scalars):
                return {**scalars, **{k.split("/", 1)[1]: v for k, v in arrays.items()
                                      if k.startswith(prefix + "/")}}

            null = meta["null_space"]
            gevp = GEVPSolution(**section("gevp", meta["gevp"]), null_space=None
                                if null is None else NullSpaceCheck(**section("null_space", null)))
            ops = meta["source_operators"]
            pivot = cls(
                settings=PivotSettings(**meta["settings"]),
                info=CorrelatorMatrixInfo(meta["operators"], True, meta["subtract_vev"]),
                rotated_operators=tuple(parse_operator(op) for op in meta["rotated_operators"]),
                gevp=gevp, norm_diagonal=arrays["norm_diagonal"],
                column_phases=arrays["column_phases"], vev_rephased=meta["vev_rephased"],
                source_transformation=None if ops is None else Transformation(
                    tuple(parse_operator(op) for op in ops), arrays["source_coefficients"]),
                diagnostics=PivotDiagnostics(**section("diagnostics", meta["diagnostics"])),
                _ensemble=EnsembleInfo(**meta["ensemble"]),
                _sampling=None if meta["sampling"] is None else SamplingInfo(**meta["sampling"]))
            pivot._validate()
            return pivot
        except PivotError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PivotError(f"invalid SS pivot file: {exc}") from exc


@dataclass(frozen=True, eq=False)
class OffDiagonalCheck:
    """Rotated off-diagonal significance, per check time.

    ``n_entries`` counts upper-triangle components (real, plus imaginary for
    CN); ``n_beyond[:, k]`` counts those beyond ``k + 1`` sigma. A time with
    any invalid component has ``max_ratios`` NaN and no percentages.
    """

    times: tuple[int, ...]
    max_ratios: np.ndarray
    n_entries: np.ndarray
    n_invalid: np.ndarray
    n_beyond: np.ndarray
    numeric_type: str
    phase_rule: str

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(int(t) for t in self.times))
        for key, dtype in (("max_ratios", float), ("n_entries", int), ("n_invalid", int),
                           ("n_beyond", int)):
            object.__setattr__(self, key, _freeze(np.asarray(getattr(self, key), dtype=dtype)))

    @property
    def by_time(self) -> xr.Dataset:
        with np.errstate(divide="ignore", invalid="ignore"):
            percent = 100 * self.n_beyond / self.n_entries[:, None]
        percent[self.n_invalid > 0] = np.nan
        result = {"max_ratio": ("time", self.max_ratios.copy()),
                  "n_entries": ("time", self.n_entries.copy()),
                  "n_invalid": ("time", self.n_invalid.copy())}
        result.update({f"beyond_{sigma}": ("time", percent[:, sigma - 1]) for sigma in range(1, 5)})
        return xr.Dataset(result, coords={"time": list(self.times)},
                          attrs={"numeric_type": self.numeric_type, "phase_rule": self.phase_rule})

    @property
    def max_ratio(self) -> float:
        if not len(self.times) or np.any(np.isnan(self.max_ratios)):
            return float("nan")
        return float(np.max(self.max_ratios))

    @property
    def percent_beyond(self) -> dict[int, float]:
        total = int(self.n_entries.sum())
        if not total or self.n_invalid.any():
            return {sigma: float("nan") for sigma in range(1, 5)}
        return {sigma: float(100 * self.n_beyond[:, sigma - 1].sum() / total) for sigma in range(1, 5)}

    def summary(self):
        n_entries, n_invalid = int(self.n_entries.sum()), int(self.n_invalid.sum())
        return dict(max_ratio=self.max_ratio, percent_beyond=self.percent_beyond,
                    n_entries=n_entries, n_invalid=n_invalid,
                    numeric_type=self.numeric_type, phase_rule=self.phase_rule,
                    unavailable_reason="no off-diagonal entries" if not n_entries else
                    "invalid components" if n_invalid else None,
                    by_time=self.by_time.to_dict())
