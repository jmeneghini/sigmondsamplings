# Single pivot: correlation-matrix rotation and overlaps

Status: plan, nothing implemented.
Scope: `sigmondsamplings/correlators/rotation.py` (new), eigensolver kernels
in `correlators/_kernels.py`, `CorrelatorMatrix.improve` and
`principal_correlators`, a `transform` fix, and later plots and
`ss query corr --pivot`. SS does the whole chain itself: pivot, rotation, and
overlaps from the fitted amplitudes. Only the single pivot is in scope.

References: `sigmond/src/sigmond/cpp/tasks/single_pivot.cc`,
`HermDiagonalizerWithMetric` in `tasks/task_utils.cc`, `task_rotate_corrs.cc`,
`task_get_from_pivot.cc`, and `sigmond/doc/operators_and_rotation.md` §I.7,
§II.6–II.7. The workflow reference is PyCALQ `fvspectrum/sigmond_rotate_corrs.py`
and `sigmond_spectrum_fits.py`.

## 1. Rules for this design

- **Sigmond's conventions are the defaults.** The numbers match Sigmond's
  single pivot unless §5 says otherwise, and every exception there has a
  reason.
- **One way to do each thing.** No alternative conventions, aliases or
  opt-in modes unless a real workflow needs them. Sigmond's diagnostics that
  cost nothing run always instead of behind flags.
- **Pythonic.**
  - The pivot is a frozen result object from a `from_matrix` classmethod (like
    `SamplingFit.from_data` and `CorrelatorIndex.from_collection`), with
    keyword-only parameters.
  - Levels and operators are addressed by label (objects or strings), never
    by position.
  - Errors are `ValueError` subclasses, data problems are warnings, and
    progress goes to `logging`.
- **Fast by construction.**
  - One vectorized kernel call per stage, and no Python loops over times,
    samples or cells.
  - Rotation reuses `transform`, which is two GEMMs.
  - Everything is NumPy/LAPACK; there is no compiled code (§6).

## 2. What Sigmond does (the defaults)

Parameters: rescaling time `τN`, metric time `τ0`, diagonalization time `τD`,
and minimum inverse condition number `ε`. Construction uses the **full sample
only**:

1. Rescale: `C'(t) = D^{-1/2} C(t) D^{-1/2}` with `D = diag |Re C_ii(τN)|`.
2. **Metric:** `eigh(C'(τ0))`. It is an error if `λ_min < alarm` (default
   `alarm = −5ε`). Keep the `N0` eigenvectors `P0` with `λ ≥ |ε λ_max|`; it is
   an error if none remain.
3. **Matrix:** `G = B̃^{-1/2} P0† C'(τD) P0 B̃^{-1/2}`, where
   `B̃ = diag(kept metric eigenvalues)`. The same alarm and cut apply,
   leaving `NP` eigenvectors `Ṽ` in **descending eigenvalue order** (level 0
   decays slowest at `τD`).
4. `R = D^{-1/2} P0 B̃^{-1/2} Ṽ` and `𝒵 = P0 B̃^{1/2} Ṽ` (`𝒵` is not rescaled).
5. With VEVs, rephase column `n` of `R` and `𝒵` so that `(R† v)_n` is real
   and positive. Rotated VEVs are stored with their imaginary part set to
   zero.
6. Improved operators (a basis that mixes linear combinations with plain
   operators) enter as `C = T† C_orig T`, and the stored rotation is `T R`.
7. **Rotate:** `Ĉ(t) = R† C(t) R` on every bin or resample, with the fixed `R`.
   The diagonal is real.
8. **Off-diagonal check** for `t ≥ τD`: `|Re Ĉ_nm|/σ` and `|Im Ĉ_nm|/σ` for
   `n < m`. It reports the maximum and the percentage beyond 1σ to 4σ.
9. **Overlaps** from fits `Ĉ_nn(t) ≈ A_n e^{-E_n t}`:
   `|𝒵_in|² |A_n| = |Z_in|² / C_ii(τN)` on every resample
   (`computeZMagnitudesSquared`). The reported table divides each operator's
   row by its full-sample sum over levels (`DoCorrMatrixZMagSquares`).
   Levels can be put in ascending fit-energy order.

Also logged by Sigmond: every eigenvalue and rank, the fractional errors of
`C_ii(τ0)` and `C_ii(τD)`, and, when requested, the resampled metric
eigenvalues (`CheckMetricErrors`) and the null-space check
(`CheckCommonMetricMatrixNullSpace`).

## 3. API

### 3.1 `SinglePivot` (`correlators/rotation.py`)

```python
@dataclass(frozen=True)
class SinglePivot:
    @classmethod
    def from_matrix(
        cls,
        matrix: CorrelatorMatrix,          # Hermitian, >= 2 operators; bins or samplings
        *,
        norm_time: int,                    # τN
        metric_time: int,                  # τ0
        diagonalize_time: int,             # τD
        min_inv_condition: float,          # ε
        negative_eigenvalue_alarm: float | None = None,  # default -5ε
        name: str = "ROT",                 # rotated labels: the basis channel + name + level
    ) -> SinglePivot: ...

    # Parameters, as given.
    norm_time: int; metric_time: int; diagonalize_time: int
    min_inv_condition: float; negative_eigenvalue_alarm: float

    # Bases.
    operators: tuple[Operator, ...]              # rows of R, in the matrix's order
    rotated_operators: tuple[GenIrrepOperator, ...]   # ROT 0..NP-1, descending eigenvalue at τD
    source_transformation: Transformation | None # when the matrix was improved/transformed

    # Numbers (read-only arrays).
    transformation: np.ndarray    # R, (N, NP); real for a real matrix
    zmatrix: np.ndarray           # 𝒵, (N, NP), same column phases as R
    norm_diagonal: np.ndarray     # full-sample Re C_ii(τN), (N,)
    eigenvalues: np.ndarray       # retained G eigenvalues, descending, (NP,)
    metric_eigenvalues: np.ndarray    # all eigenvalues of C'(τ0), ascending, (N,)
    matrix_eigenvalues: np.ndarray    # all eigenvalues of G, ascending, (N0,)

    # Checks, computed at build time.
    metric_eigenvalue_samples: SigmondData | None  # CheckMetricErrors, (level,); None for bins
    diagonal_fractional_errors: xr.DataArray       # (time: [τ0, τD], operator)
    null_space_ok: bool | None                     # None when the metric has full rank

    def rotate(self, matrix: CorrelatorMatrix) -> CorrelatorMatrix: ...
    def off_diagonal_check(self, rotated: CorrelatorMatrix, *, times=None) -> OffDiagonalCheck: ...
    def overlaps(self, amplitudes: Mapping, *, energies: Mapping | None = None,
                 normalize: bool = False) -> SigmondData: ...
    def summary(self) -> dict[str, Any]: ...       # like FitResult.summary()
    def write(self, path, *, group: str = "pivot", mode: str = "w") -> None: ...
    @classmethod
    def read(cls, path, *, group: str = "pivot") -> SinglePivot: ...


@dataclass(frozen=True)
class OffDiagonalCheck:
    max_ratio: float
    percent_beyond: dict[int, float]          # {1: ..., 2: ..., 3: ..., 4: ...}, Sigmond's four numbers
    max_ratio_by_time: xr.DataArray           # (time,)

class PivotError(ValueError): ...             # a failed build, with Sigmond's reason text
class PivotWarning(UserWarning): ...          # a data problem the result still carries
```

Workflow:

```python
m = index.matrix(basis, subtract_vev=True, sampling_info=boot, include_vevs=True)
piv = SinglePivot.from_matrix(m, norm_time=3, metric_time=5, diagonalize_time=10,
                              min_inv_condition=0.01)          # logs the build (§4)
rot = piv.rotate(m)                                            # CorrelatorMatrix over ROT n
piv.off_diagonal_check(rot)                                    # logs Sigmond's percentages
fits = {op: SamplingFit.from_data(rot.correlator(op, op).to_real(), model, params).fit()
        for op in rot.operators}
z2 = piv.overlaps({op: f.param("A") for op, f in fits.items()},
                  energies={op: f.param("E") for op, f in fits.items()})
rot.write("rotated.hdf5"); piv.write("pivots.hdf5", group=str(m.info.channel))
```

### 3.2 `from_matrix`

- **Checks (Sigmond's).**
  - The matrix is Hermitian and has at least 2 operators.
  - `ε ≥ 0`, and the alarm is `≤ 0`.
  - `τ0 ≠ τD`, `τN ≤ τ0` and `τN ≤ τD`. `τD < τ0` is allowed.
  - All three times are in the matrix.

  Anything else raises `ValueError`, naming the offending value.
- **Full sample** is `values._analyzer.full`: column 0 for samplings, the bin
  mean for bins. For a connected matrix assembled from bins, that is already
  `C̄ − v̄ v̄†`, which is what Sigmond uses.
- **Hermitian part** of the three full-sample matrices. This drops only
  `Im C_ii`, which Sigmond ignores.
- **Real or complex.** A real matrix gives a real `R`, as in Sigmond's
  `REALNUMBERS` build.
- **Failures** raise `PivotError`. The message has Sigmond's reason
  ("Metric not positive semidefinite", "Null space is dim of Matrix", ...), the
  offending eigenvalues and the cutoff, so `ε` can be changed without
  rerunning at DEBUG.
- **Checks that always run.** Both are cheap, and Sigmond makes them opt-in
  only because of log volume:
  - the null-space check. A failure gives a `PivotWarning`, because `𝒵` (the
    overlaps) is then suspect.
  - `metric_eigenvalue_samples`: one stacked `eigvalsh` of `C'(τ0)`, each
    sample rescaled by its own `C_ii(τN)`, as `CheckMetricErrors` does. It is
    `None` for bins, since eigenvalues of single bins mean nothing.

### 3.3 `rotate`

1. The input basis must be `pivot.operators` in any order (it is reordered
   with `select`), or the source operators of `source_transformation`, in
   which case the coefficients are `T @ R`. Anything else raises and lists
   the missing and extra operators. `subtract_vev` is not compared, so a pivot
   built on the connected matrix can rotate the raw one, as in Sigmond's mode
   `U`.
2. `matrix.transform(R, rotated_operators)`: two GEMMs over the whole buffer.
   VEVs follow as `R† v`, and `transformation` composes, so
   `rot.transformation` leads back to the stored operators.
3. Take the Hermitian part of the **rotated** values. It is identical to
   projecting the input, since `(R† C R)† = R† C† R`, and cheaper, because
   `NP ≤ N`. Rotated VEVs keep only their real part.

Sigmond's four `RotateMode`s are just which matrix you pass. Bins and
samplings, connected and raw, all go through the same call. Because `R` is
linear, rotating bins and then resampling equals rotating samplings exactly.
For `tmin`/`tmax`, use `sel_times`.

### 3.4 `overlaps`

The input is `amplitudes`: `{rotated operator (object or string):
SigmondSampling}`, for example `fit.param("A")`. The value on each sample `s`
is `|𝒵_in|² |A_n^{(s)}|`, one broadcast of `(N, NP) × (NP, S)`.

- The result is a `SigmondData` with dims `(operator, level)`. Operator coords
  are the basis, and level coords are the rotated labels.
- The default is Sigmond's `|Z_in|²/C_ii(τN)`. `normalize=True` divides each
  operator's row by its full-sample sum, which is the reported table and plot
  convention.
- `energies=` (same keys) orders the level axis by full-sample energy, which
  is `ReorderByFitEnergy`. Labels are never renamed. The rotated matrix
  reorders with `rot.select(z2.coords["level"])`.
- A missing or extra level raises. Amplitudes must share one sampling plan
  with each other, and bins raise.

### 3.5 `off_diagonal_check`

For `times=None`, it uses the matrix times `≥ τD`. For each `n < m` it takes
`|full|/err` of `Re` and of `Im`, counted separately as Sigmond does. It
returns `OffDiagonalCheck` and logs one INFO line. It reads the rotated
buffer once through the analyzer; nothing is recomputed per cell.

### 3.6 `CorrelatorMatrix` changes

- **`improve(basis, definitions)`**: Sigmond's improved operators, addressed
  by label.
  - `definitions = {new GI op: {original op: coefficient}}`.
  - `basis` is the target order, mixing improved and plain operators.
  - It builds `T` (a unit column for each plain operator) and calls
    `transform`. `transform` learns to keep an original label for a column
    that is exactly the unit vector selecting that operator. Every other new
    label must still be a GI operator.
- **Fix in `transform`: momentum-averaged bases** (`matrix.py:829`). PyCALQ's
  usual channels are `PSQ=n` with no row, and today `transform` rejects them,
  so neither `improve` nor the pivot could run on them. Allow a `PSQ`-only
  basis when the new labels carry exactly its channel. Still reject `PSQ`-only
  labels on a definite-momentum basis, which is the existing test.
- **`principal_correlators`** (§3.7).

### 3.7 `principal_correlators` (choosing `τ0`/`τD`/`ε`)

```python
CorrelatorMatrix.principal_correlators(
    metric_time: int, *, norm_time: int | None = None,
    min_inv_condition: float = 0.0, pin_time: int | None = None,
) -> SigmondData          # (time, level), real
```

This is Sigmond's construction with `τD = t`, evaluated at `t`, for every
`t`:

1. The metric is taken once from the full sample: `W = D^{-1/2} P0 B̃^{-1/2}`,
   with the same cut as the pivot.
2. At each `t`, `eigh` the full-sample `W† C(t) W`. The value on each sample is
   the diagonal `V(t)† W† C_s(t) W V(t)`. That is linear, has no sorting bias,
   and needs no per-sample eigen-solves or level matching.
3. **Pinning.** Start at `pin_time`: by default the latest time whose solve
   succeeds. Walk outward in both directions, matching each `t`'s orthonormal
   eigenvectors to the previous time's by overlap with
   `scipy.optimize.linear_sum_assignment`. At `τ0` (where `G = 1`), reuse the
   previous vectors. A matched overlap below 0.7 gives a `PivotWarning` naming
   the times. Level `n` therefore follows one state across `t`. With
   `pin_time = τD` and the pivot's `norm_time` and `ε`, it is `ROT n`.

The condition number of `C'(τ0)` needed to choose `ε` is on the pivot
(`metric_eigenvalues`, `metric_eigenvalue_samples`), so there is no separate
function for it.

## 4. Logging and warnings

These follow the existing conventions: `logger =
logging.getLogger(__name__)`, and INFO blocks with aligned labels as in
`SamplingFit`.

| When | Level | Content |
| --- | --- | --- |
| `from_matrix` succeeds | INFO | One block:<br>• channel, `N`, bins or samplings;<br>• `τN/τ0/τD` and `ε`;<br>• metric rank `N0/N` and the smallest kept and largest metric eigenvalue;<br>• levels `NP/N0` and the retained `G` eigenvalues;<br>• condition number of `C'(τ0)` with its error (from the metric samples);<br>• the null-space result;<br>• the largest fractional error of `C_ii(τ0)` and of `C_ii(τD)`, with its operator;<br>• elapsed time. |
| `from_matrix` | DEBUG | Every eigenvalue (metric and `G`), all fractional errors, and the coefficient table `R` by operator and level (`PrintTransformationMatrix`). |
| `rotate` | INFO | `N → NP`, times, bins or samplings, elapsed time. |
| `off_diagonal_check` | INFO | `max |Ĉ_nm|/σ = …; beyond 1σ/2σ/3σ/4σ: …%`, Sigmond's wording. |
| `overlaps` | DEBUG | The level order used (by energy or by pivot). |

`PivotWarning` is used for data problems that stay in the result, so they
show without any logging configured:
- a non-positive `Re C_ii(τN)` (the `|…|` is used, as in Sigmond);
- a failed null-space check;
- a zero rotated VEV that cannot fix its phase;
- a weak pinning overlap.

Everything else that is merely informative (levels dropped by the cut,
off-diagonal percentages) is logged, not warned.

`__repr__` fits on one line, for example
`SinglePivot(N=12→10, τN=3 τ0=5 τD=10, ε=0.01, isotriplet P=(0,0,0) T1u_1)`.
`summary()` returns the INFO block as a dict for notebooks and files.

## 5. Where SS differs from Sigmond (and why)

| Sigmond | SS | why |
| --- | --- | --- |
| Rows of `R`/`𝒵` in `std::set` order | The basis order you give | SS never uses `icode` order. Results are identical after permuting rows. |
| Eigenvector phases are whatever LAPACK returns | Column `n` phased so that `𝒵[i*, n]` is real and positive, where `i* = argmax_i |𝒵_in|` (lowest `i` on ties). The VEV rule still wins. | Reproducible off-diagonals and tests. Diagonals, energies and overlaps are unaffected. |
| Diagonal only for `t < τD` | Full `NP × NP` at every time | One code path. The written file is a superset of Sigmond's. |
| The null-space and metric-error checks are opt-in | They always run (§3.2) | They cost almost nothing and catch bad pivots. |

**Not carried over:**
- `SetImaginaryPartsZero` and `SetToZero`: rarely used noise-reduction masks.
- A "raw" overlap normalization: multiplying by `pivot.norm_diagonal` is one
  line.
- `max_condition_number` (PyCALQ's `1/ε`).
- Sigmond pivot files: SS computes the overlaps itself.
- The rotate modes, `DiagonalOnlyTime`/`RemoveOffDiag`, and xmgrace plots.

Each can be added when a workflow needs it.

## 6. Performance

A prototype on synthetic complex data, on a loaded machine (compare the
ratios, not the absolute times):

| N × T × S (buffer) | rotate: `transform` (2 GEMMs) | Python loop over (t, s) | stacked `eigvalsh` |
| --- | --- | --- | --- |
| 12 × 48 × 201 (22 MB) | 47 ms | 78 ms | 59 ms |
| 12 × 48 × 2001 (221 MB) | 0.44 s | — | 0.58 s |
| 30 × 48 × 201 (139 MB) | 0.54 s | — | 0.33 s |

| Stage | Cost | How |
| --- | --- | --- |
| Build | a few `N × N` solves, plus one stacked `eigvalsh` over `S` for the metric samples | `_kernels.metric_solve`, `gevp` |
| Rotate | Memory-bound; one intermediate of `16 T N NP S` bytes | Existing `apply_matrix` (`tensordot` → one BLAS GEMM per side). The Hermitian part is taken on the smaller rotated result. |
| Off-diagonal check | One pass over the rotated buffer | Analyzer `full` and `error` on the whole slice |
| Overlaps | `N × NP × S` multiplications | One broadcast |
| `principal_correlators` | `T` full-sample `eigh` + one batched `V† W† C W` + `T` tiny assignments | `np.matmul` on `(time, sample, N, N)` views; no per-sample eigen-solves |

A diagonal-only rotation saves about 45% but needs a second code path. It is
left out until a workload with thousands of samples needs it. Chunked
rotation straight from the index, for bases too large for memory, is also
left out.

`_kernels.py` gains pure-NumPy functions next to the effective-energy ones:
`metric_solve`, `gevp` (steps 1–4 on one matrix pair), `fix_phases`,
`hermitian_part` and `pin_levels`. `np.linalg.eigh` (`*heevd`) replaces
`zheev`: the eigenvalues agree to rounding, and the eigenvectors agree up to
the phase that `fix_phases` sets.

## 7. Tests

- **Exact spectra.** `C(t) = Z diag(e^{-E t}) Z†`:
  - on the full sample, `Ĉ(τ0) = 1` and `Ĉ(τD)` is diagonal;
  - without noise, `Ĉ_nn(t) = e^{-E_n(t−τ0)}`;
  - `overlaps` with `A_n = e^{E_n τ0}` gives `|Z_in|²/C_ii(τN)`, and each
    `normalize=True` row sums to 1 on the full sample.
- **Cuts and errors.** A duplicated operator gives `N0 = N−1`. An `ε` above
  every eigenvalue, and a metric eigenvalue below the alarm, raise
  `PivotError` with the eigenvalues in the message. A non-positive `C_ii(τN)`
  warns and uses `|…|`.
- **Phases and order.** Repeated builds are identical. Permuting the basis
  permutes the rows of `R` only. With VEVs, the full-sample `R† v` is real and
  positive, and the rotated VEVs have zero imaginary part.
- **Linearity.** Rotating bins and resampling equals rotating samplings.
  Rotating connected data equals `rot(raw) − v̂ v̂†` on the full sample.
- **Improved and averaged bases.** `improve` with a mixed basis matches
  Sigmond's `setup_improved_operators`. A pivot built on it rotates the
  original matrix identically. A `PSQ=1` basis rotates to `PSQ=1 … ROT n`.
- **Persistence.** `write`/`read` round-trip, with several channels in one
  file.
- **Logging.** `caplog` sees the INFO block fields, and the warnings fire
  exactly in their cases.
- **Diagnostics.** `principal_correlators` is 1 at `metric_time`, and equals
  the pivot's `Ĉ_nn(τD)` at `τD`. On a constructed crossing, pinning keeps the
  curves apart where sorting would swap them.
- **Sigmond oracle** (skipped without the bindings): the same synthetic
  `HermMat` file through `sigmond.Pivot`. Compare:
  - the rotated diagonals per sample, to `~1e-10`;
  - off-diagonals up to phase;
  - `computeZMagnitudesSquaredPython` against `overlaps`;
  - the metric-error log against `metric_eigenvalue_samples`.

## 8. Phases

1. `_kernels` (`metric_solve`, `gevp`, `fix_phases`, `hermitian_part`), the
   `transform` fix for momentum-averaged bases, `SinglePivot.from_matrix`,
   `rotate`, logging, `summary`/`__repr__`, and the exact-spectrum, error
   and logging tests.
2. `overlaps`, `off_diagonal_check`, `improve`, `write`/`read`, and the
   Sigmond oracle tests.
3. `principal_correlators` (`pin_levels`).
4. Plots, reusing the views plotters:
   - rotated matrices through `CorrelatorMatrixPlotter`;
   - `plot_eigenvalues` for the principal correlators;
   - `plot_overlaps` (bars and a heatmap of `normalize=True` overlaps).

   Then `ss query corr FILE --matrix 0 --pivot τN,τ0,τD --min-inv-cond ε
   [--write-rotated OUT] [--plot …]`, which prints the build block and the
   off-diagonal check.

## 9. Open questions

1. **`improve` in phase 2 or later.** PyCALQ has an `improved_operators`
   option, but does anyone use it on current data? If not, it and the
   unit-column rule in `transform` wait.
2. **Rotated-file layout.** `rot.write` produces standard `HermMat`
   (`SubVEV`) keys for `ROT n`. PyCALQ also groups them per channel with an
   `Info`/`Header`. Is that needed by anything downstream, now that SS does
   the overlaps?
