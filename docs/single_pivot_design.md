# Single pivot: GEVP, rotation and overlaps

Status: implementation baseline is available; see
[single_pivot_usage.md](single_pivot_usage.md). The solver, fixed-pivot workflow,
improved bases, overlaps, diagnostics, SS-native persistence and user surfaces
are implemented. CN and genuine RN live/golden checks are recorded. Performance
checks target the expected workload of at most 20 operators; the full benchmark
sweep remains unrecorded. Phase 5 extensions remain separate.

The goal is Sigmond's single-pivot numerical workflow through SS's labeled,
sampling-aware API. Reuse SS assembly, transformations, statistics, fitting,
metadata, writers and plotters. New code belongs in `correlators/pivot.py`
and the ndarray kernels in `correlators/_kernels.py`. The rolling pivot is
out of scope. The time-dependent diagnostic in §4.7 is a separate, later
feature, not a requirement for single-pivot parity.

## 1. Reference and compatibility contract

This review used the local Sigmond checkout at
`926e10ee52a2becef96c2f8ec96a729487089d55`, specifically:

- `../sigmond/src/sigmond/cpp/tasks/single_pivot.{h,cc}`: construction,
  improved operators, VEVs, rotation, fit-energy reordering and overlaps;
- `task_utils.cc`: `HermDiagonalizerWithMetric`,
  `RealSymDiagonalizerWithMetric`, `doRescaleByDiagonals`,
  `doRescaleTransformation`, `doMatrixRotation` and `doVectorRotation`;
- `task_rotate_corrs.cc` and `pivoter.h`: the task interface and writing;
- `../sigmond/src/sigmond/cpp/pybind/pysigmond.cc`: the actual bindings;
- `../sigmond/doc/operators_and_rotation.md` §I.7 and §II.6–II.7, as
  explanatory material. The C++ implementation wins where the prose differs.

Workflow references are PyCALQ's `fvspectrum/sigmond_rotate_corrs.py` and
`sigmond_spectrum_fits.py`, particularly improved operators and
`samplings_all`. Real task input and logs are in
`test_data/2rotate_corrs/logs/`.

The compatibility promise has three parts:

1. **Scientific quantities:** the same rescaling, two eigenvalue cuts,
   fixed full-sample pivot, rotation, VEV treatment, amplitude convention
   and full-sample overlap normalization as the selected Sigmond build.
2. **Observable conventions:** original/improved bases, GI labels, levels,
   HermMat/SubVEV keys and all four rotation workflows remain usable.
3. **Diagnostics:** compare only quantities with the same sampling plan,
   basis, component count and eigenvector gauge. Real/imaginary checks and
   individual degenerate-state overlaps are not gauge-invariant (§9.2).

Every claimed parity has a test. Undefined arithmetic and confirmed upstream
bugs have explicit exceptions (§7), not silent numerical "improvements".
Do not repair the Sigmond checkout as part of SS implementation: pin the
reference, then test a later upstream fix against a new reference version.

SS conventions:

- Objects and coordinate strings address operators and levels; numerical
  arrays have documented axes. No public integer-level lookup is needed.
- `SinglePivot.from_matrix` is keyword-only after `matrix`, and returns an
  immutable result. Copy and freeze owned arrays, including nested solution
  arrays and source coefficients; a frozen dataclass alone is insufficient.
  Store immutable metadata/diagnostic snapshots and return defensive copies
  for mutable `EnsembleInfo`, `SamplingInfo`, `SigmondData` and xarray views.
  Use explicit numerical equality tests, not generated dataclass equality
  on ndarrays.
- `PivotError` derives from `ValueError`; recoverable problems issue
  `PivotWarning`. Logging never prints directly, and public results expose
  the same diagnostics without requiring logging.
- Batch numerical work over samples and cells. Small loops for metadata,
  time chunking, failure isolation and state matching are allowed. Avoid
  a blanket "no loops" rule that conflicts with bounded memory or diagnostics.

### 1.1 Reuse existing SS APIs

| Responsibility | Existing API | New work |
| --- | --- | --- |
| Assembly, Hermitian completion, joint resampling, VEV subtraction | `CorrelatorIndex.matrix`, `CorrelatorMatrixInfo`, `IncompleteCorrelatorError` | Pivot validation only |
| Labels and channels | `parse_operator`, `op.coordinate`, `GenIrrepOperator.from_channel`, `with_index` | Validate rotated labels |
| Basis order and provenance | `matrix.select`, `sel_times`, `transform`, `Transformation` | Extend `transform` for averaged channels and mixed plain/improved labels |
| Statistics and components | `SigmondData.full_sample_value`, `.error`, `.component`, shared `_Analyzer` | No second jackknife/bootstrap implementation |
| Samples and coordinates | `SigmondData.stack`, `.sel`, `.apply_matrix`, `.from_data`, `SigmondBins.resample` | Thin wrapping around batched kernels |
| Fits and parameters | `SamplingFit.from_data(...).fit_resampled()`, `FitResult.param` | Accept parameter samplings; repair existing result metadata (§4.4) |
| Data files | `matrix.to_collection`, `matrix.write`, `SingleEnsembleCollection`, `SigmondWriter`, `SigmondLoader` | Filtering export keys and pivot-specific persistence |
| Observable metadata | `ArrayObsInfo`, element infos and the ObsMeta registry | Register a derived metadata type only if required for round-trip provenance |
| Plots and CLI | `CorrelatorMatrixPlotter`, `CorrelatorPlotter`, sampling/series plotters, existing `ss query corr` pipeline | New views/options after numerical acceptance |

Prefer public statistics in orchestration. Kernels can use the shared analyzer
on ndarray slices to avoid constructing repeated xarray objects. Do not call
`matrix.normalized` for pivot rescaling: that display API rejects nonpositive
diagonals, whereas Sigmond takes their absolute values.

## 2. Numerical construction

For a Hermitian matrix, write

```
C_ij(t) = Σ_n Z_in Z_jn* exp(-E_n t)
C(τD) v = λ C(τ0) v
Ĉ(t) = R† C(t) R
```

The single pivot solves once, on the full sample. The fixed `R` acts on every
bin or sampling, including sampling 0. It is never rebuilt per resample.
Thus raw rotation commutes with resampling to floating-point tolerance.
For connected data, subtract VEVs after joint resampling, using the existing
matrix assembler; the binwise product of VEVs estimates a different quantity.

### 2.1 Build conventions are explicit

Sigmond selects its real/complex diagonalizer at **compile time**, not from
whether a particular input buffer happens to have imaginary components.
The two implementations in this checkout differ:

| Step | Complex build (CN) | Real build (RN) |
| --- | --- | --- |
| Metric and reduced-matrix cut | `abs(ε * λ_max)` | Same, but replace by `ε` when signed `λ_max < ε` |
| Null-space diagnostic cut on `A` | `abs(ε * λ_max(A))` | Same; no RN floor here |
| VEV phase at exactly zero | Complex `0/0` | Sign rule chooses `+1` |
| Off-diagonal components | Real and imaginary separately | Real only |
| Inverse condition accessor | `getInvCondNum` | Not exposed by this diagonalizer |

Use `numeric_type="complex"` by default, matching the usual CN workflow.
`numeric_type="real"` reproduces RN behavior and rejects nonzero imaginary
input, VEVs and coefficients. Keep real buffers and real coefficients real
when possible, but dtype does not choose the cutoff policy or component count.
Persist this choice. Test real-valued inputs against CN as well as RN; they
are different oracle cases. SS computes the retained `G` inverse condition in
both cases; only CN has that accessor to compare.

### 2.2 Inputs, rescaling and the two cuts

Inputs are nonnegative integer times `τN ≤ τ0`, `τN ≤ τD`, `τ0 != τD`, finite
`ε ≥ 0`, and a finite nonpositive alarm (default `-5ε`). `τD < τ0` is allowed.
Then descending eigenvalue order generally corresponds to *higher* energy
first; only for `τD > τ0` does level 0 decay slowest. Fit-energy reordering
remains available in either case.

Read the full-sample matrices through SS. For samplings this is column 0;
for raw bins it is the mean. A connected full sample is
`C_raw,full - v_full v_full†`, never the mean over connected resamples.
Stored samplings supply their own column 0; SS must not reconstruct it.

Sigmond reads the upper triangle and the real diagonal. At the pivot boundary,
canonicalize this way, on all matrices used by the build and rotation. Do not
silently average disagreeing triangles in the pivot. SS's **assembly** already
averages ordered correlator directions when both exist; parity starts from
that assembled input, which is also what the oracle must write to Sigmond.
An arbitrary directly constructed matrix must not acquire a second, different
Hermitian convention. Record any discarded imaginary diagonal in diagnostics.

1. **Optional construction masks.** `set_imaginary_parts_zero` and labeled
   `zero_elements` reproduce `SetImaginaryPartsZero` and `SetToZero`.
   Apply to private copies of the three build matrices, and to the metric
   diagnostic samples. They affect pivot construction only; Sigmond's
   subsequent rotation uses the unmasked correlators. Persist the masks,
   and qualify pivot identities accordingly. Do not modify caller buffers.
2. **Rescale.** With `d_i = abs(Re C_ii(τN))`, set
   `C'(t) = D^-1/2 C(t) D^-1/2`.
   Negative diagonals warn but use `abs`, as Sigmond does. Exactly zero or
   nonfinite normalization factors raise an input error before LAPACK; an
   infinite scale is not a valid pivot. If `τN = τ0`, the diagonal is `+1`
   only for positive normalization diagonals (otherwise it is `-1`).
3. **Metric.** `eigh(B)`, `B=C'(τ0)`, gives ascending `b`. Fail if
   `b_min < alarm`, then apply the selected §2.1 cutoff. Keep `b >= cutoff`,
   giving eigenvectors `P0` and positive eigenvalues `b0`. Fail if all are
   cut. A retained **exact zero** at `ε=0` cannot be inverted: raise a clear
   metric error with no invented Sigmond code, rather than clamp it or
   silently discard it. Rounding-level negatives remain fatal at alarm 0.
4. **Reduced matrix.** Form
   `G = diag(b0^-1/2) P0† A P0 diag(b0^-1/2)`, `A=C'(τD)`.
   Apply the alarm and the §2.1 cutoff again. Fail if none remain. Store
   retained eigenvalues and vectors `V` in descending order. Retained zero
   `G` eigenvalues are allowed if the metric is invertible; define a zero
   maximum's inverse condition as NaN with an explicit diagnostic.
5. **Vectors.** Compute

   ```
   Y = P0 diag(b0^-1/2) V
   R = D^-1/2 Y
   𝒵 = P0 diag(b0^1/2) V
   ```

   `𝒵` stays in the rescaled pivot basis. Always,
   `Y† B Y = I`, `Y† A Y = Λ` and `𝒵 = B Y`, to rounding.
   Reconstruction `A = 𝒵 Λ 𝒵†` requires that no nonzero contribution was
   removed in **either** cut and that `A` is supported in the retained metric
   subspace. A cutoff-defined "common null space" alone does not imply this.
   Otherwise the right-hand side reconstructs only the retained contribution.
   Report the reconstruction residual; do not assert exact reconstruction
   for truncated cases.
6. **Phases.** For a connected pivot with complete attached VEVs, let
   `w=R†v_full`, and multiply columns of `R` and `𝒵` by `w/abs(w)`.
   This makes the new full-sample VEV real and nonnegative. For exactly zero
   complex `w`, warn and use phase 1; RN already uses phase 1.
   Without that VEV rule, choose the largest-magnitude `𝒵` entry in each
   column and multiply by `conj(z)/abs(z)`. Break exact magnitude ties by
   canonical operator coordinate, not current row position. This is an SS
   gauge convention, not a numerical parity claim for native component logs.
7. **Improved operators.** An improved matrix is `T† C_orig T`.
   Keep `R` and `𝒵` with rows in the **pivot** basis, and the existing
   `Transformation` mapping source operators into that basis. Rotating source
   data uses `T @ R`. Sigmond pivot-file key 0 and its printed transformation
   use source rows; key 1 uses pivot rows. Never compare those arrays without
   identifying the appropriate basis.

### 2.3 Null-space diagnostic

Run the corrected check when the metric has discarded vectors. Diagonalize
`A` itself and identify its cutoff-defined null space. For each discarded
metric vector `d`, record

```
projection = Σ_a abs(a† d)^2
matrix_element = abs(d† A d)
ok = min(projection) > 0.9
```

The projection is the **squared** norm, matching Sigmond's log. Individual
entries depend on the chosen basis within a degenerate discarded eigenspace;
subspace/projector comparisons must account for that.

In the pinned source, both diagonalizers increment `Anulldim` before reading
`Aeigvecs(:, Anulldim)`, selecting the wrong column and potentially reading
past the end when all of `A` is cut. They also return `-5` with
`checkNullSpace=true` and a full-rank metric because the containment flag was
never set. Neither is a reason to warn about a valid SS pivot. Treat full rank
as vacuous containment, with `null_space=None`. A genuine failed corrected
check warns, but does not alter the retained solve or raise. Do not assume
Sigmond's diagnostic option is entirely informational: its all-null branch
can exit early. Capture that as a reference limitation in oracle fixtures.

### 2.4 Overlap convention

If a rotated diagonal is fitted to `A_n exp(-E_n t)`, evaluate on **every**
sampling, including the full sample,

```
z2[i,n,s] = abs(𝒵[i,n])^2 * abs(A[n,s])
```

For an exact, untruncated spectrum with positive normalization diagonals this
is `abs(Z_physical[i,n])^2 / d_i`. The conventional name
`abs(Z)^2/C_ii(τN)` assumes positivity. Multiplying by `abs(norm_diagonal)`
undoes that normalization; multiplying by a signed negative diagonal does not.
Truncation, masks and incomplete null-space containment qualify the physical
interpretation, even though the Sigmond formula is still reproduced.

`computeZMagnitudesSquared` optionally writes these unnormalized samplings.
`DoCorrMatrixZMagSquares` normalizes the estimates displayed in its log by
one **full-sample row sum**, not by resampled row sums. SS supports both with
`normalize=False/True`. Verify full estimates and symmetric errors against
the table, and use the unnormalized output file for samplewise comparisons.

Energy ordering is a stable sort of full-sample fitted energies. Apply one
permutation to labels, amplitudes and `𝒵` columns; never reorder each sample.

## 3. Layers and kernels

```
ndarray kernels → solve_gevp → SinglePivot and labeled diagnostics → plots/CLI
```

Keep private kernel result structures private. The public `GEVPSolution` owns
copied, read-only arrays. Kernels accept ndarray buffers, keep the sample axis
last, and do not know SS types, xarray, operator labels or logging.

| Kernel | Responsibility |
| --- | --- |
| Hermitian upper-triangle reconstruction | Sigmond matrix-read convention |
| Rescale factors | Absolute real diagonal, validity mask |
| Metric solve | Eigenvalues, retained mask, whitening, selected cutoff |
| Reduced solve | `G`, retained mask, descending vectors |
| Null-space check | Correct columns, squared projections |
| Pivot vectors and phases | `Y`, `𝒵`, unit column phases |
| Matrix/vector contraction | `R† C R` / `R† v`, shared with `transform` |
| Off-diagonal counts | Separate component ratios, fixed sigma edges |

Use NumPy/LAPACK for the default implementation; NumPy's `*heevd/*syevd`
is not Sigmond's `zheev/dsyev`. Eigenvalues agree to numerical accuracy,
not bitwise. Phases and degenerate eigenspaces need the comparisons in §9.2.
SciPy is already a dependency for later assignment/pinning. Add an alternate
LAPACK driver only if oracle evidence shows a material compatibility need.

## 4. API

### 4.1 `solve_gevp`

```python
def solve_gevp(
    A, B, *, min_inv_condition: float,
    negative_eigenvalue_alarm: float | None = None,
    numeric_type: Literal["complex", "real"] = "complex",
) -> GEVPSolution: ...

@dataclass(frozen=True, eq=False)
class GEVPSolution:
    eigenvalues: np.ndarray           # retained G spectrum, descending (NP,)
    eigenvectors: np.ndarray          # Y (N, NP)
    zmatrix: np.ndarray               # B Y (N, NP)
    metric_eigenvalues: np.ndarray    # all B eigenvalues, ascending (N,)
    matrix_eigenvalues: np.ndarray    # all G eigenvalues, ascending (N0,)
    metric_whitening: np.ndarray      # P0 diag(b0^-1/2), (N, N0), before G cut
    metric_rank: int
    rank: int
    metric_cutoff: float
    matrix_cutoff: float
    inverse_condition: float          # retained G min/max; not B's condition
    null_space: NullSpaceCheck | None
    reconstruction_residual: float
```

Accept finite nonempty square Hermitian arrays of equal shape. Reject
non-Hermitian off-diagonals here, with a documented tolerance; do not infer
Hermiticity from a LAPACK triangle default. Make diagonals real consistently.
No `τN` rescaling is implicit. Return the raw solver gauge; `SinglePivot`
applies its phase rule and records it. Zero metric eigenvalues and all-cut
checks precede any division or out-of-range indexing.

`NullSpaceCheck` contains `A`'s eigenvalues, its cutoff, matrix/metric null
dimensions, squared `projections`, `matrix_elements` and `ok`.

### 4.2 `SinglePivot`

```python
@dataclass(frozen=True, eq=False)
class SinglePivot:
    @classmethod
    def from_matrix(
        cls, matrix: CorrelatorMatrix, *,
        norm_time: int, metric_time: int, diagonalize_time: int,
        min_inv_condition: float,
        negative_eigenvalue_alarm: float | None = None,
        numeric_type: Literal["complex", "real"] = "complex",
        name: str = "ROT",
        set_imaginary_parts_zero: bool = False,
        zero_elements: Sequence[tuple[Operator | str, Operator | str]] = (),
    ) -> SinglePivot: ...

    # Parameters, metadata and immutable build diagnostics.
    norm_time: int; metric_time: int; diagonalize_time: int
    min_inv_condition: float; negative_eigenvalue_alarm: float
    numeric_type: Literal["complex", "real"]
    info: CorrelatorMatrixInfo         # pivot basis + flags
    ensemble_info: EnsembleInfo
    sampling_info: SamplingInfo | None  # build plan; rotation can use a new plan
    operators: tuple[Operator, ...]     # rows of R and 𝒵
    rotated_operators: tuple[GenIrrepOperator, ...]
    source_transformation: Transformation | None
    transformation: np.ndarray        # R in pivot basis (N, NP)
    zmatrix: np.ndarray               # same columns/phases, pivot basis (N, NP)
    norm_diagonal: np.ndarray         # signed full-sample real diagonal (N,)
    gevp: GEVPSolution                # raw solver gauge; distinct from phased R/𝒵
    column_phases: np.ndarray
    vev_rephased: bool
    # Also persist construction masks and diagnostic availability/reasons.
    metric_eigenvalue_samples: SigmondData | None
    diagonal_fractional_errors: xr.DataArray  # (stage: metric/matrix, operator)

    def rotate(self, matrix: CorrelatorMatrix) -> CorrelatorMatrix: ...
    def off_diagonal_check(
        self, rotated: CorrelatorMatrix, *, times=None,
    ) -> OffDiagonalCheck: ...
    def overlaps(
        self, amplitudes: Mapping[Operator | str, SigmondSampling], *,
        energies: Mapping[Operator | str, SigmondSampling] | None = None,
        normalize: bool = False,
    ) -> SigmondData: ...
    def summary(self) -> dict[str, Any]: ...
    def write(self, path, *, group="pivot", mode="a", overwrite=False) -> None: ...
    @classmethod
    def read(cls, path, *, group="pivot") -> SinglePivot: ...
```

No pivot object owns caller data or lazily reloads it. Its metadata includes
channel, subtraction convention, source basis, masks and phase rule. A
connected pivot without attached VEVs is allowed, but logs that rephasing was
unavailable. `include_vevs=True` on assembly remains the normal workflow.
Reject connected bins: their bin mean and errors do not represent the
samplewise connected estimator. Assemble connected samplings through the
existing index first, even when the eventual output workflow is bins.

Example (with `index`, `basis`, `model` and resolved `params` already supplied):

```python
jack = SamplingInfo.jackknife_for(index.ensemble_info)
m = index.matrix(basis, subtract_vev=True, sampling_info=jack, include_vevs=True)
piv = SinglePivot.from_matrix(
    m, norm_time=3, metric_time=3, diagonalize_time=6,
    min_inv_condition=1 / 150,
)
rot = piv.rotate(m)
check = piv.off_diagonal_check(rot)
fits = {
    op: SamplingFit.from_data(
        rot.correlator(op, op).to_real(), model, params,
        ensemble_info=rot.values.ensemble_info,
    ).fit_resampled(progress=False)
    for op in rot.operators
}
z2 = piv.overlaps(
    {op: fit.param("A") for op, fit in fits.items()},
    energies={op: fit.param("E") for op, fit in fits.items()},
    normalize=True,
)
rot.write("rotated.hdf5")
piv.write("pivots.hdf5", group=str(m.info.channel))
```

Build diagnostics:

- `metric_eigenvalue_samples` uses each sampling's own normalization
  diagonal, includes sampling 0 and sorts ascending independently, matching
  `CheckMetricErrors`. It describes uncertainty in metric eigenvalues; it
  never influences the fixed pivot or tracks eigenvector identities.
- For bins, it is `None` with an availability reason. Obtain this diagnostic
  by assembling samplings through `index.matrix(..., sampling_info=...)`.
  Never diagonalize individual bins and report their errors as metric errors.
- Invalid diagnostic resamples produce NaN entries and one aggregated
  warning; preserve samples rather than drop them or use nan-aware error
  estimators. Full-sample build failures remain fatal. Sigmond's enabled
  `CheckMetricErrors` can fail the whole build; this is an explicit SS
  diagnostic-only difference.
- `diagonal_fractional_errors` uses `diagonal(component="real")` and the
  shared error/full statistics. Its `stage` axis has metric and matrix
  entries even if time values would otherwise coincide in other diagnostics.
  Always include improved-basis diagnostics. For raw bins, the bin standard
  error equals unrebinned jackknife error for this linear quantity. It is
  not a bootstrap error or an error for nonlinear connected bins.

### 4.3 Rotation and VEV workflows

`rotate` validates a Hermitian input, matching channel and ensemble, finite
values, and one of two bases: the pivot basis, or the exact recorded source
basis. Reorder by label with `select`. In the source case use `T @ R`.
Reject missing/extra operators rather than use a silent intersection.
Reject connected bins here as well. Validate resolved ensemble identity,
including extents and stream count: the current `EnsembleInfo.__eq__` does
not compare those fields, so equality alone is insufficient. Reuse the
existing ensemble field/inline-string helpers and compare additional fields
explicitly; a pivot should not quietly change global equality semantics.
The input sampling plan can differ from the build plan: the fixed full-sample
rotation can legitimately be reused on bins, jackknife or bootstrap data.
Its numerical provenance remains tied to the build ensemble.

Use `matrix.transform(coefficients, rotated_operators)` for contraction,
metadata and composed provenance. Canonicalize input as in §2.2, and clear
stale `stored`/`key_form` provenance rather than marking generated values as
stored data. Keep `omitted_times`. `transform` itself remains a general
linear operation; pivot-specific Hermitian/VEV rules belong in `rotate`.

Write `C_s` for **raw** sampled correlators and `v_s` for VEVs. With
`w_s = R† v_s`:

| Workflow | SS operation | Result |
| --- | --- | --- |
| B: bins | Build pivot on full/connected samplings, then `rotate(raw_bins)` | Raw rotated bins and, when available for a connected pivot, real VEV bins |
| S: samplings | `rotate(connected_samplings)` | `R†(C_s-v_s v_s†)R`; SubVEV keys, no VEVs attached |
| U: samplings_unsubt | `rotate(raw_samplings)` | Raw sampled rotation and real VEV samplings for a connected pivot |
| A: samplings_all | Rotate both raw and connected matrices with the same pivot | Both key sets plus real VEVs; combine existing collections for writing |

Mode S has already been subtracted. Do not subtract again or rebuild it from
real rotated VEVs. Sigmond does **not** rotate or write VEVs in mode S.
For B/U, a connected pivot with available input VEVs returns
`q_s = Re(w_s)` as real data (complex file writers can emit zero Im keys).
A missing VEV attachment must not be invented. If VEVs were unavailable at
build time, retain that skipped-rotation decision even if another input later
attaches VEVs; rebuild the pivot with VEVs for that workflow. For a pivot built without
VEV subtraction, attached VEVs do not trigger Sigmond rephasing or VEV
rotation; omit them from the pivot rotation result. Use generic `transform`
for an unrestricted complex VEV basis change.

Connected values assembled from B/U are
`R† C_s R - q_s q_s†`. They agree with S on the full sample **only when the
pivot was VEV-rephased and the relevant full-sample VEVs are available**.
On resamples their difference is

```
w_s w_s† - q_s q_s†
```

For diagonals this is `Im(w_s)^2`. Off-diagonals also have imaginary terms
linear in the VEV fluctuations; calling the whole difference "second order
in the noise" is incorrect. Test both workflows against their own oracle.
Do not promise equality if VEV rephasing was skipped or a different VEV
source was supplied.

The current Sigmond U implementation also computes connected samplings
internally; SS need only reproduce its exported raw/VEV result, not hidden
scratch data. A combined A export can use
`SingleEnsembleCollection([*raw_rot.to_collection(),
*connected_rot.to_collection(include_vevs=False)])`, then the existing
writer. No parallel rotation-mode implementation is required.

Compute the full `NP × NP` matrix at all requested times by default.
`sel_times` selects times; `diagonal()` obtains diagonal values. Plain `sel`
returns `SigmondData`, so it is **not** a substitute for choosing which keys
a `CorrelatorMatrix.write` call emits. Add one export filter shared by
`to_collection`/`write` for diagonal-only-before-time and remove-off-diagonal
behavior. This preserves Sigmond's key coverage without requiring a sparse
matrix with fake zero cells or a second numerical rotation kernel.

### 4.4 Overlaps

Resolve mapping keys with the existing operator parser and coordinate rules.
Normalize keys *before* checking duplicates, missing and extra levels (an
object and its string can otherwise supply the same level twice).

Validate each value as a real scalar `SigmondSampling`, finite including
sampling 0, from one ensemble and with identical `SamplingInfo` and sample
length. Reject bins, incomplete/synthetic fit samplings and mixed plans with
level-specific messages when invalid samples or provenance identify them.
A bare `SigmondSampling` does not by itself prove how a finite sample buffer
was produced; do not claim that this API can detect every synthetic fit.
Use existing compatibility helpers, supplemented
by ensemble/type/finiteness checks: generic stack compatibility alone does
not currently check ensemble identity. Equal plans assume aligned resample
indices, as elsewhere in SS; equal metadata cannot prove common RNG history.
The fit plan need not equal the build plan because the pivot is fixed.
Compare resolved ensemble fields as in `rotate`. Pass `ensemble_info` to
`SamplingFit.from_data` explicitly in this workflow. There is also a concrete
existing metadata bug to fix before this example can pass overlap validation:
`SamplingFit._build_result` calls `ParamResolved.observable_info(name)`
without setting the fit ensemble, so parameter samplings currently retain
`INDEP_ENSEMBLE` even when the fit has an explicit ensemble. Set their
observable metadata from the configured fit ensemble in the existing result
builder, preserving parameter name/latex metadata. Separately, let
`from_data` infer the data ensemble when its ensemble argument is omitted,
while preserving an explicit override. Test fitted parameter provenance,
including writer/read-back, rather than weakening overlap validation or
retagging parameters silently inside `overlaps`. The example above assumes
this planned prerequisite is implemented.

Use `SigmondData.stack(..., dim_name="level", coords=...)`, then broadcast
`abs(𝒵)^2` against `abs(amplitudes)` once. Build a `SigmondData` of dimensions
`(operator, level)` via the existing constructor with canonical string coords
and unique element observable infos. Preserve ensemble, plan, level identities
and normalization provenance. Do not reuse amplitude element keys for every
operator, which would collide when converting to a collection.

`normalize=True` divides by each full-sample row sum. A zero/nonfinite sum
raises an overlap error naming the operator. Finite negative amplitudes keep
Sigmond's `abs` on each sample, with sign-change counts in DEBUG.
If `energies` is present, validate it against the amplitude plan and apply a
stable full-sample energy sort. Labels remain the **original** rotated labels;
`rot.select(z2.coords["level"])` follows the same permutation.

A future fit-result convenience adapter can extract named parameters and
call this API; it must not assume every fit uses parameter names `A`/`E` or
that an arbitrary fitted parameter has the required exponential amplitude
normalization. No refitting or fit-model inference is part of `overlaps`.

### 4.5 Off-diagonal checks

```python
@dataclass(frozen=True, eq=False)
class OffDiagonalCheck:
    by_time: xr.Dataset  # max_ratio, beyond_1..beyond_4 (%), n_entries, n_invalid
    max_ratio: float
    percent_beyond: Mapping[int, float]  # pooled counts, not mean of percentages
    def summary(self) -> dict[str, Any]: ...
```

Validate rotated labels, channel, ensemble, Hermiticity and component
convention. Requested times must exist; default to available `t >= τD`.
If none are available, return an empty check with NaN aggregate statistics.
For `NP=1` there are no off-diagonal entries: counts are zero and percentages
NaN, with an explicit reason, not division by zero.

Gather pairs `n<m`, obtain **separate** real and imaginary full estimates and
errors through `.component`/the shared analyzer, and apply `_ratio`. Never
use the complex combined error from `matrix.significance` here. CN counts
`NP(NP-1)` entries even for real-valued buffers; RN counts half that.
Use `<=` edges at 1, 2, 3 and 4 sigma. Zero error gives infinity for nonzero
value and zero for zero value, the existing SS rule. Nonfinite values/errors
are recorded as invalid; affected percentages are unavailable rather than
silently reducing the denominator. Log one line per time.

For raw bins without subtraction, bin standard errors are jackknife-equivalent.
For B/U from a connected pivot, Sigmond's off-diagonal check requests connected
estimates regardless of the exported raw keys. Assemble connected samplings
from the rotated raw values and real VEVs using the requested default sampling
plan, then check that matrix. A check of raw B/U data remains useful but is
not this Sigmond diagnostic. Bootstrap checks require actual bootstrap
samplings; no implicit bootstrap from a bin standard error.

Componentwise checks depend on column phases. Store the phase convention in
the result/provenance and apply the oracle's gauge rules in §9.2.

### 4.6 Improved bases and `transform`

`CorrelatorMatrix.improve(basis, definitions)` is a thin coefficient-builder:

```
definitions = {new_operator: {source_operator: coefficient}}
```

Parse all labels first, reject canonical duplicates, and validate target
basis uniqueness/channel, definitions not in the target basis, missing source
terms, finite coefficients and empty/all-zero definitions. Do not require all
improved operators to use the same terms; that is a PyCALQ configuration
restriction, not a Sigmond algebra requirement. Plain target operators use
unit selection columns. Persist the full source transformation, including
rectangular/repeatedly composed transformations.
For a Hermitian input, canonicalize its real diagonal/upper triangle before
this transform, matching Sigmond's original-matrix reader. Transforming an
imaginary source diagonal and discarding only the improved diagonal later
can contaminate improved off-diagonals. Test that ordering explicitly. A
general non-Hermitian basis change continues to use `transform` directly.

**Coefficient convention:** coefficients are `T[i,a]`, so
`O_new[a] = Σ_i conj(T[i,a]) O_source[i]` and `C_new=T† C T`, matching both
Sigmond's `setup_improved_operators` and SS `transform`. Do not advertise them
as unconjugated coefficients of `O_new`. Include genuinely complex terms in
examples/tests to catch this ambiguity.

Extend `transform` to preserve any original label (including BL) only for an
exact matching unit-selection column. Other new labels must be GI labels.
Permit averaged `PSQ` channels by exact `Channel` equality (momentum and row
remain part of that equality); keep rejecting a `PSQ`-only label for a basis
with definite momentum. Generate ROT labels with
`GenIrrepOperator.from_channel`, preserving channel averaging and row.

Use the same contraction kernel for pivot rotation and general transforms,
but keep non-Hermitian transforms and complex VEVs supported. Test that the
optimization does not project or discard either.

### 4.7 Time-dependent diagnostic (separate phase)

The originally proposed `principal_correlators` are **not** generalized
eigenvalues on each resample. They are correlators projected onto vectors
chosen independently at each time from the full sample. State that explicitly
in docs and plot labels; do not claim nonlinear GEVP-estimator error parity.

Prefer `pivot.projected_correlators(matrix, *, pin_time=None) -> SigmondData`
for this later feature. It reuses the pivot's normalization, metric cut,
alarm, masks and numeric type instead of adding inconsistent defaults to
`CorrelatorMatrix`. Use the full-sample whitening matrix once, diagonalize
`W† C_full(t) W` at each time, and evaluate the fixed-per-time vectors on all
samples. Do not use their eigenvalues as resample values.
Obtain `W = D^-1/2 * gevp.metric_whitening` from persisted build state;
retained pivot vectors alone cannot recover all `N0` metric directions when
the G cut removed levels. Apply construction masks consistently to the
diagnostic inputs, without mutating the matrix.

Before implementing, specify a fixed level domain. Retain all `N0` metric
levels for this diagnostic, order by a pin-time solve, and expose per-time
cut/alarm masks. Applying the single-pivot `G` cut independently at every
time gives changing rank and cannot populate an ordinary `(time, level)`
array without a validity contract. Invalid times stay NaN on every sample;
warn once with the affected times. Isolate failed eigensolves so one bad
matrix does not abort the whole stacked batch.

Pin by maximizing **squared** overlaps in the whitened orthonormal basis with
`scipy.optimize.linear_sum_assignment`, walking through successful times on
both sides of `pin_time`. Default pin time is the pivot's `diagonalize_time`;
it must be available and valid, not an arbitrary latest noisy time. Treat
`τ0`'s identity matrix and degenerate clusters as subspaces; individual states
inside an exact degeneracy have no unique identity. A low matching score is
an ambiguity warning, not evidence that a physical state was tracked.

At `τD`, retained pivot levels match the single-pivot diagonal after accounting
for order/gauge. At `τ0`, the **full sample** is 1; resamples generally are not
because the metric is fixed. With construction masks, compare the masked
input when asserting these identities. This diagnostic does not have a
single-pivot task oracle and does not replace a rolling pivot.

### 4.8 Persistence and interchange

Use a versioned SS HDF5 group for a complete pivot. Persist parameters,
numeric type, masks, ensemble/build plan, matrix flags, canonical operator
labels, both bases, source coefficients, phased `R`/`𝒵`, phases, signed
normalization diagonal, the solver result and all build diagnostics (including
metric samples and their infos/coords), plus availability reasons. Reuse SS
metadata/plan/label serialization. Never pickle Python objects.

`mode="a"`, `overwrite=False` is the default: append a new group without
truncating sibling channels; reject an existing target group. Explicit
`mode="w"` replaces a file only with overwrite authorization, following
writer semantics. Validate and stage the target group before replacing it,
so a failed write cannot destroy a previous valid pivot. Share file/group
validation and backup conventions with the existing I/O layer.

Read eagerly, validate the schema version, dimensions, ranks, finiteness of
required arrays, operator uniqueness, phases and source/pivot coefficient
shapes, then copy/freeze arrays. Diagnostics may contain documented NaNs.
Round-trip behavior and stored build diagnostics exactly; elapsed time and
logging side effects are not equality requirements.

Sigmond's `WritePivotToFile` uses `IOMap<UIntKey, ArrayBuf>` with file ID
`Sigmond--SinglePivotFile-CN` or `-RN`, XML header
`SinglePivotOfCorrMat`, key 0 `T @ R` in source `std::set` order, and key 1
`𝒵` in pivot order. Its header lacks alarm, normalization diagonal, source
coefficients, diagnostic samples and masks. It cannot supply a complete SS
build result without additional inputs. Do not fabricate those fields.

Implement a shared **low-level** HDF5 interchange reader for oracle use;
keep production `SinglePivot.read` restricted to complete SS pivots initially.
Interchange import/export is a tracked compatibility milestone before claiming
full replacement for workflows that use `ReadPivotFromFile`. A future importer
must accept the missing information or return a clearly distinguished imported
rotation object with unavailable build diagnostics. Export must map both bases
by Sigmond operator order and retain RN/CN identifiers. Test ArrayBuf layout,
complex encoding and bracketed `file.hdf5[group]` addressing on actual files.

Rotated **data** remain standard Sigmond bins/samplings files written through
SS; no new correlator format. Test grouped `Info`/`Header` and downstream
PyCALQ/Sigmond read-back before calling that workflow compatible.

## 5. Errors and warnings

`PivotError(ValueError)` carries `.stage` (`parameters`, `input`, `metric`,
`matrix`, `overlaps`), `.code` (Sigmond code or None), and eigenvalues/cutoff
when available. Include operator/time/sample context and numerical thresholds.
Keep the original `LinAlgError` as the cause. Missing cells/times keep the
existing `IncompleteCorrelatorError`; I/O uses established I/O errors.

| Case | SS behavior | Sigmond comparison |
| --- | --- | --- |
| Non-Hermitian basis, fewer than 2 operators | Input error | Constructor rejects |
| Invalid/nonfinite times, cut or alarm | Parameter error | Shared domain failures; stronger SS finite/type checks |
| Missing time/cell | `IncompleteCorrelatorError` | Matrix read fails |
| Negative normalization diagonal | Warn, then `abs` | Sigmond silently uses `abs` |
| Zero/nonfinite normalization diagonal | Input error | Undefined upstream arithmetic; explicit divergence |
| Metric alarm | Metric code −1 | Same threshold |
| Metric eigensolver failure | Metric code −2 | Unit-injected failure, not a guaranteed live oracle trigger |
| All metric eigenvalues cut | Metric code −3 | Same |
| Retained metric zero at ε=0 | Metric error, code None | Upstream inverse square root is undefined |
| Reduced-matrix alarm | Matrix code −6 | Same |
| Reduced eigensolver failure | Matrix code −4 | Unit-injected failure |
| All reduced eigenvalues cut | Matrix code −3 | Same, except upstream CN out-of-range condition-ratio read |
| Failed corrected null-space containment | Warn; result remains valid as a truncated solve | Upstream −5 tolerated, subject to §2.3 bugs |
| Exactly zero rotated complex VEV | Warn, phase 1 | CN undefined; RN already phase 1 |
| Connected pivot without attached VEVs | INFO availability reason | Upstream skips rephasing |
| Invalid diagnostic resample | Warn, preserve NaNs | Upstream enabled check can fail build |
| Missing/extra/duplicate fit levels, bad plans/data or zero normalization row | Overlap error | Shared domain plus SS type/metadata checks |
| Bad improved definition | Existing `ValueError`/`TypeError` style | Shared domain, canonical duplicate detection in SS |

Warnings are aggregated by stage and point to the user's call site. A clean
build has no warnings. Do not demand a Sigmond warning log marker for cases
where its implementation is silent or performs undefined arithmetic.

## 6. Logging and result ownership

Use the module logger. Build `pivot.summary()` solely from immutable build
state. `off_diagonal_check` returns its own `summary()`; overlap ordering is
represented by output coordinates and logged at DEBUG. Calling either must
not mutate a pivot or make a later pivot write depend on call history.
The CLI can combine these reports explicitly.

Use one flat snake-case namespace: Sigmond tag names for shared fields and
plain descriptive names for SS-only fields (no prefix). Include structured operator coordinates, ranks, all/retained
eigenvalues, cutoffs, null-space diagnostics and diagnostic availability.

| Source | Content | Logging |
| --- | --- | --- |
| `CreatePivot/ResamplingMode` | Actual sampling plan; bins explicitly distinguished | INFO |
| `DiagonalCorrelatorFractionalErrors` | Real diagonal fractional errors by stage/operator | INFO extremes, DEBUG full arrays |
| `RescaledMetricConditionVariation` | Eigenvalue full estimates/errors, all levels | INFO extremes, DEBUG all; absent reason for bins |
| `AnalyzeMetric` | Operator count, spectrum, rank, retained values | INFO rank/extremes, DEBUG spectrum |
| `AnalyzeMatrix` | `G` spectrum/rank and common-null-space diagnostic | INFO rank/verdict, DEBUG detail |
| `TransformationMatrix` | `T @ R` on source labels, plus SS pivot-basis matrix | DEBUG |
| `OffDiagonalChecks` | Per-time component counts/max/percentages | INFO per time on the check result |
| `ReorderEnergies` | Original labels, stable permutation, energies | DEBUG for overlap operation |
| SS fields | Numeric type, masks, phases, residuals, availability, elapsed time | INFO concise / DEBUG detail |

A nonlinear condition-number estimate/error, if shown, must be computed on
samplings using their correlated eigenvalue ratios; never derive its error
from the errors of two marginal eigenvalues. An indefinite/all-null metric
has no ordinary positive condition number; show its spectrum and availability
reason instead. Do not label the retained `G` ratio as the metric condition.

`__repr__` is one line with basis size/rank, times, cut and channel. Timings
are informational and excluded from numerical log parity. XML precision is
field-specific (`Coefficient` string formatting can differ from `putReal`);
calibrate comparison tolerances against the real logs rather than assume
all fields have 12 significant digits.

## 7. Explicit differences and deferred functionality

| Upstream behavior | SS behavior | Required evidence |
| --- | --- | --- |
| Wrong null-vector column and full-rank −5 | Correct subspaces; full rank passes | Separate divergence fixtures; compare invariant diagnostics |
| Optional null-space check | Always when metric is truncated | No effect on retained vectors; upstream early-failure branch documented |
| LAPACK phases | Canonical phase unless VEV phase applies | Gauge-aligned rotations; phase-dependent checks qualified |
| `std::set` row order | Caller basis order | Map source and pivot rows independently |
| Undefined zero scales/metric inverse/complex VEV phase | Clear error or defined phase fallback | SS assertions; upstream failure/NaN/unsafe behavior recorded, not copied |
| Diagnostic resample failure aborts enabled check | Diagnostic NaNs and warning | Build still uses valid full sample |
| Diagonal-only numerical rotation before τD | Full rotation; filtered export supports same coverage | Compare shared keys and values; benchmark extra cost |
| Missing improved-basis fractional-error log | SS computes it | Skip that unavailable upstream field |
| Zero-error component division yields NaN/inf | SS `_ratio` convention | Explicit zero-error fixtures |
| Mode S without subtraction writes no keys | SS exports meaningful raw rotated data | Source-confirmed bug; live regression test, not intended empty output |

Keep construction masks, improved operators and all B/S/U/A workflows in the
numerical scope. They are existing functionality, not discarded because of
assumed rarity. Preserve diagram/key coverage via the export filter.
The task currently parses `RemoveOffDiag` but `Pivot::doRotation` and
`writeRotated` pass `false` and use `τD` internally; distinguish low-level
capabilities from flags that are actually honored by the current task driver.

Deferred: rolling pivots, legacy fstream pivot writing, xmgrace output and
memory-name registries. Reusing the immutable object covers `AssignName`/
`GetFromMemory` in Python. Sigmond pivot-file interchange has the explicit
milestone in §4.8; SS-native persistence does not replace that compatibility.

## 8. Performance

Optimize after a correct baseline and oracle exist. The current `transform`
uses two `_contract`/`tensordot` contractions. Measure the proposed batched
matmul/einsum implementation on the actual strides and BLAS configuration;
do not assume every reshape is copy-free or that every moveaxis copies.
Include noncontiguous inputs and complex transforms of real buffers.

The shared kernel contracts sink and source axes into a preallocated output.
Chunk times/samples if needed to keep memory bounded. Avoid casting the entire
real input to complex simply to choose CN semantics. The optional diagonal-only
numerical kernel remains a later optimization if full output is a measured
bottleneck; the export filter does not require it.

Benchmark `N ∈ {4,12,30}`, retained ranks including substantial truncation,
`T=48`, and sample lengths `{201,2001}`, real/complex buffers, bins/samplings,
VEVs and improved bases. Use fixed seeds, warmups, repeated medians, reported
BLAS/thread budgets and isolated peak-RSS measurements. Separate build,
metric diagnostics, rotation, collection conversion, file write and total.
Compare SS and `sigmond_batch` on the same exported key coverage. Also report
the cost of SS's full-matrix superset, so startup/I/O and extra cells do not
make the comparison misleading.

Targets, not portable CI timing assertions:

- Full-sample build near 20 ms for N=30 on the recorded reference machine;
  resampled metric eigensolves reported separately.
- Rotation near 10× a buffer copy, with no regression for small cases.
- Peak workspace bounded by output plus one contraction intermediate and
  documented chunk temporaries, beyond the resident input.
- End-to-end time competitive with Sigmond; report regressions and explain
  them. Do not require SS to win every row regardless of output coverage.

## 9. Validation

### 9.1 SS and mathematical tests

- Positive-definite and exactly rank-deficient GEVPs: metric orthogonality,
  diagonalization, `𝒵=B Y`, order, cutoffs and ranks. Test reconstruction
  only under §2.2's conditions and residuals when either cut removes signal.
- CN and RN cutoff floors, exact-cut equality, all-cut cases, negative alarms,
  `ε>1` (currently allowed upstream), exact-zero metric with `ε=0`, nonfinite
  inputs and unit-injected LAPACK failures. No relaxed PSD tolerance hidden
  behind the alarm.
- Exact spectra with positive diagonals: `Ĉ_full(τ0)=I`, diagonal at τD,
  `Ĉ_nn(t)=exp(-E_n(t-τ0))` in the untruncated case, and overlaps using
  `A_n=exp(E_n τ0)`. Test both directions of τD−τ0 and energy reordering.
- Fixed-pivot raw bins/resamplings commute to floating-point tolerance;
  connected rotation subtracts once. Verify full-sample and resample B/U/S
  differences, including complex off-diagonals and unavailable rephasing.
- VEV phase, exact-zero phase fallback, missing VEVs, RN sign handling,
  and construction-mask/build-versus-rotation behavior.
- Basis permutations for nondegenerate states, tie handling, improved/source
  rotation equivalence, rectangular and composed transformations, mixed BL/GI
  labels, exact channel equality for averaged/definite momentum and rows.
  Compare projectors for degeneracies, not arbitrary individual vectors.
- Scalar real fits, canonical duplicate keys, missing/extra levels, ensemble/
  plan/length mismatch, failed fit NaNs, negative amplitudes, energy ties,
  sample-independent row normalization and zero row errors. Check collection
  keys and metadata for derived overlap cells.
- Component diagnostics: CN/RN counts, exact sigma edges, zero error, invalid
  samples, NP=1 and empty times; B/U connected check uses the chosen plan.
- Read/write multiple groups, protection on collisions, schema errors,
  interrupted writes, complete diagnostics, read-only arrays and no caller
  mutation. Round-trip through existing data writers/readers.
- General `transform` remains valid for non-Hermitian values and complex VEVs.
- `caplog`/warnings: reproducible summaries, no hidden print/progress,
  no call-history mutation, and aggregated warnings in their stated cases.

### 9.2 Sigmond oracle

**Phase 0 establishes the harness before implementation.**

The review environment currently imports `sigmond` but `import sigmond.tasks`
fails with `No module named 'sigmond._core.tasks'`. Verify the installed
extension/editable-loader resolution before deciding whether reinstalling
`pip install -e ../sigmond` is the appropriate repair. Do not make one
machine's reinstall command a production dependency.

The bindings *do* expose `Pivot`, `XMLHandler`, `LogHelper.output`, rotation,
fit-info insertion and overlap operations. They do not expose the metric
diagonalizer directly. Use `sigmond_batch` for the task/log/file oracle because
it exercises the actual PyCALQ path and provides process isolation, not because
logging is unavailable through bindings.

Add a `sigmond_oracle` pytest marker and a centralized availability fixture.
`SS_REQUIRE_SIGMOND=1` makes unavailable requested oracle dependencies fail,
including the existing effective-energy oracle tests. Distinguish binary
availability, binding availability, CN/RN build and version. A detected
runtime error is a test failure, not a new skip. Unit tests remain runnable
without installing Sigmond.

`tests/sigmond_oracle.py` should provide:

- Input writers using SS's `SigmondWriter` and canonical matrix keys;
  `CorrelatorMatrixInfo.to_sigmond_xml` and `Operator.to_xml` generate XML.
  Avoid manual operator strings and a second data-file serializer.
- Generated task XML, a subprocess with a timeout, captured stdout/stderr,
  parsed task-status/errors and verified expected output datasets.
  A zero process exit code alone is insufficient: tasks/write errors can
  live in the log. Expected failure fixtures assert the relevant stage.
- Correct grouped file addressing, actual ensemble extents/streams/bin count,
  and sampling headers. Preserve rebinning/omission metadata; test the simple
  unrebinned case first rather than assume every input needs a fixed inline
  ensemble string.
- Task construction with metric/null-space diagnostics, printed source
  transformation, pivot output and rotated data. Overlap tasks insert known
  amplitude/energy samplings so fitting differences are not conflated with
  pivot parity; test reorder and ZMagSquares separately.
- A low-level pivot reader for both RN/CN headers/ArrayBuf layouts and separate
  source/pivot row mappings. Never infer `std::set` order from sorted display
  strings. Decode the header order and validate against known operator indices
  or the binding comparison before associating numeric rows with labels.
- Log parsing keyed by canonical labels and times, tested on the existing
  real logs, and a staged report (`build`, `rotation`, `overlaps`).

For bootstrap oracle cases supply identical SS-written samplings. SS's NumPy
bootstrapper does not reproduce Sigmond's bootstrapper from a shared seed.
For B/U connected bootstrap checks, supply the identical rotated-piece
samplings or explicitly restrict bin-generated checks to jackknife. Linear
jackknife from bins agrees when binning/omissions and sample ordering agree.

Keep small goldens (`N≤6`, `T≤16`, sample length ≤41), with seeds, generated
input XML/data, log, output arrays, source and pivot labels, numeric type,
Sigmond commit/build information and binary hash. Include generator version
and artifact hashes so regeneration is reviewable. The current `.gitignore`
ignores `data/`, `*.hdf5` and `*.npz`; use a tracked fixture directory with
explicit ignore exceptions when implementing it. Do not silently depend on
ignored local files. Goldens run everywhere; live comparisons additionally
run when available, and are mandatory for release parity claims.

Cover real/complex-valued inputs against CN, genuine RN fixtures separately,
B/S/U/A, raw and connected VEV cases, exact/noisy/near-degenerate spectra,
metric and G truncation, both time directions, improved/averaged bases,
construction masks, source/pivot row order and error/divergence cases.
Do not run an unsafe upstream all-null diagnostic in-process; isolate it and
record an unsafe reference outcome rather than demand a specific crash.

**Numerical comparison policy:**

- Scale tolerances with both metric whitening conditioning and retained G
  conditioning, matrix norms and reconstruction residuals. The retained G
  inverse condition alone cannot bound error amplification from the metric.
  Use relative plus absolute tolerances and an explicit ceiling; a fixture
  whose tolerance becomes meaningless is an ill-conditioned fixture, not
  an automatic pass. Cut-boundary fixtures test masks under controlled exact
  spectra and separately document near-threshold LAPACK sensitivity.
- For nondegenerate columns, find unit phases with
  `R_ss ≈ R_sig diag(p)`, using the proper source/pivot rows. Then
  `C_ss ≈ diag(p)† C_sig diag(p)` and
  `v_ss ≈ diag(p)† v_sig`. This fixes the conjugation direction in comparisons.
  Apply the **same** phases to `𝒵`.
- Native real/imaginary sigma buckets cannot be compared after an arbitrary
  phase rotation by adjusting percentages or marginal errors. Align the
  **whole sampled buffer**, then recompute component estimates/errors and
  compare checks in that common gauge. Native log percentages are compared
  only where the gauge is compatible (e.g. nonzero VEV rephasing fixes it),
  or in controlled fixtures with matching component axes. Otherwise declare
  the log field phase-dependent. A separate gauge-invariant diagnostic would
  be an SS addition, not Sigmond parity.
- Near/exact degenerate clusters: compare eigenvalues, retained subspaces,
  whitened projectors and block rotations. Only exactly degenerate subspaces
  permit arbitrary unitary alignment without changing the diagonal solve.
  Individual rotated resample diagonals and `abs(𝒵_in)^2 abs(A_n)` generally
  change under such a mixing; do not promise per-level parity for unresolved
  clusters. In near-degenerate cases require a well-conditioned identified
  basis or report the ambiguity; no fixed `1e-8` cluster rule hides a failure.
- Compare metric diagnostic eigenvalue full samples and resamples directly
  (sorted ascending, with their own normalization per sample). Compare
  available build log fields at their serialization precision.
- For corrected null-space checks, compare invariant subspaces and use
  nondegenerate diagnostic fixtures for per-vector log entries. The wrong
  upstream projection values can differ in more than just the final verdict.
- Compare overlaps samplewise against the unnormalized output, then compare
  normalized full estimates/errors against the ZMagSquares table. Energy
  order is exact, including stable ties and original label mapping.
- Sigma bucket-edge cases are tested explicitly. Random fixtures stay away
  from edges, or report a tightly bounded edge ambiguity; do not exclude
  entries from percentages while still claiming the original denominator.

Separate domain failure parity, SS validation errors, injected numerical
failures and upstream undefined behavior. Each §7 exception has a named test.

### 9.3 Benchmarks

`benchmarks/pivot.py` reports §8 measurements outside CI. Numerical acceptance
precedes performance tuning. Performance changes rerun the same meaningful
rotation and metadata regression checks; do not repeatedly broaden unrelated
tests once those pass.

## 10. Implementation phases and acceptance

0. **Reference harness:** inventory CN/RN/binding availability; verify Sigmond
   reads SS-written bins/samplings and grouped output; parse existing logs;
   implement required-oracle behavior and tracked first goldens. Confirm
   mode S's unsubtracted empty-write bug and diagnostic flag limitations.
1. **Solver and basis groundwork:** ndarray kernels, `solve_gevp`, errors,
   numeric policies, cuts/null-space diagnostics; averaged/mixed-label
   `transform` and thin `improve`. Accept identities, rank/cut fixtures and
   oracle `R`/`𝒵` comparisons before optimizing contractions.
2. **Full single-pivot workflow:** `from_matrix`, masks, phases, build
   diagnostics, rotation and B/S/U/A VEV semantics. Accept samplewise parity,
   mode differences, improved/source basis equivalence and keyed export.
3. **Fits, overlaps and durable reuse:** fix fit-result ensemble metadata and
   data-ensemble inference, then real fit samplings, stable energy
   reorder, both overlap normalizations, off-diagonal checks, immutable
   reports and complete SS persistence. Accept oracle/round-trip tests and
   downstream rotated-file read-back. Document remaining pivot interchange.
4. **Performance and user surfaces:** benchmark and optimize the shared
   contraction; plug into existing plotters and CLI assembly/render pipeline.
   Reuse matrix diagonal/grid/heatmap views and offer overlap bars/heatmaps
   carrying the chosen normalization and level labels. Later projected
   diagnostics must be labeled as projected values rather than samplewise
   eigenvalues. Provide a dedicated `ss pivot` command (sharing `ss query corr`'s assembly) with explicit
   sampling/VEV requirements and `--write-rotated`; avoid direct print calls
   in numerical APIs. Mirror exports in lazy package declarations and stubs.
5. **Separate extensions:** time-dependent projected diagnostic (§4.7) and
   production Sigmond pivot interchange (§4.8), with their own acceptance.
   A rolling pivot remains a distinct future design.

Each numerical phase passes unit/golden tests and its relevant live oracle
under `SS_REQUIRE_SIGMOND=1`; unavailable RN binaries leave RN parity clearly
unverified until a genuine RN run is recorded. Performance targets are not
phase gates on arbitrary machines. No implementation phase is complete merely
because the optional oracle silently skipped.

## 11. Remaining decisions requiring evidence

- Which downstream consumers require Sigmond pivot import/export versus only
  grouped rotated data? Exercise the PyCALQ read/fit path before declaring
  SS-native pivot storage sufficient for those workflows.
- Does full rotation dominate a representative N≈30, thousands-of-samplings
  run after the shared contraction optimization? Only that measurement
  justifies a dedicated diagonal-only numerical kernel.
- Which Sigmond upstream revision fixes the null-space and mode-S writer
  defects? Keep old-version divergence fixtures and add new-version parity
  rather than changing the pinned oracle invisibly.
