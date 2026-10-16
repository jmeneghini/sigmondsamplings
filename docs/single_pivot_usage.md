# Fixed single-pivot GEVP

`SinglePivot` builds once on the supplied full estimator and applies the same
rotation to every sampling, including sampling zero. It supports complex and
real build policies, two eigenvalue cuts, construction masks, improved bases,
connected correlators and the B/S/U/A workflows described in
[single_pivot_design.md](single_pivot_design.md).

```python
from sigmondsamplings import SinglePivot, SamplingInfo

plan = SamplingInfo.jackknife_for(index.ensemble_info)
matrix = index.matrix(
    basis, sampling_info=plan, subtract_vev=True, include_vevs=True,
)
pivot = SinglePivot.from_matrix(
    matrix, norm_time=3, metric_time=3, diagonalize_time=6,
    min_inv_condition=1 / 150,
)
rotated = pivot.rotate(matrix)
report = pivot.off_diagonal_check(rotated)

# Fit rotated.correlator(op, op).to_real() using SamplingFit.from_data.
# Parameter samplings now inherit the data ensemble unless explicitly overridden.
# fits: {rotated operator: FitResult}; "A"/"E" are the parameter names.
z2 = pivot.overlaps(fits, amplitude="A", energy="E", normalize=True)
rotated.write(
    "rotated.hdf5", diagonal_only_before_time=6, upper_triangle_only=True,
)
pivot.write("pivots.hdf5", group="channel")
restored = SinglePivot.read("pivots.hdf5", group="channel")
```

The `fits` mapping is keyed by the rotated operators (fit results, or the
amplitude samplings themselves with `energies=` samplings). Parameters must be
complete, finite, real scalar samplings from one ensemble and one sampling plan.
The exponential amplitude convention is `A exp(-E t)`. Overlaps use
`abs(zmatrix)**2 * abs(A)`; `normalize=True` divides every sampling by the
**full-sample** sum of its operator row. Energy ordering is a stable full-sample
sort and preserves the original rotated labels. The returned cells carry
`OverlapObsInfo` metadata through the ordinary Sigmond writer/loader.

A pivot holds three parts: `pivot.settings` (a frozen `PivotSettings`: the
three times, cut, alarm, numeric type and construction masks), the solve
itself (`pivot.gevp`, `norm_diagonal`, `column_phases`, `rotated_operators`,
`source_transformation`) and `pivot.diagnostics` (a frozen `PivotDiagnostics`).
`pivot.transformation` and `pivot.zmatrix` are derived from the solve, the
normalization diagonal and the phases; `pivot.gevp.eigenvectors` and
`pivot.gevp.zmatrix` are the raw solver gauge before normalization and phases.
The pivot and its arrays are immutable; ensemble/sampling metadata and the
labeled diagnostic views are copies. `summary()` is independent of subsequent
rotations, checks and overlap calls. Numerical failures raise `PivotError`; recoverable
issues emit `PivotWarning` and remain in the build diagnostics.

`solve_gevp(A, B, min_inv_condition=...)` is also public. It has no implicit
normalization-time rescaling and returns the raw solver gauge, including the
whitening matrix, retained ranks, both spectra/cutoffs and reconstruction
residual. It rejects non-Hermitian off-diagonals (`rtol=1e-10, atol=1e-12`).
The labeled pivot instead follows Sigmond's upper-triangle/real-diagonal reader
convention. Retained exact-zero metric eigenvalues cannot be inverted.

For improved operators, use `matrix.improve(basis, definitions)`, where each
column's coefficients are `T[i,a]` in `T† C T`. The operator coefficients are
therefore **conjugated**. Original BL labels can be retained only by exact
unit-selection columns. GI labels can define arbitrary linear combinations,
and averaged channels are supported by exact channel equality. Composed source
coefficients remain available on the pivot; it can rotate either the entire
recorded source basis or the entire pivot basis, in any input label order.

For B/U/A, rotate the raw bins or raw samplings with the connected pivot.
Available VEVs are rotated only when the pivot was built with connected data
and attached VEVs; these results contain the **real** rotated VEVs. For S, rotate
the already connected samplings; no second subtraction occurs and no VEVs are
attached to the result. B/U connected checks jointly resample bins using the
build plan before subtracting the rotated real VEV product. Their connected
resamples can differ from S by the complex VEV fluctuation terms.

`numeric_type="complex"` is the default even for real buffers. Rotated matrices
retain that convention for writing: CN exports contain explicit zero Im keys
without casting the numerical matrix to complex. `numeric_type="real"` selects
RN's cutoff floor/component policy and rejects nonzero imaginary input.
Generic writers can request `SigmondWriter(include_imaginary=True)` explicitly.
`upper_triangle_only`, `diagonal_only_before_time`, and `remove_off_diagonal`
filter exported keys, while numerical rotation still computes the full matrix.

`ss pivot` uses the same matrix assembly and plot pipeline as `ss query corr`:

```bash
ss pivot input.hdf5 --matrix 0 --sampling jackknife --subtract-vev \
  --norm-time 3 --metric-time 3 --diagonalize-time 6 --format json \
  --write-pivot pivot.hdf5 --pivot-group A1g --write-rotated rotated.hdf5 \
  --diagonal-only-before-time 6 --upper-triangle-only
```

`--min-inv-condition` defaults to 1/150. Existing output groups are kept unless
`--overwrite` is given.

For stored samplings, omit `--sampling`. All three pivot times are explicit.
Add `--plot diagonal`, `grid`, `heatmap`, or `effective-energy` to view the
rotation using the existing correlator plotters. In Python,
`OverlapPlotter(z2).plot_bars(operator)` and `.plot_heatmap()` show overlap errors,
original level labels and the chosen normalization convention.

## Validation and remaining milestones

The pinned CN and RN reference is Sigmond commit
`926e10ee52a2becef96c2f8ec96a729487089d55`. The process-isolated harness validates
task errors and actual output datasets, decodes source/pivot row order from the
pivot header, and aligns one common column gauge for rotations and overlaps.
Tracked fixtures include input XML/data, logs, native pivot/rotated output,
arrays, generator version and artifact/binary hashes. Regenerate explicitly with
`PYTHONPATH=. python tests/generate_pivot_goldens.py`. There are 11 CN and nine RN
fixtures, including stored bootstrap samplings. To regenerate both builds, set
`SS_SIGMOND_BATCH_REAL` to a genuine REALNUMBERS executable and pass
`--numeric-type both`.

```bash
python -m pytest tests/test_pivot.py tests/test_pivot_oracle.py
SS_REQUIRE_SIGMOND=1 SS_SIGMOND_BATCH_REAL=/path/to/real/sigmond_batch python -m pytest tests/test_pivot_oracle.py tests/test_correlator_effective.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python benchmarks/pivot.py --ops 20 --samples 2001 --complex --rank 8
```

`SS_REQUIRE_SIGMOND=1` fails requested oracle tests when binaries/bindings are
unavailable. The harness handles the installed editable extension/stubs shadowing
issue using distribution metadata; this is not a production dependency.
CN live checks cover real/complex buffers, B/S/U/A, masks, improved bases, metric
truncation, both time directions, unnormalized overlap samplings, stable energy
ordering and the normalized estimate/error table. A genuine RN build checks
both time directions, metric truncation, improved bases, stored bootstrap
samples, and all four connected VEV workflows. Unit tests cover RN cutoffs,
undefined arithmetic, degenerate projectors, sampling diagnostics, immutable
ownership, persistence failures and downstream file read-back. RN references
were built outside the source tree, with no upstream source modifications:

```bash
cmake -S ../sigmond -B /tmp/ss-sigmond-rn -DNUMBERS=real -DSKIP_SIGMOND_QUERY=ON -DSIGMOND_GENERATE_STUBS=OFF
cmake --build /tmp/ss-sigmond-rn --target sigmond_batch_cli -j 2
```

The native pivot header is required to identify RN/CN, so a real-valued CN buffer
cannot accidentally stand in for an RN reference. Live overlap comparisons
currently use the CN bindings; RN rotation coverage does not claim RN overlap
binding parity.

`write`/`read` use a versioned **SS-native** HDF5 format (schema 2): JSON
metadata for settings, labels and scalars, and one dataset per array. With
`overwrite=True` an existing group is deleted and rewritten. They are not
production Sigmond pivot-file interchange. Production interchange and time-dependent projected diagnostics
remain the separate extensions in design phase 5; rolling pivots remain out of
scope. Benchmark results report full solve, complete build/diagnostics, kernel,
rotation, collection, file write and total separately. Performance targets are
not asserted in CI, and the full benchmark sweep has not been recorded. The
practical range is up to 20 operators. Matrix products batch over samples and
chunk over time, including Hermitian completion, with a roughly 32 MiB temporary
budget (at least one time slice). Process peak RSS also includes fixture
construction and the old-kernel comparison; it is not isolated rotation memory.
