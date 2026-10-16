# Observable metadata implementation log

## Phase 1: cell preservation and immutable particles (2026-10-13)

Scope follows phase 1 of [the plan](observable_metadata_plan.md).
Existing working-tree changes were retained. Later phase status is recorded below.

### Baseline

Before editing, recorded revision
`5b7841572aa93523dd1a9fbb02a8abd1e625df23` and working-tree fingerprint
`d91dd4ab15f1ec7273822c21a0d7d646505177f639fdcf8292d41c5a8d9eccd8`.
The fingerprint method and all failing/error/skipped test IDs and reasons are
in [the baseline record](observable_metadata_phase1_baseline.json).

- Array/particle baseline: 97 passed.
- Full-suite baseline: 1329 passed, 96 failed, 6 errors, 65 skipped.
  Failures include missing local data and pivot fixtures as well as other
  pre-existing failures; the record preserves the individual reasons.

### Changes

- Stack and concatenate keep the supplied per-cell infos when parent infos
  differ. The array can have mixed metadata without replacing its cells.
- Negative stack axes resolve against spatial dimensions, keeping the sample
  axis last and aligned with the metadata. Out-of-range spatial axes reject.
- `Particle` is a frozen dataclass with normalized names and generated
  equality/hash. Equality with strings was deliberately removed; compare
  `.name` or another `Particle`. Parsing, labels, repr, nonnegative momentum
  validation and momentum-only ordering are retained.
- Regression tests cover mixed/shared parents, bins/jackknife/bootstrap,
  stack axes, concatenate, scalar selection, collections, transpose, diagonal
  extraction, default cell infos, particle mutation rejection and identity.

No correlator/operator APIs were removed in this phase. Numerical kernels were
unchanged. This phase does not establish new Sigmond agreement; unavailable
native checks remain recorded as baseline skips.

### Validation

Focused checks (array, particle, energy collection and kinematics): 190 passed;
six additional out-of-range stack-axis cases also passed in the full suite.

Full suite after implementation: **1366 passed, 96 failed, 6 errors, 65 skipped**.
Comparing JUnit results with the baseline found no new failures or errors and
no changes to existing outcomes. The old string-equality test was replaced
deliberately; all 38 new test cases passed (net increase: 37 passing tests).

Phase 1 is complete. Phase 2 begins with shared `SectorInfo` fields, forwarding
facets, channel conversion and compatible energy/general persistence.

`git diff --check` and parsing of changed Python files passed. Ruff could not run
because it is not installed in the active Python environment.

## Phase 2: shared physical sectors (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`1bd2fe9fba8ecfc7f57bcf23ab3d49c162675844b62dfa9653373b61320a5e39`.
The method and individual failing/error/skipped test IDs are in
[the phase 2 baseline record](observable_metadata_phase2_baseline.json).
Baseline: 1366 passed, 96 failed, 6 errors, 65 skipped.

### Changes

- Frozen `SectorInfo` owns PSQ and irrep alongside the existing physical fields.
  PSQ must be a nonnegative integer; irreps are validated. Exact equality/hash,
  partial compatibility, display and serialization include the new fields.
- Ordinary observable/energy/array constructors accept keyword-only
  `sector_info`; legacy PSQ/irrep arguments merge without erasing known fields
  and reject contradictions. Writable facets replace the frozen sector, so
  shallow metadata copies can be edited independently.
- Physical facets and `strangeness`/Sigmond-vocabulary `isospin` aliases are
  available to filters, DataFrames and CLI attribute listings. Summaries read
  public physical properties instead of relying only on `vars()`.
- `Channel.to_sector()` supplies isospin, strangeness, PSQ and irrep. Correlator
  and VEV sectors derive from native channels and are read-only. Correlators
  expose both source and sink channels; unequal physical sectors give an unknown
  common sector. Opaque keys retain their native text and parse diagnostics.
- Observable equality/hash includes the whole sector in the shared base.
  `EnergyObsInfo.sector` remains the compatibility `(psq, irrep)` pair.
- `EnergyObsInfo.from_channel()` requires ensemble context explicitly. Its
  optional `rot_idx` required bringing forward the small rotation-index portion
  of phase 5: validation, attrs, equality/hash, labels and separate `ROTn` naming
  and parsing. No binding inference or energy ordering was introduced.
- SS HDF5 stores nested sectors through existing `ObsMeta` records, with
  recursive NumPy normalization. Energy/general flat PSQ/irrep and SH `ni_pairs`
  remain generated compatibility projections. Legacy flat sector fields and
  aliases read at the boundary; contradictions reject. SH sector edits keep
  their legacy particle momentum projection consistent.

No correlator/operator APIs were removed. Existing native keys and numerical
kernels were retained. Tests cover reading native keys without SS annotations,
but independent Sigmond agreement remains unverified because the native
bindings/query runtime is unavailable (see recorded skips).

### Validation

The expanded focused suite passed 371 tests, with 19 native-reference skips.
Final SH-projection and unequal-channel regression cases were added afterward
and passed in a 123-test focused run.

Final full suite: **1415 passed, 96 failed, 6 errors, 65 skipped**. Comparing
JUnit results against the phase 2 baseline found no changes to existing test
outcomes; all 49 new sector tests passed. Changed Python files also parse.

Phase 2 is complete. Phases 3–7 remain pending. Phase 3 introduces immutable
element snapshots and at-coordinate cells, canonical identities/codecs, explicit
pivot bindings and per-cell effective-energy metadata, then removes the old
array-level effective-energy metadata API.

`git diff --check` passed. Ruff remains unavailable in the active environment.

## Phase 3: element identities, rotation bindings and effective cells (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`515f681c012306f6e116f6f84c7e86c9d3dd6280d2399572f89f2662fcac777c`.
The method and failing/error/skipped test IDs are in
[the phase 3 baseline record](observable_metadata_phase3_baseline.json).
Baseline: 1415 passed, 96 failed, 6 errors, 65 skipped. Existing unrelated
working-tree changes were retained.

### Changes

- Added `metadata.py` with registered `AtObsInfo`, immutable
  `EffectiveEnergyInfo`, `CorrelatorSource`, `SeriesInfo`, `RotationBinding`
  and optional `EnergyTarget` snapshots. Records contain frozen keys/scalars
  and serialized source/context identities, with no live model, callable or
  mutable source-info reference. Optional target access reconstructs independent
  lab-energy metadata, retaining reference normalization, without assigning an
  energy-sorted level.
- At-coordinate identities include typed coordinates, complete source/settings
  and normal observable/ensemble context. Versioned, canonical identity records
  generate readable names with 32 SHA-256 hexadecimal characters; integer and
  floating coordinates and adjacent finite floating values remain distinct.
  Labels and overrides are persisted separately and do not affect identity.
  Equality/hash use metadata only, without inspecting sample buffers.
- Correlator cells expose time-independent `.element` and native `.time`, plus
  separate sink/source rotation bindings. Only a known diagonal with equal
  bindings has `rot_idx`; raw GI names, operator indices and transformation
  presence establish no binding. Renaming a native key clears its bindings.
- `SinglePivot.rotate()` annotates its values before returning the matrix.
  Selection, reordering, transpose, diagonal/scalar extraction and collections
  preserve bindings. Exact unit selections through basis transforms preserve
  existing bindings per output mode; mixing establishes none for mixed modes.
- Effective-energy calculations validate source metadata/context along time,
  including native form/component/bindings and recorded time. Each output cell
  is `AtObsInfo(EffectiveEnergyInfo(...), "time", source_time)`; each non-time
  position shares one element along time. Resolved extent, kind, step and finite
  constant subtraction are recorded. Numeric arrays use generic descriptors
  from parent identity and named non-time coordinates, without parsing physics
  from placeholder names. The numerical kernels and failure policies are unchanged.
- Nested metadata uses the existing SS HDF5 `ObsMeta` table. Eager/lazy reads
  reconstruct registered kinds and reject unsupported versions/kinds,
  contradictory identity records and names, and conflicting duplicate table
  records. Generated-ID collisions and raw-key collisions between different
  pivot bindings reject, including append with `overwrite=True`; distinct raw
  operator namespaces are required. No sample-based deduplication was added.
- Removed `EffectiveEnergyObsInfo` and its exports directly. Its replacement is
  per-cell `AtObsInfo`/`EffectiveEnergyInfo` with plain `ArrayObsInfo` at array
  level. Updated plotter metadata lookup, exports/type declarations, the example
  notebook, current API/design docs and numerical tests. Plot offsets resolve
  per plotted series and apply by default only to the time dimension.

Rich source/element metadata is guaranteed only for SS HDF5. Native fstream and
legacy Sigmond formats retain keys and samples but cannot carry all these
annotations. Native key text remains separate from SS annotations.

### Validation

All **43 new element/rotation/codec tests passed**. They cover the full
matrix → pivot → diagonal → effective energy → scalar/collection → HDF5 → read
path, explicit `shape_by` reconstruction, multiple pivots/settings/components,
generic arrays, immutable snapshots/targets, binding preservation and clearing,
off-diagonals without scalar modes, finite/typed identities, label overrides,
collision rejection and malformed/unsupported records.

The existing `test_plot_offset` cases and explicit-extent test were deliberately
updated to inspect `cell.element`, replacing their removed array-info assertions.
Their numerical/offset coverage remains. A replacement test verifies that the
old class/export is absent; no compatibility alias was introduced.

Expanded focused checks: 208 passed, 8 native-reference skips, followed by
43/43 passing new tests after final presentation/target checks. Final full
suite: **1458 passed, 96 failed, 6 errors, 65 skipped**. JUnit comparison against
the phase 3 baseline found no changes to any existing test outcome.

Changed Python sources parse and `git diff --check` passes. Independent Sigmond
agreement remains **unverified**: native bindings/query checks are unavailable
and retain their recorded skips. Supported closed-form numerical checks pass.

Phase 3 implementation is complete. Phases 4–7 remain pending. Phase 4 adds the
computed `.ts` view, shared label helpers and full time-series plotter adoption;
it will recover elements/offset/extent directly from these durable cells.

## Phase 4: computed time-series views and shared labels (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`0d52e56bf692dc84f2f92c46c84d0c9240e095e4a99bca26b3e4bb236ddca500`.
The method and failing/error/skipped test IDs are in
[the phase 4 baseline record](observable_metadata_phase4_baseline.json).
Baseline: 1458 passed, 96 failed, 6 errors, 65 skipped. Unrelated working-tree
changes were retained.

### Changes

- Added public `TimeSeriesView`/`TimeAxis` and computed `SigmondData.ts`.
  Named time coordinates must be unique finite real numbers. Floating values
  retain their type/coordinates for plotting and selection; the effective-energy
  calculator retains its integer-time requirement.
- `.ts.elements` validates complete source context along time and returns an
  object DataArray over non-time dimensions, scalar for a one-dimensional
  series. Actual at-time elements and native time-independent correlator keys
  remain accessible; built-in placeholders yield generic serializable records.
- `.ts.axis.offset` resolves effective-element offsets or zero; `.extent`
  resolves recorded calculation extents, otherwise each series' cell ensemble.
  Common queries reject differences, including known versus unknown extent.
  No independent axis state, boundary field or cache is attached to the data.
- `.ts.at(t)` delegates exactly to `sel(time=t)`. `.window(a, b)` uses an
  inclusive coordinate mask, preserves input order and rejects reversed bounds.
  Empty time axes allow coordinate/window access but reject element/axis queries.
  `.effective_energy()` and `.plot()` delegate to the existing calculator and
  `SeriesPlotter`.
- Plotters resolve each series after splitting stacks, recover metadata through
  the computed view and honor explicit `x_offset`. Default time offsets apply
  only to the time dimension. Fit overlays use their selected series' settings
  and display offset without changing fit coordinates or reevaluating models.
  Shared y labels avoid claiming the first rotation mode for a multi-mode plot.
- Added small backend-independent helpers in `labels.py` for literal
  `{VAR}`/`{var}` substitution, energy/mass symbols, reference denominators,
  effective-energy decoration and rotation labels. At-time/correlator cells,
  general energy labels and plotters use them; existing general/energy label
  entry points and default conventions remain available.
- Explicit label overrides, including empty strings, work through ordinary,
  energy/SH and keyed metadata and survive direct codecs and SS HDF5 reads.
  Overrides remain outside identity/hash. Removed the obsolete private
  `_reconstructs_own_label` persistence hint; generated defaults are reconstructed,
  while explicit overrides are always written.
- Updated the README and example notebook to demonstrate `.ts` and explicit
  collection-to-time-series reconstruction.

### Validation

**42 new tests passed**: 40 computed-view/label/codec tests and two fit-overlay
regressions. Coverage includes unsorted inclusive windows, empty and fractional
coordinates, scalar times, generic/native/effective elements, transpose and
HDF5 reconstruction, mixed offsets/extents, inconsistent source context,
reference-normalized labels, label overrides, and unchanged fit coordinates.

Expanded focused suite: 287 passed, 19 native-reference skips; the final
40-test time-series suite also passed after the last direct-codec checks.
Full suite: **1500 passed, 96 failed, 6 errors, 65 skipped**. JUnit comparison
against the phase 4 baseline found no changes to existing test outcomes.
Changed Python sources parse, the example notebook remains valid JSON, and
`git diff --check` passes. Ruff remains unavailable in the active environment.

Numerical kernels and symmetric-kind assumptions are unchanged. Native Sigmond
verification remains unavailable, with the recorded reference checks skipped.

Phase 4 is complete. Phases 5–7 remain pending. Phase 5 adds explicit energy
roles, centralized NI naming and copied energy-level ordering.

## Phase 5: energy roles, NI names and copied level ordering (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`9c2521d85f05ba775d506a9352dfefdb5aeedc1032d2a37bdcfb268788600e36`.
The fingerprint method and failing/error/skipped IDs are in
[the phase 5 baseline record](observable_metadata_phase5_baseline.json).
Baseline: 1500 passed, 96 failed, 6 errors, 65 skipped. Unrelated working-tree
changes were retained.

### Changes

- Added explicit energy roles outside `SectorInfo`: `hadron` on the SH
  specialization, ordered `ni_particles` for NI sums/shift baselines/`qcmsq`,
  and `non_interacting` for NI sums. `particle` aliases SH `hadron`; `particles`
  is now a read-only compatibility projection. SH momentum comes from its
  authoritative sector. Invalid SH shift/NI roles and conflicting legacy/new
  inputs reject construction and codec loading.
- Legacy adapters use observable kind, energy type and the NI marker rather
  than constituent count alone. New attrs include explicit roles alongside
  legacy flat sector and `ni_pairs` projections. Eager and lazy SS HDF5 reads
  preserve the roles, reference mode, and sector. Analysis summaries expose
  the new properties and compatibility facets.
- Centralized NI names as `[PSQp_]elab_ni_constituents[_ROTn][_ref]`. Unknown
  total PSQ remains unknown. Name parsing retains constituent order, repeated
  particles, arbitrary counts and per-constituent momenta; shift parsing also
  keeps constituent momenta separate from total PSQ. Kinematics constructs
  explicit NI metadata through this naming path and resolves SH masses from
  their hadron role. Shift-particle assignment writes `ni_particles`.
- Added `with_level_indices(overwrite=False)` to energy collections. It returns
  the same concrete collection kind with independent metadata and shared sample
  buffers. It ranks finite real full-sample values once, stably, by ensemble
  and exact `SectorInfo`; collection order and resamples remain unchanged.
  `rot_idx` is preserved. Unknown PSQ/irrep, missing or mixed energy conventions,
  mixed reference conventions, duplicate non-null rotations, SH/NI entries and
  existing assignments without explicit overwrite reject ranking.
- Interacting-spectrum views exclude NI sums. Generic collection deduplication
  remains metadata-based. Exact sectors can still map to the same legacy
  storage address; the writer rejects such collisions. README documents the
  roles, ordering API and identity requirements.

### Validation

51 new tests cover role aliases and rejection rules, old/new codec conflicts,
known/unknown total-PSQ NI names, references and rotations, repeated constituents,
shift parsing, eager/lazy HDF5, exact-sector/ensemble ranking, stable ties,
independent metadata/shared buffers, overwrite behavior, invalid ranking inputs,
interacting views and legacy-address collisions.

The focused energy/kinematics/sector/time-series suite initially passed all
244 tests; the final role suite passed all 51 after additional boundary checks.
Native Sigmond agreement remains unverified: its reference checks remain skipped
in this environment. No numerical kernels or generic deduplication were changed.

Final full suite: **1551 passed, 96 failed, 6 errors, 65 skipped**. JUnit comparison
against the phase 5 baseline found no changes to existing test outcomes; all
51 added tests passed. Changed Python sources parse and `git diff --check`
passes. Ruff remains unavailable in the active environment.

Phase 5 is complete. Phases 6–7 remain pending. Phase 6 audits arithmetic
metadata identity across scalar and N-D workflows.

## Phase 6: audited arithmetic identity (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`241e6a5cf066ddbc60a08cd1d04341b76efe7780bc75245bcfdc4a368c04e832`.
The method and failing/error/skipped IDs are in
[the phase 6 baseline record](observable_metadata_phase6_baseline.json).
Baseline: 1551 passed, 96 failed, 6 errors, 65 skipped. Unrelated changes were
retained.

### Changes

- Added public `DerivedObsInfo` and immutable `DerivedElement` in
  `arithmetic.py`. Versioned identities contain the operation, ordered operand
  identifiers, exact scalar arguments and deterministic numeric-array value
  fingerprints. They retain no live operand graph and never compare data
  buffers to establish observable identity. Labels render parenthesized
  expressions and remain outside identity, including in later operations.
- Scalar and N-D ufunc paths use the same per-cell policy. Every supported
  value-transforming ufunc derives metadata, including unary plus, scaling by
  one, addition of zero and equal-input arithmetic. `E` and `-E` coexist.
  Broadcast dimensions resolve the corresponding input cell rather than
  preserving an array parent's metadata or inventing undifferentiated
  `mixed_operation` placeholders.
- Unary/scalar results retain source sector context; multiple data operands
  retain only fields known and equal across all operands. Energy roles, level
  assignment and pivot-mode claims are cleared. Derived source sectors are
  immutable snapshots. Component projection remains an explicit domain method;
  complex dtype promotion is a representation-only no-op.
- Time-aligned arithmetic produces derived elements and `AtObsInfo` cells,
  retaining scalar time access, inclusive windows and reconstruction after SS
  HDF5 reads. Built-in generic placeholders obtain series context from their
  named array coordinates. Empty or mixed-parent arrays still permit arithmetic
  without claiming a common time source. Matching time offsets/extents survive;
  differing source offsets reject a common-axis query. Native correlator and
  effective-energy source identifiers retain their input distinctions without
  claiming that the result is the original source or rotation mode.
- Registered concrete codecs for derived scalar and at-time records, including
  operand presentation labels. Unknown versions, altered identities and
  contradictory generated storage names reject reads. Identity snapshots strip
  all supported presentation fields before identifying operands.
- Audited reference normalization, kinematics, bin resampling/statistics,
  model evaluations, fitting, range buffers and batch edits. Explicit reference
  normalization builds intended energy/reference metadata, including per-cell
  broadcast N-D output. Generic normalization of existing derived/at-time data
  gets an explicit reference derivation. Bin statistics on generated infos use
  new derivations rather than attempting to suffix immutable storage names.
  Model evaluation replaces metadata on a view so a function returning a
  parameter directly cannot rename the parameter. The range-buffer zero-span
  fallback now has its own descriptor rather than masquerading as subtraction.
- Updated README arithmetic guidance. Deliberately replaced the obsolete
  `mixed_operation` assertion in
  `test_to_collection_rebuilds_keys_after_arithmetic` with a derived-element
  assertion; the matrix domain wrapper still reconstructs native export keys
  and the numerical matrix remains unchanged. Generic deduplication and
  numerical kernels were not changed.

### Validation

43 new tests cover scalar/N-D identity, ordered operands and exact constants,
numeric-array snapshots, common-sector context, stable generated names,
cleared energy/pivot claims, time access and HDF5 reconstruction, labels outside
identity, frozen metadata, reference normalization, bin statistics, safe model
parameter returns, editor independence, generic numeric time arrays, transformed
pivot/effective series, conflicting offsets, version/identity contradictions,
empty/mixed-parent arithmetic, range-buffer fallback and broadcast reference
metadata.

The initial audit suite passed 364 tests. Arithmetic/matrix/plotting checks passed
96 tests before the last boundary additions. The expanded fitting/statistics/
bins/reference/time-series audit passed 228 tests with two pre-existing skips;
the final arithmetic/reference subset passed 51 tests before the final parent
metadata refinement. Changed Python sources parse and `git diff --check` passes.
Native Sigmond verification remains unavailable, with the existing reference
checks skipped. No claim of native agreement is made.

Final full suite: **1594 passed, 96 failed, 6 errors, 65 skipped**. JUnit
comparison against the phase 6 baseline found no changes to existing outcomes;
all 43 added tests passed. The one existing arithmetic assertion was deliberately
updated as described above. Ruff remains unavailable in the active environment.

Phase 6 is complete. Phase 7 remains: prediction wrapping and `model.series`.

## Phase 7: sampled prediction wrapping and model series (2026-10-13)

### Baseline

Revision: `5b7841572aa93523dd1a9fbb02a8abd1e625df23`.
Working-tree fingerprint:
`7ae766fdbc34619f8140ad1974d75a87106087b8ff5b63ac6d1ae5451e6b0f10`.
The method and failing/error/skipped IDs are in
[the phase 7 baseline record](observable_metadata_phase7_baseline.json).
Baseline: 1594 passed, 96 failed, 6 errors, 65 skipped. Unrelated changes were
retained.

### Changes

- Added frozen public `PredictionSeriesInfo`: a fixed series ID, numeric
  coordinate convention/dimension, optional explicit sector/target, and
  presentation labels. It stores no model, callable or live parameter/sample
  container. Default physical context is empty; parameter names never establish
  an energy role or sector. An explicit target can supply its known sector,
  while contradictory known sector/target fields reject construction. Prediction
  cells do not claim a fitted energy, level assignment or rotation mode.
- Added one internal `_wrap_predictions` helper taking existing scalar
  predictions, coordinates, dimension and the immutable descriptor. It creates
  `SigmondData` with `AtObsInfo` cells and preserves sample values exactly.
  Coordinates must be nonempty, unique, finite real numbers. Floating
  coordinates remain floating for selection/plotting; effective-energy
  calculation retains its integer-time requirement.
- `_prediction_descriptor` generates the ID once from coordinates, normalized
  prediction samples, sampling conventions and input/context identity fields.
  The buffer encoding fixes float64/complex128 dtype, little-endian byte order,
  C layout and NaN payload/sign encoding. Normalization is for fingerprinting
  only: original prediction buffers, including NaNs/infinities and complex
  values, are preserved. Later metadata names/equality/hash never read buffers.
- `SigmondModelFunc.series(x, dim="time", ...)` uses existing evaluation once per
  coordinate, then wraps it. Sampled independent variables retain uncertainty
  propagation and use full-sample values as display coordinates. Existing
  scalar/list-returning `__call__` is unchanged. Model label substitution now
  shares the literal-token helper and replaces both `{VAR}` and `{var}` while
  retaining ordinary LaTeX braces.
- `FitResult.theory_series()` wraps stored `theory`, `x` and `x_name` without
  reevaluating a model or accessing an optimizer/callable. Synthetic/linearized
  and reloaded predictions preserve their exact samples. Optional presentation
  and explicit physical context are accepted; no energy-specific fit accessor
  or parameter-to-level inference was added.
- Added prediction element codecs and separate presentation fields through
  `ObsMeta`. Eager/lazy SS HDF5 reads and collection reconstruction retain series
  IDs, coordinate convention, labels and explicit context. Unknown versions,
  changed IDs and coordinate/descriptor contradictions reject reads.
- Fixed fit-theory persistence ordering: HDF5 lexical key order placed index 10
  before index 2, misaligning predictions and stored coordinates. `write_theory`
  now records `PredictionOrder` alongside the existing fit-coordinate group
  annotations. `from_hdf5` validates it and restores the original row order;
  files without this optional annotation retain the prior loading behavior.
  Prediction values/observable codecs and existing coordinate attributes are
  otherwise unchanged.
- Updated README examples for both series entry points and prediction identity.
  Rich prediction annotations use SS HDF5; fstream/native legacy formats do not
  preserve all SS annotations, as documented for the earlier phases.

### Validation

36 new tests cover scalar/list/sampled-x evaluation, one evaluation per point,
fractional times, explicit dimensions/context/targets, distinct prediction sets
with identical parameter names, deterministic normalized IDs, labels outside
identity, immutable snapshots and hashing independent of buffers, synthetic and
NaN/infinite samples, complex predictions, eager/lazy HDF5 reconstruction,
callable-free reloaded fits, multi-digit prediction ordering, invalid coordinates
and dimensions, unsupported versions, identity contradictions and invalid saved
order annotations. Existing model evaluation results remain numerically equal.

The initial fitting/plotting/arithmetic suite passed 145 tests. The expanded
prediction/fitting/plotting suite passed **190 tests with one pre-existing skip**.
The first full run with 34 new tests passed 1628 tests and retained the baseline's
96 failures, 6 errors and 65 skips, with no existing outcome changes. Two further
complex persistence tests passed in the expanded audit. Changed Python sources
parse and `git diff --check` passes. Native Sigmond agreement remains unverified;
its existing reference checks remain unavailable/skipped in this environment.

Final full suite: **1630 passed, 96 failed, 6 errors, 65 skipped**. JUnit
comparison against the phase 7 baseline found no changes to existing outcomes;
all 36 added tests passed. Ruff remains unavailable in the active environment.

Phase 7 is complete. All seven implementation phases are complete. The native
Sigmond agreement gate remains unverified because the required bindings/fixtures
are unavailable; recorded skips and pre-existing fixture failures remain.
