# Consolidated key and info proposal

Status: proposed, 2026-10-16. API examples are sketches, not implemented APIs.
This specifies the recommended end state without Python API compatibility
constraints. It incorporates the Sigmond source comparison and the requirement
to preserve SS physical metadata, extensions, NumPy/xarray behavior and queries.
It supersedes alternative API shapes discussed in `observable_info_redesign_review.md`;
that document remains the source audit and architecture rationale.

## Decision

Use immutable native key values underneath immutable SS observable definitions.
An SS `ObservableKey` identifies an exact semantic definition, component and
evaluation point. One composed `ObservableInfo` adds presentation and user
annotations. Keep `SectorInfo` as a common, typed physical facet.

`SigmondData` owns samples and sample context, with aligned metadata for scalars
and arrays. Estimate references and storage addresses are separate keys.
Collections preserve occurrences. Domain indices are views over those records.

```text
NativeOperatorKey ──> OperatorKey ──> CorrelatorDef / VEVDef
                         │
                  sector + optional mode binding

CorrelatorDef / VEVDef / EnergyDef / EffectiveEnergyDef / ...
                         │
                 ObservableKey
             definition + part + point
                         │
                 ObservableInfo
             key + display + annotations
                         │
                   SigmondData
       samples + sample context + aligned infos
       + estimate refs + origins/layout/provenance
```

## 1. Keys have explicit purposes

| Value | Identifies | Equality/hash |
| --- | --- | --- |
| `NativeOperatorKey` | Exact BL/GI operator identity understood by Sigmond | Canonical native identity |
| `NativeObservableKey` | Exact scalar component record understood by Sigmond | Canonical native identity, including form flags, simplicity and component |
| `OperatorKey` | An SS-qualified operator, including sector enrichment and optional pivot binding | Exact normalized semantic fields |
| `ObservableKey` | What a value represents, including mathematical definition/component/point | Exact normalized semantic fields |
| `EstimateRef` | One particular typed sampled result from a producer/output | Persisted producer/output identity |
| `StorageAddress` | A native record in an artifact/group | Artifact/group/native key |
| `EnsembleKey` | Symbolic ensemble identity used for grouping | Explicit canonical ensemble identifier |

There is no universal key for all these purposes. Two fits can share an energy
key and have different estimate refs. Two SS-qualified pivot operators can
encode to the same native operator string. One estimate can be stored at more
than one address. Two observations can have the same native key and different
known SS physical metadata.

Keys are transitively immutable, hashable values. No key reads sample buffers,
calls a label renderer, or serializes a general info object to establish equality.
Python's `hash()` is only an in-process hash; persistent references and codec
records have separately specified stable encodings.

## 2. The native key layer follows Sigmond

Expose read-only native values through the small core, with validated factories
and replacement constructors. Do not expose mutators on objects usable as keys.
Reusing canonical parsing/encoding avoids a second independent Python parser.

The conceptual native observable union is:

```text
NativeObservableKey =
    NativeVEVKey(operator, part)
  | NativeCorrelatorKey(sink, source, time, form, part)
  | NativeNamedKey(name, index, estimator_kind, part)
  | NativeVacuumKey
```

`form` preserves the ordered/HermMat and raw/SubVEV distinctions. Native parts
are real or imaginary. Native named indices are file/observable-key indices,
not spectrum levels or array positions. Opaque keys preserve unsupported input
and diagnostics without pretending to validate known semantics.

The native layer is an encoding/validation authority and also a source of
observable contracts. It is not the owner of SS annotations or sample buffers.

## 3. SS definitions contain typed scientific meaning

`ObservableDefinition` is a small protocol implemented by frozen value records.
It has a registered `kind`, a `sector_info` facet, a category and an estimation
contract. These are value variants, not a deep inheritance hierarchy with shared
mutable constructor state.

The initial vocabulary is:

| Definition | Significant fields |
| --- | --- |
| `VEVDef` | Qualified operator |
| `CorrelatorDef` | Qualified sink/source, raw/connected definition, ordered/Hermitian construction convention |
| `EnergyDef` | Resolved sector, state, lab/CM frame, optional shift reference and normalization reference |
| `MomentumSquaredDef` | Sector/state, constituent kinematics, normalization convention |
| `NonInteractingEnergyDef` | Constituents with their momenta, frame and normalization |
| `EffectiveEnergyDef` | Source family key, algorithm kind, step, constant subtraction, required temporal convention |
| `OverlapDef` | Basis operator, pivot-qualified mode and normalization definition |
| `FitParameterDef` | Model namespace and parameter token when no physical target is declared |
| `PredictionDef` | Model/output definition and optional declared target |
| `DerivedDef` | Bounded operation description, semantic operands and arguments |
| `NamedDef` | Namespace/token, declared estimation contract, resolved sector and registered semantic fields |

`NamedDef` supports new observables without requiring a new compiled class.
Native named imports with unknown meaning receive a conservative namespace;
they do not acquire energy semantics from a parameter named `E` alone.
Registered extensions use immutable canonical values and versioned schemas.
Unknown extension payloads remain opaque and round-trip.

Scientific settings that define a finite-time quantity belong to its definition.
The sampled inputs, fitted window, producing run, optimizer details and data
fingerprints belong to estimate provenance. For example, two windows estimating
one declared energy retain the same `EnergyDef`; effective-energy prescriptions
with different steps have different definitions.

State is a separate discriminated value:

```text
EnergyState =
    SingleHadron(particle, momentum_specification)
  | AssignedLevel(assignment_scheme, index)
  | PivotMode(pivot_ref, index)
  | UnassignedState
```

The energy's sector is the authoritative shared physical classification. State
kinematics constrain it and are validated against it. Momentum vectors determine
PSQ where available. There is no unchecked second PSQ stored in mutable state.

A pivot mode is local to its pivot, not an assigned physical spectrum level.
Noninteracting reference constituents do not identify the measured state.
`qcmsq` maps to momentum squared rather than being an energy subtype.
Reference values describe the physical shift/normalization; the particular
sampled reference used for calculation appears in provenance.

## 4. ObservableKey adds component and evaluation point

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class ObservableKey:
    definition: ObservableDefinition
    part: Part = Part.VALUE
    point: Point = Point()
```

`Part.VALUE` is the complete value in the definition's declared codomain.
Complex observables also support real and imaginary component keys. Real
projection of an intrinsically real definition canonicalizes to its complete
value. A real-only imported component does not become a complete complex
measurement just because its numeric dtype is promoted.

`Point` is a canonical immutable mapping of semantic parameter names to typed
values, such as `time=8`. A family key has time unbound; its at-time key binds
time. A fully selected scalar has one evaluated key. Axis names and array
positions are not automatically semantic parameters.

```python
family = ObservableKey(CorrelatorDef(sink, source, connected=False))
point = family.at(time=8)
real_point = point.component("real")
```

There is no separate `AtObsInfo` hierarchy and no generated storage-name field
inside this key. Parameter normalization follows the variable contract: native
correlator time is integral, while sampled prediction coordinates can be real.
No display rounding enters identity. Layout dimension rename preserves the
semantic variable name and updates only its binding.

`key.category`, `key.estimator_kind`, `key.kind`, `key.sector_info`, and useful
physical facets are read-only projections of the definition, component and point.

## 5. Keep SectorInfo, with enrichment and one authoritative resolution

Retain the common frozen `SectorInfo` for charges, isospin, spin/parities,
PSQ and irrep. Keep exact equality, including unspecified fields, and validate
the supported fields consistently. Unknown is not a wildcard in equality.

Provide distinct operations:

| Operation | Meaning |
| --- | --- |
| `a == b` | Exact normalized stored values |
| `a.compatible_with(b)` | Specified fields do not conflict |
| `a.unify_known(b)` | Combine complementary knowledge; reject conflicts |
| `a.common(b)` | Retain fields both specify equally |
| `a.matches(query)` | Every specified query field is known and equal in `a` |

Resolve an owned sector by unifying native/definition constraints with explicitly
supplied physical information at construction. For instance, a key-known irrep
and PSQ can coexist with an SS-supplied baryon number. A conflicting irrep raises;
an extra known charge is retained. A physics enrichment produces a new semantic
key and never mutates keys already held by a dictionary or index. It does not
automatically rename the native record or allocate a storage address.

The shared facet does not require identical storage in every definition:

- `EnergyDef` and `NamedDef` own a resolved sector.
- `OperatorKey` owns its resolved enrichment over native constraints.
- `VEVDef` projects its operator's sector.
- `CorrelatorDef` retains both endpoint sectors; `sector_info` is their common
  summary. Enrichment of endpoints is explicit, not an ambiguous retag of the
  common summary.
- `EffectiveEnergyDef` projects its source's eligible sector.
- Generic `DerivedDef` does not inherit physical classification merely because
  operands have matching fields. Registered domain operations can establish it.

All expose the same convenient read interface. `.psq` and `.irrep` forward to
this authoritative facet. No independently assignable duplicate attributes exist.

## 6. Qualify operator identity without changing native encoding

```python
@dataclass(frozen=True, slots=True)
class OperatorKey:
    native: NativeOperatorKey
    sector_info: SectorInfo
    mode: ModeBinding | None = None

@dataclass(frozen=True, slots=True)
class ModeBinding:
    pivot_ref: PivotRef
    index: int
```

Factories resolve native sector constraints and validate an optional mode's
channel. A pivot ref identifies the actual pivot artifact/build, not just an
eight-character hash of settings and labels. Sink/source bindings are independent.

Two pivots with mode 2 remain distinct even if their GI text is the same.
Native serialization unwraps `native`; SS metadata retains the qualification.
Native export collisions are handled by groups or an explicit operator-name
mapping, not by dropping one semantic key.

This replaces separate mutable rotation attributes on correlator infos and
JSON snapshots containing copies of the same operator meaning.

## 7. One ObservableInfo record preserves rich attrs

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class ObservableInfo:
    key: ObservableKey
    display: DisplayInfo
    annotations: FrozenMetadata

    __hash__ = None

    @property
    def sector_info(self):
        return self.key.sector_info
```

The sketches omit constructor defaults and validation plumbing. Every reachable
field is immutable. `FrozenMetadata` owns recursively canonicalized values; a
proxy over caller-owned mutable storage is insufficient.

Keep three explicit roles for SS attributes:

| Role | Location | Effect |
| --- | --- | --- |
| Physical/definitional metadata | Typed definition/key | Participates in semantic equality and domain validation |
| Labels and display overrides | `DisplayInfo` | Changes presentation without changing semantic or estimate identity |
| Free user tags/notes | Namespaced `annotations` | Persisted and explicitly queryable, excluded from semantic key equality |

Energy roles, normalization references, effective prescriptions and mode bindings
are not loose user tags. They remain typed scientific metadata. New fields with
scientific meaning must declare a schema/definition projection; putting them
in annotations does not silently authorize numerical behavior.

There are no `EnergyObsInfo`, `SHEnergyObsInfo`, `CorrelatorObsInfo`, `AtObsInfo`,
`DerivedObsInfo` or `ArrayElementObsInfo` subclasses in the end state. Their
scientific content becomes the typed definitions/keys. Their labels and extra
annotations live in the same common info record. Native/physics kind tags need
not be the same as Python class names.

Queries can expose registered conveniences such as `kind`, `is_energy`, `hadron`,
`irrep` and `psq`. Type-specific structured access uses the actual definition.
There is no generic `__getattr__` delegation into arbitrary bags of attributes.

Info structural equality can compare the whole record for inspection, including
display/annotations. It is deliberately unhashable. Code choosing a dictionary
key must choose `info.key`, `data.estimate`, or a storage address explicitly.

## 8. Preserve Sigmond's two classifications

| Definition | Category | Estimator kind |
| --- | --- | --- |
| VEV | Primary | Simple |
| Raw correlator, including qualified/rotated operators | Primary | Simple |
| Connected correlator | Primary | Nonsimple |
| Named configuration-level observable | Secondary | Simple, explicitly declared |
| Fitted energy, effective energy, ratio of means | Secondary | Nonsimple |

Known definitions derive these properties. Named definitions declare their
estimation contract, defaulting safely to ensemble-level/nonsimple. Opaque
definitions may have unknown capabilities; operations needing those capabilities
must resolve them. These are not independently writable booleans.

The classifications survive component projection, point evaluation and
resampling. Storage as jackknife/bootstrap does not make a simple observable
nonsimple. A numeric vector is not evidence that an observable is simple.
Native bins require a valid simple definition and valid bin context.

Simple means configuration-level expectation semantics, not merely that an
operation is numerically easy. A nonlinear per-bin transform is a separately
defined statistic; it does not become the corresponding function of means.
Connected correlators use jointly resampled raw correlator/VEV estimates before
subtraction. Operator transformations that define new correlators can retain
primary/simple status through their explicit producer contract.

Rename the old `op_type` to the appropriate estimation contract. Retain `s`/`n`
only as native serialization spellings.

## 9. Ensemble, bins and sampling info are separate values

```text
EnsembleInfo
  ensemble_key
  geometry (Nx, Ny, Nz, Nt), streams, measurement metadata
  weighting/physical conventions, typed scientific extensions

BinsInfo
  ensemble_info
  canonical selection/omissions
  binning policy and applied weighting specification
  derived num_bins

SamplingPlan = JackknifePlan | BootstrapPlan
  jackknife count resolves from the bin context
  bootstrap records generator/convention, actual seed, skip and count

SampleContext
  ensemble/bin specifications
  representation and sample-axis convention
  realized alignment information
```

Full info equality and hashing use all canonical semantic fields. In particular,
geometry, streams and supported scientific metadata must not be dropped from
ensemble identity comparisons merely to fix the current hash contract.
Symbolic grouping uses `EnsembleKey`; associated descriptions must be validated
or explicitly refined. Unknown metadata is not silently treated as equal to
known metadata.

No unchecked `tweak_info`/`extra_params` controls computation. Keep/omit inputs
normalize relative to original measurements; omissions precede rebinning.
Already binned arrays record that state and are not rebinned implicitly again.
File-declared counts must agree with the normalized plan. Supported weighting
is explicit; unknown weighting cannot be interpreted as unweighted by default.

Sample compatibility is a separate predicate. Same-ensemble grouping, equal
descriptions and aligned draws answer different questions. Different metadata
does not establish zero covariance. Joint contexts explicitly identify ensemble
members and relationships rather than substituting an `indep` ensemble.

The Python records own the richer scientific description. Native adapters consume
validated supported fields; native ID-only equality is not the Python equality
contract. Native bootstrap generation remains the reference for compatible draws.

## 10. Data, scalars and arrays share this model

The concrete class shape, accessors and buffer/metadata ownership are specified
in [the SigmondData proposal](sigmond_data_redesign_proposal.md). `SigmondBins`
and `SigmondSampling` can remain convenience constructors for one shape-generic
concrete class; the examples below do not require a new subclass hierarchy.

```text
SigmondData
  samples/backend                 spatial_shape + (sample_length,)
  context                         shared SampleContext
  layout                          dimensions, index/auxiliary/scalar coords
  infos                           aligned field of ObservableInfo
  estimates                       aligned field of EstimateRef
  origins                         optional component StorageAddresses
  producers                       shared provenance/artifact references
  display                         container title/defaults, no physical identity
```

The scalar case has spatial shape `()`. Scalar `.info`, `.key` and `.estimate`
return single values. For arrays, `.infos`, `.keys` and `.estimates` are aligned
read-only fields. `.obs` supplies common facet projections across those fields.
There is no independent array-level physical info competing with its cells.

Private metadata fields can be constant, dense, or bound to coordinates. An
operator matrix and a time series need not eagerly allocate one key/info object
per cell. The dense fallback handles heterogeneous cells. This is a storage
optimization under one public semantic model, not another identity system.

Bindings explicitly associate a semantic parameter with an axis or auxiliary
coordinate. A time dimension can be named `t`; rename updates its binding and
preserves physical time. Layout-only trial axes repeat quantities legitimately.
Selecting a semantic coordinate retains the bound scalar value in the point.

For a single validated time-series family, `.ts.key` exposes its key with time
unbound. `.sel(time=8).key` evaluates that key at time 8. This replaces the
interim `.ts.observable` suggestion with one explicit key vocabulary. A family
request with unselected heterogeneous/non-time choices raises or uses the
corresponding field API; it never guesses a family from equal display labels.

Missingness is explicit. Flattening/exporting does not invent observed zero
cells. Flattening retains a layout manifest so arbitrary trial/coordinate axes
can be reconstructed without inserting them into scientific identity.

One dense array has a declared sample space. Mixed-context collections remain
valid; array assembly selects a homogeneous group or requires an explicit joint
context. This preserves multi-ensemble workflows without implicit sample pairing.

## 11. Mutability, estimate refs and operation rules

Public infos, keys, contexts and data handles are immutable. Samples/coordinates
have safe ownership and read-only public access. Library-owned immutable storage
can be shared; ordinary construction detaches caller-owned writable buffers.
Lazy materialization caches are internal state and do not change meaning.

`EstimateRef(producer_ref, output_selector)` identifies a typed result. Allocate
refs once at producer boundaries. Do not derive them by repeatedly hashing all
sample values. Copies, selection and layout changes preserve selected refs.
Display/annotation changes preserve them too. A calculation or explicit semantic
reinterpretation creates a new typed output linked to prior inputs, even when
it can share numeric memory. This preserves interpretation history.

An importer without a persisted artifact identity cannot guarantee stable result
identity across arbitrary file copies. Use an import-scoped artifact identity
and persist it on SS export; do not promise that a filesystem path is permanent.

`SigmondData` is unhashable. Its `==` uses object identity; explicit `.equals()`
compares datasets under documented numerical/metadata options. Semantic and
estimate comparisons use their keys directly.

| Operation | Required behavior |
| --- | --- |
| Slice, transpose, diagonal, axis rename | Preserve selected keys, refs and scalar semantic points |
| Stack/concatenate/broadcast | Preserve input keys/refs and validate context/alignment; repeated values are valid |
| Component projection | Construct canonical component keys; preserve component origin where known |
| Generic arithmetic | New derived definition and output refs; no automatic energy/sector/primary claim |
| Domain energy conversion/normalization | Compute and build the precise frame/reference definition together |
| Effective energy | Build prescription/source family key, bind output times, record sampled inputs; no fitted-level claim |
| Operator/pivot transformation | New qualified operators and correlator keys, with actual pivot/artifact provenance |
| Fit parameter | Use model-declared target if validated, otherwise generic parameter definition |
| Metadata enrichment | New validated key/interpretation, no automatic native rename |
| Display/annotation replacement | Same semantic key/ref, new immutable info |

One canonical producer policy handles corresponding scalar and array operations.
Pointwise derivations factor shared semantic evaluation coordinates where valid;
unequal source-time operations retain their explicit time relationships rather
than claiming an unsupported common time. Family binding and scalar key
construction must commute. Long histories use shared references rather than
recursive JSON snapshots in every cell.

Edits use replacement or a transaction. Callbacks see original values, all
dependent changes validate together, and a failure leaves the input untouched.
Freezing an editor detaches its writable state from the returned result.
No filtered view writes through into a parent collection.

## 12. Collections and indices

`ObservableCollection` is an ordered immutable sequence of scalar data handles.
It preserves different estimates of one quantity and repeated occurrences of
one estimate. Construction never deduplicates by info/key/buffer equality.

Registered projections supply shared physical facets and kind-specific fields.
Queries, sorting and grouping use an immutable metadata table and do not load
samples. Annotation queries have an explicit namespace. Unknown facet names
raise, missing/not-applicable values have documented handling, and query
semantics do not depend on the first row's type.

```python
levels.filter(kind="energy", psq=0, irrep="A1g")
levels.group_by("sector_info")
levels.filter(annotation__study="window_scan")
```

Dictionary export requires a selected key and a collision policy. Deduplication
and keyed merges are explicit. A unique lookup reports ambiguity rather than
choosing one estimate. Array pivots reject duplicate coordinate tuples unless
the layout includes the dimension that distinguishes them.

`CorrelatorIndex` is an immutable view for a chosen sample context. It indexes
SS definitions, qualified operators, points, components and native forms, with
links to observation rows. Native lookup is a separate index scoped to source
artifact/estimate choices. Sector enrichment can change an exact SS key while
native lookup continues to identify its stored record.

Two pivots with colliding native keys remain distinct semantically. Multiple
estimates of one semantic key remain multiple observations. `filter(mode_index=2)`
can select across pivots; a unique mode lookup requires pivot qualification or
otherwise reports ambiguity. Coverage and basis discovery remain metadata-only.

Matrices validate their basis/form against the same metadata their values own.
Exports use that authoritative metadata and preserve both endpoint bindings.
They never regenerate primary identity solely from axis names or coerce generic
derived values back into raw correlators implicitly.

## 13. Persistence and names

Native keys and numerical records retain Sigmond semantics. SS writes a versioned
consolidated metadata table containing definitions, sector enrichments, operator
qualifications, presentation, annotations, refs and layout/provenance records.
Shared definitions and artifacts are interned/referenced rather than repeated
in every component row.

On read: parse the native record, decode SS definitions/attrs, unify compatible
physical knowledge, validate the correspondence, then construct immutable infos.
Preserve unknown payloads; reject contradictory known fields. Re/Im fusion
requires compatible shared semantic/context information.

On write: select/allocate destination addresses, project supported definitions
to native keys, preflight collisions and context/classification constraints,
then write numeric records and SS metadata together. A typed secondary definition
can have a suggested name, but no universally required generated name.

Retain original addresses/spellings as origins. Destinations can reuse them for
unchanged exports or use an explicit mapping. Name allocation is not key equality.
Native-only export must explicitly acknowledge any unsupported metadata loss.

There is no general `info.name` or `info.index` that means several things.
Use display labels for presentation, native named-key fields for storage,
`AssignedLevel.index` for spectrum assignment, `ModeBinding.index` for a pivot
mode, and coordinates/positions for layout. Convenience labels are rendered from
definition plus overrides and never require `update_name()`.

## 14. Illustrative analysis

All constructors and methods below describe the target API. Inputs such as
`samples`, `ctx`, `index` and model configuration are supplied by the application.

```python
sector = SectorInfo(B=0, S=0, twoI=2, irrep="A1g", psq=0)
energy_key = ObservableKey(EnergyDef(
    sector_info=sector,
    state=AssignedLevel(assignment_scheme="spectrum-v1", index=0),
    frame="lab",
))
info = ObservableInfo(
    key=energy_key,
    display=DisplayInfo(latex=r"E_0"),
    annotations={"study": "window_scan"},
)
energy = SigmondSampling(samples, info=info, context=ctx)

assert energy.info.sector_info == sector
assert energy.info.psq == 0
pretty = energy.with_display(latex=r"E_{\mathrm{ground}}")
assert pretty.key == energy.key
assert pretty.estimate == energy.estimate

C = index.matrix(operators, times=range(3, 25))
pivot = SinglePivot.from_matrix(C, norm_time=3, metric_time=5, diagonalize_time=10)
series = pivot.rotate(C).diagonal(component="real").isel(operator=2)
family = series.ts.key
assert series.sel(time=8).key == family.at(time=8)

eff = series.ts.effective_energy(kind="forward", step=1)
assert eff.sel(time=8).key == eff.ts.key.at(time=8)
assert eff.ts.axis.offset == 0.5

rows = eff.to_collection()
rows.filter(kind="effective_energy", psq=0, mode_index=2)
```

Fit parameters can carry an explicitly declared `EnergyDef` target. Different
fit windows then produce different estimate refs with equal semantic keys.
Neither a parameter name nor an effective-energy point assigns a spectrum level
automatically. The exact names of conveniences can be adjusted without changing
the ownership and identity contracts.

## 15. Implementation boundary and gates

The native core owns canonical native values/codecs, exact sampling conventions,
native/legacy I/O primitives and selected Sigmond domain kernels. Python owns
SS definitions/info composition, enriched equality, arrays, queries, producers,
presentation and SS extension metadata. Native routines consume arrays and
validated typed settings. They do not own a competing public observable registry.

Implement one scalar/array/correlator vertical path before generalizing all
definitions. Required gates include:

1. Native classifier/encoding agreement and immutable hash stability.
2. Additional sector fields survive read/write; contradictions reject and native
   storage identity does not silently change on enrichment.
3. Two fit estimates, two pivot bindings and duplicate collection occurrences
   survive all supported conversions.
4. Scalar selection, array operations and collection export agree on keys/refs;
   both matrix export paths preserve bindings.
5. Binning/omissions/weighting/counts and bootstrap skip/seed behavior meet the
   declared native conventions, with unknown alignment handled explicitly.
6. Lazy queries never read samples; metadata field factorization does not change
   semantics or require allocating a full object grid.
7. Atomic editing, read-only buffer ownership and editor freeze isolation hold.
8. Codecs round-trip typed attrs, presentation, annotations and opaque extensions;
   transport schema changes do not change a canonical semantic definition.

This proposal changes documentation only. It does not implement the APIs or
establish new numerical parity results.
