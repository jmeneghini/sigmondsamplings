# Observable metadata: a design from the current code outward

The consolidated end-state recommendation is now in
[the key/info proposal](key_info_redesign_proposal.md). Use that document for
the proposed API and ownership contracts; this review retains the code survey,
Sigmond comparison, findings and the evolution of the recommendation.

Review date: 2026-10-16. This is a design review, not an implementation.
It examines the working tree, including its substantial uncommitted work.
The recommendations deliberately assume no Python API compatibility requirement.
Sigmond's observable and sampling semantics are the reference model, alongside
native file interoperability. Python ergonomics can change representation and
add capabilities; numerical or identity differences require explicit decisions.

The recommendation is to replace the mutable info inheritance hierarchy with
small immutable quantity values, a separate sample context, and one metadata
model for scalar and array data. Collections preserve observations rather than
deduplicating descriptions. Correlator indexing becomes a specialized view of
those observations. Typed primary/secondary and simple/nonsimple classification
remains part of the observable contract. Names and codecs cease to define
scientific identity by themselves.

## What is actually implemented

The seven phases in `observable_metadata_implementation.md` are recorded as
complete. The working code contains their immutable particles, sectors,
element snapshots, rotation bindings, effective-energy cells, arithmetic
descriptors, time-series views, and prediction series.

`observable_info_mutation_plan.md` and `quantity_redesign_plan.md` are proposals.
They describe different possible successors, not the current runtime model.
For example, `Particle` is already frozen and hashable, despite the quantity
proposal describing it as mutable. This review treats code as the authority.
It does not inherit the proposals' compatibility or mutability decisions.

### Type survey

| Current type | Actual responsibility | Recommended destination |
| --- | --- | --- |
| `ObservableInfo` | Free name/index, file encoding flags, ensemble, sector, label, equality/hash | One immutable scalar metadata view; named quantities cover the generic case |
| `EnergyObsInfo` | Energy meaning, state/particle roles, normalization, storage naming, labels | Composed energy quantity and state/reference values |
| `SHEnergyObsInfo` | Single-hadron constraints and alternate naming | Explicit single-hadron state inside an energy quantity |
| `_KeyedObsInfo` | Native text, lazy parsing, error state, key-derived facets | Native-key codec plus typed or opaque decoded value |
| `CorrelatorObsInfo`, `VEVObsInfo` | Operator meaning wrapped in mutable storage context | Correlator/VEV quantities; native keys remain transport values |
| `OverlapObsInfo` | Operator, pivot level, normalization and solve conventions | Overlap quantity plus pivot artifact reference and provenance |
| `ArrayObsInfo` | Container title/context and a callable naming placeholders | Container display metadata; shared context belongs to data |
| `ArrayElementObsInfo` | Parent pointer, positional index, eagerly generated name | Generic bound quantity or explicit cell quantity; no parent pointer |
| `AtObsInfo` | One element at one coordinate, generated storage name, duplicated context | A quantity evaluated at named semantic coordinates |
| `SeriesInfo` | JSON snapshot identifying a generic source series | A named quantity/family and ordinary coordinate binding |
| `CorrelatorSource` | Correlator definition, flags, component, rotation context | Composed correlator quantity and source estimate reference |
| `EffectiveEnergyInfo` | Finite-time definition and optional energy target | Effective-energy quantity; target is a separate explicit interpretation |
| `EnergyTarget` | Immutable snapshot of selected energy/ensemble fields | Energy quantity/state target; ensemble stays in data context |
| `DerivedElement`, `DerivedObsInfo` | Operation, operand fingerprints, source context, labels | Generic derived quantity plus estimate provenance |
| `PredictionSeriesInfo` | Prediction-set identity, axis, optional physics, labels | Prediction quantity plus producing run/output reference |
| `RotationBinding` | Pivot-qualified local mode | Retain the concept, with a durable pivot artifact identity |
| `SectorInfo`, `Particle` | Physical value objects | Retain and strengthen validation; reuse rather than duplicate |
| Operators, `Channel`, `CorrelatorInfo`, `VEVInfo` | Validated immutable domain values | Retain the useful values; separate native encoding from semantic composition |
| `CorrelatorAtTimeInfo` | Native correlator key, including stored-form flags | Native address key, mapped to a point quantity on import |
| `CorrelatorMatrixInfo` | Ordered basis plus requested assembly rules | Separate validated basis and assembly request |
| `EnsembleInfo` | Ensemble description mixed with binning/selection tweaks | Immutable ensemble description; binning/selection move to sample context |
| `SamplingInfo` | Statistical convention and generator settings | Immutable resampling plan, distinct from the realized sample alignment |

The numerical architecture is worth retaining: `SigmondData` already handles
scalar and N-D samples together, with the sample axis last, and analyzer
dispatch supplies bins/jackknife/bootstrap statistics. Array class names are
already aliases. The redesign need not add another array hierarchy.

## Sigmond's contracts as design constraints

Added after auditing the local Sigmond source checkout at revision `926e10e`.
This section refines the original recommendation: Sigmond is the semantic
reference, not only a file format to support. Sources below are in the sibling
`../sigmond/src/sigmond/cpp` tree. This was a source comparison; importing
Sigmond's native Python bindings was unavailable in this interpreter.

### Primary/secondary and simple/nonsimple are independent classifications

`observables/mcobs_info.h:11-36` defines both. Primary observables identify
operator VEVs and correlators; secondary observables include named user-defined
quantities and fit parameters. Simple observables have a single-configuration
integrand whose expectation can be estimated directly from bins. Nonsimple
observables require an ensemble-level construction.

| Definition | Category | Estimation contract |
| --- | --- | --- |
| Operator VEV | Primary | Simple |
| Unsubtracted correlator at time t | Primary | Simple |
| VEV-subtracted correlator at time t | Primary | Nonsimple |
| Named configuration-level observable | Secondary | Simple, explicitly declared |
| Fitted energy, effective energy, ratio of means | Secondary | Nonsimple |

The secondary/simple combination is operationally supported: named constructors
accept `simple=true`, and `data_handling/bins_handler.h:228-232` allows simple
keys without requiring primary classification. Do not infer behavior solely
from the header comment suggesting secondary quantities only use sampling files.
The default vacuum key is also a primary/simple special case; preserve its
defined behavior if implementing it rather than treating it as an operator VEV.

Primary does not mean original, unprocessed, or lacking provenance. A rotated
GI correlator is still primary. A connected correlator remains primary even
though it requires a nonlinear combination of ensemble estimates.

Use a typed observable definition that exposes `category`, `kind`, and
`estimator_kind`. Derive the primary category and simple status from known
definitions; for a generic named observable require a declared estimation
contract, with a safe nonsimple default. Opaque unknown definitions can carry
unknown capabilities. Avoid separately writable booleans that can contradict
the definition. The existing `op_type` letters `s`/`n` encode simple/nonsimple,
not primary/secondary or a physical operator type; replace the misleading name
in the Python API and retain its native encoding in the codec.

This classification is independent of bins/jackknife/bootstrap storage. A raw
correlator remains simple after resampling. Nonsimple does not become simple
because someone wraps a numeric vector in `SigmondBins`.

The contract must drive behavior. In Sigmond,
`analysis/mcobs_handler.cc:702` resamples a simple observable directly, while a
nonsimple connected correlator is constructed from jointly resampled raw
correlator and VEV estimates. Preserve that operation order. Native bins export
must reject nonsimple definitions; arrays expose per-cell classification, with
an optional uniform summary. `Component` and `At` preserve the appropriate
classification of their underlying definition. Generic arithmetic obtains a
new classification from a proven producer contract, not from its first operand.

In particular, distinguish a function of ensemble means from a mean of a
configuration-level function. Nonlinear functions applied to already aggregated
bin means also cannot silently claim the original configuration-level meaning.
An explicit per-bin operation is an extension with its own definition/context.

### Follow Sigmond's ensemble/binning/sampling separation

Sigmond already has separate `MCEnsembleInfo`, `MCBinsInfo`, and `MCSamplingInfo`.
Adopt that conceptual separation using immutable Python values:

```text
EnsembleInfo: canonical ensemble key, geometry, streams, measurement metadata,
              weighting/physical conventions and validated semantic extensions
BinsInfo:     ensemble description, canonical omission set, rebin policy
SamplingInfo: Jackknife plan or Bootstrap plan
SampleContext: these values plus actual sample alignment/representation
```

`MCEnsembleInfo::operator==` compares its ID string
(`data_handling/ensemble_info.h:119`). This is conditional on Sigmond's model:
known IDs resolve through an ensemble database, and explicitly constructed
unknown ensembles append measurement/geometry fields to that ID string.
`MCBinsInfo::operator==` additionally compares rebinning and omissions
(`data_handling/bins_info.cc:127`). Python's currently broader, freely supplied
metadata does not justify copying ID-only equality unchecked.

For the redesign, full `EnsembleInfo` value equality and hashing include all
canonical semantic fields, including geometry, streams and supported weighting
or physics metadata. Generate both operations from the same frozen fields.
Do not repair the existing contract by removing additional metadata from the
hash. Use an explicit ensemble key for symbolic same-ensemble grouping, and
validate descriptions/refinements associated with that key.

Replace `tweak_info` with typed `BinsInfo` fields. Parse `KeepFirst`, `KeepLast`,
and `Omissions` into a canonical omission set relative to original measurements;
apply omissions before rebinning. Derive `num_bins` from the normalized plan,
then validate any declared file count. Additional extension metadata must have
a schema and immutable canonical values. Free annotations belong outside this
identity. Do not skip unhashable semantic fields or lowercase arbitrary values
as a substitute for validation.

Do not use full-description inequality as evidence of statistical independence.
The existing covariance/grouping sites that compare ensemble objects need an
audit when equality changes. Different omission/binning metadata can refer to
overlapping measurements from the same ensemble; safe sample pairing requires
the sample-context relationship, not automatically zero covariance.

Jackknife count resolves from `BinsInfo`, as in Sigmond's
`MCSamplingInfo::getNumberOfReSamplings`; it should not be an independently
contradictory plan field. Bootstrap parameters require an explicit generator
contract, count, resolved seed and skip convention. Sigmond seed zero requests
a time-derived seed; NumPy `RandomState(0)` is deterministic. Resolve and record
the actual seed. Equal nominal settings across different generators do not
establish matching draws.

### Current departures: preserve deliberately or fix explicitly

| Current behavior | Assessment and design response |
| --- | --- |
| Complex arrays fuse Sigmond's real-valued component records | Useful ergonomic extension; preserve separate native component identities and matching context |
| Shape-generic arrays and metadata-only collection queries | Useful extension; preserve classifications per cell without another array hierarchy |
| Connected correlator assembly jointly resamples components before subtraction | Consistent with Sigmond's essential nonlinear operation order; preserve it |
| Raw and connected `CorrelatorObsInfo` both default `op_type` to `n` | Misleading internal flag; XML export derives native simplicity from the key, but Python should expose the correct derived contract |
| Generic named `SigmondBins` accepts default `op_type='n'` and writer emits that key | Semantic gap: Sigmond's bins writer rejects nonsimple keys; require a valid simple definition for native bins export |
| `tweak_info['omissions']` is retained as text but bin counts ignore it | Confirmed gap: 12 measurements, omissions 2/7/11, rebin 2 reports 6 bins; Sigmond's formula gives 4 |
| Already binned input can be rebinned again during `resample()` from the original header's factor | Confirmed gap: a 6-bin trace with 12 original measurements/rebin 2 is treated as 3 blocks and rejects a 6-sample jackknife plan |
| `num_bins` can disagree with its implied rebin plan | Confirmed gap: 10 measurements and requested 6 bins yields rebin 1, which implies 10 bins; validate normalized state |
| Bootstrap code stores `boot_skip` but never uses it | Confirmed gap: skip 0 and 127 produced identical samples; Sigmond explicitly advances the generator between resamples |
| NumPy generator/seed behavior differs from Sigmond | Native draw equivalence is unverified. Choose a tested Sigmond-compatible default, or identify the alternate generator explicitly; do not imply equivalence through identical headers |
| Weighted ensemble metadata is deliberately ignored by the current parser | An explicit support limitation, not proof of equivalent unweighted semantics; preserve/report the convention and prevent unsupported numerical interpretation |

For each numerical departure, record whether it is a bug, unsupported behavior,
or intentional extension, and add a Sigmond-reference check where applicable.
Source inspection can establish a missing parameter application or wrong count;
native execution is still required to establish full numerical agreement.

## Findings that determine the design

The focused run below passed **326 tests**. Separate small reproductions
confirmed the failures of invariants in this table. A passing existing suite
therefore does not establish that the present ownership model is sound.

| Finding | Evidence in current source | Consequence |
| --- | --- | --- |
| Equal ensembles can have different hashes | `info.py:451`, `info.py:461`: equality omits geometry/streams; hashing includes them | Equal objects can occupy two set entries |
| Mutable keys become unreachable | `ObservableInfo.__hash__` reads assignable identity fields | After inserting an info into a dictionary, changing its name made membership false |
| Different buffers disappear on construction | `observable_collection.py:343`; data equality in `data.py:622` ignores samples | Two samplings with identical metadata and different numbers became one collection item |
| Overlap meaning is absent from equality | `correlators/observables.py:313` inherits base equality/hash | Two overlaps with different operators, otherwise identical arguments, compared equal |
| Parent and cell metadata disagree | `data.py:421` replaces only the parent; `sampling_array.py:97` stores parent pointers and copied facts | An array retagged to another ensemble retained cells from the original ensemble; sectors also disagreed |
| Names drift from energy meaning | `energy_levels.py:993` and `update_name` | Changing PSQ retained the old stored name until explicit synchronization |
| Cache invalidation has holes | `observable_collection.py:585`, `:650` versus direct assignments | Cached PSQ remained 0 after the contained info was changed to PSQ 1 |
| Read-only mappings do not freeze their arrays | `data.py:116`, `:208`, `:213` | Exposed coordinate arrays are writable; infos can be replaced directly |
| Equivalent export paths lose different information | `correlators/matrix.py:1145` rebuilds infos with `_matrix_infos` | A rotated matrix cell had a rotation binding; `matrix.to_collection()` dropped it; `matrix.values.to_collection()` retained it |
| The index uses a narrower key than the observation | `correlators/index.py:106` | Two different pivot bindings with the same native key were rejected as duplicate semantic keys |

Additional structural weaknesses follow directly from inspection:

- `ObservableInfo.copy()` copies `__dict__` without validation. Nested ensemble,
  sampling and parent objects remain shared. A frozen outer dataclass containing
  these objects would still not be transitively immutable.
- Collection batch edits apply setters sequentially. Validation can fail after
  earlier fields or earlier items have changed. Callbacks can observe partially
  edited state. The array editor also shares coordinate arrays with its source.
- Collection `+`, `-`, `to_dict`, constructor deduplication, and the correlator
  index use different notions of identity. There is no safe universal equality
  rule that can make these all interchangeable.
- `observable_record` obtains identity by serializing `to_attrs()` and removing
  a blacklist of label fields. Adding a presentation field can inadvertently
  change derived identity. Representation versions and ensemble descriptions
  also leak into these identities.
- `_KeyedObsInfo`, `AtObsInfo`, and `DerivedObsInfo` delete base-class state
  because their sectors are derived. This is an inheritance mismatch, not a
  reason immutable value classes cannot work.
- `source_elements()` repeatedly reconstructs and checks series identity from
  every time cell. That work compensates for metadata that could have been
  represented once as a series definition bound to time.
- Sample compatibility for resampled data checks method/settings and length
  (`analysis.py:118`), which do not establish common draws or bin membership.
  Cross-ensemble arithmetic can fall back to `INDEP_ENSEMBLE`; independence is
  a relation between sample sources, not a replacement ensemble.
- `SinglePivot.identifier` hashes settings, labels, ensemble and optional source
  coefficients, but not the solved rotation or input estimate. Two builds with
  the same setup can therefore share a mode namespace despite different solves.
- Generic arithmetic retains common sector fields by agreement alone. Agreement
  is useful source context but is not a general physics rule for a ratio,
  product, or function of observables.

## The central distinction: three different identities

These are different questions and should have different keys:

| Question | Key | Example |
| --- | --- | --- |
| What quantity is estimated? | Immutable `Quantity` | Lab energy of an assigned state in a sector |
| Which particular estimate is this? | `EstimateRef(producer, output)` | Energy parameter from a particular fit/window |
| Where is a component stored? | Scoped `StorageAddress` | Artifact/group/native key/Re component |

Two fit windows can estimate the same quantity and produce different estimates.
One estimate can be copied to another address. A complex estimate can have two
component addresses. Two pivots can use the same GI text while defining different
operators. None of these cases should require a fake physical distinction, a
special equality override, or a storage name injected into the quantity.

Quantity equality compares exact, normalized semantic fields. It includes the
declared sector where that sector is part of the definition. Unknown fields are
not wildcards in equality. Labels, file names, sample values, ensemble context,
and fit windows are not quantity identity. Pivot-qualified states naturally
refer to an ensemble-specific artifact and are consequently local to that pivot.

Estimate references are allocated once at construction/producer boundaries and
persisted. Outputs identify a parameter, original cell, or component, not just a
run. Selection, reorder and copying preserve them. Computation, resampling, and
explicit reinterpretation create new outputs with input references. Display
changes do not. IDs are not recomputed by hashing sample buffers on metadata
access. Content fingerprints are optional integrity/reproducibility tools.

For a native file without stored artifact IDs, an importer cannot promise a
durable identity across arbitrary copies/renames without an external identity
record or a content fingerprint. Allocate an import identity, reuse it within
that import, and persist it on SS export. Do not disguise a path string as a
permanent scientific identity.

`SigmondData` and metadata envelopes should be unhashable. Remove metadata-based
data equality. Use ordinary object identity for `==` at the container level and
provide explicit `equals()` for comparing numerical datasets, with documented
metadata/NaN options. Quantity comparison and estimate comparison use their
respective value keys directly. Collections never use container equality to
decide whether an observation deserves to exist.

## Small immutable values, composed rather than inherited

Use frozen, slotted dataclasses for the domain values. Normalize aliases and
validate all dependent fields in one construction step. Store tuples and
recursively frozen mappings; do not put mutable infos, dicts, arrays, callables,
or live containers inside a semantic key. A read-only proxy over a caller-owned
dict is insufficient. Atomic `replace()` constructs and validates a new value.

Definitions also expose the Sigmond category and estimation contract described
above. Those contracts participate in definition validation and native key
conversion. A generic named definition declared simple is distinct from an
ensemble-level definition with the same label; those choices estimate different
things and Sigmond encodes their distinction in the key.

The existing constructor/property conflicts are reasons to remove the deep
hierarchy. They are not limitations of a fresh composition-based model.

The initial quantity vocabulary should cover actual producers:

| Quantity/value | Composition |
| --- | --- |
| `NamedQuantity` | Namespace, stable user key, canonical semantic attributes, optional sector |
| `Energy` | State description, frame, optional explicit shift and normalization references |
| `MomentumSquared` | State/channel, constituent kinematics and normalization; separate from energy |
| `NonInteractingEnergy` | Constituent particles and their momenta, energy convention |
| `Correlator` | Independently identified sink/source and connected/raw definition |
| `VEV` | Identified operator |
| `EffectiveEnergy` | Source quantity and finite-time prescription: kind, step, constant subtraction and required temporal context |
| `Overlap` | Basis operator, pivot-qualified mode and normalization definition |
| `FitParameter` | Model namespace and parameter key when there is no declared physical interpretation |
| `Prediction` | Declared model/output meaning and optional target; producing fit is provenance |
| `Derived` | Operation and semantic operands/arguments, without automatically claiming energy or sector |
| `At` | Quantity/family evaluated at named semantic coordinates |
| `Component` | Quantity and real/imaginary part |

Use a discriminated state value inside `Energy`: single hadron, explicitly
assigned spectrum level, pivot mode, or unassigned channel state. Keep
single-hadron identity, constituents of a noninteracting reference, assigned
level number and local pivot mode separate. A single constituent does not prove
a single-hadron measured state. A mode number does not prove energy ordering.

Represent existing `elab`, `ecm`, `delab`, `decm`, and `qcmsq` through explicit
frame/shift/normalization definitions. `qcmsq` is not an energy subtype. The
shift reference identifies what was subtracted; the normalization reference
identifies the scale. Which sampled reference was used belongs to provenance.
If unit conversion is performed, its convention must be explicit too.

Retain `SectorInfo` as a partial physical classification and the existing exact
operator channel as a more resolved value. Validate their projections, including
momentum/PSQ agreement. Do not force a partially specified energy sector to
pretend it is a fully resolved operator channel with a row and momentum vector.

Give sectors separate, named operations:

- Exact equality for keys.
- Directional `matches(query)`: specified query fields must be known and equal.
- `compatible_with(other)`: known fields do not conflict.
- `unify_known(other)`: merge complementary information, rejecting conflicts.
- `common(other)`: retain only shared known fields, for summaries.

Compatibility is not equality and is not transitive. `common` must not become a
universal arithmetic propagation rule. For correlators, preserve both endpoint
sectors even when a common summary is empty. Labels and queries can expose both.

Pure value modules should not import plotting, numerical fitting, storage, or
their codecs. Registration describes codecs and facet projections. Registration
must reject duplicate kind ownership. Unknown kinds have an opaque immutable
representation, with diagnostics, rather than claiming known physics.

## Sample context belongs to data

An observable definition does not own the Monte Carlo ensemble or sample plan.
`SigmondData.context` does. The context is immutable and contains references to:

1. The ensemble description and actual measurement/bin selection.
2. Binning/omission/reweighting conventions relevant to these samples.
3. The bins/jackknife/bootstrap axis convention and resampling plan.
4. The realized alignment domain: which indexed draws/bins correspond.

Split ensemble physics/geometry from selection and rebinning. Fix equality and
hashing by deriving both from the same canonical fields. A stable ensemble
reference can support grouping; exact description equality and explicit
refinement handle incomplete headers. Neither uses wildcard equality.

One dense array has one declared sample space. Collections can contain different
contexts, lengths and lazy records. Building an array from different contexts
requires an explicit joint context/coupling policy, or selecting a homogeneous
group. A joint context names its ensemble members and independence/correlation
blocks; it never replaces them with an `indep` sentinel. An empty collection has
no inferred context; a typed empty array can retain an explicitly supplied one.

Equal RNG seeds and sample counts alone do not prove alignment. Imports without
sufficient evidence carry an unknown alignment relationship. An explicit
user/producer assertion can establish it and should be recorded. This makes the
existing caution in `CorrelatorIndex` enforceable without inventing certainty.
Implementing a general multi-source uncertainty engine is a separate numerical
change; until supported, reject unsupported joint operations clearly.

## One scalar/array metadata model

Keep `SigmondData` shape-generic. Its conceptual state is:

```text
SigmondData
  samples/backend          S + (N,), read-only public access
  context                  sample interpretation and alignment
  layout                   named axes, index and auxiliary coordinates
  metadata                 aligned fields of quantities, estimates, display, origins
  display                  optional container title/default presentation
```

`ObservableInfo` can remain as one frozen scalar record returned by
`data.metadata.at(index)`: quantity, estimate reference, display and origins.
It has no subclasses and owns no parallel ensemble/sampling context. It is a
view of the canonical metadata, not a second state store. Structural envelope
comparison is for inspection; the envelope is not a hash key.

An internal `Field[T]` handles constant, dense and coordinate-bound metadata.
It is aligned to a subset of spatial axes, never to the Monte Carlo axis.
Selection, transpose and broadcasting operate on this field together with the
samples. A correlator matrix can represent operators on sink/source axes, a
shared definition, and time coordinates without allocating thousands of nearly
identical mutable Python objects. Heterogeneous cells use the dense fallback.

Use typed producer bindings for series/matrices. The binding explicitly maps a
semantic variable such as time to a coordinate; arbitrary dimension names are
not enough to establish physics. Bindings retain scalar selected coordinates.
Renaming a layout axis updates the binding without renaming the physical
variable. Unknown/literal strings are never implicitly coordinate references.
Start with a dense field and a small number of producer-specific factorizations;
do not build a general expression compiler before measurements justify it.

Make access predictable:

- `data.quantity` returns the selected scalar cell's quantity; it raises for
  N-D data, directing the caller to select a cell or inspect `quantities`.
- `data.quantities` is a read-only/lazy field over the full spatial shape,
  including time. Materializing an object grid is explicit.
- `series.ts.observable` exposes the time-independent family after validating
  the series binding. It does not infer a family from merely equal labels.
- `data.estimate` is scalar-only; `data.estimates` has the full spatial shape.
- A cell selected from an array has exactly the same quantity, context and
  estimate as that cell exported through the collection interface.

This avoids a global dimension-role system in which `.quantity` sometimes
means an entire function and sometimes a point depending on unselected axes.
The distinction between a family and an evaluated quantity is explicit in
`.ts.observable` versus `.quantity`. Producer bindings provide the needed
semantics without making users classify every axis as semantic/replicate/etc.

Remove `ArrayObsInfo`, placeholder parent pointers and `name_fn` from canonical
identity. A raw named array uses a namespaced family evaluated at its original
semantic coordinates. Layout-only axes may repeat quantities; estimate refs
still distinguish outputs. A title change never changes cells. A semantic
reinterpretation explicitly replaces the relevant quantity field and validates
the result; there is no ambiguous operation that retags only the parent.

Missing cells require a validity/coverage mask. A zero-filled buffer is not an
observation. Flattening and exporting omit absent cells unless the caller
explicitly requests a fill operation, which creates derived output estimates.

## Mutability and editing

Freeze all public metadata, including presentation and annotations. Labels can
remain easy to change through `with_display(latex=...)`, sharing samples,
quantity and estimate refs. Mutable labels are not necessary for notebook
ergonomics and make shared-view behavior harder to reason about.

Presentation overrides and user tags are separate from semantic facets. Tags
can be queried through an explicit namespace; being queryable does not make an
annotation scientific identity. Do not allow arbitrary attributes on infos.
Rendering receives quantity plus display overrides, with no storage naming
side effects.

The data handle is immutable too. Exposed samples, coordinates and metadata
storage are owned or safely shared and read-only. Lazy materialization caches
and memoized metadata are permitted internal mutations. This is essential:
freezing infos alone does not protect a cached collection/index if a contained
sampling can rebind its info or change its samples.

Constructors must detach caller-owned writable arrays by default. Merely marking
a view read-only does not stop an external owner changing its buffer. Internal
trusted construction can share library-owned frozen storage; an advanced borrowed
buffer API must state its ownership/lifetime contract explicitly. This applies
to coordinate and object-array storage as well as sample buffers.

Use an explicit editor for batch work. The editor owns writable buffers and
replacement metadata; `freeze()` validates the complete candidate, detaches
editor-owned memory and returns a new object. Subsequent editor changes must
not alter the returned result. Semantic/sample edits get new estimate refs;
display edits retain refs. Evaluate replacement callbacks against the original
row, not against successively edited fields. An invalid row aborts the whole
transaction without modifying the input.

For scoped collection edits, use `new = collection.replace_where(query, transform)`.
The parent and previously filtered views remain valid snapshots. CLI edit
programs can use a local transaction and return the final collection. Retaining
write-through `filter(...).obs.update(...)` would require a shared mutable store
and revisions; it is unnecessary for this library's analysis objects.

## Collections and the correlator index

`ObservableCollection` should be an ordered immutable sequence of scalar data
handles, preserving every occurrence. It is not a set or a dictionary keyed by
infos, quantities, or even estimates. Repeating the same view is valid too.

Concatenation appends. Grouping by quantity returns groups. An explicit
`deduplicate(by=...)` states the intended policy; key-based merges require an
explicit conflict rule. Dictionary export requires a caller-chosen key and
rejects duplicates rather than silently overwriting. `find_one` reports
ambiguity. Difference/drop use row selection or an explicit key policy.

A cached metadata table supports `filter`, `sort`, `group_by`, `unique`, facets
and DataFrame export without loading samples. Build it from registered
projections, not `vars()`, `dir()`, or the first item's type. Unknown values and
not-applicable fields have documented handling; unknown facet names raise.
Numeric metrics belong to an explicit values namespace and may materialize data.
Immutable rows and membership make metadata caches valid for the snapshot.

Array flattening retains an optional layout manifest with original positions,
axis definitions and coordinate labels. That makes array → collection → array
lossless even for layout-only trial axes with repeated quantities. Without a
manifest, `to_array(shape_by=...)` is an explicit pivot: duplicate coordinate
tuples raise, and missing cells retain missingness. Merging unrelated manifests
requires selecting one or supplying the new layout; it cannot guess a trial axis.

Energy collection APIs become validated domain views/accessors over this common
collection, sharing its queries and storage. Do not maintain another identity
or mutation model for spectra. Preserve useful domain operations and replace
the inheritance boilerplate.

`CorrelatorIndex` remains useful. It should be a specialized immutable index over
a collection and a selected sample context, with mappings to observation rows.
It is not the authority that owns or regenerates observation meaning.

Its selectors distinguish:

- Typed sink/source identities, including independent pivot bindings.
- Time and component availability.
- Native stored key form and raw/connected conventions.
- Multiple estimates or source artifacts for otherwise matching quantities.

Two estimates for one physical correlator are an ambiguity to resolve, not
automatically duplicate corruption. Different pivots are different bound
operators even if their GI encodings match. Native-key lookup is scoped to a
selected artifact/estimate set and reports remaining ambiguity. Preserve native
`HermMat` form as an encoding/convention; the flag alone is not a numerical
Hermiticity check. Preserve the existing explicit assembly/completion rules.

Coverage, available bases, operators and sorted series times remain metadata-only.
Only data requests read selected samples. Cache the index on the immutable
collection; no defensive info copies, mutation counters or refresh-after-edit
protocol is needed. New collections have their own index cache. Keep native-key
and semantic lookup APIs explicitly named rather than overloading `keys()`.

`CorrelatorMatrix` remains a domain wrapper for assembly, transforms, VEVs,
coverage and diagnostics. The values' bound quantity metadata is authoritative;
the basis/request is validated against it. Export uses that same metadata and
cannot recreate a correlator merely because axes are named sink/source/time.
Generic arithmetic that changes meaning cannot be written back as raw
correlators implicitly. An intentional reinterpretation/export mapping must
be explicit.

## Operations define metadata, not equality heuristics

| Operation | Meaning and result identity |
| --- | --- |
| Slice, transpose, rename, diagonal selection | Select/remap existing quantities and estimate refs; preserve selected scalar semantic coordinates |
| Stack/concatenate | Preserve observations and refs, including repeated quantities; declare layout and validate sample context |
| Real/imaginary projection | `Component(quantity, part)` and an explicit output/component ref; select the corresponding origin address where available |
| Numeric dtype promotion | Change representation; it does not recover an unmeasured imaginary component |
| Generic arithmetic, clipping, unwrap, contraction | New derived meaning and output refs unless a registered producer supplies a more precise contract |
| Explicit energy frame/shift/normalization helper | Compute samples and construct the new physical definition together |
| Matrix basis transformation | New identified operators/correlators, with transformation/input refs in provenance |
| Effective energy | New finite-time quantity, source estimate refs and algorithm settings; no automatic claim of a fitted eigenenergy |
| Fit | Generic parameters unless the model/user explicitly declares a physical target; window/run in provenance |
| Prediction | New sampled output of the declared model; fits with different parameters produce different estimates of the same prediction quantity |
| Semantic assignment/correction | Explicit reinterpretation with a new typed result ref, sharing sample memory and linking the prior interpretation |
| Label/tag change | Presentation replacement only; preserve quantities and estimate refs |

For example, `C / C` must not remain a correlator, and `E + E` must not remain
the original level energy. Even scalar rescaling should default to a derivation
unless a domain operation explicitly declares the normalization semantics.
Shared sectors alone do not justify retaining a physical type.

Operation metadata should be produced once per operation/series, then bound
across cells. Keep shallow descriptions and shared provenance references instead
of expanding nested history into every name, cell or label. Provenance stores
estimate/artifact refs, operation and settings, not strong references to all
input data arrays. Missing referenced artifacts are explicitly unresolved.

Keep source-time coordinates distinct from display offsets. Offsets derived
from an effective-energy convention are not separately writable. Extent does
not imply periodic boundary conditions. Preserve direction, component and
boundary conventions that actually affect calculation.

Model parameter names such as `E` are not declarations of energy semantics.
An optional model-owned target mapping can produce an `Energy` quantity after
validation; otherwise use `FitParameter`. Assigned spectrum level and pivot
mode remain different state descriptions. These are metadata contracts around
the existing fit driver, not a reason to rewrite the optimizer or kernels.

## Storage and serialization

Codecs import domain values. Domain values do not call codecs to define equality.
Keep a distinction between transport schema versions and semantic definition
versions. A transport revision should not change a quantity that decodes to the
same canonical value. Persistent IDs use a specified canonical encoding, never
Python's process-dependent `hash()`.

A storage address includes artifact/group scope, typed native key and component.
Flat Sigmond keys retain name, index and op_type; correlator keys retain their
native form flags. Original spelling can be preserved in a source locator.
Generated names are suggestions allocated at export, not required properties
of every quantity. A quantity with incomplete known physics can still exist.

Keep loaded origins for traceability and optional copying. A derived output has
no current output address merely because an input was stored somewhere. Labels,
renaming a destination, and copying files do not redefine physical meaning.

Continue using a consolidated metadata table: its avoidance of per-dataset
attribute reads is a good existing design. Intern shared quantity definitions,
sample contexts, producers and pivot artifacts; rows reference them. Re/Im
components must agree on shared identity/context before fusion.

Decode native keys into typed quantities where supported. Preserve unknown
keys/SS payloads opaquely with diagnostics; do not pretend to understand their
physics. Conflicting known metadata is an error. Codecs declare facet/label
projections explicitly, so display fields cannot enter identity by accident.

Before writing, allocate all destination addresses and preflight collisions.
Repeated quantities/estimates can need different addresses or an explicit
export deduplication decision. Two pivots with colliding native GI encodings
require separate groups or an explicit operator-name mapping. SS annotations
alone cannot make identical native keys coexist in one native namespace.
Native-only exports that cannot retain semantics need an explicit lossy mode.
Do not silently omit conflicting rows or invent physical labels to fit a key.

## What to keep, what to remove

Keep shape-generic data/analyzer dispatch, named exact alignment, immutable
operators/particles/sectors, explicit rotation bindings, metadata-only discovery,
lazy sample reads, coverage checks, numerical correlator/GEVP/effective kernels,
and consolidated HDF5 metadata.

Remove the mutable info subclass hierarchy, stored/canonical-name synchronization,
parent-pointer placeholders, serialization-derived semantic equality, mutable
hash keys, automatic collection deduplication, first-item query introspection,
implicit write-through view edits and reconstruction of scientific identity
at export. `refresh_correlators()` and defensive metadata deep copies become
unnecessary only after the containing data handles are immutable too.

Compared with `observable_info_mutation_plan.md`, freezing only identity while
retaining mutable presentation and assignable data handles is an intermediate
repair, not the desired end state. It leaves ownership and identity conflated.
Its claimed cache simplification does not hold if samplings can rebind their
metadata after an index/table is built.

Compared with `quantity_redesign_plan.md`, retain the separation of meaning,
estimates, storage and provenance. Drop compatibility envelopes as a migration
goal, mandatory global axis-role classifications, and estimate-keyed collection
uniqueness. Keep scalar quantity access simple, use explicit family access,
and let a collection preserve repeated occurrences of the same estimate.

## Implementation language and the native boundary

For a complete rebuild, prefer a Python-owned public/data model over a small,
purpose-built native Sigmond core. Reuse scientific implementations and native
encodings; design bindings around bulk arrays and value records. This is closer
to rebuilding the bindings than to maintaining a complete independent Python
implementation of Sigmond, but the Python layer remains substantial.

The current binding module, `sigmond/cpp/pybind/pysigmond.cc`, exposes handlers,
mutable info classes and tasks. Its `RVector.array()` goes through a standard
vector conversion; it has no explicit NumPy array/buffer binding or GIL-release
interface. `MCObsHandler` owns per-observable maps, lazy reads, sampling-mode
state and caches; pivots retain handler references. The bindings already contain
necessary lifetime policies. These are coherent implementation choices for the
existing engine, but a small Python wrapper alone cannot turn that ownership
model into independent NumPy/xarray datasets.

The appropriate division is:

| Layer | Owns |
| --- | --- |
| Python data and metadata | `SigmondData`, immutable enriched definitions/context, layout, coordinate binding, estimate/provenance refs and domain result objects |
| Python array and collection API | NumPy protocols, xarray alignment/selection, metadata queries, lazy request planning, editing, plotting, CLI and custom model composition |
| Native value/encoding core | Canonical operator/native observable parsing, validation, primary/simple conventions and exact native key serialization |
| Native numerical core | Exact bootstrap draw generation and selected Sigmond-specific effective-energy/pivot algorithms and diagnostics |
| Native I/O adapters | Native/legacy formats and batched record access where the native implementation is valuable; Python retains SS metadata/layout transport |
| Backend integrations | Optional full Sigmond session/task interoperability and native fitting; Python fitting remains available for arbitrary user models |

For shared values, give one representation authority. A primary definition can
contain a native immutable operator/key value, with additional mode/physical
metadata composed in Python. Do not maintain two independent parsers or two
mutable copies of its native identity. Native `MCEnsembleInfo` ID equality is
not automatically the equality of Python's richer `EnsembleInfo`; define the
adapter and the intentional stronger Python validation explicitly.

The public sample store is NumPy-compatible memory. Native routines consume
blocks and return arrays/results. They do not require uploading every Python
operation into an `MCObsHandler` registry. Expose read-only native values or
replacement factories rather than setters on objects used as Python keys.

Illustrative private interfaces, not implemented functions:

```python
indices = _native.bootstrap_indices(n_bins, plan)
values = reader.read_many(keys, representation=representation)
energy, failed = _native.effective_energy(values, times, settings)
solution = _native.solve_pivot(matrix_values, settings)
```

A Python user still writes `data / reference`, `.sel(...)`, `.ts.effective_energy()`
and collection filters. Array arithmetic executes through NumPy; labeled alignment
and result metadata are handled in Python. A native call is appropriate for an
entire selected numerical operation, not one binding crossing per sample/cell.
NumPy/SciPy already run their expensive array/linear-algebra work in compiled
code; introducing bindings does not itself establish a performance advantage.

Specify dtype, shape/sample-axis placement, strides, output ownership and failure
diagnostics for every bulk interface. Share buffers where ownership is safe;
allow an explicit bounded conversion/copy when required. Release the GIL during
native work only after validation and only where the code is safe for concurrent
use. Never return an array view invalidated by clearing a handler's cache.

This needs some C++ refactoring as well as binding changes. Extract numerical
operations from XML/task/handler orchestration where feasible, preserving their
algorithms and reference tests. A pivot whose implementation is deeply coupled
to handler state can initially use a contained session adapter; its returned
arrays/artifact state must still obey the Python ownership model.

Separate the minimal compiled core from the full task/plotting/fitting build.
The existing CMake extension links all six component libraries, and real/complex
and precision choices are global compile-time settings. New runtime dtype
contracts and packaging cannot be obtained by changing pybind signatures alone.
Ship supported wheels for the small core; ordinary analysis should not require
a separately configured batch executable, ensemble database path or plotting
stack. Keep a full-session integration as an explicit backend.

Prioritize native reuse where independent implementations have already drifted:
bootstrap skip/seed/draw conventions, native encodings and file handling, and
algorithm-specific numerical decisions. Simple statistics and general array
operations can stay NumPy implementations verified against the semantic contract.
Do not implement interchangeable backends for every function speculatively.

Before committing to broad extraction, prove a small vertical prototype: native
key decoding and batched reads, exact bootstrap indices, ordinary NumPy/xarray
selection/arithmetic, a native effective-energy or pivot operation, then
collection and file round-trip. Measure wall time, conversion/copy cost and peak
memory separately from scientific parity. The existing oracle harness remains
useful for tests; production should not need temporary files and subprocesses
to cross the Python/native boundary.

## How the proposed API feels

These are target API sketches, not runnable examples for the current package.
`loader`, `ops`, and `ctx` are supplied inputs; `fit1` and `fit2` are fits with
different windows and an explicitly declared common energy target.

```python
mass = SigmondSampling(
    samples,
    quantity=Energy(state=SingleHadron("pi", psq=0), frame="lab"),
    context=ctx,
)
pretty = mass.with_display(latex=r"m_\pi")
assert pretty.quantity == mass.quantity
assert pretty.estimate == mass.estimate

C = loader.correlators.matrix(ops, times=range(3, 25))
pivot = SinglePivot.from_matrix(C, norm_time=3, metric_time=5, diagonalize_time=10)
D = pivot.rotate(C).diagonal(component="real")
c2 = D.isel(operator=2)
c2.ts.observable                 # real-component correlator family, pivot-qualified
c2.sel(time=8).quantity           # that quantity evaluated at source time 8
c2.sel(time=8).estimate           # the corresponding sampled output

eff = c2.ts.effective_energy(kind="forward", step=1)
eff.ts.observable                # finite-time effective-energy definition
eff.ts.axis.offset               # derived display offset, 0.5
eff.sel(time=8).quantity          # effective-energy point at source time 8

assert fit1.param("E").quantity == fit2.param("E").quantity
assert fit1.param("E").estimate != fit2.param("E").estimate
trials = ObservableCollection([fit1.param("E"), fit2.param("E")])
assert len(trials) == 2
trials.group_by_quantity()       # one quantity, both estimates

changed = trials.replace_where(
    {"irrep": "A1g"},
    lambda data: data.with_display(latex=r"E_0"),
)
# trials, changed, and earlier filtered views remain independent snapshots.
```

Ordinary users work with samples, quantities, selectors, domain methods and
labels. Producers create estimate refs and bindings automatically. Storage
allocation, field implementations, codec registration and provenance resolvers
remain extension APIs rather than constructor requirements for every notebook.

## Implementation order and acceptance criteria

1. **Establish values and ownership.** Implement immutable quantity/context/ref
   values, frozen display, codecs and explicit comparisons. Map every current
   info field to a destination. Establish read-only data/coordinate ownership
   and editor freeze isolation. No compatibility hierarchy is required.
2. **Establish scalar/array/collection behavior together.** Implement the common
   metadata field with a dense fallback, scalar selection, duplicate-preserving
   collections, facet projections, transactional edits and layout manifests.
   Prove conversions preserve quantities, refs, context and missingness.
3. **Complete the correlator path.** Import/index → matrix → pivot → diagonal →
   component → effective energy → collection → SS file → lazy reload. Use a
   durable pivot artifact ref. Export through wrapper and values must agree.
   Add producer-specific factorization only where it simplifies this path.
4. **Complete energies, fits and predictions.** Physical state/reference
   composition, explicit domain transformations, optional model targets,
   repeated fit-window estimates, prediction provenance and spectra views.
5. **Measure and simplify.** Benchmark large matrix discovery/construction,
   selection, metadata queries, serialization size and peak memory. Remove
   obsolete snapshots/caches and eager metadata allocation after the replacement
   meets correctness and performance gates. Update exports/stubs/docs together.

Acceptance tests should establish laws, not merely mirror constructors:

- Equal keys have equal hashes, and reachable state cannot change either.
- Display changes leave quantities/estimate refs unchanged; semantic changes
  cannot overwrite a result in an existing index.
- Scalar operations and array cell operations yield the same quantities and
  corresponding output refs under the declared producer mapping.
- Selection/reorder/copy preserve original refs and selected semantic coordinates.
- Two windows for one energy, two pivots with mode 2, different normalization
  references, and same-name different buffers all survive collection conversion.
- Matrix export preserves both endpoint rotation bindings; arbitrary derived
  values cannot regain correlator identity through their coordinates.
- Failed multi-field/multi-row edits leave inputs untouched. Mutating an editor
  after freeze cannot alter a frozen result.
- No metadata query/index operation reads lazy samples. Cache construction does
  not require a full quantity object grid or content hash of sample buffers.
- Unknown alignment is not treated as verified coupling; mixed ensembles do not
  disappear into a placeholder ensemble. Supported joint context is explicit.
- Missing cells remain absent; scalar and auxiliary coordinates survive selection
  and array/collection/file reconstruction where the format supports them.
- Primary/secondary and simple/nonsimple classifications agree with Sigmond for
  VEVs, raw/connected correlators and named simple/nonsimple definitions. Wrappers
  and array selection preserve them. Nonsimple definitions cannot masquerade as
  native bins.
- Omissions and keep ranges normalize before rebinning, bin counts agree with
  the plan, and resampling already binned data does not reapply source blocking.
  Native-compatible bootstrap tests cover skip and resolved seed behavior.
- Codec/display changes cannot silently alter semantic identity. Unknown payloads
  remain inspectable; known contradictions and export collisions are diagnosed.
- Existing numerical oracle checks continue to establish numerical equivalence
  for kernels, resampling, fit covariance and pivot conventions.

### Validation performed for this review

```sh
python -m pytest -q \
  tests/test_observable_sectors.py tests/test_observable_elements.py \
  tests/test_arithmetic_metadata.py tests/test_sampling_array.py \
  tests/test_time_series.py tests/test_prediction_series.py \
  tests/test_correlator_index.py tests/test_collection_mutation.py \
  --disable-warnings --tb=short
```

Result: **326 passed in 3.77s**. Additional in-memory reproductions confirmed all
ten findings in the main findings table. The full suite and native Sigmond
oracles were not run for this design review; historic suite counts in existing
implementation logs are not a current full-suite result. No runtime source or
existing design document was changed by this review.

The follow-up Sigmond source audit additionally reproduced ignored omissions,
reapplied blocking, contradictory bin-count inputs and an unused `boot_skip`,
using in-memory Python examples. Native Sigmond Python execution was unavailable
(`ModuleNotFoundError: No module named 'sigmond'`); native behavior cited in the
comparison comes from the inspected C++ implementations, not a runtime parity
claim. The follow-up changes only this design review.
