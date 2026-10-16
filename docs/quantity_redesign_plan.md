# Quantities, observable addresses, and labeled data

Status: proposed design and implementation plan. No API in this document is
implemented by this change. Examples labeled **target API** describe the intended
interface; existing entry points remain supported during migration.

## Read this first

The redesign separates five responsibilities:

| Responsibility | Object | Answers |
| --- | --- | --- |
| Meaning | Immutable quantity key | What does this value represent? |
| Estimate identity | Stable estimate reference | Which particular computed or loaded estimate is this? |
| Storage | Immutable observable address | Which stored dataset/component identifies it? |
| Layout | Coordinates and a quantity template | What does each array cell represent? |
| History | Provenance record | How was this result produced? |

`SigmondData` continues to own samples, statistics, ensemble and sampling
context. It supports scalars and arrays. Correlator matrices and pivots continue
to provide domain operations; this redesign changes their metadata plumbing,
not their numerical algorithms.

The principal benefit is reliable meaning through a complete analysis:
correlator → pivot mode → effective energy → fit parameter → spectrum.
Selection preserves identity, transformations create the appropriate new
identity, and file conversion preserves addresses without guessing from labels.

Recommended implementation order: establish value keys and codecs, complete one
pivot-diagonal workflow, then generalize. Do not replace all observable classes
before proving that workflow.

## Everyday API and extension API

Everyday analysis uses `SigmondData`, coordinates, `.quantity`/`.quantities`,
`.estimate`/`.estimates`, `.ensemble_info`, `.ts`, filters, and domain operations.
`.address` and `.provenance` are inspectable when needed; ordinary analysis does
not require constructing them. Producers create templates, bindings and references.

The extension API contains `QuantityTemplate`, `CoordRef`, quantity registration,
codecs, address allocation and producer metadata contracts. These belong in
producer/extension documentation, not the first session example. Advanced users
may construct physical quantity keys for model targets or state assignments.

The end state makes data metadata first-class. `ObservableInfo` and its subclasses
are compatibility adapters for old constructors, imports, filters and files;
new domain code must not use their equality to decide semantic or estimate
identity. Deprecating adapters is a release decision, not a prerequisite for the
new API. Their continued existence must not force two competing internal models.

## A session using the target API

The following is **target API**, not runnable against the current package.
`loader`, `ops`, and `fit_options` are supplied by the application; `single_exp`
is a proposed standard fit-time model that declares its own energy semantics.

```python
from sigmondsamplings import SigmondData, SinglePivot
from sigmondsamplings.fitting import single_exp

C = loader.correlators.matrix(ops, times=range(3, 25))
pivot = SinglePivot.from_matrix(
    C, norm_time=3, metric_time=5, diagonalize_time=10,
)
D = pivot.rotate(C).diagonal(component="real")
# Layout: (time, operator); mode bindings are auxiliary operator metadata.
D.dim_roles                       # {"time": "parameter", "operator": "semantic"}
D.quantities                      # DataArray of component-correlator keys over operator

c2 = D.isel(operator=2)
eff = c2.ts.effective_energy(kind="forward", step=1)
eff.time.offset                   # 0.5: display position relative to source t
eff.quantity                      # EffectiveEnergy(of=Component(Correlator(...), "re"), ...)
eff.quantity.estimates            # Energy(channel=..., state=PivotModeKey(..., 2))
eff.sel(time=8).quantity           # TimePoint(EffectiveEnergy(...), time=8)
eff.ts.at(8).quantity              # exactly the same key as sel(time=8)

fit = c2.ts.fit(single_exp, window=(6, 18), **fit_options)
E2 = fit.param("E")               # SigmondSampling, not a bare quantity
E2.quantity                      # Energy(channel=..., state=PivotModeKey(..., 2))
E2.estimate                      # EstimateRef(run_id, output="parameter/E")
E2.provenance.inputs              # input estimate references, resolvable when retained
fit.param("A").quantity          # generic FitParameter unless declared
E2.ensemble_info                  # first-class ensemble context
eff.ts.plot(overlay=fit)
```

`window=(6, 18)` includes both endpoints in source-time coordinates. `.ts.fit`
delegates to the existing fit driver and still requires its parameter/minimizer
options. `c2.ts.window(6, 18).ts.fit(...)` is equivalent; the convenience argument
records the same selected input/window in provenance.

Fit `single_exp` to the correlator `c2`, as above. Fitting an effective-energy
series requires a model for effective energies (for example a declared plateau
model), not the correlator's exponential model. Typed model bindings must
validate the input quantity, including supported component wrappers.

A mode index is local to one pivot. Assigning a physical spectrum level is a
separate, explicit step:

```python
assigned = E2.assign_level(2, scheme="A1g-spectrum-v1")
assigned.quantity                # Energy(..., state=LevelKey(..., level=2))

spectrum = SigmondData.stack(
    assigned_energies, dim_name="level", coords=level_keys, role="semantic",
)
a1g = spectrum.to_collection().filter(irrep="A1g", level_index=2)
```

For a fit-window study, stack along a replicate axis. Duplicate quantities are
valid and never deduplicated by quantity equality:

```python
trials = SigmondData.stack(
    trial_energies, dim_name="fit", coords=fit_ids, role="replicate",
)
trials.quantities                # repeated Energy keys over fit
trials.estimates                 # distinct EstimateRefs over fit
first = trials.sel(fit=fit_ids[0])
second = trials.sel(fit=fit_ids[1])
first.quantity == second.quantity  # can be True
first.estimate != second.estimate  # True for different producing runs
by_estimate = {d.estimate: d for d in trial_energies}
```

A dictionary keyed by quantity is a deliberate semantic grouping and cannot
represent multiple estimates per key without lists. Provide collection
`group_by_quantity()` returning quantity → list of estimates; do not encourage
quantity-keyed dictionaries as a replacement for estimate collections.

Canonical documentation uses `.sel(time=t)` for selection and `.ts` for temporal
calculation, fitting and plotting. `.ts.at`, `CorrelatorMatrix.at`, free
`effective_energy`, `SamplingFit.from_data`, and plotters remain delegating
compatibility or lower-level entry points, with links to the canonical examples.

## Value model

### Channels and states

Use one immutable `Channel` value across energy and correlator metadata, composed
with the existing immutable `SectorInfo` for charges. Keep conversion methods
for the current operator channel spelling and energy `irrep`/`psq` spelling.

| Value | Fields / rule |
| --- | --- |
| `Channel` | `irrep`, `psq`, optional `momentum`, optional `row`, optional `flavor: SectorInfo` |
| `PivotModeKey` | validated channel, stable pivot identity, local mode index |
| `LevelKey` | channel, assignment scheme identifier, physical level index |
| `ParticleKey` | canonical particle name, optional nonnegative `psq` |
| `ReferenceKey` | structured normalization identity, initially a canonical particle name |

Equality and hashing use exact stored fields, including `None`. A partially
known channel differs from a fully specified one. `candidate.matches(query)` is
directional: every supplied query field must agree with a known candidate field.
Symmetric compatibility remains a separate operation. Neither changes equality.
If momentum and `psq` are both supplied, validate their consistency; if only
momentum is supplied, derive `psq`. Normalize inputs once at construction.

Keep `SectorInfo` rather than flattening its fields into every class. Do not
infer absent charges from names or equate row-resolved and row-averaged channels.

`PivotModeKey` is semantic metadata attached to a GI operator, not a new Sigmond
operator encoding. Existing parser validation accepts only the current operator
types; changing that hierarchy would add migration cost without helping export.
Bind each operator independently using an immutable `ModeBound(operator, mode)`
value in semantic auxiliary coordinates. A correlator's sink and source each
accept an operator or such a binding; rotated off-diagonals retain both modes,
and overlaps bind only their level-side operator. Each binding validates channel
agreement. Sigmond serialization unwraps the original operator; SS metadata
retains the binding. Do not add a single diagonal-only `mode` field to Correlator.

`level_index` projects only assigned `LevelKey.level`. A pivot-mode query requires
both `pivot` and `mode`; `filter(mode=2)` without a pivot raises a descriptive
error instead of selecting unrelated pivots. Legacy energies with a stored
level index map to a documented legacy assignment scheme, not to pivot modes.
Their existing filter behavior remains supported by that adapter.

Quantity keys omit ensemble context, enabling semantic grouping across ensembles.
This does not establish physical state correspondence: exact channel fields and
assignment scheme still must agree, and pivot-qualified modes are intentionally
local. Cross-ensemble matching requires explicit state assignment under a shared
scheme or a domain-specific matching policy, never level number alone.

A pivot identity must survive persistence and distinguish different source
bases/rotations. Audit the existing `SinglePivot.identifier` before reuse. The
identifier is a stable reference to a recorded pivot artifact, not merely its
display name; the full transformation belongs in provenance/artifact storage.
The current eight-character identifier hashes ensemble, labels, settings and
source coefficients, but not the solved rotation/input estimate. It is useful
as a suggested name, not sufficient as the new artifact identity. Persist a
producer/artifact reference that distinguishes separate builds and their inputs.

### Quantity keys

Use small frozen value classes, with transitively immutable fields:

| Quantity | Meaning / required distinctions |
| --- | --- |
| `Energy` | channel, optional state, lab/CM frame, optional shift reference, optional normalization reference, particle description |
| `RelativeMomentumSquared` | channel, optional state, decay particles, optional normalization reference; maps legacy `qcmsq` |
| `NonInteractingEnergy` | particle keys with momenta, frame, optional normalization reference |
| `Correlator` | independently bound sink/source keys and connected/subtraction semantics |
| `VEV` | existing operator key |
| `EffectiveEnergy` | source correlator, kind, step, constant-subtraction specification, relevant temporal model context |
| `Overlap` | basis operator, mode-bound level operator, normalization convention |
| `FitParameter` | parameter name and model namespace when no physical mapping is declared; run identity belongs to the estimate |
| `Theory` | fit reference and source quantity |
| `DerivedQuantity` | bounded operation description and operand quantity keys |
| `TimePoint` | temporal quantity and source time |
| `Component` | source quantity and `part="re"` or `"im"`; distinct from the complex quantity |
| `NamedQuantity` | namespaced kind, schema version, canonical immutable fields and display metadata |

For shifted energies, the shift reference identifies the noninteracting energy
being subtracted; it is not a boolean. For normalized energies, distinguish the
reference identity from the particular sampling used to compute the ratio;
the latter belongs in provenance. Different uncertainty treatments retain
different provenance even when the physical quantity key is the same.

The table defines categories, not permission to invent new physics conversions.
Existing `elab`, `ecm`, `delab`, `decm`, and `qcmsq` must have explicit, validated
legacy mappings. `Particle` currently is mutable and unhashable: convert it to
`ParticleKey` at boundaries before inserting it into any frozen key.

Do not classify an energy as single-hadron solely from tuple length: a shift's
reference particles do not identify the measured state. Use an explicit
single-hadron descriptor where supported by legacy metadata.

The minimal quantity contract is exact equality/hash, a kind discriminator, and
a display label. Serialization uses codecs. Temporal evaluation and energy
targets are capabilities, rather than mandatory methods on every quantity:

- Temporal quantities support `at(t)` returning `TimePoint`.
- Correlators and effective energies expose an optional `estimates` energy
  target. A pivot binding supplies a mode identity; raw correlators can supply
  channel-only targets when supported, without fabricating a level number.
- Quantity replacement returns a new validated key.
- A codec may generate a suggested storage name. A quantity does not universally
  require a representable Sigmond name.

An effective-energy target is a labeling/model contract, not a claim that a
finite-time estimate has isolated a single eigenstate.

### Extension contract

The taxonomy is open. A quantity extension registers a namespaced kind with the
existing `obs_kind` infrastructure through an adapter, preserving legacy tags.
The registry rejects conflicting registrations; built-ins own reserved names.
A registration provides:

1. An immutable value type with validated, transitively hashable semantic fields.
2. A versioned codec with encode/decode and any legacy mapping.
3. A label renderer and optional facet projections/suggested export names.
4. Optional temporal, component and model-target capabilities.
5. Equality/hash, malformed-input, unknown-version and round-trip tests.

No numerical/plotting imports are required to construct a key. Labels do not
participate in semantic equality. Kind/version and canonical semantic fields do.
Optional facets do not automatically authorize arithmetic propagation.

`NamedQuantity` is the serializable fallback for matrix elements, decay
constants, phase shifts, ratios or other new results without a custom class.
Its fields use a restricted canonical value tree: scalars, immutable sequences,
key-sorted mappings and registered nested keys; reject arbitrary Python objects
and non-finite identity values. Display metadata stays outside equality. A
named quantity can round-trip without executable plugin code. An unknown custom
payload is preserved opaquely rather than reinterpreted as a known quantity.

### Storage, envelopes, and provenance

`ObservableAddress(name, index, op_type, re_im)` describes one stored component.
Its identity is scoped by ensemble and source/group context. Keep `op_type`:
dropping it from the proposed envelope would lose an existing address field.
A complex quantity may have both real and imaginary component addresses.

`ObservableInfo` becomes a compatibility view combining quantity, optional
address, ensemble context, and label overrides. Expose common filter fields
through an explicit facet adapter; avoid unrestricted `__getattr__` delegation.
For example, `irrep` projects `quantity.channel.irrep`; `level_index` projects
only an assigned physical level, while pivot and mode are separate facets.

Keep existing `ObservableInfo` equality/hash behavior during the transition.
Provide explicit semantic comparison through quantity equality and explicit
storage comparison through addresses. Do not silently re-key collections when
new quantity metadata is attached.

`data.estimate` is the identity of one estimate, distinct from its quantity,
address and provenance. Use `EstimateRef(producer_id, output_key)`: producer IDs
are persisted opaque run/derivation IDs; output keys distinguish parameters,
components and cells from that producer. A fit run ID alone cannot identify both
its amplitude and energy. Mint IDs once at a producer boundary, never on property
access. Numerical derivations and explicit semantic reassignment create new
references; pure selections/reordering/copies preserve per-cell references.
Broadcasting repeats existing references rather than pretending to create data.

`.estimates` exposes a requested DataArray of per-cell references over all data
dims, including parameter axes. `.estimate` is available for one selected cell;
a non-scalar series directs callers to `.estimates`. Loaded legacy estimates use
stable source-artifact/group/component identity, persisted when re-exported.
Different sampling representations are distinct estimates with shared provenance
links. Define artifact identity without mutable path names or reading all sample
buffers merely to access metadata. Duplicate references with conflicting sample
context are errors; repeated views of the same estimate are valid.

`data.address` exposes the selected cell's optional stored component address
(or real/imaginary address pair for complex data). Arrays retain the extension
address mapping without requiring everyday users to allocate it. Derived values
have no output address until exported. `data.ensemble_info` is first-class.

`data.provenance` has a readable representation and exposes `operation`, `inputs`
and `settings`. Inputs are estimate references, not mandatory strong references
to live data objects. A resolver can retrieve retained metadata/artifacts; an
unavailable input is an explicit unresolved reference. Walking history must not
materialize sample buffers or retain every intermediate array in memory.

Provenance records operations, input references, fit settings/windows, pivot
references, and approximation choices. It does not participate in quantity
equality. Use shared records/references instead of duplicating full history in
every cell. Preserve sampling-plan compatibility checks in the existing data
layer; quantity equality never proves sample alignment.

## Array layout and templates

The array metadata contains:

1. Named dimensions, explicit dimension roles and index coordinates.
2. Auxiliary coordinates over any subset of dimensions, including scalar coords.
3. A quantity specification: a constant key, a validated template, or a quantity
   coordinate for heterogeneous cells.
4. Estimate mappings and optional address/provenance mappings, independent of
   semantic coordinates.

Use explicit coordinate references, not overloaded strings:

```python
# Target API: "forward" is a literal; CoordRef("sink") is a reference.
matrix_q = QuantityTemplate(
    Correlator, sink=CoordRef("sink"), source=CoordRef("source"),
)
diagonal_q = QuantityTemplate(
    Correlator, sink=CoordRef("bound_operator"), source=CoordRef("bound_operator"),
)
eff_q = QuantityTemplate(
    EffectiveEnergy, of=diagonal_q, kind="forward", step=1,
)
mixed_q = QuantityTemplate.from_coord("quantity")
```

Validate templates at construction: referenced coordinates exist, dimension
dependencies are compatible, constructors accept the resolved values, and
bindings agree with operator/channel metadata. Resolve keys lazily without
building one Python object per sample or every array cell.

### Dimension roles and predictable quantity access

`data.dim_roles` is producer-owned metadata, preserved through rename/selection.
Roles are explicit and do not depend on spelling a dimension `time` or `fit`:

| Role | Meaning | Examples |
| --- | --- | --- |
| `semantic` | chooses physical identity/template fields | operator, sink, source, assigned level |
| `parameter` | evaluates a quantity along an independent variable | time; later energy for a phase-shift series |
| `replicate` | distinguishes estimates without asserting new physical meaning | fit trial, analysis variant |

The Monte Carlo sample axis is internal, never a replicate layout dimension.
A parameter axis has a descriptor declaring its variable kind; time descriptors
supply `TimeAxis` and `.ts`. Design the role schema to accommodate other parameter
axes, but implement only time evaluation in this migration. Do not build a general
parameter-axis fitting framework speculatively. Template dependencies must agree
with roles; raw constructors declare roles explicitly. Legacy adapters infer
only established conventions and leave ambiguous axes for caller declaration.

Quantity access follows shape/roles, never an equality scan of all cells:

- `.quantity` returns the series quantity once semantic and replicate dims are
  selected. Unselected parameter dims represent a function, so a single
  correlator time series returns `Correlator` (or `Component(Correlator, "re")`).
- Selecting time into a scalar coord evaluates it: `.sel(time=8).quantity` and
  `.ts.at(8).quantity` both return `TimePoint(series_quantity, 8)`.
- `.quantity` raises when any semantic or replicate dimension remains, even
  when all keys happen to be equal, and directs callers to `.quantities`.
- `.quantities` returns an `xr.DataArray` over remaining semantic and replicate
  dims, with scalar selected parameter coordinates evaluated. It omits
  unselected parameter dims: `D.quantities` spans operator; a single series has
  a zero-dimensional quantity DataArray. Selecting a time on the data first
  produces evaluated keys. No parallel `quantity_at` selection API is needed.

`.quantities` constructs metadata keys only when requested and may cache against
immutable metadata. A normal object-valued xarray DataArray is not inherently
lazy: budget its materialization and permit selection on the data first. Samples
remain untouched. Producer internals resolve individual templates without
materializing a complete key grid.

Retain readable integer/string index coords where convenient; typed auxiliary
coords can hold operators, modes and quantities. The initial implementation
does not require custom xarray indexes for arbitrary dataclasses. Current
`SigmondData._from_parts` retains dimension coords only, so auxiliary coordinate
support is an explicit deliverable, not an assumed xarray feature.

An `.infos` compatibility view derives envelopes on demand. It combines semantic
keys with retained addresses and provenance; cache only with immutable metadata
ownership or a documented invalidation rule. Heterogeneous legacy arrays may
retain a quantity/address coordinate rather than fitting a homogeneous template.

Missing cells remain explicitly missing. A rectangular template does not prove
that every dataset exists; preserve coverage masks and do not export nonexistent
observables because a quantity can be synthesized for their coordinates.

## Operation contract

| Operation | Quantity behavior | Addresses / history |
| --- | --- | --- |
| Selection, transpose, dimension rename | preserve meaning and selected scalar coords | subset/remap addresses; no new measurement |
| Stack/concat | declare new axis role; preserve each input quantity | preserve estimate references/history; reject conflicting references, not repeated views |
| Broadcast | repeat meaning with explicit alignment rules | repeated views are not newly stored measurements |
| Diagonal extraction | resolve sink/source together; preserve mode bindings | subset addresses |
| Real/imaginary projection | create `Component(of=q, part="re"/"im")` | resolve component addresses; distinct component output reference |
| Scalar rescaling of a correlator | retain correlator meaning; record changed normalization | invalidate output address, record operation |
| Generic arithmetic / reductions | produce bounded `DerivedQuantity`, or unknown quantity if unsupported | invalidate output address, retain input references |
| Explicit energy normalization/shift/frame conversion | create the specified physical quantity after validation | invalidate output address, record reference samples/settings |
| Matrix basis change | create new correlators with new operator keys | record transformation and invalidate addresses |
| Effective energy | create `EffectiveEnergy`; preserve eligible energy target | record algorithm/context and invalidate addresses |
| Fit parameter | declared physical mapping, otherwise `FitParameter` | retain fit run/model/window reference |

`Component(Correlator, "re")` differs from the complex correlator and from its
imaginary component. Effective-energy producers accept supported real component
quantities, retaining the underlying correlator's eligible energy target.
Component projection is idempotent for the same part; unsupported compositions
require defined semantics rather than blindly nesting wrappers.

Metadata dispatch follows the operation, not just whether operands have equal
keys. `C / C` must not keep `Correlator`; `E + E` must not automatically keep
`Energy`. Explicit helpers such as `E.normalized_by(m_pi)` perform the sample
arithmetic and semantic update together, after existing ensemble/sampling checks.
They do not alter numerical uncertainty propagation.

Keep derived expressions bounded: use shared provenance references for long
histories instead of recursively expanding arbitrarily large trees in labels or
HDF5 attrs. Unsupported ufuncs must not silently preserve the old meaning.

## Time and fit semantics

`TimeAxis` exposes read-only source values, display offset, optional extent,
boundary convention, and time direction. Source values stay distinct from
display coordinates. Temporal model context that changes quantity meaning is
also included in the corresponding quantity/codec; display offset is derived
from the effective-energy convention so two copies cannot drift apart.

- `.ts.at(t)` selects a source time; `.ts.window(a, b)` selects an inclusive
  source-time window and preserves irregular spacing/order according to existing
  selection conventions.
- `.ts.effective_energy(...)` delegates to the current kernel, retaining all
  four kinds, constant subtraction, neighbor validation, and failure policies.
- `.ts.plot()` uses display coordinates and quantity labels. It delegates to the
  current plotters rather than creating another plotting backend.
- `.ts.fit()` binds source coordinates by default. A model may explicitly request
  display coordinates; the bound choice is recorded in the fit result.
- `.ts.fold()` requires an explicit supported boundary/reflection convention.
  Never infer periodicity from the existence of an extent. Preserve the numerical
  conventions for forward/backward operators and connected correlators.

Fit semantics are optional capabilities on the existing model protocol. Standard
fit-time models declare their own typed bindings, e.g. `EnergyOfInput()` for the
single-exponential energy parameter. Ordinary calls do not require a separate
`ModelSemantics` object. Custom models may supply typed `targets={"E0": key0,
"E1": key1}` overrides; validate names, channel compatibility and available
input targets. An unresolved standard binding reports a diagnostic rather than
silently assigning a level. Arbitrary callables without bindings produce
`FitParameter`; parameter spelling is never a physics declaration.

Initially support one energy target for one temporal series. Multi-exponential
and multi-channel models require explicit bindings. The post-fit factories in
`fitting/model_func.py` are not fit-time models today: create/adapt a standard
fit-time model in `fitting/model.py` (or a small dedicated model module), and share
semantic declarations with post-fit wrappers where appropriate. Do not add
metadata allocation to `predict` or objective evaluation. `fit.param("E")`
continues to return sampling data; `fit.theory_series()` returns a `Theory`
quantity with appropriate time context and estimate references.

## Persistence and collection behavior

Continue reading existing HDF5 addresses, `obs_kind` tags, energy attrs, explicit
labels, and files without SS metadata. Legacy names are authoritative addresses,
not automatically regenerated from newly attached quantity keys.

Add a versioned optional quantity/estimate/provenance payload to SS metadata. Use the
existing consolidated `ObsMeta` mechanism where practical. Keep legacy fields
for kinds that older readers understand; new kinds need an explicit documented
fallback rather than masquerading as a legacy energy type. Unknown versions or
kinds preserve the raw payload and address with a generic envelope; do not
silently discard annotations. Conflicting known metadata produces an actionable
diagnostic and follows a documented precedence policy.

At the writer boundary, retain authoritative addresses for unchanged stored
measurements. For derived outputs, default `write(..., names="suggested")` uses
codec suggestions and deterministic estimate/output qualifiers when needed.
Suggestions allocate destination addresses, never reuse input addresses from
provenance. Allow explicit name/index mappings for controlled exports. Preflight
the full destination, report all remaining collisions with conflicting estimates
and suggested remedies, and write nothing on preflight failure. No silent
quantity-based deduplication or implicit overwrite; validate Sigmond limits.
Persist allocated addresses and estimate IDs in output metadata. Writing a copy
does not create a new estimate solely because its address changes.

The new collection core keys entries by estimate reference, regardless of stored
or derived status, with separate quantity and scoped-address indexes. Old
ObservableInfo-keyed construction/lookups use explicit compatibility adapters
until phased out. Do not switch old hash semantics silently. Array → collection
→ array conversion preserves duplicate quantities, estimate references,
ordering, sample alignment, components, labels and missingness.
`group_by_quantity()` provides semantic grouping; storage lookup uses scoped
addresses. Stacking permits duplicate quantities and repeated views, but rejects
one estimate reference paired with conflicting quantity/sampling context.
Deduplication, when requested, is explicit and by estimate, never by quantity.

## Proposed source layout

```text
sigmondsamplings/
  quantities/
    __init__.py          # small, lazy public exports
    base.py              # value/capability contracts and common kinds
    channel.py           # unified Channel; legacy conversion helpers
    energy.py            # ParticleKey, energy/state/reference keys
    temporal.py          # EffectiveEnergy, TimePoint, Theory
    derived.py           # bounded derived quantity descriptions
    named.py             # NamedQuantity canonical fallback
    registry.py          # namespaced registration adapter and capabilities
    templates.py         # extension API: CoordRef, template resolution/validation
    facets.py            # explicit filter/label projections
  metadata.py            # addresses, estimate references, provenance records
  axis_roles.py          # explicit roles and parameter descriptors
  time_axis.py           # time parameter descriptor and .ts facade
  info.py                # existing infos; transitional envelope adapters
  energy_levels.py       # existing API; adapters to quantity keys
  data.py                # buffers/stats; quantity and layout access points
  sampling_array.py      # dimension + auxiliary-coordinate transformations
  correlators/
    operators.py         # existing operator encodings; imports Channel
    keys.py              # retain CorrelatorInfo/VEVInfo; expose quantity contracts
    matrix.py            # domain operations and template construction
    pivot.py             # mode bindings and provenance; unchanged solve
    effective.py         # numerical calculation; called through .ts
  fitting/
    semantics.py         # typed model-owned bindings and target overrides
    result.py            # quantities on params/theory; existing result API
  io/
    quantity_codec.py    # versioned codec + legacy attr mapping
    obsmeta.py           # consolidated transport, no domain constructors
tests/
  test_quantity_keys.py
  test_quantity_templates.py
  test_quantity_operations.py
  test_quantity_roundtrip.py
  test_estimate_identity.py
  test_quantity_extensions.py
  test_time_axis.py
  test_fit_semantics.py
```

Keep pure value modules free of xarray, plotting, fitting and I/O imports. Existing
correlator keys should be reused/adapted, not recreated as a parallel hierarchy.
`SectorInfo` can remain in `info.py` initially; avoid runtime import cycles with
local conversion imports. Codecs import domain types; domain types do not import
codecs. Templates describe layout independently of numeric algorithms.

## Migration plan and completion gates

### Phase 0 — Record behavior and settle contracts

Record revision, working-tree diff fingerprint, environment, optional
dependencies, relevant pytest results, skips and failure identities. This checkout
has substantial existing changes: a revision alone does not identify the baseline.
Do not assume the previously quoted count of 78 failures is current. Avoid an
unreviewed permanent xfail list.

Write characterization fixtures for legacy energy kinds, duplicate estimates,
real/imaginary addresses, a pivot diagonal, effective energies, overlaps and old
files without `ObsMeta`. Confirm channel matching, state assignment, export
allocation and metadata precedence semantics before implementing their APIs.

Gate: each old field has a documented destination, and representative fixtures
can distinguish numerical changes from metadata changes.

### Phase 1 — Add immutable keys and legacy adapters

Implement `ParticleKey`, channel conversion, energy/state/component keys,
operator-side mode bindings, estimate/provenance value contracts, registry/fallback,
facet adapters and versioned codecs. Add read-only `.quantity` projections to
existing infos. Until old infos stop mutating, build projections from current
attrs rather than caching stale quantity snapshots. Retain old identity/hash and
constructor behavior. New keys are immutable; legacy mutation stays supported.

Gate: every existing energy kind and supported correlator key maps losslessly;
hashing works transitively; old files and collection keys retain behavior.

### Phase 2 — Complete one vertical workflow

Add auxiliary/scalar-coordinate support, explicit axis roles, templates,
`.quantity`/`.quantities`, per-cell estimate/address mappings and `.ts`
effective-energy/selection operations for pivot diagonals. Attach pivot-qualified
mode keys without changing GI encoding. Provide derived `.infos` for this path.
Support array/collection conversion and write/read of its metadata. Introduce the
operation dispatcher here so unsupported arithmetic cannot preserve misleading
identity while new arrays are in use.

Gate: matrix → pivot diagonal → effective energy → scalar selection → collection
→ file → reload preserves samples, semantic keys, source/display time, and
provenance, and estimate references. `sel(time=t)` and `.ts.at(t)` produce equal
keys. Two pivots with local mode 2 stay distinct; bound off-diagonal keys retain
both modes. Replicate-axis trials survive collection conversion. Irregular-time
and missing coverage cases are handled explicitly.

### Phase 3 — Add fit targets and spectra

Add typed model-owned semantics, target overrides, a standard fit-time
single-exponential model, `.ts.fit` delegation and fit parameter mapping, then
explicit level assignment and spectrum stacking. Keep raw-mode and assigned-level
filtering distinct. Generic models still work without semantics.

Gate: changing a fit window creates a distinct estimate without changing its
energy target; stack/export preserves both. Model bindings survive serialization
where supported; unsupported callable serialization is reported explicitly.

### Phase 4 — Generalize arrays and temporal views

Extend templates to full matrices, VEVs, overlaps and heterogeneous quantities.
Add `.ts.plot` delegation; implement folding only after specifying
its boundary and reflection rules. Preserve standalone effective-energy and
existing plotter entry points as wrappers.

Gate: selection, diagonal extraction, transpose, rename, broadcasting, stacking
and supported arithmetic satisfy the operation table. Existing pivot oracle and
effective-energy checks show no numerical regressions. Benchmark template
resolution and collection/export boundaries against existing scripts before
removing eager metadata allocation; avoid per-sample key construction.

### Phase 5 — Consolidate compatibility envelopes

Make first-class metadata and estimate-keyed collection internals the canonical
model. Convert observable subclasses to compatibility adapters, migrate mutation
sites to validated replacement/re-keying operations, and document deprecations.
`AttributeAccessor.update`, collection mutation, `io/edit.py`, and CLI edits must
be audited before disabling legacy mutation. Update public exports and `.pyi`
stubs at each phase, rather than waiting until the end.

Gate: supported legacy imports, constructors, filters and file paths work through
adapters. Replacement cannot corrupt hashed collections. Removal of compatibility
APIs requires a separate release decision after downstream use is audited.

## Implementation handoff for agents

Implement one phase per reviewable change; do not interpret this document as
authorization to rewrite the whole package at once. Start each phase by inspecting
the current tree and applicable instructions, since the API may have evolved.

Each change should report:

- Which phase and contracts it implements.
- Public API additions and compatibility behavior.
- Representative legacy and target round-trip results.
- Numerical checks relevant to the touched path; pre-existing failures separately.
- Any unresolved contract or unavailable optional dependency/oracle.

Use existing tests as the base: `test_sampling_array.py`, `test_nd_primitives.py`,
`test_collection_mutation.py`, `test_selection.py`, `test_loader_writer.py`,
`test_slatmeta.py`, `test_correlator_matrix_views.py`, `test_correlator_effective.py`,
`test_pivot.py`, `test_pivot_oracle.py`, and `test_fitting_from_data.py`. Add tests
for the new behavioral contracts, especially duplicate identity and transformation
semantics; constructor snapshots alone are insufficient. Run optional oracle
checks when available and report skipped checks accurately.

Do not change sampling order, full-estimator conventions, covariance calculation,
GEVP normalization/gauge, connected-correlator workflows, or fit optimization
behavior as an incidental part of metadata migration.

Related existing designs:
[operator/correlator API](operators_correlators_api_design.md),
[correlator views](correlator_views_design.md),
[single pivot](single_pivot_design.md),
[pivot usage](single_pivot_usage.md), and
[HDF5 structure](sigmond_hdf5_structure.md).
