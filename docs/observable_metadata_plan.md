# Observable metadata: sectors, time series and energy levels

Status: proposed implementation plan, revised 2026-10-13. Examples are **target
API**, not claims about the current implementation. This supersedes the scope
of `quantity_redesign_plan.md`, which remains a record of the larger design
considered. The open questions in the previous version are resolved below.

## Goals and constraints

1. Physical-sector metadata has one home: immutable `SectorInfo`. Producers
   populate what they know; consumers read what they need.
2. Effective-energy cells retain their source, calculation settings, sector and
   rotation context through selection, collections and SS HDF5 files.
3. Time-dependent elements and values at coordinates share a small convention.
4. GEVP rotation indices remain distinct from energy-sorted level indices.
5. Array rearrangement preserves cell metadata; arithmetic describes its result.
6. Energy and particle roles are explicit, with legacy decoding at the boundary.

`ObservableInfo` remains the metadata API. Keep attribute filtering and the
generic data, sampling, energy and fitting interfaces except for explicit
changes in this plan. No quantity-key hierarchy, templates, estimate references
or provenance graph. Callable serialization and automatic correlator-fit
parameter semantics are outside this change.

**No backward compatibility is required for SS correlator or operator APIs.**
This includes their constructors, class names, imports, accessors, labels,
matrix/pivot/effective-energy entry points and old SS-specific annotations.
Change or remove them directly when that simplifies the implementation; update
callers, exports, type stubs, examples and tests in the same phase. Add no
compatibility wrappers, aliases or deprecation period solely for these APIs.

The required contract is agreement with Sigmond: native operator/correlator
representations, key meaning and flags, file interoperability, and numerical
conventions/results for supported operations. Canonical text may change when
it represents the same native key. These requirements do not preserve an old
SS Python interface or metadata schema. Energy/general API compatibility
described elsewhere in this plan is unaffected by this exception.

## Decisions

| Topic | Decision |
| --- | --- |
| Physical sectors | Extend frozen `SectorInfo` with PSQ/irrep; every info exposes it, defaulting to an empty sector. |
| Familiar facets | Forward existing attributes to the sector; do not store independent copies. |
| Partial sectors | `None` is unknown. Equality/hash are exact; partial compatibility is separate. |
| Correlator cells | Use a typed native-key representation with `.element`/`.time` because the Sigmond key has its own semantics; existing SS classes/interfaces may change. |
| Rotation | Attach explicit pivot bindings when rotating. Never infer them from `transformation`, names or raw indices. Display `ROT n`. |
| Level assignment | `with_level_indices()` returns copied metadata; no in-place variant initially. |
| Time | `.ts` is a computed view. No independently stored axis state or boundary field. |
| Labels | Retain `latex_str`; share small helpers, not a symbol/decorations object model. |
| Arithmetic | Value-changing unary, scalar and data/data operations create derived metadata. |
| Prediction series | Share wrapping of sampled predictions, not model evaluation. Never store a live model as metadata. |
| SS correlator/operator APIs | Clean breaks are allowed; no compatibility facade or deprecation requirement. Remove array-level `EffectiveEnergyObsInfo`. |
| Sigmond agreement | Validate native operator/key semantics, file interoperability and numerical conventions against Sigmond. |

## 1. Shared physical sectors

### Fields and authority

Extend the existing frozen, hashable `SectorInfo` in `info.py` with `psq` and
`irrep`. Keep its existing fields: `B`, `Q`, `S`, `C`, `twoI`, `twoI3`, `twoJ`,
`Gpar`, `Cpar`, `par`. All fields default to `None`; zero is a specified value.
Preserve recognized-irrep validation and require nonnegative integer PSQ.
Update hash/equality, compatibility, repr and display helpers together.

Every `ObservableInfo` exposes `sector_info`, defaulting to `SectorInfo()`.
Ordinary infos store it. Correlator/VEV infos derive it from the parsed channel,
optionally caching that immutable result. They do not keep an independently
editable sector. Where raw loading of a native key unsupported by the SS parser
is possible, retain an opaque key and parse diagnostic with an unknown sector.
Do not guess its physics or reinterpret it as a different observable.

Keep `irrep`, `psq` and existing charge/isospin spellings as forwarding
properties. `strangeness` aliases `S`; an `isospin` string is derived from `twoI`
using Sigmond's vocabulary. Expose the remaining sector fields as facets. Keep
aliases that aid analysis, without requiring old correlator-only spellings.
Do not store a second string-isospin or strangeness value.

Update attribute discovery, CLI listings and collection/DataFrame summaries to
include these public properties; inspecting `vars(info)` alone no longer finds
the physical fields. Keep the existing attribute-filter syntax.

Add keyword-only `sector_info=` while retaining legacy `irrep=`/`psq=` arguments
for energy/general infos. Correlator/operator constructors may adopt the new
representation directly. Normalize once: overlapping known values must agree;
unspecified legacy values do not overwrite a populated sector. Reject conflicts.
Existing writable facets on ordinary infos replace the frozen sector with
`dataclasses.replace`; keyed facets remain read-only. Sharing a sector cannot
cause cross-info edits.

`Channel.to_sector()` fills `twoI`, `S`, `psq` and `irrep`. Exact momentum vectors
and irrep rows stay on `Channel`, describing the operator representation.
Preserve both source and sink channels on correlators. Do not infer missing
charges from particle names or labels.

### Identity and consumers

Include the full sector in observable equality/hash through the shared base
implementation, without repeating its fields in each subclass. Equality is
exact, including `None`. Keep `compatible_with()` as symmetric partial
compatibility; never use it for dictionary keys, deduplication or level grouping.

Keep the energy `.sector` property as the existing `(psq, irrep)` compatibility
pair. New code that distinguishes physical channels uses `.sector_info`.
Rotation/level indices, energy type, normalization and particle roles stay
outside the sector: sharing a sector does not mean sharing an estimate.

Add `EnergyObsInfo.from_channel(channel, *, ensemble_info,
energy_type="elab", rot_idx=None)`. It uses `channel.to_sector()` and requires
the ensemble explicitly. Effective energies copy their source's immutable
sector directly; they do not need an intermediate energy info to obtain facets.

## 2. Preserve cells through rearrangement

Fix `sampling_array._joined` independently. Selection, transpose, diagonal
extraction and stack/concatenate retain the corresponding per-cell infos even
when array-level infos differ. A mixed array can use `ArrayObsInfo`; that does
not justify replacing its cells with placeholders. Scalar selection retains
the selected cell's info.

Keep existing sampling/ensemble compatibility checks; this does not expand
which data can be combined. Collection-to-array reconstruction remains explicit
through `shape_by`, with duplicate target cells rejected.

## 3. Elements, coordinate values and persistence

### Small shared convention

- At-time infos expose `.element` and `.time`.
- A correlator cell's `.element` exposes its time-independent correlator;
  `.time` comes from the native at-time key. Flags/component and rotation
  bindings stay on the cell. Keep this typed distinction for the Sigmond key's
  semantics, not for compatibility with existing SS classes. Renaming or
  consolidating those classes is allowed. No `AtObsInfo` inheritance is needed.
- Add registered `AtObsInfo(element, coord, value)`. It exposes `.coord` and
  `.value`, and `.time == value` for `coord == "time"`, otherwise `None`.
- Use a structural convention/helper for consumers, not a universal element
  inheritance hierarchy.

New element records are immutable metadata snapshots. Reuse frozen keys and
`SectorInfo`; freeze normalized fields of mutable source infos. Do not retain
live mutable infos, model parameters or callables inside frozen records. If an
accessor returns an `EnergyObsInfo`, reconstruct an independent object with the
correct ensemble context.

Forward only `sector_info`, its field aliases and `rot_idx` where applicable.
Unavailable facets return `None` consistently. Do not forward arbitrary
attributes: `name`, `index`, `obs_kind`, classification and serialization belong
to the actual observable. Effective-energy cells are not fitted energy levels
for spectrum collection purposes.

### Identity before labels

An at-coordinate identity includes the complete element/source identity,
coordinate name and typed value, plus normal observable/ensemble context. An
effective-energy identity includes source form/component, pivot bindings,
kind, step, resolved calculation extent and constant subtraction.

Identity excludes display labels and label overrides. Normalize NumPy scalars;
encode numbers deterministically without display rounding. New coordinate
values and calculation constants must be finite.

Generate new at/derived storage names from a canonical versioned identity
record: readable kind prefix plus `ID` and 32 hexadecimal SHA-256 characters.
Retain the full identity record in metadata. Never use process-dependent
Python `hash()` for persistent identifiers. Equal generated names with unequal
records are an error, not a merge. Later equality/hash uses metadata only and
must not read sample buffers.

Correlator/operator text must represent the intended Sigmond key; old SS
spellings need not be retained. Legacy energy names keep their conventions,
apart from the explicit rotation token below. Independent legacy energy
estimates with otherwise identical metadata still require explicit namespaces.
This plan does not introduce a general estimate-identity system.

### Nested metadata codec

Use the existing HDF5 `ObsMeta` mechanism with recursive JSON normalization.
Separate wrapper fields from the nested element; do not flatten discriminators:

```json
{
  "obs_kind": "at",
  "metadata_version": 1,
  "coord": "time",
  "value": 8,
  "element": {
    "element_kind": "effective_energy",
    "sector_info": {"twoI": 0, "S": 0, "psq": 0, "irrep": "A1g"},
    "source": {
      "source_kind": "correlator",
      "correlator": "canonical CorrelatorInfo.to_sigmond() text",
      "hermitian": true,
      "subtract_vev": false,
      "re_im": "re",
      "sink_rotation": {"pivot_id": "stable pivot identifier", "index": 2},
      "source_rotation": {"pivot_id": "stable pivot identifier", "index": 2}
    },
    "kind": "forward",
    "step": 1,
    "time_extent": 64,
    "subtract_const": 0.0
  }
}
```

The strings above illustrate field contents. Decode `correlator` through a
Sigmond-conformant parser; its SS interface may change. Generic sources instead
encode `source_kind="series"`, a parent/source identity and named non-time
coordinates; prediction sources also
carry the series identifier described in section 8. Encode missing bindings
as null or consistently omit them. The scalar `rot_idx` is derived from equal
sink/source bindings, not stored a third time.

The wrapper forwards its element's sector; it does not store another copy.
Keep labels outside the identity portion. Decode registered metadata kinds
only, with descriptive errors for unsupported versions/kinds; never execute a
serialized callable or silently reconstruct the wrong observable type.

## 4. Explicit rotation and effective-energy metadata

### Pivot bindings

`SinglePivot.rotate()` attaches immutable `(pivot_id, index)` bindings using
the existing `SinglePivot.identifier`. Correlator cells have separate sink and
source bindings; only a diagonal with the same binding on both sides exposes
one `rot_idx`. Display `ROT 3`, without a pivot name. Keep the identifier for
source identity and disambiguation.

Attach bindings to `values` before returning the matrix. Diagonal/correlator
extraction, scalar selection and `to_collection()` then preserve them without
the enclosing matrix. Include them in cell equality/hash and SS attrs; preserve
the native `<CorrT>` meaning, including flags and component. Canonicalize text
as appropriate for Sigmond rather than preserving old SS text verbatim.

A non-null `transformation`, GI name or raw operator index does not establish
a pivot binding. General transforms/improvement create none. Exact selection
and reordering preserve bindings; mixing modes clears them. A fresh pivot
rotation establishes fresh bindings.

Raw rotated matrices with identical Sigmond operator labels need distinct
operator namespaces to coexist in a file; reject duplicate storage keys even
when annotations differ. Generated effective-energy names include the binding
and can coexist across pivots without changing raw-key encoding.

### Effective-energy elements and targets

Add frozen `EffectiveEnergyInfo` containing a source snapshot, `sector_info`,
`kind`, positive integer `step`, the `time_extent` actually resolved for the
calculation and `subtract_const`. Its `rot_idx` comes only from an unambiguous
rotated diagonal; `time_offset` uses existing `effective_time_offset()` rules.

Propagate a correlator's sector only when sink and source agree. Preserve both
source descriptions even when no common sector or scalar mode can be assigned.
A raw-basis index is never an energy-state assignment.

An optional known energy target is an immutable snapshot. Its `.energy` view
can reconstruct lab-energy metadata with sector, rotation index and ensemble;
it must not invent an energy-sorted `level_index` or fit-parameter meaning.
Filtering and time handling do not depend on this optional target.

Effective-energy signatures and SS interfaces may change. Preserve supported
Sigmond calculation types, source-time/offset conventions and numerical
results; define failure handling explicitly and test it. Reuse the numerical
kernels unless a Sigmond discrepancy requires correction. Each output cell gets
`AtObsInfo(EffectiveEnergyInfo(...), "time", t)`. Build one element per non-time
position and share it along time.

Plain numeric arrays remain supported through generic series descriptors from
parent identity and named non-time coordinates. Do not parse physical meaning
from placeholder names. Explicit element metadata must agree along time,
including flags/component/bindings; diagnose inconsistency instead of selecting
the first time entry's metadata.

### Array metadata

Use plain `ArrayObsInfo` at array level; complete element/cell metadata is the
source of truth. Remove array-level `EffectiveEnergyObsInfo` and its exports
directly, with no compatibility facade. Update plotters, imports, stubs,
examples and tests to use the cell/computed-view API in the same phase.

## 5. Computed time-series access

`data.ts` requires a named `"time"` dimension with unique numeric coordinates.
It attaches no independent axis state to `SigmondData`:

| Member | Contract |
| --- | --- |
| `times` | Existing source-time coordinates, without display offset. |
| `elements` | DataArray over non-time dims; validate common elements and source context along time. Scalar DataArray for a 1-D series. |
| `axis.offset` | Common computed effective-element offset, otherwise zero. Raise when element offsets differ. |
| `axis.extent` | Common recorded calculation extent, or ensemble extent for generic series. Raise when element values differ, including known versus unknown. |
| `at(t)` | Exactly `sel(time=t)`; selected cells retain `.time`. |
| `window(a, b)` | Inclusive coordinate mask, preserving input order; reject reversed bounds. |
| `effective_energy(...)`, `plot(...)` | Delegate to the existing calculator and `SeriesPlotter`. |

For built-in placeholders, `elements` may construct a generic descriptor from
parent identity and non-time coordinates. Explicit elements require consistency.
An empty time axis permits coordinates/windows, but element/offset/extent
queries raise because no element can be established.

Stacked series may have different offsets. Plotting splits series first and
resolves each offset separately; a common scalar query raises with conflicting
values. Explicit `x_offset` still overrides the plotting default. Apply the
default time offset only when plotting the time dimension.

Calculation extent precedence stays explicit `time_extent=` then ensemble
extent. Record the resolved value on the effective element. Display offset
never alters selection, fit coordinates or calculation. No boundary inference
or change to symmetric-kind numerical assumptions is introduced.

Selection, transpose, stacking and HDF5 reads recover time behavior from cells
and coordinates. Rebuilding an array still requires explicit `shape_by`; this
does not promise recovery of an arbitrary original layout.

## 6. Energy roles and level ordering

### Particles and roles

Make `Particle` a frozen dataclass `(name, psq=None)`. Keep constructor
normalization/validation, `from_string`, string/label formatting and existing
momentum ordering. Drop equality with strings; do not introduce alphabetical
ordering as an incidental dataclass change.

Keep these fields outside `SectorInfo`:

| Field | Meaning |
| --- | --- |
| `hadron` (`str` or `None`) | This single hadron's energy; momentum comes from the sector PSQ. |
| `ni_particles: tuple[Particle, ...]` | NI constituents or the channel/baseline for a shift or `qcmsq`. |
| `non_interacting: bool` | This observable is itself the NI sum. |
| `energy_type`, `ref_particle` | Representation and normalization. |
| `rot_idx`, `level_index` | Local pivot mode and explicit energy ordering, independently. |

Keep `SHEnergyObsInfo` and its obs-kind tag as the SH validator/label
specialization, with `particle` as an alias of `hadron`. Single-hadron
construction uses this specialization; reject `hadron` on a plain energy info
rather than silently changing its classification. Reject SH shift/`qcmsq`,
SH+NI combinations and conflicting legacy/new role arguments.

Keep read-only `particles` as a compatibility view. Legacy constructors and
file adapters map it by obs kind, energy type and the legacy NI marker, not
tuple length alone. For SH the view builds a particle from `hadron` and PSQ.
Update `set_shift_particles()` and other writers to assign `ni_particles`;
`kinematics._particle_name` reads the SH role. Enforce operand counts in the
specific kinematic operation, not a universal two-particle limit on NI sums.

Centralize NI naming as `[PSQ{p}_]elab_ni_{constituents}[_ref]`. Preserve
unknown-total-PSQ names, reference suffix, constituent order and arbitrary
constituent count. Constituent momentum squares do not determine total PSQ.
Other existing names remain unchanged without `rot_idx`. Add `ROT{n}` before
`_ref` in general energy names; it must not parse as `level_index`.

### Copied level assignment

Add `with_level_indices(*, overwrite=False)` to energy collections. Return the
same concrete collection kind with independent metadata; buffers may be shared.

- Accept interacting levels, not SH/NI entries or effective-energy time
  samples. Require known PSQ and irrep.
- Group by ensemble and exact `sector_info`, never wildcard compatibility.
  Partially and fully specified sectors remain separate groups.
- Require one energy type and reference convention per group; reject unlike
  representations. Users select one estimate per intended state. Repeated
  non-null `rot_idx` in a group is an ambiguity, not an extra level.
- Sort finite real full-sample values ascending, preserving input order for
  ties; assign zero-based indices once. Never sort each resample independently.
- Refuse existing assignments unless `overwrite=True`. Ranking is local to
  the supplied collection; it does not establish that omitted states are
  physically absent. Regenerate canonical names only on the copies.

Validate `rot_idx` as a nonnegative integer and include it in energy
equality/hash, repr, labels, parsers, attrs and filters. Neither index implies
or overwrites the other. Legacy names do not encode every newly known sector
field; require explicit namespaces if legacy storage keys collide, and reject
such writes instead of discarding metadata.

## 7. Arithmetic metadata

Implement separately after auditing fitting, statistics, kinematics and edit
operations. Use the same policy for scalar and N-D ufunc paths:

| Operation | Result metadata |
| --- | --- |
| Selection, transpose, diagonal extraction, stack/concatenate | Corresponding cell infos survive. |
| Value-changing unary/scalar operations (`-E`, `conj(C)`, `2*E`, `2/E`, `E**2`, `log(E)`, `sqrt(E)`, `abs(E)`) | Derived metadata with operation and operand identity. |
| Data/data operations (`C/C`, `E+E`) | Derived metadata even for equal input infos. |
| Explicit reference normalization, basis change, effective energy | Domain method creates the intended typed output. |

Do not decide preservation by operand count. Start with derived metadata for
all value-transforming arithmetic; any no-op exception must be explicit.
Names/identity include operation, operand order and unrounded scalar arguments;
labels show the expression with parentheses. Numeric array operands contribute
a deterministic value fingerprint instead of being treated as harmless scaling.

Use lightweight `DerivedObsInfo`/element metadata with ordered operand
identifiers, operation arguments and presentation fields, not a graph of live
sources. Time-wise operations on consistently annotated inputs build a derived
element and at-time cells so time access remains available.

Retain source sector context for unary/scalar operations. For multiple data
operands retain only fields known and equal in every operand; do not fill one
operand's unknown fields from another. Clear energy classification, level
assignment and pivot-mode claims unless a domain method establishes them.
Source context does not claim that the result is the original observable.

`create_ref_sampling` continues building its intended output metadata. Audit
helpers that rename or mutate intermediate results so they neither alter shared
input metadata nor leave derived results mislabeled. No symbolic simplifier,
unit algebra or expression-history browser is added.

## 8. Labels and sampled prediction series

Keep `latex_str` and general/energy public label entry points; correlator and
operator labeling APIs may change to fit the new metadata. Share small physics
helpers in `sigmondsamplings/labels.py`: coordinate substitution, energy symbols
and effective-energy formatting. Backend conversion stays in `slat.plotting.text`.
Reuse `{VAR}`/`{var}` through literal token replacement, not general Python
formatting of LaTeX braces. Handle symbolic/value coordinates, SH masses,
effective decoration, rotation labels and reference denominators explicitly.
Preserve label overrides; they never affect identity.

Add one internal wrapping helper taking sampled predictions, coordinates,
dimension name and an immutable serializable series descriptor:

- `SigmondModelFunc.series(x, dim="time")` uses existing evaluation, then wraps
  the predictions. Keep the existing list/scalar-returning `__call__`.
- `FitResult.theory_series()` wraps existing `theory`, `x` and `x_name` without
  reevaluation or a callable requirement. Preserve reloaded and synthetic /
  linearized predictions exactly.
- A frozen generic prediction-series descriptor carries label/coordinate
  convention, optional explicitly supplied sector/target and a series ID.
  Do not store a live `SigmondModelFunc` or infer physics from parameter names.
  Without explicit physical context its sector is empty.
- Generate the ID once from existing coordinates, a normalized prediction
  buffer and identity fields, excluding labels. This distinguishes prediction
  sets with identical parameter names without later data reads during metadata
  hashing. The normalized encoding must fix dtype/byte order and NaN encoding.

The helper returns `SigmondData` with `AtObsInfo` cells. Numeric plotting does
not coerce floating coordinates into lattice times; effective-energy
calculation still requires integer times. `FitResult` stays general, without
energy-specific accessors or automatic parameter-to-level mapping.

## 9. Sigmond agreement and file compatibility

Native Sigmond operator/correlator data must remain readable without SS-specific
annotations. Export native keys/operators and sample data that Sigmond accepts,
with the intended flags, components and meaning. Test canonicalization by native
key equivalence rather than requiring the old SS byte representation.

No adapters or older schema output are required for SS-only correlator/operator
metadata. The new schema serves the new API directly. Keep SS annotations
separate from native key text so they cannot alter its interpretation.

Write nested records through `ObsMeta`, with recursive NumPy normalization;
no sidecar format. Store each sector at its authoritative owner. Keyed infos
serialize sector/bindings as annotations while retaining Sigmond key semantics.

For energy/general infos, read both new nested sector/role fields and old flat
attrs, including SH `ni_pairs`. Reject contradictory known values. During
migration writers retain legacy flat energy `irrep`/`psq` and `ni_pairs`
projections, generated from the
authoritative fields, alongside explicit new roles. They are wire-format
compatibility projections, not independently editable state.

Test energy/general API compatibility except for deliberate changes such as
`Particle == str` removal and arithmetic identity. Replace obsolete correlator /
operator API tests with tests of the new design and Sigmond behavior; do not
recreate removed APIs merely to satisfy old tests.

Rich metadata is not recoverable from generated names alone. Its persistence
gate applies to SS HDF5; fstream/legacy Sigmond formats preserve samples/keys but
cannot carry all annotations. Document that limitation rather than promising
equivalence. Do not change generic collection deduplication into a comparison
of sample data.

## Target workflow

The application supplies `C`, a pivot and ensemble `ens`. This is target API:

```python
sector = SectorInfo(twoI=0, S=0, psq=0, irrep="A1g")
E = EnergyObsInfo(sector_info=sector, ensemble_info=ens, energy_type="elab")
E.irrep == E.sector_info.irrep

D = pivot.rotate(C).diagonal(component="real")
eff = D.ts.effective_energy(kind="forward", step=1)
eff.ts.axis.offset                  # 0.5, computed
eff.ts.elements                    # one effective element per operator
cell = eff.ts.at(8).isel(operator=2)
cell.observable_info.time          # 8, not 8.5
cell.observable_info.rot_idx       # 2; distinct from level_index
cell.observable_info.sector_info   # all known sector fields

selected = eff.to_collection().filter(irrep="A1g", psq=0, rot_idx=2)
# Existing SS HDF5 entry points write/read these cells.
# Rebuild with SigmondData.from_collection(..., shape_by=["time"]).

ranked = fitted_energies.with_level_indices()
```

The last line starts with explicit fitted-energy metadata; it does not infer it
from arbitrary correlator-fit parameter names.

## Implementation order and acceptance

Before each phase record revision, working-tree fingerprint and failing/skipped
test IDs. Preserve unrelated changes. Keep phases independently reviewable and
avoid unrelated numerical or plotting-backend refactors alongside metadata work.
Record intentional correlator/operator API removals and their replacement tests
so baseline comparison does not turn obsolete behavior into a requirement.

| Phase | Work | Acceptance gate |
| --- | --- | --- |
| 1 | Preserve stacked/concatenated cells; freeze `Particle`. | Mixed parents retain cells; scalar selection/collections agree; particle parsing, normalization, hashing and momentum ordering stay correct. Update string-equality tests deliberately. |
| 2 | Shared sectors, aliases, channel conversion and legacy energy/general codecs. | Fields survive construction, copying, filters and HDF5; conflicts reject; exact/partial behavior is tested; supported native keys and opaque-key diagnostics work. |
| 3 | At/element identity and codecs, pivot bindings and effective cells; remove `EffectiveEnergyObsInfo`. | Matrix → pivot → diagonal → effective energy → selection → collection → write → read retains source/form, sector, pivot ID, rotation index, settings and time. Native key semantics and numerical results agree with Sigmond; callers/exports/tests use the new API. |
| 4 | Computed `.ts`, labels and plotter adoption. | Rebuilt arrays recover offset/extent; unsorted inclusive windows work; scalar cells retain time; mixed offsets plot per series and reject a common query; label overrides work through the new API. |
| 5 | Energy roles, NI naming and copied level assignment. | Legacy SH/NI/energy roles recover; unknown-PSQ/reference NI names work; ranked copies leave inputs untouched and reject incomparable/ambiguous spectra. |
| 6 | Audited arithmetic policy. | Scalar/N-D paths agree; `E` and `-E` coexist; operation order/constants affect identity; fitting/stats/edit/reference workflows remain correct. |
| 7 | Prediction wrapping and `model.series`. | Existing evaluations/predictions stay unchanged; reloaded theory needs no callable; distinct prediction sets remain distinct and metadata round-trips. |

Phase 3 also tests multiple kinds/steps/extents and multiple pivots with the
same displayed `rot_idx` in one effective-energy collection/file. Cover generic
numeric and parsed inputs, off-diagonals without a scalar mode, raw GI operators
without bindings, arbitrary mode mixing, transpose/time selection, generated-ID
collisions and unsupported metadata versions. Assert complete metadata/keys,
not just labels. Compare operator/key parsing and serialization, file reads /
writes, pivot rotations and effective-energy results with Sigmond using native
bindings, fixtures and oracle tests. Retain numerical coverage while updating
its SS call sites. If native checks are unavailable, record the skipped checks
and do not claim that Sigmond agreement was verified.

Primary areas: `info.py`, `energy_levels.py`, `energy_level_collection.py`,
`sampling_array.py`, correlator observable/matrix/pivot/effective modules,
I/O codecs, `plotting/series.py` and model-function/result series wrappers.
Keep metadata/label helpers small; add no general registry beyond the concrete
codec requirements.

## Deferred, with triggers

| Deferred | Add when |
| --- | --- |
| Stored axis metadata / boundary conditions | A consumer needs information unavailable from coordinates, element settings and the ensemble. |
| Quantity hierarchy, templates, auxiliary coordinates or lazy infos | Profiling shows material cell-construction cost, or a real transform cannot express its metadata. |
| General estimate references / provenance graph | Source/calculation identities and explicit namespaces stop distinguishing required estimates. |
| Separate storage-address abstraction | One meaning needs several addresses beyond current component handling. |
| General dimension roles / parameter-axis algebra | A second parameter axis has a concrete supported workflow. |
| Symbol/decorations label representation | Small formatting helpers fail for an actual shared requirement. |
| Correlator-fit parameter semantics | A concrete workflow needs explicit parameter/energy bindings. |
| Removal of energy/general legacy attrs | A separate release decision establishes migration policy; no such gate applies to correlator/operator APIs. |
