# Observable info mutation: frozen identity, free presentation

Status: proposed, revised 2026-10-15. This plan replaces an earlier proposal
to convert every `ObservableInfo` class into a frozen dataclass. Examples show
the **target API**, not the current implementation.

## Problem

`ObservableInfo` and its direct subclasses, together with `EnsembleInfo` and
`SamplingInfo`, are the last mutable layer of the metadata model. The rest is
already frozen: `SectorInfo`, `SeriesInfo`, `EnergyTarget`, `DerivedElement`,
`PredictionSeriesInfo`, `EffectiveEnergyInfo`, and the correlator keys
(`CorrelatorInfo`, `CorrelatorAtTimeInfo`, `VEVInfo`).

### Symptoms

| Symptom | Where |
|---|---|
| Classes whose sector is computed run `del self._sector_info` after the base `__init__` stores it | `_KeyedObsInfo`, `AtObsInfo`, `DerivedObsInfo` |
| An energy name goes stale when a field changes, until someone calls `update_name()` | `EnergyObsInfo`, 5 library call sites |
| `obs.update` mutates info objects in place, so an info's hash changes while a dict holds it as a key | `observable_collection.py:269` |
| `obs.update` depends on argument order and isn't atomic. Each setter validates against the half-edited state | energy setters (`hadron`, `ni_particles`, `non_interacting`, `energy_type`) |
| `EnsembleInfo.__eq__` ignores the lattice extents and `num_streams`, but `__hash__` includes them | `info.py:451-484` |

These were confirmed by running the current code:

```python
c.obs.update(psq=1)              # name stays PSQ0_A1g_elab_0 although psq is now 1
s.observable_info in c.to_dict() # False: the hash changed under the dict

pair = (Particle("pi", 0), Particle("pi", 1))
c.obs.update(non_interacting=True, ni_particles=[pair])  # ValueError: NI energy requires constituents
c.obs.update(ni_particles=[pair], non_interacting=True)  # succeeds, but the name stays PSQ1_A1u_elab_0

a, b = EnsembleInfo("E250", 100, 96, 192), EnsembleInfo("E250", 100)
a == b, len({a, b})              # (True, 2). The same happens to infos that carry them
```

Other defects found during the review:

- `_coerce_array_obs_info` changes the `name_fn` of the `ArrayObsInfo` that
  the caller passed in (`sampling_array.py:143`).
- The `SigmondData.ensemble_info` setter mutates the info, which other
  samplings may share (`data.py:242`).
- `--set` checks attribute names with `dir()` (`selection.py:247`). Read-only
  properties and methods pass the check, so `setattr` then fails, or
  overwrites the method.

### Root causes

A survey of info usage, the collection classes and the correlator API found
three structural causes behind these symptoms.

**1. Identity is defined five different ways.**

| Layer | What counts as "the same observable" |
|---|---|
| HDF5 writer | the dataset key from `(name, index)`, with Re/Im folded together (`writer.py:421-485`) |
| In-memory `__eq__`/`__hash__` | every field, including the sector and the energy roles |
| Correlators | the typed frozen key plus the rotation bindings |
| N-D arrays | coords plus the array info. Cell infos are placeholders, and `CorrelatorMatrix.to_collection` rebuilds them from the coords |
| `AtObsInfo` and `DerivedObsInfo` | a sha256 of the identity record, which is also the generated name |

As a result, two infos that differ only in their sector are distinct in
memory, yet the writer rejects them as a "Duplicate logical observable". The
plan keeps this difference on purpose (settled decision 1). Identity in
memory is finer than the dataset key, so a conflicting retag survives until
the writer reports it by name, instead of being merged away silently.

**2. `name` means four different things.** It's a free label on plain
infos. On energies it's either the canonical name or a legacy name. On
correlators and VEVs it's the key text, and on `AtObsInfo` and
`DerivedObsInfo` it's a hash of the identity record. Only the energy name is
both stored and derivable from other fields, which is why it drifts.

**3. Mutability forces defensive code throughout the library.**

- `CorrelatorIndex.from_collection` copies every info. The loader exposes
  `refresh_correlators()` so that edits reach the index (`index.py:34`,
  `loader.py:182-192`).
- `SinglePivot` deep-copies `EnsembleInfo` and `SamplingInfo` when it's
  built, and again on every read (`pivot.py:177-184, 302-303, 340-344`).
- Collections carry `_shared_attr_cache`, `_mutation_version` and
  `clear_shared_attr_cache()` to invalidate cached values after edits.
- Metadata elements store their parent infos as JSON snapshots, and plot
  markers snapshot their samplings.

### How the code uses infos

- **Library:** the most-read attributes are `ensemble_info` (26 reads),
  `name` (17), `latex_str` (7) and `index` (7). Energy fields are read
  mostly through `isinstance(…, EnergyObsInfo)` branches (14).
- **Research scripts:**
  - Assign `observable_info.name` about 10 times and `latex_str` about 8
    times.
  - Set `energy_type` and then call `update_name()` twice.
  - Attach ad-hoc attributes in one script (`paper_referee.py`: `rot_i`,
    `rot_j`, `t`).
  - Use collections mainly through `filter` (79 calls), `interacting_spectra`
    (60), `filter_by_spec` (26) and `to_dataframe`. `group_by` and
    `shared_attr` are rare.
- **Energy collections:** the 13 energy-specific accessors (`irreps`,
  `psqs`, `group_by_irrep`, …) are generic column operations under different
  names.

## Alternatives considered

| Design | Verdict |
|---|---|
| **A. Copy-on-write only.** All library edits go through `replace()` | Needed in any case (steps 1–2), but on its own it leaves identity split five ways and keeps the snapshots and caches |
| **B. Frozen identity, free presentation** | **Adopted** (steps 3–4) |
| **C. Columnar metadata index on collections** | Adopted as a follow-on to B (step 6) |
| **D. Collections as 1-D labeled arrays** (metadata as xarray coords; `from_collection(shape_by)` becomes unstack) | The biggest unification, but deferred. Collections can hold samples of different lengths, several ensembles, and lazily loaded datasets, and fits, plotting and arithmetic all use the per-item `SigmondSampling` API. C delivers most of the benefit |
| **E. Frozen dataclasses** | Rejected; see below |

### Why not frozen dataclasses

1. **A dataclass field can't be overridden by a property.** `AtObsInfo`,
   `DerivedObsInfo` and `_KeyedObsInfo` compute `name` (and `sector_info`)
   as properties. A dataclass subclass that overrides a base field with a
   read-only property crashes in its generated `__init__`
   (`AttributeError: property 'name' ... has no setter`, checked).
2. **Constructor signatures differ across the hierarchy.**
   - The base class takes `(name, index, op_type, re_im, ensemble_info,
     latex_str)`.
   - `EnergyObsInfo` takes `(name, index, ensemble_info, irrep, psq,
     energy_type, particles, ...)`.
   - `SHEnergyObsInfo` takes `particle` positionally.
   - The legacy arguments `irrep=`, `psq=` and `particles=` aren't fields at
     all.

   Every class would still need a hand-written `__init__`.
3. **It freezes presentation too.** `latex_str` and ad-hoc tags play no part
   in identity, and the research code edits them in place.
4. **It doesn't fix legacy names or storage identity.** An `explicit_name`
   field would leave a loaded legacy name stale after `replace(psq=1)`.
5. **It makes a file-format change without saying so.** That proposal applied
   `SectorInfo.common` to correlator sink and source sectors; step 5 explains
   why that breaks loading of existing files.

## Design

> Every field of an info is either **identity** or **presentation**.
> Identity is frozen and defines `__eq__`, `__hash__`, the name, and every
> attribute you can filter on. Presentation is mutable and never takes part
> in equality, hashing or filtering. Identity changes only through
> `info.replace(...)`, which returns a new info built by the constructor.
> Presentation can be edited in place, because such an edit can't change a
> hash or the result of a query.

There's no third category. An earlier draft had "facts" (frozen but left out
of `__eq__`/`__hash__`) for the sector of plain infos; settled decision 1
removed it.

Today's `__eq__` already compares every identity field and leaves out
`latex_str`. The split therefore doesn't change which infos compare equal.
It makes the existing rule explicit and enforced. The only behaviour changes
are that energy names are derived from their fields, and that identity can no
longer be assigned in place.

The writer's dataset key, `(name, index)`, is coarser than identity. Two
identities that share a key can exist in memory, and the writer's
duplicate check rejects them when they're written together.

**Identity** consists of the kind-specific fields plus a shared context of
`index`, `op_type`, `re_im` and `ensemble_info`:

| Kind | Identity fields |
|---|---|
| plain `ObservableInfo`, `ArrayObsInfo` | `name`, `sector_info` (plus `name_fn` for arrays) |
| `EnergyObsInfo`, `SHEnergyObsInfo` | `sector_info`, `energy_type`, `level_index`, `rot_idx`, `ref_particle`, `hadron`, `ni_particles`, `non_interacting` |
| `CorrelatorObsInfo`, `VEVObsInfo` | the typed key (and the rotation bindings for correlators) |
| `AtObsInfo`, `DerivedObsInfo` | the element or derivation, the coordinate, and the value |
| `OverlapObsInfo` | the operator, the level, `pivot_id`, `normalize`, and the three times |
| `ArrayElementObsInfo` | `parent`, `array_index` |

Every kind includes the sector in its identity, either as a field (plain,
array and energy infos) or through its key or element (correlators, VEVs,
`AtObsInfo`, `DerivedObsInfo`, array cells through the parent).

**Presentation** fields are mutable and never compared:

- `latex_str`;
- `storage_name`, the spelling the writer uses when it differs from the
  encoded name (see "Names" below);
- `tags`, a free dict that replaces ad-hoc attributes.

### Names

`name` is an encoding of identity for every kind:

| Kind | `name` |
|---|---|
| plain | the name as given. It's part of the identity, along with the sector |
| energy | `canonical_name` |
| correlator, VEV | the key text |
| `AtObsInfo`, `DerivedObsInfo` | the generated hash name |

A legacy spelling read from a file becomes `storage_name`. The writer uses
`storage_name` when it's set and `name` otherwise. **Any `replace()` that
changes identity drops `storage_name`**, because the alias belonged to the
identity it named. This rule replaces both `update_name()` and the earlier
heuristic that checked whether a pinned name still parsed consistently.

## Plan

Each step ships on its own.

### Step 0: fix the `EnsembleInfo` hash

Make `__hash__` use only the fields that `__eq__` compares: `name`,
`num_measurements`, `num_bins`, and the hashable entries of `tweak_info`. Add a
regression test asserting `len({a, b}) == 1` for the example above, and the
matching case for two infos that carry those ensembles. Collection `+` and
`-` build sets of infos, so files whose headers differ only in whether the
extents are present currently fail to deduplicate or subtract.

### Step 1: add `replace()` by round-tripping through the constructor

- Every info class implements `_init_args() -> dict`, which returns the
  keyword arguments that rebuild it. `OverlapObsInfo`, `AtObsInfo`,
  `DerivedObsInfo` and `_KeyedObsInfo` already have this shape in their
  `from_attrs`.
- `ObservableInfo.replace(**changes)` lives in `info.py` and does four
  things:
  1. Maps aliases: `strangeness` to `S`, and `particle` to `hadron` on
     `SHEnergyObsInfo`.
  2. Routes sector facets (`psq`, `irrep`, `S`, …) into `sector_info` when
     `sector_info` is one of the class's init args. Otherwise it raises
     `AttributeError` with a message such as "`psq` is derived from the
     correlator key; change `name`/`key` instead".
  3. Rejects unknown names and lists the settable ones.
  4. Returns `type(self)(**{**self._init_args(), **changes})`.
- `settable_attributes()` returns the init args plus the routed facets.
- `copy()` becomes `replace()` with no arguments.
- The base `__init__` stores `_sector_info` only for classes that own their
  sector, decided by a class flag. This removes the three
  `del self._sector_info` lines.

Tests:
- For one example of every registered obs kind (parametrize over
  `obs_kinds()`): `info.replace() == info` and
  `type(info.replace()) is type(info)`.
- A change to several fields at once either applies completely or raises.
- Keyed facets still raise (`test_observable_sectors.py:102-116` stays as it
  is).

This step changes no behaviour.

### Step 2: copy-on-write everywhere in the library

| Site | Change |
|---|---|
| `observable_collection.py:226-235` (`obs.replace`) | `info.replace(**resolved)`, wrapped by `with_observable_info` |
| `observable_collection.py:269` (`obs.update`) | `sampling.observable_info = info.replace(**resolved)`. Keep `setattr` for extractors that don't target the info |
| `data.py:242` (`ensemble_info` setter) | rebind `observable_info` |
| `data.py:399-407` (`component`) | `replace(re_im=...)` on the array info and on each cell |
| `energy_level_collection.py:109-110, 196-197, 551-552, 637-638` | `replace(...)` |
| `sampling.py:186-195` (`ref_observable_info`) | `replace(ref_particle=...)` / `replace(name=...)` |
| `io/loader.py:361-362` | `replace(latex_str=...)` |
| `sampling_array.py:143` | return `replace(name_fn=...)` instead of mutating the caller's info |
| `io/edit.py` `_check_settable` | validate against `settable_attributes()` |

`filter` shares sampling objects with its parent collection, so rebinding
`sampling.observable_info` keeps scoped edits visible in the parent. The old
info objects stay unchanged and keep a stable hash, including any held in
user dicts or shared by other samplings.
`LazySigmondSampling.observable_info` is a plain attribute (`lazy.py:238`),
so it can be rebound the same way.

Tests:
- The dict-key and argument-order reproductions above.
- A scoped `update` shows through the parent collection.
- `_coerce_array_obs_info` leaves its argument unchanged.

### Step 3: declare identity and derive energy names

- Each class declares its identity fields and its presentation fields. That
  one declaration drives four things:
  - `__eq__` and `__hash__`, implemented once in the base class in place of
    the per-class overrides;
  - `settable_attributes()`, which then separates identity (changed with
    `replace`) from presentation (assigned in place);
  - the step 4 guard;
  - the step 6 table columns.

  Equality doesn't change for any non-energy kind. Today's overrides already
  compare exactly these fields.
- `name` becomes a property that encodes identity, following the table in
  "Names".
  - `EnergyObsInfo` drops its stored name and `update_name()`. A deprecated
    no-op alias is kept for one release.
  - The 5 library call sites go away.
- The loader passes `storage_name` when a dataset name differs from the
  encoded name.
  - The writer writes every observable under `storage_name or name`.
  - **Collision guard:** if two datasets in one file encode to the same
    energy identity, the loader raises and names both datasets (settled
    decision 3).
- `--set` and `obs.update` accept presentation fields and `tags`. `ss edit
  --rename` becomes unnecessary, since identity edits always rename.
- Tests:
  - Round-trip of `storage_name` through HDF5.
  - An identity edit drops `storage_name`.
  - The collision guard raises.
  - For every non-energy kind, equality before and after the change agrees
    on a fixed set of info pairs.
  - The writer still rejects two identities that share a dataset key
    (`test_hdf5_rejects_colliding_native_storage_keys` stays as it is).
  - `test_sector_fields_participate_in_exact_identity_and_partial_compatibility`
    stays as it is.

### Step 4: freeze identity

- Identity attributes get a `__setattr__` guard. For one release it
  issues a `DeprecationWarning` that points to `replace()` and
  `SigmondData.with_observable_info`, and from then on it raises
  `AttributeError`. Run the research repositories against the warning
  release.
- Presentation fields stay freely assignable. `samp.observable_info.latex_str
  = ...` keeps working, and `info.tags["rot_i"] = 3` replaces ad-hoc
  attributes.
- Freeze `EnsembleInfo` and `SamplingInfo` the same way, storing `tweak_info`
  and `extra_params` as read-only mappings.
- Once infos can't change identity, delete the defensive code they required:
  - the info copies in `CorrelatorIndex.from_collection`;
  - `refresh_correlators()`, or reduce it to an index rebuild;
  - the deep copies in `SinglePivot`;
  - `_mutation_version` and `clear_shared_attr_cache()`. The shared-attr
    cache stays valid because it reads only identity.
- Migration in the research code:
  - Replace `samp.observable_info.name = x` (about 10 sites) with
    `samp = samp.with_observable_info(samp.observable_info.replace(name=x))`.
    A `SigmondData.replace_info(**changes)` shorthand is worth adding.
  - Replace `energy_type = ...` followed by `update_name()` (2 sites) the
    same way.

### Step 5: `SectorInfo.common`, as a refactor only

- Add `SectorInfo.common(*sectors)`. It keeps the fields that every input
  specifies with the same value. Use it in `arithmetic_info`
  (`arithmetic.py:185-187`), which already inlines exactly that logic, so the
  output is identical.
- **Do not** apply it to `CorrelatorObsInfo.sector_info` or
  `CorrelatorSource.sector_info` (`metadata.py:145`) without a
  `metadata_version` bump and a decode path. Here is why:
  - `CorrelatorSource.sector_info` feeds `EffectiveEnergyInfo.to_record()`.
    That record goes into `AtObsInfo.identity_record`, which produces the
    generated `EffEn_ID…` storage name.
  - In addition, `element_from_record` checks the stored sector against the
    recomputed one.

  Effective energies of off-diagonal correlators that were written before the
  change would then fail to load.

### Step 6 (follow-on): a columnar metadata index on collections

Infos can't change identity once step 4 is in place, so a collection can
build a table of identity columns once, on first query, and never
invalidate it. Sector facets are identity for every kind, so they're always
columns. `SigmondSampling` stays the item type.

- `filter`, `group_by`, `unique`, `sort` and `shared_attr` become table
  operations. This also covers the energy shortcuts `filter_by_spec`,
  `interacting_spectra` and `single_hadron_spectra`.
- The 13 energy accessors become thin aliases of generic column queries, or
  are removed.
- `to_dataframe` and `ss query --list-attrs` read the table directly.
- The keys and coverage of `CorrelatorIndex` become queries on the same kind
  of table. This also settles the WORKLOG item about `CorrelatorIndex.keys()`
  ordering, and makes the sector columns first-class filters, which addresses
  the item "use sector info more extensively".
- Presentation fields are left out of the table and read on demand.

Measure before and after on a large correlator file, about 20 operators ×
20 × 30 times, before committing to this step.

## Settled decisions

1. **The sector stays in the identity of plain infos (decided 2026-10-15).**
   `ObservableInfo("x", sector_info=S1) != ObservableInfo("x",
   sector_info=S2)` remains true. The alternative was to treat the sector as
   a fact: frozen, but left out of `__eq__`/`__hash__`. It was rejected for
   these reasons:
   - Its original motivation, a hash that changes when the sector is edited
     in place, disappears once identity is frozen.
   - Two infos with the same name but different sectors almost always mean a
     retagging mistake. Keeping them distinct lets the writer report the
     conflict by name. Making them equal would let the collection
     constructor drop one silently, and let `+` quietly keep the right-hand
     side.
   - Every other kind has the sector in its identity, so excluding it only
     for plain infos would be an exception.
   - The survey found no code that needs such infos to compare equal.

   Consequence: there's no "facts" category, and every field is either
   identity or presentation.

2. **`storage_name` and `ss edit` (decided 2026-10-15): drop the alias and
   rename.** An identity edit drops `storage_name`, and the level is written
   under its canonical name. `ss edit --rename` is removed. Evidence:
   - The 12 current level files scanned (about 3,600 observables across
     lambda_1405, h_dibaryon, rho_nolan, NN_C103, tetraquarks and mdwf)
     contain no legacy energy names.
   - The active KBfit inputs (tetraquarks, NN_C103, h_dibaryon) already
     reference canonical names.
   - Legacy references remain only in the lambda_1405 inputs
     (`PSQ1/G1/dElab_0_ref`, `single_hadrons/S(1)_ref`) and the old rho inputs
     (`T1u_PSQ0_elab_0_ref`). They are affected only if a legacy-named file is
     edited.
3. **Legacy names that collide (decided 2026-10-15): raise.** If two
   datasets in one file encode to the same canonical energy identity, the
   loader raises and names both datasets. The same scan found no collisions,
   so there is no fallback mode that identifies such datasets by name. Step 3's
   collision guard raises instead of warning.

## Open checks

- **N-D arrays.** Replacing an array's `ArrayObsInfo` through
  `with_observable_info` leaves the cells' `parent` pointing at the old info.
  The only reader of `cell.parent` is `source_elements` (`metadata.py:514`).
  Decide whether default placeholder cells should be rebuilt when the parent
  is replaced.
- **`EnergyObsInfo` construction order.** Today's setters read partly
  initialized state through `getattr(..., default)`. `_init_args()` must
  return arguments that the constructor accepts in a single pass, for example
  `particles` and `ni_particles` together. The round-trip test covers this.
- **Silent deduplication (separate fix).** `ObservableCollection.__init__`
  removes samplings whose metadata is equal by calling
  `dict.fromkeys(data)`, without comparing their data. It therefore drops a
  sampling that has the same identity as another but different numbers,
  without saying so. This is independent of this plan. A warning when
  deduplication drops an item would make it visible.
