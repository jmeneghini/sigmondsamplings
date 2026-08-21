# `ss edit` design: editing an observable set from the CLI

Status: implemented. See section 12 for where the build deviated from this plan.
Scope: `sigmondsamplings.cli`, `sigmondsamplings.io`, plus one small extraction out of
`sigmondsamplings/cli/query.py`.

## 1. Why

The write side of the CLI can currently do three things: `convert` a file format,
`combine` several files, and `energy-tag` a file with energy attrs. There is no way to
*change* the observable set in a file — drop levels, fix a mis-parsed `psq`, rename an
observable to its canonical form, or add a derived observable — without opening a
notebook and driving the collection API by hand.

Everything needed is already in the library and already exercised by
`io/energy_tag.py`, which is in effect a single-purpose edit pipeline:

```python
loader = SigmondLoader(...)              # load
energy = [s.as_energy_level() for s in ...]   # interpret
energy_coll.set_ref(ref_particle)             # annotate
energy_coll.set_shift_particles_from_pycalq_yml(ni_yml)
(non_energy_coll + energy_coll).to_hdf5(...)  # write
```

`ss edit` generalises that skeleton: same load/interpret/annotate/write shape, but with
a selectable scope and a pluggable set of operations. `energy-tag` then becomes one
configuration of `edit` rather than its own code path.

The selection language is not new either. `ss query` already parses `attr=value` specs,
coerces scalars, validates attribute names, and offers did-you-mean suggestions
(`cli/query.py:parse_where_specs`). An edit is a query plus an action, so it should
accept exactly the filter you already vetted with `ss query`.

## 2. Non-goals

- **No in-place editing.** Like every other write command, `edit` is input -> output and
  goes through `_common.guard_output`. Sampling files are analysis inputs; silently
  rewriting one is not worth the ergonomics.
- **No new arithmetic surface.** `edit` exposes *preset* operators built from existing
  collection methods. Arbitrary expressions over observables stay in notebooks.
- **No multi-ensemble editing in v1.** The executor asserts a single ensemble, as
  `to_hdf5` already does.
- **Not a replacement for `combine`.** Merging observables in from a second file is
  deferred (section 9).

## 3. Interface

Two front doors onto one core. Flags cover the one-shot case; a recipe TOML covers
multi-step edits where different operations need different scopes.

### 3.1 Flag form

```
ss edit IN OUT [selection] [operations] [io]
```

```bash
# tag energy attrs and attach a reference particle (the old energy-tag)
ss edit in.h5 out.h5 --tag-energy --ni-yml ni.yml --ref-particle N

# add E/M_N reference levels for every non-ref level
ss edit in.h5 out.h5 --add-ref N

# fix a mis-parsed observable and resync its canonical name
ss edit in.h5 out.h5 -w name=badname_0 --set psq=2 --set irrep=E --rename

# keep only the psq=0 sector
ss edit in.h5 out.h5 -w psq=0 --only

# delete two levels
ss edit in.h5 out.h5 -w level_index=4,5 --drop

# prune to a spectrum captured earlier by `ss query energy --save`
ss edit in.h5 out.h5 --spec spectrum.toml --only
```

Selection (`-w/--where`, `--contains`, `--regex`) is byte-for-byte the `ss query`
language, since it is the same parser.

### 3.2 Recipe form

```bash
ss edit in.h5 out.h5 --recipe edits.toml
```

```toml
[[edit]]
op = "tag-energy"
ni_yml = "ni.yml"

[[edit]]
op = "set"
where = { psq = 0, energy_type = "elab" }
attrs = { irrep = "A1g" }
rename = true

[[edit]]
op = "add-ref"
particle = "N"

[[edit]]
op = "drop"
where = { level_index = [4, 5] }
```

Recipe ops execute in file order, each with its own scope. This is the only form that
can express two differently-scoped `set` operations in one pass.

`--save-recipe recipe.toml` writes the desugared op list of a flag invocation, so an
ad-hoc edit can be captured for provenance and replayed.

Flags and `--recipe` are mutually exclusive; mixing them is a `BadParameter`.

## 4. Execution model

The flag form desugars into an `EditSpec` op list in a **fixed canonical order**, so
flag order on the command line never changes the result:

| # | stage | ops | notes |
| --- | --- | --- | --- |
| 1 | interpret | `tag-energy` | `as_energy_levels()`; non-energy observables set aside |
| 2 | annotate | `set`, `set-ref`, `ni-pairs`, `rename` | mutate attrs of the scope, in place |
| 3 | derive | `add-ref` | append new observables |
| 4 | membership | `keep` (`--only`), `drop` | decide what reaches the output |
| 5 | write | — | `to_hdf5(out, group=out_group, mode="w")` |

Two rules make the scope semantics predictable:

- **Scope is a view, not a filter.** `--where` (and `--spec`) select which observables an
  operation touches; everything outside the scope passes through to the output
  unchanged. Output membership only ever changes via `--only` / `--drop`.
- **Operators resolve their inputs from the full collection, the scope only receives
  the result.** `-w psq=1 --add-ref N` finds the nucleon single-hadron sampling anywhere
  in the file, and emits ref levels only for the `psq=1` levels.

Because membership is decided at stage 4, `--only` cannot strip an observable an
operator still needs.

### 4.1 Why the executor is nearly free: scope aliases the parent

`ObservableCollection.filter` returns `self._fast_load(filtered, ...)`
(`observable_collection.py:474`) — a new collection wrapping the **same**
`SigmondSampling` objects. `copy()` is the explicit opt-in that copies. So an in-place
mutation applied to a filtered scope is visible in the parent collection with no
merge step at all:

```python
scope = collection.filter(psq=0)
scope.set_ref("N")            # collection sees it too, same objects
```

That is the whole executor for every annotate op:

```python
scope = _scope(collection, op.where)   # filter, or filter_by_spec
<mutate scope in place>
collection.clear_shared_attr_cache()
```

No split, no rejoin, and observable order in the output file is preserved exactly.

Three places break the aliasing and are the only ones needing care:

| method | behaviour | consequence |
| --- | --- | --- |
| `obs.replace(**attrs)` | copies via `with_observable_info` | result does **not** reach the parent |
| `create_ref(samp)` | extends the *scope's* `_data` list | new observables do **not** reach the parent |
| any in-place mutation | leaves `_shared_attr_cache` populated | must `clear_shared_attr_cache()` |

Section 7 closes the first two.

## 5. Models: `EditSpec` / `EditResolved`

Follows the Spec/Resolved convention already used by `spectrum_spec.py` and
`io/spec.py`: a permissive pydantic authoring surface, one `resolve()` boundary, a
canonical form the executor consumes. `StrictModel` (`extra="forbid"`) turns a typo in a
hand-edited recipe into a field-level error; `TomlConfigModel` supplies the TOML
round-trip that `--save-recipe` needs.

New module `sigmondsamplings/edit_spec.py`, alongside `spectrum_spec.py`:

Scope lives on a `ScopedOp` base the ops inherit, rather than a nested `Selector`
field, so a recipe spells its scope exactly as the CLI spells its flags — `where = {...}`
directly on the op, not `where = { where = {...} }`.

```python
class ScopedOp(StrictModel):
    """Scope for one edit op. Mirrors the ss query filter language."""
    where: dict[str, Any] = {}
    contains: str | None = None
    regex: str | None = None
    spec: str | None = None       # path to a spectrum TOML; filter_by_spec

    def scope(self, collection): ...        # -> filter_collection(...)

class TagEnergyOp(StrictModel):
    op: Literal["tag-energy"]
    ni_yml: str | None = None
    skip_missing_particles: bool = True

class SetOp(ScopedOp):
    op: Literal["set"]
    attrs: dict[str, Any]
    rename: bool = False          # update_name(strict=False) after mutating

class SetRefOp(ScopedOp):
    op: Literal["set-ref"]        # tag existing is_ref levels
    particle: str

class AddRefOp(ScopedOp):
    op: Literal["add-ref"]        # derive E/M_ref levels
    particle: str
    psq: int = 0                  # frame of the single-hadron mass used as denominator

class KeepOp(ScopedOp):
    op: Literal["keep"]           # validator: requires a non-empty scope

class DropOp(ScopedOp):
    op: Literal["drop"]           # validator: requires a non-empty scope

EditOp = Annotated[
    TagEnergyOp | SetOp | SetRefOp | AddRefOp | KeepOp | DropOp,
    Field(discriminator="op"),
]

class EditSpec(TomlConfigModel):
    edit: list[EditOp]
    def resolve(self, *, base_dir: Path) -> EditResolved: ...
```

`resolve()` is where the cross-op validation lives: reject `add-ref`/`set-ref` on a
collection with no energy interpretation, resolve `ni_yml` and `Selector.spec` against
`base_dir` the way `SamplingDataSourceSpec.resolve` does, and reject an unknown particle
name via `slat.resolve_particle_name`.

### 5.1 `--spec` lives on the scope, not as its own op

Putting the spectrum TOML path on `Selector` rather than adding a `SpecOp` means one
field buys scoping for *every* op, reusing `filter_by_spec`
(`energy_level_collection.py:487`) as the resolver:

```bash
ss edit in.h5 out.h5 --spec spectrum.toml --only        # prune to the spectrum
ss edit in.h5 out.h5 --spec spectrum.toml --drop        # remove the spectrum
ss edit in.h5 out.h5 --spec spectrum.toml --add-ref N   # ref levels for it alone
```

It pairs directly with `ss query energy --save`, which already writes that TOML.

Note this keeps the section-4 rule intact rather than special-casing: `--spec` on its own
is a scope, so pruning is `--spec s.toml --only`. Consistency across selectors is worth
more than saving one flag.

`filter_collection` composes the clauses with AND in one place, so `where` + `spec` +
`regex` combine the same way for flags and recipes alike.

## 6. Layout

```
sigmondsamplings/
    selection.py        # NEW: Selector resolution + attr=value parsing,
                        #      extracted from cli/query.py
    edit_spec.py        # NEW: EditSpec / EditResolved, op models
    io/
        edit.py         # NEW: apply_edits(collection, spec) -> collection
                        #      edit_file(in, out, spec, ...) -> Path
        energy_tag.py   # reimplemented over apply_edits
    cli/
        edit.py         # NEW: typer command
        write.py        # energy_tag() becomes a deprecated alias
        main.py         # app.command()(edit)
```

`apply_edits(collection, spec)` is the data-source-agnostic core, the same split
`run_query_view` uses: it takes an already-loaded collection so a notebook or a
project-wide multi-ensemble front-end can call it without going through a file.

### 6.1 The `selection.py` extraction

`parse_where_specs`, `_parse_scalar`, `_normalize_filter_value`, `_available_attrs`,
and `_unknown_attr_message` currently live in `cli/query.py`. The edit executor is
library code and must not import from `cli/`, so these move to
`sigmondsamplings/selection.py` and `cli/query.py` re-exports them.
`tests/test_cli_query.py` imports several by name from `cli.query`, so the re-export
keeps it green unchanged.

The extraction also gives the two front doors a shared value story: `--set psq=2`
arrives as the string `"2"` and needs the same `_parse_scalar` coercion that `--where`
already applies, while a recipe's `psq = 2` is already an `int`.

## 7. Library changes this depends on

Kept deliberately small — three of the four are single-line-ish, and none introduces a
second way to do something the collections already do.

1. **Reference particle lookup: no new method.** `EnergyLevelMixin.single_hadron_spectra`
   (`energy_level_collection.py:213`) already isolates `SHEnergyObsInfo`, so the lookup
   is one existing call:

   ```python
   matches = collection.single_hadron_spectra.filter(particle=particle, psq=psq)
   ```

   Reusing that property is preferred over filtering directly, since it is the
   established idiom — the equivalent filter is `is_single_hadron=true` (the boolean
   facet every `ObservableInfo` carries), or `obs_kind="energy_single_hadron"` to name
   the exact class. The executor only adds the error messages for zero
   matches ("no single-hadron observable for 'N'; available: ...") and for more than one.

2. **`create_ref` idempotency + return value** (`energy_level_collection.py:329`). Two
   changes to one method:
   - Guard the append on whether an equivalent ref observable already exists.
     `EnergyObsInfo.__hash__`/`__eq__` both include `ref_particle`
     (`energy_levels.py:573-599`), so a plain set membership test on `observable_info` is
     exact — no bespoke comparison needed.
   - Return the list of created samplings instead of `None`, so the executor can push
     them onto the parent collection. Non-breaking; every current caller discards it.

3. **`update_name(strict=False)`** (`energy_levels.py:569`). `canonical_name` raises when
   `irrep`/`psq`/`energy_type` are incomplete, which would abort a whole edit over one
   unparseable observable. Add `strict: bool = True` so the default preserves today's
   behaviour for `create_ref_sampling`, and the executor passes `strict=False` to warn
   and keep the existing name instead.

   No executor-side loop is needed: `AttributeAccessor.__getattr__` already returns a
   `method_proxy` for callables (`observable_collection.py:253`), so the rename op is
   literally `scope.obs.update_name(strict=False)`.

4. **`obs.update(**attrs)`: the in-place sibling of `replace`.** This is the one genuine
   addition, and it is what makes section 4.1 apply uniformly. `replace` copies each
   observable, so a scoped `set` would otherwise need a rejoin:

   ```python
   collection = (collection - scope) + scope.obs.replace(**attrs)
   ```

   That works — `__sub__`/`__add__` already exist and `__add__` is other-wins
   (`observable_collection.py:1317`), so the mutated scope must be the right operand.
   But `__add__` puts `other` first, so every `set` reshuffles observable order in the
   output file, and repeated ops shuffle progressively.

   `update()` instead setattrs on the existing `observable_info` and returns `None`,
   reusing `replace`'s scalar/list/callable value resolution through a shared
   `_resolve_value(value, idx, target)` helper so there is one implementation of the
   value semantics. The executor then has exactly one shape for every annotate op, and
   the rejoin special case disappears.

   If the extra method is judged not worth it, the rejoin one-liner above is the
   zero-new-API fallback and the only cost is output ordering.

## 8. Absorbing `energy-tag`

`add_energy_attrs(input, output, ni_yml=..., ref_particle=...)` keeps its signature and
its module, but its body becomes:

```python
spec = EditSpec(edit=[
    TagEnergyOp(op="tag-energy", ni_yml=ni_yml),
    *( [SetRefOp(op="set-ref", particle=ref_particle)] if ref_particle else [] ),
])
return edit_file(input_file, output_file, spec, in_group=..., out_group=..., overwrite=...)
```

One code path, and the existing non-energy passthrough behaviour is preserved because
`tag-energy` implements the same split `energy_tag.py` does today.

`ss energy-tag` stays registered as a thin deprecated alias that forwards to the same
call and prints a one-line note pointing at `ss edit --tag-energy`. README's CLI
paragraph gains `ss edit` and marks `energy-tag` deprecated.

## 9. Deferred

- `--merge other.h5` to union in observables from a second file, with a conflict policy.
  Overlaps `ss combine`; worth doing only if `--add-ref` needs a reference particle that
  lives in a different file.
- Kinematics-derived operators (non-interacting energies from `TwoParticleKinem`,
  ecm/elab conversion).
- Multi-ensemble edits.
- In-place editing.

## 10. Phases

1. **Selection extraction** — move the filter parser to `sigmondsamplings/selection.py`,
   add `Selector` resolution (`where` + `contains` + `regex` + `spec`), re-export from
   `cli/query.py`, confirm `tests/test_cli_query.py` is untouched and green.
2. **Library changes** — the four items in section 7. Unit tests against a synthetic
   collection, including an explicit test that mutating a filtered scope is visible in
   the parent (the aliasing section 4.1 depends on).
3. **Models + executor** — `edit_spec.py`, `io/edit.py`, `apply_edits` / `edit_file`.
4. **CLI** — `cli/edit.py`, registration, `--save-recipe`, `energy-tag` reimplemented as
   an alias. `CliRunner` tests following `tests/test_cli_query.py`.
5. **Docs** — README CLI paragraph, deprecation note.

## 11. Testing

| file | covers |
| --- | --- |
| `tests/test_edit_spec.py` | op discrimination, `extra="forbid"` rejection, TOML round trip, `resolve()` validation, `Selector` clause composition |
| `tests/test_io_edit.py` | `apply_edits`: scope leaves non-matches alone, scope aliasing reaches the parent, `--only`/`--drop`/`--spec`, `set` + `rename` incl. the incomplete-attr warning path, `add-ref` run twice, non-energy passthrough, single-ensemble assertion |
| `tests/test_cli_edit.py` | `CliRunner` over `tests/data`: flag/recipe equivalence, `--save-recipe` replay, `guard_output`, mutual exclusion errors |
| `tests/test_cli_query.py` | unchanged — proves the `selection.py` extraction was behaviour-preserving |

Two load-bearing tests:

- **Flag/recipe equivalence** — a flag invocation and a replay of its `--save-recipe`
  output produce identical files. This is what keeps the two front doors on one core.
- **`add-ref` idempotency** — running the same edit twice yields the same observable
  count, which is the regression guard for item 7.2.

## 12. What changed during the build

Five deviations from the plan above, all found by building or testing it.

1. **Scope became a base class, not a field.** A nested `Selector` model made a recipe
   read `where = { where = { psq = 0 } }`. Ops inherit `ScopedOp` instead, so the TOML
   spells its scope exactly as the CLI spells its flags.

2. **`tag-energy` must not re-interpret what is already interpreted.** The first
   `add-ref` idempotency test failed at *three* runs, not two. `as_energy_level()`
   re-parses the observable *name*, and a name records `_ref` without recording *which*
   particle — so re-tagging an already-edited file silently reset `ref_particle` (and
   would do the same to `ni_pairs`). `_tag_energy` now passes through observables that
   already carry energy metadata, matching what `from_collection` does. This mattered
   well beyond idempotency: any second edit of a real file was losing attrs.

3. **The reference lookup has to exclude reference levels.** After one `add-ref N`, the
   file holds both `PSQ0_N` and `PSQ0_N_ref` — and the ratio is *itself* a single-hadron
   observable for particle `N`, so a second run found two candidates and reported an
   ambiguity. The lookup filters `is_ref=False`.

4. **`--set` needed attribute validation in the executor.** Flags are parsed before any
   collection is loaded, so nothing could check attribute names; `--set psqq=2` silently
   created a junk attribute that nothing reads and the writer drops. The executor
   validates against the scope's observable metadata and reuses the query language's
   did-you-mean helper (now `selection.unknown_attr_message`, with a `label`).

5. **`create_ref`'s idempotency guard needed a cheap key.** Guarding after calling
   `create_ref_sampling` still paid for the division before discarding it — on a re-run,
   for every level. The metadata construction moved to
   `SigmondSampling.ref_observable_info`, which `create_ref_sampling` also uses, so the
   guard predicts the result without computing it.

Also in v1 rather than deferred: `--spec`, which is on `ScopedOp` and therefore composes
with every op.

### Testing note

`tests/data/*.hdf5` is gitignored and absent from a fresh clone, and `sigmond_query` is
not on PATH here, so `test_cli_query.py`, `test_lazy_loader.py`, `test_loader_writer.py`
and `test_scripts.py` cannot run in this environment (78 failures, unchanged by this
work). Every test added here builds its fixtures in memory via `tests/conftest.py`.
`tests/test_selection.py` deliberately re-covers the `cli.query` helpers that the
extraction moved, so the extraction is guarded by a suite that actually runs.
