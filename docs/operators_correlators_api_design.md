# Operators, correlators and correlation matrices

Reference for `sigmondsamplings/correlators/` and the decisions behind it. The
physics reference is `../../sigmond/doc/operators_and_rotation.md` (Part II),
checked against the C++ classes in `../../sigmond/src/sigmond/cpp/observables/`.

## Scope

SS represents the **labels** of operators, correlators and correlation
matrices, finds their data in existing Sigmond files (HDF5 or fstream), and
assembles labeled, complex, sample-aligned time series and matrices for
analysis and fitting. Like Sigmond, it does not construct quark operators or
run contractions. Reading every existing Sigmond file is a hard requirement;
matching Sigmond's internal design is not.

Built: Sigmond's short-form BL/GI operator labels, `CorrT` and `VEV` keys,
typed correlator/VEV observable infos, a metadata-only index, strict square
matrices with Hermitian completion, VEV subtraction and complex basis changes,
and HDF5 round trips that Sigmond reads back. Not built: the single pivot (see
the end), long-form XML and Sigmond's `icode` ordering (see *Decisions*).

## Where things live

| What | Where |
| --- | --- |
| Hadron flavors, irreps, isospins, spatial types (Sigmond's name tables) | `slat/sigmond_names.py`; irrep LaTeX in `slat.IRREP_LATEX_MAP` covers every Sigmond irrep |
| Operator labels and grammar: `Hadron`, `BasicLapHOperator`, `GenIrrepOperator`, `Channel`, `parse_operator` | `correlators/operators.py` |
| Identities: `CorrelatorInfo`, `CorrelatorAtTimeInfo`, `VEVInfo`, `CorrelatorMatrixInfo`, `parse_key` | `correlators/keys.py` |
| Observable infos: `CorrelatorObsInfo`, `VEVObsInfo` (registered obs kinds `correlator`, `vev`) | `correlators/observables.py` |
| `CorrelatorIndex` | `correlators/index.py`; `SigmondLoader.correlators` caches one |
| Assembly, `CorrelatorMatrix` | `correlators/matrix.py` |
| N-D primitives: `apply_matrix`, `transpose`, `rename_dims`, `diagonal`, `from_collection(missing="raise")`, `from_data(infos=...)` | `sampling_array.py`, reached through `SigmondData` methods |
| Batched low-level HDF5 reads: `lazy.materialize`, `lazy.stack_data` | `lazy.py` |

Labels, keys and infos are pure Python (no NumPy/xarray), so importing them is
cheap; `slat` loads its pydantic-based TOML models only on first use.

## Naming

Names follow Sigmond's classes and one rule for suffixes:

- **Operators have no suffix.** `BasicLapHOperator` and `GenIrrepOperator`
  (Sigmond's `BasicLapHOperatorInfo` / `GenIrrepOperatorInfo`, "BL" / "GI")
  and their union `Operator`. An operator object *is* the operator; no data
  object competes for the name.
- **`...Info` is a sample-free identity** of something that has data, as in
  Sigmond: `CorrelatorInfo` (a correlator `C_ij` for all times, from its sink
  and source), `CorrelatorAtTimeInfo` (`C_ij(t)` with the key flags),
  `VEVInfo`, and `CorrelatorMatrixInfo` (basis plus assembly rules; it writes
  Sigmond's `<CorrelatorMatrixInfo>`). They are frozen and hashable.
- **`...ObsInfo` is the metadata of a stored observable** (`ObservableInfo`
  subclasses): `CorrelatorObsInfo` wraps a `CorrelatorAtTimeInfo` the way
  Sigmond's `MCObsInfo` does; `VEVObsInfo`, `EffectiveEnergyObsInfo`.
- **The data** is a `SigmondData` or a `CorrelatorMatrix`, whose identity is
  `CorrelatorMatrix.info`.

Attributes use the same words: `CorrelatorAtTimeInfo.correlator` and
`CorrelatorObsInfo.correlator` (so `group_by("correlator")`),
`CorrelatorIndex.correlator_infos` (identities) versus
`CorrelatorIndex.correlator(...)` (data). "Key" is kept only where the HDF5
name is meant (`parse_key`, `ObsInfo.key`, `key_form`).

## Labels

Operators are frozen dataclasses with defaults resolved at construction, so
`str(op)` is Sigmond's canonical short form and `op.coordinate` is that form
wrapped as `BL{...}` / `GI{...}`. **Equality and hashing use the coordinate**
(computed once): the short form encodes every field, as Sigmond's own integer
encoding does. Operators pickle by their text, since a cached string hash is
only valid within one process. Parsing is memoized (`lru_cache`), so the few
labels that recur at every time and matrix cell are parsed once per process.

- `BasicLapHOperator`: the vacuum (`BasicLapHOperator.vacuum()`), one hadron
  (`BasicLapHOperator.single(...)`), or several (explicit total isospin, irrep and
  row; CG occurrence ids; `bwd`). Tetraquarks take the long flavor name plus
  `color=±1`; the short `tq...p/m` names convert.
- `GenIrrepOperator`: isospin, strangeness, definite momentum **or** `PSQ`,
  irrep, optional row, name and index. `PSQ=0` becomes `P=(0,0,0)`.
- `Channel`: isospin, strangeness, momentum or `PSQ`, irrep and row, compared
  exactly (an unspecified field is a distinct value, not a wildcard).
  `Channel.to_sector()` gives the isospin and strangeness as a `SectorInfo`.
- `latex_str` on hadrons, operators and the observable infos, built from the
  `slat` tables. Sigmond's `omega` is the baryon Ω, not the meson that
  `slat.PARTICLE_LATEX_MAP` calls `omega`, so operators use their own column.

Keys print the **sink first** (`<CorrT>BL{sink} GI{source} time=5 HermMat
SubVEV</CorrT>`), with flags in Sigmond's required order; the reader accepts
either order and uses a brace-aware scanner. Sigmond's bit-field limits are
checked only on export (`to_sigmond()`). `CorrelatorMatrixInfo.to_sigmond_xml()`
writes a `<CorrelatorMatrixInfo>` for Sigmond input (Sigmond reindexes the
basis in its own order).

## Observable infos and the index

`SigmondLoader._parse_observable_key` gives every `<CorrT>` observable a
`CorrelatorObsInfo` and every `<VEV>` observable a `VEVObsInfo`. Their typed
`key` is parsed on first use and exposed as attributes (`sink`, `source`,
`correlator`, `time`, `hermitian`, `subtract_vev`, and the sink channel's `irrep`,
`psq`, `momentum`, `isospin`, `strangeness`). They therefore work in
`filter`, `group_by`, `ss query --where` and `--list-attrs` with no correlator
code in those layers. The info keeps the file's key text as its name, so
writes reproduce the original key.

**Tolerance.** A key that does not parse (for instance a flavor or irrep SS
does not know yet) never breaks a load: the info's `key` is `None`,
`key_error` says why, and the quantum-number attributes are `None`.
`CorrelatorIndex.from_collection` lists such names in `unparsed` with one
warning; `strict=True` raises instead. Labels built in Python are validated
strictly.

`CorrelatorIndex` maps typed keys to the original (possibly lazy) observables
without reading samples. It offers `operators`, `channels` (operators grouped
by channel: the candidate bases), `times`, `correlator_infos`, `vev_operators`, `get`,
`series_times`, `matrix(...)` and `correlator(...)`. One index covers one
ensemble, sampling plan, kind (bins or samplings) and sample length; mixed
collections raise. Duplicate semantic keys raise even when spelled differently.
Equal `SamplingInfo` does not prove that two independently generated bootstrap
files used the same draws.

## Matrix assembly contract

`index.matrix(basis, times=None, *, hermitian=True, subtract_vev=False,
key_form=None, sampling_info=None, include_vevs=False)`, where `basis` is a
sequence of operators (objects or short strings, kept in order) or a
`CorrelatorMatrixInfo`. `index.correlator(sink, source, ...)` applies the same
rules to one correlator and returns a 1-D `SigmondData` over `time`.

- **Basis.** Rejected: empty, duplicates, the vacuum, several channels, and a
  mix of forward and backward operators (Sigmond's own check for this is
  commented out; averaging them is wrong).
- **Key form.** The stored key flags used are the request's key form:
  `"plain"`, `"HermMat"`, `"SubVEV"` or `"HermMat SubVEV"`. One form per
  request, never mixed. A Hermitian request may read `HermMat` or unflagged
  keys, a non-Hermitian one only unflagged keys. A connected request reads
  stored `SubVEV` keys or derives every cell from raw keys and VEVs. If exactly
  one allowed form is stored for the basis it is used; if several are, the
  request raises, names them with their cell counts, and `key_form=` chooses.
  If none is, the error names the forms that are stored. A Sigmond file
  normally holds one form: Sigmond averages `HermMat` pairs on the raw
  per-configuration data before binning (`BLCorrelatorDataHandler::getSymData`)
  and reads the flipped key of a `HermMat` record by conjugation
  (`MCObsHandler`).
- **Times.** Explicit `times` must all be complete, or the error lists the
  missing cells per time. `times=None` takes every complete time and warns
  about observed times left out (`matrix.omitted_times`). Nothing is
  zero-filled, interpolated or wrapped.
- **Hermitian completion.** At each time: `(C_ij + conj(C_ji)) / 2` sample by
  sample when both directions are stored, the stored direction (or the
  conjugate of the flipped one) when only one is. Diagonals keep their
  recorded complex value; `check_hermitian(tol)` reports residuals and
  imaginary diagonals in units of the statistical error.
- **VEV subtraction.** `C_conn[i,j] = C[i,j] - v[i] * conj(v[j])` at every
  sample, including the full estimate. From bins, the correlator and VEV bins
  are stacked and resampled jointly (`sampling_info=` required), then
  subtracted, which keeps the nonlinear full-sample estimator and the
  correlations between cells. Stored `SubVEV` data are never subtracted again.
  Missing VEVs are all reported before any sample is read.
- **Bins.** A raw bins request stays bins unless `sampling_info=` asks for
  samplings (correlators and any attached VEVs are resampled jointly).
- **Lone imaginary parts.** A record with only an `Im` dataset is indexed but
  raises if used; a lone `Re` record is a real correlator.

`CorrelatorMatrix` holds `info` (its `CorrelatorMatrixInfo`), `values` (an ordinary N-D `SigmondData` with
dims `(time, sink, source)` and string coords `op.coordinate`), optional
`vevs` (dim `operator`), and how the matrix was read: `key_form`, `stored` (a
boolean `(time, sink, source)` array, true where that ordered direction was
itself stored, so averaged and conjugated cells can be told apart) and
`omitted_times`. It provides `sel`/`correlator`/`select` (accepting operator
objects or strings), `at`, `diagonal(component="complex"|"real")`,
`sel_times`, `transform`, `check_hermitian`, `to_collection` and `write`.

**Identity lives in the coords.** N-D arithmetic replaces per-cell infos with
placeholders when operands differ, so `to_collection()` rebuilds every cell's
`CorrT` key from the coords and `spec` rather than trusting the infos. Cell
infos are `CorrelatorObsInfo`s whose key text is generated only when needed.

**Basis change.** `O_new[a] = sum_i conj(T[i,a]) O_old[i]`, so
`matrix.transform(T, new_gi_operators)` gives `C_new = T^H C T` and
`v_new = T^H v`, over all times and samples through `apply_matrix`. `T` has
shape `(N_old, N_new)`, must be finite, and the new GI labels must share the
basis channel with the same definite momentum (`PSQ`-only labels are
rejected). The coefficients are kept in `matrix.transformation` and compose
across repeated transforms; GI labels alone never encode them.

## Performance

`benchmarks/correlator_matrix.py`, 12 operators × 48 times × 201 bootstrap
samples, complex correlators with both directions stored (13,824 datasets):

| Step | Before | After |
| --- | --- | --- |
| Lazy load, SS file (with `ObsMeta`) | 1.13 s | 0.42 s |
| Lazy load, Sigmond-layout file (no `ObsMeta`) | 2.86 s | 1.05 s |
| Index build | 0.13 s | 0.08 s |
| Full matrix, lazy reads | 2.9 s | 0.8 s |

What changed: root-group discovery no longer visits every dataset;
`ObsMeta`-less files open each dataset once through `h5py.h5d` and read attrs
only when present; matrix reads go through `lazy.materialize`, which opens
each file once and reads every dataset straight into one preallocated array
with the low-level API (about 2.7× faster than `group[name][:]`); assembly
gathers each cell's rows directly into the output with vectorized Hermitian
completion; operator hashes and coordinates are computed once.

HDF5 metadata and per-dataset reads are CPU-bound and dominate lazy loading.
The measurements were taken on a machine under heavy memory pressure (swap
in use), where freshly allocated arrays fill about 25× slower than warm ones.
The in-memory assembly figures from that machine are inflated, and were not
tuned further. Memory is the complex buffer, `16 T N² S` bytes, plus one
lightweight info per cell.

## Decisions

- **Short forms only.** Sigmond's HDF5 dataset names are always the short form
  (`MCObsInfo::serialize` writes `output(xml, false)`), so reading files never
  needs long-form XML. Long operator/correlator XML, `icode` ordering, and
  reading `<CorrelatorMatrixInfo>` with `<AssignName>`/`<Name>` were removed.
  They matter only for Sigmond task XML and pivot files, which SS does not
  read; add them with a fixture when a workflow needs them.
- **Key forms are chosen, not ranked.** An earlier version ranked candidate
  routes and forms by coverage. That covered combinations Sigmond does not
  produce, and could pick a form silently. Now ambiguity is an error with a
  `key_form=` remedy.
- **Provenance is a mask.** A per-cell operation code plus source-key table
  was reconstructible from coords, flags and the `stored` mask, and needed
  alignment bookkeeping in every slicing method.
- **No `symmetry=` modes.** `hermitian` determines completion;
  `check_hermitian()` is the statistical check.
- **Tetraquark `D2`/`D3` quirk is reproduced.** Sigmond's `shortwrite_hadron`
  masks a shifted tetraquark flavor code when choosing which displacement to
  omit. SS matches it byte for byte (all 96 spellings are tested against C++),
  because Sigmond looks records up by exact key text.
- **Parsing is not reimplemented from the bindings at runtime.** The `sigmond`
  Python bindings (used by `sigmond_scripts`) need a C++ build, so they are
  the test oracle only. When they are installed, tests compare canonical
  strings with C++; when `sigmond_query` is on `PATH`, a test requires it to
  read back SS-written keys. The installed Sigmond build is older than the
  source tree and drops `bwd`, so backwards operators are checked against the
  source grammar only.

## Open items

- **Fitting handoff (done for one series).** `SamplingFit.from_data(series,
  model, param_set, x=None)` takes a 1-D `SigmondData` and uses the values of
  its dim `x` (default: its only dim) as the independent variable. The model
  is an `XModel`, e.g. `CallableXModel(func(x, theta), names)`, and is bound to
  those values in data order. The values are also kept on `fit.x` /
  `fit.x_name`. Complex series are rejected until projected (`.to_real()` or
  `diagonal(component="real")`), and bins until resampled. `from_model` with
  `SamplingStats` is unchanged. Still open: simultaneous fits over a 2-D series
  (e.g. `(time, operator)`), and carrying `x` onto `FitResult` (its `theory`
  is still a flat `SamplingStats`, and `x` is not written to HDF5).
- **`sigmond_query` and mixed records.** The installed `sigmond_query`
  rejected one file mixing real-only and complex records under particular
  keys; the C++ key test writes every record as complex.
- **Single pivot (planned).** See `single_pivot_design.md`: `SinglePivot`
  in `correlators/rotation.py`, rotation through `transform`, overlaps, and
  the eigenvalue diagnostic (`principal_correlators`) that was listed here.
  It also plans lifting the definite-momentum restriction on `transform` for
  momentum-averaged (`PSQ`-only) bases.
