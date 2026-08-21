# Sigmond HDF5 file structure

This note describes the on-disk layout of a Sigmond **samplings** or **bins**
HDF5 file as written and read by this package (`sigmondsamplings/io/`), and the
one rule that drives the whole API: **one root group holds exactly one
ensemble.**

## Terminology: groups, not paths

A Sigmond HDF5 file is not flat. Each block of data lives in an HDF5 **group**
that contains the block's header and its per-observable datasets. We call that
block's top node its **root group**. Throughout the API this is just `group`:

| Concept | API |
| --- | --- |
| Root group to read | `SigmondLoader(file, group=...)`, `loader.group`, `ss query … --group` |
| Root group to write | `writer.write_hdf5(file, …, group=...)`, `collection.to_hdf5(file, group=...)`, `ss combine … --out-group` |
| Default when unspecified | `DEFAULT_GROUP = "data"` |

A "group" here is an *in-file* address (POSIX-style `/`, no filesystem meaning),
distinct from the *filesystem* path of the `.hdf5` file itself. Root groups may
be nested (e.g. `isotriplet/P0A1g`); they are discovered as any group that
contains a `Values` child, at any depth.

## Layout

```
file.hdf5
├── Info/                         global file metadata (exactly one per file)
│   ├── FIdentifier   = "Sigmond--SamplingsFile"  or  "Sigmond--BinsFile"
│   └── Endianness    = "L"
│
├── data/                         ← a root group (name is the `group` argument)
│   ├── Header        = "<SigmondSamplingsFile>…</…>"   (XML, see below)
│   ├── IncludeCKS    = "N"
│   ├── Values/                   one dataset per observable component
│   │   ├── <obs-key-A>   float64[num_samples]   (real, or the Re half)
│   │   ├── <obs-key-A-Im> float64[num_samples]  (Im half, complex obs only)
│   │   └── …
│   └── ObsMeta       consolidated per-dataset metadata table (see below)
│
├── isotriplet/P0A1g/             ← another, independent root group …
│   ├── Header  ·  IncludeCKS  ·  Values/  ·  ObsMeta
│   └── …                         … with its own ensemble + observables
└── …
```

`Info` is global and shared. Everything else lives under a root group, and a
file may contain **several** root groups side by side.

## One root group = one ensemble

The root group's `Header` carries exactly **one** `<MCEnsembleInfo>` (and, for a
samplings file, exactly one resampling description). So every observable in a
root group shares the same ensemble and the same sampling scheme — that is an
invariant of the format, not a convention. The loader enforces it: a
`SingleEnsembleCollection` read from one group rejects mixed ensembles.

Consequences:

- To store **multiple ensembles** in one file, write **multiple root groups**
  (one per ensemble), e.g. `data/cls21_c103`, `data/cls21_d200`.
- To read several ensembles together, point `MultiSigmondLoader` at one
  `(file, group)` per ensemble — across one file or many. It concatenates them
  into a `MultiEnsembleCollection` and verifies the resampling scheme matches.
- Auto-detection (`group=None`) only works when a file holds a **single** root
  group; otherwise you must name the group.

## The `Header` XML

A samplings header:

```xml
<SigmondSamplingsFile>
  <MCBinsInfo>
    <MCEnsembleInfo>clover_s32_t256_ud860_s743</MCEnsembleInfo>
    <NumberOfMeasurements>412</NumberOfMeasurements>
    <NumberOfBins>412</NumberOfBins>
    <!-- optional <TweakEnsemble> … </TweakEnsemble> -->
  </MCBinsInfo>
  <MCSamplingInfo>
    <!-- exactly one of: -->
    <Bootstrapper>
      <NumberResamplings>800</NumberResamplings>
      <Seed>6754</Seed>
      <BootSkip>127</BootSkip>
    </Bootstrapper>
    <!-- or <Jackknife/> (simple)  or  <Jackkniffer><NumberResamplings>…</…></Jackkniffer> -->
  </MCSamplingInfo>
</SigmondSamplingsFile>
```

A **bins** file uses `<SigmondBinsFile>` and has **no** `<MCSamplingInfo>`
section — raw bins carry no resampling metadata. The file's kind is also
recorded redundantly in `/Info/FIdentifier`; the loader cross-checks the two.

## `Values` datasets

- One dataset per observable **component**, stored as `float64`. A complex
  observable is split into two datasets (real and imaginary).
- The dataset name is the observable's XML key made HDF5-safe: since `/` is the
  HDF5 path separator, `/` → `|` and the closing-tag `</` → `<|`. The loader
  reverses this on read, so CorrT/raw-XML observable names round-trip verbatim.

## `ObsMeta`

A single 1-D variable-length UTF-8 dataset beside `Values`, holding one JSON
object per `Values` dataset: its `key`, sample `shape`, `dtype`, and optional
annotations such as explicit `latex_str` labels or energy metadata (`obs_kind`,
`irrep`, `psq`, `energy_type`, `level_index`, `ref_particle`, `ni_pairs`). One
read recovers all per-observable metadata, and the lazy loader can report shapes
without opening a single sample dataset. Files predating this table (and real
Sigmond output) simply lack `ObsMeta`; readers fall back to per-dataset HDF5
attributes.

`obs_kind` is the class discriminator: the loader looks it up in the registry
that `ObservableInfo` subclasses join via `@register_obs_kind` and calls that
class's `from_attrs`, so a read reproduces the exact type that was written
rather than guessing from the dataset name. Current tags are `energy` and
`energy_single_hadron` (alias `energy_sh` accepted on read); an absent tag means
a plain `ObservableInfo`. Since energy types regenerate their LaTeX label from
these attrs, `latex_str` is stored only for observables that do not.
